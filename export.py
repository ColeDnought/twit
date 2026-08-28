"""Export and quantize a trained GaborNet for ESP32-S3 deployment.

Two things make the training model unfit for a microcontroller runtime, and this
module fixes both:

1. The Gabor front-end recomputes its filters every forward with sin/cos/exp. Those
   ops don't quantize and aren't supported by ESP-DL / TFLite-Micro. Since inference
   doesn't learn, we *bake* the trained filters into a static Conv1d weight -- the
   graph then reduces to Conv1d + elementwise magnitude + the CNN head.
2. The variable-length masking path (masked_fill / -inf / adaptive_max_pool1d) is
   dynamic. On-device you record a fixed window, so we drop masking entirely.

The exported ONNX takes a fixed [1, 1, clip_len] float input and returns a [1] logit.
The CLI validates its operators, then uses representative audio and ESP-PPQ to produce
an int8 (or int16) .espdl model. Pass --skip-espdl to stop after ONNX export.
"""
import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models import GaborFilter, load_model


# Runtime operators supported by ESP-DL 3.3 on ESP32-S3. Constant and Identity are
# resolved by the converter rather than emitted as runtime operators.
_ESPDL_INTEGER_OPS = frozenset({
    "Add", "AveragePool", "Clip", "Concat", "Conv", "ConvTranspose",
    "DepthToSpace", "Div", "Elu", "Equal", "Exp", "Flatten", "Gather", "Gemm",
    "GlobalAveragePool", "Greater", "GreaterOrEqual", "GRU", "HardSigmoid",
    "HardSwish", "LayerNormalization", "LeakyRelu", "Less", "LessOrEqual", "Log",
    "LogSoftmax", "LSTM", "MatMul", "MaxPool", "Mod", "Mul", "Neg", "Pad", "Pow",
    "PRelu", "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp",
    "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum",
    "ReduceSumSquare", "Relu", "Reshape", "Resize", "ReverseSequence", "ScatterND",
    "Sigmoid", "Slice", "Softmax", "SpaceToDepth", "Split", "Sqrt", "Squeeze",
    "Sub", "Swish", "Tanh", "Transpose", "Unsqueeze",
})
_CONVERTER_ONLY_OPS = frozenset({"Constant", "Identity"})


class BakedGaborNet(nn.Module):
    """Inference-only GaborNet: static Conv1d front-end, fixed-length input, no mask."""

    def __init__(self, model: nn.Module):
        super().__init__()
        gf: GaborFilter = model.feature_extractor
        K = gf.t.shape[-1]
        self.n_filters = int(gf.center_freq.shape[0])
        self.eps = 1e-8

        # bake the trained filterbank into a plain conv weight (no runtime trig/exp)
        self.conv = nn.Conv1d(1, 2 * self.n_filters, K, stride=gf.stride,
                              padding=K // 2, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(gf.build_kernel())

        self.head = model.head  # reused as-is; called with mask=None below

    def forward(self, x):                          # x: [B, 1, L] fixed L
        out = self.conv(x)                         # [B, 2*n_filters, T]
        cos_out, sin_out = torch.split(
            out, [self.n_filters, self.n_filters], dim=1
        )
        feats = torch.sqrt(cos_out ** 2 + sin_out ** 2 + self.eps)  # [B, n_filters, T]
        return self.head(feats)                    # mask defaults to None


def export_onnx(checkpoint: str, out_path: str, clip_seconds: float = 1.0,
                opset: int = 18):
    """Load a save_model() checkpoint, bake it, export ONNX, and verify parity."""
    model = load_model(checkpoint)
    baked = BakedGaborNet(model).eval()

    sr = model.feature_extractor.sr
    clip_len = int(round(clip_seconds * sr))
    example = torch.randn(1, 1, clip_len)

    # sanity: baked (no mask) must equal the original fed a full-length window
    with torch.no_grad():
        ref = model(example, torch.tensor([clip_len]))
        got = baked(example)
    diff = (ref - got).abs().max().item()
    print(f"baked-vs-original max abs diff: {diff:.3e}")

    torch.onnx.export(
        baked, (example,), out_path,
        input_names=["audio"], output_names=["logit"],
        opset_version=opset, dynamo=False,
    )
    print(f"wrote {out_path}  (input [1, 1, {clip_len}] @ {sr} Hz)")

    _verify_onnx(out_path, example, got)
    check_espdl_operator_support(out_path)
    return out_path


def check_espdl_operator_support(onnx_path: str | Path) -> Counter:
    """Validate ONNX and fail early if it contains unsupported ESP-DL int operators."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX validation requires the 'onnx' package") from exc

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    counts = Counter(node.op_type for node in model.graph.node)
    runtime_ops = set(counts) - _CONVERTER_ONLY_OPS
    unsupported = sorted(runtime_ops - _ESPDL_INTEGER_OPS)

    summary = ", ".join(f"{op} ({counts[op]})" for op in sorted(runtime_ops))
    print(f"ONNX operators: {summary}")
    if unsupported:
        raise RuntimeError(
            "ONNX contains operators not supported by ESP-DL integer inference: "
            + ", ".join(unsupported)
        )
    print("ESP-DL integer operator check passed")
    return counts


def _verify_onnx(onnx_path, example, torch_out):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping runtime parity check")
        return
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"audio": example.numpy()})[0]
    diff = (torch.from_numpy(ort_out).reshape(-1) - torch_out.reshape(-1)).abs().max().item()
    print(f"onnxruntime-vs-torch max abs diff: {diff:.3e}")


class _FixedWindowCalibrationDataset(Dataset):
    """Deterministic fixed windows spread across the training audio collection."""

    def __init__(self, dataset: Dataset, clip_len: int, sample_count: int):
        self.dataset = dataset
        self.clip_len = clip_len
        self.sample_count = min(sample_count, len(dataset))

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        if self.sample_count == 1:
            dataset_index = 0
        else:
            dataset_index = round(index * (len(self.dataset) - 1) / (self.sample_count - 1))

        audio, _ = self.dataset[dataset_index]
        audio = audio.float().flatten() / 32768.0
        if audio.numel() < self.clip_len:
            audio = F.pad(audio, (0, self.clip_len - audio.numel()))
        elif audio.numel() > self.clip_len:
            max_start = audio.numel() - self.clip_len
            start = round(index * max_start / max(self.sample_count - 1, 1))
            audio = audio[start:start + self.clip_len]
        return audio.unsqueeze(0)


def convert_to_espdl(
    onnx_path: str | Path,
    espdl_path: str | Path,
    *,
    data_root: str | Path,
    sample_rate: int,
    clip_len: int,
    bits: int = 8,
    calibration_steps: int = 32,
    calibration_batch_size: int = 32,
    device: str = "cpu",
):
    """Quantize a checked ONNX model with representative audio and export .espdl."""
    if calibration_steps < 1 or calibration_batch_size < 1:
        raise ValueError("calibration steps and batch size must both be positive")
    if bits not in (8, 16):
        raise ValueError("ESP-DL quantization supports 8 or 16 bits")

    try:
        from esp_ppq.api import espdl_quantize_onnx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "ESP-DL conversion requires ESP-PPQ; install it using the "
            "instructions in the ESP-DL documentation"
        ) from exc

    from data.load_data import WarblrbDataset

    check_espdl_operator_support(onnx_path)
    source = WarblrbDataset(root_dir=data_root, sample_rate=sample_rate)
    if not len(source):
        raise ValueError(f"no calibration audio found under {data_root}")
    calibration = _FixedWindowCalibrationDataset(
        source,
        clip_len=clip_len,
        sample_count=calibration_steps * calibration_batch_size,
    )
    loader = DataLoader(
        calibration,
        batch_size=calibration_batch_size,
        shuffle=False,
    )
    actual_steps = min(calibration_steps, len(loader))
    test_input = next(iter(loader))[:1].to(device)

    print(
        f"quantizing {onnx_path} to {bits}-bit {espdl_path} with "
        f"{len(calibration)} calibration clips"
    )
    graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=loader,
        calib_steps=actual_steps,
        input_shape=None,
        inputs=test_input,
        target="esp32s3",
        num_of_bits=bits,
        collate_fn=lambda batch: batch.to(device),
        dispatching_override=None,
        device=device,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )
    print(f"wrote {espdl_path}")
    return graph


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="path to a save_model() .pt checkpoint")
    p.add_argument("-o", "--out", default="gabornet.onnx")
    p.add_argument("--espdl-out", help="output path (default: ONNX path with .espdl suffix)")
    p.add_argument("--skip-espdl", action="store_true", help="only validate and export ONNX")
    p.add_argument("--clip-seconds", type=float, default=1.0)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--data-root", default="data", help="WarblrB dataset root for calibration")
    p.add_argument("--bits", type=int, choices=(8, 16), default=8)
    p.add_argument("--calibration-steps", type=int, default=32)
    p.add_argument("--calibration-batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu", help="ESP-PPQ device, e.g. cpu or cuda")
    args = p.parse_args()
    export_onnx(args.checkpoint, args.out, args.clip_seconds, args.opset)

    if not args.skip_espdl:
        model = load_model(args.checkpoint)
        sample_rate = model.feature_extractor.sr
        clip_len = int(round(args.clip_seconds * sample_rate))
        espdl_out = args.espdl_out or str(Path(args.out).with_suffix(".espdl"))
        convert_to_espdl(
            args.out,
            espdl_out,
            data_root=args.data_root,
            sample_rate=sample_rate,
            clip_len=clip_len,
            bits=args.bits,
            calibration_steps=args.calibration_steps,
            calibration_batch_size=args.calibration_batch_size,
            device=args.device,
        )
