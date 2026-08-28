"""Export a trained GaborNet to ONNX for on-device (ESP32-S3) deployment.

Two things make the training model unfit for a microcontroller runtime, and this
module fixes both:

1. The Gabor front-end recomputes its filters every forward with sin/cos/exp. Those
   ops don't quantize and aren't supported by ESP-DL / TFLite-Micro. Since inference
   doesn't learn, we *bake* the trained filters into a static Conv1d weight -- the
   graph then reduces to Conv1d + elementwise magnitude + the CNN head.
2. The variable-length masking path (masked_fill / -inf / adaptive_max_pool1d) is
   dynamic. On-device you record a fixed window, so we drop masking entirely.

The exported ONNX takes a fixed [1, 1, clip_len] float input and returns a [1] logit.
Feed it to esp-ppq (-> .espdl) or an ONNX->TFLite int8 flow next.
"""
import argparse

import torch
import torch.nn as nn

from models import GaborFilter, load_model


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
        cos_out, sin_out = out.chunk(2, dim=1)
        feats = torch.sqrt(cos_out ** 2 + sin_out ** 2 + self.eps)  # [B, n_filters, T]
        return self.head(feats)                    # mask defaults to None


def export_onnx(checkpoint: str, out_path: str, clip_seconds: float = 1.0,
                opset: int = 17):
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
    return out_path


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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="path to a save_model() .pt checkpoint")
    p.add_argument("-o", "--out", default="gabornet.onnx")
    p.add_argument("--clip-seconds", type=float, default=1.0)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    export_onnx(args.checkpoint, args.out, args.clip_seconds, args.opset)
