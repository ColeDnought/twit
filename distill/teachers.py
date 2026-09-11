"""Swappable bioacoustic teachers that turn a clip into per-window embeddings.

Every teacher exposes the same tiny contract (`ClipTeacher`): a native sample
rate, a window length, and `embed(waveform, sr) -> [n_windows, D]` (plus an
optional `embed_batch` that extract.py prefers for throughput). `extract.py` feeds
each teacher the shared 32 kHz clip cache and never needs to know which model
produced the embeddings. Heavy / conflicting dependencies (the `birdnet` package
for BirdNET, JAX+TF for Perch) are imported lazily so importing this module stays
cheap and the Perch path can live in its own isolated environment.
"""

from __future__ import annotations

from math import ceil, gcd
from typing import Protocol, runtime_checkable

import numpy as np


def _resample(waveform: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Polyphase-resample a mono float waveform to the teacher's native rate."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if sr == target_sr:
        return waveform
    from scipy.signal import resample_poly

    divisor = gcd(int(sr), int(target_sr))
    return resample_poly(waveform, target_sr // divisor, sr // divisor).astype(np.float32)


def frame_windows(
    waveform: np.ndarray, window_samples: int, hop_samples: int | None = None
) -> np.ndarray:
    """Slice a 1-D waveform into [n_windows, window_samples], zero-padding the tail.

    A final partial hop is snapped back to end-align the last full window so no
    audio near the clip end is dropped.
    """
    hop_samples = hop_samples or window_samples
    n = len(waveform)
    if n <= window_samples:
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[:n] = waveform
        return padded[None, :]
    starts = list(range(0, n - window_samples + 1, hop_samples))
    if starts[-1] + window_samples < n:
        starts.append(n - window_samples)
    return np.stack([waveform[start : start + window_samples] for start in starts])


@runtime_checkable
class ClipTeacher(Protocol):
    name: str
    native_sr: int
    window_seconds: float

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Return [n_windows, D] embeddings for a mono float waveform in [-1, 1]."""
        ...

    def embed_batch(self, waveforms: list[np.ndarray], sr: int) -> list[np.ndarray]:
        """Return one [n_windows, D] array per input clip (extract.py prefers this)."""
        ...


class BirdNetTeacher:
    """BirdNET acoustic embeddings via the official `birdnet` package.

    V3.0 runs at 32 kHz / 3 s segments / 1280-d embeddings on the ONNX backend
    (bundled onnxruntime, no TensorFlow). The package handles segmentation and any
    resampling from the input rate, so `embed` just forwards `(waveform, sr)`. Short
    clips are padded to whole 3 s segments; those silence tails read as "no bird" and
    fall out of the clip-level max-pool, so batching mixed-length clips is safe.
    """

    name = "birdnet"

    def __init__(
        self,
        version: str = "3.0",
        backend: str = "onnx",
        precision: str = "fp32",
        bandpass_fmax: int = 15000,
        device: str = "CPU",  # onnx backend only accepts "CPU"/"GPU"; no mps/cuda on macOS
        onnx_batch_size: int = 16,
    ):
        import birdnet

        self.model = birdnet.load("acoustic", version, backend, precision=precision)
        self.native_sr = int(self.model.get_sample_rate())
        self.window_seconds = float(self.model.get_segment_size_s())
        self.bandpass_fmax = bandpass_fmax
        self.device = device
        self.onnx_batch_size = onnx_batch_size

    def _encode(self, waveforms: list[np.ndarray], sr: int) -> list[np.ndarray]:
        inputs = [(np.asarray(w, dtype=np.float32).reshape(-1), int(sr)) for w in waveforms]
        result = self.model.encode_arrays(
            inputs,
            bandpass_fmax=self.bandpass_fmax,
            device=self.device,
            batch_size=self.onnx_batch_size,
        )
        emb = np.asarray(result.embeddings, dtype=np.float32)  # [B, max_segments, D]
        hop = result.hop_duration_s
        # slice each clip back to its own segment count (batch pads short clips)
        return [emb[i, : max(1, ceil(dur / hop))] for i, dur in enumerate(result.input_durations)]

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        return self._encode([waveform], sr)[0]

    def embed_batch(self, waveforms: list[np.ndarray], sr: int) -> list[np.ndarray]:
        return self._encode(waveforms, sr)


class PerchTeacher:
    """Google Perch 2.0 embeddings (1536-d) via perch_hoplite.

    Needs TensorFlow >= 2.20; run this teacher inside the isolated `.venv-perch`
    env (perch-hoplite[tf]). `perch_v2_cpu` is the CPU-only preset (Apple Silicon
    has no CUDA, and the plain `perch_v2` preset targets a GPU); `perch_v2_gpu`
    exists for CUDA hosts. The model self-frames at 32 kHz / 5 s windows and emits
    embeddings [frames, channels, 1536]; everything downstream is plain numpy, so
    only extract.py needs this environment.
    """

    name = "perch"

    def __init__(
        self,
        model_name: str = "perch_v2_cpu",
        native_sr: int = 32000,
        window_seconds: float = 5.0,
    ):
        self.native_sr = native_sr
        self.window_seconds = window_seconds
        try:
            from perch_hoplite.zoo import model_configs  # type: ignore[import-not-found]
        except ImportError:
            try:
                from chirp.inference import models as model_configs  # type: ignore[import-not-found]  # older layout
            except ImportError as exc:
                raise RuntimeError(
                    "PerchTeacher needs perch_hoplite (or chirp) with JAX/TF; "
                    "run extract.py for --teacher perch inside the isolated env."
                ) from exc
        self.model = model_configs.load_model_by_name(model_name)

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        wav = _resample(waveform, sr, self.native_sr)
        outputs = self.model.embed(wav)
        emb = np.asarray(outputs.embeddings, dtype=np.float32)
        if emb.ndim == 3:  # [frames, channels, D] -> average the channel axis
            emb = emb.mean(axis=1)
        return emb.reshape(emb.shape[0], -1)

    def embed_batch(self, waveforms: list[np.ndarray], sr: int) -> list[np.ndarray]:
        return [self.embed(w, sr) for w in waveforms]


def build_teacher(name: str, **kwargs) -> ClipTeacher:
    """Factory: map a teacher name to its constructor (keeps CLIs teacher-agnostic)."""
    builders = {"birdnet": BirdNetTeacher, "perch": PerchTeacher}
    if name not in builders:
        raise ValueError(f"unknown teacher {name!r}; choose from {sorted(builders)}")
    return builders[name](**kwargs)
