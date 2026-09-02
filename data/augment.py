"""Train-only waveform augmentation for robust, device-realistic bird detection.

The parametric microphone filter includes a DC-blocking high-pass approximation.
Firmware must apply a matching high-pass (nominally 120 Hz) before inference so
the training and deployment signal chains remain consistent.
"""

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import torch
import torch.nn.functional as F


_EPS = 1e-12


@dataclass(frozen=True)
class AugmentConfig:
    """Configuration for Tier 1/2 waveform and device-input augmentation."""

    enabled: bool = True
    apply_prob: float = 1.0
    sample_rate: int = 16_000

    time_shift_prob: float = 0.5
    max_time_shift_seconds: float = 0.5
    random_crop_prob: float = 0.0
    min_crop_fraction: float = 0.75
    crop_positive_clips: bool = False

    background_mix_prob: float = 0.5
    background_snr_db: tuple[float, float] = (-5.0, 20.0)
    pink_noise_prob: float = 0.5
    pink_noise_snr_db: tuple[float, float] = (3.0, 25.0)

    mic_filter_enabled: bool = True
    mic_filter_taps: int = 63
    mic_filter_variants: int = 8
    mic_highpass_hz: tuple[float, float] = (80.0, 160.0)
    mic_upper_shelf_db: tuple[float, float] = (0.5, 2.5)
    mic_nyquist_droop_db: tuple[float, float] = (-10.0, -5.0)

    level_dbfs: tuple[float, float] = (-60.0, -20.0)
    self_noise_dbfs: float = -90.0
    requantize_int16: bool = True

    def __post_init__(self) -> None:
        probabilities = (
            self.apply_prob,
            self.time_shift_prob,
            self.random_crop_prob,
            self.background_mix_prob,
            self.pink_noise_prob,
        )
        if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
            raise ValueError("augmentation probabilities must be in [0, 1]")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.max_time_shift_seconds < 0:
            raise ValueError("max_time_shift_seconds cannot be negative")
        if not 0.0 < self.min_crop_fraction <= 1.0:
            raise ValueError("min_crop_fraction must be in (0, 1]")
        if self.mic_filter_taps < 3 or self.mic_filter_taps % 2 == 0:
            raise ValueError("mic_filter_taps must be an odd integer >= 3")
        if self.mic_filter_variants < 1:
            raise ValueError("mic_filter_variants must be positive")
        for name in (
            "background_snr_db",
            "pink_noise_snr_db",
            "mic_highpass_hz",
            "mic_upper_shelf_db",
            "mic_nyquist_droop_db",
            "level_dbfs",
        ):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} must be ordered (low, high)")


class AudioAugmenter:
    """Apply label-preserving waveform transforms before batch padding.

    Background donors are drawn only from clips whose human label is negative,
    so mixing preserves binary presence labels and does not alter the effective
    class prior. Random cropping is disabled by default and, unless explicitly
    requested, is restricted to negatives because a weakly labeled positive can
    lose its only bird event when cropped.
    """

    def __init__(self, config: AugmentConfig):
        self.config = config
        self._mic_kernels = self._build_mic_kernel_bank() if config.mic_filter_enabled else None

    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        return x.square().mean().clamp_min(_EPS).sqrt()

    @staticmethod
    def _uniform(bounds: tuple[float, float]) -> float:
        low, high = bounds
        if low == high:
            return low
        return low + (high - low) * torch.rand(()).item()

    def _time_transform(self, x: torch.Tensor, is_positive: bool) -> torch.Tensor:
        cfg = self.config
        if x.numel() > 1 and torch.rand(()).item() < cfg.time_shift_prob:
            max_shift = min(
                x.numel() - 1,
                int(round(cfg.max_time_shift_seconds * cfg.sample_rate)),
            )
            if max_shift > 0:
                shift = int(torch.randint(-max_shift, max_shift + 1, ()).item())
                x = torch.roll(x, shift)

        may_crop = cfg.crop_positive_clips or not is_positive
        if (
            may_crop
            and x.numel() > 1
            and torch.rand(()).item() < cfg.random_crop_prob
        ):
            min_length = max(1, ceil(x.numel() * cfg.min_crop_fraction))
            crop_length = int(torch.randint(min_length, x.numel() + 1, ()).item())
            start = int(torch.randint(0, x.numel() - crop_length + 1, ()).item())
            x = x[start : start + crop_length]
        return x

    @staticmethod
    def _fit_background(background: torch.Tensor, length: int) -> torch.Tensor:
        if background.numel() >= length:
            max_start = background.numel() - length
            start = int(torch.randint(0, max_start + 1, ()).item())
            return background[start : start + length]
        repeats = ceil(length / max(background.numel(), 1))
        return background.repeat(repeats)[:length]

    def _mix_at_snr(
        self,
        signal: torch.Tensor,
        addition: torch.Tensor,
        snr_db: float,
    ) -> torch.Tensor:
        addition_rms = self._rms(addition)
        if addition_rms.item() <= _EPS**0.5:
            return signal
        signal_rms = self._rms(signal)
        scale = signal_rms / (addition_rms * (10.0 ** (snr_db / 20.0)))
        return signal + addition * scale

    @staticmethod
    def _pink_noise_like(x: torch.Tensor) -> torch.Tensor:
        """Generate unit-RMS FFT-shaped pink noise on x's device."""
        n = x.numel()
        spectrum = torch.fft.rfft(torch.randn(n, device=x.device, dtype=x.dtype))
        frequencies = torch.fft.rfftfreq(n, device=x.device, dtype=x.dtype)
        shaping = frequencies.clamp_min(1.0 / max(n, 1)).rsqrt()
        shaping[0] = 0.0
        noise = torch.fft.irfft(spectrum * shaping, n=n)
        return noise / noise.square().mean().clamp_min(_EPS).sqrt()

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _build_mic_kernel(
        self,
        highpass_hz: float,
        upper_shelf_db: float,
        nyquist_droop_db: float,
    ) -> torch.Tensor:
        cfg = self.config
        n_fft = max(256, 4 * cfg.mic_filter_taps)
        if n_fft % 2:
            n_fft += 1
        frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / cfg.sample_rate)
        nyquist = cfg.sample_rate / 2.0

        highpass = frequencies / torch.sqrt(frequencies.square() + highpass_hz**2)
        shelf = self._smoothstep((frequencies - 3_500.0) / 2_500.0)
        droop = self._smoothstep(
            (frequencies - 6_500.0) / max(nyquist - 6_500.0, 1.0)
        )
        response_db = upper_shelf_db * shelf + nyquist_droop_db * droop
        response = highpass * torch.pow(10.0, response_db / 20.0)

        reference_bin = torch.argmin((frequencies - 1_000.0).abs())
        response = response / response[reference_bin].clamp_min(_EPS)
        impulse = torch.fft.irfft(response, n=n_fft)
        impulse = torch.roll(impulse, n_fft // 2)
        center = n_fft // 2
        radius = cfg.mic_filter_taps // 2
        kernel = impulse[center - radius : center + radius + 1]
        kernel = kernel * torch.hann_window(cfg.mic_filter_taps, periodic=False)
        kernel = kernel - kernel.mean()

        actual = torch.fft.rfft(kernel, n=n_fft).abs()
        kernel = kernel / actual[reference_bin].clamp_min(_EPS)
        return kernel.float()

    def _build_mic_kernel_bank(self) -> torch.Tensor:
        cfg = self.config
        kernels = []
        count = cfg.mic_filter_variants
        for index in range(count):
            fraction = (index + 0.5) / count
            reverse_fraction = 1.0 - fraction
            hp = cfg.mic_highpass_hz[0] + fraction * (
                cfg.mic_highpass_hz[1] - cfg.mic_highpass_hz[0]
            )
            shelf = cfg.mic_upper_shelf_db[0] + reverse_fraction * (
                cfg.mic_upper_shelf_db[1] - cfg.mic_upper_shelf_db[0]
            )
            droop = cfg.mic_nyquist_droop_db[0] + fraction * (
                cfg.mic_nyquist_droop_db[1] - cfg.mic_nyquist_droop_db[0]
            )
            kernels.append(self._build_mic_kernel(hp, shelf, droop))
        return torch.stack(kernels)

    def _apply_mic_filter(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        if self._mic_kernels is None or not waveforms:
            return waveforms
        lengths = [waveform.numel() for waveform in waveforms]
        padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        indices = torch.randint(0, self._mic_kernels.shape[0], (len(waveforms),))
        kernels = self._mic_kernels[indices].to(device=padded.device, dtype=padded.dtype)

        # Grouped convolution applies a different FIR to each batch item at once.
        channels_first = padded.unsqueeze(0)
        filtered = F.conv1d(
            channels_first,
            kernels.unsqueeze(1),
            padding=self.config.mic_filter_taps // 2,
            groups=len(waveforms),
        ).squeeze(0)
        return [filtered[index, :length] for index, length in enumerate(lengths)]

    def _set_level_and_quantize(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        target_rms = 10.0 ** (self._uniform(cfg.level_dbfs) / 20.0)
        x = x * (target_rms / self._rms(x))

        noise_rms = 10.0 ** (cfg.self_noise_dbfs / 20.0)
        if noise_rms > 0.0:
            noise = torch.randn_like(x)
            noise = noise / self._rms(noise) * noise_rms
            x = x + noise

        x = x.clamp(-1.0, 1.0)
        if cfg.requantize_int16:
            x = torch.round(x * 32_768.0).clamp(-32_768.0, 32_767.0) / 32_768.0
        return x

    def __call__(
        self,
        waveforms: Sequence[torch.Tensor],
        labels: Sequence[bool | int | float],
    ) -> tuple[list[torch.Tensor], Sequence[bool | int | float]]:
        if len(waveforms) != len(labels):
            raise ValueError("waveforms and labels must have the same length")
        xs = [waveform.float().flatten() for waveform in waveforms]
        if not self.config.enabled:
            return xs, labels

        positives = [bool(label) for label in labels]
        selected = [
            torch.rand(()).item() < self.config.apply_prob for _ in waveforms
        ]
        xs = [
            self._time_transform(waveform, is_positive) if augment else waveform
            for waveform, is_positive, augment in zip(xs, positives, selected)
        ]

        donor_indices = [index for index, positive in enumerate(positives) if not positive]
        pre_mix = [waveform.clone() for waveform in xs]
        if donor_indices:
            for index, signal in enumerate(xs):
                if (
                    not selected[index]
                    or torch.rand(()).item() >= self.config.background_mix_prob
                ):
                    continue
                choices = [candidate for candidate in donor_indices if candidate != index]
                if not choices:
                    choices = donor_indices
                donor_index = choices[int(torch.randint(0, len(choices), ()).item())]
                background = self._fit_background(pre_mix[donor_index], signal.numel())
                xs[index] = self._mix_at_snr(
                    signal,
                    background,
                    self._uniform(self.config.background_snr_db),
                )

        for index, signal in enumerate(xs):
            if (
                selected[index]
                and torch.rand(()).item() < self.config.pink_noise_prob
            ):
                xs[index] = self._mix_at_snr(
                    signal,
                    self._pink_noise_like(signal),
                    self._uniform(self.config.pink_noise_snr_db),
                )

        selected_indices = [index for index, augment in enumerate(selected) if augment]
        filtered = self._apply_mic_filter([xs[index] for index in selected_indices])
        for index, waveform in zip(selected_indices, filtered):
            xs[index] = self._set_level_and_quantize(waveform)
        return xs, labels
