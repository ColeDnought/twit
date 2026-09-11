from concurrent.futures import ProcessPoolExecutor
from csv import DictReader
from functools import partial
from math import gcd
from pathlib import Path
from shutil import copyfile
from typing import Callable, Protocol, Sequence
from urllib.request import urlopen

import os
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import Subset


DATASET_DIRS = ["warblrb10k_public", "ff1010bird", "BirdVox-DCASE-20k"]
METADATA_URLS = {
    "warblrb10k_public": "https://ndownloader.figshare.com/files/10853306",
    "ff1010bird": "https://ndownloader.figshare.com/files/10853303",
    "BirdVox-DCASE-20k": "https://ndownloader.figshare.com/files/10853300",
}


def _resample_file(args: tuple[Path, Path, int]) -> None:
    """Resample one wav to target_sr and write it to the cache; skips work already done."""
    src_path, dst_path, target_sr = args
    if dst_path.exists():
        return
    sr, data = wavfile.read(src_path)
    if sr != target_sr:
        # polyphase resample to the model's expected rate (dev sets are 44.1 kHz)
        g = gcd(sr, target_sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(data.dtype)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(dst_path, target_sr, data)


def _download_metadata(url: str, destination: Path) -> None:
    """Download a small label CSV atomically; audio is never downloaded here."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    try:
        with urlopen(url, timeout=30) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _find_metadata(root_dir: str | Path, dataset_name: str) -> Path | None:
    """Locate a dataset's metadata.csv, preferring the raw layout then any rate cache.

    Labels are sample-rate independent, so a metadata.csv copied into any
    data/.cache/srN/<dataset>/ is just as valid as the raw one.
    """
    root = Path(root_dir)
    candidates = [root / dataset_name / "metadata.csv"]
    cache = root / ".cache"
    if cache.is_dir():
        candidates += sorted(cache.glob(f"sr*/{dataset_name}/metadata.csv"))
    return next((path for path in candidates if path.exists()), None)


def list_items(root_dir: str | Path = "data") -> list[tuple[str, bool]]:
    """(itemid, label) for every clip in TwitDataset index order.

    Reproduces the exact DATASET_DIRS x CSV-row ordering that TwitDataset builds,
    so a seeded random_split over range(len(items)) partitions the same clips the
    training notebook does -- letting the distillation probe stay leak-free.
    """
    items: list[tuple[str, bool]] = []
    for dataset_name in DATASET_DIRS:
        metadata = _find_metadata(root_dir, dataset_name)
        if metadata is None:
            raise FileNotFoundError(
                f"no metadata.csv found for {dataset_name} under {root_dir}"
            )
        with open(metadata, "r") as f:
            for row in DictReader(f):
                items.append((row["itemid"], bool(int(row["hasbird"]))))
    return items


def aggregate_labels(
    root_dir: str | Path = "data",
    cache_path: str | Path | None = None,
    rebuild: bool = False,
) -> dict[str, int]:
    """Merge the per-dataset metadata.csv files into one cached {itemid: label} table.

    This single file is the shared source of truth for the distillation tooling
    (distill/build_targets.py) so labels/itemid keys stay consistent without
    re-parsing every dataset CSV on each run. Built once, then reused.
    """
    root = Path(root_dir)
    cache_path = Path(cache_path) if cache_path is not None else root / ".cache" / "labels.npz"
    if cache_path.exists() and not rebuild:
        return load_labels(cache_path)

    items = list_items(root)
    itemids = np.array([itemid for itemid, _ in items])
    labels = np.array([int(label) for _, label in items], dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, itemids=itemids, labels=labels)
    return dict(zip(itemids.tolist(), labels.tolist()))


def load_labels(cache_path: str | Path = "data/.cache/labels.npz") -> dict[str, int]:
    """Read the aggregated {itemid: label} table written by aggregate_labels."""
    data = np.load(cache_path, allow_pickle=False)
    return dict(zip(data["itemids"].tolist(), data["labels"].tolist()))


class TwitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str | Path = "data", sample_rate: int = 16000, cache_workers: int | None = None):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate  # every cached wav is resampled to this rate
        self.cache_dir = self.root_dir / ".cache" / f"sr{sample_rate}"
        self.labels: list[tuple[Path, bool]] = []
        to_resample: list[tuple[Path, Path, int]] = []
        for dataset_dir in DATASET_DIRS:
            to_resample.extend(self._load_dataset(self.root_dir / dataset_dir))

        if to_resample:
            # One-time cache build: pay the resample cost once (in parallel across
            # cores) instead of on every __getitem__ call of every training epoch.
            print(f"Resampling {len(to_resample)} file(s) to {sample_rate} Hz (cached under {self.cache_dir})...")
            with ProcessPoolExecutor(max_workers=cache_workers or os.cpu_count()) as pool:
                list(pool.map(_resample_file, to_resample))

    def _load_dataset(self, dataset_dir: Path) -> list[tuple[Path, Path, int]]:
        cache_dataset_dir = self.cache_dir / dataset_dir.name
        metadata = dataset_dir / "metadata.csv"
        cached_metadata = cache_dataset_dir / "metadata.csv"
        if metadata.exists():
            if not cached_metadata.exists():
                cached_metadata.parent.mkdir(parents=True, exist_ok=True)
                copyfile(metadata, cached_metadata)
            metadata = cached_metadata
        elif cached_metadata.exists():
            metadata = cached_metadata
        elif cache_dataset_dir.is_dir() and next(cache_dataset_dir.glob("*.wav"), None):
            print(f"Recovering labels for cached dataset {dataset_dir.name}...")
            try:
                _download_metadata(METADATA_URLS[dataset_dir.name], cached_metadata)
            except Exception as exc:
                raise FileNotFoundError(
                    f"cached audio exists at {cache_dataset_dir}, but its labels are "
                    f"missing and could not be downloaded from the official DCASE URL"
                ) from exc
            metadata = cached_metadata
        else:
            raise FileNotFoundError(
                f"missing {metadata}; build the dataset before constructing TwitDataset"
            )

        to_resample: list[tuple[Path, Path, int]] = []
        with open(metadata, "r") as f:
            for row in DictReader(f):
                src_path = dataset_dir / f"{row['itemid']}.wav"
                cache_path = cache_dataset_dir / src_path.name
                if not cache_path.exists():
                    to_resample.append((src_path, cache_path, self.sample_rate))
                self.labels.append((cache_path, bool(int(row["hasbird"]))))
        return to_resample

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, bool]:
        audio_path, label = self.labels[idx]
        # already resampled to self.sample_rate when the cache was built
        _, data = wavfile.read(audio_path)
        return torch.from_numpy(data.astype("float32")), label

class WaveformAugmenter(Protocol):
    def __call__(
        self,
        waveforms: Sequence[torch.Tensor],
        labels: Sequence[bool | int | float],
    ) -> tuple[list[torch.Tensor], Sequence[bool | int | float]]: ...


def _pad_and_stack(
    xs: Sequence[torch.Tensor],
    ys: Sequence[bool | int | float],
    ts: Sequence[float] | None = None,
):
    """Pad normalized variable-length waveforms and retain their valid lengths.

    When per-clip teacher targets `ts` are supplied (distillation path), they are
    stacked and appended, so the batch becomes (x, y, lengths, targets).
    """
    lengths = torch.tensor([x.shape[-1] for x in xs])
    x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    y = torch.tensor(ys, dtype=torch.float32)
    if ts is None:
        return x.unsqueeze(1), y, lengths
    t = torch.tensor(ts, dtype=torch.float32)
    return x.unsqueeze(1), y, lengths, t


def collate_fn(batch):
    """Unaugmented eval collate: normalize, pad, and return valid lengths."""
    xs, ys = zip(*batch)
    xs = [x.float() / 32768.0 for x in xs]
    return _pad_and_stack(xs, ys)


def _augmenting_collate(batch, augmenter: WaveformAugmenter):
    # DistillDataset yields (waveform, label, teacher_logit); plain sets yield (waveform, label).
    if len(batch[0]) == 3:
        xs, ys, ts = zip(*batch)
    else:
        xs, ys, ts = *zip(*batch), None
    xs = [x.float() / 32768.0 for x in xs]
    xs, ys = augmenter(xs, ys)
    return _pad_and_stack(xs, ys, ts)


def make_augmenting_collate(augmenter: WaveformAugmenter) -> Callable:
    """Build a picklable train-only collate around a waveform augmenter."""
    return partial(_augmenting_collate, augmenter=augmenter)


def get_class_imbalance(dataset: torch.utils.data.Dataset) -> torch.Tensor:
    """neg/pos count ratio for BCEWithLogitsLoss pos_weight (train split, not global)."""
    if isinstance(dataset, Subset):
        base, indices = dataset.dataset, dataset.indices
    else:
        base, indices = dataset, range(len(dataset))
    ys = [base.labels[i][1] for i in indices]
    n_pos = sum(ys)
    n_neg = len(ys) - n_pos
    return torch.tensor(n_neg / n_pos)


def dataset_itemid(dataset: torch.utils.data.Dataset, idx: int) -> str:
    """itemid (wav stem) for position `idx`, transparently unwrapping Subsets."""
    while isinstance(dataset, Subset):
        idx = dataset.indices[idx]
        dataset = dataset.dataset
    return dataset.labels[idx][0].stem


class DistillDataset(torch.utils.data.Dataset):
    """Attach a per-clip teacher logit (looked up by itemid) to a (waveform, label) set.

    Wrap the train split only: __getitem__ returns (waveform, label, teacher_logit).
    `targets` may be a path to a build_targets.py npz (loaded here) or a preloaded
    {itemid: logit} dict. Clips absent from the table yield NaN so the Trainer drops
    them from the KD term while still using their hard label. The underlying dataset
    is untouched, so the eval split keeps using the plain collate_fn.
    """

    def __init__(self, base: torch.utils.data.Dataset, targets: str | Path | dict[str, float]):
        self.base = base
        if isinstance(targets, (str, Path)):
            data = np.load(targets, allow_pickle=False)
            targets = dict(zip(data["itemids"].tolist(), data["logits"].tolist()))
        self.targets = targets

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        waveform, label = self.base[idx]
        itemid = dataset_itemid(self.base, idx)
        logit = self.targets.get(itemid, float("nan"))
        return waveform, label, float(logit)
