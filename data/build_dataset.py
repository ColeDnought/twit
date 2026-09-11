#!/usr/bin/env python3
"""Download DCASE2018 Task 3 "Bird Audio Detection" development datasets.

Downloads the three development sets (freefield1010, warblrb10k, and
BirdVox-DCASE-20k) into the layout expected by load_data.py: each dataset gets
its own self-contained folder, e.g. <data-dir>/warblrb10k_public/, containing
a "metadata.csv" (itemid,datasetid,hasbird) plus that dataset's "<itemid>.wav"
files sitting flat alongside it.

See: https://dcase.community/challenge2018/task-bird-audio-detection

No network requests are made on import - only when this script is run (or
build_dataset() is called explicitly). Combined size of all three WAV sets is
roughly 25 GB, so pick --datasets deliberately on a metered connection.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dir_name: str  # folder under data_dir holding metadata.csv + this dataset's wavs
    wav_url: str
    wav_zip_name: str
    labels_url: str
    approx_wav_size_gb: float


DATASETS: dict[str, DatasetSpec] = {
    "freefield1010": DatasetSpec(
        key="freefield1010",
        dir_name="ff1010bird",
        wav_url="https://archive.org/download/ff1010bird/ff1010bird_wav.zip",
        wav_zip_name="ff1010bird_wav.zip",
        labels_url="https://ndownloader.figshare.com/files/10853303",
        approx_wav_size_gb=5.8,
    ),
    "warblrb10k": DatasetSpec(
        key="warblrb10k",
        dir_name="warblrb10k_public",
        wav_url="https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip",
        wav_zip_name="warblrb10k_public_wav.zip",
        labels_url="https://ndownloader.figshare.com/files/10853306",
        approx_wav_size_gb=4.3,
    ),
    "birdvox_dcase_20k": DatasetSpec(
        key="birdvox_dcase_20k",
        dir_name="BirdVox-DCASE-20k",
        wav_url="https://zenodo.org/record/1208080/files/BirdVox-DCASE-20k.zip",
        wav_zip_name="BirdVox-DCASE-20k.zip",
        labels_url="https://ndownloader.figshare.com/files/10853300",
        approx_wav_size_gb=15.4,
    ),
}


def _download(url: str, dest: Path, *, resume: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() and resume else 0
    req = urllib.request.Request(url)
    if existing:
        req.add_header("Range", f"bytes={existing}-")

    with urllib.request.urlopen(req) as resp:
        status = getattr(resp, "status", 200)
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) + existing if total and status == 206 else (int(total) if total else None)
        if status == 200 and existing:
            dest.unlink()
            existing = 0
            total_bytes = int(total) if total else None

        mode = "ab" if existing else "wb"
        downloaded = existing
        with dest.open(mode) as f:
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = 100 * downloaded / total_bytes
                    print(f"\r{dest.name}: {downloaded / 1e6:.1f}/{total_bytes / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
                else:
                    print(f"\r{dest.name}: {downloaded / 1e6:.1f} MB", end="", flush=True)
    print()


def _wavs_present(dataset_dir: Path) -> bool:
    return dataset_dir.is_dir() and any(dataset_dir.glob("*.wav"))


def _build_one(spec: DatasetSpec, data_dir: Path, *, force: bool) -> None:
    dataset_dir = data_dir / spec.dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / spec.wav_zip_name
    labels_path = dataset_dir / "metadata.csv"

    print(f"\n=== {spec.key} ===")
    if force or not _wavs_present(dataset_dir):
        print(f"Downloading WAVs (~{spec.approx_wav_size_gb:.1f} GB) from {spec.wav_url}")
        _download(spec.wav_url, zip_path)
        print(f"Extracting {zip_path} -> {dataset_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dataset_dir)
        # Every zip contains a single top-level "wav/" folder; flatten it so
        # the wav files sit directly alongside metadata.csv in dataset_dir.
        extracted_wav_dir = dataset_dir / "wav"
        for wav_file in extracted_wav_dir.glob("*.wav"):
            wav_file.rename(dataset_dir / wav_file.name)
        extracted_wav_dir.rmdir()
        zip_path.unlink(missing_ok=True)
    else:
        print(f"Skipping WAV download for {spec.key}; found files in {dataset_dir}")

    if force or not labels_path.exists():
        print(f"Downloading labels from {spec.labels_url}")
        _download(spec.labels_url, labels_path, resume=False)
    else:
        print(f"Skipping labels download for {spec.key}; found {labels_path}")

    n_wavs = len(list(dataset_dir.glob("*.wav")))
    print(f"{spec.key}: {n_wavs} wavs + {labels_path.name} in {dataset_dir}")


DEFAULT_DATA_DIR = Path(__file__).resolve().parent


def build_dataset(
    data_dir: str | Path | None = None,
    dataset_keys: list[str] | None = None,
    *,
    force: bool = False,
) -> None:
    """Download the DCASE dev set(s) into `data_dir` (defaults to this package's dir).

    Callable inline from a notebook, e.g. `build_dataset()` for all three sets or
    `build_dataset(dataset_keys=["freefield1010"])` for one. Idempotent: existing
    wavs/labels are skipped unless `force=True`. `dataset_keys` accepts "all".
    """
    data_dir = (Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    keys = dataset_keys or list(DATASETS)
    if "all" in keys:
        keys = list(DATASETS)
    unknown = set(keys) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {sorted(unknown)}. Choose from {list(DATASETS)}")

    for key in keys:
        _build_one(DATASETS[key], data_dir, force=force)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory to hold one subfolder per dataset (default: this script's directory)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[*DATASETS, "all"],
        default=["all"],
        help="Which development dataset(s) to download (default: all three, ~25.5 GB combined)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist")
    args = parser.parse_args(argv)
    keys = list(DATASETS) if "all" in args.datasets else args.datasets
    build_dataset(args.data_dir, keys, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
