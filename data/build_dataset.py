#!/usr/bin/env python3
"""Download the Warblr10k public WAV set and labels into the layout expected by load_data.py."""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

WAV_URL = "https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip"
LABELS_URL = "https://ndownloader.figshare.com/files/10853306"
LABELS_NAME = "warblrb10k_public_metadata_2018.csv"


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


def _wavs_present(wav_dir: Path) -> bool:
    return wav_dir.is_dir() and any(wav_dir.glob("*.wav"))


def build_dataset(data_dir: Path, *, force: bool = False) -> None:
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = data_dir / "wav"
    zip_path = data_dir / "warblrb10k_public_wav.zip"
    labels_dir = data_dir / "warblrb10k_public_wav"
    labels_path = labels_dir / LABELS_NAME

    if force or not _wavs_present(wav_dir):
        print(f"Downloading WAVs from {WAV_URL}")
        _download(WAV_URL, zip_path)
        print(f"Extracting {zip_path} -> {data_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
        zip_path.unlink(missing_ok=True)
    else:
        print(f"Skipping WAV download; found files in {wav_dir}")

    if force or not labels_path.exists():
        print(f"Downloading labels from {LABELS_URL}")
        _download(LABELS_URL, labels_path, resume=False)
    else:
        print(f"Skipping labels download; found {labels_path}")

    n_wavs = len(list(wav_dir.glob("*.wav"))) if wav_dir.exists() else 0
    print(f"development wavs: {n_wavs} in {wav_dir}")
    print(f"labels: {labels_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for wav/ and warblrb10k_public_wav/ (default: this script's directory)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist")
    args = parser.parse_args(argv)
    build_dataset(args.data_dir, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
