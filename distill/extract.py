#!/usr/bin/env python3
"""Dump per-clip teacher embeddings from the full-band clip cache.

Iterates the 32 kHz TwitDataset cache, embeds each clip through the chosen teacher
(in batches for throughput), and writes distill/emb/{teacher}.npz keyed by itemid.
Run once per teacher. BirdNET uses the official `birdnet` package (bundled ONNX
runtime, no TensorFlow); Perch runs in the isolated `.venv-perch` TF env.

Run from the repo root so `data` and `distill` are importable:
    python -m distill.extract --teacher birdnet
    python -m distill.extract --teacher birdnet --birdnet-version 3.0 --batch-size 64
    .venv-perch/bin/python -m distill.extract --teacher perch   # isolated TF env
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.load_data import TwitDataset
from distill.store import save_embeddings
from distill.teachers import build_teacher


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", required=True, choices=["birdnet", "perch"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--sample-rate", type=int, default=32000, help="clip cache rate to read")
    parser.add_argument("--batch-size", type=int, default=64, help="clips per teacher.embed_batch call")
    parser.add_argument("--out", help="output npz (default: distill/emb/{teacher}.npz)")
    parser.add_argument("--limit", type=int, help="debug: only embed the first N clips")
    parser.add_argument("--birdnet-version", default="3.0")
    parser.add_argument("--birdnet-backend", default="onnx")
    args = parser.parse_args(argv)

    teacher_kwargs: dict = {}
    if args.teacher == "birdnet":
        teacher_kwargs.update(version=args.birdnet_version, backend=args.birdnet_backend)
    teacher = build_teacher(args.teacher, **teacher_kwargs)
    print(
        f"teacher={args.teacher} native_sr={teacher.native_sr} "
        f"window={teacher.window_seconds}s reading sr{args.sample_rate} cache"
    )

    dataset = TwitDataset(root_dir=args.data_root, sample_rate=args.sample_rate)
    count = len(dataset) if not args.limit else min(args.limit, len(dataset))

    embeddings: dict[str, np.ndarray] = {}
    for start in tqdm(range(0, count, args.batch_size)):
        stop = min(start + args.batch_size, count)
        waveforms, itemids = [], []
        for index in range(start, stop):
            waveform, _ = dataset[index]
            # TwitDataset returns int16-as-float; teachers expect PCM in [-1, 1]
            waveforms.append(waveform.numpy().astype(np.float32) / 32768.0)
            itemids.append(dataset.labels[index][0].stem)
        for itemid, emb in zip(itemids, teacher.embed_batch(waveforms, args.sample_rate)):
            embeddings[itemid] = emb

    out_path = Path(args.out) if args.out else Path("distill/emb") / f"{args.teacher}.npz"
    save_embeddings(out_path, embeddings)
    dim = next(iter(embeddings.values())).shape[-1] if embeddings else 0
    print(f"wrote {out_path}  ({len(embeddings)} clips, D={dim})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
