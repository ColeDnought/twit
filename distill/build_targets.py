#!/usr/bin/env python3
"""Turn teacher embeddings into a clip-level soft-logit table (+ report the ceiling).

Fits a linear probe (LogisticRegression) on the TRAIN split's per-window
embeddings using your aggregated labels, max-pools the per-window logits to a
clip logit (matching the student's masked global-max presence pooling), then:
  - writes distill/targets/{name}.npz = {itemid: logit} for TRAIN clips (KD targets)
  - prints the held-out TEST AUC = the teacher's usable ceiling on your task

The seeded split is reconstructed exactly as train.ipynb builds it, so the probe
never sees the student's test clips. Pass several teachers to average their
clip logits into an ensemble.

Run from the repo root so `data` and `distill` are importable:
    python -m distill.build_targets --teachers birdnet
    python -m distill.build_targets --teachers birdnet perch          # -> ensemble.npz
    python -m distill.build_targets --teachers perch --name perch --C 0.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data.load_data import aggregate_labels, list_items
from distill.store import load_embeddings, save_targets


def reconstruct_split(root_dir: str, seed: int, train_frac: float):
    """Rebuild train.ipynb's seeded 90/10 split as ordered itemid lists.

    Mirrors `random_split(ds, [train_frac, 1-train_frac], Generator().manual_seed(seed))`
    over range(len(ds)); the permutation depends only on length + seed, so the
    partition matches clip-for-clip.
    """
    items = list_items(root_dir)
    generator = torch.Generator().manual_seed(seed)
    # round the complement so 1 - 0.9 == 0.1 exactly; float dust (0.0999...) would
    # shift random_split's floor and move one clip across the train/test boundary.
    test_frac = round(1.0 - train_frac, 12)
    train_subset, test_subset = torch.utils.data.random_split(
        range(len(items)), [train_frac, test_frac], generator=generator
    )
    train_ids = [items[i][0] for i in train_subset.indices]
    test_ids = [items[i][0] for i in test_subset.indices]
    return train_ids, test_ids


def probe_clip_logits(embeddings, train_ids, test_ids, labels, C):
    """Fit a per-window logistic probe on TRAIN, return clip-level logits per split.

    Every window inherits its clip's (weak) label for fitting; the clip logit is
    the max over its windows' decision scores.
    """
    features, targets = [], []
    for itemid in train_ids:
        windows = embeddings.get(itemid)
        if windows is None:
            continue
        windows = np.asarray(windows, dtype=np.float32).reshape(len(windows), -1)
        features.append(windows)
        targets.append(np.full(len(windows), labels[itemid], dtype=np.int64))
    if not features:
        raise ValueError("no TRAIN embeddings overlap the split; check itemid keys")

    probe = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
    probe.fit(np.concatenate(features), np.concatenate(targets))

    def clip_logit(windows):
        windows = np.asarray(windows, dtype=np.float32).reshape(len(windows), -1)
        return float(probe.decision_function(windows).max())

    train_logits = {i: clip_logit(embeddings[i]) for i in train_ids if i in embeddings}
    test_logits = {i: clip_logit(embeddings[i]) for i in test_ids if i in embeddings}
    return train_logits, test_logits


def report_auc(tag, logits, labels):
    ids = [i for i in logits if i in labels]
    auc = roc_auc_score([labels[i] for i in ids], [logits[i] for i in ids])
    print(f"[{tag}] test AUC ceiling: {auc:.4f}  ({len(ids)} test clips)")
    return auc


def _average(dicts):
    common = set(dicts[0]).intersection(*dicts[1:])
    return {i: float(np.mean([d[i] for d in dicts])) for i in common}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teachers", nargs="+", required=True, help="emb file stem(s) under --emb-dir")
    parser.add_argument("--emb-dir", default="distill/emb")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=42, help="must match the notebook SEED")
    parser.add_argument("--train-frac", type=float, default=0.9)
    parser.add_argument("--C", type=float, default=1.0, help="LogisticRegression inverse-reg strength")
    parser.add_argument("--name", help="output name (default: teacher name, or 'ensemble')")
    parser.add_argument("--out", help="output npz (default: distill/targets/{name}.npz)")
    args = parser.parse_args(argv)

    train_ids, test_ids = reconstruct_split(args.data_root, args.seed, args.train_frac)
    labels = aggregate_labels(args.data_root)
    print(f"split: {len(train_ids)} train / {len(test_ids)} test clips (seed={args.seed})")

    per_teacher_train, per_teacher_test = [], []
    for teacher in args.teachers:
        embeddings = load_embeddings(Path(args.emb_dir) / f"{teacher}.npz")
        train_logits, test_logits = probe_clip_logits(embeddings, train_ids, test_ids, labels, args.C)
        report_auc(teacher, test_logits, labels)
        per_teacher_train.append(train_logits)
        per_teacher_test.append(test_logits)

    if len(args.teachers) > 1:
        train_targets = _average(per_teacher_train)
        report_auc("ensemble", _average(per_teacher_test), labels)
        default_name = "ensemble"
    else:
        train_targets = per_teacher_train[0]
        default_name = args.teachers[0]

    name = args.name or default_name
    out_path = Path(args.out) if args.out else Path("distill/targets") / f"{name}.npz"
    save_targets(out_path, train_targets)
    print(f"wrote {out_path}  ({len(train_targets)} TRAIN soft targets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
