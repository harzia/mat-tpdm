#!/usr/bin/env python3
"""Sanity-check that the loader reproduces natural patch sampling."""
from __future__ import annotations
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from alcu_dataset import build_train_val_datasets

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-root", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--orientation", choices=("xy", "xz"), required=True)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train_ds, val_ds = build_train_val_datasets(
        args.processed_root, args.split, args.orientation,
        patch_size=384, train_samples=args.n,
        val_samples=min(1024, args.n), seed=args.seed,
    )
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    n = crossing = 0
    band_sum = 0.0
    vmin, vmax = float("inf"), float("-inf")
    for batch in loader:
        x = batch["image"]
        n += x.shape[0]
        crossing += int(batch["crosses_interface"].sum())
        band_sum += float(batch["interface_band_fraction"].sum())
        vmin = min(vmin, float(x.min()))
        vmax = max(vmax, float(x.max()))
    print(f"orientation: {args.orientation}")
    print(f"samples: {n}")
    print(f"P(crossing): {crossing / n:.6f}")
    print(f"mean interface-band fraction: {band_sum / n:.8f}")
    print(f"normalized data range: [{vmin:.6f}, {vmax:.6f}]")
    print(f"train volumes: {len(train_ds.volume_dirs)}")
    print(f"val volumes: {len(val_ds.volume_dirs)}")

if __name__ == "__main__":
    main()
