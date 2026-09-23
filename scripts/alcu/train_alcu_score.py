#!/usr/bin/env python3
"""Train one TPDM constituent score model (XY or XZ) on Al-Cu patches.

Uses yang-song/score_sde_pytorch for NCSN++, VE-SDE loss, EMA, optimizer helpers,
and checkpoint format, while using the native PyTorch loader in alcu_dataset.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from alcu_dataset import build_train_val_datasets
from alcu_config import get_config




def save_checkpoint(path, state):
    payload = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }
    torch.save(payload, path)


def restore_checkpoint(path, state, device):
    path = Path(path)
    if not path.exists():
        return state
    payload = torch.load(path, map_location=device)
    state["optimizer"].load_state_dict(payload["optimizer"])
    state["model"].load_state_dict(payload["model"], strict=False)
    state["ema"].load_state_dict(payload["ema"])
    state["step"] = int(payload["step"])
    return state

def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def config_to_plain(config):
    d = config.to_dict()
    d["device"] = str(config.device)
    return d


def mean_validation_loss(eval_step_fn, state, loader, device, max_batches):
    losses = []
    crossings = 0
    samples = 0
    it = iter(loader)
    for _ in range(max_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        x = batch["image"].to(device, non_blocking=True, dtype=torch.float32)
        loss = eval_step_fn(state, x)
        losses.append(float(loss.item()))
        crossings += int(batch["crosses_interface"].sum())
        samples += int(x.shape[0])
    return (
        float(np.mean(losses)) if losses else float("nan"),
        crossings / samples if samples else float("nan"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--score-sde-root", type=Path, required=True)
    p.add_argument("--processed-root", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--orientation", choices=("xy", "xz"), required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--patch-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--n-iters", type=int, default=300000)
    p.add_argument("--train-samples-per-epoch", type=int, default=50000)
    p.add_argument("--val-samples", type=int, default=1024)
    p.add_argument("--val-batches", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-freq", type=int, default=50)
    p.add_argument("--eval-freq", type=int, default=1000)
    p.add_argument("--checkpoint-freq", type=int, default=10000)
    p.add_argument("--preemption-freq", type=int, default=1000)
    args = p.parse_args()

    score_sde_root = args.score_sde_root.resolve()
    if not score_sde_root.exists():
        raise SystemExit(f"score_sde_pytorch not found: {score_sde_root}")
    sys.path.insert(0, str(score_sde_root))

    # ncsnpp import registers the model with models.utils.
    from models import ncsnpp  # noqa: F401
    from models import utils as mutils
    from models.ema import ExponentialMovingAverage
    import losses
    import sde_lib

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = get_config(args.batch_size, args.n_iters, args.patch_size)
    config.seed = args.seed

    workdir = args.workdir.resolve()
    checkpoint_dir = workdir / "checkpoints"
    meta_dir = workdir / "checkpoints-meta"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    writer = SummaryWriter(str(workdir / "tensorboard"))

    with open(workdir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "orientation": args.orientation,
            "processed_root": str(args.processed_root.resolve()),
            "split": str(args.split.resolve()),
            "loader": {
                "natural_sampling": "uniform timepoint -> uniform valid patch",
                "patch_size": args.patch_size,
                "normalization": "clip phi to [-1,1], then map to [0,1]",
                "train_samples_per_epoch": args.train_samples_per_epoch,
                "val_samples": args.val_samples,
                "seed": args.seed,
            },
            "score_sde_config": config_to_plain(config),
        }, f, indent=2, default=str)

    train_ds, val_ds = build_train_val_datasets(
        args.processed_root, args.split, args.orientation,
        patch_size=args.patch_size,
        train_samples=args.train_samples_per_epoch,
        val_samples=args.val_samples,
        seed=args.seed,
        interface_eps=0.1,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=False, persistent_workers=args.num_workers > 0,
    )
    train_iter = cycle(train_loader)

    logging.info("orientation=%s train_volumes=%d val_volumes=%d",
                 args.orientation, len(train_ds.volume_dirs), len(val_ds.volume_dirs))
    logging.info("device=%s GPUs=%d batch_size=%d patch=%d",
                 config.device, torch.cuda.device_count(), args.batch_size, args.patch_size)

    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = {"optimizer": optimizer, "model": score_model, "ema": ema, "step": 0}

    meta_path = meta_dir / "checkpoint.pth"
    state = restore_checkpoint(str(meta_path), state, config.device)
    initial_step = int(state["step"])

    sde = sde_lib.VESDE(
        sigma_min=config.model.sigma_min,
        sigma_max=config.model.sigma_max,
        N=config.model.num_scales,
    )
    optimize_fn = losses.optimization_manager(config)
    train_step_fn = losses.get_step_fn(
        sde, train=True, optimize_fn=optimize_fn,
        reduce_mean=config.training.reduce_mean,
        continuous=config.training.continuous,
        likelihood_weighting=config.training.likelihood_weighting,
    )
    eval_step_fn = losses.get_step_fn(
        sde, train=False, optimize_fn=optimize_fn,
        reduce_mean=config.training.reduce_mean,
        continuous=config.training.continuous,
        likelihood_weighting=config.training.likelihood_weighting,
    )

    logging.info("Starting at step %d; target=%d", initial_step, args.n_iters)
    t0 = time.time()

    for _ in range(initial_step, args.n_iters):
        batch = next(train_iter)
        x = batch["image"].to(config.device, non_blocking=True, dtype=torch.float32)
        loss = train_step_fn(state, x)
        step = int(state["step"])

        crossing_rate = float(batch["crosses_interface"].float().mean())
        band_fraction = float(batch["interface_band_fraction"].float().mean())

        if step % args.log_freq == 0:
            logging.info(
                "step=%d loss=%.6e crossing=%.3f band_fraction=%.6g elapsed=%.1fs",
                step, float(loss.item()), crossing_rate, band_fraction, time.time()-t0)
            writer.add_scalar("train/loss", float(loss.item()), step)
            writer.add_scalar("train/interface_crossing_rate", crossing_rate, step)
            writer.add_scalar("train/interface_band_fraction", band_fraction, step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

        if step % args.preemption_freq == 0:
            save_checkpoint(str(meta_path), state)

        if step % args.eval_freq == 0:
            val_loss, val_crossing = mean_validation_loss(
                eval_step_fn, state, val_loader, config.device, args.val_batches)
            logging.info("step=%d val_loss=%.6e val_crossing=%.3f",
                         step, val_loss, val_crossing)
            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("val/interface_crossing_rate", val_crossing, step)

        if step % args.checkpoint_freq == 0 or step == args.n_iters:
            ckpt_path = checkpoint_dir / f"checkpoint_{step:09d}.pth"
            save_checkpoint(str(ckpt_path), state)
            save_checkpoint(str(meta_path), state)
            logging.info("Saved %s", ckpt_path)

    writer.flush()
    writer.close()
    logging.info("Training complete.")

if __name__ == "__main__":
    main()
