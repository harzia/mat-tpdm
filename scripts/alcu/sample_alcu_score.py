#!/usr/bin/env python3
"""Unconditional 2D samples from an Al-Cu XY/XZ VE-SDE NCSN++ checkpoint.

Uses Yang Song's original score_sde_pytorch predictor/corrector or ODE sampler.
Checkpoint EMA is copied into the network before sampling. Writes raw float32
network-space values (not clipped), plus clipped preview PNGs.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from alcu_config import get_config


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as file:
        np.save(file, array, allow_pickle=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


def save_png_atomic(path: Path, raw_normalized: np.ndarray) -> None:
    # Only the preview is clipped. Scientific arrays stay unmodified.
    im = Image.fromarray(np.rint(np.clip(raw_normalized, 0, 1) * 255).astype('uint8'), 'L')
    temp = path.with_name(path.stem + '.tmp.png')
    im.save(temp)
    os.replace(temp, path)


def dump_json_atomic(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score-sde-root', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--orientation', choices=('xy','xz'), required=True)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--num-samples', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--patch-size', type=int, default=384)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--method', choices=('pc','ode'), default='pc')
    p.add_argument('--predictor', choices=('reverse_diffusion','euler_maruyama','ancestral_sampling'), default='reverse_diffusion')
    p.add_argument('--corrector', choices=('langevin','none','ald'), default='langevin')
    p.add_argument('--snr', type=float, default=0.15)
    p.add_argument('--corrector-steps', type=int, default=1)
    p.add_argument('--steps', type=int, default=2000,
                   help='Sampling VE-SDE discretization; 2000 matches training config')
    p.add_argument('--eps', type=float, default=1e-5)
    p.add_argument('--no-denoise', action='store_true')
    args = p.parse_args()
    if args.num_samples < 1 or args.batch_size < 1 or args.patch_size <= 0 or args.steps < 2:
        p.error('num-samples and batch-size must be positive, steps >= 2')
    if not (0 < args.eps < 1):
        p.error('eps must lie strictly between 0 and 1')
    if args.method == 'pc' and args.snr <= 0 and args.corrector != 'none':
        p.error('snr must be positive for nontrivial corrector')
    if args.corrector_steps < 1:
        p.error('corrector-steps must be >= 1')
    return args


def main():
    args = parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    score_root = args.score_sde_root.resolve(strict=True)
    sys.path.insert(0, str(score_root))

    from models import ncsnpp  # noqa: F401 - model registry
    from models import utils as mutils
    from models.ema import ExponentialMovingAverage
    import sde_lib
    import sampling

    if not torch.cuda.is_available():
        raise SystemExit('This 384x384 NCSN++ sampler expects a CUDA GPU.')
    torch.backends.cudnn.benchmark = False
    config = get_config(batch_size=args.batch_size, n_iters=300000, image_size=args.patch_size)
    config.device = torch.device('cuda:0')
    if not config.training.continuous or config.data.centered:
        raise RuntimeError('Unexpected train config: expected continuous VE-SDE with centered=False')
    if config.model.name != 'ncsnpp' or config.data.num_channels != 1:
        raise RuntimeError('Unexpected model config')

    # Keep run identity independent of any mutable files and of process startup time.
    checkpoint_stat = checkpoint.stat()
    run_id = {
        'orientation': args.orientation,
        'checkpoint': str(checkpoint),
        'checkpoint_size': checkpoint_stat.st_size,
        'checkpoint_mtime_ns': checkpoint_stat.st_mtime_ns,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'patch_size': args.patch_size,
        'seed': args.seed,
        'method': args.method,
        'predictor': args.predictor if args.method == 'pc' else None,
        'corrector': args.corrector if args.method == 'pc' else None,
        'snr': args.snr if args.method == 'pc' else None,
        'corrector_steps': args.corrector_steps if args.method == 'pc' else None,
        'sampling_steps': args.steps,
        'eps': args.eps,
        'denoise': not args.no_denoise,
        'model_sigma_min': float(config.model.sigma_min),
        'model_sigma_max': float(config.model.sigma_max),
        'model_training_scales': int(config.model.num_scales),
        'data_centered': bool(config.data.centered),
    }
    outdir = args.output_dir.resolve()
    images_dir = outdir / 'png'
    arrays_dir = outdir / 'npy'
    outdir.mkdir(parents=True, exist_ok=True)
    identity_path = outdir / 'run_config.json'
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding='utf-8'))
        differing = [key for key, value in run_id.items() if existing.get(key) != value]
        if differing:
            raise SystemExit('Output directory belongs to a different run (' + ', '.join(differing) + '). Use another --output-dir.')
    elif list(outdir.glob('*.npy')) or (outdir / 'npy').exists():
        raise SystemExit('Output directory already has outputs but no run_config.json; use a new path.')
    images_dir.mkdir(exist_ok=True)
    arrays_dir.mkdir(exist_ok=True)

    training_config_path = checkpoint.parent.parent / 'training_config.json'
    if training_config_path.is_file():
        training_record = json.loads(training_config_path.read_text(encoding='utf-8'))
        if training_record.get('orientation') != args.orientation:
            raise RuntimeError(f'Training orientation in {training_config_path} differs from {args.orientation}')
        train_cfg = training_record.get('score_sde_config', {})
        if int(train_cfg.get('data', {}).get('image_size', args.patch_size)) != args.patch_size:
            raise RuntimeError('Sampling patch size differs from training image_size')
        model_cfg = train_cfg.get('model', {})
        for key in ('sigma_min', 'sigma_max', 'num_scales'):
            if key in model_cfg and float(model_cfg[key]) != float(getattr(config.model, key)):
                raise RuntimeError(f'Training config {key} differs from sampling model config')
        print('Verified training configuration:', training_config_path, flush=True)

    print('Loading checkpoint:', checkpoint, flush=True)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or not {'model','ema','step'} <= payload.keys():
        raise RuntimeError('Checkpoint must have model, ema and step')
    step = int(payload['step'])
    print(f'Loaded checkpoint step={step} keys={sorted(payload)}', flush=True)
    # Model architecture and param order match training exactly (DataParallel wrapper).
    model = mutils.create_model(config)
    model.load_state_dict(payload['model'], strict=True)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    ema.load_state_dict(payload['ema'])
    ema.copy_to(model.parameters())
    model.eval()
    del payload, ema
    gc.collect()

    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding='utf-8')).get('checkpoint_step') != step:
            raise RuntimeError('Checkpoint step changed since first run; select a fresh output path')
    else:
        dump_json_atomic(identity_path, {**run_id, 'checkpoint_step': step,
                                        'created_utc': datetime.now(timezone.utc).isoformat()})

    config.sampling = type(config.model)()
    config.sampling.method = args.method
    config.sampling.predictor = args.predictor
    config.sampling.corrector = args.corrector
    config.sampling.snr = args.snr
    config.sampling.n_steps_each = args.corrector_steps
    config.sampling.probability_flow = False
    config.sampling.noise_removal = not args.no_denoise

    # These samples were trained directly on x=(clip(phi,-1,1)+1)/2.
    # centered=False => no second normalization, inverse_scaler is identity.
    inverse_scaler = lambda x: x

    total_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    for batch_idx in range(total_batches):
        start = batch_idx * args.batch_size
        end = min(args.num_samples, start + args.batch_size)
        batch_indices = range(start, end)
        missing = [i for i in batch_indices if not (arrays_dir / f'sample_{i:05d}.npy').is_file()]
        if not missing:
            print(f'Batch {batch_idx + 1}/{total_batches}: already saved, skipping', flush=True)
        else:
            batch_seed = args.seed + batch_idx
            torch.manual_seed(batch_seed)
            torch.cuda.manual_seed_all(batch_seed)
            np.random.seed(batch_seed % (2**32))
            random.seed(batch_seed)
            shape = (end - start, 1, args.patch_size, args.patch_size)
            sde = sde_lib.VESDE(sigma_min=config.model.sigma_min,
                                sigma_max=config.model.sigma_max, N=args.steps)
            sde.discrete_sigmas = sde.discrete_sigmas.to(config.device)
            sampler = sampling.get_sampling_fn(config, sde, shape, inverse_scaler, args.eps)
            print(f'Batch {batch_idx+1}/{total_batches}: generating samples {start}..{end-1} '
                  f'(seed={batch_seed})', flush=True)
            with torch.inference_mode():
                generated, nfe = sampler(model)
            generated = generated.detach().float().cpu().numpy()
            if generated.shape != shape:
                raise RuntimeError(f'Unexpected sampler shape {generated.shape}, expected {shape}')
            for pos, i in enumerate(batch_indices):
                sample = np.array(generated[pos, 0], dtype=np.float32, copy=True)
                if not np.all(np.isfinite(sample)):
                    raise RuntimeError(f'Nonfinite sample #{i}: not saving')
                dst = arrays_dir / f'sample_{i:05d}.npy'
                if not dst.exists():
                    save_npy_atomic(dst, sample)
                save_png_atomic(images_dir / f'sample_{i:05d}.png', sample)
            print(f'Batch {batch_idx+1}/{total_batches} done, NFE={nfe}', flush=True)
            del generated
            torch.cuda.empty_cache()
        # Repair missing previews after interruption, without resampling.
        for i in batch_indices:
            png = images_dir / f'sample_{i:05d}.png'
            if not png.exists():
                src = np.load(arrays_dir / f'sample_{i:05d}.npy', allow_pickle=False)
                save_png_atomic(png, src)

    print(f'COMPLETE: {args.num_samples} raw normalized .npy files and clipped PNGs in {outdir}', flush=True)


if __name__ == '__main__':
    main()