#!/usr/bin/env python3
"""Rank the longest phi=0 interface boundaries in existing Al-Cu patch samples.

Reads existing distribution-evaluation per_patch_metrics.csv; it does NOT run
score-SDE, perform diffusion sampling, or load any held-out test volumes.
Generated images are read from the saved raw .npy samples; selected real patches
are reconstructed from the exact train/val (volume, coordinate) rows in CSV.

The ranking metric matches evaluate_alcu_distributions.py:
    number of 4-neighbor pixel pairs whose phi signs differ / patch pixel count.
This is a 2D pixel-grid proxy, NOT Euclidean contour length or 3D surface area.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCORE = 'interface_edge_length_per_pixel_area'
GROUPS = ('generated', 'train', 'val')


def font_at(size: int):
    paths = (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    )
    for name in paths:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def parse_rows(path: Path) -> dict[str, list[dict]]:
    rows = {k: [] for k in GROUPS}
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        needed = {'source', 'index', 'volume', 'fixed_index', 'start_a', 'start_b', SCORE}
        missing = needed.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: columns missing: {sorted(missing)}')
        for record in reader:
            source = record['source']
            if source not in rows:
                raise ValueError(f'Unknown source {source!r} in {path}')
            record['index'] = int(record['index'])
            record[SCORE] = float(record[SCORE])
            if not math.isfinite(record[SCORE]):
                raise ValueError(f'Non-finite score for {source} #{record["index"]}')
            rows[source].append(record)
    for source, values in rows.items():
        if not values:
            raise ValueError(f'No {source} rows found in {path}')
        ids = [v['index'] for v in values]
        if len(set(ids)) != len(ids):
            raise ValueError(f'Duplicate {source} sample indices')
        values.sort(key=lambda r: (-r[SCORE], r['index']))
    return rows


def get_patch(record: dict, orientation: str, samples_dir: Path,
              processed_root: Path, summary: dict, patch_size: int) -> np.ndarray:
    source = record['source']
    idx = record['index']
    if source == 'generated':
        path = samples_dir / 'npy' / f'sample_{idx:05d}.npy'
        x = np.load(path, allow_pickle=False)
    else:
        name = record['volume']
        allowed = summary['train_timepoints'] if source == 'train' else summary['validation_timepoints']
        if name not in allowed:
            raise ValueError(f'Refusing unexpected {source} volume {name!r} for #{idx}')
        vol_path = processed_root / name / 'phi_volume.npy'
        vol = np.load(vol_path, mmap_mode='r', allow_pickle=False)
        fixed, a, b = (int(record[key]) for key in ('fixed_index', 'start_a', 'start_b'))
        if orientation == 'xy':
            phi = vol[fixed, a:a + patch_size, b:b + patch_size]
        else:
            phi = vol[a:a + patch_size, fixed, b:b + patch_size]
        # Exactly the same normalization as alcu_dataset.py.
        x = (np.clip(np.asarray(phi, dtype=np.float32), -1.0, 1.0) + 1.0) * 0.5
    if x.shape != (patch_size, patch_size):
        raise ValueError(f'{source} #{idx}: expected {patch_size}x{patch_size}, got {x.shape}')
    if not np.isfinite(x).all():
        raise ValueError(f'{source} #{idx}: non-finite pixels')
    return x


def interface_grid(x: np.ndarray) -> tuple[np.ndarray, int, float]:
    positive = (x > 0.5)  # phi>0, since normalized x=(phi+1)/2
    edge_mask = np.zeros(x.shape, dtype=bool)
    vertical = (positive[1:, :] != positive[:-1, :])
    horizontal = (positive[:, 1:] != positive[:, :-1])
    edge_count = int(np.count_nonzero(vertical) + np.count_nonzero(horizontal))
    # Highlight pixels on BOTH sides of each sign-changing edge.
    edge_mask[1:, :] |= vertical
    edge_mask[:-1, :] |= vertical
    edge_mask[:, 1:] |= horizontal
    edge_mask[:, :-1] |= horizontal
    return edge_mask, edge_count, float(np.mean(positive))


def image_tile(x: np.ndarray, output_size: int, highlight: bool) -> Image.Image:
    gray = np.uint8(np.rint(np.clip(x, 0.0, 1.0) * 255))
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    if highlight:
        mask, _, _ = interface_grid(x)
        # Magenta marks the 2D phi=0 sign-changing pixel boundary.
        rgb[mask] = (255, 0, 235)
    return Image.fromarray(rgb, 'RGB').resize((output_size, output_size), Image.Resampling.NEAREST if highlight else Image.Resampling.BILINEAR)


def save_montage(path: Path, blocks: list[tuple[str, dict, np.ndarray, int, float]],
                 cols: int, tile: int, highlight: bool, title: str, subtitle: str) -> None:
    label_h, top_h, bottom_margin = 49, 72, 16
    num_rows = math.ceil(len(blocks) / cols)
    width, height = cols * tile, top_h + num_rows * (tile + label_h) + bottom_margin
    canvas = Image.new('RGB', (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, font=font_at(19), fill=(245,245,245))
    draw.text((12, 36), subtitle, font=font_at(12), fill=(190,190,190))
    for rank, (source, row, x, edges, positive_fraction) in enumerate(blocks):
        col, r = rank % cols, rank // cols
        xloc = col * tile
        yloc = top_h + r * (tile + label_h)
        score = row[SCORE]
        name = {'generated':'GEN', 'train':'TRAIN', 'val':'VAL'}[source]
        draw.text((xloc + 7, yloc + 5),
                  f'{name} #{row["index"]:03d}  L={score:.6f}',
                  font=font_at(13), fill=(255,255,255))
        draw.text((xloc + 7, yloc + 26),
                  f'{edges} edges   +phase={positive_fraction:.3f}',
                  font=font_at(11), fill=(190,190,190))
        canvas.paste(image_tile(x, tile, highlight), (xloc, yloc + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.png')
    canvas.save(tmp)
    tmp.replace(path)


def build_block(record, orientation, samples_dir, processed_root, summary, patch_size):
    x = get_patch(record, orientation, samples_dir, processed_root, summary, patch_size)
    _, count, positive = interface_grid(x)
    score = count / x.size
    if not math.isclose(score, record[SCORE], rel_tol=0, abs_tol=1e-7):
        raise ValueError(f'{orientation} {record["source"]} #{record["index"]}: stored score '
                         f'{record[SCORE]:.9f} != recomputed {score:.9f}. '
                         'Check that evaluation and sample directories match.')
    return record['source'], record, x, count, positive


def run_orientation(args, orientation: str):
    samples_dir = args.samples_root / f'alcu_{orientation}_unconditional' / args.sample_subdir
    eval_dir = args.evaluation_root / orientation
    out_dir = args.output_root / orientation
    meta = json.loads((samples_dir / 'run_config.json').read_text(encoding='utf-8'))
    summary = json.loads((eval_dir / 'summary.json').read_text(encoding='utf-8'))
    if meta.get('orientation') != orientation or summary.get('orientation') != orientation:
        raise ValueError(f'{orientation}: orientation mismatch between meta and evaluation')
    if meta.get('checkpoint_step') != summary.get('checkpoint_step'):
        raise ValueError(f'{orientation}: checkpoint mismatch')
    patch_size = int(meta['patch_size'])
    rows = parse_rows(eval_dir / 'per_patch_metrics.csv')
    if len(rows['generated']) != int(meta['num_samples']):
        raise ValueError(f'{orientation}: generated CSV count differs from run_config.json')
    for source in ('train','val'):
        n = summary['comparisons'][source]['n']
        if len(rows[source]) != n:
            raise ValueError(f'{orientation}: {source} CSV count differs from summary.json')

    out_dir.mkdir(parents=True, exist_ok=True)
    # Export ALL ranked metrics, not just the 12 selected examples.
    fields = ['source','rank','index',SCORE,'edge_count','positive_phase_fraction',
              'volume','fixed_index','start_a','start_b']
    ranking_path = out_dir / 'interface_length_rankings.csv'
    with ranking_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for source in GROUPS:
            for rank, r in enumerate(rows[source], start=1):
                writer.writerow({
                    'source':source, 'rank':rank, 'index':r['index'],
                    SCORE:r[SCORE],
                    'edge_count':round(r[SCORE]*patch_size*patch_size),
                    'positive_phase_fraction':r.get('positive_phase_fraction',''),
                    'volume':r.get('volume',''),
                    'fixed_index':r.get('fixed_index',''),
                    'start_a':r.get('start_a',''), 'start_b':r.get('start_b',''),
                })

    top = min(args.top_k, len(rows['generated']))
    generated_blocks = [build_block(r,orientation,samples_dir,args.processed_root,summary,patch_size)
                        for r in rows['generated'][:top]]
    subtitle = 'Ranked by 4-neighbor phi=0 sign changes / patch area; selected extremes, NOT representative'
    save_montage(out_dir/'generated_top_interfaces_raw.png', generated_blocks,
                 args.cols,args.tile,False,
                 f'{orientation.upper()}: top {top} GENERATED patches by interface length',subtitle)
    save_montage(out_dir/'generated_top_interfaces_overlay.png', generated_blocks,
                 args.cols,args.tile,True,
                 f'{orientation.upper()}: top {top} GENERATED interfaces (MAGENTA)',subtitle)

    # 3-row comparison, top four per source; these are independent NOT paired.
    compare_blocks = []
    for source in GROUPS:
        best = [build_block(r,orientation,samples_dir,args.processed_root,summary,patch_size)
                for r in rows[source][:args.cols]]
        compare_blocks.extend(best)
    save_montage(out_dir/'top_interfaces_generated_train_val.png', compare_blocks,
                 args.cols,args.tile,True,
                 f'{orientation.upper()}: top {args.cols} patches by interface length in EACH distribution',
                 'Rows: GENERATED / TRAIN / VALIDATION (NOT spatially paired); magenta = phi=0 boundary')

    print(f'\n{orientation.upper()} — ranked {len(rows["generated"])} generated, '
          f'{len(rows["train"])} train, {len(rows["val"])} validation patches')
    for rank, (source, r, _, count, pos) in enumerate(generated_blocks, 1):
        print(f'  GEN rank {rank:2d}: sample_{r["index"]:05d}.npy '
              f'L={r[SCORE]:.7f}, edges={count}, positive={pos:.4f}')
    print(f'  Wrote plots and {ranking_path}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples-root', required=True, type=Path,
                   help='Root containing alcu_xy_unconditional/ and alcu_xz_unconditional/')
    p.add_argument('--evaluation-root', required=True, type=Path,
                   help='Directory with xy/ and xz/ evaluation outputs')
    p.add_argument('--processed-root', required=True, type=Path,
                   help='Root of training/validation phase-field volumes; test is never used')
    p.add_argument('--sample-subdir', default='step_300000_pc_langevin_128')
    p.add_argument('--output-root', required=True, type=Path)
    p.add_argument('--orientation',choices=('xy','xz','both'),default='both')
    p.add_argument('--top-k',type=int,default=12)
    p.add_argument('--cols',type=int,default=4)
    p.add_argument('--tile',type=int,default=320)
    args=p.parse_args()
    if args.top_k<1 or args.cols<1 or args.tile<140:
        p.error('--top-k/--cols must be positive and --tile must be >=140')
    for orientation in ('xy','xz') if args.orientation=='both' else (args.orientation,):
        run_orientation(args,orientation)


if __name__=='__main__':
    main()