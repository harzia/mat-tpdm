#!/usr/bin/env python3
"""Compare unconditional score-SDE 2D samples with deterministic validation patches.

Reads only the 'val' split (never touches the test volumes). Uses robust morphology
proxies, not paired reconstruction metrics: generated samples are unconditional.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from alcu_dataset import PhaseFieldPatchDataset, load_split


def json_dump_atomic(path: Path, data):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sample_metrics(x: np.ndarray, eps: float) -> dict:
    # x is raw network-space: do not clip for scientific measurements.
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite values in a sample')
    phi = 2.0 * x - 1.0
    interface = np.abs(phi) < eps
    return {
        'normalized_mean': float(x.mean(dtype=np.float64)),
        'normalized_std': float(x.std(dtype=np.float64)),
        'normalized_min': float(x.min()),
        'normalized_max': float(x.max()),
        'out_of_normalized_range_fraction': float(np.mean((x < 0.0) | (x > 1.0))),
        'phi_mean': float(phi.mean(dtype=np.float64)),
        'phi_std': float(phi.std(dtype=np.float64)),
        'crosses_interface': bool(phi.min() < 0.0 and phi.max() > 0.0),
        'interface_band_fraction': float(interface.mean()),
        'positive_phase_fraction': float(np.mean(phi > 0.0)),
        'negative_phase_fraction': float(np.mean(phi < 0.0)),
    }


def aggregate(rows: list[dict]):
    fields = ['normalized_mean','normalized_std', 'out_of_normalized_range_fraction',
              'phi_mean','phi_std','interface_band_fraction',
              'positive_phase_fraction','negative_phase_fraction']
    return {
        'n': len(rows),
        'crossing_probability': float(np.mean([float(r['crosses_interface']) for r in rows])),
        **{key + '_mean': float(np.mean([r[key] for r in rows])) for key in fields},
        **{key + '_median': float(np.median([r[key] for r in rows])) for key in fields},
    }


def preview_array(x: np.ndarray, size=256) -> Image.Image:
    # Clipping is exclusively for the visual preview.
    image = Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8), 'L')
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return image.convert('RGB')


def make_montage(src_gen, src_val, n: int, path: Path, tile: int=256, cols: int=4):
    cols = min(cols, n)
    rows = int(math.ceil(n / cols))
    # Two panels: each line of generated images is followed by corresponding
    # validation references (not spatially paired).
    height = 2 * rows * (tile + 24)
    im = Image.new('RGB', (cols * tile, height), (28,28,28))
    draw = ImageDraw.Draw(im)
    for i in range(n):
        col, row = i % cols, i // cols
        x = col * tile
        g_y = (2*row)*(tile+24)
        v_y = (2*row+1)*(tile+24)
        im.paste(preview_array(src_gen(i), tile), (x,g_y+24))
        im.paste(preview_array(src_val(i), tile), (x,v_y+24))
        draw.text((x+5,g_y+5), f'GENERATED {i:03d}', fill=(255,255,255))
        draw.text((x+5,v_y+5), f'VALIDATION {i:03d}', fill=(255,255,255))
    tmp = path.with_name(path.stem + '.tmp.png')
    im.save(tmp)
    os.replace(tmp, path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-root', required=True, type=Path)
    p.add_argument('--split', required=True, type=Path)
    p.add_argument('--orientation', required=True, choices=('xy','xz'))
    p.add_argument('--samples-dir', required=True, type=Path)
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Defaults to SAMPLES_DIR/evaluation')
    p.add_argument('--validation-seed', type=int, default=43,
                   help='train_alcu_score.py val uses training seed 42 + 1')
    p.add_argument('--interface-eps', type=float, default=0.1)
    p.add_argument('--max-montage', type=int, default=16)
    args = p.parse_args()

    sample_dir=args.samples_dir.resolve(strict=True)
    meta=json.loads((sample_dir/'run_config.json').read_text(encoding='utf-8'))
    if args.orientation != meta['orientation']:
        raise SystemExit('Orientation mismatch between --orientation and run_config.json')
    psize=int(meta['patch_size'])
    count=int(meta['num_samples'])
    generated_paths=[sample_dir/'npy'/f'sample_{i:05d}.npy' for i in range(count)]
    missing=[str(p) for p in generated_paths if not p.exists()]
    if missing:
        raise SystemExit(f'Missing {len(missing)} generated .npy files; first={missing[0]}')
    split=load_split(args.split)
    valdirs=[args.processed_root/name for name in split['val']]
    ds=PhaseFieldPatchDataset(valdirs,args.orientation,
                              patch_size=psize,samples=count,
                              deterministic=True, seed=args.validation_seed,
                              interface_eps=args.interface_eps,return_metadata=True)
    out=(args.output_dir or sample_dir/'evaluation').resolve()
    out.mkdir(parents=True,exist_ok=True)
    gen_rows=[]
    val_rows=[]
    hist_edges=np.linspace(-1.0,1.0,65)
    hist_gen=np.zeros(64,dtype=np.int64)
    hist_val=np.zeros(64,dtype=np.int64)
    gen_outside=0
    val_outside=0
    print(f'Evaluating {count} {args.orientation.upper()} samples vs {count} fixed validation patches',flush=True)
    for i,path in enumerate(generated_paths):
        generated=np.load(path,allow_pickle=False)
        if generated.shape!=(psize,psize):
            raise RuntimeError(f'{path}: expected {(psize,psize)}, got {generated.shape}')
        item=ds[i]
        real=item['image'][0].numpy()
        gen_rows.append({'sample':i,**sample_metrics(generated,args.interface_eps)})
        val_rows.append({'sample':i,'volume':item['volume_name'],
                         **sample_metrics(real,args.interface_eps)})
        phi_g=2*generated-1
        phi_v=2*real-1
        hist_gen+=np.histogram(phi_g,bins=hist_edges)[0]
        hist_val+=np.histogram(phi_v,bins=hist_edges)[0]
        gen_outside+=int(np.count_nonzero((phi_g<-1)|(phi_g>1)))
        val_outside+=int(np.count_nonzero((phi_v<-1)|(phi_v>1)))
    with (out/'per_sample_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['source','sample','volume',*list(gen_rows[0].keys())[1:]]
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for r in gen_rows: writer.writerow({'source':'generated','volume':'',**r})
        for r in val_rows: writer.writerow({'source':'validation',**r})
    with (out/'phi_histogram.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['phi_low','phi_high','generated_count','validation_count'])
        for i in range(64):
            writer.writerow([float(hist_edges[i]),float(hist_edges[i+1]),
                             int(hist_gen[i]),int(hist_val[i])])
    summary={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'orientation':args.orientation,
        'checkpoint_step':meta['checkpoint_step'],
        'generated':aggregate(gen_rows),
        'validation':aggregate(val_rows),
        'validation_volumes':split['val'],
        'validation_seed':args.validation_seed,
        'interface_eps_phi':args.interface_eps,
        'phi_histogram_domain':[-1.0,1.0],
        'phi_histogram_generated_outside_count':gen_outside,
        'phi_histogram_validation_outside_count':val_outside,
        'notes':['Unconditional generated images and validation images are NOT spatially paired.',
                 'Generated arrays are measured before clipping; preview PNGs are clipped to [0,1].',
                 'This is a distribution sanity check, not an X-ray reconstruction evaluation.',
                 'Test split is never loaded.']
    }
    json_dump_atomic(out/'summary.json',summary)
    n=min(count,args.max_montage)
    if n>0:
        make_montage(lambda i: np.load(generated_paths[i],allow_pickle=False),
                     lambda i: ds[i]['image'][0].numpy(), n,
                     out/'generated_vs_validation.png')
    print(json.dumps({'generated':summary['generated'], 'validation':summary['validation']},indent=2))
    print('Saved',out,flush=True)


if __name__=='__main__':
    main()