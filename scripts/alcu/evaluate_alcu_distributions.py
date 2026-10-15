#!/usr/bin/env python3
"""Distributional sanity check for Al-Cu XY/XZ unconditional score-SDE samples.

Exactly reproduces the training *sampling distribution* for real references:
choose a train/validation timepoint uniformly, then a uniformly random valid
XY/XZ patch within that volume. All reference draws are deterministic.

The test split is never loaded. Generated samples are unconditional and are
NOT spatially matched to the reference patches. Keep originals in raw x-space.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

# This module already exists alongside this script in harzia/mat-tpdm.
from alcu_dataset import PhaseFieldPatchDataset


METRIC_KEYS = (
    'crosses_interface',
    'positive_phase_fraction',
    'interface_band_fraction',
    'interface_edge_length_per_pixel_area',
    'positive_components_min32',
    'mean_normalized',
    'std_normalized',
    'below_0_fraction',
    'above_1_fraction',
)


def read_split(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        value = json.load(f)
    for key in ('train', 'val'):
        if not isinstance(value.get(key), list) or not value[key]:
            raise ValueError(f'Missing or empty split {key!r}')
    all_sets = {key: set(value.get(key, [])) for key in ('train','val','test')}
    if any(all_sets[a] & all_sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Train, val and test overlap. Refusing to continue.')
    return value


def inspect_patch(x: np.ndarray, eps: float) -> dict:
    """Return sample metrics in network space, without modifying x.

    Interface length is a *2D pixel-grid proxy*: the count of adjacent pixels
    on different sides of phi=0 divided by patch area. A value of 0 means
    no resolvable interface. This is not physical 3D interface area.
    """
    x = np.asarray(x)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError(f'Expected finite 2D image, got {x.shape}')
    phi = 2.0 * x.astype(np.float32, copy=False) - 1.0
    positive = phi > 0
    neg = phi < 0
    crosses = bool(positive.any() and neg.any())
    edges = (np.count_nonzero(positive[1:, :] != positive[:-1, :]) +
             np.count_nonzero(positive[:, 1:] != positive[:, :-1]))

    # Count only positive components with >=32 pixels, to suppress tiny specks.
    # 8-connected in 2D. Includes components touching patch edges.
    labels, n_labels = ndimage.label(positive, structure=np.ones((3,3), dtype=np.uint8))
    if n_labels:
        sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
        components = int(np.count_nonzero(sizes[1:] >= 32))
    else:
        components = 0

    below = x[x < 0.0]
    above = x[x > 1.0]
    # Out-of-range values matter quantitatively, especially when the previews
    # clip to [0,1]. Magnitudes are in normalized network space.
    return {
        'crosses_interface': int(crosses),
        'positive_phase_fraction': float(np.mean(positive)),
        'negative_phase_fraction': float(np.mean(neg)),
        'interface_band_fraction': float(np.mean(np.abs(phi) < eps)),
        'interface_edge_length_per_pixel_area': float(edges / x.size),
        'positive_components_min32': components,
        'mean_normalized': float(x.mean(dtype=np.float64)),
        'std_normalized': float(x.std(dtype=np.float64)),
        'min_normalized': float(x.min()),
        'max_normalized': float(x.max()),
        'below_0_fraction': float(below.size / x.size),
        'above_1_fraction': float(above.size / x.size),
        'mean_below_0_magnitude': float(-below.mean()) if below.size else 0.0,
        'max_below_0_magnitude': float(-below.min()) if below.size else 0.0,
        'mean_above_1_magnitude': float((above - 1).mean()) if above.size else 0.0,
        'max_above_1_magnitude': float((above - 1).max()) if above.size else 0.0,
    }


def save_json(path: Path, record: dict) -> None:
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)


def aggregate(rows: list[dict]) -> dict:
    out = {'n': len(rows)}
    fields = [k for k in rows[0] if k not in ('source','index','volume','fixed_index','start_a','start_b')]
    for key in fields:
        v = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[key] = {
            'mean': float(np.mean(v)),
            'median': float(np.median(v)),
            'p05': float(np.quantile(v,.05)),
            'p95': float(np.quantile(v,.95)),
            'min': float(v.min()),
            'max': float(v.max()),
        }
    return out


def make_reference_ds(root: Path, split_names: list[str], orientation: str,
                      patch_size: int, n: int, seed: int) -> PhaseFieldPatchDataset:
    # Only selected train / validation names are passed; test paths are never opened.
    paths = [root / name for name in split_names]
    for p in paths:
        if not (p / 'phi_volume.npy').is_file():
            raise FileNotFoundError(p / 'phi_volume.npy')
    return PhaseFieldPatchDataset(
        paths,
        orientation=orientation,
        patch_size=patch_size,
        samples=n,
        deterministic=True,
        seed=seed,
        return_metadata=True,
    )


def png_tile(array: np.ndarray, tile: int) -> Image.Image:
    img = Image.fromarray(np.uint8(np.rint(np.clip(array,0,1)*255)), 'L')
    return img.resize((tile, tile), Image.Resampling.BILINEAR).convert('RGB')


def write_montage(path: Path, examples: dict[str, list[tuple[int,np.ndarray]]],
                  cols: int=4, tile: int=216, label_h: int=30):
    """Each source is its own row; same column is NOT a matched sample."""
    groups = [('generated','GENERATED (unconditional)'),
              ('train','TRAIN (natural)'), ('val','VALIDATION (natural)')]
    canvas = Image.new('RGB', (cols*tile, len(groups)*(tile+label_h+10)), (27,27,27))
    draw = ImageDraw.Draw(canvas)
    for gi,(group,label) in enumerate(groups):
        y = gi*(tile+label_h+10)
        for col, (index, array) in enumerate(examples.get(group, [])[:cols]):
            x = col*tile
            canvas.paste(png_tile(array,tile), (x,y+label_h))
            draw.text((x+6,y+6),f'{label}  #{index:03d}',fill=(255,255,255))
        if not examples.get(group):
            draw.text((4,y+label_h+10),f'No qualifying {group} samples', fill=(255,200,200))
    tmp=path.with_name(path.stem+'.tmp.png')
    canvas.save(tmp)
    os.replace(tmp,path)


def write_ecdf_figure(path: Path, data: dict[str,list[dict]]):
    stats = [
        ('positive_phase_fraction','Positive-phase area fraction'),
        ('interface_band_fraction','Diffuse-interface pixel fraction |phi| < 0.1'),
        ('interface_edge_length_per_pixel_area','2D sign-change edge length / pixel area'),
        ('mean_normalized','Mean normalized intensity (raw)'),
    ]
    colors={'generated':'#D97706','train':'#2563EB','val':'#059669'}
    fig, axs = plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for ax,(key,label) in zip(axs.ravel(),stats):
        for source in ('generated','train','val'):
            v=np.sort([r[key] for r in data[source]])
            p=np.arange(1,len(v)+1,dtype=float)/len(v)
            ax.step(v,p,where='post',label=f'{source} (n={len(v)})', color=colors[source], lw=1.8)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel('Empirical cumulative probability')
        ax.set_ylim(0,1.01)
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.suptitle('Unconditional generations versus natural train/validation patches\nNot spatially paired; timepoint distributions differ',fontsize=13)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def write_phi_histogram(path: Path, data: dict[str,np.ndarray], bin_edges: np.ndarray):
    fig,ax=plt.subplots(figsize=(9,5))
    colors={'generated':'#D97706','train':'#2563EB','val':'#059669'}
    centers=.5*(bin_edges[1:]+bin_edges[:-1])
    for k in ('generated','train','val'):
        hist=data[k].astype(np.float64)
        hist/=max(1,hist.sum())
        ax.step(centers,hist,label=k,color=colors[k],linewidth=1.8,where='mid')
    ax.set_xlabel('phi ([-1,1] histogram only)')
    ax.set_ylabel('Fraction of in-range pixels per bin')
    ax.set_title('Phase-field value distribution; out-of-range generated pixels excluded')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-root',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--samples-dir',type=Path,required=True)
    p.add_argument('--orientation',choices=('xy','xz'),required=True)
    p.add_argument('--output-dir',type=Path,help='Default: SAMPLES_DIR/distribution_evaluation')
    p.add_argument('--train-count',type=int,default=512)
    p.add_argument('--val-count',type=int,default=512)
    p.add_argument('--train-seed',type=int,default=1042)
    p.add_argument('--val-seed',type=int,default=43,
                   help='43 reproduces the existing validation sampler sequence')
    p.add_argument('--interface-eps',type=float,default=.1)
    p.add_argument('--montage-cols',type=int,default=4)
    args=p.parse_args()
    if args.train_count<1 or args.val_count<1 or args.interface_eps<=0 or args.montage_cols<1:
        p.error('Counts, interface eps and montage cols must be positive')
    split=read_split(args.split)
    meta=json.loads((args.samples_dir/'run_config.json').read_text(encoding='utf-8'))
    if meta.get('orientation')!=args.orientation:
        raise ValueError(f'Samples are orientation {meta.get("orientation")!r}, expected {args.orientation}')
    patch_size=int(meta['patch_size'])
    n=int(meta['num_samples'])
    if n<=0:
        raise ValueError('Invalid generated sample count')
    sample_files=[args.samples_dir/'npy'/f'sample_{i:05d}.npy' for i in range(n)]
    not_found=[str(path) for path in sample_files if not path.is_file()]
    if not_found:
        raise FileNotFoundError(f'{len(not_found)} generated samples missing, first: {not_found[0]}')
    out=(args.output_dir or (args.samples_dir/'distribution_evaluation')).resolve()
    out.mkdir(parents=True,exist_ok=True)
    print(f'Orientation={args.orientation}, generated n={n}, train n={args.train_count}, val n={args.val_count}',flush=True)
    print(f'Excluding test split ({len(split.get("test",[]))} timepoints) entirely.',flush=True)
    dss={
        'train': make_reference_ds(args.processed_root,split['train'],args.orientation,patch_size,args.train_count,args.train_seed),
        'val': make_reference_ds(args.processed_root,split['val'],args.orientation,patch_size,args.val_count,args.val_seed),
    }
    data={key:[] for key in ('generated','train','val')}
    preview={key:[] for key in data}
    interface_preview={key:[] for key in data}
    edges=np.linspace(-1.0,1.0,101)
    phi_hist={key:np.zeros(len(edges)-1,dtype=np.int64) for key in data}
    out_of_hist={key:0 for key in data}
    for source in ('generated','train','val'):
        total=n if source=='generated' else len(dss[source])
        for i in range(total):
            if source=='generated':
                x=np.load(sample_files[i],allow_pickle=False)
                volume=''
                fixed=a=b=''
            else:
                item=dss[source][i]
                x=item['image'][0].numpy()
                volume=str(item['volume_name'])
                fixed=int(item['fixed_index'])
                a=int(item['start_a'])
                b=int(item['start_b'])
            if x.shape!=(patch_size,patch_size):
                raise ValueError(f'{source} #{i}: unexpected {x.shape} patch shape')
            row={'source':source,'index':i,'volume':volume,
                 'fixed_index':fixed,'start_a':a,'start_b':b,
                 **inspect_patch(x,args.interface_eps)}
            data[source].append(row)
            phi=2.0*x-1.0
            phi_hist[source]+=np.histogram(phi,bins=edges)[0]
            out_of_hist[source]+=int(np.count_nonzero((phi<-1)|(phi>1)))
            if len(preview[source])<args.montage_cols:
                preview[source].append((i,np.array(x,copy=True)))
            if row['crosses_interface'] and len(interface_preview[source])<args.montage_cols:
                interface_preview[source].append((i,np.array(x,copy=True)))
            if (i+1)%128==0 or i+1==total:
                print(f'  {source} {i+1}/{total}',flush=True)

    csv_path=out/'per_patch_metrics.csv'
    fields=list(data['generated'][0].keys())
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for source in ('generated','train','val'):
            writer.writerows(data[source])
    write_montage(out/'unfiltered_montage.png',preview,cols=args.montage_cols)
    write_montage(out/'interface_crossing_only_montage.png',interface_preview,cols=args.montage_cols)
    write_ecdf_figure(out/'distribution_ecdfs.png',data)
    write_phi_histogram(out/'phi_histogram.png',phi_hist,edges)
    with (out/'phi_histogram.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['phi_bin_start','phi_bin_end','generated_count','train_count','val_count'])
        for i in range(len(edges)-1):
            writer.writerow([float(edges[i]),float(edges[i+1]),
                             int(phi_hist['generated'][i]),int(phi_hist['train'][i]),int(phi_hist['val'][i])])
    summary={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'orientation':args.orientation,
        'checkpoint_step':meta.get('checkpoint_step'),
        'patch_size':patch_size,
        'interface_eps_phi':args.interface_eps,
        'train_seed':args.train_seed,'val_seed':args.val_seed,
        'train_timepoints':split['train'], 'validation_timepoints':split['val'],
        'test_timepoints_loaded':0,
        'sampling_rule':'uniform timepoint, then uniform valid crop (independent per orientation)',
        'comparisons':{source:aggregate(data[source]) for source in ('generated','train','val')},
        'histograms_phi_outside_minus1_plus1_counts':out_of_hist,
        'warnings':[
            'Unconditional generated patches do not target or correspond to the validation timepoints.',
            'No images are paired. Pixelwise fidelity metrics (PSNR/SSIM) are not appropriate.',
            'The train-vs-validation morphology distribution may shift over solidification time.',
            'The edge length is a 2D grid sign-change proxy, not 3D surface area.',
            'The interface-crossing-only montage is explicitly selected and not representative.',
            'With 16 generations, uncertainty is substantial; use >=128 for a more stable comparison.',
            'Generated raw values are not clipped for statistics; displayed previews are clipped.',
        ],
    }
    save_json(out/'summary.json',summary)
    print('\nCROSSING / POSITIVE PHASE / INTERFACE BAND / OUT-OF-RANGE:')
    for source in ('generated','train','val'):
        s=summary['comparisons'][source]
        print(f'{source:>9} ({s["n"]:>4}): crossing={s["crosses_interface"]["mean"]:.4f} '
              f'positive={s["positive_phase_fraction"]["mean"]:.4f} '
              f'band={s["interface_band_fraction"]["mean"]:.6f} '
              f'outside={s["below_0_fraction"]["mean"] + s["above_1_fraction"]["mean"]:.4f}',flush=True)
    print('Wrote',out,flush=True)

if __name__ == '__main__':
    main()
