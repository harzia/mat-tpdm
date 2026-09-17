#!/usr/bin/env python3
"""
Exactly measure the natural 384x384 XY/XZ patch distribution in processed
phase-field volumes.

Natural reference distribution (recommended for later training):
  1) choose a timepoint uniformly;
  2) choose a valid patch uniformly inside that timepoint.

A patch crosses the phi=0 interface when it contains at least one phi<0 and
at least one phi>0 value.  Interface density is the fraction of pixels with
abs(phi) < eps (default eps=0.1).

The implementation scans each memmapped phi_volume.npy once in contiguous
z-slabs.  It does not generate crop files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Stats:
    total: int
    crossing: int
    band_present: int
    band_sum_all: int
    band_sum_crossing: int
    crossing_hist: np.ndarray


def timestep_from_name(name: str) -> Optional[int]:
    m = re.search(r"solution-(\d+)", name)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[0]) if nums else None


def discover(root: Path) -> list[Path]:
    paths = list(root.glob("*/phi_volume.npy"))
    paths.sort(key=lambda p: (
        timestep_from_name(p.parent.name) is None,
        timestep_from_name(p.parent.name) or 0,
        p.parent.name,
    ))
    if not paths:
        raise RuntimeError(f"No */phi_volume.npy files found under {root}")
    return paths


def sliding_sum_z(rows: np.ndarray, p: int) -> np.ndarray:
    """rows: (nz, n_other, n_xstarts) -> p-long sliding sums along z."""
    cs = np.zeros((rows.shape[0] + 1,) + rows.shape[1:], dtype=np.int64)
    np.cumsum(rows, axis=0, dtype=np.int64, out=cs[1:])
    return cs[p:] - cs[:-p]


def make_stats(neg: np.ndarray, pos: np.ndarray, band: np.ndarray, area: int) -> Stats:
    crossing = (neg > 0) & (pos > 0)
    n_cross = int(np.count_nonzero(crossing))
    hist = np.bincount(
        band[crossing].astype(np.int64, copy=False).ravel(),
        minlength=area + 1,
    ).astype(np.int64, copy=False)
    return Stats(
        total=int(band.size),
        crossing=n_cross,
        band_present=int(np.count_nonzero(band > 0)),
        band_sum_all=int(np.sum(band, dtype=np.int64)),
        band_sum_crossing=int(np.sum(band[crossing], dtype=np.int64)),
        crossing_hist=hist,
    )


def analyze_volume(path: Path, p: int, eps: float, slab_depth: int) -> dict[str, Stats]:
    vol = np.load(path, mmap_mode="r")
    if vol.ndim != 3:
        raise RuntimeError(f"{path}: expected 3D volume, got {vol.shape}")

    nz, ny, nx = map(int, vol.shape)
    if p > min(ny, nx, nz):
        raise RuntimeError(f"{path}: patch size {p} does not fit {vol.shape}")

    ys = ny - p + 1
    xs = nx - p + 1
    area = p * p

    # This exact implementation is intentionally optimized for the current
    # near-full-plane 384-of-385 crops.
    if ys * xs > 256 or xs > 64:
        raise RuntimeError(
            f"{path}: too many crop starts for this optimized implementation "
            f"(XY starts={ys*xs}, XZ x-starts={xs})."
        )

    # XY patch counts: one entry for every (z, y_start, x_start).
    xy_neg = np.empty((nz, ys, xs), dtype=np.int32)
    xy_pos = np.empty_like(xy_neg)
    xy_band = np.empty_like(xy_neg)

    # XZ row counts: number of mask pixels across each p-wide x interval.
    # Later summed over every p-long z interval.
    xz_neg_rows = np.empty((nz, ny, xs), dtype=np.int32)
    xz_pos_rows = np.empty_like(xz_neg_rows)
    xz_band_rows = np.empty_like(xz_neg_rows)

    for z0 in range(0, nz, slab_depth):
        z1 = min(nz, z0 + slab_depth)
        data = np.asarray(vol[z0:z1], dtype=np.float32)

        if not np.all(np.isfinite(data)):
            bad = int(np.count_nonzero(~np.isfinite(data)))
            raise RuntimeError(f"{path}: {bad:,} non-finite values in z[{z0}:{z1})")

        neg = data < 0.0
        pos = data > 0.0
        band = np.abs(data) < eps

        for yi in range(ys):
            ysl = slice(yi, yi + p)
            for xi in range(xs):
                xsl = slice(xi, xi + p)
                xy_neg[z0:z1, yi, xi] = neg[:, ysl, xsl].sum((1, 2), dtype=np.int32)
                xy_pos[z0:z1, yi, xi] = pos[:, ysl, xsl].sum((1, 2), dtype=np.int32)
                xy_band[z0:z1, yi, xi] = band[:, ysl, xsl].sum((1, 2), dtype=np.int32)

        for xi in range(xs):
            xsl = slice(xi, xi + p)
            xz_neg_rows[z0:z1, :, xi] = neg[:, :, xsl].sum(2, dtype=np.int32)
            xz_pos_rows[z0:z1, :, xi] = pos[:, :, xsl].sum(2, dtype=np.int32)
            xz_band_rows[z0:z1, :, xi] = band[:, :, xsl].sum(2, dtype=np.int32)

        print(f"    z [{z0}:{z1}) / {nz}", flush=True)

    xy = make_stats(xy_neg, xy_pos, xy_band, area)

    xz_neg = sliding_sum_z(xz_neg_rows, p)
    xz_pos = sliding_sum_z(xz_pos_rows, p)
    xz_band = sliding_sum_z(xz_band_rows, p)
    xz = make_stats(xz_neg, xz_pos, xz_band, area)

    return {"xy": xy, "xz": xz}


def hist_quantile(hist: np.ndarray, q: float, area: int) -> Optional[float]:
    total = float(hist.sum())
    if total <= 0:
        return None
    c = np.cumsum(hist, dtype=np.float64)
    idx = int(np.searchsorted(c, q * total, side="left"))
    return idx / area


def hist_summary(hist: np.ndarray, area: int) -> dict:
    total = float(hist.sum())
    if total <= 0:
        return {"mass": 0.0, "mean": None, "quantiles": {}}
    k = np.arange(hist.size, dtype=np.float64)
    mean = float(np.dot(k, hist) / (area * total))
    qs = {}
    for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        qs[f"p{int(q*100):02d}"] = hist_quantile(hist, q, area)
    nz = np.flatnonzero(hist)
    if nz.size:
        qs["min"] = float(nz[0] / area)
        qs["max"] = float(nz[-1] / area)
    return {"mass": total, "mean": mean, "quantiles": qs}


def write_hist_csv(
    path: Path,
    pooled_hist: np.ndarray,
    equal_joint_hist: np.ndarray,
    pooled_total: int,
    pooled_crossing: int,
    equal_crossing_prob: float,
    area: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "band_pixel_count",
            "interface_fraction",
            "pooled_patch_count",
            "pooled_joint_probability",
            "pooled_conditional_probability_given_crossing",
            "equal_timepoint_joint_probability",
            "equal_timepoint_conditional_probability_given_crossing",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        indices = np.flatnonzero((pooled_hist > 0) | (equal_joint_hist > 0))
        for k in indices:
            pj = pooled_hist[k] / pooled_total if pooled_total else 0.0
            pc = pooled_hist[k] / pooled_crossing if pooled_crossing else 0.0
            ej = float(equal_joint_hist[k])
            ec = ej / equal_crossing_prob if equal_crossing_prob > 0 else 0.0
            w.writerow({
                "band_pixel_count": int(k),
                "interface_fraction": k / area,
                "pooled_patch_count": int(pooled_hist[k]),
                "pooled_joint_probability": pj,
                "pooled_conditional_probability_given_crossing": pc,
                "equal_timepoint_joint_probability": ej,
                "equal_timepoint_conditional_probability_given_crossing": ec,
            })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--patch-size", type=int, default=384)
    ap.add_argument("--interface-eps", type=float, default=0.1)
    ap.add_argument("--slab-depth", type=int, default=32)
    args = ap.parse_args()

    root = args.processed_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths = discover(root)
    nvol = len(paths)
    p = args.patch_size
    area = p * p

    oris = ("xy", "xz")
    pooled_hist = {o: np.zeros(area + 1, dtype=np.int64) for o in oris}
    pooled_total = {o: 0 for o in oris}
    pooled_cross = {o: 0 for o in oris}
    pooled_band_present = {o: 0 for o in oris}
    pooled_band_sum = {o: 0 for o in oris}

    # sum_i P_i(band_count=k AND crossing). Divide by nvol at the end.
    equal_joint_hist_sum = {o: np.zeros(area + 1, dtype=np.float64) for o in oris}
    equal_cross_prob_sum = {o: 0.0 for o in oris}
    equal_band_prob_sum = {o: 0.0 for o in oris}
    equal_mean_band_sum = {o: 0.0 for o in oris}

    rows = []

    print(f"Found {nvol} volumes")
    print(f"Patch: {p}x{p}; interface band: |phi| < {args.interface_eps}")
    print("Reference weighting: uniform timepoint -> uniform valid patch")

    for i, path in enumerate(paths, 1):
        name = path.parent.name
        timestep = timestep_from_name(name)
        print("\n" + "=" * 78)
        print(f"[{i}/{nvol}] {name}")
        print("=" * 78)

        result = analyze_volume(path, p, args.interface_eps, args.slab_depth)

        for ori in oris:
            s = result[ori]
            pooled_hist[ori] += s.crossing_hist
            pooled_total[ori] += s.total
            pooled_cross[ori] += s.crossing
            pooled_band_present[ori] += s.band_present
            pooled_band_sum[ori] += s.band_sum_all

            equal_joint_hist_sum[ori] += s.crossing_hist.astype(np.float64) / s.total
            equal_cross_prob_sum[ori] += s.crossing / s.total
            equal_band_prob_sum[ori] += s.band_present / s.total
            equal_mean_band_sum[ori] += s.band_sum_all / (s.total * area)

            row = {
                "volume": name,
                "timestep": "" if timestep is None else timestep,
                "orientation": ori,
                "total_patches": s.total,
                "crossing_patches": s.crossing,
                "crossing_probability": s.crossing / s.total,
                "bulk_probability": 1.0 - s.crossing / s.total,
                "band_present_patches": s.band_present,
                "band_present_probability": s.band_present / s.total,
                "mean_band_fraction_all_patches": s.band_sum_all / (s.total * area),
                "mean_band_fraction_crossing_patches": (
                    s.band_sum_crossing / (s.crossing * area) if s.crossing else math.nan
                ),
            }
            rows.append(row)
            print(
                f"  {ori.upper()}: P(crossing)={row['crossing_probability']:.6f}, "
                f"P(any band)={row['band_present_probability']:.6f}"
            )

    with (out / "per_volume.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    report = {
        "processed_root": str(root),
        "n_volumes": nvol,
        "patch_size": p,
        "patch_area": area,
        "interface_crossing_definition": "at least one phi<0 and at least one phi>0",
        "interface_band_definition": f"abs(phi) < {args.interface_eps}",
        "natural_sampling_definition": "choose timepoint uniformly, then choose a valid patch uniformly within that timepoint",
        "orientations": {},
    }

    for ori in oris:
        equal_joint = equal_joint_hist_sum[ori] / nvol
        equal_cross = equal_cross_prob_sum[ori] / nvol
        equal_band = equal_band_prob_sum[ori] / nvol
        equal_mean_band = equal_mean_band_sum[ori] / nvol

        report["orientations"][ori] = {
            "equal_timepoint_weighting": {
                "crossing_probability": equal_cross,
                "bulk_probability": 1.0 - equal_cross,
                "band_present_probability": equal_band,
                "mean_band_fraction_all_patches": equal_mean_band,
                "interface_density_given_crossing": hist_summary(equal_joint, area),
            },
            "pooled_patch_weighting": {
                "total_patches": pooled_total[ori],
                "crossing_patches": pooled_cross[ori],
                "crossing_probability": pooled_cross[ori] / pooled_total[ori],
                "bulk_probability": 1.0 - pooled_cross[ori] / pooled_total[ori],
                "band_present_probability": pooled_band_present[ori] / pooled_total[ori],
                "mean_band_fraction_all_patches": pooled_band_sum[ori] / (pooled_total[ori] * area),
                "interface_density_given_crossing": hist_summary(pooled_hist[ori], area),
            },
        }

        write_hist_csv(
            out / f"{ori}_interface_density_histogram.csv",
            pooled_hist[ori], equal_joint, pooled_total[ori], pooled_cross[ori],
            equal_cross, area,
        )

    with (out / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print("NATURAL PATCH DISTRIBUTION (uniform timepoint -> uniform patch)")
    print("=" * 78)
    for ori in oris:
        s = report["orientations"][ori]["equal_timepoint_weighting"]
        print(f"\n{ori.upper()}")
        print(f"  P(crosses phi=0) : {s['crossing_probability']:.6f}")
        print(f"  P(no crossing)    : {s['bulk_probability']:.6f}")
        print(f"  P(any |phi|<eps)  : {s['band_present_probability']:.6f}")
        print(f"  Mean band fraction: {s['mean_band_fraction_all_patches']:.8f}")
        q = s["interface_density_given_crossing"]["quantiles"]
        if q:
            print("  Density quantiles among crossing patches:")
            for key in ("min","p01","p05","p10","p25","p50","p75","p90","p95","p99","max"):
                if key in q:
                    print(f"    {key:>3}: {q[key]:.8f}")

    print("\nWrote:")
    print(out / "report.json")
    print(out / "per_volume.csv")
    print(out / "xy_interface_density_histogram.csv")
    print(out / "xz_interface_density_histogram.csv")


if __name__ == "__main__":
    main()