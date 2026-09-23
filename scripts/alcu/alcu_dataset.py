#!/usr/bin/env python3
"""Natural 384x384 patch loaders for processed Al-Cu phase-field volumes.

Sampling distribution:
  1. choose a timepoint uniformly;
  2. choose a valid patch uniformly inside that timepoint.

XY: choose z, y_start, x_start uniformly.
XZ: choose y, z_start, x_start uniformly.

phi is clipped to [-1, 1] and mapped to [0, 1] on the fly. The stored
phi_volume.npy is never modified.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


def _timestep(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 10**30


def discover_volume_dirs(processed_root: Path) -> list[Path]:
    processed_root = Path(processed_root)
    dirs = [
        p.parent for p in processed_root.glob("*/phi_volume.npy")
        if (p.parent / "metadata.json").exists()
    ]
    dirs.sort(key=lambda p: (_timestep(p.name), p.name))
    if not dirs:
        raise RuntimeError(
            f"No processed volumes found under {processed_root}. "
            "Expected */phi_volume.npy and */metadata.json."
        )
    return dirs


def load_split(split_path: Path) -> dict:
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    if "train" not in split or "val" not in split:
        raise ValueError("Split JSON must contain 'train' and 'val' lists.")
    train = list(split["train"])
    val = list(split["val"])
    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(f"Train/validation leakage: {sorted(overlap)}")
    if not train or not val:
        raise ValueError("Both train and val splits must be non-empty.")
    return {"train": train, "val": val}


def resolve_split_dirs(processed_root: Path, split_path: Path):
    all_dirs = discover_volume_dirs(processed_root)
    by_name = {p.name: p for p in all_dirs}
    split = load_split(split_path)
    missing = [n for n in split["train"] + split["val"] if n not in by_name]
    if missing:
        raise ValueError(
            "Split references missing processed directories: "
            + ", ".join(sorted(set(missing)))
        )
    return ([by_name[n] for n in split["train"]],
            [by_name[n] for n in split["val"]])


@dataclass(frozen=True)
class PatchDescriptor:
    volume_index: int
    fixed_index: int
    start_a: int
    start_b: int


class PhaseFieldPatchDataset(Dataset):
    """Natural stochastic training patches or fixed deterministic val patches."""

    def __init__(
        self,
        volume_dirs: Sequence[Path],
        orientation: str,
        patch_size: int = 384,
        samples: int = 10000,
        deterministic: bool = False,
        seed: int = 42,
        interface_eps: float = 0.1,
        return_metadata: bool = True,
    ) -> None:
        if orientation not in ("xy", "xz"):
            raise ValueError("orientation must be 'xy' or 'xz'")
        if patch_size <= 0 or samples <= 0 or not volume_dirs:
            raise ValueError("invalid patch_size/samples/volume_dirs")

        self.volume_dirs = [Path(p) for p in volume_dirs]
        self.orientation = orientation
        self.patch_size = int(patch_size)
        self.samples = int(samples)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.interface_eps = float(interface_eps)
        self.return_metadata = bool(return_metadata)
        self.volume_paths = [p / "phi_volume.npy" for p in self.volume_dirs]
        self.shapes = []

        for p in self.volume_paths:
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 3:
                raise RuntimeError(f"{p}: expected 3D, got {arr.shape}")
            shape = tuple(int(v) for v in arr.shape)
            nz, ny, nx = shape
            if orientation == "xy" and (patch_size > ny or patch_size > nx):
                raise RuntimeError(f"{p}: patch does not fit XY plane {ny}x{nx}")
            if orientation == "xz" and (patch_size > nz or patch_size > nx):
                raise RuntimeError(f"{p}: patch does not fit XZ plane {nz}x{nx}")
            self.shapes.append(shape)
            del arr

        self._memmaps: Dict[int, np.ndarray] = {}
        self._rng: Optional[np.random.Generator] = None
        self._rng_worker_seed: Optional[int] = None

    def __len__(self):
        return self.samples

    def _worker_rng(self):
        info = get_worker_info()
        worker_seed = int(torch.initial_seed() % (2**63 - 1))
        worker_seed ^= self.seed if info is None else (self.seed + 1000003 * info.id)
        if self._rng is None or self._rng_worker_seed != worker_seed:
            self._rng = np.random.default_rng(worker_seed)
            self._rng_worker_seed = worker_seed
        return self._rng

    def _open_volume(self, volume_index: int):
        arr = self._memmaps.get(volume_index)
        if arr is None:
            arr = np.load(self.volume_paths[volume_index], mmap_mode="r")
            self._memmaps[volume_index] = arr
        return arr

    def _sample_descriptor(self, rng):
        vi = int(rng.integers(0, len(self.volume_paths)))
        nz, ny, nx = self.shapes[vi]
        p = self.patch_size
        if self.orientation == "xy":
            return PatchDescriptor(
                vi,
                int(rng.integers(0, nz)),
                int(rng.integers(0, ny - p + 1)),
                int(rng.integers(0, nx - p + 1)),
            )
        return PatchDescriptor(
            vi,
            int(rng.integers(0, ny)),
            int(rng.integers(0, nz - p + 1)),
            int(rng.integers(0, nx - p + 1)),
        )

    def _descriptor_for_index(self, index: int):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        return self._sample_descriptor(rng)

    def _read_patch(self, d: PatchDescriptor):
        v = self._open_volume(d.volume_index)
        p = self.patch_size
        if self.orientation == "xy":
            patch = v[d.fixed_index, d.start_a:d.start_a+p, d.start_b:d.start_b+p]
        else:
            patch = v[d.start_a:d.start_a+p, d.fixed_index, d.start_b:d.start_b+p]
        patch = np.asarray(patch, dtype=np.float32).copy()
        if patch.shape != (p, p):
            raise RuntimeError(f"Unexpected patch shape {patch.shape}; {d}")
        if not np.all(np.isfinite(patch)):
            raise RuntimeError(f"Non-finite phi values; {d}")
        return patch

    def __getitem__(self, index: int):
        d = (self._descriptor_for_index(index) if self.deterministic
             else self._sample_descriptor(self._worker_rng()))
        phi = self._read_patch(d)

        phi_min = float(phi.min())
        phi_max = float(phi.max())
        crosses = bool(phi_min < 0.0 and phi_max > 0.0)
        band_fraction = float(np.mean(np.abs(phi) < self.interface_eps))

        np.clip(phi, -1.0, 1.0, out=phi)
        phi = (phi + 1.0) * 0.5
        image = torch.from_numpy(phi).unsqueeze(0)

        if not self.return_metadata:
            return image

        return {
            "image": image,
            "crosses_interface": torch.tensor(crosses, dtype=torch.bool),
            "interface_band_fraction": torch.tensor(band_fraction, dtype=torch.float32),
            "volume_index": torch.tensor(d.volume_index, dtype=torch.int64),
            "fixed_index": torch.tensor(d.fixed_index, dtype=torch.int64),
            "start_a": torch.tensor(d.start_a, dtype=torch.int64),
            "start_b": torch.tensor(d.start_b, dtype=torch.int64),
            "volume_name": self.volume_dirs[d.volume_index].name,
        }


def build_train_val_datasets(
    processed_root: Path,
    split_path: Path,
    orientation: str,
    patch_size: int = 384,
    train_samples: int = 100000,
    val_samples: int = 1024,
    seed: int = 42,
    interface_eps: float = 0.1,
):
    train_dirs, val_dirs = resolve_split_dirs(processed_root, split_path)
    train_ds = PhaseFieldPatchDataset(
        train_dirs, orientation, patch_size, train_samples,
        deterministic=False, seed=seed, interface_eps=interface_eps,
        return_metadata=True,
    )
    val_ds = PhaseFieldPatchDataset(
        val_dirs, orientation, patch_size, val_samples,
        deterministic=True, seed=seed + 1, interface_eps=interface_eps,
        return_metadata=True,
    )
    return train_ds, val_ds
