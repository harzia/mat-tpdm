#!/usr/bin/env python3
"""Create a whole-timepoint train/validation/test split.

The current training loader consumes ``train`` and ``val``. ``test`` is stored
in the same JSON so those volumes can be kept untouched for later TPDM
reconstruction evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def timestep(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 10**30


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-root", type=Path, required=True)
    p.add_argument("--val", nargs="+", required=True)
    p.add_argument("--test", nargs="+", default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    dirs = sorted(
        [
            p.parent.name
            for p in args.processed_root.glob("*/phi_volume.npy")
            if (p.parent / "metadata.json").exists()
        ],
        key=lambda x: (timestep(x), x),
    )
    if not dirs:
        raise SystemExit(
            f"No processed volumes found under {args.processed_root}"
        )

    val = list(dict.fromkeys(args.val))
    test = list(dict.fromkeys(args.test))

    overlap = set(val) & set(test)
    if overlap:
        raise SystemExit(
            "Validation/test overlap: " + ", ".join(sorted(overlap))
        )

    missing = [name for name in val + test if name not in dirs]
    if missing:
        raise SystemExit(
            "Requested split names not found: "
            + ", ".join(sorted(set(missing)))
        )

    held_out = set(val) | set(test)
    train = [name for name in dirs if name not in held_out]
    if not train:
        raise SystemExit("Held-out sets consumed all volumes.")

    payload = {"train": train, "val": val, "test": test}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, args.output)

    print(f"Total: {len(dirs)}")
    print(f"Train: {len(train)}")
    print(f"Val:   {len(val)}")
    print(f"Test:  {len(test)}")
    print(f"Wrote: {args.output}")
    print("Validation:", " ".join(val))
    print("Test:", " ".join(test))


if __name__ == "__main__":
    main()
