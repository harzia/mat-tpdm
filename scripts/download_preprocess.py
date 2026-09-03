#!/usr/bin/env python3
"""
Download Al-Cu phase-field VTU files from Materials Commons dataset 2710,
then invoke preprocess_phi.py on each downloaded file.

Expected dataset layout:

    /PF_Simulation_Results/
        solution-....vtu
        solution-....vtu
        early_times/
            solution-....vtu
            ...

Authentication:
    export MC_API_TOKEN="..."

Useful commands:

    # Verify discovery before downloading anything:
    python download_and_preprocess.py --list-only

    # Process one discovered VTU:
    python download_and_preprocess.py --index 0 \
        --scratch ./scratch \
        --output-root ./processed

    # Process all discovered VTUs sequentially:
    python download_and_preprocess.py \
        --scratch ./scratch \
        --output-root ./processed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

import materials_commons.api as mcapi


DEFAULT_DATASET_ID = 2710
PF_ROOT = PurePosixPath("/PF_Simulation_Results")
EARLY_ROOT = PF_ROOT / "early_times"

DEFAULT_SPACING = 1.11328125
DEFAULT_SLAB_DEPTH = 32
DEFAULT_LONGITUDINAL = "xz"


@dataclass(frozen=True)
class RemoteVTU:
    file_id: int
    name: str
    path: str
    size: Optional[int]
    category: str
    timestep: Optional[int]


def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "unknown"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def timestep_from_name(name: str) -> Optional[int]:
    """
    Extract the last integer from the filename stem.

    Example:
        solution-05000000.vtu -> 5000000
    """
    nums = re.findall(r"\d+", Path(name).stem)
    return int(nums[-1]) if nums else None


def join_remote(parent: PurePosixPath, name: str) -> PurePosixPath:
    return PurePosixPath(str(parent).rstrip("/")) / name


def is_directory(obj) -> bool:
    return str(getattr(obj, "mime_type", "")).lower() == "directory"


def list_directory_by_path(client: mcapi.Client, dataset_id: int, path: PurePosixPath):
    """
    List one directory in a published dataset.

    We try the canonical absolute path first. A relative-path retry is included
    because Materials Commons deployments/versions have historically varied in
    how path strings are accepted.
    """
    candidates = [str(path)]
    relative = str(path).lstrip("/")
    if relative and relative not in candidates:
        candidates.append(relative)

    errors = []
    for candidate in candidates:
        try:
            items = client.list_published_dataset_directory_by_path(
                dataset_id,
                candidate,
            )
            return list(items)
        except Exception as exc:
            errors.append((candidate, exc))

    details = "\n".join(
        f"  path={candidate!r}: {type(exc).__name__}: {exc}"
        for candidate, exc in errors
    )
    raise RuntimeError(
        f"Could not list published dataset directory {path}.\n{details}"
    )


def walk_directory(
    client: mcapi.Client,
    dataset_id: int,
    directory: PurePosixPath,
):
    """
    Yield (file_object, full_remote_path) for all regular files recursively
    beneath 'directory'.
    """
    items = list_directory_by_path(client, dataset_id, directory)

    for item in items:
        name = getattr(item, "name", None)
        if not name:
            continue

        full_path = join_remote(directory, str(name))

        if is_directory(item):
            yield from walk_directory(
                client=client,
                dataset_id=dataset_id,
                directory=full_path,
            )
        else:
            yield item, full_path


def root_debug_listing(client: mcapi.Client, dataset_id: int) -> list[str]:
    """
    Best-effort diagnostic listing of the published dataset root.

    This is used only when PF_Simulation_Results cannot be found/listed.
    """
    result = []
    for candidate in (PurePosixPath("/"), PurePosixPath(".")):
        try:
            items = list_directory_by_path(client, dataset_id, candidate)
            for item in items:
                kind = "DIR " if is_directory(item) else "FILE"
                result.append(
                    f"{kind}  name={getattr(item, 'name', None)!r}  "
                    f"id={getattr(item, 'id', None)!r}  "
                    f"path={getattr(item, 'path', None)!r}"
                )
            if result:
                return result
        except Exception:
            pass
    return result


def discover_vtus(client: mcapi.Client, dataset_id: int) -> list[RemoteVTU]:
    """
    Discover exactly:
      * VTUs directly inside /PF_Simulation_Results      -> category "main"
      * VTUs recursively inside /PF_Simulation_Results/early_times
                                                        -> category "early_times"

    Other subdirectories under PF_Simulation_Results are ignored deliberately.
    """
    try:
        top_items = list_directory_by_path(client, dataset_id, PF_ROOT)
    except Exception as exc:
        root_listing = root_debug_listing(client, dataset_id)
        diagnostic = ""
        if root_listing:
            diagnostic = (
                "\n\nPublished dataset root contained:\n"
                + "\n".join("  " + line for line in root_listing[:100])
            )
        raise RuntimeError(
            "Failed to list /PF_Simulation_Results. "
            "This is a directory-discovery problem, not a 'no VTUs' result."
            f"\nOriginal error: {exc}"
            f"{diagnostic}"
        ) from exc

    selected: list[RemoteVTU] = []
    early_dir_found = False

    for item in top_items:
        name = getattr(item, "name", None)
        if not name:
            continue

        if is_directory(item):
            if str(name) == "early_times":
                early_dir_found = True
            continue

        if str(name).lower().endswith(".vtu"):
            file_id = getattr(item, "id", None)
            if file_id is None:
                raise RuntimeError(
                    f"VTU {name!r} has no Materials Commons file id."
                )

            remote_path = join_remote(PF_ROOT, str(name))
            selected.append(
                RemoteVTU(
                    file_id=int(file_id),
                    name=str(name),
                    path=str(remote_path),
                    size=(
                        int(getattr(item, "size"))
                        if getattr(item, "size", None) is not None
                        else None
                    ),
                    category="main",
                    timestep=timestep_from_name(str(name)),
                )
            )

    if early_dir_found:
        for item, remote_path in walk_directory(
            client=client,
            dataset_id=dataset_id,
            directory=EARLY_ROOT,
        ):
            name = getattr(item, "name", None)
            if not name or not str(name).lower().endswith(".vtu"):
                continue

            file_id = getattr(item, "id", None)
            if file_id is None:
                raise RuntimeError(
                    f"VTU {name!r} has no Materials Commons file id."
                )

            selected.append(
                RemoteVTU(
                    file_id=int(file_id),
                    name=str(name),
                    path=str(remote_path),
                    size=(
                        int(getattr(item, "size"))
                        if getattr(item, "size", None) is not None
                        else None
                    ),
                    category="early_times",
                    timestep=timestep_from_name(str(name)),
                )
            )

    selected.sort(
        key=lambda x: (
            x.timestep is None,
            x.timestep if x.timestep is not None else 0,
            0 if x.category == "early_times" else 1,
            x.path,
            x.file_id,
        )
    )

    if not selected:
        preview = []
        for item in top_items[:100]:
            preview.append(
                f"{'DIR ' if is_directory(item) else 'FILE'}  "
                f"name={getattr(item, 'name', None)!r}  "
                f"id={getattr(item, 'id', None)!r}  "
                f"mime_type={getattr(item, 'mime_type', None)!r}  "
                f"path={getattr(item, 'path', None)!r}"
            )
        raise RuntimeError(
            "PF_Simulation_Results was listed successfully, but no matching "
            ".vtu files were discovered.\n\nDirectory contents seen by the API:\n"
            + "\n".join("  " + line for line in preview)
        )

    return selected


def filter_vtus(vtus: Iterable[RemoteVTU], source: str) -> list[RemoteVTU]:
    if source == "all":
        return list(vtus)
    return [v for v in vtus if v.category == source]


def build_output_name_map(all_vtus: Iterable[RemoteVTU]) -> dict[int, str]:
    """
    Use a clean basename when unique:

        solution-05000000

    If the same basename appears in multiple remote directories, use:

        solution-05000000__mc12345

    This keeps processed/ flat without risking overwrites.
    """
    vtus = list(all_vtus)
    stems = [Path(v.name).stem for v in vtus]
    counts = Counter(stems)

    result = {}
    for v in vtus:
        stem = Path(v.name).stem
        if counts[stem] == 1:
            result[v.file_id] = stem
        else:
            result[v.file_id] = f"{stem}__mc{v.file_id}"
    return result


def output_dir_for(
    vtu: RemoteVTU,
    output_root: Path,
    output_names: dict[int, str],
) -> Path:
    return output_root / output_names[vtu.file_id]


def expected_outputs(output_dir: Path, longitudinal: str) -> list[Path]:
    return [
        output_dir / "phi_volume.npy",
        output_dir / "metadata.json",
        output_dir / "slices" / "xy_manifest.csv",
        output_dir / "slices" / f"{longitudinal}_manifest.csv",
    ]


def is_complete(output_dir: Path, longitudinal: str) -> bool:
    return all(p.exists() for p in expected_outputs(output_dir, longitudinal))


def safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def download_one(
    client: mcapi.Client,
    dataset_id: int,
    vtu: RemoteVTU,
    scratch_root: Path,
    retries: int,
) -> Path:
    """
    Download one source VTU into temporary scratch storage.

    A .part path is used so an interrupted transfer is never confused with a
    complete source file. If Materials Commons reports a file size, it is
    verified before the .part file is renamed.
    """
    file_dir = scratch_root / str(vtu.file_id)
    file_dir.mkdir(parents=True, exist_ok=True)

    final_path = file_dir / vtu.name
    part_path = final_path.with_suffix(final_path.suffix + ".part")

    if final_path.exists():
        if vtu.size is None or final_path.stat().st_size == vtu.size:
            print(f"Scratch download already present: {final_path}", flush=True)
            return final_path
        print(
            "Existing scratch file has the wrong size; redownloading.",
            flush=True,
        )
        final_path.unlink()

    last_exc: Optional[BaseException] = None

    for attempt in range(1, retries + 1):
        safe_remove(part_path)
        try:
            print(
                f"Downloading attempt {attempt}/{retries}: "
                f"{vtu.path} ({human_bytes(vtu.size)})",
                flush=True,
            )

            client.download_published_dataset_file(
                dataset_id,
                vtu.file_id,
                str(part_path),
            )

            local_size = part_path.stat().st_size
            if vtu.size is not None and local_size != vtu.size:
                raise RuntimeError(
                    f"Downloaded size mismatch for {vtu.path}: "
                    f"remote={vtu.size:,}, local={local_size:,}"
                )

            part_path.replace(final_path)
            print(f"Downloaded to {final_path}", flush=True)
            return final_path

        except BaseException as exc:
            last_exc = exc
            safe_remove(part_path)

            if attempt == retries:
                break

            delay = min(60, 5 * (2 ** (attempt - 1)))
            print(
                f"Download failed: {exc}\nRetrying in {delay}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to download {vtu.path} after {retries} attempts"
    ) from last_exc


def run_preprocess(
    preprocess_script: Path,
    vtu_path: Path,
    output_dir: Path,
    spacing: float,
    slab_depth: int,
    longitudinal: str,
) -> None:
    cmd = [
        sys.executable,
        str(preprocess_script),
        "all",
        str(vtu_path),
        "--output",
        str(output_dir),
        "--spacing",
        str(spacing),
        "--slab-depth",
        str(slab_depth),
        "--longitudinal",
        longitudinal,
    ]

    print("\nRunning preprocessing:", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def validate_preprocess(
    output_dir: Path,
    longitudinal: str,
    allow_invalid: bool,
) -> dict:
    required = [
        output_dir / "phi_volume.npy",
        output_dir / "metadata.json",
        output_dir / "slices" / "xy_manifest.csv",
        output_dir / "slices" / f"{longitudinal}_manifest.csv",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "Preprocessing returned successfully but expected outputs are "
            "missing:\n" + "\n".join(missing)
        )

    with open(output_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    invalid = int(metadata.get("invalid_points", 0))
    if invalid and not allow_invalid:
        raise RuntimeError(
            f"Preprocessed volume contains {invalid:,} invalid Cartesian "
            "points. Refusing to mark this VTU complete. Use --allow-invalid "
            "only after investigating the reason."
        )

    return metadata



def process_one(
    client: mcapi.Client,
    dataset_id: int,
    vtu: RemoteVTU,
    index: int,
    total: int,
    scratch_root: Path,
    output_root: Path,
    output_names: dict[int, str],
    preprocess_script: Path,
    spacing: float,
    slab_depth: int,
    longitudinal: str,
    retries: int,
    keep_download: bool,
    allow_invalid: bool,
    force: bool,
) -> None:

    out_dir = output_dir_for(vtu, output_root, output_names)

    print("\n" + "=" * 80, flush=True)
    print(f"VTU {index + 1}/{total}", flush=True)
    print(f"Remote:   {vtu.path}", flush=True)
    print(f"File id:  {vtu.file_id}", flush=True)
    print(f"Size:     {human_bytes(vtu.size)}", flush=True)
    print(f"Source:   {vtu.category}", flush=True)
    print(f"Output:   {out_dir}", flush=True)
    print("=" * 80, flush=True)

    if is_complete(out_dir, longitudinal) and not force:
        print("Already complete; skipping.", flush=True)
        return

    # The existing resampler is not resumable. Remove an incomplete output
    # directory before trying this file again.
    if out_dir.exists():
        if force or not is_complete(out_dir, longitudinal):
            print(
                f"Removing incomplete/forced output: {out_dir}",
                flush=True,
            )
            shutil.rmtree(out_dir)

    vtu_path = download_one(
        client=client,
        dataset_id=dataset_id,
        vtu=vtu,
        scratch_root=scratch_root,
        retries=retries,
    )

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        run_preprocess(
            preprocess_script=preprocess_script,
            vtu_path=vtu_path,
            output_dir=out_dir,
            spacing=spacing,
            slab_depth=slab_depth,
            longitudinal=longitudinal,
        )

        metadata = validate_preprocess(
            output_dir=out_dir,
            longitudinal=longitudinal,
            allow_invalid=allow_invalid,
        )

        print(f"SUCCESS: {vtu.path}", flush=True)

    finally:
        if not keep_download and vtu_path.exists():
            per_file_dir = vtu_path.parent
            print(
                f"Removing scratch download: {per_file_dir}",
                flush=True,
            )
            shutil.rmtree(per_file_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Download phase-field VTUs from Materials Commons and preprocess "
            "phi onto the validated Cartesian grid."
        )
    )

    p.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)

    p.add_argument(
        "--source",
        choices=("all", "main", "early_times"),
        default="all",
        help=(
            "Select all VTUs, only direct PF_Simulation_Results VTUs, or only "
            "early_times VTUs. Default: all."
        ),
    )

    p.add_argument(
        "--list-only",
        action="store_true",
        help="Discover/list VTUs in exact processing order, then exit.",
    )

    p.add_argument(
        "--index",
        type=int,
        default=None,
        help=(
            "Process only this zero-based index from the selected list. If "
            "omitted and JOB_COMPLETION_INDEX is set, that environment value "
            "is used. If neither exists, process all selected VTUs sequentially."
        ),
    )

    p.add_argument(
        "--scratch",
        type=Path,
        default=Path("./scratch"),
        help="Temporary source-VTU download directory.",
    )

    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("./processed"),
        help="Flat processed-data root.",
    )

    p.add_argument(
        "--preprocess-script",
        type=Path,
        default=Path(__file__).resolve().with_name("preprocess_phi.py"),
        help="Path to the validated preprocess_phi.py.",
    )

    p.add_argument("--spacing", type=float, default=DEFAULT_SPACING)
    p.add_argument("--slab-depth", type=int, default=DEFAULT_SLAB_DEPTH)

    p.add_argument(
        "--longitudinal",
        choices=("xz", "yz"),
        default=DEFAULT_LONGITUDINAL,
    )

    p.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Whole-file download attempts. Default: 3.",
    )

    p.add_argument(
        "--keep-download",
        action="store_true",
        help="Keep downloaded source VTU after preprocessing.",
    )

    p.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Permit metadata.invalid_points > 0 and still mark success.",
    )

    p.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if all expected preprocessing outputs already exist.",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    token = os.getenv("MC_API_TOKEN")
    if not token:
        raise SystemExit(
            "MC_API_TOKEN is not set. Put your Materials Commons API token in "
            "that environment variable (or expose a Kubernetes Secret as it)."
        )

    if args.retries < 1:
        raise SystemExit("--retries must be >= 1")
    if args.slab_depth < 1:
        raise SystemExit("--slab-depth must be >= 1")

    preprocess_script = args.preprocess_script.resolve()
    if not preprocess_script.exists():
        raise SystemExit(
            f"Preprocess script not found: {preprocess_script}"
        )

    args.scratch.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    client = mcapi.Client(token)

    print(
        f"Discovering dataset {args.dataset_id} under {PF_ROOT} ...",
        flush=True,
    )

    # Always discover the complete set first so output names remain stable even
    # when --source filters the processing subset.
    all_vtus = discover_vtus(client, args.dataset_id)
    output_names = build_output_name_map(all_vtus)
    vtus = filter_vtus(all_vtus, args.source)

    print(
        f"Discovered {len(all_vtus)} total VTU(s); "
        f"{len(vtus)} selected by --source={args.source}.",
        flush=True,
    )

    if not vtus:
        raise SystemExit(
            f"No VTUs remain after --source={args.source}."
        )

    if args.list_only:
        print()
        for i, v in enumerate(vtus):
            timestep = "-" if v.timestep is None else str(v.timestep)
            print(
                f"{i:4d}  "
                f"{v.category:11s}  "
                f"{human_bytes(v.size):>11s}  "
                f"t={timestep:>10s}  "
                f"out={output_names[v.file_id]:30s}  "
                f"{v.path}"
            )
        return

    index = args.index

    if index is None:
        env_index = os.getenv("JOB_COMPLETION_INDEX")
        if env_index not in (None, ""):
            try:
                index = int(env_index)
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid JOB_COMPLETION_INDEX={env_index!r}"
                ) from exc

    if index is not None:
        if not 0 <= index < len(vtus):
            raise SystemExit(
                f"Index {index} is out of range; "
                f"selected VTU count is {len(vtus)}."
            )
        work = [(index, vtus[index])]
    else:
        work = list(enumerate(vtus))

    for i, vtu in work:
        process_one(
            client=client,
            dataset_id=args.dataset_id,
            vtu=vtu,
            index=i,
            total=len(vtus),
            scratch_root=args.scratch.resolve(),
            output_root=args.output_root.resolve(),
            output_names=output_names,
            preprocess_script=preprocess_script,
            spacing=args.spacing,
            slab_depth=args.slab_depth,
            longitudinal=args.longitudinal,
            retries=args.retries,
            keep_download=args.keep_download,
            allow_invalid=args.allow_invalid,
            force=args.force,
        )


if __name__ == "__main__":
    main()
