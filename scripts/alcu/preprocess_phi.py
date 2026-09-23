import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk


DEFAULT_SPACING = 1.11328125

def human_bytes(n):
    n = float(n)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024

    return f"{n:.2f} PB"


def read_phi_only(filename):
    """
    Read geometry/connectivity and only PointData['phi'].

    Avoids loading:
        U
        mu
        concen
        d_temp
    """

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(filename))
    reader.UpdateInformation()

    point_selection = reader.GetPointDataArraySelection()
    cell_selection = reader.GetCellDataArraySelection()

    available = [
        point_selection.GetArrayName(i)
        for i in range(point_selection.GetNumberOfArrays())
    ]

    if "phi" not in available:
        raise RuntimeError(
            f"'phi' not found. Available PointData arrays: {available}"
        )

    point_selection.DisableAllArrays()
    point_selection.EnableArray("phi")

    cell_selection.DisableAllArrays()

    print("Loading VTU geometry/connectivity + phi...")
    reader.Update()

    return pv.wrap(reader.GetOutput())


def calculate_grid(bounds, spacing):
    """
    Determine regular-grid dimensions.

    We use rounding because the chosen spacing is expected to
    divide the physical domain exactly.
    """

    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    lx = xmax - xmin
    ly = ymax - ymin
    lz = zmax - zmin

    nx_cells = int(round(lx / spacing))
    ny_cells = int(round(ly / spacing))
    nz_cells = int(round(lz / spacing))

    nx = nx_cells + 1
    ny = ny_cells + 1
    nz = nz_cells + 1

    reconstructed = (
        nx_cells * spacing,
        ny_cells * spacing,
        nz_cells * spacing,
    )

    errors = (
        abs(reconstructed[0] - lx),
        abs(reconstructed[1] - ly),
        abs(reconstructed[2] - lz),
    )

    tolerance = spacing * 1e-4

    if any(e > tolerance for e in errors):
        print(
            "\nWARNING: spacing does not divide the domain "
            "exactly."
        )
        print("Extent errors:", errors)

    return nx, ny, nz


def resample_vtu(
    vtu_path,
    output_dir,
    spacing,
    slab_depth,
    overwrite=False,
):
    """
    Resample phi from the adaptive/unstructured mesh onto a
    uniform Cartesian grid.

    Output volume convention:

        volume[z, y, x]

    Data type:

        float32
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume_path = output_dir / "phi_volume.npy"
    metadata_path = output_dir / "metadata.json"

    if volume_path.exists() and not overwrite:
        raise FileExistsError(
            f"{volume_path} already exists.\n"
            "Use --overwrite if you really want to replace it."
        )

    mesh = read_phi_only(vtu_path)

    bounds = tuple(float(v) for v in mesh.bounds)

    nx, ny, nz = calculate_grid(
        bounds,
        spacing,
    )

    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    n_points = nx * ny * nz
    volume_bytes = n_points * 4

    print("\n" + "=" * 70)
    print("TARGET CARTESIAN GRID")
    print("=" * 70)

    print(f"Spacing: {spacing}")
    print(f"x points: {nx}")
    print(f"y points: {ny}")
    print(f"z points: {nz}")

    print(
        f"\nStored NumPy shape (z,y,x): "
        f"{nz} x {ny} x {nx}"
    )

    print(f"Total points: {n_points:,}")
    print(
        f"phi float32 storage: "
        f"{human_bytes(volume_bytes)}"
    )

    print("\nBounds:")
    print(f"  x: {xmin} -> {xmax}")
    print(f"  y: {ymin} -> {ymax}")
    print(f"  z: {zmin} -> {zmax}")

    volume = np.lib.format.open_memmap(
        volume_path,
        mode="w+",
        dtype=np.float32,
        shape=(nz, ny, nx),
    )

    total_invalid = 0
    global_phi_min = np.inf
    global_phi_max = -np.inf

    n_slabs = int(
        np.ceil(nz / slab_depth)
    )

    print("\n" + "=" * 70)
    print("RESAMPLING")
    print("=" * 70)

    print(
        f"Processing {n_slabs} slabs "
        f"with up to {slab_depth} z planes each."
    )

    for slab_number, start_z in enumerate(
        range(0, nz, slab_depth),
        start=1,
    ):

        end_z = min(
            start_z + slab_depth,
            nz,
        )

        local_nz = end_z - start_z

        slab_origin_z = (
            zmin + start_z * spacing
        )

        print(
            f"\nSlab {slab_number}/{n_slabs}"
            f"  z indices [{start_z}:{end_z})"
            f"  physical z={slab_origin_z:.6g}"
        )

        grid = pv.ImageData()

        grid.origin = (
            xmin,
            ymin,
            slab_origin_z,
        )

        grid.spacing = (
            spacing,
            spacing,
            spacing,
        )

        grid.dimensions = (
            nx,
            ny,
            local_nz,
        )

        query_points = (
            nx * ny * local_nz
        )

        print(
            f"  query points: "
            f"{query_points:,}"
        )


        sampled = grid.sample(
            mesh,
            pass_cell_data=False,
            pass_point_data=True,
        )

        phi_flat = np.asarray(
            sampled.point_data["phi"],
            dtype=np.float32,
        )

        # VTK ImageData ordering:
        #
        # x varies fastest, then y, then z.
        #
        # Convert:
        #
        # flattened VTK
        #     ->
        # (x,y,z)
        #     ->
        # (z,y,x)
        #
        phi_xyz = phi_flat.reshape(
            (nx, ny, local_nz),
            order="F",
        )

        phi_zyx = np.transpose(
            phi_xyz,
            (2, 1, 0),
        )


        if (
            "vtkValidPointMask"
            in sampled.point_data
        ):
            mask_flat = np.asarray(
                sampled.point_data[
                    "vtkValidPointMask"
                ]
            )

            mask_xyz = mask_flat.reshape(
                (nx, ny, local_nz),
                order="F",
            )

            mask_zyx = np.transpose(
                mask_xyz,
                (2, 1, 0),
            )

            invalid = (
                mask_zyx == 0
            )

            invalid_count = int(
                np.sum(invalid)
            )

            total_invalid += invalid_count

            if invalid_count:
                print(
                    f"  WARNING: "
                    f"{invalid_count:,} invalid "
                    "sample points"
                )

                # Do not create artificial phi=0 boundaries.
                phi_zyx = phi_zyx.copy()
                phi_zyx[invalid] = np.nan


        finite = phi_zyx[
            np.isfinite(phi_zyx)
        ]

        if finite.size:
            slab_min = float(
                np.min(finite)
            )

            slab_max = float(
                np.max(finite)
            )

            global_phi_min = min(
                global_phi_min,
                slab_min,
            )

            global_phi_max = max(
                global_phi_max,
                slab_max,
            )

            print(
                f"  phi range: "
                f"{slab_min:.6g} "
                f"to {slab_max:.6g}"
            )

        volume[start_z:end_z, :, :] = (
            phi_zyx
        )

        volume.flush()

        del sampled
        del grid
        del phi_flat
        del phi_xyz
        del phi_zyx

        gc.collect()

    volume.flush()
    del volume

    metadata = {
        "source_vtu": str(
            Path(vtu_path).resolve()
        ),
        "field": "phi",
        "dtype": "float32",
        "axis_order": "zyx",
        "shape_zyx": [
            nz,
            ny,
            nx,
        ],
        "spacing_xyz": [
            spacing,
            spacing,
            spacing,
        ],
        "bounds_xyz": [
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        ],
        "phi_min": (
            float(global_phi_min)
            if np.isfinite(global_phi_min)
            else None
        ),
        "phi_max": (
            float(global_phi_max)
            if np.isfinite(global_phi_max)
            else None
        ),
        "invalid_points": int(
            total_invalid
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("RESAMPLING COMPLETE")
    print("=" * 70)

    print(f"Volume:   {volume_path}")
    print(f"Metadata: {metadata_path}")

    print(
        f"\nFinal phi range: "
        f"{global_phi_min:.8g} "
        f"to {global_phi_max:.8g}"
    )

    print(
        f"Invalid Cartesian points: "
        f"{total_invalid:,}"
    )

    if total_invalid:
        print(
            "\nWARNING: invalid points were written as NaN. "
            "We should investigate before training."
        )

    del mesh
    gc.collect()

    return volume_path, metadata_path


def slice_stats(array):
    """
    Return slice statistics without assuming all values are valid.
    """

    finite = array[
        np.isfinite(array)
    ]

    if finite.size == 0:
        return (
            np.nan,
            np.nan,
            False,
            True,
        )

    min_phi = float(
        np.min(finite)
    )

    max_phi = float(
        np.max(finite)
    )

    contains_interface = (
        min_phi <= 0.0 <= max_phi
    )

    contains_nan = bool(
        np.any(~np.isfinite(array))
    )

    return (
        min_phi,
        max_phi,
        contains_interface,
        contains_nan,
    )


def save_slice_manifest(
    volume_path,
    metadata_path,
    output_dir,
    longitudinal="xz",
    slice_step=1,
    only_interface=False,
    write_files=False,
    clip_phi=False,
):
    """
    Prepare XY and perpendicular longitudinal slice datasets.

    The regular volume is memory-mapped, so it does not need to
    be loaded completely into memory.

    XY:
        volume[z, :, :]
        shape = (y, x)

    XZ:
        volume[:, y, :]
        shape = (z, x)

    YZ:
        volume[:, :, x]
        shape = (z, y)
    """

    volume_path = Path(volume_path)
    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    volume = np.load(
        volume_path,
        mmap_mode="r",
    )

    nz, ny, nx = volume.shape

    expected = tuple(
        metadata["shape_zyx"]
    )

    if volume.shape != expected:
        raise RuntimeError(
            f"Volume shape {volume.shape} "
            f"does not match metadata {expected}."
        )

    spacing_x, spacing_y, spacing_z = (
        metadata["spacing_xyz"]
    )

    (
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
    ) = metadata["bounds_xyz"]

    if longitudinal not in (
        "xz",
        "yz",
    ):
        raise ValueError(
            "longitudinal must be xz or yz"
        )

    print("\n" + "=" * 70)
    print("PREPARING SLICE DATASETS")
    print("=" * 70)

    print(
        f"Volume shape (z,y,x): "
        f"{volume.shape}"
    )

    print(
        f"XY slice shape: "
        f"{ny} x {nx}"
    )

    if longitudinal == "xz":
        print(
            f"XZ slice shape: "
            f"{nz} x {nx}"
        )
    else:
        print(
            f"YZ slice shape: "
            f"{nz} x {ny}"
        )

    xy_dir = output_dir / "xy"
    long_dir = (
        output_dir / longitudinal
    )

    if write_files:
        xy_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        long_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    xy_manifest = (
        output_dir / "xy_manifest.csv"
    )

    long_manifest = (
        output_dir /
        f"{longitudinal}_manifest.csv"
    )

    xy_saved = 0

    with open(
        xy_manifest,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "orientation",
                "index",
                "coordinate",
                "rows",
                "cols",
                "min_phi",
                "max_phi",
                "contains_interface",
                "contains_nan",
                "file",
            ],
        )

        writer.writeheader()

        for z_idx in range(
            0,
            nz,
            slice_step,
        ):

            arr = np.asarray(
                volume[z_idx, :, :],
                dtype=np.float32,
            )

            (
                min_phi,
                max_phi,
                contains_interface,
                contains_nan,
            ) = slice_stats(arr)

            if (
                only_interface
                and not contains_interface
            ):
                continue

            filename = ""

            if write_files:

                save_arr = np.array(
                    arr,
                    dtype=np.float32,
                    copy=True,
                )

                if clip_phi:
                    np.clip(
                        save_arr,
                        -1.0,
                        1.0,
                        out=save_arr,
                    )

                file_path = (
                    xy_dir /
                    f"phi_xy_z{z_idx:04d}.npy"
                )

                np.save(
                    file_path,
                    save_arr,
                )

                filename = str(
                    file_path.resolve()
                )

            coordinate = (
                zmin
                + z_idx * spacing_z
            )

            writer.writerow({
                "orientation": "xy",
                "index": z_idx,
                "coordinate": coordinate,
                "rows": ny,
                "cols": nx,
                "min_phi": min_phi,
                "max_phi": max_phi,
                "contains_interface": (
                    int(contains_interface)
                ),
                "contains_nan": (
                    int(contains_nan)
                ),
                "file": filename,
            })

            xy_saved += 1

    long_saved = 0

    with open(
        long_manifest,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "orientation",
                "index",
                "coordinate",
                "rows",
                "cols",
                "min_phi",
                "max_phi",
                "contains_interface",
                "contains_nan",
                "file",
            ],
        )

        writer.writeheader()

        if longitudinal == "xz":

            indices = range(
                0,
                ny,
                slice_step,
            )

            for y_idx in indices:

                arr = np.asarray(
                    volume[:, y_idx, :],
                    dtype=np.float32,
                )

                (
                    min_phi,
                    max_phi,
                    contains_interface,
                    contains_nan,
                ) = slice_stats(arr)

                if (
                    only_interface
                    and not contains_interface
                ):
                    continue

                filename = ""

                if write_files:

                    save_arr = np.array(
                        arr,
                        dtype=np.float32,
                        copy=True,
                    )

                    if clip_phi:
                        np.clip(
                            save_arr,
                            -1.0,
                            1.0,
                            out=save_arr,
                        )

                    file_path = (
                        long_dir /
                        f"phi_xz_y{y_idx:04d}.npy"
                    )

                    np.save(
                        file_path,
                        save_arr,
                    )

                    filename = str(
                        file_path.resolve()
                    )

                coordinate = (
                    ymin
                    + y_idx * spacing_y
                )

                writer.writerow({
                    "orientation": "xz",
                    "index": y_idx,
                    "coordinate": coordinate,
                    "rows": nz,
                    "cols": nx,
                    "min_phi": min_phi,
                    "max_phi": max_phi,
                    "contains_interface": (
                        int(contains_interface)
                    ),
                    "contains_nan": (
                        int(contains_nan)
                    ),
                    "file": filename,
                })

                long_saved += 1

        else:

            indices = range(
                0,
                nx,
                slice_step,
            )

            for x_idx in indices:

                arr = np.asarray(
                    volume[:, :, x_idx],
                    dtype=np.float32,
                )

                (
                    min_phi,
                    max_phi,
                    contains_interface,
                    contains_nan,
                ) = slice_stats(arr)

                if (
                    only_interface
                    and not contains_interface
                ):
                    continue

                filename = ""

                if write_files:

                    save_arr = np.array(
                        arr,
                        dtype=np.float32,
                        copy=True,
                    )

                    if clip_phi:
                        np.clip(
                            save_arr,
                            -1.0,
                            1.0,
                            out=save_arr,
                        )

                    file_path = (
                        long_dir /
                        f"phi_yz_x{x_idx:04d}.npy"
                    )

                    np.save(
                        file_path,
                        save_arr,
                    )

                    filename = str(
                        file_path.resolve()
                    )

                coordinate = (
                    xmin
                    + x_idx * spacing_x
                )

                writer.writerow({
                    "orientation": "yz",
                    "index": x_idx,
                    "coordinate": coordinate,
                    "rows": nz,
                    "cols": ny,
                    "min_phi": min_phi,
                    "max_phi": max_phi,
                    "contains_interface": (
                        int(contains_interface)
                    ),
                    "contains_nan": (
                        int(contains_nan)
                    ),
                    "file": filename,
                })

                long_saved += 1

    print("\nSlice preparation complete.")

    print(
        f"XY slices indexed: "
        f"{xy_saved}"
    )

    print(
        f"{longitudinal.upper()} slices indexed: "
        f"{long_saved}"
    )

    print(f"\nXY manifest:")
    print(xy_manifest)

    print(
        f"\n{longitudinal.upper()} manifest:"
    )
    print(long_manifest)

    if write_files:
        print(
            "\nIndividual float32 .npy "
            "slice files were also written."
        )
    else:
        print(
            "\nNo duplicate slice files were written. "
            "The manifests can later be used with a "
            "memory-mapped PyTorch Dataset."
        )


def add_resample_arguments(parser):

    parser.add_argument(
        "vtu",
        help="Input VTU file",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--spacing",
        type=float,
        default=DEFAULT_SPACING,
        help=(
            "Cartesian spacing. "
            f"Default: {DEFAULT_SPACING}"
        ),
    )

    parser.add_argument(
        "--slab-depth",
        type=int,
        default=32,
        help=(
            "Number of z planes sampled at once. "
            "Reduce if memory is tight. Default: 32."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing phi_volume.npy",
    )


def add_slice_arguments(parser):

    parser.add_argument(
        "--longitudinal",
        choices=[
            "xz",
            "yz",
        ],
        default="xz",
        help=(
            "Perpendicular longitudinal orientation. "
            "Default: xz."
        ),
    )

    parser.add_argument(
        "--slice-step",
        type=int,
        default=1,
        help=(
            "Use every Nth slice. Default: 1."
        ),
    )

    parser.add_argument(
        "--only-interface",
        action="store_true",
        help=(
            "Keep only slices containing both "
            "positive and negative phi."
        ),
    )

    parser.add_argument(
        "--write-slice-files",
        action="store_true",
        help=(
            "Write individual .npy files for every slice. "
            "Without this flag, only manifests are created."
        ),
    )

    parser.add_argument(
        "--clip-phi",
        action="store_true",
        help=(
            "Clip saved slice files to [-1,1]. "
            "The main volume is never clipped."
        ),
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Resample phase-field VTU data onto a uniform "
            "Cartesian grid and prepare TPDM slice datasets."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    p_resample = subparsers.add_parser(
        "resample",
        help="VTU -> regular phi_volume.npy",
    )

    add_resample_arguments(
        p_resample
    )


    p_slices = subparsers.add_parser(
        "slices",
        help="Regular volume -> slice manifests/files",
    )

    p_slices.add_argument(
        "volume",
        help="phi_volume.npy",
    )

    p_slices.add_argument(
        "--metadata",
        required=True,
        help="metadata.json",
    )

    p_slices.add_argument(
        "--output",
        required=True,
        help="Slice output directory",
    )

    add_slice_arguments(
        p_slices
    )

    p_all = subparsers.add_parser(
        "all",
        help=(
            "Run resampling and slice preparation "
            "in one command"
        ),
    )

    add_resample_arguments(
        p_all
    )

    add_slice_arguments(
        p_all
    )

    args = parser.parse_args()


    if args.command == "resample":

        resample_vtu(
            vtu_path=args.vtu,
            output_dir=args.output,
            spacing=args.spacing,
            slab_depth=args.slab_depth,
            overwrite=args.overwrite,
        )

    elif args.command == "slices":

        save_slice_manifest(
            volume_path=args.volume,
            metadata_path=args.metadata,
            output_dir=args.output,
            longitudinal=args.longitudinal,
            slice_step=args.slice_step,
            only_interface=args.only_interface,
            write_files=args.write_slice_files,
            clip_phi=args.clip_phi,
        )

    elif args.command == "all":

        output_dir = Path(
            args.output
        )

        volume_path, metadata_path = (
            resample_vtu(
                vtu_path=args.vtu,
                output_dir=output_dir,
                spacing=args.spacing,
                slab_depth=args.slab_depth,
                overwrite=args.overwrite,
            )
        )

        save_slice_manifest(
            volume_path=volume_path,
            metadata_path=metadata_path,
            output_dir=(
                output_dir / "slices"
            ),
            longitudinal=args.longitudinal,
            slice_step=args.slice_step,
            only_interface=args.only_interface,
            write_files=args.write_slice_files,
            clip_phi=args.clip_phi,
        )


if __name__ == "__main__":
    main()