#!/usr/bin/env python
"""clev2er.tools.compute_fes2022_tides

Pre-compute FES2022 tide corrections for CryoSat-2 L1b files, writing one tide
file per L1b file for later use by the cryotempo chain's
``alg_fes2022_tide_correction`` algorithm.

Baseline-F replaces FES2014b with FES2022. This tool replaces the CPOM
``compute_tidal_elevations_from_pytmd.py`` script, which required a separate
``cats`` conda environment: it runs in the standard cryotempo poetry
environment, as pyTMD is now a project dependency.

Three components are computed at each L1b 20Hz nadir location:

    ocean_tide_20     FES2022 ocean tide
    load_tide_20      FES2022 ocean tide loading (a separate pyTMD model)
    ocean_tide_eq_20  long-period equilibrium tide (analytic, no model files)

**Why this is a pre-processor and not an on-the-fly chain algorithm**

Reading the FES2022 model costs ~110s, and unlike CATS2023 that cost cannot be
avoided per file in the chain (run_chain.py spawns one process per L1b file, so
nothing can be cached between files). Here we process many files in one
process, so the model is read *once* per hemisphere, cropped to the bounding
box of the files being processed, after which each file costs ~0.4s. That is
~200x faster than calling pyTMD.compute.tide_elevations per file, and gives
bit-identical results.

Example use, for one month of SARIn L1b files:

    compute_fes2022_tides.py -id $L1B_BASE_DIR/SIN/2019/05 \\
        -d $FES2022_BASE_DIR/SIN/2019/05

The output directory is the one the chain expects for that mode/year/month, ie
<fes2022_base_dir>/<SIN|LRM>/<YYYY>/<MM>/. Existing outputs are skipped unless
--overwrite is given, so a run can be resumed.

"""

import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyTMD.compute
import pyTMD.io
import pyTMD.predict
import timescale
from netCDF4 import Dataset  # pylint:disable=E0611

log = logging.getLogger(__name__)

# CryoSat-2 time_20_ku is TAI seconds since this epoch
CS2_EPOCH = (2000, 1, 1, 0, 0, 0)
CS2_TIME_STANDARD = "TAI"

# stored as int32 with a 1mm scale factor, matching the FES2014b files
# produced for baselines B-E
SCALE_FACTOR = 0.001

# padding (degrees) added around the bounding box the model is cropped to
BBOX_BUFFER_DEG = 1.0


def read_l1b_nadir(l1b_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """read the 20Hz nadir locations and times from a CryoSat L1b file

    Args:
        l1b_file (str): path of the L1b file

    Returns:
        (lats, lons, delta_time) : nadir latitude (deg), longitude (deg, in
        -180..180) and TAI seconds since CS2_EPOCH

    Raises:
        OSError : if the file can not be read
        KeyError : if the expected variables are missing
    """
    with Dataset(l1b_file) as nc:
        lats = nc.variables["lat_20_ku"][:].data.astype(np.float64)
        lons = nc.variables["lon_20_ku"][:].data.astype(np.float64)
        delta_time = nc.variables["time_20_ku"][:].data.astype(np.float64)
    # normalise longitude to -180..180 so bounding boxes are well defined
    lons = ((lons + 180.0) % 360.0) - 180.0
    return lats, lons, delta_time


def tide_from_constants(
    model: Any,
    constants: Any,
    lons: np.ndarray,
    lats: np.ndarray,
    delta_time: np.ndarray,
    extrapolate: bool = False,
    cutoff_km: float = 10.0,
) -> np.ndarray:
    """predict the tide at drift locations from pre-read model constants

    This reproduces pyTMD.compute.tide_elevations for TYPE='drift', but using
    constants that have already been read (and cropped), so that the expensive
    model read is not repeated for every file. Verified to give bit-identical
    results to tide_elevations.

    Args:
        model: pyTMD model object
        constants: constants previously returned by model.read_constants()
        lons (np.ndarray): longitudes in degrees
        lats (np.ndarray): latitudes in degrees
        delta_time (np.ndarray): TAI seconds since CS2_EPOCH
        extrapolate (bool): nearest-neighbour extrapolation beyond the grid
        cutoff_km (float): extrapolation cutoff, only used if extrapolate

    Returns:
        np.ndarray : tide elevation in metres, NaN outside the model domain
    """
    tscale = timescale.from_deltatime(delta_time, epoch=CS2_EPOCH, standard=CS2_TIME_STANDARD)
    # pyTMD's minor constituent inference underflows harmlessly; the CS2
    # retracker modules set np.seterr("raise") at import, so scope a relaxation
    with np.errstate(under="ignore"):
        amp, phase = model.interpolate_constants(
            lons,
            lats,
            constants=constants,
            method="linear",
            extrapolate=extrapolate,
            cutoff=cutoff_km,
        )
        harmonic = np.ma.array(amp * np.exp(-1j * phase * np.pi / 180.0), mask=amp.mask)
        tide = pyTMD.predict.drift(
            tscale.tide,
            harmonic,
            constants.fields,
            deltat=tscale.tt_ut1,
            corrections=model.corrections,
        )
        tide += pyTMD.predict.infer_minor(
            tscale.tide,
            harmonic,
            constants.fields,
            deltat=tscale.tt_ut1,
            corrections=model.corrections,
        )
    return np.ma.filled(tide, np.nan).astype(np.float64).flatten()


def equilibrium_tide(lons: np.ndarray, lats: np.ndarray, delta_time: np.ndarray) -> np.ndarray:
    """long-period equilibrium tide (analytic, requires no model files)

    Args:
        lons (np.ndarray): longitudes in degrees
        lats (np.ndarray): latitudes in degrees
        delta_time (np.ndarray): TAI seconds since CS2_EPOCH

    Returns:
        np.ndarray : equilibrium tide in metres
    """
    with np.errstate(under="ignore"):
        lpet = pyTMD.compute.LPET_elevations(
            x=lons,
            y=lats,
            delta_time=delta_time,
            EPSG=4326,
            EPOCH=CS2_EPOCH,
            TYPE="drift",
            TIME=CS2_TIME_STANDARD,
        )
    return np.ma.filled(lpet, np.nan).astype(np.float64).flatten()


def write_tide_file(
    out_file: str,
    ocean_tide: np.ndarray,
    ocean_tide_eq: np.ndarray,
    load_tide: np.ndarray,
) -> None:
    """write the per-L1b tide file

    The layout matches the FES2014b files used by baselines B-E: int32 values
    with a 1mm scale factor, on a 'measurements' dimension. Values outside the
    model domain are stored as zero, as they were for FES2014b.

    Args:
        out_file (str): output file path
        ocean_tide (np.ndarray): ocean tide, metres
        ocean_tide_eq (np.ndarray): equilibrium long period tide, metres
        load_tide (np.ndarray): ocean tide loading, metres
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with Dataset(out_file, "w", format="NETCDF4") as nc_out:
        nc_out.createDimension("measurements", ocean_tide.size)
        for name, values, comment in (
            ("ocean_tide_20", ocean_tide, "FES2022 ocean tide"),
            (
                "ocean_tide_eq_20",
                ocean_tide_eq,
                "long period equilibrium ocean tide",
            ),
            ("load_tide_20", load_tide, "FES2022 ocean tide loading"),
        ):
            nc_var = nc_out.createVariable(name, "i4", ("measurements",))
            nc_var.units = "m"
            nc_var.scale_factor = SCALE_FACTOR
            nc_var.add_offset = 0.0
            nc_var.comment = comment
            nc_var[:] = np.nan_to_num(values)

        nc_out.tide_model = "FES2022"
        nc_out.reference = "https://doi.org/10.24400/527896/A01-2024.004"
        nc_out.created_by = "clev2er.tools.compute_fes2022_tides"


def bounding_box(file_bounds: list[tuple[float, float, float, float]]) -> list[float]:
    """union bounding box of the supplied per-file bounds, plus a buffer

    Args:
        file_bounds: list of (lon_min, lon_max, lat_min, lat_max) per file

    Returns:
        [W, E, S, N] suitable for pyTMD's crop bounds
    """
    lon_min = min(b[0] for b in file_bounds) - BBOX_BUFFER_DEG
    lon_max = max(b[1] for b in file_bounds) + BBOX_BUFFER_DEG
    lat_min = min(b[2] for b in file_bounds) - BBOX_BUFFER_DEG
    lat_max = max(b[3] for b in file_bounds) + BBOX_BUFFER_DEG
    return [
        max(lon_min, -180.0),
        min(lon_max, 180.0),
        max(lat_min, -90.0),
        min(lat_max, 90.0),
    ]


def process_group(
    l1b_files: list[str],
    nadir: dict,
    bounds: list[float],
    models_dir: str,
    out_dir: str,
    no_load_tide: bool,
    extrapolate: bool = False,
    cutoff_km: float = 10.0,
) -> tuple[int, int]:
    """compute and write tide files for one group of L1b files

    The FES2022 ocean and load models are read once each, cropped to `bounds`,
    then applied to every file in the group.

    Args:
        l1b_files: L1b file paths in this group
        nadir: mapping of l1b file path -> (lats, lons, delta_time)
        bounds: [W, E, S, N] to crop the model to
        models_dir: pyTMD tide models directory
        out_dir: output directory
        no_load_tide: if True, write zeros for load_tide_20
        extrapolate: nearest-neighbour extrapolation beyond the model grid
        cutoff_km: extrapolation cutoff, only used if extrapolate

    Returns:
        (num_written, num_errors)
    """
    log.info(
        "Reading FES2022 ocean tide model cropped to "
        "lon %.1f..%.1f lat %.1f..%.1f (this takes a couple of minutes)",
        *bounds,
    )
    tic = time.time()
    ocean_model = pyTMD.io.model(models_dir).elevation("FES2022")
    ocean_constants = ocean_model.read_constants(
        type=ocean_model.type, crop=True, bounds=bounds, method="linear"
    )
    log.info("FES2022 ocean tide model read in %.1fs", time.time() - tic)

    load_model = None
    load_constants = None
    if not no_load_tide:
        tic = time.time()
        load_model = pyTMD.io.model(models_dir).elevation("FES2022_load")
        load_constants = load_model.read_constants(
            type=load_model.type, crop=True, bounds=bounds, method="linear"
        )
        log.info("FES2022 load tide model read in %.1fs", time.time() - tic)

    num_written = 0
    num_errors = 0
    for l1b_file in l1b_files:
        lats, lons, delta_time = nadir[l1b_file]
        out_file = os.path.join(out_dir, f"{Path(l1b_file).name[:-3]}.fes2022.nc")
        try:
            ocean_tide = tide_from_constants(
                ocean_model, ocean_constants, lons, lats, delta_time, extrapolate, cutoff_km
            )
            if load_model is not None:
                load_tide = tide_from_constants(
                    load_model, load_constants, lons, lats, delta_time, extrapolate, cutoff_km
                )
            else:
                load_tide = np.zeros_like(ocean_tide)
            ocean_tide_eq = equilibrium_tide(lons, lats, delta_time)

            write_tide_file(out_file, ocean_tide, ocean_tide_eq, load_tide)
        except (OSError, ValueError, KeyError) as exc:
            log.error("Failed for %s : %s", l1b_file, exc)
            num_errors += 1
            continue

        num_written += 1
        log.info(
            "[%d/%d] %s : %d of %d records with an ocean tide",
            num_written,
            len(l1b_files),
            Path(out_file).name,
            int(np.count_nonzero(np.isfinite(ocean_tide) & (ocean_tide != 0.0))),
            ocean_tide.size,
        )

    return num_written, num_errors


def main() -> int:
    """main function, see module docstring for use"""
    parser = argparse.ArgumentParser(
        description="Pre-compute FES2022 tide corrections for CryoSat L1b files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--l1b_dir", "-id", required=True, help="directory containing the input L1b files"
    )
    parser.add_argument(
        "--outdir", "-d", required=True, help="directory to write the tide files to"
    )
    parser.add_argument(
        "--models_dir",
        default=os.environ.get("PYTMD_TIDE_MODELS_DIR", ""),
        help="pyTMD tide models directory (default $PYTMD_TIDE_MODELS_DIR)",
    )
    parser.add_argument(
        "--max_files", type=int, default=0, help="only process this many files (0 = all)"
    )
    parser.add_argument(
        "--no_load_tide",
        action="store_true",
        help=(
            "skip the FES2022 load tide and write zeros. Only for testing when the "
            "fes2022b/load_tide model files are not available"
        ),
    )
    parser.add_argument(
        "--extrapolate",
        action="store_true",
        help=(
            "nearest-neighbour extrapolation to points outside the model grid. Off by "
            "default, matching the baselines B-E pre-processor. FES2022 has a finer, "
            "stricter coastal land mask than FES2014b, so fewer coastal points get a "
            "value without this; but extrapolated values are less reliable"
        ),
    )
    parser.add_argument(
        "--cutoff_km",
        type=float,
        default=10.0,
        help="extrapolation cutoff in km, only used with --extrapolate",
    )
    parser.add_argument("--overwrite", action="store_true", help="re-create existing outputs")
    parser.add_argument("--debug", action="store_true", help="debug level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.models_dir:
        log.error("no tide models directory: pass --models_dir or set PYTMD_TIDE_MODELS_DIR")
        return 1
    if not os.path.isdir(args.models_dir):
        log.error("tide models directory %s not found", args.models_dir)
        return 1
    if not os.path.isdir(args.l1b_dir):
        log.error("L1b directory %s not found", args.l1b_dir)
        return 1

    l1b_files = sorted(glob.glob(os.path.join(args.l1b_dir, "**", "CS_*1B*.nc"), recursive=True))
    if args.max_files:
        l1b_files = l1b_files[: args.max_files]
    if not l1b_files:
        log.error("no L1b files found in %s", args.l1b_dir)
        return 1
    log.info("Found %d L1b files in %s", len(l1b_files), args.l1b_dir)

    if not args.overwrite:
        todo = [
            f
            for f in l1b_files
            if not os.path.exists(os.path.join(args.outdir, f"{Path(f).name[:-3]}.fes2022.nc"))
        ]
        if len(todo) != len(l1b_files):
            log.info(
                "%d of %d already have tide files, skipping (use --overwrite to redo)",
                len(l1b_files) - len(todo),
                len(l1b_files),
            )
        l1b_files = todo
        if not l1b_files:
            log.info("Nothing to do")
            return 0

    # Read the nadir tracks up front. This is cheap, and lets the model be
    # cropped to the area actually covered, which is what makes the batch fast.
    nadir = {}
    north_files: list[str] = []
    south_files: list[str] = []
    north_bounds: list[tuple[float, float, float, float]] = []
    south_bounds: list[tuple[float, float, float, float]] = []
    for l1b_file in l1b_files:
        try:
            lats, lons, delta_time = read_l1b_nadir(l1b_file)
        except (OSError, KeyError) as exc:
            log.error("Could not read %s : %s", l1b_file, exc)
            continue
        nadir[l1b_file] = (lats, lons, delta_time)
        this_bounds = (lons.min(), lons.max(), lats.min(), lats.max())
        # split by hemisphere so each group crops to a compact bounding box
        if np.nanmean(lats) >= 0:
            north_files.append(l1b_file)
            north_bounds.append(this_bounds)
        else:
            south_files.append(l1b_file)
            south_bounds.append(this_bounds)

    log.info(
        "%d northern hemisphere, %d southern hemisphere files to process",
        len(north_files),
        len(south_files),
    )

    total_written = 0
    total_errors = 0
    tic = time.time()
    for group_files, group_bounds, label in (
        (north_files, north_bounds, "northern"),
        (south_files, south_bounds, "southern"),
    ):
        if not group_files:
            continue
        log.info("--- %s hemisphere: %d files ---", label, len(group_files))
        written, errors = process_group(
            group_files,
            nadir,
            bounding_box(group_bounds),
            args.models_dir,
            args.outdir,
            args.no_load_tide,
            args.extrapolate,
            args.cutoff_km,
        )
        total_written += written
        total_errors += errors

    elapsed = time.time() - tic
    log.info(
        "Wrote %d tide files (%d errors) in %.1fs (%.2fs per file)",
        total_written,
        total_errors,
        elapsed,
        elapsed / max(total_written, 1),
    )
    if args.no_load_tide:
        log.warning(
            "--no_load_tide was used: load_tide_20 is ZERO in these files. "
            "They are NOT suitable for production processing."
        )
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
