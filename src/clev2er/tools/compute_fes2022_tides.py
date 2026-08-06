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

    ocean_tide_20     FES2022 ocean tide, all 34 constituents
    load_tide_20      FES2022 ocean tide loading (a separate pyTMD model)
    ocean_tide_eq_20  long-period equilibrium tide (analytic, no model files)

The ocean tide uses the extrapolated FES2022 product and includes the
long-period constituents, both to match baselines B-E, which used pyfes with
ocean_tide_extrapolated.ini. Verified on a Greenland granule against the B-E
FES2014b file: ocean tide rms 0.53cm, load tide rms 0.06cm.

See LONG_PERIOD_CONSTITUENTS for an open question about whether the resulting
ocean_tide_20 + ocean_tide_eq_20 sum double counts the long-period tide. That
behaviour is inherited from B-E and is reproduced here deliberately.

**Why this is a pre-processor and not an on-the-fly chain algorithm**

Reading the FES2022 model costs ~110s, and unlike CATS2023 that cost cannot be
avoided per file in the chain (run_chain.py spawns one process per L1b file, so
nothing can be cached between files). Here we process many files in one
process, so the model is read *once* per hemisphere, cropped to the bounding
box of the files being processed, after which each file costs ~0.4s. That is
~200x faster than calling pyTMD.compute.tide_elevations per file, and gives
bit-identical results.

**Memory**

Cropping bounds the run time, but NOT the peak memory. Measured, peak is
essentially independent of the bbox area (2.1 GB for a 5x5 degree crop versus
1.8 GB for 50x30) and scales instead with the number of constituents, of which
FES2022 has 34: a full read needs many GB. The two models are therefore read
and released one at a time, and a `--max_memory_gb` guard (default 60% of
physical RAM) aborts the run rather than letting the machine go into swap and
be killed by the OS.

On a large production server the guard never triggers and the defaults are
right. On a workstation, use `--constituents m2 s2 n2 k1` to develop and test
cheaply - noting that output is then incomplete and not for production.

Example use, for one month of SARIn L1b files:

    compute_fes2022_tides.py -id $L1B_BASE_DIR/SIN/2019/05 \\
        -d $FES2022_BASE_DIR/SIN/2019/05

The output directory is the one the chain expects for that mode/year/month, ie
<fes2022_base_dir>/<SIN|LRM>/<YYYY>/<MM>/. Existing outputs are skipped unless
--overwrite is given, so a run can be resumed.

"""

import argparse
import gc
import glob
import logging
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

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

# fraction of physical RAM used as the default memory budget
DEFAULT_MEMORY_FRACTION = 0.6

# The 7 long-period constituents that both FES2014 and FES2022 provide.
#
# These are INCLUDED in ocean_tide_20 by default, because that is what
# baselines B-E did. B-E used pyfes:
#     ocean_tide_20, ocean_tide_eq_20, _ = short_tide.calculate(...)
# and although pyfes documents the first return as the tide and the second as
# the long-period tide, the stored ocean_tide_20 demonstrably contains the
# dynamic long-period signal: on a Greenland granule, FES2022 including these
# constituents agrees with FES2014b to rms 0.53cm, and excluding them degrades
# that to rms 3.37cm - a shift the size of the long-period tide itself.
#
# NOTE (open question, inherited from B-E): alg_geo_corrections adds
# ocean_tide_20 + ocean_tide_eq_20 over northern floating ice and ocean. Since
# ocean_tide_20 already carries a dynamic long-period tide (rms 2.7cm) and
# ocean_tide_eq_20 is an equilibrium long-period tide (rms 2.9cm) which is 0.95
# correlated with it, that sum appears to count the long-period tide twice.
# This behaviour is reproduced here deliberately, to keep baseline-F comparable
# with B-E, but it is worth a science review. --exclude_long_period exists so
# the alternative can be generated and assessed without editing code.
LONG_PERIOD_CONSTITUENTS = ("mf", "mm", "msf", "msqm", "mtm", "sa", "ssa")

# Ocean tide model. Baselines B-E used FES2014's *extrapolated* ocean tide
# (ocean_tide_extrapolated.ini), so the extrapolated FES2022 product is the
# like-for-like successor: using the plain one instead loses ~55% of the valid
# records over Greenland, because FES2022's finer coastal land mask is much
# stricter than FES2014's.
#
# The load tide is deliberately NOT the extrapolated variant: B-E used the
# plain load_tide.ini.
DEFAULT_OCEAN_MODEL = "FES2022_extrapolated"
LOAD_MODEL = "FES2022_load"


def peak_memory_gb() -> float:
    """peak resident memory of this process, in GB

    ru_maxrss is bytes on macOS and kilobytes on Linux.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024**3 if sys.platform == "darwin" else peak / 1024**2


def physical_memory_gb() -> float:
    """total physical RAM in GB, or 0.0 if it cannot be determined"""
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / 1024**3
        return (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        )  # pragma: no cover - linux only
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def start_memory_guard(budget_gb: float) -> None:
    """abort the process if peak memory exceeds `budget_gb`

    Reading the FES2022 constituents is memory hungry and, importantly,
    cropping does NOT bound it. On a workstation an unbounded run can drive the
    machine into swap and get the process (or the desktop) killed by the OS. It
    is far better to fail here, with a message saying what to do about it.

    On a large production server the budget will simply never be reached.

    Args:
        budget_gb (float): abort if peak resident memory exceeds this
    """

    def _watch() -> None:
        while True:
            peak = peak_memory_gb()
            if peak > budget_gb:
                log.error(
                    "MEMORY GUARD: peak memory %.1f GB exceeded the %.1f GB budget - aborting. "
                    "Re-run with a larger --max_memory_gb if the machine has the RAM, or "
                    "reduce --constituents, or run on a larger machine.",
                    peak,
                    budget_gb,
                )
                sys.stderr.flush()
                os._exit(2)  # pylint: disable=protected-access
            time.sleep(0.25)

    threading.Thread(target=_watch, daemon=True).start()


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
    constituents: list[str] | None = None,
    ocean_model: str = DEFAULT_OCEAN_MODEL,
    exclude_long_period: bool = False,
) -> tuple[int, int]:
    """compute and write tide files for one group of L1b files

    Each model is read once, cropped to `bounds`, and applied to every file in
    the group. The two models are read and released **one at a time**: peak
    memory is dominated by holding one model's constituents, so overlapping
    them would roughly double it for no benefit.

    Note that cropping bounds the *time* cost, not the peak memory: measured,
    peak is essentially independent of the bbox area (2.1 GB for a 5x5 degree
    crop vs 1.8 GB for 50x30) and scales instead with the number of
    constituents. See --max_memory_gb.

    Args:
        l1b_files: L1b file paths in this group
        nadir: mapping of l1b file path -> (lats, lons, delta_time)
        bounds: [W, E, S, N] to crop the model to
        models_dir: pyTMD tide models directory
        out_dir: output directory
        no_load_tide: if True, write zeros for load_tide_20
        extrapolate: nearest-neighbour extrapolation beyond the model grid
        cutoff_km: extrapolation cutoff, only used if extrapolate
        constituents: subset of constituents to use, or None for all 34
        ocean_model: pyTMD ocean tide model name
        exclude_long_period: drop the long-period constituents (not the B-E
            behaviour; for assessing the possible double count only)

    Returns:
        (num_written, num_errors)
    """

    def tides_for_all_files(model_name: str) -> dict[str, np.ndarray]:
        """read one model, apply it to every file in the group, then release it"""
        model = pyTMD.io.model(models_dir).elevation(model_name)

        # All constituents by default, long-period included, as per baselines
        # B-E. See LONG_PERIOD_CONSTITUENTS.
        wanted = list(constituents) if constituents else list(model.constituents)
        dropped: list[str] = []
        if exclude_long_period:
            dropped = [c for c in wanted if c in LONG_PERIOD_CONSTITUENTS]
            wanted = [c for c in wanted if c not in LONG_PERIOD_CONSTITUENTS]

        log.info(
            "Reading %s cropped to lon %.1f..%.1f lat %.1f..%.1f (%d constituents%s). "
            "This is one file per constituent and takes several minutes",
            model_name,
            *bounds,
            len(wanted),
            f", excluding long-period {','.join(dropped)}" if dropped else "",
        )
        tic = time.time()
        constants = model.read_constants(
            type=model.type,
            crop=True,
            bounds=bounds,
            method="linear",
            constituents=wanted,
        )
        log.info(
            "%s read in %.1fs (peak memory so far %.1f GB)",
            model_name,
            time.time() - tic,
            peak_memory_gb(),
        )
        out = {}
        report = progress_logger(len(l1b_files), f"{model_name} tides computed")
        for filenum, l1b_file in enumerate(l1b_files, start=1):
            lats, lons, delta_time = nadir[l1b_file]
            out[l1b_file] = tide_from_constants(
                model, constants, lons, lats, delta_time, extrapolate, cutoff_km
            )
            report(filenum)
        # release before the next model is read
        del constants, model
        gc.collect()
        return out

    ocean_tides = tides_for_all_files(ocean_model)
    load_tides = {} if no_load_tide else tides_for_all_files(LOAD_MODEL)

    num_written = 0
    num_errors = 0
    report = progress_logger(len(l1b_files), "tide files written")
    for filenum, l1b_file in enumerate(l1b_files, start=1):
        lats, lons, delta_time = nadir[l1b_file]
        out_file = os.path.join(out_dir, f"{Path(l1b_file).name[:-3]}.fes2022.nc")
        try:
            ocean_tide = ocean_tides[l1b_file]
            load_tide = load_tides.get(l1b_file)
            if load_tide is None:
                load_tide = np.zeros_like(ocean_tide)
            ocean_tide_eq = equilibrium_tide(lons, lats, delta_time)

            write_tide_file(out_file, ocean_tide, ocean_tide_eq, load_tide)
        except (OSError, ValueError, KeyError) as exc:
            log.error("Failed for %s : %s", l1b_file, exc)
            num_errors += 1
            continue

        num_written += 1
        report(filenum)
        # per-file detail at debug level: for a month this is thousands of lines
        log.debug(
            "%s : %d of %d records with an ocean tide",
            Path(out_file).name,
            int(np.count_nonzero(np.isfinite(ocean_tide) & (ocean_tide != 0.0))),
            ocean_tide.size,
        )

    return num_written, num_errors


def format_duration(seconds: float) -> str:
    """a short human readable duration, eg '2m30s' or '1h05m'"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def progress_logger(total: int, what: str, interval: float = 15.0) -> Callable[[int], None]:
    """make a callable that reports progress through a long loop

    Reports at most every `interval` seconds, and always on the final item, so
    that slow stages (reading thousands of L1b files over a network filesystem,
    or computing tides for a whole month) show that they are making progress
    rather than appearing to hang.

    Args:
        total (int): number of items in the loop
        what (str): short description of the work, used in the message
        interval (float): minimum seconds between reports

    Returns:
        a function to call with the number of items completed so far
    """
    start = time.time()
    last = [start]

    def report(done: int) -> None:
        now = time.time()
        if done < total and now - last[0] < interval:
            return
        last[0] = now
        elapsed = now - start
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else 0.0
        log.info(
            "  %s: %d/%d (%.0f%%), %.1f/s, ~%s remaining",
            what,
            done,
            total,
            100.0 * done / total if total else 100.0,
            rate,
            format_duration(remaining),
        )

    return report


def directories_valid(models_dir: str, l1b_dir: str) -> bool:
    """check the input directories exist

    Args:
        models_dir (str): pyTMD tide models directory
        l1b_dir (str): directory holding the input L1b files

    Returns:
        bool : True if both are usable
    """
    if not models_dir:
        log.error("no tide models directory: pass --models_dir or set PYTMD_TIDE_MODELS_DIR")
        return False
    if not os.path.isdir(models_dir):
        log.error("tide models directory %s not found", models_dir)
        return False
    if not os.path.isdir(l1b_dir):
        log.error("L1b directory %s not found", l1b_dir)
        return False
    return True


def models_available(models_dir: str, ocean_model: str, no_load_tide: bool) -> bool:
    """check the required tide models resolve, before starting a long run

    Args:
        models_dir (str): pyTMD tide models directory
        ocean_model (str): ocean tide model name
        no_load_tide (bool): if True the load tide model is not needed

    Returns:
        bool : True if every required model resolves
    """
    for model_name in [ocean_model] + ([] if no_load_tide else [LOAD_MODEL]):
        try:
            pyTMD.io.model(models_dir).elevation(model_name)
        except FileNotFoundError as exc:
            log.error("Model %s is not available: missing %s", model_name, exc)
            if model_name == DEFAULT_OCEAN_MODEL:
                log.error(
                    "Download the extrapolated FES2022 ocean tide from AVISO into "
                    "%s/fes2022b/ocean_tide_extrapolated/ (34 files, decompress any .xz), "
                    "or pass --ocean_model FES2022 to use the non-extrapolated product "
                    "- noting that loses roughly half the valid records near the coast",
                    models_dir,
                )
            return False
    return True


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
    parser.add_argument(
        "--max_memory_gb",
        type=float,
        default=0.0,
        help=(
            # NB the %% is required: ArgumentDefaultsHelpFormatter runs help
            # strings through %-formatting, so a literal % raises a TypeError
            "abort if peak memory exceeds this. 0 means auto, ie "
            f"{DEFAULT_MEMORY_FRACTION * 100:.0f}%% of physical RAM. Reading the FES2022 "
            "constituents is memory hungry and cropping does not bound it, so this "
            "guard stops a run taking a workstation down. On a large server it "
            "will never trigger"
        ),
    )
    parser.add_argument(
        "--constituents",
        nargs="+",
        default=None,
        help=(
            "only use these constituents, eg 'm2 s2 n2 k1'. Peak memory and run time "
            "scale with the number of constituents, so this makes local testing on a "
            "small machine practical. NOT for production: the output is incomplete"
        ),
    )
    parser.add_argument(
        "--ocean_model",
        default=DEFAULT_OCEAN_MODEL,
        help=(
            "pyTMD ocean tide model. The default is the extrapolated product, which is "
            "the like-for-like successor to the extrapolated FES2014 ocean tide used by "
            "baselines B-E. Pass FES2022 to use the non-extrapolated one, which loses "
            "roughly half the valid records near the Greenland coast"
        ),
    )
    parser.add_argument(
        "--exclude_long_period",
        action="store_true",
        help=(
            "drop the long-period constituents from the ocean and load tides. NOT the "
            "baselines B-E behaviour - B-E included them, and excluding them shifts the "
            "ocean tide by ~3cm rms. Provided so the possible long-period double count "
            "(see LONG_PERIOD_CONSTITUENTS) can be assessed"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="re-create existing outputs")
    parser.add_argument("--debug", action="store_true", help="debug level logging")
    args = parser.parse_args()

    # force=True: something in the import chain (pyTMD/clev2er) already
    # configures the root logger, which would otherwise make this a no-op and
    # silence the whole run - including the memory guard's abort message.
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,
    )

    budget_gb = args.max_memory_gb
    if budget_gb <= 0:
        physical_gb = physical_memory_gb()
        budget_gb = physical_gb * DEFAULT_MEMORY_FRACTION if physical_gb else 8.0
    log.info(
        "Memory budget %.1f GB (physical RAM %.1f GB). Peak scales with the number of "
        "constituents, not the area processed.",
        budget_gb,
        physical_memory_gb(),
    )
    start_memory_guard(budget_gb)

    if not directories_valid(args.models_dir, args.l1b_dir):
        return 1

    if not models_available(args.models_dir, args.ocean_model, args.no_load_tide):
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

    # Read the nadir tracks up front, so the models can be cropped to the area
    # actually covered. Only the coordinate variables are read, but over a
    # network filesystem this is still several minutes for a month of files.
    log.info(
        "Reading nadir tracks from %d L1b files, to find the area to crop the models to",
        len(l1b_files),
    )
    tic = time.time()
    report = progress_logger(len(l1b_files), "scanned")
    nadir = {}
    north_files: list[str] = []
    south_files: list[str] = []
    north_bounds: list[tuple[float, float, float, float]] = []
    south_bounds: list[tuple[float, float, float, float]] = []
    for filenum, l1b_file in enumerate(l1b_files, start=1):
        report(filenum)
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
    log.info("Scanned %d L1b files in %s", len(l1b_files), format_duration(time.time() - tic))

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
            args.constituents,
            args.ocean_model,
            args.exclude_long_period,
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
