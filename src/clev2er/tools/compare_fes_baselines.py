#!/usr/bin/env python
"""Compare pre-computed FES2022 tide files against the baselines B-E FES2014b
files, over every granule that exists in both.

Reports each variable separately, and the sum that alg_geo_corrections actually
applies over northern floating ice and ocean (ocean_tide_20 + ocean_tide_eq_20).

The question this answers: is the ~1.2cm offset seen on a single granule a
systematic bias, or one sample of a slowly varying difference? A distribution
straddling zero means the latter.

Usage:
    compare_fes.py [glob-under-FES2022_BASE_DIR]
    compare_fes.py 'SIN/2020/01/*.fes2022.nc'      # default
"""

import glob
import os
import sys

import numpy as np
from netCDF4 import Dataset  # pylint:disable=E0611

VARIABLES = ("ocean_tide_20", "ocean_tide_eq_20", "load_tide_20")


def read_vars(path: str) -> dict[str, np.ndarray]:
    """read the tide variables from a tide file

    Args:
        path (str): tide file path

    Returns:
        dict : {variable name: values in metres}, masked/missing as 0.0
    """
    with Dataset(path) as nc:
        return {v: np.ma.filled(nc.variables[v][:], 0.0).astype(np.float64) for v in VARIABLES}


def main() -> int:
    """compare every FES2022 tide file with its FES2014b counterpart

    Returns:
        int : 0 on success, 1 if no comparable granules were found
    """
    new_base = os.environ["FES2022_BASE_DIR"]
    old_base = os.environ["FES2014B_BASE_DIR"]
    pattern = sys.argv[1] if len(sys.argv) > 1 else "SIN/2020/01/*.fes2022.nc"

    new_files = sorted(glob.glob(os.path.join(new_base, pattern)))
    print(f"{len(new_files)} FES2022 files matching {pattern}")

    # accumulate per-granule means, and all point differences
    per_granule: dict[str, list[float]] = {v: [] for v in (*VARIABLES, "applied_sum")}
    all_points: dict[str, list[np.ndarray]] = {v: [] for v in (*VARIABLES, "applied_sum")}
    dates = []
    n_pairs = 0
    coverage = []

    for new_f in new_files:
        rel = os.path.relpath(new_f, new_base)
        old_f = os.path.join(old_base, rel).replace(".fes2022.nc", ".fes2014b.nc")
        if not os.path.exists(old_f):
            continue
        try:
            new_v, old_v = read_vars(new_f), read_vars(old_f)
        except (OSError, KeyError) as exc:
            print(f"  skipping {os.path.basename(new_f)}: {exc}")
            continue
        if new_v["ocean_tide_20"].size != old_v["ocean_tide_20"].size:
            print(f"  skipping {os.path.basename(new_f)}: record count differs")
            continue

        n_pairs += 1
        # granule start date, from the CryoSat filename
        dates.append(os.path.basename(new_f)[19:27])
        coverage.append(
            (
                int(np.count_nonzero(old_v["ocean_tide_20"])),
                int(np.count_nonzero(new_v["ocean_tide_20"])),
            )
        )

        # valid in BOTH ocean tides: the points the chain would actually correct
        mask = (old_v["ocean_tide_20"] != 0) & (new_v["ocean_tide_20"] != 0)
        if not mask.any():
            continue
        for v in VARIABLES:
            d = (new_v[v][mask] - old_v[v][mask]) * 100.0
            per_granule[v].append(d.mean())
            all_points[v].append(d)
        applied_new = new_v["ocean_tide_20"] + new_v["ocean_tide_eq_20"]
        applied_old = old_v["ocean_tide_20"] + old_v["ocean_tide_eq_20"]
        d = (applied_new[mask] - applied_old[mask]) * 100.0
        per_granule["applied_sum"].append(d.mean())
        all_points["applied_sum"].append(d)

    if not n_pairs:
        print("No granules found in both baselines - check the paths and pattern")
        return 1

    old_cov = sum(c[0] for c in coverage)
    new_cov = sum(c[1] for c in coverage)
    print(f"\n{n_pairs} granules in both baselines, " f"{min(dates)}..{max(dates)}")
    print(
        f"ocean tide coverage: FES2014b {old_cov} records, FES2022 {new_cov} "
        f"({100.0 * new_cov / old_cov - 100:+.1f}%)"
    )

    print(
        f"\n{'variable':18s} {'pts':>9} {'mean cm':>9} {'rms cm':>8} "
        f"{'per-granule mean cm':>28}"
    )
    print(f"{'':18s} {'':>9} {'':>9} {'':>8} {'mean':>8} {'sd':>8} {'min':>8} {'max':>8}")
    for v in (*VARIABLES, "applied_sum"):
        pts = np.concatenate(all_points[v])
        gran = np.array(per_granule[v])
        print(
            f"{v:18s} {pts.size:9d} {pts.mean():9.3f} {np.sqrt((pts**2).mean()):8.3f} "
            f"{gran.mean():8.3f} {gran.std():8.3f} {gran.min():8.3f} {gran.max():8.3f}"
        )

    gran = np.array(per_granule["ocean_tide_eq_20"])
    frac_neg = float((gran < 0).mean())
    print(
        f"\nocean_tide_eq_20 per-granule means: {100 * frac_neg:.0f}% negative, "
        f"{100 * (1 - frac_neg):.0f}% positive"
    )
    print(
        "  a systematic bias would be almost all one sign;\n"
        "  a slowly varying difference should straddle zero across the month."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
