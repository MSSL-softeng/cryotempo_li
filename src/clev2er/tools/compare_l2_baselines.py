#!/usr/bin/env python
"""Compare CryoTEMPO Land Ice L2 products from two baselines.

This measures what a chain evolution actually did to the product, rather than
to an intermediate quantity. It is the definitive check for the baseline-F tide
evolutions: unlike comparing the tide files directly, the corrections are only
applied where the chain applies them, so extrapolated tide values over grounded
ice - which the chain computes but never uses - cannot skew the result.

Granules are paired by their filename with the trailing _<BVVV> baseline and
version stripped. Records are matched on `time`, NOT by index: a changed
correction can change which records survive the height filters, so the two
products may hold different numbers of records for the same granule.

Differences are reported per surface type, read from the product's own
flag_meanings so the mapping cannot drift.

Usage:
    compare_l2_baselines.py <baseline_a> <baseline_b> [options]

      --zone GREENL|ANTARC   default GREENL
      --year YYYY            default 2020
      --month MM             default 01
      --base DIR             product base dir, default $CT_PRODUCT_BASEDIR

Example, FES2014b/CATS2008a (E001) against FES2022/CATS2023 (F010):

    compare_l2_baselines.py E001 F010 --zone GREENL --year 2020 --month 01
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
from netCDF4 import Dataset  # pylint:disable=E0611

# the baseline/version suffix at the end of a TDP filename, eg _E001
BASELINE_SUFFIX = re.compile(r"_[A-Z]\d{3}\.nc$")


def granule_key(path: str) -> str:
    """filename with the _<BVVV> baseline/version suffix removed

    Args:
        path (str): product file path

    Returns:
        str : key that is the same for the same granule in either baseline
    """
    return BASELINE_SUFFIX.sub("", os.path.basename(path))


def surface_type_names(dataset: Dataset) -> dict[int, str]:
    """map surface_type values to names, from the product's own metadata

    Args:
        dataset (Dataset): an open L2 product

    Returns:
        dict : {flag value: name}
    """
    try:
        meanings = dataset.variables["surface_type"].flag_meanings.split()
    except (KeyError, AttributeError):
        return {}
    # names can carry a parenthesised comment, eg non_greenland_land(...)
    return {i: name.split("(")[0] for i, name in enumerate(meanings)}


def read_product(path: str) -> dict[str, np.ndarray]:
    """read the fields needed for the comparison

    Args:
        path (str): product file path

    Returns:
        dict of arrays, plus 'surface_type_names'
    """
    with Dataset(path) as nc:
        out = {
            name: np.ma.filled(nc.variables[name][:], np.nan).astype(np.float64)
            for name in ("time", "elevation", "latitude", "longitude")
        }
        out["surface_type"] = np.ma.filled(nc.variables["surface_type"][:], -128).astype(int)
        out["names"] = surface_type_names(nc)
    return out


def describe(label: str, values: np.ndarray) -> None:
    """print a one line summary of an elevation difference, in cm"""
    if values.size == 0:
        print(f"  {label:22s} {'(no records)':>12}")
        return
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        print(f"  {label:22s} {'(all NaN)':>12}")
        return
    diff = finite * 100.0
    print(
        f"  {label:22s} {finite.size:9d} pts  mean {diff.mean():+8.3f}  "
        f"rms {np.sqrt((diff**2).mean()):8.3f}  "
        f"p50 {np.percentile(np.abs(diff), 50):7.3f}  "
        f"p99 {np.percentile(np.abs(diff), 99):8.3f}  "
        f"max {np.abs(diff).max():9.3f}   (cm)"
    )


def compare_pair(path_a: str, path_b: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """compare one granule

    Args:
        path_a (str): product from the first baseline
        path_b (str): product from the second baseline

    Returns:
        (elevation differences b-a, surface types, (n_a, n_b, n_matched))
    """
    prod_a, prod_b = read_product(path_a), read_product(path_b)
    # match on time: a changed correction can change which records pass the
    # height filters, so the products need not be record aligned
    _, idx_a, idx_b = np.intersect1d(prod_a["time"], prod_b["time"], return_indices=True)
    counts = (prod_a["time"].size, prod_b["time"].size, idx_a.size)
    if idx_a.size == 0:
        return np.array([]), np.array([]), counts
    diff = prod_b["elevation"][idx_b] - prod_a["elevation"][idx_a]
    return diff, prod_a["surface_type"][idx_a], counts


def main() -> int:
    """compare two baselines of the L2 product

    Returns:
        int : 0 on success, 1 if no granules could be paired
    """
    parser = argparse.ArgumentParser(
        description="Compare CryoTEMPO Land Ice L2 products from two baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("baseline_a", help="reference baseline and version, eg E001")
    parser.add_argument("baseline_b", help="baseline and version to compare, eg F010")
    parser.add_argument("--zone", default="GREENL", choices=["GREENL", "ANTARC"])
    parser.add_argument("--year", default="2020")
    parser.add_argument("--month", default="01")
    parser.add_argument(
        "--base",
        default=os.environ.get("CT_PRODUCT_BASEDIR", ""),
        help="product base directory (default $CT_PRODUCT_BASEDIR)",
    )
    args = parser.parse_args()

    if not args.base:
        print("no product base directory: pass --base or set CT_PRODUCT_BASEDIR")
        return 1

    def product_dir(baseline: str) -> str:
        return os.path.join(
            args.base, baseline[0], baseline[1:], "LAND_ICE", args.zone, args.year, args.month
        )

    dir_a, dir_b = product_dir(args.baseline_a), product_dir(args.baseline_b)
    files_a = {granule_key(f): f for f in glob.glob(os.path.join(dir_a, "*.nc"))}
    files_b = {granule_key(f): f for f in glob.glob(os.path.join(dir_b, "*.nc"))}
    shared = sorted(set(files_a) & set(files_b))

    print(f"{args.baseline_a}: {len(files_a)} granules in {dir_a}")
    print(f"{args.baseline_b}: {len(files_b)} granules in {dir_b}")
    print(f"paired: {len(shared)}")
    if not shared:
        print("Nothing to compare - check the baselines, zone and date")
        return 1
    only_a, only_b = set(files_a) - set(files_b), set(files_b) - set(files_a)
    if only_a or only_b:
        print(
            f"  only in {args.baseline_a}: {len(only_a)}, only in {args.baseline_b}: {len(only_b)}"
        )

    all_diff: list[np.ndarray] = []
    all_stype: list[np.ndarray] = []
    names: dict[int, str] = {}
    total_a = total_b = total_matched = 0
    for key in shared:
        try:
            diff, stype, counts = compare_pair(files_a[key], files_b[key])
        except (OSError, KeyError) as exc:
            print(f"  skipping {key}: {exc}")
            continue
        total_a += counts[0]
        total_b += counts[1]
        total_matched += counts[2]
        if diff.size:
            all_diff.append(diff)
            all_stype.append(stype)
        if not names:
            with Dataset(files_a[key]) as nc:
                names = surface_type_names(nc)

    if not all_diff:
        print("No records could be matched by time")
        return 1

    diff = np.concatenate(all_diff)
    stype = np.concatenate(all_stype)
    print(
        f"\nrecords: {args.baseline_a} {total_a}, {args.baseline_b} {total_b} "
        f"({100.0 * total_b / total_a - 100:+.2f}%), matched on time {total_matched}"
    )

    print("\nelevation difference, " f"{args.baseline_b} minus {args.baseline_a}:")
    describe("all surfaces", diff)
    for value in sorted(set(stype.tolist())):
        label = names.get(value, f"surface_type {value}")
        describe(f"{value}: {label}", diff[stype == value])

    # count only records with a valid elevation in BOTH, so records dropped by
    # one baseline do not dilute the percentage
    comparable = np.isfinite(diff)
    changed = comparable & (np.abs(diff) > 1e-4)
    print(
        f"\n{int(changed.sum())} of {int(comparable.sum())} records with a valid elevation "
        f"in both changed ({100.0 * changed.sum() / max(comparable.sum(), 1):.2f}%)"
    )
    if changed.any():
        moved = np.abs(diff[changed]) * 100.0
        print(f"  of those: median |change| {np.median(moved):.2f} cm, max {moved.max():.2f} cm")
    print(
        "\nWhich surface types should move depends on the evolution. The tide\n"
        "evolutions touch only ocean and floating ice, so a difference over grounded\n"
        "ice would indicate a correction applied where it should not be; a geolocation\n"
        "change can legitimately move any surface type."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
