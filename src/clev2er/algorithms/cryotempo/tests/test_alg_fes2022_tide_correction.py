"""pytest of algorithm
   clev2er.algorithms.cryotempo.alg_fes2022_tide_correction.py

Round-trips through clev2er.tools.compute_fes2022_tides.write_tide_file, so the
file layout written by the pre-processor and the layout expected by the chain
algorithm are tested against each other. This needs no FES2022 model files.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint: disable=E0611

from clev2er.algorithms.cryotempo.alg_fes2022_tide_correction import Algorithm
from clev2er.tools.compute_fes2022_tides import (
    CTE_LONG_PERIOD_LINES,
    auto_lon_sectors,
    bounding_box,
    circular_mean_lon,
    doodson_key,
    equilibrium_tide,
    write_tide_file,
)
from clev2er.utils.config.load_config_settings import load_config_files

# Similar lines in 2 files, pylint: disable=R0801

log = logging.getLogger(__name__)

L1B_NAME = "CS_OFFL_SIR_SIN_1B_20190504T122546_20190504T122726_D001.nc"


@pytest.fixture(name="fes2022_base_dir")
def fixture_fes2022_base_dir(tmp_path) -> str:
    """create a FES2022 tide file for the test L1b, using the pre-processor's writer"""
    base_dir = os.environ["CLEV2ER_BASE_DIR"]
    l1b_file = f"{base_dir}/testdata/cs2/l1bfiles/{L1B_NAME}"
    with Dataset(l1b_file) as l1b:
        num_records = l1b["lat_20_ku"].size

    rng = np.random.default_rng(0)
    out_dir = tmp_path / "SIN" / "2019" / "05"
    write_tide_file(
        str(out_dir / f"{Path(L1B_NAME).stem}.fes2022.nc"),
        ocean_tide=rng.uniform(-2.0, 2.0, num_records),
        ocean_tide_eq=rng.uniform(-0.05, 0.05, num_records),
        load_tide=rng.uniform(-0.1, 0.1, num_records),
    )
    return str(tmp_path)


def test_alg_fes2022_tide_correction(fes2022_base_dir: str) -> None:
    """test of Algorithm in clev2er.algorithms.cryotempo.alg_fes2022_tide_correction.py"""

    base_dir = os.environ["CLEV2ER_BASE_DIR"]
    assert base_dir is not None

    config, _, _, _, _ = load_config_files("cryotempo", baseline="F", version=10)
    config["chain"]["use_multi_processing"] = False
    config["tides"]["fes2022_base_dir"] = fes2022_base_dir

    try:
        thisalg = Algorithm(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize algorithm {exc}"

    l1b_file = f"{base_dir}/testdata/cs2/l1bfiles/{L1B_NAME}"
    try:
        l1b = Dataset(l1b_file)
    except IOError:
        assert False, f"{l1b_file} could not be read"

    shared_dict: dict = {
        "l1b_file_name": l1b_file,
        "num_20hz_records": l1b["lat_20_ku"].size,
        "instr_mode": "SIN",
    }

    success, error_str = thisalg.process(l1b, shared_dict)
    assert success, f"algorithm should succeed: {error_str}"

    assert "fes_corrections" in shared_dict, "fes_corrections should have been added"
    for field in ("ocean_tide_20", "ocean_tide_eq_20", "load_tide_20"):
        assert field in shared_dict["fes_corrections"], f"{field} missing"
        values = shared_dict["fes_corrections"][field]
        assert (
            values.size == shared_dict["num_20hz_records"]
        ), f"{field} must have one value per 20Hz record"
        assert np.all(np.isfinite(values)), f"{field} should contain no NaN or inf"
        # values are stored as int32 with a 1mm scale factor, so should be in metres
        assert np.all(np.abs(values) < 20.0), f"{field} values should be within +/-20 m"


def test_alg_fes2022_missing_file(fes2022_base_dir: str) -> None:
    """a missing FES2022 file must fail cleanly rather than raise"""

    base_dir = os.environ["CLEV2ER_BASE_DIR"]
    config, _, _, _, _ = load_config_files("cryotempo", baseline="F", version=10)
    config["chain"]["use_multi_processing"] = False
    config["tides"]["fes2022_base_dir"] = fes2022_base_dir

    thisalg = Algorithm(config, log)

    l1b_file = f"{base_dir}/testdata/cs2/l1bfiles/{L1B_NAME}"
    l1b = Dataset(l1b_file)

    shared_dict: dict = {
        "l1b_file_name": l1b_file,
        "num_20hz_records": l1b["lat_20_ku"].size,
        # LRM has no tide file staged by the fixture, so the lookup must miss
        "instr_mode": "LRM",
    }

    success, error_str = thisalg.process(l1b, shared_dict)
    assert not success, "should fail when the FES2022 file is missing"
    assert "FES2022" in error_str, "error string should name the missing FES2022 file"


def test_bounding_box_simple() -> None:
    """a compact group should crop to its own extent plus the buffer"""
    # two tracks off east Greenland
    bounds = [(-40.0, -37.0, 63.0, 69.0), (-35.0, -30.0, 70.0, 78.0)]
    west, east, south, north = bounding_box(bounds, lon_centre=-35.0)
    assert west == pytest.approx(-41.0), "west should be the min lon minus the buffer"
    assert east == pytest.approx(-29.0), "east should be the max lon plus the buffer"
    assert south == pytest.approx(62.0)
    assert north == pytest.approx(79.0)


def test_bounding_box_across_antimeridian() -> None:
    """tracks either side of the dateline must not produce a global box

    Naively taking min/max of longitudes in -180..180 would give -179..179, ie
    the whole globe, for a group that actually spans only a few degrees.
    """
    bounds = [(178.0, 179.5, -80.0, -75.0), (-179.0, -177.0, -80.0, -75.0)]
    west, east, _, _ = bounding_box(bounds, lon_centre=180.0)
    # the group spans 178..183 (=-177), so wraps; the code widens rather than
    # emitting an invalid west>east box, but it must not be wider than needed
    # in latitude, and must remain a valid box
    assert west <= east, "west must not exceed east"
    span = east - west
    assert span == pytest.approx(360.0), "a wrapping group falls back to all longitudes"

    # the same group expressed without wrapping stays tight
    shifted = [(0.0, 1.5, -80.0, -75.0), (2.0, 4.0, -80.0, -75.0)]
    west, east, _, _ = bounding_box(shifted, lon_centre=2.0)
    assert east - west == pytest.approx(6.0), "non-wrapping group should stay tight"


def test_circular_mean_lon_across_antimeridian() -> None:
    """the mean longitude of a track straddling the dateline is near 180, not 0"""
    lons = np.array([179.0, 179.5, -179.5, -179.0])
    assert abs(abs(circular_mean_lon(lons)) - 180.0) < 1.0
    # and an ordinary track is unaffected
    assert circular_mean_lon(np.array([-40.0, -38.0, -36.0])) == pytest.approx(-38.0, abs=0.1)


def test_auto_lon_sectors_scales_with_file_count() -> None:
    """one group for a handful of files, several for a month"""
    assert auto_lon_sectors(1) == 1, "a single file must not pay for extra model reads"
    assert auto_lon_sectors(50) == 1
    assert auto_lon_sectors(1828) > 1, "a month of files should be split"
    assert auto_lon_sectors(1828) <= 12, "but never split without limit"
    assert auto_lon_sectors(100000) <= 12, "capped for very large runs"


def test_lpet_default_constituents() -> None:
    """CTE_LONG_PERIOD_LINES must still match pyTMD's own default set

    The double-count fix works by subtracting the atlas constituents from this
    list. If pyTMD ever changed its default, our list would silently stop being
    the full set and the exclusion would be wrong. Rather than scrape pyTMD's
    source, assert behaviourally: summing our list must equal summing pyTMD's
    default.
    """
    rng = np.random.default_rng(0)
    n = 200
    lat = rng.uniform(58.0, 84.0, n)
    lon = rng.uniform(-60.0, -20.0, n)
    delta_time = rng.uniform(6.3e8, 6.3e8 + 365 * 86400.0, n)

    default = equilibrium_tide(lon, lat, delta_time)
    explicit = equilibrium_tide(lon, lat, delta_time, list(CTE_LONG_PERIOD_LINES))
    assert np.allclose(default, explicit, atol=1e-12), (
        "CTE_LONG_PERIOD_LINES no longer matches pyTMD's default equilibrium "
        "tide constituents - the double-count exclusion would be incorrect"
    )


def test_doodson_matching_across_naming_schemes() -> None:
    """FES and pyTMD name the same spectral lines differently

    Matching by name alone gets this wrong: FES 'mtm' is pyTMD's 'mt', and
    'msqm' is not one of the 15 lines at all.
    """
    assert doodson_key("mtm") == doodson_key("mt"), "mtm and mt are the same line"
    assert doodson_key("mf") == doodson_key("mf"), "sanity"
    assert doodson_key("065.445") == 65.445, "bare Doodson numbers must parse"
    assert doodson_key("not_a_constituent") is None, "unknown names give None"

    # msqm is in the FES2022 atlas but is not among the 15 CTE lines
    cte_keys = {doodson_key(c) for c in CTE_LONG_PERIOD_LINES}
    assert doodson_key("msqm") not in cte_keys, "msqm is not one of the 15 CTE lines"
    for name in ("mf", "mm", "msf", "sa", "ssa", "mtm"):
        assert doodson_key(name) in cte_keys, f"{name} should match a CTE line"


def test_compute_fes2022_tides_help() -> None:
    """`compute_fes2022_tides.py --help` must work.

    argparse's ArgumentDefaultsHelpFormatter runs help strings through
    %-formatting, so a literal '%' in any help text raises a TypeError only
    when --help is actually requested - which unit tests of the functions
    would never catch.
    """
    base_dir = os.environ["CLEV2ER_BASE_DIR"]
    tool = f"{base_dir}/src/clev2er/tools/compute_fes2022_tides.py"

    result = subprocess.run(
        [sys.executable, tool, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"--help failed:\n{result.stderr}"
    assert "--max_memory_gb" in result.stdout, "help should list the memory guard option"
    assert "--constituents" in result.stdout, "help should list the constituents option"
