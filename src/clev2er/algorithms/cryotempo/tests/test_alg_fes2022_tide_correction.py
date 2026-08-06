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
from clev2er.tools.compute_fes2022_tides import write_tide_file
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
