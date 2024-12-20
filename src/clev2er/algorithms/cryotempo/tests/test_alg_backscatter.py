"""pytest of algorithm
   clev2er.algorithms.cryotempo.alg_backscatter.py
"""

import logging
import os

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint: disable=E0611

from clev2er.algorithms.cryotempo.alg_backscatter import Algorithm
from clev2er.utils.config.load_config_settings import load_config_files

# Similar lines in 2 files, pylint: disable=R0801

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "l1b_file",
    [
        ("CS_OFFL_SIR_SIN_1B_20190504T122546_20190504T122726_D001.nc"),
        ("CS_OFFL_SIR_LRM_1B_20200930T231147_20200930T232634_D001.nc"),
    ],
)
def test_alg_backscatter(l1b_file) -> None:
    """test of clev2er.algorithms.cryotempo.alg_backscatter.py"""

    base_dir = os.environ["CLEV2ER_BASE_DIR"]
    assert base_dir is not None

    # Load merged config file for chain
    config, _, _, _, _ = load_config_files("cryotempo")

    # Set to Sequential Processing
    config["chain"]["use_multi_processing"] = False

    # Initialise the Algorithm
    try:
        thisalg = Algorithm(config, log)  # no config used for this alg
    except KeyError as exc:
        assert False, f"Could not initialize algorithm {exc}"

    # -------------------------------------------------------------------------
    # Test with L1b file

    l1b_file = f"{base_dir}/testdata/cs2/l1bfiles/{l1b_file}"
    try:
        l1b = Dataset(l1b_file)
        log.info("Opened %s", l1b_file)
    except IOError:
        assert False, f"{l1b_file} could not be read"

    # Run  Algorithm.process()
    shared_dict = {}

    # setup dummy shared_dict results from other algorithms
    if "SIR_SIN_1B" in l1b_file:
        shared_dict["instr_mode"] = "SIN"
    if "SIR_LRM_1B" in l1b_file:
        shared_dict["instr_mode"] = "LRM"
    shared_dict["pwr_at_rtrk_point"] = np.full_like(l1b["lat_20_ku"][:].data, 100.0)
    shared_dict["range_cor_20_ku"] = np.full_like(l1b["lat_20_ku"][:].data, 70000.0)

    # Run the alg process
    success, _ = thisalg.process(l1b, shared_dict)
    assert success, "algorithm should not fail"

    assert "sig0_20_ku" in shared_dict, "sig0_20_ku not in shared_dict"
