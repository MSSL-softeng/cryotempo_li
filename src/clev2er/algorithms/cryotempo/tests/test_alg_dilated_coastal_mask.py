"""pytest of algorithm
   clev2er.algorithms.cryotempo.alg_dilated_coastal_mask.py
"""

import logging
import os

import numpy as np
from netCDF4 import Dataset  # pylint: disable=E0611

from clev2er.algorithms.cryotempo.alg_dilated_coastal_mask import Algorithm
from clev2er.utils.config.load_config_settings import load_config_files

# Similar lines in 2 files, pylint: disable=R0801
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals

log = logging.getLogger(__name__)


def test_alg_dilated_coastal_mask() -> None:
    """test of Algorithm in clev2er.algorithms.cryotempo.alg_dilated_coastal_mask.py"""

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
    # Test with L1b file that has 71% inside Antarctic mask

    l1b_file = (
        f"{base_dir}/testdata/cs2/l1bfiles/"
        "CS_OFFL_SIR_SIN_1B_20190504T122546_20190504T122726_D001.nc"
    )
    try:
        l1b = Dataset(l1b_file)
        log.info("Opened %s", l1b_file)
    except IOError:
        assert False, f"{l1b_file} could not be read"

    # Run  Algorithm.process()
    shared_dict = {}

    # setup dummy shared_dict results from other algorithms
    shared_dict["l1b_file_name"] = l1b_file
    shared_dict["num_20hz_records"] = l1b["lat_20_ku"].size
    shared_dict["hemisphere"] = "south"

    shared_dict["lats_nadir"] = l1b["lat_20_ku"][:].data
    shared_dict["lons_nadir"] = l1b["lon_20_ku"][:].data % 360.0  # [-180,+180E] -> 0..360E

    # This should fail, as file is outside cryosphere
    success, _ = thisalg.process(l1b, shared_dict)

    assert success, "Should not fail"
    assert "dilated_surface_mask" in shared_dict, "dilated_surface_mask should have been added"

    num_in_mask = np.count_nonzero(shared_dict["dilated_surface_mask"])
    num_records = len(shared_dict["lats_nadir"])
    assert (num_in_mask * 100.0 / num_records) > 70.0, "% in mask should > 70%"

    # -------------------------------------------------------------------------
    # Test with L1b file that has 100% inside Greenland dilated mask

    l1b_file = (
        f"{base_dir}/testdata/cs2/l1bfiles/"
        "CS_LTA__SIR_LRM_1B_20200930T235609_20200930T235758_E001.nc"
    )
    try:
        l1b = Dataset(l1b_file)
        log.info("Opened %s", l1b_file)
    except IOError:
        assert False, f"{l1b_file} could not be read"

    # Run  Algorithm.process()
    shared_dict = {}

    # setup dummy shared_dict results from other algorithms
    shared_dict["l1b_file_name"] = l1b_file
    shared_dict["num_20hz_records"] = l1b["lat_20_ku"].size
    shared_dict["hemisphere"] = "north"

    shared_dict["lats_nadir"] = l1b["lat_20_ku"][:].data
    shared_dict["lons_nadir"] = l1b["lon_20_ku"][:].data % 360.0  # [-180,+180E] -> 0..360E

    # This should fail, as file is outside cryosphere
    success, _ = thisalg.process(l1b, shared_dict)

    assert success, "Should not fail"
    assert "dilated_surface_mask" in shared_dict, "dilated_surface_mask should have been added"

    num_in_mask = np.count_nonzero(shared_dict["dilated_surface_mask"])
    num_records = len(shared_dict["lats_nadir"])
    assert (num_in_mask * 100.0 / num_records) > 99.0, "% in mask should > 99%"
