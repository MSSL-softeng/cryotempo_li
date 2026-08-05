"""pytest of algorithm
   clev2er.algorithms.cryotempo.alg_cats2023_tide_correction.py
"""

import logging
import os

import numpy as np
from netCDF4 import Dataset  # pylint: disable=E0611

from clev2er.algorithms.cryotempo.alg_cats2023_tide_correction import Algorithm
from clev2er.algorithms.cryotempo.alg_surface_type import Algorithm as SurfaceType
from clev2er.utils.config.load_config_settings import load_config_files

# Similar lines in 2 files, pylint: disable=R0801

log = logging.getLogger(__name__)


def _shared_dict_for(l1b, l1b_file: str, hemisphere: str, instr_mode: str) -> dict:
    """build the minimal shared_dict this algorithm requires"""
    shared_dict: dict = {}
    shared_dict["l1b_file_name"] = l1b_file
    shared_dict["num_20hz_records"] = l1b["lat_20_ku"].size
    shared_dict["hemisphere"] = hemisphere
    shared_dict["instr_mode"] = instr_mode
    shared_dict["lats_nadir"] = l1b["lat_20_ku"][:].data
    shared_dict["lons_nadir"] = l1b["lon_20_ku"][:].data % 360.0  # [-180,+180E] -> 0..360E
    return shared_dict


def test_alg_cats2023_tide_correction() -> None:
    """test of Algorithm in clev2er.algorithms.cryotempo.alg_cats2023_tide_correction.py

    Baseline-F computes the CATS2008-v2023 tide on the fly, rather than reading the
    pre-computed CATS2008a file used in baselines B-E.
    """

    base_dir = os.environ["CLEV2ER_BASE_DIR"]
    assert base_dir is not None

    # Load merged config file for chain
    config, _, _, _, _ = load_config_files("cryotempo", baseline="F", version=10)

    # Set to Sequential Processing
    config["chain"]["use_multi_processing"] = False

    try:
        surface_type = SurfaceType(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize SurfaceType algorithm {exc}"

    # Initialise the Algorithm
    try:
        thisalg = Algorithm(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize algorithm {exc}"

    # -------------------------------------------------------------------------
    # Test with SIN L1b file over Antarctica

    l1b_file = (
        f"{base_dir}/testdata/cs2/l1bfiles/"
        "CS_OFFL_SIR_SIN_1B_20190504T122546_20190504T122726_D001.nc"
    )
    try:
        l1b = Dataset(l1b_file)
        log.info("Opened %s", l1b_file)
    except IOError:
        assert False, f"{l1b_file} could not be read"

    shared_dict = _shared_dict_for(l1b, l1b_file, "south", "SIN")

    success, _ = surface_type.process(l1b, shared_dict)
    assert success, "surface_type algorithm should not fail"

    success, _ = thisalg.process(l1b, shared_dict)

    assert success, "Should succeed computing the CATS2023 tide"
    assert "cats_tide" in shared_dict, "cats_tide should have been added"
    assert shared_dict[
        "cats_tide_required"
    ], "cats_tide_required should have been added and be True"

    cats_tide = shared_dict["cats_tide"]

    # The tide array must be parallel to the L1b 20Hz records, as
    # alg_geo_corrections indexes it with floating_ice_locations/ocean_locations
    assert (
        cats_tide.size == shared_dict["num_20hz_records"]
    ), "cats_tide must have one value per 20Hz record"

    # NaNs outside the model domain are replaced by zero, so all values are finite
    assert np.all(np.isfinite(cats_tide)), "cats_tide should contain no NaN or inf"

    # Ocean tides around Antarctica are a few metres at most; anything larger
    # indicates the wrong model, units, or time standard
    assert np.all(np.abs(cats_tide) < 10.0), "cats_tide values should be within +/-10 m"

    # This track crosses the CATS domain, so some points must have a non-zero tide
    assert np.any(cats_tide != 0.0), "some records should have a computed tide value"


def test_alg_cats2023_tide_correction_skips() -> None:
    """the algorithm must not compute a tide outside its domain of applicability:
    northern hemisphere, or non-SIN mode. In both cases it must return success
    (the file is not skipped) with cats_tide_required False.
    """

    base_dir = os.environ["CLEV2ER_BASE_DIR"]
    config, _, _, _, _ = load_config_files("cryotempo", baseline="F", version=10)
    config["chain"]["use_multi_processing"] = False

    thisalg = Algorithm(config, log)

    l1b_file = (
        f"{base_dir}/testdata/cs2/l1bfiles/"
        "CS_OFFL_SIR_SIN_1B_20190504T122546_20190504T122726_D001.nc"
    )
    l1b = Dataset(l1b_file)

    # northern hemisphere -> no CATS tide
    shared_dict = _shared_dict_for(l1b, l1b_file, "north", "SIN")
    success, _ = thisalg.process(l1b, shared_dict)
    assert success, "should return success (file not skipped) for northern hemisphere"
    assert not shared_dict["cats_tide_required"], "no CATS tide in northern hemisphere"
    assert "cats_tide" not in shared_dict, "cats_tide should not be set"

    # LRM mode -> no CATS tide
    shared_dict = _shared_dict_for(l1b, l1b_file, "south", "LRM")
    success, _ = thisalg.process(l1b, shared_dict)
    assert success, "should return success (file not skipped) for LRM mode"
    assert not shared_dict["cats_tide_required"], "no CATS tide in LRM mode"
    assert "cats_tide" not in shared_dict, "cats_tide should not be set"
