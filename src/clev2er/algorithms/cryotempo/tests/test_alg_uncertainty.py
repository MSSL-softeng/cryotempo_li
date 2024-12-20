"""pytest of algorithm
   clev2er.algorithms.cryotempo.alg_uncertainty.py
"""

import logging
import os
from typing import Any, Dict

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint: disable=E0611

from clev2er.algorithms.cryotempo.alg_backscatter import Algorithm as Backscatter
from clev2er.algorithms.cryotempo.alg_basin_ids import Algorithm as BasinIds
from clev2er.algorithms.cryotempo.alg_cats2008a_tide_correction import (
    Algorithm as Cats2008a,
)
from clev2er.algorithms.cryotempo.alg_dilated_coastal_mask import (
    Algorithm as CoastalMask,
)
from clev2er.algorithms.cryotempo.alg_fes2014b_tide_correction import (
    Algorithm as Fes2014b,
)
from clev2er.algorithms.cryotempo.alg_filter_height import Algorithm as FilterHeight
from clev2er.algorithms.cryotempo.alg_geo_corrections import Algorithm as GeoCorrections
from clev2er.algorithms.cryotempo.alg_geolocate_lrm import Algorithm as Geolocate_Lrm
from clev2er.algorithms.cryotempo.alg_geolocate_sin import Algorithm as Geolocate_Sin
from clev2er.algorithms.cryotempo.alg_identify_file import Algorithm as IdentifyFile
from clev2er.algorithms.cryotempo.alg_ref_dem import Algorithm as RefDem
from clev2er.algorithms.cryotempo.alg_retrack import Algorithm as Retracker
from clev2er.algorithms.cryotempo.alg_skip_on_area_bounds import Algorithm as SkipArea
from clev2er.algorithms.cryotempo.alg_skip_on_mode import Algorithm as SkipMode
from clev2er.algorithms.cryotempo.alg_surface_type import Algorithm as SurfaceType
from clev2er.algorithms.cryotempo.alg_uncertainty import Algorithm
from clev2er.algorithms.cryotempo.alg_waveform_quality import (
    Algorithm as WaveformQuality,
)
from clev2er.utils.config.load_config_settings import load_config_files

# Similar lines in 2 files, pylint: disable=R0801
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "l1b_file",
    [
        ("CS_OFFL_SIR_SIN_1B_20190504T122546_20190504T122726_D001.nc"),  # SIN, over AIS
        ("CS_OFFL_SIR_SIN_1B_20190511T005631_20190511T005758_D001.nc"),  # SIN, over GIS
        ("CS_OFFL_SIR_LRM_1B_20200911T023800_20200911T024631_D001.nc"),  # LRM, over AIS
        ("CS_LTA__SIR_LRM_1B_20200930T235609_20200930T235758_E001.nc"),  # LRM, over GRN
    ],
)
def test_alg_uncertainty(l1b_file) -> None:
    """test of clev2er.algorithms.cryotempo.alg_uncertainty.py"""

    base_dir = os.environ["CLEV2ER_BASE_DIR"]
    assert base_dir is not None

    # Load merged config file for chain
    config, _, _, _, _ = load_config_files("cryotempo")

    # Set to Sequential Processing
    config["chain"]["use_multi_processing"] = False

    # Initialise any other Algorithms required by test

    try:
        identify_file = IdentifyFile(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize IdentifyFile algorithm {exc}"

    try:
        surface_type = SurfaceType(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize SurfaceType algorithm {exc}"

    try:
        skip_mode = SkipMode(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize SkipMode algorithm {exc}"

    try:
        fes2014b = Fes2014b(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize Fes2014b algorithm {exc}"

    try:
        cats2008a = Cats2008a(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize Cats2008a algorithm {exc}"

    try:
        geo_corrections = GeoCorrections(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize GeoCorrections algorithm {exc}"

    try:
        skip_area = SkipArea(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize SkipArea algorithm {exc}"

    try:
        coastal_mask = CoastalMask(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize CoastalMask algorithm {exc}"

    try:
        waveform_quality = WaveformQuality(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize WaveformQuality algorithm {exc}"

    try:
        retracker = Retracker(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize algorithm {exc}"

    try:
        backscatter = Backscatter(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize algorithm {exc}"

    try:
        geolocate_lrm = Geolocate_Lrm(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize algorithm {exc}"

    try:
        geolocate_sin = Geolocate_Sin(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize algorithm {exc}"

    try:
        filter_height = FilterHeight(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize FilterHeight algorithm {exc}"

    try:
        basin_ids = BasinIds(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize BasinIds algorithm {exc}"

    try:
        ref_dem = RefDem(config, log)
    except KeyError as exc:
        assert False, f"Could not initialize RefDem algorithm {exc}"

    # Initialise the Algorithm being tested
    try:
        thisalg = Algorithm(config, log)  # no config used for this alg
    except (KeyError, FileNotFoundError) as exc:
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
    shared_dict: Dict[str, Any] = {}

    # setup dummy shared_dict results from other algorithms

    shared_dict["l1b_file_name"] = l1b_file

    # mock the final lat/lon from nadir

    # Run other alg process required by test to fill in
    # required shared_dict parameters

    success, _ = identify_file.process(l1b, shared_dict)
    assert success, "identify_file algorithm should not fail"

    success, _ = skip_mode.process(l1b, shared_dict)
    assert success, "skip_mode algorithm should not fail"

    success, _ = skip_area.process(l1b, shared_dict)
    assert success, "skip_area algorithm should not fail"

    success, _ = surface_type.process(l1b, shared_dict)
    assert success, "surface_type algorithm should not fail"

    success, _ = coastal_mask.process(l1b, shared_dict)
    assert success, "coastal_mask algorithm should not fail"

    success, _ = cats2008a.process(l1b, shared_dict)
    assert success, "cats2008a algorithm should not fail"
    success, _ = fes2014b.process(l1b, shared_dict)
    assert success, "fes2014b algorithm should not fail"

    success, _ = geo_corrections.process(l1b, shared_dict)
    assert success, "geo_corrections algorithm should not fail"

    success, _ = waveform_quality.process(l1b, shared_dict)
    assert success, "waveform quality algorithm should not fail"

    success, _ = retracker.process(l1b, shared_dict)
    assert success, "retracker algorithm should not fail"

    success, _ = backscatter.process(l1b, shared_dict)
    assert success, "backscatter algorithm should not fail"

    success, _ = geolocate_lrm.process(l1b, shared_dict)
    assert success, "geolocate_lrm algorithm should not fail"

    success, _ = geolocate_sin.process(l1b, shared_dict)
    assert success, "geolocate_sin algorithm should not fail"

    success, _ = basin_ids.process(l1b, shared_dict)
    assert success, "basin_ids algorithm should not fail"

    success, _ = ref_dem.process(l1b, shared_dict)
    assert success, "ref_dem algorithm should not fail"

    success, _ = filter_height.process(l1b, shared_dict)
    assert success, "filter_height algorithm should not fail"

    # Run the alg process
    success, _ = thisalg.process(l1b, shared_dict)
    assert success, "algorithm should not fail"

    # Test outputs from algorithm

    assert "uncertainty" in shared_dict, "uncertainty not in shared_dict"

    min_uncertainty = np.nanmin(shared_dict["uncertainty"])
    max_uncertainty = np.nanmax(shared_dict["uncertainty"])

    assert 0.0 < min_uncertainty < 0.3
    assert max_uncertainty < 2.5

    log.info("min_uncertainty %f", min_uncertainty)
    log.info("max_uncertainty %f", max_uncertainty)

    num_invalid = np.count_nonzero(np.isnan(shared_dict["uncertainty"]))
    num_valid = np.count_nonzero(~np.isnan(shared_dict["uncertainty"]))
    log.info("num_valid %d", num_valid)
    log.info("num_invalid %d", num_invalid)
