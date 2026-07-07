"""pytest of algorithm
   clev2er.algorithms.cryotempo.alg_uncertainty.py
"""

import logging
import os
from typing import Any, Dict

import numpy as np
import pyproj
import pytest
import zarr
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
from clev2er.utils.uncertainty.tests.test_luts import write_test_lut

# Similar lines in 2 files, pylint: disable=R0801
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unit test with synthetic LUTs and rastermaps (no external data needed)
# ---------------------------------------------------------------------------


def _write_zarr_pair(directory, stem, values, binsize=100.0):
    """Write <stem>.zarr (+ _flipped companion) with the extent attrs RasterMap expects.

    ``values`` row 0 is the top row of the map (maximum y); grid origin is (0, 0).
    """
    nrows, ncols = values.shape
    top_y = (nrows - 1) * binsize
    right_x = (ncols - 1) * binsize
    main = zarr.open_array(str(directory / f"{stem}.zarr"), mode="w", shape=values.shape)
    main[:] = values
    main.attrs.update(
        {
            "ncols": ncols,
            "nrows": nrows,
            "top_l": (0.0, top_y),
            "top_r": (right_x, top_y),
            "bottom_l": (0.0, 0.0),
            "binsize": binsize,
        }
    )
    flipped = zarr.open_array(str(directory / f"{stem}_flipped.zarr"), mode="w", shape=values.shape)
    flipped[:] = np.flip(values, 0)


def _make_synthetic_config(tmp_path) -> Dict[str, Any]:
    """Build a minimal chain config with synthetic LUTs and slope/roughness rastermaps.

    LUT tables are constant per (region, mode) so the expected uncertainty values are
    unambiguous: antarctica sin=1.5, lrm=0.7; greenland sin=2.5, lrm=1.2. The synthetic
    slope (0.5 deg) and roughness (0.2 m) maps are constant and within the LUT axis
    ranges. Map grids cover x,y = 0..700 m (8x8 at 100 m) at the projection origin.
    """
    lut_dir = tmp_path / "luts"
    lut_dir.mkdir()
    map_dir = tmp_path / "maps"
    map_dir.mkdir()

    axes_4d = {
        "slope": np.array([0.2, 0.6, 1.0]),
        "roughness": np.array([0.1, 0.3]),
        "power": np.array([5.0, 15.0]),
        "coherence": np.array([0.6, 0.9]),
    }
    axes_3d = {k: v for k, v in axes_4d.items() if k != "coherence"}
    shape_4d = tuple(v.size for v in axes_4d.values())
    shape_3d = tuple(v.size for v in axes_3d.values())

    lut_values = {
        ("ant", "sin"): 1.5,
        ("ant", "lrm"): 0.7,
        ("grn", "sin"): 2.5,
        ("grn", "lrm"): 1.2,
    }
    for (prefix, mode), value in lut_values.items():
        if mode == "sin":
            write_test_lut(
                lut_dir / f"{prefix}_uncertainty_4d_sin_v01.nc",
                axes_4d,
                np.full(shape_4d, value),
            )
        else:
            write_test_lut(
                lut_dir / f"{prefix}_uncertainty_3d_lrm_v01.nc",
                axes_3d,
                np.full(shape_3d, value),
            )

    slope_values = np.full((8, 8), 0.5)
    roughness_values = np.full((8, 8), 0.2)
    _write_zarr_pair(map_dir, "rema_filled_100m_slope_svd_9x9", slope_values)
    _write_zarr_pair(map_dir, "rema_filled_100m_roughness_range_svd_9x9", roughness_values)
    _write_zarr_pair(map_dir, "arcticdem_cropped_100m_slope_svd_9x9", slope_values)
    _write_zarr_pair(map_dir, "arcticdem_cropped_100m_roughness_range_svd_9x9", roughness_values)

    return {
        "chain": {"use_multi_processing": False},
        "uncertainty_tables": {
            "base_dir": str(lut_dir),
            "antarctica": {
                "sin": {
                    "filename": "ant_uncertainty_4d_sin_v01.nc",
                    "covariates": "slope,roughness,power,coherence",
                },
                "lrm": {
                    "filename": "ant_uncertainty_3d_lrm_v01.nc",
                    "covariates": "slope,roughness,power",
                },
            },
            "greenland": {
                "sin": {
                    "filename": "grn_uncertainty_4d_sin_v01.nc",
                    "covariates": "slope,roughness,power,coherence",
                },
                "lrm": {
                    "filename": "grn_uncertainty_3d_lrm_v01.nc",
                    "covariates": "slope,roughness,power",
                },
            },
        },
        "surface_slopes": {
            "antarctica": {
                "raster_map_name": "rema_filled_100m_slope_svd_9x9_zarr",
                "directory": str(map_dir),
            },
            "greenland": {
                "raster_map_name": "arcticdem_cropped_100m_slope_svd_9x9",
                "directory": str(map_dir),
            },
        },
        "surface_roughness": {
            "antarctica": {
                "raster_map_name": "rema_filled_100m_roughness_range_svd_9x9_zarr",
                "directory": str(map_dir),
            },
            "greenland": {
                "raster_map_name": "arcticdem_cropped_100m_roughness_range_svd_9x9_zarr",
                "directory": str(map_dir),
            },
        },
    }


def _latlon_on_synthetic_map(xs, ys, epsg=3031):
    """Convert map-projection x/y to (lats, lons) for the synthetic map grids."""
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(epsg), pyproj.CRS.from_epsg(4326), always_xy=True
    )
    lons, lats = transformer.transform(np.asarray(xs, float), np.asarray(ys, float))
    return np.asarray(lats), np.asarray(lons)


@pytest.fixture(name="dummy_l1b")
def dummy_l1b_fixture(tmp_path):
    """A minimal netCDF4 Dataset (process_setup only checks the type)."""
    with Dataset(tmp_path / "dummy_l1b.nc", "w") as nc:
        nc.createDimension("time", 1)
    with Dataset(tmp_path / "dummy_l1b.nc") as nc:
        yield nc


def test_alg_uncertainty_synthetic(tmp_path, dummy_l1b) -> None:
    """SIN uses the 4D LUT, LRM the 3D LUT; NaN covariates/off-map points give NaN"""
    config = _make_synthetic_config(tmp_path)
    thisalg = Algorithm(config, log)

    # two on-map points + one off-map point (x=5000 is outside the 0..700 m grid)
    lats, lons = _latlon_on_synthetic_map([350.0, 150.0, 5000.0], [350.0, 250.0, 5000.0])

    shared_dict: Dict[str, Any] = {
        "hemisphere": "south",
        "instr_mode": "SIN",
        "latitudes": lats,
        "longitudes": lons,
        "sig0_20_ku": np.array([10.0, np.nan, 10.0]),
        "coherence_at_rtrk_point": np.array([0.7, 0.8, 0.7]),
    }
    success, _ = thisalg.process(dummy_l1b, shared_dict)
    assert success
    unc = shared_dict["uncertainty"]
    assert np.isclose(unc[0], 1.5)  # antarctica SIN LUT value
    assert np.isnan(unc[1])  # NaN sig0 -> NaN uncertainty
    assert np.isnan(unc[2])  # off-map -> NaN slope/roughness -> NaN uncertainty

    # LRM mode needs no coherence and uses the 3D LUT
    shared_dict_lrm: Dict[str, Any] = {
        "hemisphere": "south",
        "instr_mode": "LRM",
        "latitudes": lats[:2],
        "longitudes": lons[:2],
        "sig0_20_ku": np.array([10.0, 10.0]),
    }
    success, _ = thisalg.process(dummy_l1b, shared_dict_lrm)
    assert success
    assert np.allclose(shared_dict_lrm["uncertainty"], 0.7)

    # unsupported mode fails with a clear error
    shared_dict_sar = dict(shared_dict_lrm, instr_mode="SAR")
    success, error_str = thisalg.process(dummy_l1b, shared_dict_sar)
    assert not success and "SAR" in error_str


def test_alg_uncertainty_synthetic_greenland(tmp_path, dummy_l1b) -> None:
    """northern hemisphere selects the greenland LUT set"""
    config = _make_synthetic_config(tmp_path)
    thisalg = Algorithm(config, log)

    lats, lons = _latlon_on_synthetic_map([350.0], [350.0], epsg=3413)
    shared_dict: Dict[str, Any] = {
        "hemisphere": "north",
        "instr_mode": "LRM",
        "latitudes": lats,
        "longitudes": lons,
        "sig0_20_ku": np.array([10.0]),
    }
    success, _ = thisalg.process(dummy_l1b, shared_dict)
    assert success
    assert np.isclose(shared_dict["uncertainty"][0], 1.2)


def test_alg_uncertainty_grn_only(tmp_path, dummy_l1b) -> None:
    """grn_only runs skip Antarctic resources; southern files get uncertainty None"""
    config = _make_synthetic_config(tmp_path)
    config["grn_only"] = True
    thisalg = Algorithm(config, log)
    assert "antarctica" not in thisalg.uncertainty_luts

    shared_dict: Dict[str, Any] = {"hemisphere": "south", "instr_mode": "SIN"}
    success, _ = thisalg.process(dummy_l1b, shared_dict)
    assert success
    assert shared_dict["uncertainty"] is None


# ---------------------------------------------------------------------------
# Integration test with real L1b files and the chain config (requires the
# generated LUTs, slope/roughness rastermaps and CryoSat test data)
# ---------------------------------------------------------------------------


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

    # Skip (rather than fail) while the generated 3D/4D LUT files are not yet in
    # place: they are produced from CryoTEMPO-minus-IS2 differences by
    # create_cryotempo_luts_from_is2_diffs in the CLEV2ER repository.
    first_lut = (
        f"{config['uncertainty_tables']['base_dir']}/"
        f"{config['uncertainty_tables']['antarctica']['sin']['filename']}"
    )
    if not os.path.isfile(first_lut):
        pytest.skip(f"3D/4D uncertainty LUTs not found ({first_lut})")

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
        thisalg = Algorithm(config, log)
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

    # LUT values are median |dh| screened at |dh| <= 15 m, so all finite
    # uncertainties must lie in (0, 15)
    assert 0.0 < min_uncertainty <= max_uncertainty < 15.0

    log.info("min_uncertainty %f", min_uncertainty)
    log.info("max_uncertainty %f", max_uncertainty)

    num_invalid = np.count_nonzero(np.isnan(shared_dict["uncertainty"]))
    num_valid = np.count_nonzero(~np.isnan(shared_dict["uncertainty"]))
    log.info("num_valid %d", num_valid)
    log.info("num_invalid %d", num_invalid)
