"""pytests for Dem class"""

import logging

import numpy as np
import pytest

from clev2er.utils.dems.dems import Dem

pytestmark = pytest.mark.requires_external_data

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "dem_name,lats,lons,elevs",
    [
        (
            "arcticdem_100m_greenland_v4.1_zarr",
            [79.3280254299693],
            [-34.42389],
            [1983.98],
        ),  # GIS location, elevations from CS2 CryoTEMPO Baseline-B
        ("rema_gapless_100m_zarr", [-77], [106], [3516]),  # Vostok
        ("rema_gapless_1km_zarr", [-77], [106], [3516]),  # Vostok
        ("rema_ant_1km_v2_zarr", [-77], [106], [3516]),  # Vostok
        ("arcticdem_1km_greenland_v4.1_zarr", [79.328], [-34.424], [1983.98]),
        # (
        #     "awi_grn_1km",
        #     [79.3280254299693],
        #     [-34.42389],
        #     [1985],
        # ),  # GIS location, elevations from CS2 CryoTEMPO Baseline-B
        # (
        #     "arcticdem_1km",
        #     [79.3280254299693],
        #     [-34.42389],
        #     [1983.98],
        # ),  # GIS location, elevations from CS2 CryoTEMPO Baseline-B
        # (
        #     "arcticdem_1km_v4.1",
        #     [79.3280254299693],
        #     [-34.42389],
        #     [1983.98],
        # ),  # GIS location, elevations from CS2 CryoTEMPO Baseline-B
        # (
        #     "arcticdem_100m_greenland",
        #     [79.3280254299693],
        #     [-34.42389],
        #     [1983.98],
        # ),  # GIS location, elevations from CS2 CryoTEMPO Baseline-B
        # (
        #     "arcticdem_100m_greenland_v4.1",
        #     [79.3280254299693],
        #     [-34.42389],
        #     [1983.98],
        # ),  # GIS location, elevations from CS2 CryoTEMPO Baseline-B
        # (
        #     "arcticdem_1km_greenland_v4.1",
        #     [79.3280254299693],
        #     [-34.42389],
        #     [1983.98],
        # ),  # GIS location, elevations from CS2 CryoTEMPO Baseline-B
        # ("rema_ant_1km", [-77], [106], [3516]),  # Vostok
        # ("rema_ant_1km_v2", [-77], [106], [3516]),  # Vostok
        # ("rema_ant_200m", [-77], [106], [3516]),  # Vostok
        # ("rema_gapless_100m", [-77], [106], [3516]),  # Vostok
        # ("rema_gapless_1km", [-77], [106], [3516]),  # Vostok
        # ("awi_ant_1km_floating", [-80], [182], [0]),  # Ross Ice Shelf
        # ("awi_ant_1km_grounded", [-77], [106], [3516]),  # Vostok
        # ("awi_ant_1km", [-77], [106], [3516]),  # Vostok),# Vostok
    ],
)
def test_dems(dem_name, lats, lons, elevs):
    """load DEMs and test interpolated elevations to tolerance of 1m

    Args:
        dem_name (str): _description_
        lats (np.ndarray): latitude values
        lons (np.ndarray): longitude values
        elevs (np.ndarray: expected elevation values
    """
    thisdem = Dem(dem_name)

    if len(lats) > 0:
        dem_elevs = thisdem.interp_dem(lats, lons, xy_is_latlon=True)
        np.testing.assert_allclose(elevs, dem_elevs, atol=1.0)


@pytest.mark.parametrize(
    "dem_name_zarr,lats,lons,elevs",
    [
        ("rema_gapless_1km_zarr", [-77], [106], [3516]),  # Vostok
        ("rema_gapless_100m_zarr", [-77], [106], [3516]),  # Vostok
    ],
)
def test_dems_zarr(dem_name_zarr, lats, lons, elevs):
    """load Zarr format DEMs and test interpolated elevations to tolerance of 1m

    Args:
        dem_name_zarr (str): _description_
        lats (np.ndarray): latitude values
        lons (np.ndarray): longitude values
        elevs (np.ndarray: expected elevation values
    """
    thisdem = Dem(dem_name_zarr)

    if len(lats) > 0:
        dem_elevs = thisdem.interp_dem(lats, lons, xy_is_latlon=True)
        np.testing.assert_allclose(elevs, dem_elevs, atol=1.0)


@pytest.mark.parametrize(
    "dem_name,dem_name_zarr,lats,lons",
    [
        ("rema_gapless_1km", "rema_gapless_1km_zarr", [-77], [106]),  # Vostok
    ],
)
def test_compare_dems_zarr_and_tiff(dem_name, dem_name_zarr, lats, lons):
    """Compare zarr and tiff DEMs to a tolerance of 0.001m

    Args:
        dem_name (str): name of Dem object using Tiff format
        dem_name_zarr (str): name of Dem object using Zarr format
        lats (np.ndarray): latitude values
        lons (np.ndarray): longitude values
        elevs (np.ndarray: expected elevation values
    """
    thisdem_tiff = Dem(dem_name)
    thisdem_zarr = Dem(dem_name_zarr)

    if len(lats) > 0:
        dem_elevs_tiff = thisdem_tiff.interp_dem(lats, lons, xy_is_latlon=True)
        dem_elevs_zarr = thisdem_zarr.interp_dem(lats, lons, xy_is_latlon=True)
        np.testing.assert_allclose(dem_elevs_tiff, dem_elevs_zarr, atol=0.001)
