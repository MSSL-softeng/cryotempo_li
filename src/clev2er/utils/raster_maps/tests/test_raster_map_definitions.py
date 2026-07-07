"""Tests for clev2er.utils.raster_maps.raster_map_definitions."""

import numpy as np
import pytest
import zarr

from clev2er.utils.raster_maps.raster_map_definitions import (
    all_raster_maps,
    raster_map_from_name,
)

# pylint: disable=missing-function-docstring


def _write_zarr_pair(directory, stem, values, binsize=100.0):
    """Write <stem>.zarr (+ _flipped companion) with the extent attrs RasterMap expects.

    ``values`` row 0 is the top row of the map (maximum y).
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


def test_raster_map_from_name_zarr(tmp_path):
    """a named slope map loads from a directory and interpolates zarr values"""
    # values[j, i] = 10*j + i, row 0 at the top of an 8x8, 100 m grid
    values = (10.0 * np.arange(8)[:, None] + np.arange(8)[None, :]).astype(np.float32)
    _write_zarr_pair(tmp_path, "rema_filled_100m_slope_svd_9x9", values)

    rmap = raster_map_from_name(
        "rema_filled_100m_slope_svd_9x9_zarr", directory=tmp_path, map_type="slope"
    )

    assert rmap.metadata["map_name"] == "rema_filled_100m_slope_svd_9x9_zarr"
    assert "xref" in rmap.metadata
    # exact grid point: x=300 -> col 3; y=300 -> row (700-300)/100 = 4 -> 43
    out = rmap.interpolate(np.array([300.0]), np.array([300.0]), method="linear")
    assert np.isclose(out[0], 43.0)
    # off-map points return NaN
    out = rmap.interpolate(np.array([-500.0]), np.array([300.0]), method="linear")
    assert np.isnan(out[0])


def test_raster_map_from_name_errors(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        raster_map_from_name("no_such_map", directory=tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        raster_map_from_name("rema_filled_100m_slope_svd_9x9_zarr")
    with pytest.raises(ValueError, match="Invalid map_type"):
        raster_map_from_name(
            "rema_filled_100m_slope_svd_9x9_zarr", directory=tmp_path, map_type="dem"
        )


def test_uncertainty_map_definitions_present():
    """the four maps the CryoTEMPO uncertainty algorithm needs are defined"""
    for name in (
        "rema_filled_100m_slope_svd_9x9_zarr",
        "rema_filled_100m_roughness_range_svd_9x9_zarr",
        "arcticdem_cropped_100m_slope_svd_9x9",
        "arcticdem_cropped_100m_roughness_range_svd_9x9_zarr",
    ):
        assert name in all_raster_maps
        assert "default_filename" in all_raster_maps[name]
        assert "xref" in all_raster_maps[name]
