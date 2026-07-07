"""Tests for the `RasterMap` class."""

import numpy as np
import pyproj
import pytest

from clev2er.utils.raster_maps.raster_map import RasterMap

# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


@pytest.fixture
def raster_map(tmp_path):
    xdem = np.linspace(0, 4, 5)
    # `RasterMap.ys` should be in descending order
    ydem = np.linspace(0, 3, 4)[::-1]
    # `RasterMap.values` matches, so `values[0, 0]` has coordinates `xs[0]`, `ys[0]`
    zdem = np.array(
        [
            # x=0, y=0 is bottom-left corner in this array, value 0.0
            # x increases to the right, y increases upwards
            # Max value is 19.0 at x=4, y=3 (top-right corner)
            [15, 16, 17, 18, 19],
            [10, 11, 12, 13, 14],
            [5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4],
        ]
    ).astype(float)
    out_path = tmp_path / "test_raster_map.npz"
    np.savez(
        out_path,
        xdem=xdem,
        ydem=ydem,
        zdem=zdem,
        mindemx=xdem.min(),
        mindemy=ydem.min(),
        binsize=xdem[1] - xdem[0],
    )
    return RasterMap(
        filepath=out_path,
        projection=pyproj.CRS.from_epsg(3031),
    )


@pytest.mark.parametrize(
    ("x", "y", "expected", "method"),
    [
        (0.5, 0.5, 3.0, "linear"),
        (0.4, 0.4, 0.0, "nearest"),
        (0.6, 0.6, 6.0, "nearest"),
        (1.0, 1.0, 6.0, "linear"),
        (4.0, 3.0, 19.0, "linear"),
        (4.5, 3.5, np.nan, "linear"),
    ],
)
def test_interpolate(raster_map, x, y, expected, method):
    data = raster_map.interpolate(x=[x], y=[y], method=method)
    assert len(data) == 1, "interpolation should return a single value"
    np.testing.assert_equal(
        data[0],
        expected,
        err_msg=f"interpolated value should be {expected} but is {data[0]}",
    )


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (0.0, 0.0, 15.0),
        (4.0, 3.0, 4.0),
        (3.6, 2.6, 4.0),
    ],
)
def test_grid_lookup(raster_map, x, y, expected):
    data = raster_map.grid_lookup(x=[x], y=[y])
    assert len(data) == 1, "interpolation should return a single value"
    np.testing.assert_equal(
        data[0],
        expected,
        err_msg=f"interpolated value should be {expected} but is {data[0]}",
    )
