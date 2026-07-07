"""pytest for clev2er.utils.uncertainty.luts (MultiDimUncLut)"""

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from clev2er.utils.uncertainty.luts import MultiDimUncLut


def write_test_lut(path, axes, table):
    """Write a minimal N-D uncertainty LUT NetCDF in the format MultiDimUncLut reads.

    Args:
        path: output file path.
        axes: ordered mapping covariate name -> 1-D sample points.
        table: N-D uncertainty array matching the axis sizes.
    """
    covariates = list(axes)
    with Dataset(path, "w", format="NETCDF4") as nc:
        for cov, points in axes.items():
            nc.createDimension(f"n_{cov}", points.size)
            var = nc.createVariable(cov, "f8", (f"n_{cov}",))
            var[:] = points
        uvar = nc.createVariable(
            "uncertainty",
            "f4",
            tuple(f"n_{c}" for c in covariates),
            fill_value=np.float32(np.nan),
        )
        uvar[:] = table.astype(np.float32)
        nc.covariates = ",".join(covariates)


@pytest.fixture(name="lut_path")
def lut_path_fixture(tmp_path):
    """A 3-D slope x power x coherence LUT with value = slope + power."""
    slope = np.array([0.0, 1.0, 2.0])
    power = np.array([10.0, 20.0])
    coherence = np.array([0.5, 0.7, 0.9, 1.0])
    table = slope[:, None, None] + power[None, :, None] + 0.0 * coherence[None, None, :]
    path = tmp_path / "tst_uncertainty_3d_sin_v01.nc"
    write_test_lut(path, {"slope": slope, "power": power, "coherence": coherence}, table)
    return path


def test_full_covariates_interpolation(lut_path):
    """values at and between sample points; clipping outside the range"""
    lut = MultiDimUncLut(str(lut_path), covariates=["slope", "power", "coherence"])
    assert lut.active_covariates == ["slope", "power", "coherence"]

    unc = lut.get_uncertainty(
        slope=np.array([0.0, 0.5, 2.0, 99.0]),
        power=np.array([10.0, 15.0, 20.0, 99.0]),
        coherence=np.array([0.5, 0.8, 1.0, 1.0]),
    )
    assert np.allclose(unc, [10.0, 15.5, 22.0, 22.0])  # last value clipped to axis maxima


def test_subset_marginalises_axes(lut_path):
    """a covariate subset nanmedian-marginalises the unused axes"""
    lut = MultiDimUncLut(str(lut_path), covariates=["slope"])
    assert lut.active_covariates == ["slope"]
    # median over power axis of (slope + power) = slope + 15
    unc = lut.get_uncertainty(slope=np.array([0.0, 1.0, 2.0]))
    assert np.allclose(unc, [15.0, 16.0, 17.0])


def test_nan_inputs_and_errors(lut_path):
    """NaN inputs give NaN; wrong covariates raise"""
    lut = MultiDimUncLut(str(lut_path), covariates=["slope", "power", "coherence"])
    unc = lut.get_uncertainty(
        slope=np.array([np.nan, 1.0]),
        power=np.array([10.0, 10.0]),
        coherence=np.array([0.9, 0.9]),
    )
    assert np.isnan(unc[0]) and np.isclose(unc[1], 11.0)

    with pytest.raises(KeyError, match="missing"):
        lut.get_uncertainty(slope=np.array([1.0]))
    with pytest.raises(ValueError, match="not present"):
        MultiDimUncLut(str(lut_path), covariates=["slope", "roughness"])
    with pytest.raises(OSError, match="not found"):
        MultiDimUncLut("/no/such/file.nc", covariates=["slope"])
