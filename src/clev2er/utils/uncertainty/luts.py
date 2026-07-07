"""clev2er.utils.uncertainty.luts

Multi-dimensional elevation-uncertainty LUT reader, ported from the CLEV2ER
land-ice chain (``clev2er.utils.uncertainty.luts.MultiDimUncLut``) for the
CryoTEMPO 3D/4D uncertainty algorithm. Reads the LUT NetCDF files produced by
``clev2er.utils.uncertainty.create_cryotempo_luts_from_is2_diffs`` (in the
CLEV2ER repository) from CryoTEMPO-minus-ICESat-2 elevation differences.
"""

import logging
import os
import warnings
from typing import Sequence

import numpy as np
from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

log = logging.getLogger(__name__)


class MultiDimUncLut:  # pylint: disable=too-few-public-methods
    """Multi-dimensional elevation uncertainty LUT.

    Loads an N-D uncertainty LUT NetCDF. The file declares its full
    covariate set (e.g. ``slope,roughness,power,coherence``) via the
    ``covariates`` global attribute and stores one sample-points variable per
    covariate (named after the covariate) plus an N-D ``uncertainty`` array.

    At load time the caller passes the subset of covariates to use. Any LUT
    axis whose covariate is *not* in that subset is marginalised away by
    taking the ``nanmedian`` along that axis. The remaining axes (preserved
    in their original file/axis order) are wired into a
    :class:`scipy.interpolate.RegularGridInterpolator` for linear N-D
    interpolation.

    Out-of-range covariate values are clipped to the LUT's per-axis sample
    range (matching the legacy 1-D LUT behaviour). NaN inputs and
    LUT cells containing NaN both produce NaN outputs.
    """

    def __init__(
        self,
        path: str,
        covariates: Sequence[str],
        thislog: logging.Logger | None = None,
    ):
        self.log = thislog if thislog is not None else log
        if not os.path.isfile(path):
            self.log.error("uncertainty LUT %s not found", path)
            raise OSError(f"uncertainty LUT not found: {path}")

        if not covariates:
            raise ValueError(f"empty covariates list for LUT {path}")

        with Dataset(path) as nc:
            file_covariates = [c.strip() for c in str(nc.covariates).split(",") if c.strip()]
            file_coords = {
                name: np.asarray(nc[name][:], dtype=np.float64) for name in file_covariates
            }
            # `_FillValue` is NaN in the LUT writer; `.filled(np.nan)` makes
            # masked cells explicit NaN floats for the interpolator to handle.
            table = nc["uncertainty"][:].filled(np.nan).astype(np.float64)

        unknown = [c for c in covariates if c not in file_covariates]
        if unknown:
            raise ValueError(
                f"configured covariates {unknown} not present in LUT {path} "
                f"(available: {file_covariates})"
            )

        # Marginalise inactive axes via nanmedian. Process from the highest
        # axis index downwards so the remaining axis indices stay valid.
        inactive_axes = [i for i, c in enumerate(file_covariates) if c not in covariates]
        with warnings.catch_warnings():
            # nanmedian over an all-NaN slice emits a RuntimeWarning; the
            # resulting NaN is the desired passthrough behaviour.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for axis in sorted(inactive_axes, reverse=True):
                table = np.nanmedian(table, axis=axis)

        self.active_covariates: list[str] = [c for c in file_covariates if c in covariates]
        self.coords: list[NDArray[np.float64]] = [file_coords[c] for c in self.active_covariates]
        self.table = table

        self.interpolator = RegularGridInterpolator(
            self.coords,
            self.table,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        self._axis_min = np.array([c[0] for c in self.coords], dtype=np.float64)
        self._axis_max = np.array([c[-1] for c in self.coords], dtype=np.float64)

        self.log.debug(
            "loaded LUT %s: file_covariates=%s, active=%s, table shape=%s",
            path,
            file_covariates,
            self.active_covariates,
            self.table.shape,
        )

    def get_uncertainty(self, **covariate_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate uncertainty at the supplied covariate sample points.

        All active covariates (``self.active_covariates``) must be supplied as
        equal-length 1-D arrays. Values outside the LUT range on any axis are
        clipped to the nearest sample point on that axis. NaN inputs (on any
        axis) and LUT cells containing NaN both yield NaN outputs.

        Args:
            **covariate_values: mapping covariate name -> 1-D ndarray of
                sample values. Must contain exactly the covariates returned by
                ``self.active_covariates``.

        Returns:
            1-D ``ndarray[float64]`` of uncertainties.
        """
        missing = [c for c in self.active_covariates if c not in covariate_values]
        if missing:
            raise KeyError(f"missing covariates {missing}; required: {self.active_covariates}")
        extras = [c for c in covariate_values if c not in self.active_covariates]
        if extras:
            raise KeyError(f"unexpected covariates {extras}; allowed: {self.active_covariates}")

        n = -1
        for c in self.active_covariates:
            arr = np.asarray(covariate_values[c], dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(f"covariate '{c}' must be 1-D, got ndim={arr.ndim}")
            if n < 0:
                n = arr.size
            elif arr.size != n:
                raise ValueError(f"covariate '{c}' length {arr.size} != {n} (from first covariate)")

        if n == 0:
            return np.empty(0, dtype=np.float64)

        points = np.empty((n, len(self.active_covariates)), dtype=np.float64)
        for i, c in enumerate(self.active_covariates):
            v = np.asarray(covariate_values[c], dtype=np.float64)
            # Clip to [axis_min, axis_max] preserving NaN positions.
            clipped = np.where(np.isnan(v), v, np.clip(v, self._axis_min[i], self._axis_max[i]))
            points[:, i] = clipped

        nan_rows = np.any(np.isnan(points), axis=1)
        result = np.full(n, np.nan, dtype=np.float64)
        if not np.all(nan_rows):
            result[~nan_rows] = self.interpolator(points[~nan_rows])
        return result
