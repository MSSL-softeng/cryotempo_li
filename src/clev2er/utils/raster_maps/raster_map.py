"""Class to handle loading and interpolation of raster maps stored on disk, such as DEMs."""

from pathlib import Path
from typing import Literal, Self

import netCDF4
import numpy as np
import pyproj
import rasterio
import rasterio.errors
import zarr
from scipy.interpolate import RegularGridInterpolator, interpn
from scipy.ndimage import gaussian_filter
from tifffile import imread  # to support large TIFF files

# Linting complains due to the many attributes needed to store raster data and coordinates.
# Better long-term solution would be something like xarray to keep the data in a structured object.
#
# pylint: disable=too-many-instance-attributes
# pylint: disable=attribute-defined-outside-init


def _get_geotiff_extent(filepath: Path) -> tuple[int, int, int, int, int, int, int]:
    """Extract extent information from a GeoTIFF file.

    Args:
        filepath (Path): path of GeoTIFF file

    Raises:
        ValueError: If the pixel width and height are not equal.
        IOError: If the GeoTIFF file cannot be read.

    Returns:
        tuple(int,int,int,int,int,int,int):
            - width
            - height
            - top_left
            - top_right
            - bottom_left
            - bottom_right
            - pixel_width
    """
    try:
        with rasterio.open(filepath) as dataset:
            transform = dataset.transform
            width = dataset.width
            height = dataset.height

            top_left = transform * (0, 0)
            top_right = transform * (width, 0)
            bottom_left = transform * (0, height)
            bottom_right = transform * (width, height)

            pixel_width = transform[0]
            pixel_height = -transform[4]  # Negative because the height is
            # typically negative in GeoTIFFs
            if int(pixel_width) != int(pixel_height):
                raise ValueError(f"pixel_width {pixel_width} != pixel_height {pixel_height}")
    except rasterio.errors.RasterioIOError as exc:
        raise IOError(f"Could not read GeoTIFF: {exc}") from exc
    return (
        width,
        height,
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        pixel_width,
    )


InterpolationMethod = Literal[
    "linear", "nearest", "slinear", "cubic", "quintic", "pchip", "splinef2d"
]


class RasterMap:
    """Provides utilities for working with spatial raster data stored on disk, such as DEMs.

    Attributes:
        filepath (Path): Path to the raster file on disk.
        projection (pyproj.CRS): Coordinate reference system of the raster data.
        fileformat (str): File format of the raster data (e.g., '.tif', '.npz', '.zarr').
        dtype (type): Data type of the raster values (default: np.float32).
        void_value (int | float): Value representing void or no-data in the raster (default: -9999).
        metadata (dict | None): Optional dictionary containing additional metadata about the raster.

        xs (np.ndarray): 1D array of x coordinates corresponding to the raster columns.
        ys (np.ndarray): 1D array of y coordinates corresponding to the raster rows.
        ys_flipped (np.ndarray): Flipped version of `ys`.
        values (np.ndarray): 2D array containing the raster values.
        values_flipped (np.ndarray): Copy of `values` with y-axis flipped to match `ys_flipped`.
        min_x (float): Minimum x coordinate of the raster extent.
        min_y (float): Minimum y coordinate of the raster extent.
        binsize (float): Grid resolution of the raster in meters.
    """

    LATLON_CRS = pyproj.CRS("EPSG:4326")

    def __init__(
        self,
        filepath: Path,
        projection: pyproj.CRS,
        fileformat: str | None = None,
        dtype: type = np.float32,
        void_value: int | float = -9999,
        metadata: dict | None = None,
    ):
        if fileformat is None:
            fileformat = filepath.suffix.lower()
        self.filepath = filepath
        self.projection = projection
        self.fileformat = fileformat
        self.dtype = dtype
        self.void_value = void_value
        self.metadata = metadata
        self._load()

        self.xy_to_lonlat_transformer = pyproj.Transformer.from_proj(
            self.projection, RasterMap.LATLON_CRS, always_xy=True
        )
        self.lonlat_to_xy_transformer = pyproj.Transformer.from_proj(
            RasterMap.LATLON_CRS, self.projection, always_xy=True
        )

    def _load(self) -> Self:
        """Load the raster values and coordinates into memory."""
        match self.fileformat:
            case ".npz":
                self._load_npz()
            case ".zarr":
                self._load_zarr()
            case ".tif" | ".tiff":
                self._load_geotiff()
            case ".nc":
                self._load_netcdf()
            case _:
                raise ValueError(f"Unsupported file format: {self.fileformat}")
        return self

    def _load_npz(self):
        data = np.load(self.filepath, allow_pickle=True)
        self.values = data["zdem"]
        self.values_flipped = np.flip(self.values, 0)
        self.xs = data["xdem"]
        self.ys = data["ydem"]
        self.ys_flipped = np.flip(self.ys)
        self.min_x = data["mindemx"]
        self.min_y = data["mindemy"]
        self.binsize = data["binsize"]

    def _load_zarr(self):
        try:
            values = zarr.open_array(self.filepath, mode="r")
        except Exception as exc:
            raise IOError(f"Failed to open Zarr file: {self.filepath} {exc}") from exc

        ncols = values.attrs["ncols"]
        nrows = values.attrs["nrows"]
        top_l = values.attrs["top_l"]
        top_r = values.attrs["top_r"]
        bottom_l = values.attrs["bottom_l"]
        binsize = values.attrs["binsize"]

        self.xs = np.linspace(top_l[0], top_r[0], ncols, endpoint=True)
        self.ys_flipped = np.linspace(bottom_l[1], top_l[1], nrows, endpoint=True)
        self.ys = np.flip(self.ys_flipped)
        self.min_x = self.xs.min()
        self.min_y = self.ys.min()
        self.binsize = binsize  # grid resolution in m
        self.values = values

        if not self.filepath.suffix.lower() == ".zarr":
            raise ValueError(
                f"Filepath `{self.filepath}` does have '.zarr' suffix. "
                "Cannot locate flipped zarr file by name."
            )
        flipped_dem = self.filepath.with_name(self.filepath.stem + "_flipped.zarr")
        try:
            self.values_flipped = zarr.open_array(flipped_dem, mode="r")
        except Exception as exc:
            raise IOError(f"Failed to open Zarr file: {flipped_dem} {exc}") from exc

    def _load_geotiff(self):
        (
            ncols,
            nrows,
            top_l,
            top_r,
            bottom_l,
            _,
            binsize,
        ) = _get_geotiff_extent(self.filepath)

        values = imread(self.filepath)
        if not isinstance(values, np.ndarray):
            raise TypeError(f"Geotiff image type not supported : {type(values)}")
        self.values = values
        self.values_flipped = np.flip(self.values, 0)

        # Set void data to Nan
        if self.void_value:
            void_data = np.where(self.values == self.void_value)
            if np.any(void_data):
                self.values[void_data] = np.nan

        self.xs = np.linspace(top_l[0], top_r[0], ncols, endpoint=True)
        self.ys_flipped = np.linspace(bottom_l[1], top_l[1], nrows, endpoint=True)
        self.ys = np.flip(self.ys_flipped)
        self.min_x = self.xs.min()
        self.min_y = self.ys.min()
        self.binsize = binsize  # grid resolution in m

    def _load_netcdf(self):
        with netCDF4.Dataset(self.filepath, mode="r") as ds:  # pylint: disable=E1101
            if len(ds.variables) == 1:
                # If there is only one variable, assume it contains the map values
                mask_var = list(ds.variables.keys())[0]
            else:
                # Otherwise, the variable name should be given, or default to "mask"
                mask_var = (self.metadata or {}).get("nc_varname", "mask")

            # Some map files have their y-axis flipped, this should be noted in the metadata
            flip = (self.metadata or {}).get("nc_flip", False)
            # Some map files have x/y transposed, this should be noted in the metadata.
            transpose = (self.metadata or {}).get("nc_transpose", False)

            self.values = ds.variables[mask_var][:].data
            if transpose:
                self.values = self.values.T
            self.values_flipped = np.flip(self.values, 0)
            if flip:
                self.values, self.values_flipped = self.values_flipped, self.values

            # The grid coordinates are either in variables, or inferred by grid origin and binsize
            self.binsize = (
                ds.__dict__.get("spacing")
                or ds.__dict__.get("binsize")
                or ds.__dict__.get("posting")
            )
            if self.binsize is None:
                # If there is no binsize attribute, coordinate variables should be available
                xs = ds.variables.get("x")
                ys = ds.variables.get("y")
                if xs is None or ys is None:
                    raise ValueError("Neither binsize nor coordinates found in netCDF file")
                self.xs = xs[:].data
                self.ys = ys[:].data
                self.ys_flipped = np.flip(self.ys)
                self.min_x = self.xs.min()
                self.min_y = self.ys.min()
                self.binsize = self.xs[1] - self.xs[0]
                return

            # Recreate the x/y coordinate arrays from binsize and min/max attributes
            xmin = ds.__dict__.get("xmin", ds.__dict__.get("min_x", ds.__dict__.get("minxm")))
            if xmin is None:
                raise ValueError("xmin or minxm attribute not found in netCDF file")
            self.xs = np.arange(xmin, xmin + self.values.shape[1] * self.binsize, self.binsize)

            ymax = ds.__dict__.get("ymax")
            ymin = ds.__dict__.get("minym", ds.__dict__.get("min_y"))
            if ymax is None and ymin is None:
                raise ValueError("ymax or minym attribute not found in netCDF file")

            if ymin is None:
                ymin = ymax - (self.values.shape[0] - 1) * self.binsize

            self.ys_flipped = np.arange(
                ymin, ymin + self.values.shape[0] * self.binsize, self.binsize
            )
            self.ys = np.flip(self.ys_flipped)

            self.min_x = self.xs.min()
            self.min_y = self.ys.min()

    def get_segment(
        self, segment_bounds: list, grid_xy: bool = True, flatten: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """return a cropped segment of the map, flattened or as a grid

        Args:
            segment_bounds (List): [(minx,maxx),(miny,maxy)]
            grid_xy (bool, optional): return segment as a grid. Defaults to True.
            flatten (bool, optional): return segment as flattened list. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (xdem,ydem,zdem)
        """
        # ----------------------------------------------------------------------
        # Get coord bounds as index bounds
        # ----------------------------------------------------------------------

        minx_ind = (np.absolute(segment_bounds[0][0] - self.xs)).argmin()
        maxx_ind = (np.absolute(segment_bounds[0][1] - self.xs)).argmin()
        miny_ind = (np.absolute(segment_bounds[1][0] - self.ys)).argmin()
        maxy_ind = (np.absolute(segment_bounds[1][1] - self.ys)).argmin()

        # ----------------------------------------------------------------------
        # Crop full dem coords to segment bounds
        # ----------------------------------------------------------------------

        values = self.values[maxy_ind:miny_ind, minx_ind:maxx_ind]
        xs = self.xs[minx_ind:maxx_ind]
        ys = self.ys[maxy_ind:miny_ind]

        if grid_xy is True:
            xs, ys = np.meshgrid(xs, ys)

            # Set x,y to nan where z is nan
            zdem_nan = np.isnan(values)
            xs[zdem_nan] = np.nan
            ys[zdem_nan] = np.nan

        # ----------------------------------------------------------------------
        # Return, flattened if requested
        # ----------------------------------------------------------------------

        if flatten is False:
            return (xs, ys, values)

        return (xs.flatten(), ys.flatten(), values.flatten())

    # Maximum bounding-box slab size (bytes) before _chunked_interpolate switches
    # from a single zarr read to tiled reads. 200 MB comfortably covers any
    # along-track strip while keeping peak memory bounded for full-grid queries.
    _MAX_SLAB_BYTES = 200 * 1024 * 1024

    def _chunked_interpolate(
        self, x: np.ndarray, y: np.ndarray, method: InterpolationMethod
    ) -> np.ndarray:
        """Interpolate the map, reading only the zarr region needed for the queries.

        For queries whose bounding box fits in ``_MAX_SLAB_BYTES`` (the normal
        along-track case), a single sub-array is read and interpolated — same code
        path and same values as before. For queries that span a large fraction of
        the grid (e.g. full-continent plots), the bounding box is split into tiles
        and each tile reads its own zarr slab with a 1-pixel halo, so memory stays
        bounded.

        Args:
            x (np.ndarray): Array of x coordinates in the map's projection (in meters).
            y (np.ndarray): Array of y coordinates in the map's projection (in meters).
            method (InterpolationMethod): Interpolation method to use.

        Returns:
            np.ndarray: Interpolated map values at the specified coordinates.
        """
        results = np.full_like(x, np.nan, dtype=np.float64)

        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        if not valid_mask.any():
            return results

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        # Bounding box of valid points in zarr index space, with ±1 pad
        xi_lo = max(int(np.searchsorted(self.xs, x_valid.min())) - 1, 0)
        xi_hi = min(int(np.searchsorted(self.xs, x_valid.max())) + 1, len(self.xs) - 1)
        yi_lo = max(int(np.searchsorted(self.ys_flipped, y_valid.min())) - 1, 0)
        yi_hi = min(
            int(np.searchsorted(self.ys_flipped, y_valid.max())) + 1,
            len(self.ys_flipped) - 1,
        )

        bytes_per_cell = self.values_flipped.dtype.itemsize
        bbox_cells = (yi_hi - yi_lo + 1) * (xi_hi - xi_lo + 1)

        if bbox_cells * bytes_per_cell <= self._MAX_SLAB_BYTES:
            # Fast path: single read covers all queries (unchanged behaviour)
            results[valid_mask] = self._interpolate_slab(
                yi_lo, yi_hi, xi_lo, xi_hi, y_valid, x_valid, method
            )
            return results

        # Tiled path: each tile reads its own zarr slab + 1-pixel halo.
        # Halo of 1 is sufficient for 'linear' and 'nearest'; widen if higher-
        # order methods are ever used.
        max_cells_per_tile = self._MAX_SLAB_BYTES // bytes_per_cell
        tile_side = max(int(np.sqrt(max_cells_per_tile)), 1)
        halo = 1

        # Assign each valid point to a tile by its nearest grid-cell index
        xi_of_pt = np.clip(np.searchsorted(self.xs, x_valid) - 1, xi_lo, xi_hi)
        yi_of_pt = np.clip(np.searchsorted(self.ys_flipped, y_valid) - 1, yi_lo, yi_hi)
        tx_of_pt = (xi_of_pt - xi_lo) // tile_side
        ty_of_pt = (yi_of_pt - yi_lo) // tile_side

        interp_values = np.full(x_valid.shape, np.nan, dtype=np.float64)
        for ty in np.unique(ty_of_pt):
            same_ty = ty_of_pt == ty
            for tx in np.unique(tx_of_pt[same_ty]):
                in_tile = same_ty & (tx_of_pt == tx)
                t_yi_lo = max(yi_lo + int(ty) * tile_side - halo, 0)
                t_yi_hi = min(
                    yi_lo + (int(ty) + 1) * tile_side - 1 + halo,
                    len(self.ys_flipped) - 1,
                )
                t_xi_lo = max(xi_lo + int(tx) * tile_side - halo, 0)
                t_xi_hi = min(
                    xi_lo + (int(tx) + 1) * tile_side - 1 + halo,
                    len(self.xs) - 1,
                )
                interp_values[in_tile] = self._interpolate_slab(
                    t_yi_lo,
                    t_yi_hi,
                    t_xi_lo,
                    t_xi_hi,
                    y_valid[in_tile],
                    x_valid[in_tile],
                    method,
                )

        results[valid_mask] = interp_values
        return results

    def _interpolate_slab(
        self,
        yi_lo: int,
        yi_hi: int,
        xi_lo: int,
        xi_hi: int,
        y_pts: np.ndarray,
        x_pts: np.ndarray,
        method: InterpolationMethod,
    ) -> np.ndarray:
        """Read one zarr slab and interpolate the given points against it."""
        sub_zarr = np.array(self.values_flipped[yi_lo : yi_hi + 1, xi_lo : xi_hi + 1])
        sub_yy = self.ys_flipped[yi_lo : yi_hi + 1]
        sub_xx = self.xs[xi_lo : xi_hi + 1]
        interpolator = RegularGridInterpolator(
            (sub_yy, sub_xx),
            sub_zarr,
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )
        return interpolator(np.vstack((y_pts, x_pts)).T)

    def grid_lookup(
        self,
        x,
        y,
        xy_is_latlon=False,
        unknown_value=np.nan,
    ) -> np.ndarray:
        """Lookup grid values at specified coordinates.

        This assumes that the raster map is a regular grid defined by xs[0], ys[-1], and binsize.
        The input coordinates are binned into the grid cells, and those grid values are returned.
        Values that fall outside the grid are assigned `unknown_value`.

        Args:
            x (np.ndarray): x cartesian coordinates in the map's projection in m, or lat values
            y (np.ndarray): y cartesian coordinates in the map's projection in m, or lon values
            xy_is_latlon (bool, optional): if True, x,y are lat, lon values. Defaults to False.
            unknown_value (int | float, optional):
                value to return for points outside the map bounds. Defaults to np.nan.

        Returns:
            np.ndarray: resulting grid values, same shape as inputs x and y
        """
        x = np.array(x)
        y = np.array(y)

        if xy_is_latlon:
            x, y = self.lonlat_to_xy_transformer.transform(y, x)

        # handle invalid coordinates (e.g. NaN) by only processing valid coordinates
        # and leaving invalid ones as unknown_value
        valid_coord_mask = np.isfinite(x) & np.isfinite(y)
        valid_coord_idxs = np.where(valid_coord_mask)[0]

        out: np.ndarray = np.full(len(x), unknown_value, dtype=self.dtype)
        i = np.around((x[valid_coord_mask] - self.xs[0]) / self.binsize).astype(int)
        j = np.around((y[valid_coord_mask] - self.ys[-1]) / self.binsize).astype(int)

        for idx, (ii, jj) in zip(valid_coord_idxs, zip(i, j)):
            if 0 <= ii < self.values.shape[1] and 0 <= jj < self.values.shape[0]:
                out[idx] = self.values[jj, ii]
        return out

    def interpolate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: InterpolationMethod = "linear",
        xy_is_latlon: bool = False,
        unknown_value: int | float = np.nan,
    ) -> np.ndarray:
        """Interpolate map to return values corresponding to
           cartesian x,y in map's projection or lat,lon values

        Args:
            x (np.ndarray): x cartesian coordinates in the map's projection in m, or lat values
            y (np.ndarray): y cartesian coordinates in the map's projection in m, or lon values
            method (InterpolationMethod, optional): Interpolation method. Defaults to "linear".
            xy_is_latlon (bool, optional): if True, x,y are lat, lon values. Defaults to False.
            unknown_value (int | float, optional):
                value to return for points outside the map bounds. Defaults to np.nan.

        Returns:
            np.ndarray: interpolated map values
        """
        x = np.array(x)
        y = np.array(y)

        if xy_is_latlon:
            x, y = self.lonlat_to_xy_transformer.transform(y, x)

        # If values is actually a zarr array instead of a numpy array
        # then we use the pre-flipped zarr version, but need to convert it to
        # a numpy array first (which is slow)
        if self.fileformat == ".zarr":
            return self._chunked_interpolate(x, y, method)

        return interpn(
            (self.ys_flipped, self.xs),
            self.values_flipped,
            (y, x),
            method=method,
            bounds_error=False,
            fill_value=unknown_value,
        )

    def gaussian_smooth(self, sigma=1.0) -> Self:
        """Perform an in-place gaussian smooth on the map.

        Uses a weighted approach to handle NaN values.

        Args:
            sigma (float, optional): degree of smoothing in standard deviations, defaults to 1.0
        """
        new_values = self.values.copy()
        nan_mask = np.isnan(self.values)
        new_values[nan_mask] = 0
        new_values = gaussian_filter(new_values, sigma=sigma)
        www = np.ones_like(self.values)
        www[nan_mask] = 0
        f_www = gaussian_filter(www, sigma=sigma)
        self.values = new_values / f_www
        return self

    def hillshade(self, azimuth=225, pitch=45) -> Self:
        """In-place conversion of an elevation map to a hillshade value between 0..255

        Args:
            azimuth (float, optional): angle in degrees (0..360). Defaults to 225.
            pitch (float, optional): angle in degrees (0..90). Defaults to 45.
        """
        azimuth = 360.0 - azimuth

        x, y = np.gradient(self.values)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)
        azimuthrad = azimuth * np.pi / 180.0
        altituderad = pitch * np.pi / 180.0

        shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope) * np.cos(
            (azimuthrad - np.pi / 2.0) - aspect
        )

        self.values = 255 * (shaded + 1) / 2
        return self
