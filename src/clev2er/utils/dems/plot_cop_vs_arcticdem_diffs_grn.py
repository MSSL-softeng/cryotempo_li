# mypy: ignore-errors
"""plot differences between ArcticDEM and COP-90 DEM over Greenland"""

# pylint: skip-file        # Pylint: ignore this entire file
# mypy: ignore-errors      # mypy: suppress all type errors in this file
# ruff: noqa               # Ruff: ignore all lint rules in this file

import netCDF4 as nc
import numpy as np
from pygeodesy.geoids import GeoidPGM
from read_cop_dem_greenland import extract_elevations_nn_and_bilinear_minmax

from clev2er.utils.dems.dems import Dem


def get_geoid_undulation(lat: float, lon: float) -> float:
    """
    Return the geoid undulation (in meters) at the given latitude and longitude
    based on an EGM96 geoid model.
    """
    # The .height() method should return the undulation (N), which is positive if the geoid is above the ellipsoid.
    return geoid.height(lat, lon)


def greenland_meshgrid(degree_spacing: float):
    """
    Generates a rectangular mesh grid over Greenland and returns flattened
    numpy arrays of latitude and longitude values.

    Parameters:
        degree_spacing (float): The spacing in degrees between grid points
                                (e.g., 1.0 or 0.1).

    Returns:
        tuple: A tuple containing two numpy arrays:
            - lat_flat: flattened array of latitude values.
            - lon_flat: flattened array of longitude values.
    """
    # Define the approximate bounding box for Greenland
    min_lat, max_lat = 59.0, 84.0  # latitudes from 60°N to 83°N
    min_lon, max_lon = -74.0, -12.0  # longitudes from 74°W to 12°W

    # Create arrays for latitude and longitude using the desired spacing
    latitudes = np.arange(min_lat, max_lat + degree_spacing, degree_spacing)
    longitudes = np.arange(min_lon, max_lon + degree_spacing, degree_spacing)

    # Generate a 2D mesh grid
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    # Flatten the mesh grid arrays before returning
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()

    return lat_flat, lon_flat


spacing = 0.07  # spacing in degrees; try 0.1 for a finer grid
latitudes, longitudes = greenland_meshgrid(spacing)

geoid = GeoidPGM("testdata/geoids/EGM2008_5.pgm")

geoid_corr = get_geoid_undulation(latitudes, longitudes)

# Directory containing your downloaded DEM tiles.
dem_directory = "/cpdata/SATS/RA/DEMS/COP90"
# For 90 m DEMs, use dem_product "30" (resulting in filenames like "Copernicus_DSM_30_...").
dem_product = "30"

# Obtain all 4 sets of elevation values.
nn_elev, bil_elev, min_elev, max_elev = extract_elevations_nn_and_bilinear_minmax(
    latitudes, longitudes, dem_directory, dem_product
)

thisdem = Dem("arcticdem_100m_greenland_v4.1_zarr")
arctic_dem_elevs = thisdem.interp_dem(latitudes, longitudes, method="linear", xy_is_latlon=True)

bias = 0.0
bil_elev += np.array(geoid_corr) + bias
elevation_diffs = arctic_dem_elevs - bil_elev

n_points = latitudes.shape[0]

# Create and open a new netCDF file for writing
output_filename = "/tmp/greenland_dem_diffs_linear.nc"
ds = nc.Dataset(output_filename, "w", format="NETCDF4")

# Create a single dimension 'point' to store each linear array
ds.createDimension("point", n_points)

# Create variables for latitude, longitude, and elevation difference.
lat_var = ds.createVariable("latitude", np.float32, ("point",))
lon_var = ds.createVariable("longitude", np.float32, ("point",))
elev_diff_var = ds.createVariable("elevation_diff", np.float32, ("point",))

# Set attributes for the variables
lat_var.units = "degrees_north"
lon_var.units = "degrees_east"
elev_diff_var.units = "meters"

# Write the data to the variables
lat_var[:] = latitudes
lon_var[:] = longitudes
elev_diff_var[:] = elevation_diffs

# Optionally, add some global attributes
ds.description = "Linear arrays of latitudes, longitudes, and elevation differences between ArcticDEM and COP-90 DEM over Greenland"
ds.source = "Computed using clev2er and custom netCDF writer example"

# Close the netCDF file to write data to disk
ds.close()
print(f"NetCDF file written to {output_filename}")
