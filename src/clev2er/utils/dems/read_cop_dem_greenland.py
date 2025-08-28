# mypy: ignore-errors

import logging
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import map_coordinates

# pylint: skip-file        # Pylint: ignore this entire file
# ruff: noqa               # Ruff: ignore all lint rules in this file


# Optional logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")
logger = logging.getLogger("DEM_Elevation_Extractor")


def average_pixel_spacing(dem_tile_path: str) -> float:
    """
    Calculate the average ground distance in meters between adjacent pixel centers
    in a DEM tile.
    """
    with rasterio.open(dem_tile_path) as src:
        transform = src.transform
        center0 = transform * (0.5, 0.5)
        center_right = transform * (1.5, 0.5)
        center_down = transform * (0.5, 1.5)
    # Here we simply compute Euclidean distances in degrees (approximation) for demonstration.
    spacing_h = np.linalg.norm(np.subtract(center_right, center0))
    spacing_v = np.linalg.norm(np.subtract(center_down, center0))
    avg_spacing = (spacing_h + spacing_v) / 2.0
    return avg_spacing


def compute_tile_filename(lat: float, lon: float, dem_product: str = "30") -> str:
    """
    Given a latitude and longitude, compute the DEM tile file name.

    For DEM files named as:
      Copernicus_DSM_{product}_N{lat_base:02d}_00_W{lon_base:03d}_00_DEM.tif
    we assume:
      - The tile covers 1° in latitude. For a coordinate lat, the tile's latitude part is floor(lat)
        (e.g., for 75.5, floor(75.5)=75 gives "N75_00").
      - For longitudes in the Western Hemisphere (negative values), the tile is computed as:
            tile_lon = floor(abs(lon)) + 1.
        For example, for lon = –57.3, floor(57.3)=57 and tile_lon = 57+1 = 58 which gives "W058_00".
    """
    tile_lat = int(np.floor(lat))
    tile_lon = int(np.floor(abs(lon))) + 1
    lat_part = f"N{tile_lat:02d}_00"
    lon_part = f"W{tile_lon:03d}_00"
    filename = f"Copernicus_DSM_{dem_product}_{lat_part}_{lon_part}_DEM.tif"
    return filename


def extract_elevations_nn_and_bilinear_minmax(
    latitudes: np.ndarray, longitudes: np.ndarray, dem_directory: str, dem_product: str = "30"
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    For the input coordinate arrays (latitudes, longitudes), select the relevant DEM tile files
    based on their file names and compute:
      - Nearest neighbor (NN) elevations (using Rasterio's sample()),
      - Bilinearly interpolated elevations (using SciPy's map_coordinates),
      - The minimum and maximum values of the 4 surrounding pixels used in bilinear interpolation.

    Coordinates for which no DEM file is found will remain as NaN.

    Parameters:
      latitudes (np.ndarray): 1D array of latitudes (degrees N).
      longitudes (np.ndarray): 1D array of longitudes (degrees, negative for West).
      dem_directory (str): Directory containing DEM tile GeoTIFF files.
      dem_product (str): The string code in the DEM filenames (e.g., "30" for 90 m DEMs).

    Returns:
      tuple: Four 1D numpy arrays:
            (nearest_neighbor_elevations, bilinear_elevations,
             min_neighbor_elevations, max_neighbor_elevations)
    """
    if latitudes.shape != longitudes.shape:
        raise ValueError("latitudes and longitudes must have the same shape.")

    n_points = len(latitudes)
    elev_nn = np.full(n_points, np.nan, dtype=float)
    elev_bilinear = np.full(n_points, np.nan, dtype=float)
    elev_min = np.full(n_points, np.nan, dtype=float)
    elev_max = np.full(n_points, np.nan, dtype=float)

    # Group points by DEM tile filename.
    tile_to_points = {}
    for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
        tile_filename = compute_tile_filename(lat, lon, dem_product)
        tile_to_points.setdefault(tile_filename, []).append((i, (lon, lat)))

    dem_dir = Path(dem_directory)
    logger.info("Processing %d points across %d unique tiles...", n_points, len(tile_to_points))

    # Process each DEM tile.
    for tile_filename, items in tile_to_points.items():
        tile_path = dem_dir / tile_filename
        if not tile_path.exists():
            # logger.warning(
            #     "Tile file %s not found; %d point(s) will remain NaN.", tile_filename, len(items)
            # )
            continue

        indices = [item[0] for item in items]
        coords = [item[1] for item in items]  # (lon, lat) tuples

        try:
            with rasterio.open(tile_path) as src:
                # --- Nearest Neighbor Sampling ---
                nn_samples = list(src.sample(coords))
                nn_values = np.array([sample[0] for sample in nn_samples], dtype=float)

                # --- Bilinear Interpolation ---
                dem_array = src.read(1).astype(np.float64)
                inv_transform = ~src.transform
                frac_coords = np.array([inv_transform * (lon, lat) for lon, lat in coords])
                cols = frac_coords[:, 0]
                rows = frac_coords[:, 1]
                bilinear_values = map_coordinates(dem_array, [rows, cols], order=1, mode="nearest")

                # --- Compute Surrounding Neighbors' Min and Max ---
                n_rows, n_cols = dem_array.shape
                min_vals = []
                max_vals = []
                for r, c in zip(rows, cols):
                    # Identify the indices of the 4 neighbor pixels.
                    r0 = int(np.floor(r))
                    r1 = int(np.ceil(r))
                    c0 = int(np.floor(c))
                    c1 = int(np.ceil(c))
                    # Ensure indices are within bounds.
                    r0 = np.clip(r0, 0, n_rows - 1)
                    r1 = np.clip(r1, 0, n_rows - 1)
                    c0 = np.clip(c0, 0, n_cols - 1)
                    c1 = np.clip(c1, 0, n_cols - 1)
                    neighbors = [
                        dem_array[r0, c0],
                        dem_array[r0, c1],
                        dem_array[r1, c0],
                        dem_array[r1, c1],
                    ]
                    min_vals.append(np.min(neighbors))
                    max_vals.append(np.max(neighbors))
                min_vals = np.array(min_vals, dtype=float)
                max_vals = np.array(max_vals, dtype=float)

                # Update the global arrays.
                for idx, nn_val, bil_val, mn_val, mx_val in zip(
                    indices, nn_values, bilinear_values, min_vals, max_vals
                ):
                    elev_nn[idx] = nn_val
                    elev_bilinear[idx] = bil_val
                    elev_min[idx] = mn_val
                    elev_max[idx] = mx_val

            # logger.info("Processed tile %s for %d point(s).", tile_filename, len(coords))
        except Exception as e:
            logger.error("Error processing tile %s: %s", tile_filename, e)

    return elev_nn, elev_bilinear, elev_min, elev_max


# Example usage:
if __name__ == "__main__":
    # Example input coordinates (adjust as needed).
    latitudes = np.array([75.5, 75.6, 75.7, 61.3, 79.3280254299693])
    longitudes = np.array([-57.5, -57.4, -57.3, -68.9, -34.42389])

    # Directory containing your downloaded DEM tiles.
    dem_directory = "/cpdata/SATS/RA/DEMS/COP90"
    # For 90 m DEMs, use dem_product "30" (resulting in filenames like "Copernicus_DSM_30_...").
    dem_product = "30"

    # Obtain all 4 sets of elevation values.
    nn_elev, bil_elev, min_elev, max_elev = extract_elevations_nn_and_bilinear_minmax(
        latitudes, longitudes, dem_directory, dem_product
    )

    # Print out the results.
    for lat, lon, nn_val, bil_val, mn_val, mx_val in zip(
        latitudes, longitudes, nn_elev, bil_elev, min_elev, max_elev
    ):
        print(
            f"({lat:.4f}, {lon:.4f}) -> NN Elevation: {nn_val}, Bilinear Elevation: {bil_val}, "
            f"Min Surrounding: {mn_val}, Max Surrounding: {mx_val}"
        )
