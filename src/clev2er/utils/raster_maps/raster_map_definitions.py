"""Load slope and roughness raster maps from predefined sources by name.

Ported (trimmed) from the CLEV2ER land-ice chain's
``clev2er.utils.raster_maps.raster_map_definitions``: only the slope/roughness
map definitions needed by the CryoTEMPO elevation-uncertainty algorithm
(``alg_uncertainty``) are kept, so the LUT covariates the chain interpolates at
runtime are, by construction, the same datasets the uncertainty LUTs were
trained on.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import numpy as np
import pyproj

from clev2er.utils.raster_maps.raster_map import RasterMap


class RasterMapInfo(TypedDict):
    """Information about a raster map, used to construct RasterMap objects from names.

    Attributes:
        description (str): A human-readable description of the map.
        default_filename (str, optional): The filename of the map within its directory.
        crs (str):
            The coordinate reference system of the map, in EPSG code format (e.g., "EPSG:3031").
        version (str): The version of the map dataset.
        src_institute (str, optional): The institution that produced the map.
        src_url (str, optional): A URL where the map data can be accessed or downloaded.
        reference_year (int, optional): The reference year for the data in the map.
        dtype (type, optional):
            The data type to use when loading the map values. Defaults to np.float32.
        void_value (int or float, optional):
            The value used to represent void or no-data areas in the map. Defaults to -9999.
        nc_varname, nc_flip, nc_transpose (optional): NetCDF-source loading controls.
        xref (str, optional):
            Short human-readable provenance string for use as an L2 product cross-reference.
            If omitted, consumers should fall back to the map name.
    """

    description: str
    default_filename: NotRequired[str]
    crs: str
    version: str
    src_institute: NotRequired[str]
    src_url: NotRequired[str]
    reference_year: NotRequired[int]
    dtype: NotRequired[type]
    void_value: NotRequired[int | float]
    nc_varname: NotRequired[str]
    nc_flip: NotRequired[bool]
    nc_transpose: NotRequired[bool]
    xref: NotRequired[str]


all_roughness_maps: dict[str, RasterMapInfo] = {
    "arcticdem_cropped_100m_roughness_range_svd_9x9_zarr": {
        "description": "roughness from ArcticDEM, Greenland subarea v4.1",
        "default_filename": "arcticdem_cropped_100m_roughness_range_svd_9x9.zarr",
        "crs": "EPSG:3413",
        "version": "4.1",
        "src_institute": "Univ of Lancaster/PGC",
        "xref": (
            "roughness from ArcticDEM v4.1 (Greenland subarea) using SVD 9x9, "
            "Univ of Lancaster/PGC, arcticdem_cropped_100m_roughness_range_svd_9x9.zarr"
        ),
    },
    "rema_filled_100m_roughness_range_svd_9x9_zarr": {
        "description": "roughness from REMA v2 (filled) using SVD 9x9",
        "default_filename": "rema_filled_100m_roughness_range_svd_9x9.zarr",
        "crs": "EPSG:3031",
        "version": "4.1",
        "src_institute": "Lancaster/PGC",
        "xref": (
            "roughness from REMA v2 (filled) using SVD 9x9, "
            "Lancaster/PGC, rema_filled_100m_roughness_range_svd_9x9.zarr"
        ),
    },
}

all_slope_maps: dict[str, RasterMapInfo] = {
    "arcticdem_cropped_100m_slope_svd_9x9": {
        "description": "Arctic Slopes from Univ of Lancaster derived from ArcticDEM v4.1",
        "default_filename": "arcticdem_cropped_100m_slope_svd_9x9.zarr",
        "crs": "EPSG:3413",
        "version": "1.0",
        "src_institute": "Lancaster",
        "xref": (
            "slope from ArcticDEM v4.1 (Greenland subarea) using SVD 9x9, "
            "Univ of Lancaster, arcticdem_cropped_100m_slope_svd_9x9.zarr"
        ),
    },
    "rema_filled_100m_slope_svd_9x9_zarr": {
        "description": "slope from REMA v2 (filled) using SVD 9x9",
        "default_filename": "rema_filled_100m_slope_svd_9x9.zarr",
        "crs": "EPSG:3031",
        "version": "4.1",
        "src_institute": "Lancaster/PGC",
        "xref": (
            "slope from REMA v2 (filled) using SVD 9x9, "
            "Lancaster/PGC, rema_filled_100m_slope_svd_9x9.zarr"
        ),
    },
}

all_raster_maps: dict[str, RasterMapInfo] = {
    **all_roughness_maps,
    **all_slope_maps,
}


@lru_cache(maxsize=256)
def raster_map_from_name(
    name: str,
    directory: Path | None = None,
    filepath: Path | None = None,
    map_type: Literal["roughness", "slope"] | None = None,
) -> RasterMap:
    """Load a predefined raster map by name.

    Must provide either `directory` or `filepath` but not both.

    Args:
        name: The name of the raster map to load.
        directory:
            The directory where the raster map file is located.
            If provided, the default filename from the map info will be appended to this directory
            to construct the full file path.
        filepath:
            The full file path to the raster map file.
            If provided, this will be used directly instead of constructing a path from the
            directory and default filename.
        map_type:
            The type of raster map to load.
            If provided, only maps of this type will be considered when looking up the map by name.
            Must be "roughness" or "slope"; or None to consider all map types.

    """
    if (directory is None) == (filepath is None):
        raise ValueError("Must specify exactly one of `directory` or `filepath`.")
    match map_type:
        case "roughness":
            maps = all_roughness_maps
        case "slope":
            maps = all_slope_maps
        case None:
            maps = all_raster_maps
        case _:
            raise ValueError(
                f"Invalid map_type '{map_type}'. Must be 'roughness', 'slope', or None."
            )
    try:
        info = maps[name]
    except KeyError as e:
        raise ValueError(
            f"Raster map with name '{name}' not found. "
            f"Known raster map names {(f'of type {map_type}' if map_type else '')}: "
            f"{list(maps.keys())}"
        ) from e

    if directory is not None:
        default_filename = info.get("default_filename", None)
        if default_filename is None:
            raise ValueError(
                f"Raster map '{name}' does not have a default filename. "
                "Must specify `filepath` instead of `directory`."
            )
        filepath = Path(directory) / default_filename
    else:
        assert filepath is not None  # for type checker
        filepath = Path(filepath)

    projection = pyproj.CRS(info["crs"])
    dtype = info.get("dtype", np.float32)
    void_value = info.get("void_value", -9999)
    metadata = {"map_name": name} | {
        k: v for k, v in info.items() if k not in {"path", "crs", "dtype", "void_value"}
    }

    return RasterMap(
        filepath=filepath,
        projection=projection,
        dtype=dtype,
        void_value=void_value,
        metadata=metadata,
    )
