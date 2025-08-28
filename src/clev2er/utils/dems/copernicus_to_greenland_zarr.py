# mypy: ignore-errors
#!/usr/bin/env python3
"""
copernicus_to_greenland_zarr_match_layout.py

New options to address blank margins and ensure exact alignment with your other DEMs:

- --match-zarr /path/to/reference.zarr
    Use the reference Zarr's transform + shape as the target grid (exact extents, resolution, orientation).
    Ignores --bbox/--res/--chunks unless you override them explicitly.

- --tight-to-tiles [--buffer-m 0]
    Derive the target grid bounds from the union of all input tile footprints in EPSG:3413
    (using transform_bounds with densify), then align to the requested resolution. Optional buffer in meters.

These options help remove empty borders (like the southern band you saw) or perfectly match an
existing ArcticDEM-derived grid.
"""

# pylint: skip-file        # Pylint: ignore this entire file
# mypy: ignore-errors      # mypy: suppress all type errors in this file
# ruff: noqa               # Ruff: ignore all lint rules in this file

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import rasterio
import zarr
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine, from_bounds, from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform
from zarr import Blosc

# ------------------------------ Geoid helpers ------------------------------


class GeoidBackend:
    def height(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GeographicLibGeoid(GeoidBackend):
    def __init__(self, model: str = "egm96-5"):
        try:
            from geographiclib.geoid import Geoid  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "geographiclib is required for --geoid-backend geographiclib.\n"
                "Install it (pip install geographiclib) and the EGM dataset, e.g.:\n"
                "  python -m geographiclib.geoid egm96-5"
            ) from exc
        self._geoid = Geoid(model)

    def height(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        out = np.empty(lat.shape, dtype=np.float64)
        it = np.nditer([lat, lon, out], op_flags=[["readonly"], ["readonly"], ["writeonly"]])
        for phi, lam, o in it:
            o[...] = self._geoid.Height(float(phi), float(lam))
        return out


class PyGeodesyGeoid(GeoidBackend):
    def __init__(self, pgm_path: str, kind: int = 1):
        try:
            import pygeodesy as _pg
            from pygeodesy.geoids import GeoidPGM  # type: ignore

            print(
                f"[pygeodesy] using {getattr(_pg, '__version__', 'unknown')} from {getattr(_pg, '__file__', '')}"
            )
            print(f"[env] python={sys.executable}")
        except Exception as exc:
            raise RuntimeError(
                "pygeodesy is required for --geoid-backend pygeodesy.\n"
                "pip install pygeodesy (and scipy if using kind -1 or -3)."
            ) from exc
        if not os.path.exists(pgm_path):
            raise FileNotFoundError(f"PGM not found: {pgm_path}")
        try:
            self._geoid = GeoidPGM(pgm_path, kind=kind)
        except Exception as e:
            print(
                f"[WARN] GeoidPGM(kind={kind}) failed: {e}\n       Falling back to kind=1 (bilinear).",
                file=sys.stderr,
            )
            self._geoid = GeoidPGM(pgm_path, kind=1)

    def height(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        out = np.empty(lat.shape, dtype=np.float64)
        it = np.nditer([lat, lon, out], op_flags=[["readonly"], ["readonly"], ["writeonly"]])
        for phi, lam, o in it:
            o[...] = self._geoid.height(float(phi), float(lam))
        return out


# ------------------------------ Core logic ---------------------------------


@dataclass
class BBoxLL:
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float


CRS_WKT_FOR_ATTR = 'LOCAL_CS["WGS 84 / NSIDC Sea Ice Polar Stereographic North",UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mosaic Copernicus DEM tiles to ArcticDEM-compatible Zarr over Greenland (EPSG:3413) with geoid correction (h=H+N)."
    )
    p.add_argument(
        "--input-dir", required=True, help="Directory containing Copernicus DEM GeoTIFF tiles."
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output Zarr store path (directory). Will be overwritten, and a flipped companion will be created.",
    )
    p.add_argument("--res", type=float, default=90.0, help="Target pixel size in meters (binsize).")
    p.add_argument(
        "--chunks",
        type=int,
        nargs=2,
        default=(82, 145),
        metavar=("CHUNK_ROWS", "CHUNK_COLS"),
        help="Zarr chunk sizes (rows cols). Default mirrors your ArcticDEM style.",
    )
    p.add_argument(
        "--dtype", default="<f4", choices=("<f4", "<f8"), help="Output dtype (endianness explicit)."
    )
    p.add_argument(
        "--resampling",
        default="average",
        choices=("average", "bilinear"),
        help="Resampling mode when reprojecting tiles (default: average).",
    )
    p.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing pixels when tiles overlap.",
    )
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=(-75.0, 58.0, -10.0, 85.0),
        help="WGS84 bbox for Greenland sub-area.",
    )
    # Geoid
    p.add_argument(
        "--geoid-backend",
        default="geographiclib",
        choices=("geographiclib", "pygeodesy"),
        help="EGM model provider.",
    )
    p.add_argument(
        "--geoid-model",
        default="egm96-5",
        help="For geographiclib backend (e.g., egm96-5, egm2008-5). Ignored by pygeodesy.",
    )
    p.add_argument(
        "--geoid-pgm",
        default=None,
        help="For pygeodesy backend: path to egm*.pgm (egm96-5.pgm or egm2008-5.pgm).",
    )
    p.add_argument(
        "--geoid-kind",
        type=int,
        default=1,
        choices=(1, 3, -1, -3),
        help="pygeodesy interpolation: 1=bilinear, 3=bicubic, -1/-3 use SciPy. Default 1.",
    )
    p.add_argument(
        "--geoid-step-minutes",
        type=float,
        default=5.0,
        help="Coarse geoid grid spacing in arc-minutes for on-the-fly reprojection (default 5).",
    )
    # Void handling
    p.add_argument(
        "--void-value",
        type=float,
        default=-9999.0,
        help="Value used for nodata/void and as the Zarr .zarray fill_value. Default -9999.",
    )
    # Extent control
    p.add_argument(
        "--match-zarr",
        default=None,
        help="Use transform & shape from an existing Zarr to define target grid.",
    )
    p.add_argument(
        "--tight-to-tiles",
        action="store_true",
        help="Derive target grid bounds from union of tile footprints.",
    )
    p.add_argument(
        "--buffer-m",
        type=float,
        default=0.0,
        help="Buffer (meters) to expand bounds when --tight-to-tiles.",
    )
    return p.parse_args()


def find_tiles(input_dir: str) -> List[str]:
    exts = (".tif", ".tiff", ".TIF", ".TIFF")
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(exts)]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No GeoTIFFs found in {input_dir}.")
    return files


def bounds_intersect(b1, b2) -> bool:
    l1, bm1, r1, t1 = b1
    l2, bm2, r2, t2 = b2
    return not (r1 <= l2 or r2 <= l1 or t1 <= bm2 or t2 <= bm1)


def filter_tiles_to_bbox(files: List[str], bbox_ll: BBoxLL) -> List[str]:
    selected: List[str] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        for fp in files:
            try:
                with rasterio.open(fp) as ds:
                    src_crs = ds.crs
                    b = ds.bounds
                    if src_crs is None:
                        continue
                    if src_crs.to_epsg() != 4326:
                        transformer = Transformer.from_crs(
                            src_crs, CRS.from_epsg(4326), always_xy=True
                        )
                        xs = (b.left, b.right, b.right, b.left)
                        ys = (b.bottom, b.bottom, b.top, b.top)
                        llxs, llys = transformer.transform(xs, ys)
                        left, right = min(llxs), max(llxs)
                        bottom, top = min(llys), max(llys)
                        b_ll = (left, bottom, right, top)
                    else:
                        b_ll = (b.left, b.bottom, b.right, b.top)
                    if bounds_intersect(
                        b_ll, (bbox_ll.lon_min, bbox_ll.lat_min, bbox_ll.lon_max, bbox_ll.lat_max)
                    ):
                        selected.append(fp)
            except Exception as e:
                print(f"[WARN] Could not read {fp}: {e}", file=sys.stderr)
    if not selected:
        raise RuntimeError("No tiles intersect the requested bbox.")
    return selected


def compute_target_grid_from_bbox(bbox_ll: BBoxLL, res_m: float, dst_crs: CRS):
    transformer = Transformer.from_crs(CRS.from_epsg(4326), dst_crs, always_xy=True)
    xs = [bbox_ll.lon_min, bbox_ll.lon_max, bbox_ll.lon_max, bbox_ll.lon_min]
    ys = [bbox_ll.lat_min, bbox_ll.lat_min, bbox_ll.lat_max, bbox_ll.lat_max]
    X, Y = transformer.transform(xs, ys)
    x_min, x_max = min(X), max(X)
    y_min, y_max = min(Y), max(Y)

    def align_down(v, res):
        return math.floor(v / res) * res

    def align_up(v, res):
        return math.ceil(v / res) * res

    x_min_a = align_down(x_min, res_m)
    x_max_a = align_up(x_max, res_m)
    y_min_a = align_down(y_min, res_m)
    y_max_a = align_up(y_max, res_m)
    width = int(round((x_max_a - x_min_a) / res_m))
    height = int(round((y_max_a - y_min_a) / res_m))
    transform = from_origin(x_min_a, y_max_a, res_m, res_m)
    return transform, height, width, (x_min_a, y_min_a, x_max_a, y_max_a)


def compute_target_grid_from_tiles(
    tiles: List[str], dst_crs: CRS, res_m: float, buffer_m: float = 0.0
):
    xs_min, ys_min, xs_max, ys_max = [], [], [], []
    for fp in tiles:
        with rasterio.open(fp) as ds:
            l, b, r, t = ds.bounds
            l2, b2, r2, t2 = transform_bounds(ds.crs, dst_crs, l, b, r, t, densify_pts=21)
            xs_min.append(l2)
            ys_min.append(b2)
            xs_max.append(r2)
            ys_max.append(t2)
    x_min = min(xs_min) - buffer_m
    y_min = min(ys_min) - buffer_m
    x_max = max(xs_max) + buffer_m
    y_max = max(ys_max) + buffer_m

    def align_down(v, res):
        return math.floor(v / res) * res

    def align_up(v, res):
        return math.ceil(v / res) * res

    x_min_a = align_down(x_min, res_m)
    x_max_a = align_up(x_max, res_m)
    y_min_a = align_down(y_min, res_m)
    y_max_a = align_up(y_max, res_m)
    width = int(round((x_max_a - x_min_a) / res_m))
    height = int(round((y_max_a - y_min_a) / res_m))
    transform = from_origin(x_min_a, y_max_a, res_m, res_m)
    return transform, height, width, (x_min_a, y_min_a, x_max_a, y_max_a)


def compute_target_grid_from_zarr(ref_path: str):
    arr = zarr.open_array(ref_path, mode="r")
    attrs = getattr(arr, "attrs", {})
    # prefer explicit transform from attrs
    if "transform" in attrs:
        t = attrs["transform"]
        x_min, x_res, _, y_max, _, y_step = t
        res = abs(float(x_res))
        height, width = arr.shape
        transform = Affine(x_res, 0.0, x_min, 0.0, y_step, y_max)  # y_step should be -res
        x_max = x_min + width * res
        y_min = y_max + height * y_step
        bounds = (x_min, y_min, x_max, y_max)
        return transform, height, width, bounds, res
    else:
        raise RuntimeError(f"{ref_path} missing 'transform' in .zattrs; cannot match grid.")


def build_coarse_geoid_ll(bbox_ll: BBoxLL, step_minutes: float, backend: GeoidBackend):
    step_deg = step_minutes / 60.0
    lon_min, lat_min, lon_max, lat_max = (
        bbox_ll.lon_min,
        bbox_ll.lat_min,
        bbox_ll.lon_max,
        bbox_ll.lat_max,
    )
    nx = int(math.ceil((lon_max - lon_min) / step_deg))
    ny = int(math.ceil((lat_max - lat_min) / step_deg))
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)
    xs = (np.arange(nx) + 0.5) * transform.a + transform.c
    ys = (np.arange(ny) + 0.5) * transform.e + transform.f
    lon2d, lat2d = np.meshgrid(xs, ys)
    N = backend.height(lat2d, lon2d).astype(np.float32)
    return N, transform


def create_root_zarr_array(
    path: str, shape, chunks, dtype_str: str, compressor, fill_value: float
) -> zarr.Array:
    if os.path.exists(path):
        import shutil

        shutil.rmtree(path)
    dtype = np.dtype(dtype_str)
    arr = zarr.open_array(
        path,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        order="C",
        fill_value=np.array(fill_value, dtype=dtype),
    )
    return arr


def resampling_from_name(name: str) -> Resampling:
    return Resampling.average if name.lower() == "average" else Resampling.bilinear


def reproject_tile_into_window(
    src_ds: rasterio.DatasetReader,
    dst_transform_full: rasterio.Affine,
    dst_crs: CRS,
    window: rasterio.windows.Window,
    resampling: Resampling,
) -> np.ndarray:
    win_transform = window_transform(window, dst_transform_full)
    h = int(round(window.height))
    w = int(round(window.width))
    out = np.full((h, w), np.nan, dtype=np.float32)
    src_nodata = src_ds.nodata
    reproject(
        source=rasterio.band(src_ds, 1),
        destination=out,
        src_transform=src_ds.transform,
        src_crs=src_ds.crs,
        dst_transform=win_transform,
        dst_crs=dst_crs,
        src_nodata=src_nodata,
        dst_nodata=np.nan,
        resampling=resampling,
        num_threads=2,
    )
    return out


def geoid_window_from_coarse(
    coarse_geoid_ll: np.ndarray,
    coarse_transform_ll: rasterio.Affine,
    dst_transform_full: rasterio.Affine,
    dst_crs: CRS,
    window: rasterio.windows.Window,
) -> np.ndarray:
    win_transform = window_transform(window, dst_transform_full)
    h = int(round(window.height))
    w = int(round(window.width))
    out = np.empty((h, w), dtype=np.float32)
    reproject(
        source=coarse_geoid_ll,
        destination=out,
        src_transform=coarse_transform_ll,
        src_crs=CRS.from_epsg(4326),
        dst_transform=win_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
        num_threads=2,
    )
    return out


def main():
    args = _parse_args()

    dst_crs = CRS.from_epsg(3413)

    # Geoid backend
    if args.geoid_backend == "geographiclib":
        geoid = GeographicLibGeoid(args.geoid_model)
    else:
        if not args.geoid_pgm:
            print(
                "ERROR: --geoid-backend pygeodesy requires --geoid-pgm /path/to/egm*.pgm",
                file=sys.stderr,
            )
            sys.exit(2)
        geoid = PyGeodesyGeoid(args.geoid_pgm, kind=args.geoid_kind)

    bbox_ll = BBoxLL(*args.bbox)

    # Tiles
    print("Scanning tiles...")
    tiles = filter_tiles_to_bbox(find_tiles(args.input_dir), bbox_ll)
    print(f"Found {len(tiles)} tiles intersecting bbox.")

    # Target grid selection
    if args.match_zarr:
        print(f"Matching reference grid: {args.match_zarr}")
        transform, height, width, dst_bounds, res_from_ref = compute_target_grid_from_zarr(
            args.match_zarr
        )
        # If user didn't override --res, adopt ref resolution
        if args.res is None or args.res != res_from_ref:
            print(f"Using resolution from reference: {res_from_ref} m")
            args.res = res_from_ref
    elif args.tight_to_tiles:
        print("Computing grid from union of tile footprints (tight-to-tiles)...")
        transform, height, width, dst_bounds = compute_target_grid_from_tiles(
            tiles, dst_crs, args.res, buffer_m=args.buffer_m
        )
    else:
        transform, height, width, dst_bounds = compute_target_grid_from_bbox(
            bbox_ll, args.res, dst_crs
        )
    print(f"Target grid: {width} x {height} at {args.res} m (EPSG:3413)")

    # Coarse geoid (still built over lat/lon bbox; OK even if output bounds are tighter/wider)
    print("Computing coarse geoid grid...")
    coarse_N, coarse_T = build_coarse_geoid_ll(bbox_ll, args.geoid_step_minutes, geoid)
    print(f"Coarse geoid grid shape: {coarse_N.shape}")

    # Zarr root arrays (main + flipped)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    out_path = args.output
    if out_path.endswith(".zarr"):
        base = out_path[:-5]
        flipped_path = base + "_flipped.zarr"
    else:
        base = out_path
        flipped_path = out_path + "_flipped.zarr"

    z = create_root_zarr_array(
        out_path,
        (height, width),
        tuple(args.chunks),
        args.dtype,
        compressor,
        fill_value=args.void_value,
    )
    z_flip = create_root_zarr_array(
        flipped_path,
        (height, width),
        tuple(args.chunks),
        args.dtype,
        compressor,
        fill_value=args.void_value,
    )

    # Set .zattrs
    x_min, y_min, x_max, y_max = dst_bounds
    binsize = float(args.res)
    attrs_common = {
        "binsize": binsize,
        "bottom_l": [float(x_min), float(y_min)],
        "crs": CRS_WKT_FOR_ATTR,
        "ncols": int(width),
        "nrows": int(height),
        "top_l": [float(x_min), float(y_max)],
        "top_r": [float(x_max), float(y_max)],
        "transform": [float(x_min), binsize, 0.0, float(y_max), 0.0, -binsize],
        "void_value": float(args.void_value),
    }
    z.attrs.put(attrs_common)
    z_flip.attrs.put(attrs_common)

    resampling = resampling_from_name(args.resampling)
    void = np.array(args.void_value, dtype=z.dtype)

    # Process tiles
    for idx, fp in enumerate(tiles, 1):
        print(f"[{idx}/{len(tiles)}] Reprojecting {os.path.basename(fp)} ...")
        with rasterio.open(fp) as ds:
            l, b, r, t = ds.bounds
            l_dst, b_dst, r_dst, t_dst = transform_bounds(
                ds.crs, dst_crs, l, b, r, t, densify_pts=21
            )
            window = window_from_bounds(l_dst, b_dst, r_dst, t_dst, transform)
            row_off = max(0, int(math.floor(window.row_off)))
            col_off = max(0, int(math.floor(window.col_off)))
            row_max = min(height, int(math.ceil(window.row_off + window.height)))
            col_max = min(width, int(math.ceil(window.col_off + window.width)))
            h = row_max - row_off
            w = col_max - col_off
            if h <= 0 or w <= 0:
                print("  Skipping (no overlap after transform).")
                continue
            tight_window = rasterio.windows.Window(col_off, row_off, w, h)

            reprojected = reproject_tile_into_window(
                ds, transform, dst_crs, tight_window, resampling
            )
            geoid_win = geoid_window_from_coarse(
                coarse_N, coarse_T, transform, dst_crs, tight_window
            )

            valid_dem = np.isfinite(reprojected)
            valid_geoid = np.isfinite(geoid_win)
            valid = valid_dem & valid_geoid
            reprojected[valid] = reprojected[valid] + geoid_win[valid]

            block_out = np.where(valid_dem, reprojected, void).astype(z.dtype, copy=False)

            # MAIN write
            if args.overwrite_existing:
                z[row_off : row_off + h, col_off : col_off + w] = block_out
            else:
                current = z[row_off : row_off + h, col_off : col_off + w]
                write_where = (current == void) & (block_out != void)
                if np.any(write_where):
                    merged = np.where(write_where, block_out, current)
                    z[row_off : row_off + h, col_off : col_off + w] = merged

            # FLIPPED write: vertical flip
            flip_row_start = height - (row_off + h)
            flip_row_end = height - row_off
            block_flip = block_out[::-1, :]

            if args.overwrite_existing:
                z_flip[flip_row_start:flip_row_end, col_off : col_off + w] = block_flip
            else:
                current_f = z_flip[flip_row_start:flip_row_end, col_off : col_off + w]
                write_where_f = (current_f == void) & (block_flip != void)
                if np.any(write_where_f):
                    merged_f = np.where(write_where_f, block_flip, current_f)
                    z_flip[flip_row_start:flip_row_end, col_off : col_off + w] = merged_f

    print("Done.")
    print(f" - Main Zarr:    {out_path}  shape={z.shape} chunks={z.chunks} dtype={z.dtype}")
    print(
        f" - Flipped Zarr: {flipped_path}  shape={z_flip.shape} chunks={z_flip.chunks} dtype={z_flip.dtype}"
    )
    print("Attrs keys:", list(z.attrs.keys()))


if __name__ == "__main__":
    main()
