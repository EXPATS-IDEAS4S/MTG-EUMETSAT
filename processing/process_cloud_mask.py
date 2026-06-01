#!/usr/bin/env python3

"""Unzip, crop, and optionally plot MTG FCI cloud-mask products.

The script uses `user_config.py` for:
- `process_cloud_mask` to enable/disable execution
- `cloud_mask_base` or the `download_products` cloud-mask entry as input root
- `cloud_mask_start_date` / `cloud_mask_end_date` for the date range
- `roi` and `roi_name` for cropping and path resolution
- `cloud_mask_plot` to write quicklook PNGs into a `plot/` subfolder

It scans for `.zip` archives, extracts them to a temporary directory, crops any
NetCDF files to the configured ROI, and saves the cropped NetCDF next to the
source archive.
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
import tempfile
import zipfile
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
try:
    from pyproj import CRS, Transformer
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False
    Transformer = None
from satpy import Scene


sys.path.append("/home/Daniele/codes/MTG-EUMETSAT/")
from configs.config_loader import CONFIG as cfg


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


DATE_FMT = "%Y.%m.%d"


def parse_config_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, DATE_FMT).date()


def iter_dates(start_date: dt.date, end_date: dt.date):
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def resolve_cloud_mask_base(config: dict) -> Path:
    """Resolve the source folder for cloud-mask zip archives."""
    base = config.get("cloud_mask_base")
    if base:
        return Path(base)

    for spec in config.get("download_products", []):
        if spec.get("name") == "cloud_mask" and spec.get("base"):
            return Path(spec["base"])

    return Path("/sat_data/MTG/FCI/cloud_masks/")


def resolve_cloud_mask_output(config: dict) -> Path:
    """Resolve the destination folder for processed cloud-mask outputs."""
    output_base = config.get("cloud_mask_output")
    if output_base:
        return Path(output_base)

    return Path(config.get("process_base", "/sat_data/MTG/FCI/cloud_masks_processed/"))


def candidate_roots(base: Path, roi_name: str) -> list[Path]:
    roots = []
    roi_root = base / roi_name
    if roi_root.exists():
        roots.append(roi_root)
    if base.exists():
        roots.append(base)
    return roots


def find_zip_archives(base: Path, roi_name: str, start_date: dt.date, end_date: dt.date):
    seen = set()
    for current in iter_dates(start_date, end_date):
        year = f"{current.year:04d}"
        month = f"{current.month:02d}"
        day = f"{current.day:02d}"
        for root in candidate_roots(base, roi_name):
            day_dir = root / year / month / day
            if not day_dir.exists():
                continue
            for zip_path in sorted(day_dir.glob("*.zip")):
                if zip_path not in seen:
                    seen.add(zip_path)
                    yield zip_path


def find_lat_lon_names(ds: xr.Dataset):
    """Return the first matching latitude/longitude variable pair in the dataset."""
    candidate_pairs = [
        ("lat", "lon"),
        ("latitude", "longitude"),
        ("latitude_2d", "longitude_2d"),
        ("lat_2d", "lon_2d"),
    ]

    for lat_name, lon_name in candidate_pairs:
        lat = ds.coords.get(lat_name)
        if lat is None:
            lat = ds.variables.get(lat_name)

        lon = ds.coords.get(lon_name)
        if lon is None:
            lon = ds.variables.get(lon_name)

        if lat is not None and lon is not None and lat.shape == lon.shape:
            return lat_name, lon_name

    return None, None


def minimize_cloud_mask_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Keep only the cloud-state layer and the projection metadata needed to interpret it."""
    preferred_vars = [name for name in ("cloud_state", "cma") if name in ds.data_vars]
    if not preferred_vars:
        preferred_vars = list(ds.data_vars)

    reduced = ds[preferred_vars].copy()
    if "mtg_geos_projection" in ds:
        reduced["mtg_geos_projection"] = ds["mtg_geos_projection"]

    return reduced


def crop_dataset_to_roi(ds: xr.Dataset, roi: dict, nc_path: Path = None):
    """Crop dataset to ROI. If dataset lacks explicit lat/lon grids, use
    Satpy to derive lat/lon from the file area when `nc_path` is provided.
    Returns (cropped_dataset, lat_name, lon_name) or (None, None, None).
    """
    lat_name, lon_name = find_lat_lon_names(ds)
    lat = None
    lon = None

    if lat_name is not None and lon_name is not None:
        lat = ds[lat_name].values
        lon = ds[lon_name].values

    # If pyproj is available and the dataset contains mtg_geos_projection,
    # build lon/lat from x/y using the projection parameters.
    if (lat is None or lon is None) and "mtg_geos_projection" in ds:
        if _HAS_PYPROJ:
            try:
                proj_attrs = ds["mtg_geos_projection"].attrs
                proj_keys = [
                    "perspective_point_height",
                    "longitude_of_projection_origin",
                    "sweep_angle_axis",
                    "semi_major_axis",
                    "semi_minor_axis",
                ]
                missing = [k for k in proj_keys if k not in proj_attrs or proj_attrs.get(k) is None]
                if missing:
                    log.debug("mtg_geos_projection missing keys: %s; will fallback to Satpy", missing)
                    raise RuntimeError("missing projection keys")

                h = float(proj_attrs.get("perspective_point_height"))
                lon_0 = float(proj_attrs.get("longitude_of_projection_origin"))
                sweep = proj_attrs.get("sweep_angle_axis", "x")
                a = float(proj_attrs.get("semi_major_axis"))
                b = float(proj_attrs.get("semi_minor_axis"))

                geos_def = f"+proj=geos +h={h} +lon_0={lon_0} +sweep={sweep} +a={a} +b={b} +units=m +no_defs"
                geos_crs = CRS.from_proj4(geos_def)
                wgs84 = CRS.from_epsg(4326)
                transformer = Transformer.from_crs(geos_crs, wgs84, always_xy=True)

                # x and y coordinates exist as 1D arrays in the dataset
                if "x" in ds.coords and "y" in ds.coords:
                    x = ds["x"].values
                    y = ds["y"].values
                    xv, yv = np.meshgrid(x, y, indexing="xy")
                    lon, lat = transformer.transform(xv, yv)
                    lat_name = "lat_2d"
                    lon_name = "lon_2d"
            except Exception:
                lat = None
                lon = None
        else:
            # Fallback to Satpy if pyproj not available and nc_path provided
            if nc_path is not None:
                try:
                    scn = Scene(filenames=[str(nc_path)], reader=cfg.get('cth_reader', 'fci_l2_nc'))
                    var = None
                    for candidate in ("cloud_state", "clm", "retrieved_cloud_top_height"):
                        if candidate in scn.available_dataset_names():
                            var = candidate
                            break
                    if var is None:
                        names = list(scn.available_dataset_names())
                        var = names[0] if names else None
                    if var is None:
                        return None, None, None
                    scn.load([var])
                    lon_grid, lat_grid = scn[var].attrs["area"].get_lonlats()
                    lat = lat_grid
                    lon = lon_grid
                    lat_name = "lat_2d"
                    lon_name = "lon_2d"
                except Exception:
                    return None, None, None

    if lat is None or lon is None or lat.shape != lon.shape or lat.ndim < 2:
        return None, None, None

    mask = (
        (lat >= roi["lat_min"]) & (lat <= roi["lat_max"]) &
        (lon >= roi["lon_min"]) & (lon <= roi["lon_max"])
    )
    if not np.any(mask):
        return None, None, None

    ys, xs = np.where(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    if "number_of_rows" in ds.dims and "number_of_columns" in ds.dims:
        y_dim = "number_of_rows"
        x_dim = "number_of_columns"
    else:
        y_dim, x_dim = list(ds.dims)[:2]

    cropped = ds.isel({y_dim: slice(y_min, y_max + 1), x_dim: slice(x_min, x_max + 1)})
    return cropped, lat_name, lon_name


def save_cropped_dataset(cropped: xr.Dataset, output_path: Path, compress_level: int):
    def _clean_attrs(ds: xr.Dataset):
        # Remove attributes with value None and convert numpy scalars
        def _clean(d):
            to_del = []
            for k, v in d.attrs.items():
                if v is None:
                    to_del.append(k)
                    continue
                # convert numpy scalar to python scalar
                if hasattr(v, 'item') and not isinstance(v, (str, bytes, list, tuple)):
                    try:
                        d.attrs[k] = v.item()
                    except Exception:
                        pass
            for k in to_del:
                d.attrs.pop(k, None)

        _clean(ds)
        for var in ds.variables:
            _clean(ds[var])

    _clean_attrs(cropped)

    encoding = {
        name: {"zlib": True, "complevel": compress_level}
        for name in cropped.data_vars
    }
    cropped.to_netcdf(output_path, format="NETCDF4", encoding=encoding)


def add_lat_lon_coords(ds: xr.Dataset, lat_grid, lon_grid) -> xr.Dataset:
    """Attach explicit 2D latitude/longitude coordinates to the dataset."""
    data_var = next(iter(ds.data_vars), None)
    if data_var is None:
        return ds

    dims = ds[data_var].dims
    if len(dims) < 2:
        return ds

    y_dim, x_dim = dims[-2], dims[-1]
    ds = ds.assign_coords(
        lat=((y_dim, x_dim), lat_grid),
        lon=((y_dim, x_dim), lon_grid),
    )
    return ds


def plot_cropped_dataset(cropped: xr.Dataset, plot_path: Path, roi: dict, lat_name: str, lon_name: str):
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    data_var_name = next(iter(cropped.data_vars), None)
    if data_var_name is None:
        return

    data = cropped[data_var_name]
    lat = cropped[lat_name].values
    lon = cropped[lon_name].values

    finite_values = data.values[np.isfinite(data.values)]
    if finite_values.size and np.issubdtype(data.dtype, np.integer):
        cmap = "tab20"
    else:
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    mesh = ax.pcolormesh(lon, lat, data.values, cmap=cmap, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(data_var_name)
    ax.coastlines(resolution="50m", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.2)
    ax.set_extent(
        [roi["lon_min"], roi["lon_max"], roi["lat_min"], roi["lat_max"]],
        crs=ccrs.PlateCarree(),
    )
    ax.set_title(f"{data_var_name} cropped to ROI")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_zip_archive(zip_path: Path, roi: dict, plot_enabled: bool, compress_level: int, output_base: Path, roi_name: str):
    log.info("Processing %s", zip_path)

    with tempfile.TemporaryDirectory(prefix="cloud_mask_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        with zipfile.ZipFile(zip_path, mode="r") as zf:
            zf.extractall(tmp_dir)

        nc_files = sorted(tmp_dir.rglob("*.nc"))
        if not nc_files:
            log.warning("No NetCDF files found inside %s", zip_path)
            return

        for nc_path in nc_files:
            rel_output_dir = (
                output_base
                / roi_name
                / zip_path.parent.parent.parent.name
                / zip_path.parent.parent.name
                / zip_path.parent.name
            )
            rel_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = rel_output_dir / f"{nc_path.stem}_cropped.nc"
            plot_path = rel_output_dir / "plot" / f"{nc_path.stem}_plot.png"

            if output_path.exists():
                log.info("Skipping existing file %s", output_path)
                continue

            try:
                # Use Satpy Scene with the configured L2 reader to load and crop
                reader = cfg.get('cth_reader', 'fci_l2_nc')
                scn = Scene(reader=reader, filenames=[str(nc_path)])
                names = list(scn.available_dataset_names())
                if not names:
                    log.warning("No datasets available in %s via Satpy", nc_path)
                    continue

                # prefer cloud mask variable names
                var = None
                for candidate in ("cloud_state", "clm", "cma"):
                    if candidate in names:
                        var = candidate
                        break
                if var is None:
                    var = names[0]
            
                scn.load([var])
                crop_scene = scn.crop(ll_bbox=(roi['lon_min'], roi['lat_min'], roi['lon_max'], roi['lat_max']))

                # Extract DataArray from cropped scene
                da = crop_scene[var]
                # Convert to xarray Dataset if possible
                try:
                    ds_out = da.to_dataset(name=var) if hasattr(da, 'to_dataset') else xr.Dataset({var: (da.dims, da.values)})
                except Exception:
                    ds_out = xr.Dataset({var: (da.dims, da.values)})

                # Attach explicit geolocation coordinates so the output is self-describing.
                try:
                    lon_grid, lat_grid = crop_scene[var].attrs['area'].get_lonlats()
                    ds_out = add_lat_lon_coords(ds_out, lat_grid, lon_grid)
                except Exception:
                    log.debug('Could not attach lat/lon coordinates to %s', nc_path, exc_info=True)
                

                # Minimize to preferred variables and save
                ds_min = minimize_cloud_mask_dataset(ds_out)
               
                save_cropped_dataset(ds_min, output_path, compress_level)
                log.info("Saved cropped dataset to %s", output_path)

                if plot_enabled:
                    # For plotting, try to get lat/lon from the crop_scene area
                    try:
                        lon_grid, lat_grid = crop_scene[var].attrs['area'].get_lonlats()
                        temp_ds = xr.Dataset({var: (crop_scene[var].dims, crop_scene[var].values)})
                        temp_ds = add_lat_lon_coords(temp_ds, lat_grid, lon_grid)
                        plot_cropped_dataset(temp_ds, plot_path, roi, 'lat', 'lon')
                        log.info("Saved plot to %s", plot_path)
                    except Exception:
                        log.debug('Failed to generate plot for %s', output_path, exc_info=True)
            except Exception:
                log.exception("Error processing %s", nc_path)


def main():
    if not cfg.get("process_cloud_mask", False):
        log.info("process_cloud_mask is disabled in user_config.py; nothing to do.")
        return

    roi = cfg["roi"]
    roi_name = cfg.get("roi_name", "roi")
    start_date = parse_config_date(cfg.get("cloud_mask_start_date", cfg["start_date"]))
    end_date = parse_config_date(cfg.get("cloud_mask_end_date", cfg["end_date"]))
    base = resolve_cloud_mask_base(cfg)
    output_base = resolve_cloud_mask_output(cfg)
    plot_enabled = cfg.get("cloud_mask_plot", False)
    compress_level = int(cfg.get("compress_level", 9))

    log.info("Cloud-mask base: %s", base)
    log.info("Cloud-mask output: %s", output_base)
    log.info("ROI: %s", roi_name)
    log.info("Date range: %s to %s", start_date, end_date)

    archives = list(find_zip_archives(base, roi_name, start_date, end_date))
    if not archives:
        log.info("No cloud-mask zip archives found.")
        return

    log.info("Found %d zip archive(s).", len(archives))
   
    for zip_path in archives:
        process_zip_archive(zip_path, roi, plot_enabled, compress_level, output_base, roi_name)

    log.info("Cloud-mask processing complete.")


if __name__ == "__main__":
    main()