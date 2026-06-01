#!/usr/bin/env python3

"""Process OCA/CTH NetCDF files into cropped products usable for parallax correction."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr


log = logging.getLogger(__name__)


DEFAULT_VARIABLE = "retrieved_cloud_top_height"


def find_lat_lon_names(ds: xr.Dataset):
    for lat_name, lon_name in [("lat", "lon"), ("latitude", "longitude"), ("lat_2d", "lon_2d")]:
        lat = ds.coords.get(lat_name)
        if lat is None:
            lat = ds.variables.get(lat_name)
        lon = ds.coords.get(lon_name)
        if lon is None:
            lon = ds.variables.get(lon_name)
        if lat is not None and lon is not None and lat.shape == lon.shape:
            return lat_name, lon_name
    return None, None


def crop_dataset_to_roi(ds: xr.Dataset, roi: dict, nc_path: Path, variable: str):
    from satpy import Scene

    scn = Scene(filenames=[str(nc_path)], reader="fci_l2_nc")
    scn.load([variable])
    lon_grid, lat_grid = scn[variable].attrs["area"].get_lonlats()

    mask = (
        (lat_grid >= roi["lat_min"]) & (lat_grid <= roi["lat_max"]) &
        (lon_grid >= roi["lon_min"]) & (lon_grid <= roi["lon_max"])
    )
    if not np.any(mask):
        return None, None

    ys, xs = np.where(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    cropped = ds.isel(number_of_rows=slice(y_min, y_max + 1), number_of_columns=slice(x_min, x_max + 1))
    return cropped, (lon_grid[y_min:y_max + 1, x_min:x_max + 1], lat_grid[y_min:y_max + 1, x_min:x_max + 1])


def save_reduced_dataset(ds: xr.Dataset, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, format="NETCDF4")


def process_oca_cth_files(source_dir: str | Path, output_dir: str | Path, roi: dict, variable: str = DEFAULT_VARIABLE):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(source_dir.rglob("*.nc"))
    processed = 0
    for nc_path in files:
        if output_dir in nc_path.parents:
            continue
        try:
            ds = xr.open_dataset(nc_path, engine="netcdf4")
            if variable not in ds:
                ds.close()
                continue

            cropped, _ = crop_dataset_to_roi(ds, roi, nc_path, variable)
            if cropped is None:
                ds.close()
                continue

            var_data = cropped[variable].copy(deep=True)
            proj_attrs = ds.get("mtg_geos_projection", {}).attrs if "mtg_geos_projection" in ds else {}
            var_data.attrs.update({
                "satellite_nominal_latitude": proj_attrs.get("latitude_of_projection_origin", 0.0),
                "satellite_nominal_longitude": proj_attrs.get("longitude_of_projection_origin", 0.0),
                "satellite_nominal_altitude": proj_attrs.get("perspective_point_height", 35786400.0),
            })

            ds_new = xr.Dataset(
                data_vars={variable: var_data},
                coords={k: cropped[k] for k in ["x", "y"] if k in cropped},
                attrs=ds.attrs,
            )
            if "mtg_geos_projection" in ds:
                ds_new["mtg_geos_projection"] = ds["mtg_geos_projection"]

            save_dir = output_dir / nc_path.name.split("_")[7][:8]
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / nc_path.name
            save_reduced_dataset(ds_new, out_path)
            processed += 1
            ds.close()
        except Exception as exc:
            log.warning("Failed to process %s: %s", nc_path, exc)

    return processed


def main():
    import sys

    sys.path.append("/home/Daniele/codes/MTG-EUMETSAT/")
    from configs.config_loader import CONFIG as cfg

    source_dir = cfg.get("oca_cth_source_base", cfg.get("oca_source_base", cfg.get("cth_source_base", cfg.get("cth_base", "/sat_data/MTG/FCI/oca/"))))
    output_dir = cfg.get("cth_base", "/sat_data/MTG/FCI/oca/cth/")
    roi = cfg["roi"]
    processed = process_oca_cth_files(source_dir, output_dir, roi, cfg.get("cth_variable", DEFAULT_VARIABLE))
    print(f"Processed {processed} CTH file(s).")


if __name__ == "__main__":
    main()
