"""Shared helpers for MTG processing pipelines.

These helpers keep the individual pipeline entrypoints small and focused on
workflow differences.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from utils.scene_utils import make_scene, load_and_crop, get_channel
from utils.grid_utils import regrid_data, fill_missing_data_with_interpolation
from utils.io_utils import build_time_map, list_mtg_files, list_cth_files
from utils.time_utils import compute_timestamps, extract_mtg_time, extract_cth_time


def make_datastore(config: dict):
    import importlib
    import eumdac

    consumer_key = config.get("consumer_key")
    consumer_secret = config.get("consumer_secret")
    if not (consumer_key and consumer_secret):
        cred_file = config.get("credentials_file")
        if cred_file:
            try:
                module_name = os.path.splitext(cred_file.replace("/", "."))[0]
                creds = importlib.import_module(module_name)
                consumer_key = consumer_key or getattr(creds, "consumer_key", None)
                consumer_secret = consumer_secret or getattr(creds, "consumer_secret", None)
            except Exception:
                try:
                    import credentials as creds
                    consumer_key = consumer_key or getattr(creds, "consumer_key", None)
                    consumer_secret = consumer_secret or getattr(creds, "consumer_secret", None)
                except Exception:
                    pass
        else:
            try:
                import credentials as creds
                consumer_key = consumer_key or getattr(creds, "consumer_key", None)
                consumer_secret = consumer_secret or getattr(creds, "consumer_secret", None)
            except Exception:
                pass

    if not (consumer_key and consumer_secret):
        raise ValueError("Missing EUMETSAT credentials in config or credentials module")

    token = eumdac.AccessToken((consumer_key, consumer_secret))
    return eumdac.DataStore(token)


DATE_FMT = "%Y.%m.%d"


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, DATE_FMT).date()


def iter_dates(start_date: dt.date, end_date: dt.date):
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def resolve_download_spec(config: dict, name: str):
    for spec in config.get("download_products", []):
        if spec.get("name") == name and spec.get("enabled", True):
            return spec
    return None


def resolve_product_base(config: dict, name: str, fallback: str | None = None) -> Path:
    spec = resolve_download_spec(config, name)
    if spec and spec.get("base"):
        return Path(spec["base"])
    if fallback:
        return Path(fallback)
    raise KeyError(f"No base path configured for product '{name}'")


def day_output_dir(base: str | Path, roi_name: str, day: dt.date) -> Path:
    base_path = Path(base)
    outdir = base_path / roi_name / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_geolocation_file(path: Path, longitude: np.ndarray, latitude: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        data_vars={
            "longitude": (("y", "x"), longitude.astype(np.float32)),
            "latitude": (("y", "x"), latitude.astype(np.float32)),
        },
    )
    if not path.exists():
        ds.to_netcdf(path, format="NETCDF4")
    ds.close()


def ensure_channel_data(scene, channels: list[str], roi: dict, parallax: bool = False):
    cropped = load_and_crop(scene, channels, roi, parallax=parallax)
    return cropped


def extract_geolocation(crop, channel: str, config: dict):
    channel_name = get_channel([channel], config.get("parallax", False))[0]
    lon, lat = crop[channel_name].attrs["area"].get_lonlats()
    return lon, lat


def make_regular_target_grid(roi: dict, grid_step_deg):
    lats = np.arange(roi["lat_min"], roi["lat_max"] + grid_step_deg[0], grid_step_deg[0])
    lons = np.arange(roi["lon_min"], roi["lon_max"] + grid_step_deg[1], grid_step_deg[1])
    return np.meshgrid(lats, lons, indexing="ij")


def build_channel_array(crop, channel: str, config: dict, target_grid=None):
    channel_name = get_channel([channel], config.get("parallax", False))[0]
    raw = crop[channel_name].values.astype(np.float32)

    if target_grid is None:
        return raw

    lat_src, lon_src = crop[channel_name].attrs["area"].get_lonlats()
    filled = fill_missing_data_with_interpolation(lat_src, lon_src, raw, method=config.get("interp_method", "nearest"))
    lat2d, lon2d = target_grid
    return regrid_data(lat_src, lon_src, filled, lat2d, lon2d, method=config.get("interp_method", "nearest"))


def open_scene(mtg_files, cth_files, config: dict):
    from satpy import Scene

    if config.get("parallax"):
        return make_scene(mtg_files, cth_files, config)
    return Scene(reader=config["mtg_reader"], filenames=mtg_files)


def upload_file(local_path: Path, config: dict, remote_name: str | None = None):
    upload_target = config.get("upload_target")
    if not upload_target:
        return None

    upload_command = config.get("upload_command")
    if upload_command:
        cmd = [part.format(source=str(local_path), target=upload_target, remote_name=remote_name or local_path.name) for part in upload_command]
        subprocess.run(cmd, check=True)
        return None

    target = Path(upload_target)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, target / (remote_name or local_path.name))
    return target / (remote_name or local_path.name)


def make_daily_dataset(times: list[dt.datetime], arrays: dict[str, list[np.ndarray]], x: np.ndarray, y: np.ndarray, geolocation=None):
    data_vars = {}
    coords = {
        "time": ("time", times),
        "y": ("y", y),
        "x": ("x", x),
    }
    if geolocation is not None:
        longitude, latitude = geolocation
        data_vars["longitude"] = (("y", "x"), longitude.astype(np.float32))
        data_vars["latitude"] = (("y", "x"), latitude.astype(np.float32))

    for name, values in arrays.items():
        data_vars[name] = (("time", "y", "x"), np.stack(values, axis=0).astype(np.float32))

    return xr.Dataset(data_vars=data_vars, coords=coords)


def save_daily_dataset(ds: xr.Dataset, out_path: Path, compress_level: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for var in ds.data_vars:
        if ds[var].ndim > 0:
            encoding[var] = {"zlib": True, "complevel": compress_level}
    if "time" in ds.coords:
        encoding["time"] = {"dtype": "i4", "units": "seconds since 2000-01-01"}
    ds.to_netcdf(out_path, format="NETCDF4", encoding=encoding)


def build_time_indexes(config: dict):
    timestamps = compute_timestamps(config["start_date"], config["end_date"], config["time_interval_min"])
    mtg_base = resolve_product_base(config, "fci_normal_res", config.get("download_base", None))
    mtg_files = list_mtg_files(mtg_base, timestamps, pattern=config.get("file_extension", "*.nc"))
    mtg_map = build_time_map(mtg_files, lambda f: extract_mtg_time(f, config["time_interval_min"]))

    cth_files = []
    cth_map = {}
    if config.get("parallax"):
        cth_base = Path(config.get("cth_base", ""))
        if cth_base.exists():
            cth_files = list_cth_files(cth_base, "*.nc")
            cth_map = build_time_map(cth_files, extract_cth_time)

    return timestamps, mtg_map, cth_map
