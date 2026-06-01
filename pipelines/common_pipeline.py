"""Generic workflow runner for MTG processing pipelines."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

from download.download_mtg_products import download_day_for_spec, build_download_specs
from oca.process_oca_cth import process_oca_cth_files
from pipelines.pipeline_utils import (
    day_output_dir,
    ensure_channel_data,
    extract_geolocation,
    make_daily_dataset,
    make_regular_target_grid,
    open_scene,
    resolve_download_spec,
    save_daily_dataset,
    save_geolocation_file,
    upload_file,
)
from utils.io_utils import build_time_map, list_cth_files, list_mtg_files
from utils.time_utils import compute_timestamps, extract_cth_time, extract_mtg_time
from utils.scene_utils import get_channel


def _grid_step(config: dict):
    steps = config.get("grid_step_deg", [(0.02, 0.02)])
    if isinstance(steps, (list, tuple)) and steps and isinstance(steps[0], (list, tuple)):
        return tuple(steps[0])
    return tuple(steps)


def _file_map(files, extractor):
    return build_time_map(files, extractor) if files else {}


def _first_available_crop(day_times, primary_map, cth_map, config, primary_variables):
    for ts in day_times:
        primary_files = primary_map.get(ts)
        cth_files = cth_map.get(ts) if config.get("parallax") else None
        if not primary_files or (config.get("parallax") and not cth_files):
            continue
        scene = open_scene(primary_files, cth_files, config)
        crop = ensure_channel_data(scene, primary_variables, config["roi"], parallax=config.get("parallax", False))
        return ts, crop
    return None, None


def _build_scene_array(crop, variables, config, target_grid):
    arrays = {}
    for variable in variables:
        arrays[variable] = []
        arrays[variable].append(
            np.asarray(
                crop[variable].values.astype(np.float32)
                if target_grid is None
                else crop[variable].values.astype(np.float32)
            )
        )
    return arrays


def _extract_channel_array(crop, variable, config, target_grid):
    scene_variable = get_channel([variable], config.get("parallax", False))[0]

    if target_grid is None:
        return crop[scene_variable].values.astype(np.float32)

    lon_src, lat_src = crop[scene_variable].attrs["area"].get_lonlats()
    raw = crop[scene_variable].values.astype(np.float32)
    from utils.grid_utils import fill_missing_data_with_interpolation, regrid_data

    filled = fill_missing_data_with_interpolation(lat_src, lon_src, raw, method=config.get("interp_method", "nearest"))
    lat2d, lon2d = target_grid
    return regrid_data(lat_src, lon_src, filled, lat2d, lon2d, method=config.get("interp_method", "nearest"))


def _load_secondary_scene(secondary_files, config, secondary_variables, roi):
    from satpy import Scene

    if not secondary_files or not secondary_variables:
        return None, None
    reader = config.get("secondary_reader", config.get("cma_reader", "fci_l2_nc"))
    scene = Scene(reader=reader, filenames=secondary_files)
    available = set(scene.available_dataset_names())
    selected = None
    for candidate in secondary_variables:
        if candidate in available:
            selected = candidate
            break
    if selected is None:
        return None, None
    scene.load([selected])
    crop = scene.crop(ll_bbox=(roi["lon_min"], roi["lat_min"], roi["lon_max"], roi["lat_max"]))
    return crop, selected


def _resolve_inputs_for_day(config, day: dt.date, primary_spec_name: str, secondary_spec_name: str | None = None):
    primary_spec = resolve_download_spec(config, primary_spec_name)
    secondary_spec = resolve_download_spec(config, secondary_spec_name) if secondary_spec_name else None
    if primary_spec is None:
        raise KeyError(f"Missing enabled download spec '{primary_spec_name}'")
    return primary_spec, secondary_spec


def run_pipeline_for_day(
    config: dict,
    datastore,
    day: dt.date,
    primary_spec_name: str,
    primary_reader: str,
    primary_variables: list[str],
    output_prefix: str,
    time_extractor=extract_mtg_time,
    secondary_spec_name: str | None = None,
    secondary_variables: list[str] | None = None,
    secondary_output_name: str = "cma",
    use_secondary: bool = False,
    use_parallax: bool = False,
    cth_source_base: str | Path | None = None,
):
    config = dict(config)
    config["mtg_reader"] = primary_reader
    config["parallax"] = use_parallax
    config.setdefault("secondary_reader", config.get("cma_reader", "fci_l2_nc"))

    primary_spec, secondary_spec = _resolve_inputs_for_day(config, day, primary_spec_name, secondary_spec_name)
    download_day_for_spec(datastore, primary_spec, day)
    if secondary_spec is not None:
        download_day_for_spec(datastore, secondary_spec, day)

    if use_parallax:
        if cth_source_base is None:
            cth_source_base = primary_spec["base"]
        process_oca_cth_files(cth_source_base, config.get("cth_base"), config["roi"], config.get("cth_variable", "retrieved_cloud_top_height"))

    day_times = [ts for ts in compute_timestamps(config["start_date"], config["end_date"], config["time_interval_min"])[:-1] if ts.date() == day]
    if not day_times:
        return None

    primary_base = Path(primary_spec["base"])
    primary_files = list_mtg_files(primary_base, day_times, pattern=config.get("file_extension", "*.nc"))
    primary_map = _file_map(primary_files, lambda f: time_extractor(f, config["time_interval_min"]))

    cth_map = {}
    if use_parallax and config.get("cth_base"):
        cth_files = list_cth_files(Path(config["cth_base"]), "*.nc")
        cth_map = _file_map(cth_files, extract_cth_time)

    secondary_map = {}
    if use_secondary and secondary_spec is not None:
        secondary_files = list_mtg_files(Path(secondary_spec["base"]), day_times, pattern=config.get("file_extension", "*.nc"))
        secondary_map = _file_map(secondary_files, lambda f: time_extractor(f, config["time_interval_min"]))

    reference_ts, reference_crop = _first_available_crop(day_times, primary_map, cth_map, config, primary_variables)
    if reference_crop is None:
        return None

    regular_grid = bool(config.get("regular_grid", False))
    target_grid = make_regular_target_grid(config["roi"], _grid_step(config)) if regular_grid else None

    if regular_grid:
        lon_geo, lat_geo = target_grid[1], target_grid[0]
    else:
        lon_geo, lat_geo = extract_geolocation(reference_crop, primary_variables[0], config)

    output_base = config.get(f"{output_prefix}_output_base", config.get("process_base"))
    geo_path = Path(output_base) / config["roi_name"] / "geolocation" / ("regular" if regular_grid else "native") / "geolocation.nc"
    save_geolocation_file(geo_path, lon_geo, lat_geo)

    reference_shape = lon_geo.shape
    x = np.arange(reference_shape[1], dtype=np.int32)
    y = np.arange(reference_shape[0], dtype=np.int32)

    series = {var: [] for var in primary_variables}
    if use_secondary and secondary_variables:
        series[secondary_output_name] = []
    times = []

    for ts in day_times:
        primary_files_at_ts = primary_map.get(ts)
        cth_files_at_ts = cth_map.get(ts) if use_parallax else None
        secondary_files_at_ts = secondary_map.get(ts) if use_secondary else None

        times.append(ts)

        if primary_files_at_ts and (not use_parallax or cth_files_at_ts):
            scene = open_scene(primary_files_at_ts, cth_files_at_ts, config)
            crop = ensure_channel_data(scene, primary_variables, config["roi"], parallax=use_parallax)
            for variable in primary_variables:
                series[variable].append(_extract_channel_array(crop, variable, config, target_grid))
        else:
            for variable in primary_variables:
                series[variable].append(np.full(reference_shape, np.nan, dtype=np.float32))

        if use_secondary and secondary_variables:
            secondary_crop, selected_secondary = _load_secondary_scene(secondary_files_at_ts, config, secondary_variables, config["roi"])
            if secondary_crop is None or selected_secondary is None:
                series[secondary_output_name].append(np.full(reference_shape, np.nan, dtype=np.float32))
            else:
                series[secondary_output_name].append(_extract_channel_array(secondary_crop, selected_secondary, config, target_grid))

    daily_ds = make_daily_dataset(times, series, x, y)
    daily_dir = day_output_dir(output_base, config["roi_name"], day)
    daily_path = daily_dir / f"{output_prefix}_{day:%Y%m%d}.nc"
    save_daily_dataset(daily_ds, daily_path, config["compress_level"])

    upload_file(daily_path, config)
    upload_file(geo_path, config)
    return daily_path
