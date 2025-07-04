#!/usr/bin/env python3
"""
Improved MSG+CTH processing script with modular structure,
pathlib, logging, and efficient load/crop handling.

@authors: Daniele Corradini and Claudia Acquistapace

TODO: add CTH reader and parallax correction.
TODO: add regular grid option? Or save the original lat/lon coordinates as variables?
TODO: implement the delete_chunks option to remove the original chunks after processing?
"""
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from satpy import Scene
from typing import Sequence
from collections import defaultdict
from typing import Callable, Iterable, Dict, List
import pandas as pd

from old.regrid_functions import regrid_data, fill_missing_data_with_interpolation


# --- Configuration ---
CONFIG = {\
    'output_base': Path('/data/sat/msg/mtg/fci/'),
    'mtg_base': Path('/data/trade_pc/mtg/fci/'),
    'cth_base': None,
    'file_extension': '*.nc',
    'mtg_reader': 'fci_l1c_nc',
    'cth_reader': None,  # TODO still to be defined
    'channels': ['vis_06', 'ir_105'],
    'roi': {'lon_min': 5, 'lat_min': 42, 'lon_max': 16, 'lat_max': 52},
    'parallax': False,
    'time_interval_min': 10,
    'start_date': '2025.06.29',
    'end_date': '2025.07.02', #last timestamp exlcuded
    'regular_grid': False,
    'interp_method': 'nearest',  # Interpolation method for regridding
    'grid_step_deg': [(0.015,0.010), (0.0075,0.005)],  # Regular grid step in degrees
    'compress_level': 9,
    'delete_chunks': False, #implemnt it here or in a another scirpt?
}

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# --- Helper functions ---
def compute_timestamps(start_date: str, end_date: str, step: int):
    start = datetime.strptime(start_date, '%Y.%m.%d')
    end = datetime.strptime(end_date, '%Y.%m.%d')
    ts = []
    while start <= end:
        ts.append(start)
        start += timedelta(minutes=step)
    return ts


def list_mtg_files(base: Path,
                   timestamps: Sequence[datetime],
                   pattern: str = "*.nat") -> list[Path]:
    """
    For each unique date in `timestamps`, looks under
      base/YYYY/MM/DD
    and returns all files matching `pattern`, sorted.
    """
    paths = []
    # Extract the unique date strings we need
    dates = {ts.date() for ts in timestamps}
    for d in sorted(dates):
        subdir = base / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}"
        if subdir.is_dir():
            paths.extend(subdir.rglob(pattern))
    return sorted(paths)

def list_cth_files(base: Path, pattern: str = "*.nc") -> list[Path]:
    """
    Lists all CTH files in the given base directory matching the pattern.
    TODO: complete this based on the actual CTH file structure.
    """
    #retunr empty list if base is None
    if base is None:
        return []
    if not base.is_dir():
        raise ValueError(f"CTH base path {base} is not a directory.")
    return sorted(base.rglob(pattern))


def extract_mtg_time(fname: Path, interval: int = 10) -> datetime:
    """
    Extracts the start timestamp from an MTG FCI filename and rounds it
    down to the nearest `interval` minutes.

    Assumes the filename is of the form:
        ..._<beginYYYYMMDDHHMMSS>_<endYYYYMMDDHHMMSS>_...
    where splitting on '_' gives the begin timestamp at index 8.

    Args:
        fname: Path to the MTG chunk file.
        interval: Minute‑rounding interval (e.g. 10, 5, 1).

    Returns:
        A datetime object corresponding to the begin time, with:
          - seconds set to zero
          - minutes rounded down to nearest `interval`
    """
    parts = fname.name.split('_')
    if len(parts) <= 8:
        raise ValueError(f"Unexpected MTG filename format: {fname.name}")

    # parts[8] is the “begin” tag, e.g. "20250617083000"
    begin_tag = parts[8]
    dt = datetime.strptime(begin_tag, "%Y%m%d%H%M%S")

    # round minutes down to nearest `interval`
    rounded_min = dt.minute - (dt.minute % interval)
    return dt.replace(minute=rounded_min, second=0, microsecond=0)


def extract_cth_time(fname: Path):
    tstr = fname.name.split('.')[0][5:17]
    return datetime.strptime(tstr, '%Y%m%d%H%M')


def build_time_map(
    files: Iterable[Path],
    time_extractor: Callable[[Path], datetime],
    ) -> Dict[datetime, List[str]]:
    """
    Groups files by the datetime returned from time_extractor.
    Converts each Path to str.

    Args:
        files: An iterable of Path objects.
        time_extractor: A function that maps a Path -> datetime.

    Returns:
        A dict mapping each datetime to a list of file paths (as str).
        Returns an empty dict if no files are provided.
    """
    if not files:
        return {}

    grouped = defaultdict(list)
    for f in files:
        t = time_extractor(f)
        grouped[t].append(str(f))  # Convert Path to str here

    return dict(grouped)


def load_and_crop(scene: Scene, channels, roi):
    scene.load(channels)
    crop = scene.crop(ll_bbox=(roi['lon_min'], roi['lat_min'], roi['lon_max'], roi['lat_max']))
    return crop


def make_regular_grid(roi, steps):
    grids = []
    for step in steps:
        lats = np.arange(roi['lat_min'], roi['lat_max'] + step[0], step[0])
        lons = np.arange(roi['lon_min'], roi['lon_max'] + step[1], step[1])
        grids.append(np.meshgrid(lats, lons, indexing='ij'))

    return grids


def make_scene(msg_file, cth_file, config):
    """Return a Satpy Scene, with or without CTH for parallax correction."""
    if config['parallax']:
        return Scene({
            config['mtg_reader']: [str(msg_file)],
            config['cth_reader']: [str(cth_file)]
        })
    else:
        return Scene(reader=config['mtg_reader'], filenames=msg_file)



def load_and_crop(scene, channel, roi):
    """
    Load all `channels` into `scene` and crop to the lon/lat box in `roi`.
    Returns the cropped Scene.
    """
    scene.load([channel])
    ll_bbox = (
        roi['lon_min'], roi['lat_min'],
        roi['lon_max'], roi['lat_max']
    )
    return scene.crop(ll_bbox=ll_bbox)


def preprocess_array(arr, lat_src, lon_src, target_grid, method):
    """
    - Fill NaNs via interpolation
    - Regrid from (lat_src, lon_src) to target_grid = (lat2d, lon2d)
    """
    #print(lat_src.shape, lon_src.shape, arr.shape)
    arr_filled = fill_missing_data_with_interpolation(lat_src, lon_src, arr)
    lat2d, lon2d = target_grid
    #print(target_grid)
    #print(lat2d.shape, lon2d.shape, arr_filled.shape)
    return regrid_data(lat_src, lon_src, arr_filled, lat2d, lon2d, method)



def build_data_vars(crop, ch, config, grid):
    """
    Build data variables per channel. If regular grid: regrid.
    If irregular: keep native projection, add lat/lon as variables separately.
    """
    data_vars = {}
    is_regular = config.get('regular_grid', False)

    
    raw = crop[ch].values.astype(np.float32)

    if is_regular:
        lat_src, lon_src = crop[ch].attrs['area'].get_lonlats()
        raw = preprocess_array(
            raw, lat_src, lon_src,
            target_grid=grid,
            method=config['interp_method']
        )
        data_vars[ch] = (('lat', 'lon'), raw)
    else:
        data_vars[ch] = (('y', 'x'), raw)

    return data_vars



def build_coords(crop, ch, config, grid, ts):
    """
    If regridded to regular lat/lon: return 1D lat/lon coords.
    Else: return 2D lat/lon as data variables, and x/y as indices.
    """
    is_regular = config.get('regular_grid', False)

    if is_regular:
        lat2d, lon2d = grid
        lat1d = lat2d[:, 0]
        lon1d = lon2d[0, :]
        coords = {
            'lat': (('lat',), lat1d.astype(np.float32)),
            'lon': (('lon',), lon1d.astype(np.float32)),
            'time': (('time',), [ts])
        }
    else:
        lon2d, lat2d = crop[ch].attrs['area'].get_lonlats()
        ny, nx = lat2d.shape

        # Coordinates as x/y indices only
        coords = {
            'x': (('x',), np.arange(nx, dtype=np.int32)),
            'y': (('y',), np.arange(ny, dtype=np.int32)),
            # 'time': (('time',), [ts])  # Add this if needed
        }

        # Data variables including latitude and longitude as 2D arrays
        data_vars = {
            'latitude': (('y', 'x'), lat2d.astype(np.float32)),
            'longitude': (('y', 'x'), lon2d.astype(np.float32)),
            # Add your satellite channels here as well, like:
            # ch: (('y', 'x'), crop[ch].values.astype(np.float32))
        }

        # Then create your dataset
        coords = xr.Dataset(data_vars=data_vars, coords=coords)


    return coords



def process_timestamp(ts, channel, msg_file, cth_file, config, grid=None, merge_coords=True):
    """
    High‑level function that glues together all steps:
      1) Scene creation
      2) Load & crop
      3) Data var construction (with optional fill & regrid)
      4) Coord building
      5) Assembly into xarray.Dataset
    """
    # 1) Scene
    scn = make_scene(msg_file, cth_file, config)

    # 2) Load & crop channels
    cropped = load_and_crop(scn, channel, config['roi'])

    # 3) Build data variables
    data_vars = build_data_vars(cropped, channel, config, grid)
    #print(data_vars)

    # 4) Build coords
    coords = build_coords(cropped, channel, config, grid, ts)
    #print(coords)

    # 5) Assemble and return
    if merge_coords:
        # Merge data_vars and coords into a single dict
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
    else:
        #merge only the time
        ds = xr.Dataset(data_vars=data_vars)
        ds['time'] = (('time',), [ts])
        
    return ds


def save_daily(ds, day_ts, config, channel, suffix=None):
    # Convert numpy.datetime64 to Python datetime
    day_ts = pd.to_datetime(day_ts).to_pydatetime()
    outdir = config['output_base'] / f"{day_ts.year:04d}" / f"{day_ts.month:02d}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Dynamically build suffix based on config flags
    if suffix is None:
        suffix_parts = []
        if config.get('regular_grid'):
            suffix_parts.append("regrid")
        if config.get('parallax'):
            suffix_parts.append("parallax")
        suffix = "_".join(suffix_parts) if suffix_parts else "raw"

    fname = f"{channel}_{day_ts:%Y%m%d}_{suffix}.nc"
    path = outdir / fname

    enc = {
    var: {'zlib': True, 'complevel': config['compress_level']} for var in ds.data_vars}
    enc['time'] = {'dtype': 'i4', 'units': 'seconds since 2000-01-01'}

    ds.to_netcdf(path, format='NETCDF4', encoding=enc)
    log.info(f"Saved {path}")


def create_nan_dataset(ts, channel, config, grid, regular):
    """
    Creates a dummy xarray Dataset filled with NaNs for a given timestamp and channel.
    Uses build_coords() to ensure consistent coordinate creation.
    """
    # If not regular grid, we need a dummy "crop" input with area attribute
    if not regular:
        # You can either load one dummy scene, or pass one from outside
        # Here, we use the stored original coordinates if available
        coord_path = config['output_base'] / f"{channel}_original_coords.nc"
        if not coord_path.exists():
            raise FileNotFoundError(f"Coordinate reference file not found: {coord_path}")
        coords_ds = xr.load_dataset(coord_path)
        crop = {
            channel: xr.DataArray(
                np.zeros_like(coords_ds['latitude'].values, dtype=np.float32),
                attrs={'area': type('area', (), {
                    'get_lonlats': lambda: (coords_ds['longitude'].values, coords_ds['latitude'].values)
                })()}
            )
        }
    else:
        crop = None

    # Build coords using your existing function
    coords = build_coords(crop, channel, config, grid, ts)

    # Determine shape from coordinates
    if isinstance(coords, xr.Dataset):  # not regular
        shape = coords['latitude'].shape
        dims = ('y', 'x')
    else:  # regular grid
        shape = (len(coords['lat']), len(coords['lon']))
        dims = ('lat', 'lon')

    # Create DataArray filled with NaNs
    data = xr.DataArray(np.full(shape, np.nan, dtype=np.float32),
                        dims=dims,
                        coords=coords.coords if isinstance(coords, xr.Dataset) else coords)

    # Wrap in a Dataset and assign time dimension
    ds = xr.Dataset({channel: data.expand_dims('time')})
    ds = ds.assign_coords(time=('time', [ts]))

    return ds




# --- Main workflow ---
def main():
    cfg = CONFIG
    start = cfg["start_date"]
    end   = cfg["end_date"]

    # build your list of timestamps
    timestamps = compute_timestamps(start, end, cfg["time_interval_min"])[:-1]
    #print(timestamps)

    # now only look under the yyyy/mm/dd folders for those dates
    mtg_files = list_mtg_files(cfg["mtg_base"], timestamps, pattern=cfg['file_extension'])
    print(f"Found {len(mtg_files)} MTG files.")
 
    cth_files = list_cth_files(cfg['cth_base'], '*.nc')
    mtg_map = build_time_map(mtg_files, lambda f: extract_mtg_time(f, cfg['time_interval_min']))

    cth_map = build_time_map(cth_files, extract_cth_time)
    grids = make_regular_grid(cfg['roi'], cfg['grid_step_deg']) if cfg['regular_grid'] else [None]*len(cfg['channels'])

    # Initialize tracking per channel
    daily_ds_per_channel = {ch: None for ch in cfg['channels']}
    last_day_per_channel = {ch: None for ch in cfg['channels']}

    for idx, ts in enumerate(timestamps):
        mtg_f = mtg_map.get(ts)
        cth_f = cth_map.get(ts)
        print(mtg_f, cth_f)

        missing = not mtg_f or (cfg['parallax'] and not cth_f)

        for channel, grid in zip(cfg['channels'], grids):
            print(f"Processing {ts} for channel {channel}")

            if missing:
                log.warning(f"Missing data for {ts}, creating NaN dataset.")
                ds = create_nan_dataset(ts, channel, cfg, grid, cfg['regular_grid'])
            else:
                # Save original lat/lon coords once if not regular grid
                if not cfg['regular_grid'] and idx == 0:
                    scn = make_scene(mtg_f, cth_f, cfg)
                    crop = load_and_crop(scn, channel, cfg['roi'])
                    ds_coords = build_coords(crop, channel, cfg, grid, ts)
                    ds_coords.to_netcdf(cfg['output_base'] / f"{channel}_original_coords.nc", format='NETCDF4')
                    print(f"Saved original coords for {channel} at {ts}")

                ds = process_timestamp(ts, channel, mtg_f, cth_f, cfg, grid, cfg['regular_grid'])

            # Daily concat logic (same as before)
            day = ts.day
            last_day = last_day_per_channel[channel]
            if last_day != day and daily_ds_per_channel[channel] is not None:
                save_daily(daily_ds_per_channel[channel], daily_ds_per_channel[channel].time.values[0], cfg, channel)
                daily_ds_per_channel[channel] = ds
            else:
                daily_ds_per_channel[channel] = (
                    xr.concat([daily_ds_per_channel[channel], ds], dim='time')
                    if daily_ds_per_channel[channel] is not None else ds
                )
            last_day_per_channel[channel] = day


    # Final flush
    for channel, ds_day in daily_ds_per_channel.items():
        if ds_day is not None:
            save_daily(ds_day, ds_day.time.values[0], cfg, channel)

if __name__ == '__main__':
    main()

#995179 nohup