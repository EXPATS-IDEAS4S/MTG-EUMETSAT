"""
Processing functions for handling individual timestamps and saving daily MTG datasets.

Includes:
- `process_timestamp`: orchestrates scene loading, cropping, variable/coordinate building, and Dataset assembly.
- `save_daily`: saves a daily xarray.Dataset to NetCDF with appropriate compression and naming.
"""

import xarray as xr
import pandas as pd

from scene_utils import make_scene, load_and_crop
from dataset_builder import build_data_vars, build_coords


def process_timestamp(ts, channel, msg_file, cth_file, config, grid=None, merge_coords=True):
    """
    Processes a single timestamp by:
    1. Creating a scene from MTG/CTH files.
    2. Cropping the region of interest.
    3. Building data variables (with optional regridding).
    4. Generating spatial and temporal coordinates.
    5. Assembling the final xarray.Dataset.

    Parameters:
        ts (datetime): Timestamp to process.
        channel (str): Satellite channel (e.g. 'ir_105').
        msg_file (str or Path): Path to MTG input file.
        cth_file (str or Path or None): Path to CTH file, if used.
        config (dict): Processing configuration dictionary.
        grid (tuple[np.ndarray, np.ndarray], optional): Regular grid if used.
        merge_coords (bool): Whether to include lat/lon coords in output Dataset.

    Returns:
        xr.Dataset: Final processed dataset for the timestamp.
    """
    scn = make_scene(msg_file, cth_file, config)
    cropped = load_and_crop(scn, [channel], config['roi'])
    data_vars = build_data_vars(cropped, channel, config, grid)
    coords = build_coords(cropped, channel, config, grid, ts)

    if merge_coords:
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
    else:
        ds = xr.Dataset(data_vars=data_vars)
        ds['time'] = (('time',), [ts])

    return ds


def save_daily(ds, day_ts, config, channel, suffix=None):
    """
    Saves the daily dataset to a NetCDF file with structured output path and compression.

    Parameters:
        ds (xr.Dataset): Dataset containing one or more time steps.
        day_ts (datetime or np.datetime64): Timestamp representing the day.
        config (dict): Processing configuration.
        channel (str): Channel name used in filename.
        suffix (str, optional): Suffix for the filename (e.g., "regrid_parallax").

    Returns:
        None
    """
    day_ts = pd.to_datetime(day_ts).to_pydatetime()
    outdir = config['output_base'] / f"{day_ts.year:04d}" / f"{day_ts.month:02d}"
    outdir.mkdir(parents=True, exist_ok=True)

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
        var: {'zlib': True, 'complevel': config['compress_level']}
        for var in ds.data_vars
    }
    enc['time'] = {'dtype': 'i4', 'units': 'seconds since 2000-01-01'}

    ds.to_netcdf(path, format='NETCDF4', encoding=enc)
