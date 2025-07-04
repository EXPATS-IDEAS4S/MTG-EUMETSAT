import xarray as xr
import pandas as pd

from scene_utils import make_scene, load_and_crop
from dataset_builder import build_data_vars, build_coords

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
    cropped = load_and_crop(scn, [channel], config['roi'])

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
