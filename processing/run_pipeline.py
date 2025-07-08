#!/usr/bin/env python3
"""
Satellite Data Processing Pipeline for MTG FCI satellite imagery

This script  processes Meteosat Third Generation (MTG) 
data and it uses Cloud Top Height (CTH) products for the parallax correction. 

Main features:
- Generates timestamped xarray Datasets from MTG and CTH input
- Handles missing data by inserting NaN-filled placeholders
- Supports regular and native grid processing
- Crops scenes to a user-defined region of interest (ROI)
- Performs optional parallax correction
- Groups output into daily NetCDF files
- Saves original geolocation (lat/lon) once per channel for non-regular grids

Authors:
    Daniele Corradini
    Claudia Acquistapace

TODO:
    - Implement full CTH reader integration and parallax correction
    - Consider saving original lat/lon as regular variables even with regridding
    - Add option to delete original chunked output post-processing
"""

import logging
from pathlib import Path
import xarray as xr

# --- External config ---
from config import CONFIG as cfg

# --- Internal modules ---
from time_utils import compute_timestamps,  extract_mtg_time, extract_cth_time
from io_utils import list_mtg_files, list_cth_files, build_time_map
from scene_utils import make_scene, load_and_crop, has_corrupted_files
from dataset_builder import build_coords, create_nan_dataset
from process import process_timestamp, save_daily
from grid_utils import make_regular_grid

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)


# --- Main workflow ---
def main():
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

        #check if files are corrupted

        corrupted = has_corrupted_files(mtg_f)
        missing = not mtg_f or corrupted  #or (cfg['parallax'] and not cth_f)

        for channel, grid in zip(cfg['channels'], grids):
            print(f"Processing {ts} for channel {channel}")

            if missing:
                log.warning(f"Missing or corrupted data for {ts}, creating NaN dataset.")
                ds = create_nan_dataset(ts, channel, cfg, grid)
            else:
                # Save original lat/lon coords once if not regular grid
                if not cfg['regular_grid'] and idx == 0:
                    out_path = cfg['output_base'] / f"{channel}_original_coords.nc"

                    if not out_path.exists():
                        scn = make_scene(mtg_f, cth_f, cfg)
                        crop = load_and_crop(scn, [channel], cfg['roi'])
                        ds_coords = build_coords(crop, channel, cfg, grid, ts)

                        # Create output folder if it does not exist
                        cfg['output_base'].mkdir(parents=True, exist_ok=True)

                        ds_coords.to_netcdf(out_path, format='NETCDF4')
                        print(f"Saved original coords for {channel} at {ts}")
                    else:
                        print(f"Original coords for {channel} already exist at {out_path}, skipping.")

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
    
    log.info("Processing complete.")

if __name__ == '__main__':
    main()


#2114797 nohup