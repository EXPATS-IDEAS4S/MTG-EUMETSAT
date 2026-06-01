# MTG-EUMETSAT Satellite Data Processing

This repository provides tools to download, process and visualize Meteosat Third Generation (MTG) data, focusing on the FCI instrument and OCA cloud products (CTH / CMA). It produces daily NetCDF files in a structured layout compatible with MODIS-style consumers (lat/lon saved separately, main files with dimensions `time,x,y`).

## Highlights

- Three pipeline entrypoints:
  - `pipelines/process_normal_res_pipeline.py`: Download and process normal-resolution FCI channels, optionally apply parallax correction using OCA CTH products, save daily NetCDF and upload to configured bucket.
  - `pipelines/process_high_res_pipeline.py`: Download and process high-resolution FCI channels (simpler flow — no parallax, no cloud mask), save daily NetCDF and upload.
  - `pipelines/process_oca_pipeline.py`: Download and aggregate OCA products (CTH/CMA) into daily NetCDF files and upload.
- Centralized configuration in `user_config.py` controls credentials, ROI, channel lists, grid options, and upload targets.
- Shared utilities under `utils/` (regridding, I/O helpers, scene helpers, time utilities, plotting helpers).
- NetCDF layout: channel variables are stored in daily files with dims `(time, x, y)`; geolocation (`latitude`, `longitude`) is saved separately per channel when the original (native) grid is kept.

## Repository Structure (relevant files)

- `user_config.py`: Central configuration for all pipelines (credentials, ROI, channel lists, base paths, upload settings).
- `pipelines/`: Pipeline entrypoints and shared pipeline helpers:
  - `pipelines/process_normal_res_pipeline.py`
  - `pipelines/process_high_res_pipeline.py`
  - `pipelines/process_oca_pipeline.py`
  - `pipelines/pipeline_utils.py` (common helper functions)
- `download/`: Download helpers for MTG products and chunk handling (uses `eumdac`).
- `oca/`: OCA-specific processing utilities and the `process_oca_cth.py` script.
- `processing/`: Core processing code reused by pipelines (dataset_builder, process, run_pipeline, etc.).
- `utils/`: Shared helper modules (`utils/time_utils.py`, `utils/io_utils.py`, `utils/scene_utils.py`, `utils/grid_utils.py`, `utils/plot_utils.py`).

## Quickstart

1. Update `user_config.py` with your credentials and paths. At minimum set:

   - `consumer_key`, `consumer_secret` (or a `credentials_file` module)
   - `roi`, `roi_name`
   - `download_base`, `process_base`, `output_base`
   - `channels` (list of FCI channels to process)
   - `start_date`, `end_date`, `time_interval_min`, `timestamps_per_day`
   - `regular_grid` (bool), `grid_step_deg` (if regridding)
   - `parallax` (bool), and OCA-specific `cth_base` if using parallax
   - Upload settings: `upload_bucket`, `upload_profile` (if applicable)

2. Install dependencies:

```bash
pip install -r requirements.txt
# Optional/required extras: eumdac, satpy, geopandas, shapely
```

3. Run a pipeline (examples):

```bash
# Normal-resolution FCI pipeline (with optional parallax)
python pipelines/process_normal_res_pipeline.py

# High-resolution FCI pipeline (no parallax)
python pipelines/process_high_res_pipeline.py

# OCA pipeline (CTH/CMA aggregation)
python pipelines/process_oca_pipeline.py
```

Each pipeline will:

- Ensure required products are downloaded (downloaders will skip already-present files).
- Open MTG FCI files with Satpy, load the configured channels and geolocation.
- If `regular_grid` is set, regrid channels to the configured lat/lon mesh; otherwise keep native grid and save `latitude`/`longitude` separately (one file per channel, saved once).
- When `parallax` is enabled the normal-res pipeline will look for matching OCA/CTH files for each timestamp; if not present it will attempt to download them and run the OCA processor (`oca/process_oca_cth.py`) before applying parallax.
- Merge per-timestamp datasets into daily NetCDF files with compression and then upload them to the bucket configured in `user_config.py`.

## NetCDF layout notes

- Main daily files: variables for channels and CMA/CTH are stored with dims `(time, x, y)` or `(time, y, x)` depending on the pipeline; the files are compressed using zlib.
- Geolocation: when keeping native grids the pipeline saves geolocation per channel into a separate file named like `{channel}_original_coords.nc` (stored under the `process_base/<roi_name>/` folder). Consumers that need lat/lon arrays should open that file.

## Troubleshooting

- If imports fail at runtime, ensure optional packages are installed: `satpy`, `eumdac`, `geopandas`, `shapely`.
- The pipelines are intended to be run in an environment with network access to EUMETSAT APIs and write access to the configured output and temporary folders.

## Contact

For questions or help integrating the pipelines with downstream systems, open an issue or contact the maintainers.



