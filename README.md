# MTG-EUMETSAT Satellite Data Processing

This repository provides tools to process and visualize Meteosat Third Generation (MTG) satellite data, focusing on the FCI instrument and cloud-top height (CTH) products. It supports cropping, regridding, and parallax correction to generate daily NetCDF files and time-lapse visualizations.

## Features

* Modular processing pipeline for MTG FCI data
* Support for multiple spectral channels (e.g., `vis_06`, `ir_105`)
* Region of interest (ROI) cropping and optional regridding to regular lat/lon grids
* Parallax correction for improved cloud-top height alignment
* Daily output in NetCDF format with compression
* Visualization scripts for generating plots and animated GIFs

## Repository Structure

* `config.py`: Configuration parameters (paths, channels, ROI, time range, etc.)
* `process.py`: Main processing functions to create and save datasets
* `regrid_utils.py`: Functions for spatial interpolation and grid creation
* `time_utils.py`: Timestamp management and file time extraction utilities
* `plot_day.py`: Plotting and animation of daily processed data
* Supporting modules: `scene_utils.py`, `dataset_builder.py`, etc.

## Usage

1. Configure your settings in `config.py` (paths, dates, channels, etc.).
2. Run the processing pipeline:

   ```bash
   python process.py
   ```
3. Visualize daily results with:

   ```bash
   python plot_day.py
   ```

   This generates PNG images and an animated GIF per day and channel.

## Requirements

Python 3.8+ with:

* `xarray`
* `numpy`
* `scipy`
* `matplotlib`
* `cartopy`
* `imageio`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

* Parallax correction and CTH product processing are planned but not fully implemented.
* Regular grid output and deletion of intermediate chunks are configurable.

## Contact

For questions or collaboration, contact Daniele Corradini (dcorrad1@uni-koeln.de) or Claudia Acquistapace (cacquist@uni-koeln.de).


