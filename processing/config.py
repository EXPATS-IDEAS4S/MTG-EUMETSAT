"""
Configuration dictionary for MTG processing pipeline.

This dictionary centralizes all configurable parameters used in the main processing script,
including file paths, temporal settings, selected channels, ROI, and processing options.
"""

from pathlib import Path

CONFIG = {
    "output_base": Path("/home/Daniele/data/data_for_parallax/processed/non_parallax_corrected_mtg"),  # Output directory for processed datasets
    "mtg_base": Path("/home/Daniele/data/data_for_parallax/mtg/"),               # Base directory containing raw MTG FCI input files
    "cth_base": Path("/home/Daniele/data/data_for_parallax/oca/"),                                          # Optional: directory containing Cloud Top Height files (set to None if unused)
    "file_extension": "*.nc",                                  # File extension/pattern for MTG files to process
    "mtg_reader": "fci_l1c_nc",                                # Reader module/method for MTG files
    "cth_reader": "fci_l2_nc",                                        # Optional: reader module for CTH files (if applicable)
    "save_timestamps": True,                                    # Whether to save single timestamps in the output dataset
    "channels": ["ir_105"],                          # List of MTG channels to process
    "roi": {                                                   # Region of interest for cropping the data
        "lon_min": 5,
        "lat_min": 42,
        "lon_max": 16,
        "lat_max": 52
    },

    "parallax": False,                                         # Enable parallax correction using CTH data
    "time_interval_min": 10,                                   # Time step (in minutes) between successive files
    "start_date": "2025.05.22",                                # Start date (YYYY.MM.DD) for processing period
    "end_date": "2025.05.23",                                  # End date (YYYY.MM.DD) for processing period (last date excluded)

    "regular_grid": False,                                     # Whether to regrid data to a regular lat/lon grid
    "interp_method": "nearest",                                # Interpolation method for regridding (if enabled)
    "grid_step_deg": [(0.015, 0.010), (0.0075, 0.005)],        # Grid step size (lon, lat) per channel when using regular grid

    "compress_level": 9,                                       # Compression level for NetCDF output files
    "delete_chunks": False                                     # If True, delete intermediate chunked files after final output
}
