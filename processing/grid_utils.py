"""
Grid utility functions for interpolation, regridding, and generating regular latitude/longitude meshes.

Includes:
- Interpolation of NaN values in geospatial grids.
- Regridding of 2D data between lat/lon meshes.
- Creation of regular lat/lon meshgrids over a specified ROI.
"""

import numpy as np
from scipy.interpolate import griddata


def regrid_data(old_lat, old_lon, old_data, new_lat, new_lon, method='linear'):
    """
    Regrids 2D data from a source lat/lon grid to a new target grid.

    Parameters:
        old_lat (np.ndarray): 2D array of source latitudes.
        old_lon (np.ndarray): 2D array of source longitudes.
        old_data (np.ndarray): 2D data array to regrid.
        new_lat (np.ndarray): 2D array of target latitudes.
        new_lon (np.ndarray): 2D array of target longitudes.
        method (str): Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        np.ndarray: Regridded 2D data matching the shape of `new_lat`/`new_lon`.
                    Filled with NaNs if `old_data` contains only NaNs.
    """
    if np.all(np.isnan(old_data)):
        return np.full(new_lat.shape, np.nan)

    old_coords = np.array([old_lat.ravel(), old_lon.ravel()]).T
    old_data_flat = old_data.ravel()
    new_coords = np.array([new_lat.ravel(), new_lon.ravel()]).T

    new_data_flat = griddata(old_coords, old_data_flat, new_coords, method=method)
    return new_data_flat.reshape(new_lat.shape)


def fill_missing_data_with_interpolation(lat, lon, data, method='linear'):
    """
    Fills NaN values in 2D data using spatial interpolation based on lat/lon.

    Parameters:
        lat (np.ndarray): 2D array of latitudes.
        lon (np.ndarray): 2D array of longitudes.
        data (np.ndarray): 2D data array with possible NaNs.
        method (str): Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        np.ndarray: Interpolated 2D array with NaNs filled, or all-NaN array if no valid data.
    """
    valid_mask = ~np.isnan(data)
    if not valid_mask.any():
        return np.full(data.shape, np.nan)

    valid_data = data[valid_mask]
    valid_lat = lat[valid_mask]
    valid_lon = lon[valid_mask]

    return griddata((valid_lat, valid_lon), valid_data, (lat, lon), method=method)


def make_regular_grid(roi, steps):
    """
    Generates regular lat/lon 2D meshgrids over a region of interest (ROI).

    Parameters:
        roi (dict): Bounding box with keys 'lat_min', 'lat_max', 'lon_min', 'lon_max'.
        steps (list of tuple): List of (lat_step, lon_step) tuples for grid resolution.

    Returns:
        list of tuple: List of (lat2d, lon2d) grids for each resolution step.
    """
    grids = []
    for step in steps:
        lats = np.arange(roi['lat_min'], roi['lat_max'] + step[0], step[0])
        lons = np.arange(roi['lon_min'], roi['lon_max'] + step[1], step[1])
        grids.append(np.meshgrid(lats, lons, indexing='ij'))
    return grids

