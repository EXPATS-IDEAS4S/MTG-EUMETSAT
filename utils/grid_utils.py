"""
Grid utility functions for interpolation, regridding, and generating regular latitude/longitude meshes.

Copied from `processing/grid_utils.py` to provide a central utilities package.
"""

import numpy as np
from scipy.interpolate import griddata


def regrid_data(old_lat, old_lon, old_data, new_lat, new_lon, method='linear'):
    if np.all(np.isnan(old_data)):
        return np.full(new_lat.shape, np.nan)

    old_coords = np.array([old_lat.ravel(), old_lon.ravel()]).T
    old_data_flat = old_data.ravel()
    new_coords = np.array([new_lat.ravel(), new_lon.ravel()]).T

    new_data_flat = griddata(old_coords, old_data_flat, new_coords, method=method)
    return new_data_flat.reshape(new_lat.shape)


def fill_missing_data_with_interpolation(lat, lon, data, method='linear'):
    valid_mask = ~np.isnan(data)
    if not valid_mask.any():
        return np.full(data.shape, np.nan)

    valid_data = data[valid_mask]
    valid_lat = lat[valid_mask]
    valid_lon = lon[valid_mask]

    return griddata((valid_lat, valid_lon), valid_data, (lat, lon), method=method)


def make_regular_grid(roi, steps):
    grids = []
    for step in steps:
        lats = np.arange(roi['lat_min'], roi['lat_max'] + step[0], step[0])
        lons = np.arange(roi['lon_min'], roi['lon_max'] + step[1], step[1])
        grids.append(np.meshgrid(lats, lons, indexing='ij'))
    return grids
