import numpy as np
from scipy.interpolate import griddata


def regrid_data(old_lat, old_lon, old_data, new_lat, new_lon, method='linear'):
    """
    Regrid data from an old grid to a new grid. If the old data contains only NaN values,
    returns a NaN-filled array matching the shape of the new grid.

    :param old_lat: 2D array of latitudes for the old grid.
    :param old_lon: 2D array of longitudes for the old grid.
    :param old_data: 2D array of data corresponding to the old grid.
    :param new_lat: 2D array of latitudes for the new grid.
    :param new_lon: 2D array of longitudes for the new grid.
    :param method: Interpolation method ('linear', 'nearest', 'cubic').
    :return: 2D array of regridded data corresponding to the new grid.
    """
    # Check if all data points in the old_data are NaN
    if np.all(np.isnan(old_data)):
        # Return a NaN-filled array with the same shape as the new grid
        return np.full(new_lat.shape, np.nan)

    # Flatten the old grid coordinates and data for interpolation
    old_coords = np.array([old_lat.ravel(), old_lon.ravel()]).T
    old_data_flat = old_data.ravel()

    # Create a mesh of new grid coordinates
    new_coords = np.array([new_lat.ravel(), new_lon.ravel()]).T

    # Interpolate old data to new grid using the specified method
    new_data_flat = griddata(old_coords, old_data_flat, new_coords, method=method)

    # Reshape the flattened data back into the 2D structure of the new grid
    new_data = new_data_flat.reshape(new_lat.shape)

    return new_data


def fill_missing_data_with_interpolation(lat, lon, data, method='linear'):
    """
    Fill missing data (NaN) with interpolation based on nearby values. If all data are NaN,
    returns an array of the same shape filled with NaN.

    :param lat: 2D array of latitudes.
    :param lon: 2D array of longitudes.
    :param data: 2D array of data with NaN values for missing data.
    :param method: Interpolation method ('linear', 'nearest', 'cubic').
    :return: 2D array with missing data filled or all NaN if no valid data points exist.
    """
    # Mask to identify valid (non-NaN) data points
    valid_mask = ~np.isnan(data)
    # Check if there are any valid data points
    if not valid_mask.any():
        # Return an array of NaNs with the same shape as the input data if no valid data points
        return np.full(data.shape, np.nan)

    # Coordinates and data of valid points
    valid_data = data[valid_mask]
    valid_lat = lat[valid_mask]
    valid_lon = lon[valid_mask]

    # Linear interpolation for missing data
    filled_data = griddata((valid_lat, valid_lon), valid_data, (lat, lon), method=method)

    return filled_data


def make_regular_grid(roi, steps):
    grids = []
    for step in steps:
        lats = np.arange(roi['lat_min'], roi['lat_max'] + step[0], step[0])
        lons = np.arange(roi['lon_min'], roi['lon_max'] + step[1], step[1])
        grids.append(np.meshgrid(lats, lons, indexing='ij'))

    return grids



