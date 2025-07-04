"""
Utilities for building xarray Datasets from satellite scene crops.

Includes:
- Regridding and interpolation of missing data
- Construction of data variables and spatial/temporal coordinates
- Creation of NaN-filled placeholder datasets for missing scenes
"""

import numpy as np
import xarray as xr

from grid_utils import regrid_data, fill_missing_data_with_interpolation


def preprocess_array(arr, lat_src, lon_src, target_grid, method):
    """
    Interpolates missing values and regrids data to the target lat/lon grid.

    Parameters:
        arr (np.ndarray): Input 2D array with possible NaNs.
        lat_src (np.ndarray): 2D array of source latitudes.
        lon_src (np.ndarray): 2D array of source longitudes.
        target_grid (tuple): Target (lat2d, lon2d) grid for interpolation.
        method (str): Interpolation method (e.g., 'nearest').

    Returns:
        np.ndarray: Interpolated and regridded array.
    """
    arr_filled = fill_missing_data_with_interpolation(lat_src, lon_src, arr)
    lat2d, lon2d = target_grid
    return regrid_data(lat_src, lon_src, arr_filled, lat2d, lon2d, method)


def build_data_vars(crop, ch, config, grid):
    """
    Constructs the data variables for a given channel.

    Parameters:
        crop (dict): Dictionary of cropped scene DataArrays.
        ch (str): Channel name.
        config (dict): Configuration dictionary.
        grid (tuple or None): Regular grid if applicable.

    Returns:
        dict: Dictionary of (dims, values) for xarray.Dataset.
    """
    data_vars = {}
    is_regular = config.get('regular_grid', False)
    raw = crop[ch].values.astype(np.float32)

    if is_regular:
        lat_src, lon_src = crop[ch].attrs['area'].get_lonlats()
        raw = preprocess_array(raw, lat_src, lon_src, grid, config['interp_method'])
        data_vars[ch] = (('lat', 'lon'), raw)
    else:
        data_vars[ch] = (('y', 'x'), raw)

    return data_vars


def build_coords(crop, ch, config, grid, ts):
    """
    Builds spatial and temporal coordinates for the dataset.

    Parameters:
        crop (dict): Dictionary with cropped DataArray(s).
        ch (str): Channel name.
        config (dict): Configuration dictionary.
        grid (tuple or None): Regular grid if used.
        ts (datetime): Timestamp for the data.

    Returns:
        dict or xr.Dataset: Coordinates for xarray.Dataset.
    """
    is_regular = config.get('regular_grid', False)

    if is_regular:
        lat2d, lon2d = grid
        lat1d = lat2d[:, 0]
        lon1d = lon2d[0, :]
        return {
            'lat': (('lat',), lat1d.astype(np.float32)),
            'lon': (('lon',), lon1d.astype(np.float32)),
            'time': (('time',), [ts])
        }
    else:
        lon2d, lat2d = crop[ch].attrs['area'].get_lonlats()
        ny, nx = lat2d.shape
        coords = {
            'x': (('x',), np.arange(nx, dtype=np.int32)),
            'y': (('y',), np.arange(ny, dtype=np.int32))
        }
        data_vars = {
            'latitude': (('y', 'x'), lat2d.astype(np.float32)),
            'longitude': (('y', 'x'), lon2d.astype(np.float32))
        }
        return xr.Dataset(data_vars=data_vars, coords=coords)


def create_nan_dataset(ts, channel, config, grid):
    """
    Creates a placeholder xarray.Dataset filled with NaNs for a given timestamp.

    Parameters:
        ts (datetime): Timestamp for the dataset.
        channel (str): Channel name.
        config (dict): Configuration dictionary.
        grid (tuple or None): Grid to define dataset shape.

    Returns:
        xr.Dataset: Dataset with NaN data for fallback cases.
    """
    regular = config.get('regular_grid', False)

    if not regular:
        coord_path = config['output_base'] / f"{channel}_original_coords.nc"
        if not coord_path.exists():
            raise FileNotFoundError(f"Coordinate reference file not found: {coord_path}")
        coords_ds = xr.load_dataset(coord_path)

        class MockArea:
            def get_lonlats(self):
                return coords_ds['longitude'].values, coords_ds['latitude'].values

        crop = {
            channel: xr.DataArray(
                np.zeros_like(coords_ds['latitude'].values, dtype=np.float32),
                attrs={'area': MockArea()}
            )
        }
    else:
        crop = None

    coords = build_coords(crop, channel, config, grid, ts)

    if isinstance(coords, xr.Dataset):
        shape = coords['latitude'].shape
        dims = ('y', 'x')
    else:
        shape = (len(coords['lat']), len(coords['lon']))
        dims = ('lat', 'lon')

    data = xr.DataArray(np.full(shape, np.nan, dtype=np.float32),
                        dims=dims,
                        coords=coords.coords if isinstance(coords, xr.Dataset) else coords)

    ds = xr.Dataset({channel: data.expand_dims('time')})
    ds = ds.assign_coords(time=('time', [ts]))
    return ds
