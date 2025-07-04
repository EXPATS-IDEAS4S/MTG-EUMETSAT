import numpy as np
import xarray as xr

from grid_utils import regrid_data, fill_missing_data_with_interpolation


def preprocess_array(arr, lat_src, lon_src, target_grid, method):
    """
    - Fill NaNs via interpolation
    - Regrid from (lat_src, lon_src) to target_grid = (lat2d, lon2d)
    """
    #print(lat_src.shape, lon_src.shape, arr.shape)
    arr_filled = fill_missing_data_with_interpolation(lat_src, lon_src, arr)
    lat2d, lon2d = target_grid
    #print(target_grid)
    #print(lat2d.shape, lon2d.shape, arr_filled.shape)
    return regrid_data(lat_src, lon_src, arr_filled, lat2d, lon2d, method)


def build_data_vars(crop, ch, config, grid):
    """
    Build data variables per channel. If regular grid: regrid.
    If irregular: keep native projection, add lat/lon as variables separately.
    """
    data_vars = {}
    is_regular = config.get('regular_grid', False)

    
    raw = crop[ch].values.astype(np.float32)

    if is_regular:
        lat_src, lon_src = crop[ch].attrs['area'].get_lonlats()
        raw = preprocess_array(
            raw, lat_src, lon_src,
            target_grid=grid,
            method=config['interp_method']
        )
        data_vars[ch] = (('lat', 'lon'), raw)
    else:
        data_vars[ch] = (('y', 'x'), raw)

    return data_vars




def build_coords(crop, ch, config, grid, ts):
    """
    If regridded to regular lat/lon: return 1D lat/lon coords.
    Else: return 2D lat/lon as data variables, and x/y as indices.
    """
    is_regular = config.get('regular_grid', False)

    if is_regular:
        lat2d, lon2d = grid
        lat1d = lat2d[:, 0]
        lon1d = lon2d[0, :]
        coords = {
            'lat': (('lat',), lat1d.astype(np.float32)),
            'lon': (('lon',), lon1d.astype(np.float32)),
            'time': (('time',), [ts])
        }
    else:
        lon2d, lat2d = crop[ch].attrs['area'].get_lonlats()
        ny, nx = lat2d.shape

        # Coordinates as x/y indices only
        coords = {
            'x': (('x',), np.arange(nx, dtype=np.int32)),
            'y': (('y',), np.arange(ny, dtype=np.int32)),
            # 'time': (('time',), [ts])  # Add this if needed
        }

        # Data variables including latitude and longitude as 2D arrays
        data_vars = {
            'latitude': (('y', 'x'), lat2d.astype(np.float32)),
            'longitude': (('y', 'x'), lon2d.astype(np.float32)),
            # Add your satellite channels here as well, like:
            # ch: (('y', 'x'), crop[ch].values.astype(np.float32))
        }

        # Then create your dataset
        coords = xr.Dataset(data_vars=data_vars, coords=coords)


    return coords



def create_nan_dataset(ts, channel, config, grid):
    """
    Creates a dummy xarray Dataset filled with NaNs for a given timestamp and channel.
    Uses build_coords() to ensure consistent coordinate creation.
    """
    # If not regular grid, we need a dummy "crop" input with area attribute
    regular = config.get('regular_grid', False)
    if not regular:
        # You can either load one dummy scene, or pass one from outside
        # Here, we use the stored original coordinates if available
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

    # Build coords using your existing function
    coords = build_coords(crop, channel, config, grid, ts)

    # Determine shape from coordinates
    if isinstance(coords, xr.Dataset):  # not regular
        shape = coords['latitude'].shape
        dims = ('y', 'x')
    else:  # regular grid
        shape = (len(coords['lat']), len(coords['lon']))
        dims = ('lat', 'lon')

    # Create DataArray filled with NaNs
    data = xr.DataArray(np.full(shape, np.nan, dtype=np.float32),
                        dims=dims,
                        coords=coords.coords if isinstance(coords, xr.Dataset) else coords)

    # Wrap in a Dataset and assign time dimension
    ds = xr.Dataset({channel: data.expand_dims('time')})
    ds = ds.assign_coords(time=('time', [ts]))

    return ds