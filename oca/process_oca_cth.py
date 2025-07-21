"""
Processing CTH data from CMSAF to make it compatible to Satpy for the parallax correction

@author: Daniele Corradini
"""

from glob import glob
import xarray as xr
import os
import numpy as np
from satpy import Scene

### Define Paths ###

year = '2025'
month = '05'

path_to_data = f"/data/sat/mtg/fci/oca/netcdf/{year}/{month}/*/"

out_dir = f"/data/sat/mtg/fci/oca/processing_for_parallax/{year}/{month}/"

variable = 'retrieved_cloud_top_height'  # Variable to process

#area extent from mtg files (in meters)
#area_extent_meters = (323999.9999, 4651999.9988, 1239999.9997, 4023999.999) #xmin,ymin,xmax,ymax (ymin > ymax because of the inverted Y-axis in satellite image coordinates. 
#x_min, y_min, x_max, y_max = area_extent_meters
#h_sat = 35786400
# convert to radinace

lonmin, latmin, lonmax, latmax = 5, 42, 16, 52 #EXPATS

#open all files in directory 
nc_file ='W_XX*.nc' 

fnames = sorted(glob(path_to_data+nc_file))
print(len(fnames))

#Read and process CTH data at different temporal steps 
for t, f in enumerate(fnames):
    print(f.split('/')[-1])
    # Get time info from filename
    basename = os.path.basename(f)
    time_str = basename.split('_')[7]
    year, month, day = time_str[0:4], time_str[4:6], time_str[6:8]
    print(f"Processing file for {year}-{month}-{day}")

    # Open via Satpy for lat/lon info and loading variable
    scn = Scene(filenames=[f], reader="fci_l2_nc")
    scn.load([variable])

    # Get full lat/lon grid
    lon_grid, lat_grid = scn[variable].attrs["area"].get_lonlats()

    # Compute ROI indices in pixel space
    mask = (lat_grid >= latmin) & (lat_grid <= latmax) & \
           (lon_grid >= lonmin) & (lon_grid <= lonmax)
    ys, xs = np.where(mask)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    print(f"Cropping x: {xmin} to {xmax}, y: {ymin} to {ymax}")

    # Open the dataset once (no need to open again separately)
    ds = xr.open_dataset(f, engine="netcdf4")

    if variable not in ds:
        print(f"{variable} not found in {f}")
        continue

    # Crop using pixel indices
    cropped = ds.isel(
        number_of_rows=slice(ymin, ymax + 1),
        number_of_columns=slice(xmin, xmax + 1)
    )

    # Prepare variable and attributes
    var_data = cropped[variable].copy(deep=True)
    proj_attrs = ds.get('mtg_geos_projection', {}).attrs
    var_data.attrs.update({
        'satellite_nominal_latitude': proj_attrs.get('latitude_of_projection_origin', 0.0),
        'satellite_nominal_longitude': proj_attrs.get('longitude_of_projection_origin', 0.0),
        'satellite_nominal_altitude': proj_attrs.get('perspective_point_height', 35786400.0),
    })

    # Create new cropped dataset with correct structure
    ds_new = xr.Dataset(
        data_vars={variable: var_data},
        coords={k: cropped[k] for k in ['x', 'y'] if k in cropped},
        attrs=ds.attrs
    )

    if 'mtg_geos_projection' in ds:
        ds_new['mtg_geos_projection'] = ds['mtg_geos_projection']

    # Save to NetCDF
    save_dir = os.path.join(out_dir, day)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, basename)
    ds_new.to_netcdf(out_path, format='NETCDF4')
    print(f"✅ Saved reduced dataset to {out_path}")



