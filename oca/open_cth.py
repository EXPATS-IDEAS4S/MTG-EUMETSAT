import xarray as xr
from satpy import Scene
import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature  
import matplotlib.pyplot as plt
from glob import glob

path_to_data = "/data/sat/mtg/fci/oca/processing_for_parallax/2025/05/22"
path_to_fig = "/data/sat/mtg/fci/oca/fig"
os.makedirs(path_to_fig, exist_ok=True)

#get all files in the directory
nc_file = 'W_XX*.nc'
fnames = sorted(glob(os.path.join(path_to_data, nc_file)))
print(fnames)

#filename = 'W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-2-OCA--FD------NC4E_C_EUMT_20250522234518_L2PF_OPE_20250522233000_20250522234000_N__C_0142_0000.nc' 

lonmin, latmin, lonmax, latmax = 5, 42, 16, 52 #EXPATS
variable_oca = 'retrieved_cloud_top_height'
#variable_oca = 'ctth_alti'
# variable_mtg = 'ir_105'
# #open one MTG file to get area definition
# mtg_chunks = ['/data/trade_pc/mtg/fci/2025/07/01/W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-FD--CHK-BODY---NC4E_C_EUMT_20250701001051_IDPFI_OPE_20250701000810_20250701000857_N__O_0001_0035.nc', 
#                       '/data/trade_pc/mtg/fci/2025/07/01/W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-FD--CHK-BODY---NC4E_C_EUMT_20250701001101_IDPFI_OPE_20250701000818_20250701000904_N__O_0001_0036.nc', 
#                       '/data/trade_pc/mtg/fci/2025/07/01/W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-FD--CHK-BODY---NC4E_C_EUMT_20250701001109_IDPFI_OPE_20250701000834_20250701000917_N__O_0001_0037.nc']

# scn_mtg = Scene(reader= "fci_l1c_nc", filenames=mtg_chunks)
# scn_mtg.load([variable_mtg])
# area_crop = scn_mtg[variable_mtg].attrs['area'] #area in m
# print(area_crop)

# ds = xr.open_dataset(f"{path_to_data}/{filename}" , engine='h5netcdf')   

# print(ds)

#print var names
#print(ds['mtg_geos_projection'].attrs)

# loop of the files
for filename in fnames:
    print(filename)

    #open cth as an xarray dataset
    ds_oca = xr.open_dataset(f"{filename}", engine='netcdf4')
    print(ds_oca)
    print(ds_oca['retrieved_cloud_top_height'].attrs)

    # Try to open it using satpy

    scn_oca = Scene(filenames=[f"{filename}"], reader='fci_l2_nc')

    #load variable

    scn_oca.load([variable_oca]) 

    #Crop to area of interest
    crop_scn = scn_oca.crop(ll_bbox=(lonmin, latmin, lonmax, latmax))

    #get the lat/lon coordsonly for one channel (as all of them share the same grid)

    #get coord in the cropped area
    area_crop = scn_oca[variable_oca].attrs['area'] #area in m
    sat_lon_crop, sat_lat_crop = area_crop.get_lonlats() 
    print(sat_lon_crop.shape, sat_lat_crop.shape)

    sat_data_crop = scn_oca[variable_oca].values 
    print(sat_data_crop.shape)

    # plot the data using cartopy

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.pcolormesh(sat_lon_crop, sat_lat_crop, sat_data_crop, transform=ccrs.PlateCarree(), cmap='cool', shading='auto')
    ax.set_title(f"Cloud Top Height on {filename.split('_')[7]}")
    fig.savefig(f"{path_to_fig}/cloud_top_height_{filename.split('/')[-1].split('.')[0]}.png", dpi=300, bbox_inches='tight')
    print(f"Saved figure for {filename.split('.')[0]} in {path_to_fig}/cloud_top_height_{filename.split('_')[7]}.png")