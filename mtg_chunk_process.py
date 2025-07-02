
# Import libraries
#import eumdac
import datetime
import shutil
import requests
import time
import numpy as np
import datetime
import shutil
import fnmatch
import requests
import time
import os
import zipfile
import json 
import os
from shapely.wkt import loads
from shapely.geometry import Polygon
import glob
import xarray as xr
#import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import LineString
from satpy import Scene

from domain import user_roi

from mtg_chunks import create_output_folder

def main():
    
    
    # define set of days to process
    date_start = datetime.datetime(2025, 6, 1)
    date_end = datetime.datetime(2025, 6, 2)
    
    # find all dates between date_start and date_end
    date_range = [date_start + datetime.timedelta(days=x) for x in range(0, (date_end - date_start).days + 1)]
    print('Date range:', date_range)

    for i_date, day_process in enumerate(date_range):
        
        # create output directory for the selected day of the form /yyyy/mm/dd/
        output_folder = create_output_folder(day_process, path_input='/data/trade_pc/mtg/fci/')
        print('Output folder:', output_folder)

        # loop on counts in repeat cycle:
        ds_chunk_list = []
        for i_count in range(1, 145):
            
            # read chunks for the selected 10 min times
            files_count = np.sort(glob.glob(output_folder+'*N__O_0'+str(i_count).zfill(3)+'*.nc')).tolist()
            #print(files_count)
            
            # read time stamps of files count
            time_chunk_begin = files_count[0].split('_')[8]
            time_chunk_end = files_count[-1].split('_')[9]
            print(time_chunk_begin, time_chunk_end)
            
            # convert time_sttime_chunk to datetime
            time_chunk_begin_dt = datetime.datetime.strptime(time_chunk_begin, '%Y%m%d%H%M%S')
            time_chunk_end_dt = datetime.datetime.strptime(time_chunk_end, '%Y%m%d%H%M%S')
            print('Time chunk begin:', time_chunk_begin_dt, 'Time chunk end:', time_chunk_end_dt)
            
            # build datetime for output for the day
            time_str = str(day_process.year)+str(day_process.month).zfill(2)+str(day_process.day).zfill(2)
            print(time_str)
            
            # print time stamp
            #print('Time chunk:', time_chunk)

            # define scene
            print(files_count)
            exit()
            scn = Scene(reader='fci_l1c_nc',filenames=files_count)
            
            # print scene
            print('scene ', scn)

            #print available channels
            print('Available channels:', scn.available_dataset_names())
            # #Available channels: ['ir_105', 'ir_105_earth_sun_distance', 'ir_105_index_map', 'ir_105_pixel_quality', 'ir_105_platform_altitude', 'ir_105_subsatellite_latitude', 'ir_105_subsatellite_longitude', 'ir_105_subsolar_latitude', 'ir_105_subsolar_longitude', 'ir_105_sun_satellite_distance', 'ir_105_swath_direction', 'ir_105_swath_number', 'ir_105_time', 
            #                      'ir_38', 'ir_38_earth_sun_distance', 'ir_38_index_map', 'ir_38_pixel_quality', 'ir_38_platform_altitude', 'ir_38_subsatellite_latitude', 'ir_38_subsatellite_longitude', 'ir_38_subsolar_latitude', 'ir_38_subsolar_longitude', 'ir_38_sun_satellite_distance', 'ir_38_swath_direction', 'ir_38_swath_number', 'ir_38_time', 
            #                      'nir_22', 'nir_22_earth_sun_distance', 'nir_22_index_map', 'nir_22_pixel_quality', 'nir_22_platform_altitude', 'nir_22_subsatellite_latitude', 'nir_22_subsatellite_longitude', 'nir_22_subsolar_latitude', 'nir_22_subsolar_longitude', 'nir_22_sun_satellite_distance', 'nir_22_swath_direction', 'nir_22_swath_number', 'nir_22_time', 
            #                      'vis_06', 'vis_06_earth_sun_distance', 'vis_06_index_map', 'vis_06_pixel_quality', 'vis_06_platform_altitude', 'vis_06_subsatellite_latitude', 'vis_06_subsatellite_longitude', 'vis_06_subsolar_latitude', 'vis_06_subsolar_longitude', 'vis_06_sun_satellite_distance', 'vis_06_swath_direction', 'vis_06_swath_number', 'vis_06_time']
            
            
            # loop on channels to read fields
            channels = ['ir_105', 'vis_06']
            
            # call function to read all channels at their original resolution
            ds_chunk, ds_coords_chunk = read_channels(scn, channels, user_roi)
            
            
            # store ds_coords_chunk to netcdf at first iteration
            if i_count == 1:
                ds_coords_chunk.to_netcdf(output_folder+time_str+'_MTG_fci_hrfi_lats_lons.nc')
                
            # concatenate ds_chunks to ds_chunk list if dataset is readable otherwise skip
            try:
                ds_chunk_list.append(ds_chunk)
                
                # add time variable to ds_chunk
                ds_chunk = ds_chunk.assign_coords(time=(['time'], [time_chunk_dt]))

            except Exception as e:
                print(f"Error appending chunk {i_count}: {e}")
                continue
        
        # concatenate ds_chunk along time dimension
        ds_day = xr.concat(ds_chunk_list, dim='time')
        print(ds_day)
        
        encoding_dict = {}
        for channel in channels:
            encoding_dict[channel] = {"zlib": True, "complevel": 4}
        encoding_dict['time'] = {"units": "seconds since 2000-01-01", "dtype": "i4"}

        # save the dataset to disk and compress variables
        ds_day.to_netcdf(output_folder+time_str+'_MTG_fci_hrfi_expats_no_parallax.nc', 
                        mode='w', 
                        format='NETCDF4', 
                        encoding=encoding_dict)
        
        # remove all downloaded files
        for i_count in range(1, 145):
            
            # read chunks for the selected 10 min times
            files_count = np.sort(glob.glob(output_folder+'*N__O_0'+str(i_count).zfill(3)+'*.nc')).tolist()
            
            # remove list of files
            for file in files_count:
                os.remove(file)
                print(f"Removed file {file}")
                
        print('All files removed')



  
def read_channels(scn, channels, user_roi):
    """
    Reads specified channels from a satellite scene within a user-defined ROI.
    Returns a merged dataset of the channels and their respective coordinates.
    """
    ds_list_ch = []
    ds_coords_list = []

    for ch in channels:
        # Load and crop the channel
        scn.load([ch], upper_right_corner='NE')
        scn_crop = scn.crop(ll_bbox=(
            user_roi["lon_min"],
            user_roi["lat_min"],
            user_roi["lon_max"],
            user_roi["lat_max"]
        ))

        area_crop = scn_crop[ch].attrs['area']
        lon, lat = area_crop.get_lonlats()
        val_crop = scn_crop[ch].values

        # Normalize coordinate names based on channel
        suffix = ch.split('_')[-1].replace('.', '')  # e.g., '105' from 'ir_105'
        lat_name = f'lat_{suffix}'
        lon_name = f'lon_{suffix}'

        # Create xarray Dataset for data and coordinates
        ds = xr.Dataset(
            data_vars={ch: ([lat_name, lon_name], val_crop)}
        )
        ds_coords = xr.Dataset(
            coords={
                lat_name: (lat_name, lat[:, 0]),
                lon_name: (lon_name, lon[0, :])
            }
        )
        print(ds)
        print(ds_coords)
        exit()

        ds_list_ch.append(ds)
        ds_coords_list.append(ds_coords)

    return xr.merge(ds_list_ch), xr.merge(ds_coords_list)


if __name__ == "__main__":
    main()
    
    # print end of script
    print("MTG chunk processing completed.")
    