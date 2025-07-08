import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import numpy as np
import imageio
import os
import shutil

"""
script to plot quicklooks gif animated of MTG FCI data for a list of days and a channel
authors: Claudia Acquistapace and Daniele Corradini
date: 2025-07-04
This script reads the daily files of MTG FCI data for a specific channel, plots each timestep, and saves the plots as images and a GIF.
It uses Cartopy for mapping and xarray for handling NetCDF data.
It also includes functions to load data, create output directories, and plot individual timesteps.
It then deleted the individual images after creating the GIF to save space.

launch with nohup
nohup python3 plotting/plot_quicklooks.py > plotting/plot_quicklooks.log 2>&1 &

"""

from plotting.plot_utils import channel_configs, plot_teamx_sites, domain_TEAMX, mtg_fci_daily_files_path, quicklook_browser_output_path


from pathlib import Path


def load_data_for_day(folder, channel: str, day: str, regrid: False, parallax: False):
    """
    Load data for one day for a specific channel.
    Parameters
    ----------
    folder : Path
        The folder where the daily files are stored.
    channel : str
        The channel for which to load the data.
    day : str
        The day for which to load the data, in the format 'YYYYMMDD'.
    regrid : bool, optional
        Whether to load the regridded data. Default is False.
    parallax : bool, optional
        Whether to load the parallax-corrected data. Default is False.
    Returns
    -------
    ds : xarray.Dataset
        The xarray Dataset containing the data for the specified channel and day.   
    """
    
    # Define the suffix based on regrid and parallax options
    # If both are False, use 'raw' as the suffix
    suffix_parts = []
    if regrid:
        suffix_parts.append("regrid")
    if parallax:
       suffix_parts.append("parallax")
       
    suffix = "_".join(suffix_parts) if suffix_parts else "raw"

    # Create the output directory path based on the day
    folder = Path(folder)
    outdir = folder / f"{day[0:4]}/{day[4:6]}/"
    filepath = f"{outdir}/{channel}_{day}_{suffix}.nc"
    ds = xr.open_dataset(filepath)

    return ds, suffix

def load_coords(channel: str):
    """
    Load data for one day for a specific channel.
    """
    from plotting.plot_utils import coords_file_path
    coord_file = f"{coords_file_path}/{channel}_original_coords.nc"
    ds_coords = xr.open_dataset(coord_file)
    
    return ds_coords

def plot_one_step(outdir, data, lat, lon, ts, channel, cmap='viridis', vmin=None, vmax=None, unit='', extent=None):
    """
    Plot a single timestep of data using Cartopy.
    """
    from plotting.plot_utils import channel_configs, domain_ACTA
    from plotting.plot_utils import plot_teamx_sites
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Plot data
    # print("lat shape:", lat.shape)
    # print("lon shape:", lon.shape)
    # print("data shape:", data.shape)
    img = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(f'{channel} ({unit})', fontsize=14)

    # Add features
    ax.coastlines(resolution='10m', color='orangered', linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', color='orangered', linewidth=0.5)
    ax.set_title(f"{channel} at {str(ts).split('.')[0]}", fontsize=14, fontweight='bold')

    # add teamx sites
    plot_teamx_sites(ax, color='orangered', symbol_size=10)
    
    # Set extent if provided
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    plt.tight_layout()
    
    # create output directory if it does not exist
    #outdir = Path(outdir)
 
    #print(f"Saving plot to {outdir}/{channel}_plot_{str(ts).split('.')[0]}.png")
    #fig.savefig(f"{outdir}/{channel}_plot_{str(ts).split('.')[0]}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{outdir}{channel}_plot_{str(ts).split('.')[0]}.png", dpi=300, bbox_inches='tight')

def main():
    
    # read folder for accessing data ncdf daily files 
    from plotting.plot_utils import mtg_fci_daily_files_path as folder  # 🔁 Replace with actual path
    
    # read output root for storing plots
    from plotting.plot_utils import quicklook_browser_output_path as out_root
    from plotting.plot_utils import coords_file_path 

    # Find days to plot
    # generate days string for the entire month 2025 06 
    days = [f"202505{str(i).zfill(2)}" for i in range(10, 31)]  # June 2025


    # loop on days to be plotted (month of june 2025)
    for day in days:
        
        print('Plotting data for day:', day)
        
        # loop on channels
        channels = [ "ir_105", "vis_06"]  # Replace with the desired channels
        
        for channel in channels:
            
            print(f"Processing channel: {channel}")
            
            # Check if GIF already exists for this day and channel
            gif_path = f"{out_root}{day}_{channel}_quicklook_raw_TEAMx.gif"
            print(gif_path)
            if Path(gif_path).exists():
                print(f"GIF for {day} and channel {channel} already exists, skipping...")
                continue
            
            else:
                print(f"GIF for {day} and channel {channel} does not exist, proceeding with plotting...")
            
                # read channel configuration
                cmap = channel_configs[channel]["cmap"]
                vmin = channel_configs[channel]["vmin"]
                vmax = channel_configs[channel]["vmax"]
                unit = channel_configs[channel]["unit"]

                # define domain of interest
                extent = domain_TEAMX  # replace with the desired domain
                domain_name = "TEAMx"  # replace with the desired domain name
                regrid = False
                parallax = False
                gif = True

                # create output directory for quicklooks
                outdir = Path(out_root)
                outdir.mkdir(parents=True, exist_ok=True)

                
                
                # read data for the day setting the complete path
                ds, suffix = load_data_for_day(folder, channel, day, regrid, parallax)
                print(ds)

                if not regrid:
                    print(f"{coords_file_path}/{channel}_original_coords.nc")
                    coords_ds = xr.open_dataset(f"{coords_file_path}/{channel}_original_coords.nc")#load_coords(channel)
                    lat_grid = coords_ds['latitude'].values
                    lon_grid = coords_ds['longitude'].values
                else:
                    lat = ds['lat'].values
                    lon = ds['lon'].values
                    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

                # loop over the timestamps of ds
                timestamps = ds['time'].values
                
                for t in timestamps:
                    
                    # if file does not exist, create it
                    if not Path(f"{out_root}{channel}_plot_{str(t).split('.')[0]}.png").exists():

                        #print(f"Reading and plotting time step: {t}")
                        data = ds[channel].sel(time=t).values
                    
                        plot_one_step(out_root, 
                                    data, 
                                    lat_grid, 
                                    lon_grid, 
                                    t, 
                                    channel, 
                                    cmap, 
                                    vmin, 
                                    vmax,
                                    unit, 
                                    extent)
                    else:
                        print(f"Plot for time step {t} already exists, skipping...")

                if gif:
                    
                    # check if gif already exists, then skip
                    gif_path = f"{out_root}/{day}_{channel}_quicklook_{suffix}_{domain_name}.gif"
                    if Path(gif_path).exists():
                        print(f"GIF for {day} and channel {channel} already exists, skipping...")
                        continue
                    else:
                    
                        images = []
                        for t in timestamps:
                            img_path = f"{out_root}{channel}_plot_{str(t).split('.')[0]}.png"
                            if Path(img_path).exists():
                                images.append(imageio.imread(img_path))
                            else:
                                print(f"Image for time step {t} does not exist, skipping...")
                        
                        # Save images as a GIF
                        gif_path = f"{out_root}/{day}_{channel}_quicklook_{suffix}_{domain_name}.gif"
                        imageio.mimsave(gif_path, images, duration=0.5)
                        print(f"GIF saved at {gif_path}")
                    
                        
                    # delete all individual images to save space
                    for filename in os.listdir(out_root):
                        if filename.endswith('.png'):
                            print(f"Deleting {filename} to save space...")
                            print(os.path.join(out_root, filename))
                            os.remove(os.path.join(out_root, filename))
                            
                print(f"Deleted all PNG files in {outdir}")

        # close all figures
        plt.close('all')
        
        print(f"Finished processing day: {day}")
    print("All days processed successfully. Quicklooks GIFs created.")


if __name__ == "__main__":
    main()
