"""
this module contains utility functions and color bars for plotting satellite data, deeine plotting domains a

"""

import matplotlib as mpl
import cmcrameri.cm as cmc
import cartopy.crs as ccrs
from pathlib import Path
import os
import subprocess

# definition of domains of interest
domain_German_flood =  [ 5.,    9.,    48.,   52.  ] # minlon, maxlon, minlat, maxlat
domain_expats       =  [ 5.,   16.,    42.,   51.5 ] # minlon, maxlon, minlat, maxlat
domain_joyce        =  [ 6.,   6.5,    50.8,  51.3 ] # minlon, maxlon, minlat, maxlat   
domain_ACTA         =  [ 10.73,12.0,   46.3,  47.2 ] # minlon, maxlon, minlat, maxlat
domain_TEAMX      =  [ 9.9,   12.7,    45.5,   47.4  ] # minlon, maxlon, minlat, maxlat
# 45.48592, 9.92821 47.35537, 12.78054
#quicklook browser output path
quicklook_browser_output_path = '/data/obs/campaigns/teamx/quicklooks/mtg_fci_mp4/'
# directory where mtg fci daily files of each channel are stored
mtg_fci_daily_files_path = Path('/data/trade_pc/mtg/fci/processed/no_parallax/original_grid/')
coords_file_path = Path('/data/trade_pc/mtg/fci/processed/no_parallax/original_grid/')

# config for plotting channels
channel_configs = {
    "ir_105": {
        "cmap": "gray_r",
        "vmin": 200,
        "vmax": 300,
        "unit": "K"
    },
    "vis_06": {
        "cmap": "gray",
        "vmin": 0,
        "vmax": 100,
        "unit": "Reflectance"
    }
    }







def plot_teamx_sites(ax, color, symbol_size):
    """
    Function to plot instrument positions on a map for the quicklook browser.
    Parameters
    ----------
    ax : cartopy axes
        The axes on which to plot the instrument positions.
    color : str
        The color of the instrument positions.
    symbol_size : int
        The size of the symbols used to represent the instrument positions.
    Returns
    -------
        None
        """       
    #add instrument positions for the quicklook browser
    # define a dictionary with coordinates of the sites, site names
    dict_towns = {
        'Branzol': [46.4031302, 11.32243],
        'Brixen': [46.71042, 11.65246],
        'Dornacherof':[46.49978, 11.43554], 
        'Ehrenburg': [46.79559, 11.83649],
        'Felthuner hutte': [46.60479, 11.45674],
        'Garganzone':[46.58495, 11.20144], 
        'Klobenstein':[46.53965, 11.45832], 
        'Meran':[46.67114, 11.15257], 
        'Naturns':[46.64995, 11.00418], 
        'Plose': [46.69555, 11.73333],
        'Rittenhorn': [46.61499, 11.46083],
        'Sarnthein': [46.6427, 11.35729],
        'Schwarzseespitze': [46.59605, 11.45255], 
        'Sterzing':[46.89633, 11.43214],
        'St Martin':[46.78353, 11.22874]}

    # loop on dictionary items for plotting scatter points
    for site, coords in dict_towns.items():
        lat, lon = coords
        ax.scatter(lon, lat, marker='x', color=color, s=symbol_size, transform=ccrs.PlateCarree())
        ax.text(lon + 0.01, lat - 0.01, site, color=color, transform=ccrs.PlateCarree(), ha='left', va='top', fontsize=5)
        

    return



def same_image_seq_as_mp4(out_root, images, day, channel, domain_name, fps=10):
    """
    script to save a sequence of images as an mp4 video file using ffmpeg.

    Args:
        folder_path (string): path where we want to create the mp4 file
        out_root (string): path where the images are stored
        images (list): list of image filenames to be included in the video.
        day (string): day of the images, used for naming the output file. 
        channel (string): channel name, used for naming the output file.
        domain_name (_typstringe_): name of the domain, used for naming the output file.
        fps (int, optional): _description_. Defaults to 10.
    """
    
    # Create symlinked frames for ffmpeg
    temp_dir = os.path.join(out_root, "ffmpeg/")
    os.makedirs(temp_dir, exist_ok=True)
    
    print('temp dir', temp_dir)

    # loop on the images to create symlinks in the temp directory
    for idx, img in enumerate(images):
        src = os.path.join(out_root, img)
        dst = os.path.join(temp_dir, f"frame_{idx:04d}.png")
        if not os.path.exists(dst):
            os.symlink(src, dst)

    mp4_filename = f"{day}_{channel}_quicklook_raw_{domain_name}.mp4"
    mp4_path = os.path.join(out_root, mp4_filename)
    print(f"MP4 output path: {mp4_path}")
    
    if os.path.exists(mp4_path):
        print(f"🟡 MP4 already exists, skipping: {mp4_filename}")
    else:
        print(f"🎬 Creating MP4: {mp4_path}")
        try:
            subprocess.run([
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-preset", "slow",         # compression trade-off: slower = smaller file
            "-crf", "32",              # quality vs. file size (23 is default; try 28–30)
            "-pix_fmt", "yuv420p",
            "-movflags", "faststart",
            mp4_path], check=True)
            print(f"✅ MP4 created at: {mp4_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed: {e}")

    return