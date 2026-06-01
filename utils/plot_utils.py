"""
Plotting utilities and domain/config definitions for quicklooks.

Copied from `plotting/plot_utils.py`.
"""

import matplotlib as mpl
import cmcrameri.cm as cmc
import cartopy.crs as ccrs
from pathlib import Path
import os
import subprocess

# definition of domains of interest
domain_German_flood =  [ 5.,    9.,    48.,   52.  ]
domain_expats       =  [ 5.,   16.,    42.,   51.5 ]
domain_joyce        =  [ 6.,   6.5,    50.8,  51.3 ]
domain_ACTA         =  [ 10.73,12.0,   46.3,  47.2 ]
domain_TEAMX      =  [ 9.9,   12.7,    45.5,   47.4  ]

quicklook_browser_output_path = '/data/obs/campaigns/teamx/quicklooks/mtg_fci_mp4/'
mtg_fci_daily_files_path = Path('/data/trade_pc/mtg/fci/processed/no_parallax/original_grid/')
coords_file_path = Path('/data/trade_pc/mtg/fci/processed/no_parallax/original_grid/')

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

    for site, coords in dict_towns.items():
        lat, lon = coords
        ax.scatter(lon, lat, marker='x', color=color, s=symbol_size, transform=ccrs.PlateCarree())
        ax.text(lon + 0.01, lat - 0.01, site, color=color, transform=ccrs.PlateCarree(), ha='left', va='top', fontsize=5)

    return


def same_image_seq_as_mp4(out_root, images, day, channel, domain_name, fps=10):
    temp_dir = os.path.join(out_root, "ffmpeg/")
    os.makedirs(temp_dir, exist_ok=True)

    for idx, img in enumerate(images):
        src = os.path.join(out_root, img)
        dst = os.path.join(temp_dir, f"frame_{idx:04d}.png")
        if not os.path.exists(dst):
            os.symlink(src, dst)

    mp4_filename = f"{day}_{channel}_quicklook_raw_{domain_name}.mp4"
    mp4_path = os.path.join(out_root, mp4_filename)

    if os.path.exists(mp4_path):
        print(f"🟡 MP4 already exists, skipping: {mp4_filename}")
    else:
        try:
            subprocess.run([
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "32",
            "-pix_fmt", "yuv420p",
            "-movflags", "faststart",
            mp4_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed: {e}")

    return
