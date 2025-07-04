import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import numpy as np
import imageio

def load_data_for_day(folder: Path, channel: str, day: str, suffix: str = 'raw'):
    """
    Load data for one day for a specific channel.
    """
    outdir = folder / f"{day[0:4]}/{day[4:6]}/"
    filepath = f"{outdir}/{channel}_{day}_{suffix}.nc"
    ds = xr.open_dataset(filepath)

    return ds

def load_coords(folder: Path, channel: str):
    """
    Load data for one day for a specific channel.
    """
    coord_file = f"{folder}/{channel}_original_coords.nc"
    ds_coords = xr.open_dataset(coord_file)
        
    return ds_coords

def plot_one_step(outdir, data, lat, lon, ts, channel, cmap='viridis', vmin=None, vmax=None, unit='', extent=None):
    """
    Plot a single timestep of data using Cartopy.
    """
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

    # Set extent if provided
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    plt.tight_layout()
    fig.savefig(f"{outdir}/{channel}_plot_{str(ts).split('.')[0]}.png", dpi=300, bbox_inches='tight')


def main():
    folder = Path("/data/sat/msg/mtg/fci/")  # 🔁 Replace with actual path
    day = "20250630"  # Format: YYYYMMDD
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
    channel = "ir_105"  #  Replace with the desired channel
    cmap = channel_configs[channel]["cmap"]
    vmin = channel_configs[channel]["vmin"]
    vmax = channel_configs[channel]["vmax"]
    unit = channel_configs[channel]["unit"]

    regrid = False
    parallax = False
    extent = [5, 16, 42, 52]  # [lon_min, lon_max, lat_min, lat_max]
    gif = True

    suffix_parts = []
    if regrid:
        suffix_parts.append("regrid")
    if parallax:
        suffix_parts.append("parallax")
    suffix = "_".join(suffix_parts) if suffix_parts else "raw"

    outdir = Path(f"/data/sat/msg/mtg/fci/plots/{day}/{channel}/")  # Output directory for plots
    outdir.mkdir(parents=True, exist_ok=True)

    ds = load_data_for_day(folder, channel, day, suffix)
    print(ds)

    if not regrid:
        coords_ds = load_coords(folder, channel)
        lat_grid = coords_ds['latitude'].values
        lon_grid = coords_ds['longitude'].values
    else:
        lat = ds['lat'].values
        lon = ds['lon'].values
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

    # loop over the timestamps of ds
    timestamps = ds['time'].values
    
    for t in timestamps:
        print(f"Reading and plotting time step: {t}")
        data = ds[channel].sel(time=t).values
        plot_one_step(outdir, data, lat_grid, lon_grid, t, channel, cmap, vmin, vmax, unit, extent)

    if gif:
        images = []
        for t in timestamps:
            img_path = outdir / f"{channel}_plot_{str(t).split('.')[0]}.png"
            images.append(imageio.imread(img_path))
        gif_path = outdir / f"{channel}_daily_{day}_{suffix}.gif"
        imageio.mimsave(gif_path, images, duration=0.5)
        print(f"GIF saved at {gif_path}")

if __name__ == "__main__":
    main()
