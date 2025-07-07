"""
Visualization script for daily satellite channel data using Cartopy.

Functions:
- load_data_for_day: Loads a NetCDF file for a specific channel and day.
- load_coords: Loads original latitude/longitude coordinate grids.
- plot_one_step: Plots a single time step using Cartopy with specified settings.
- main: Main routine that loads data, generates plots for each timestep, and optionally creates an animated GIF.
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import numpy as np
import imageio


def load_data_for_day(folder: Path, channel: str, day: str, suffix: str = 'raw'):
    """
    Load daily NetCDF data for a given channel.

    Parameters:
        folder (Path): Base directory of the data files.
        channel (str): Data channel name (e.g., 'ir_105').
        day (str): Day in 'YYYYMMDD' format.
        suffix (str): Filename suffix indicating processing type ('raw', 'regrid', etc.).

    Returns:
        xarray.Dataset: Loaded dataset.
    """
    outdir = folder / f"{day[0:4]}/{day[4:6]}/"
    filepath = f"{outdir}/{channel}_{day}_{suffix}.nc"
    ds = xr.open_dataset(filepath)
    return ds


def load_coords(folder: Path, channel: str):
    """
    Load original latitude and longitude coordinate grid for a channel.

    Parameters:
        folder (Path): Base directory containing coordinate files.
        channel (str): Data channel name.

    Returns:
        xarray.Dataset: Dataset containing latitude and longitude arrays.
    """
    coord_file = f"{folder}/{channel}_original_coords.nc"
    ds_coords = xr.open_dataset(coord_file)
    return ds_coords


def plot_one_step(outdir, data, lat, lon, ts, channel, cmap='viridis',
                  vmin=None, vmax=None, unit='', extent=None):
    """
    Plot a single timestep of satellite data using Cartopy.

    Parameters:
        outdir (Path): Directory to save the plot image.
        data (ndarray): 2D data array to plot.
        lat (ndarray): Latitude grid.
        lon (ndarray): Longitude grid.
        ts (datetime-like): Timestamp for the current frame.
        channel (str): Channel name for labeling.
        cmap (str): Matplotlib colormap name.
        vmin (float): Minimum value for color scaling.
        vmax (float): Maximum value for color scaling.
        unit (str): Unit label for colorbar.
        extent (list): Map extent [lon_min, lon_max, lat_min, lat_max].
    """
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    img = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                        cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(f'{channel} ({unit})', fontsize=14)

    ax.coastlines(resolution='10m', color='orangered', linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', color='orangered', linewidth=0.5)
    ax.set_title(f"{channel} at {str(ts).split('.')[0]}", fontsize=14, fontweight='bold')

    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    plt.tight_layout()
    fig.savefig(f"{outdir}/{channel}_plot_{str(ts).split('.')[0]}.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    """
    Main script execution: loads daily channel data, plots each timestep,
    and optionally compiles plots into an animated GIF.
    """
    folder = Path("/data/sat/msg/mtg/fci/")
    day = "20250630"
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

    channel = "ir_105"
    cmap = channel_configs[channel]["cmap"]
    vmin = channel_configs[channel]["vmin"]
    vmax = channel_configs[channel]["vmax"]
    unit = channel_configs[channel]["unit"]

    regrid = False
    parallax = False
    extent = [5, 16, 42, 52]
    gif = True

    suffix_parts = []
    if regrid:
        suffix_parts.append("regrid")
    if parallax:
        suffix_parts.append("parallax")
    suffix = "_".join(suffix_parts) if suffix_parts else "raw"

    outdir = Path(f"/data/sat/msg/mtg/fci/plots/{day}/{channel}/")
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

    timestamps = ds['time'].values
    for t in timestamps:
        print(f"Reading and plotting time step: {t}")
        data = ds[channel].sel(time=t).values
        plot_one_step(outdir, data, lat_grid, lon_grid, t, channel,
                      cmap, vmin, vmax, unit, extent)

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
