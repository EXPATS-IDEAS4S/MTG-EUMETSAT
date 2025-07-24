import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from glob import glob

# === Paths ===
base_path = "/home/Daniele/data/data_for_parallax"
parallax_dir = f"{base_path}/processed/parallax_corrected_mtg"
non_parallax_dir = f"{base_path}/processed/non_parallax_corrected_mtg"
coord_file = f"{base_path}/mtg/ir_105_original_coords.nc"
output_dir = f"{base_path}/plots"
variable_name = "ir_105"  # Change this to the variable you want to plot
latmin, lonmin, latmax, lonmax = 42, 5, 52, 16  # Define the region of interest
vmin, vmax = 200, 300  # Define the color scale limits for the variable

# === Load shared coordinates ===
coords_ds = xr.open_dataset(coord_file)
lat = coords_ds['latitude'].values
lon = coords_ds['longitude'].values

# === Helper: extract timestamp from filename ===
def extract_timestamp(filename):
    try:
        name = filename.split('/')[-1].split('.')[0]  # Get the last part of the path
        parts = name.split('_')
        ts_str = parts[2] 
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

# === Collect filenames ===
def collect_files(folder):
    files = glob(os.path.join(folder, "*.nc"))
    file_dict = {}
    for f in files:
        ts = extract_timestamp(f)
        if ts:
            file_dict[ts] = f
    return file_dict
    

parallax_files = collect_files(parallax_dir)
non_parallax_files = collect_files(non_parallax_dir)

# === Find common timestamps ===
common_timestamps = sorted(set(parallax_files.keys()) & set(non_parallax_files.keys()))
print(f"Found {len(common_timestamps)} common timestamps between parallax and non-parallax datasets.")
print(common_timestamps)

if not common_timestamps:
    print("No matching timestamps found.")
    exit()

# === Loop over timestamps and plot ===
for ts in common_timestamps:
    print(f"Plotting {ts}...")

    # Load datasets
    ds_parallax = xr.open_dataset(parallax_files[ts])

    ds_non_parallax = xr.open_dataset(non_parallax_files[ts])

    # Select the main variable
    data_parallax = ds_parallax["parallax_corrected_"+variable_name].squeeze()
    data_non_parallax = ds_non_parallax[variable_name].squeeze()

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 5),
                            subplot_kw={'projection': ccrs.PlateCarree()})

    #for ax, data, title in zip(axs, [data_non_parallax, data_parallax],["Non-Parallax Corrected", "Parallax Corrected"]):
    im0 = axs[0].pcolormesh(lon, lat, data_non_parallax, cmap='gray', shading='auto', vmin=vmin, vmax=vmax)
    axs[0].coastlines(color='red', linewidth=0.5)
    axs[0].add_feature(cfeature.BORDERS, linewidth=0.5, color='red')
    axs[0].set_title(f"Original")
    plt.colorbar(im0, ax=axs[0], orientation='vertical', label=variable_name, pad=0.05, shrink=0.5)
    axs[0].set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    
    im1 = axs[1].pcolormesh(lon, lat, data_parallax, cmap='gray', shading='auto', vmin=vmin, vmax=vmax)
    axs[1].coastlines(color='red', linewidth=0.5)
    axs[1].add_feature(cfeature.BORDERS, linewidth=0.5, color='red')
    axs[1].set_title(f"Parallax Corrected")
    plt.colorbar(im1, ax=axs[1], orientation='vertical', label=variable_name, pad=0.05, shrink=0.5)
    axs[1].set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

    #add a third plot for comparison
    diff = data_parallax - data_non_parallax
    im2 = axs[2].pcolormesh(lon, lat, diff, cmap='coolwarm', shading='auto', vmin=-50, vmax=50)
    axs[2].coastlines(color='red', linewidth=0.5)
    axs[2].add_feature(cfeature.BORDERS, linewidth=0.5, color='red')
    axs[2].set_title(f"Difference (Parallax - Original)")
    plt.colorbar(im2, ax=axs[2], orientation='vertical', label=f"{variable_name} Difference", pad=0.05, shrink=0.5)
    axs[2].set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

    #insert overall title with the timestamp
    plt.suptitle(f"Comparison of {variable_name} at {ts.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=16, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f"{output_dir}/{variable_name}_{ts.strftime('%Y%m%d_%H%M%S')}_parallax.png", bbox_inches='tight')
    plt.close(fig)