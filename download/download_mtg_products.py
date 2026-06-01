#!/usr/bin/env python3

"""
download_mtg_products.py

Download MTG (Meteorological GeoStationary) products from EUMETSAT using the
`eumdac` API and the configuration provided in `user_config.USER_CONFIG`.

Functionality summary
- Authenticate to EUMETSAT using API credentials.
- Build download specifications from the user's configuration (supports
    legacy flags and modern `download_products` entries).
- For each date in the configured date range, download either chunked FCI
    products (multiple .nc chunks per timestep) or single-file products.
- Organize downloads under a date/ROI-based output folder, skip already
    existing files, and support parallel downloads.

Configuration
- Credentials: either `consumer_key`/`consumer_secret` in `USER_CONFIG` or a
    credentials module/file referenced by `credentials_file`.
- ROI and `roi_name` control which chunks are relevant and where files are
    saved. Other keys: `timestamps_per_day`, `start_date`, `end_date`,
    `download_workers`, and per-product `base` output paths.

Usage
        python3 download/download_mtg_products.py

"""

# Import libraries
import datetime
import shutil
import fnmatch
import json
import glob
import os
import zipfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import importlib
import importlib.util
import numpy as np
import requests
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.wkt import loads
from shapely.geometry import Polygon, LineString
import eumdac
import sys

# read user config and credentials
sys.path.append("/home/Daniele/codes/MTG-EUMETSAT/")
from configs.config_loader import CONFIG as cfg

# credentials: prefer values from `user_config.py`; otherwise try loading
# the credentials file referenced in USER_CONFIG or a local `credentials` module.
consumer_key = cfg.get('consumer_key')
consumer_secret = cfg.get('consumer_secret')
if not (consumer_key and consumer_secret):
    cred_file = cfg.get('credentials_file')
    if cred_file:
        # Attempt to import the module by converting the path to a module name
        try:
            module_name = os.path.splitext(cred_file.replace('/', '.'))[0]
            creds = importlib.import_module(module_name)
            consumer_key = consumer_key or getattr(creds, 'consumer_key', None)
            consumer_secret = consumer_secret or getattr(creds, 'consumer_secret', None)
        except Exception:
            # fallback: try importing plain `credentials` from PYTHONPATH
            try:
                import credentials as creds
                consumer_key = consumer_key or getattr(creds, 'consumer_key', None)
                consumer_secret = consumer_secret or getattr(creds, 'consumer_secret', None)
            except Exception:
                pass
    else:
        try:
            import credentials as creds
            consumer_key = consumer_key or getattr(creds, 'consumer_key', None)
            consumer_secret = consumer_secret or getattr(creds, 'consumer_secret', None)
        except Exception:
            pass

# user ROI from config
user_roi = cfg.get('roi')


def build_download_specs(config):
    """Build the list of enabled product downloads from config.

    Prefer an explicit `download_products` list in `user_config.py`. If not
    present, attempt a minimal compatibility mapping from legacy flags.
    """
    product_specs = config.get('download_products')
    if product_specs:
        # ensure only enabled entries with required keys are returned
        return [spec for spec in product_specs if spec.get('enabled', True) and spec.get('collection_id') and spec.get('base')]

    # Compatibility fallback for older configs: map known keys to spec entries
    specs = []
    if config.get('download_fci_normal_res'):
        specs.append({
            'name': 'fci_normal_res',
            'enabled': True,
            'collection_id': config.get('collection_fci_norm_res_id') or config.get('collection_fci_norm_res_id'),
            'base': config.get('download_fci_normal_res_base'),
            'mode': 'chunked',
            'wkt_file': config.get('wkt_file', 'FCI_chunks.wkt'),
        })
    if config.get('download_fci_high_res'):
        specs.append({
            'name': 'fci_high_res',
            'enabled': True,
            'collection_id': config.get('collection_fci_high_res_id'),
            'base': config.get('download_fci_high_res_base'),
            'mode': 'chunked',
            'wkt_file': config.get('wkt_file', 'FCI_chunks.wkt'),
        })
    if config.get('download_cloud_masks'):
        specs.append({
            'name': 'cloud_mask',
            'enabled': True,
            'collection_id': config.get('collection_cloud_mask_id'),
            'base': config.get('download_cloud_masks_base'),
            'mode': 'single',
        })
    if config.get('download_oca'):
        specs.append({
            'name': 'oca',
            'enabled': True,
            'collection_id': config.get('collection_oca_id'),
            'base': config.get('download_oca_base'),
            'mode': 'single',
        })

    return [spec for spec in specs if spec.get('collection_id') and spec.get('base')]


def save_stream_to_folder(fsrc, output_folder, local_filename):
    """Save a stream to the target folder and skip if the file already exists."""
    destination = os.path.join(output_folder, local_filename)
    if os.path.exists(destination):
        print(f"File {local_filename} already exists in {output_folder}. Skipping download.")
        return False

    temporary_path = destination + '.tmp'
    with open(temporary_path, 'wb') as fdst:
        shutil.copyfileobj(fsrc, fdst)
    os.replace(temporary_path, destination)
    print(f"Saved file {destination}")
    return True


def download_single_file_products(selected_collection, dtstart, dtend, output_folder):
    """Download full-disk products that expose a single main file per product."""
    products = selected_collection.search(dtstart=dtstart, dtend=dtend)
    print(f"Found {len(products)} matching product(s).")

    for product in products:
        try:
            with product.open() as fsrc:
                local_filename = os.path.basename(fsrc.name)
                save_stream_to_folder(fsrc, output_folder, local_filename)
        except Exception as exc:
            print(f"Error downloading product {product}: {exc}")


def download_chunked_products(selected_collection, dtstart, dtend, chunk_ids, output_folder):
    """Download chunked products by filtering the matching chunk entries."""
    chunk_patterns = [f"_{cid}.nc" for cid in chunk_ids]
    products = selected_collection.search(dtstart=dtstart, dtend=dtend)
    print(f"Found {len(products)} matching timestep(s).")

    for product in products:
        for entry in product.entries:
            if any(pattern in entry for pattern in chunk_patterns):
                try:
                    with product.open(entry=entry) as fsrc:
                        local_filename = os.path.basename(fsrc.name)
                        save_stream_to_folder(fsrc, output_folder, local_filename)
                except Exception as exc:
                    print(f"Error downloading {entry}: {exc}")


def download_day_for_spec(datastore, spec, day_process):
    """Download one product spec for one day."""
    start = datetime.datetime(day_process.year, day_process.month, day_process.day, 0, 0)
    end = datetime.datetime(day_process.year, day_process.month, day_process.day, 23, 59)
    roi_name = cfg.get('roi_name', 'roi')
    output_base = spec['base']
    output_folder = create_output_folder(day_process, path_input=os.path.join(output_base, roi_name))

    collection = datastore.get_collection(spec['collection_id'])
    print(f"[{spec['name']}] Using collection {spec['collection_id']}")
    print(f"[{spec['name']}] Output folder: {output_folder}")

    if spec['mode'] == 'chunked':
        _, _, relevant_chunks = load_chunk_poligons(spec.get('wkt_file', 'FCI_chunks.wkt'), user_roi)
        print(f"[{spec['name']}] Defined ROI: {user_roi}")
        files_in = np.sort(glob.glob(os.path.join(output_folder, '*N__O_0*.nc')))
        expected = cfg['timestamps_per_day'] * len(relevant_chunks)
        print(f"[{spec['name']}] Files in output folder: {files_in}")
        if len(files_in) != expected:
            download_chunked_products(
                selected_collection=collection,
                dtstart=start,
                dtend=end,
                chunk_ids=relevant_chunks,
                output_folder=output_folder,
            )
            print(f"[{spec['name']}] Downloading files")
        else:
            print(f"[{spec['name']}] Already downloaded all the files")
    else:
        download_single_file_products(collection, start, end, output_folder)

    return True

#################################################

def main():
    """Main entrypoint: authenticate, create datastore, iterate dates, and download products per config."""

    # Feed the token object with your credentials, find yours at https://api.eumetsat.int/api-key/
    credentials = (consumer_key, consumer_secret)
    token = eumdac.AccessToken(credentials)
    print(f"This token '{token}' expires {token.expiration}") 
    #################################################

    # Create datastore object with with your token
    datastore = eumdac.DataStore(token)

    ###################################################
    # define set of days to process (from user_config)
    date_start = datetime.datetime.strptime(cfg['start_date'], '%Y.%m.%d')
    date_end = datetime.datetime.strptime(cfg['end_date'], '%Y.%m.%d')
    ###################################################

    # find all dates between date_start and date_end
    date_range = [date_start + datetime.timedelta(days=x) for x in range(0, (date_end - date_start).days + 1)]

    download_specs = build_download_specs(cfg)
    if not download_specs:
        print('No enabled products found in config.')
        return
    
    # loop on date_range
    for i_date, day_process in enumerate(date_range):
        
        print(f"Processing date {i_date + 1}/{len(date_range)}: {day_process.strftime('%Y-%m-%d')}")

        max_workers = max(1, int(cfg.get('download_workers', cfg.get('max_workers', 4))))
        worker_count = min(max_workers, len(download_specs))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(download_day_for_spec, datastore, spec, day_process) for spec in download_specs]
            for future in as_completed(futures):
                future.result()
    
    print("Download completed.")


            # Convert chunk polygons to a GeoDataFrame
            #gdf_chunks = gpd.GeoDataFrame({"chunk_id": list(chunk_polygons.keys()), "geometry": list(chunk_polygons.values())}, crs="EPSG:4326")

            # plot chunks
            #done = plot_chunks(gdf_chunks, roi_polygon)

            #print(f"Time window: from {start} to {end}.")

            # Run the function to download chunks in the time window
            #ownload_chunks_in_time_window(
            #    selected_collection=selected_collection, 
            #    dtstart=start,
            #    dtend=end, 
            ##    chunk_ids=relevant_chunks, 
            #    output_folder=output_folder)

            #print("Download completed.")

        #else:
           # print('Already downloaded all the files')
        #strasuka
        #
    # set date of yesterday
    #yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    #start = datetime.datetime(yesterday.year, yesterday.month, yesterday.day, 0, 0)
    #end = datetime.datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59)

    # build datetime for output for the day
    #time_str = str(yesterday.year)+str(yesterday.month).zfill(2)+str(yesterday.day).zfill(2)
    
    # create output directory for the selected day
    #output_folder = create_output_folder(yesterday, path_input='/data/trade_pc/mtg/fci/')

    # list files in output folder
    #files = np.sort(glob.glob(output_folder+'*N__O_0*.nc'))
    

        #print('Downloading files')
        ## Retrieve datasets that match the filter
       # products = selected_collection.search(
        #    dtstart=start,
        #    dtend=end)

       # print(f"{products.total_results} products found:")

        # Define ROI bounds (latitude and longitude bounding bbox)
        #print(f"Defined ROI: {user_roi}")

        # Load chunk polygons and find relevant chunks
        #chunk_polygons, roi_polygon, relevant_chunks = load_chunk_poligons("readers/FCI_chunks.wkt", user_roi)

        # Convert chunk polygons to a GeoDataFrame
        #gdf_chunks = gpd.GeoDataFrame({"chunk_id": list(chunk_polygons.keys()), "geometry": list(chunk_polygons.values())}, crs="EPSG:4326")

        # plot chunks
        #done = plot_chunks(gdf_chunks, roi_polygon)

        #print(f"Time window: from {start} to {end}.")

        # Run the function to download chunks in the time window
        #download_chunks_in_time_window(
        #    selected_collection=selected_collection, 
        #    dtstart=start,
        #    dtend=end, 
       #     chunk_ids=relevant_chunks, 
       #     output_folder=output_folder)

        #print("Download completed.")

    #else:
   #     print('Already downloaded all the files')
   

# This function converts the user-defined ROI to a Shapely Polygon
def convert_roi_to_poligon(roi):
    """
    Convert the user-defined ROI to a Shapely Polygon.
    """
    return Polygon([
        (roi["lon_min"], roi["lat_min"]),
        (roi["lon_min"], roi["lat_max"]),
        (roi["lon_max"], roi["lat_max"]),
        (roi["lon_max"], roi["lat_min"])
    ])

def create_output_folder(yesterday, path_input='/data/trade_pc/mtg/'):
    """
    script to create output path of the type yyyy/mm/ under the path_input folder
    input:
    - yesterday: datetime object of the date of yesterday
    - path_input: path to the folder where the output folder will be created
    output:    
    """
    # extract yyyy, mm, dd from the yesterday date
    yy = str(yesterday.year)
    mm = str(yesterday.month).zfill(2)
    dd = str(yesterday.day).zfill(2)
    
    # create output path under the provided base folder using os.path.join
    data_folder = os.path.join(path_input, yy, mm, dd)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
        print(f"Created directory {data_folder}")

    return data_folder


# This function checks if a product entry is part of the requested coverage
def get_coverage(coverage, filenames):
    chunks = []
    for pattern in coverage:
        for file in filenames:
            if fnmatch.fnmatch(file, pattern):
                chunks.append(file)
    return chunks

# Define the coverage patterns for the chunks
def load_chunk_poligons(wkt_file_path, user_roi):
    """
    Load chunk polygons from a WKT file.
    input:
    - wkt_file_path: path to the WKT file containing chunk footprints
    - user_roi: dictionary with user-defined ROI (lat_min, lat_max, lon_min, lon_max)
    output:
    - chunk_polygons: dictionary with chunk IDs and their corresponding Shapely Polygons
    - roi_polygon: Shapely Polygon representing the user-defined
    - relevant_chunks: list of chunk IDs that intersect with the user-defined ROI
    1. Load the WKT file containing chunk footprints.
    2. Parse the WKT data to create Shapely Polygons for each chunk.
    3. Convert the user-defined ROI to a Shapely Polygon.
    4. Check which chunks intersect with the ROI.
    5. Return the chunk polygons, ROI polygon, and relevant chunks.
    6. Print the number of loaded chunk footprints and relevant chunks.
    7. Return the chunk polygons, ROI polygon, and relevant chunks.
    8. Print the number of loaded chunk footprints and relevant chunks.
    9. Return the chunk polygons, ROI polygon, and relevant chunks.
    """
    # Load WKT chunk footprints
    with open(wkt_file_path, "r") as file:
        wkt_data = file.readlines()

    # Parse chunk polygons from WKT
    chunk_polygons = {}
    for line in wkt_data:
        chunk_id, wkt_poly = line.strip().split(',', 1)  # Extract chunk ID
        chunk_polygons[chunk_id] = loads(wkt_poly)  

    print(f"Loaded {len(chunk_polygons)} chunk footprints from WKT file.")
    
    # Convert user-defined ROI to a Shapely Polygon
    roi_polygon = convert_roi_to_poligon(user_roi)

    # Find chunks that intersect with ROI
    relevant_chunks = []
    for chunk_id, chunk_poly in chunk_polygons.items():
        if roi_polygon.intersects(chunk_poly):
            relevant_chunks.append(chunk_id)

    print(f"Found {len(relevant_chunks)} chunks intersecting the ROI: {relevant_chunks}")
    return chunk_polygons, roi_polygon, relevant_chunks


# function to download chunks of satellite data in a given time window
def download_chunks_in_time_window(selected_collection, dtstart, dtend, chunk_ids, output_folder):
    """
    Search for products in the given time window, download relevant .nc entries and trailer chunk (0041).
    """

    # Always ensure trailer chunk "0041" is included
    #chunk_ids.append("0041")

    if chunk_ids:
        chunk_patterns = [f"_{cid}.nc" for cid in chunk_ids]
    else:
        chunk_patterns = ["*_0000.nc"]  # pattern for full FCI L1c files (not chunked)

    # Products in time window
    products = selected_collection.search(dtstart=dtstart, dtend=dtend)
    print(f"Found {len(products)} matching timestep(s).")

    # Filter relevant entries
    for product in products:
        print(product)
        for entry in product.entries:
            if any(pattern in entry for pattern in chunk_patterns):
                try:
                    with product.open(entry=entry) as fsrc:
                        local_filename = os.path.basename(fsrc.name)
                        
                        # check if the file is already present in the output folder
                        if os.path.exists(os.path.join(output_folder, local_filename)):
                            print(f"File {local_filename} already exists in {output_folder}. Skipping download.")
                            continue
                        else:
                            print(f"Downloading file {local_filename}...")
                            with open(local_filename, 'wb') as fdst:
                                shutil.copyfileobj(fsrc, fdst)
                            print(f"Saved file {local_filename}")
                            
                            # move local file to destination folder
                            shutil.move(local_filename, output_folder)
                            print(f"Moved file to {output_folder}")
                        
                except Exception as e:
                    print(f"Error downloading {entry}: {e}")
    return()
  
# Function to plot the chunks
def plot_chunks(gdf_chunks, roi_polygon):
    """
    Plot the chunks and the user-defined ROI.
    """
    # Create a figure with Cartopy projection
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([-90, 90, -90, 90])
    ax.coastlines("50m", linewidth=0.25)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", linewidth=0.25)

    # Plot chunks with labels
    for i, row in gdf_chunks.iterrows():
        chunk_id, chunk_poly = row["chunk_id"], row["geometry"]
        if not chunk_poly.is_valid: continue
        ax.fill(*chunk_poly.exterior.xy, color=plt.cm.tab20.colors[i % 20], alpha=0.25, transform=ccrs.PlateCarree())

        # Label position inside polygon
        center_x = (chunk_poly.bounds[0] + chunk_poly.bounds[2]) / 2 
        vertical_line = LineString([(center_x, chunk_poly.bounds[1]), (center_x, chunk_poly.bounds[3])])
        label_y = vertical_line.intersection(chunk_poly).centroid.y
        ax.text(center_x, label_y, chunk_id, fontsize=6, transform=ccrs.PlateCarree(), ha="center", va="center")

    # Highlight ROI
    ax.plot(*roi_polygon.exterior.xy, color="red", linewidth=1, linestyle="dashed", transform=ccrs.PlateCarree())
    plt.title("MTG Chunk coverage extent and user ROI")
    
    # save figure
    fig.savefig("/net/ostro/mtg_plots/MTG_chunks.png", dpi=300, bbox_inches="tight")
    return()
         
 
# Main function to run the script
if __name__ == "__main__":
    main()

#997942