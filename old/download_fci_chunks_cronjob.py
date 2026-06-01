#!/usr/bin/env python3

""" Script to read MTG data following what indicated in the EUMETSAT API documentation
https://user.eumetsat.int/resources/user-guides/mtg-data-access-guide
The script downloads the data of the previous day with respect to the execution date.
It downloads the region of interest defined by the user. If the file is already downloaded, 
it is skipped.
Once download is over, it reads the chunks 

lauch command with nohup
nohup python3 download/download_fci_chunks.py > download_fci_chunks.log 2>&1 &

"""

# Import libraries
import eumdac
import datetime
import shutil
import requests
import time
import numpy as np
import eumdac
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
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import LineString
from satpy import Scene
from datetime import datetime, timezone, timedelta

# read credentials from file
from credentials import *   
from domain import user_roi

#################################################

def main():

    # Feed the token object with your credentials
    credentials = (consumer_key, consumer_secret)
    token = eumdac.AccessToken(credentials)
    print(f"This token '{token}' expires {token.expiration}")

    # Create datastore object
    datastore = eumdac.DataStore(token)

    selected_collection = datastore.get_collection('EO:EUM:DAT:0665')
    print(selected_collection)

    # Define fixed start date
    date_start = datetime.datetime(2025, 7, 3)

    # Compute yesterday dynamically
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    date_end = datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc)

    # Create date range from start to yesterday
    date_range = [date_start + datetime.timedelta(days=x) for x in range(0, (date_end - date_start).days + 1)]

    # Load chunk polygons and relevant chunks once (outside loop)
    chunk_polygons, roi_polygon, relevant_chunks = load_chunk_poligons("FCI_chunks.wkt", user_roi)
    print(f"Defined ROI: {user_roi}")

    for i_date, day_process in enumerate(date_range):

        print(f"Processing date {i_date + 1}/{len(date_range)}: {day_process.strftime('%Y-%m-%d')}")

        # Define time window for the day
        start = datetime.datetime(day_process.year, day_process.month, day_process.day, 0, 0)
        end = datetime.datetime(day_process.year, day_process.month, day_process.day, 23, 59)

        # Create output directory for the day
        output_folder = create_output_folder(day_process, path_input='/data/trade_pc/mtg/fci/')

        # List downloaded chunk files in output folder
        downloaded_files = set(os.path.basename(f) for f in glob.glob(os.path.join(output_folder, '*N__O_0*.nc')))
        print(f"{len(downloaded_files)} files already downloaded for {day_process.strftime('%Y-%m-%d')}.")

        # Build expected filenames for relevant chunks
        expected_filenames = set()
        # For each chunk, the file pattern includes chunk id like *_<chunk_id>.nc
        # We'll search products to find exact file names

        # Instead of guessing filenames, use collection search to find products for the day
        products = selected_collection.search(dtstart=start, dtend=end)
        print(f"{products.total_results} products found for {day_process.strftime('%Y-%m-%d')}.")

        # Gather expected files for relevant chunks from products metadata
        expected_filenames = set()
        for product in products:
            for entry in product.entries:
                for chunk_id in relevant_chunks:
                    if f"_{chunk_id}.nc" in entry:
                        expected_filenames.add(os.path.basename(entry))

        # Determine missing files
        missing_files = expected_filenames - downloaded_files

        if missing_files:
            print(f"{len(missing_files)} missing files to download for {day_process.strftime('%Y-%m-%d')}.")
            download_chunks_in_time_window(
                selected_collection=selected_collection,
                dtstart=start,
                dtend=end,
                chunk_ids=relevant_chunks,
                output_folder=output_folder)
        else:
            print(f"All files already downloaded for {day_process.strftime('%Y-%m-%d')}.")

    print("Download completed for all days.")

   

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
    
    # create output path  under the data folder
    data_folder = path_input+yy+'/'+mm+'/'+dd+'/'
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created directory {data_folder}")   

    return(data_folder)


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

    chunk_patterns = [f"_{cid}.nc" for cid in chunk_ids]

    # Products in time window
    products = selected_collection.search(dtstart=dtstart, dtend=dtend)
    print(f"Found {len(products)} matching timestep(s).")

    # Filter relevant entries
    for product in products:
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

#4016328