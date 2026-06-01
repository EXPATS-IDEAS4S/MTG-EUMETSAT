"""
script to download OCA data from eumetsat
https://user.eumetsat.int/resources/user-guides/mtg-fci-l2-oca-data-guide
"""

# Import necessary libraries
import os
import glob
import json
import time
import shutil
import zipfile
import fnmatch
import requests
import datetime
import numpy as np
import matplotlib.pyplot as plt
import urllib3
from urllib3.exceptions import IncompleteRead

from shapely.wkt import loads
from shapely.geometry import Polygon, LineString
import geopandas as gpd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from eumdac import AccessToken, DataStore
from satpy import Scene

# Read credentials and region of interest (ROI)
from credentials import consumer_key, consumer_secret
from domain import user_roi
from download_fci_chunks import create_output_folder
#################################################


def safe_stream_download(fsrc, local_filename, chunk_size=1024*1024, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            with open(local_filename, 'wb') as fdst:
                while True:
                    chunk = fsrc.read(chunk_size)
                    if not chunk:
                        break
                    fdst.write(chunk)
            return  # Success
        except IncompleteRead as e:
            print(f"[Retry {attempt}/{max_retries}] IncompleteRead: {e}")
            time.sleep(2 * attempt)
        except Exception as e:
            print(f"[Retry {attempt}/{max_retries}] Unexpected error: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {local_filename} after {max_retries} attempts.")

def main():
    # Authenticate using EUMETSAT API key
    credentials = (consumer_key, consumer_secret)
    token = AccessToken(credentials)
    print(f"This token '{token}' expires on {token.expiration}")

    # Create data store object
    datastore = DataStore(token)

    # Product configuration
    product_name = 'OCA'
    product_code = 'EO:EUM:DAT:0684'
    product_path = '/sat_data/MTG_norm_res/oca/'

    # Define date range to process
    date_start = datetime.datetime(2025, 4, 6)
    date_end = datetime.datetime(2025, 4, 30)
    date_list = [date_start + datetime.timedelta(days=x) for x in range((date_end - date_start).days + 1)]

    # Select product collection
    selected_collection = datastore.get_collection(product_code)
    print(f"Collection {product_name} selected: {selected_collection}")

    for target_date in date_list:
        print(f"\nProcessing date: {target_date.strftime('%Y-%m-%d')}")

        start = datetime.datetime(target_date.year, target_date.month, target_date.day, 0, 0)
        end = datetime.datetime(target_date.year, target_date.month, target_date.day, 23, 59)

        # Create output directory for the current day
        output_folder = create_output_folder(target_date, path_input=product_path)
        print(f"Output folder for {product_name} created at {output_folder}")

        # Retrieve datasets for the current day
        products = selected_collection.search(dtstart=start, dtend=end)

        # Download each product
        for product in products:
            print('***************************************************')
            print(f"Downloading product: {product}")

            with product.open() as fsrc:
                local_filename = os.path.basename(fsrc.name)
                local_path = os.path.join(output_folder, local_filename)

                if os.path.exists(local_path):
                    print(f"File {local_filename} already exists in {output_folder}. Skipping.")
                    continue

                try:
                    print(f"Downloading file {local_filename} to {local_path}...")
                    safe_stream_download(fsrc, local_path)
                    print(f"Saved file to {local_path}")
                except Exception as e:
                    print(f"❌ Error downloading {local_filename}: {e}")
                    continue

    print("\n✅ All downloads complete.")



#################################################

if __name__ == "__main__":
    main()

#1005257





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
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import LineString
from satpy import Scene
# read credentials from file
from readers.credentials import *
from download.domain import user_roi
from download.mtg_chunks import create_output_folder, load_chunk_poligons, download_chunks_in_time_window
#################################################
def main():
    # Feed the token object with your credentials, find yours at https://api.eumetsat.int/api-key/
    credentials = (consumer_key, consumer_secret)
    token = eumdac.AccessToken(credentials)
    print(f”This token ‘{token}’ expires {token.expiration}“)
    #################################################
    # Create datastore object with with your token
    datastore = eumdac.DataStore(token)
    products_names = [‘OCA’]
    products_codes = []
    products_paths = [‘/data/sat/mtg/fci/oca/‘]
    # select yesterday date
    yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    start = datetime.datetime(yesterday.year, yesterday.month, yesterday.day, 0, 0)
    end = datetime.datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59)
    # calculate the domain
    # Define ROI bounds (latitude and longitude bounding bbox)
    print(f”Defined ROI: {user_roi}“)
    # Load chunk polygons and find relevant chunks
    chunk_polygons, roi_polygon, relevant_chunks = load_chunk_poligons(“readers/FCI_chunks.wkt”, user_roi)
    # reading product name to donwload
    path_prod = ‘/data/sat/mtg/fci/gii/’
    code = ‘EO:EUM:DAT:0683’#‘FCI-2-OCA-x-FD’
    prod = ‘GII’
    print(‘Dowloading product:’, ‘GII’)
    # read collection selected
    selected_collection = datastore.get_collection(code)
    print(f”Collection {prod} selected: {selected_collection}“)
    # create output directory for the selected day
    output_folder = create_output_folder(yesterday, path_input=path_prod)
    print(f”Output folder for {prod} created at {output_folder}“)
    # Retrieve datasets that match the filter
    products = selected_collection.search(
    dtstart=start,
    dtend=end)
    # Print found products
    for product in products:
            print(product)
    # Download all found products
    for product in products:
        print((‘***************************************************’))
        print(‘downloading product:’, product)
        with product.open() as fsrc:
            local_filename = os.path.basename(fsrc.name)
            # check if the file is already present in the output folder
            if os.path.exists(os.path.join(output_folder, local_filename)):
                print(f”File {local_filename} already exists in {output_folder}. Skipping download.“)
                continue
            else:
                print(f”Downloading file {local_filename}...“)
                with open(local_filename, ‘wb’) as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                print(f”Saved file {local_filename}“)
                # move local file to destination folder
                shutil.move(local_filename, output_folder)
                print(f”Moved file to {output_folder}“)
    # postprocessing script
    #loop on all files in the output folder
if __name__ == “__main__“:
    main()

"""