"""
this module contains utility functions and color bars for plotting satellite data, deeine plotting domains a

"""

import matplotlib as mpl
import cmcrameri.cm as cmc
import cartopy.crs as ccrs
from pathlib import Path

# definition of domains of interest
domain_German_flood =  [ 5.,    9.,    48.,   52.  ] # minlon, maxlon, minlat, maxlat
domain_expats       =  [ 5.,   16.,    42.,   51.5 ] # minlon, maxlon, minlat, maxlat
domain_joyce        =  [ 6.,   6.5,    50.8,  51.3 ] # minlon, maxlon, minlat, maxlat   
domain_ACTA         =  [ 10.73,12.0,   46.3,  47.2 ] # minlon, maxlon, minlat, maxlat
domain_TEAMX      =  [ 9.9,   12.7,    45.5,   47.4  ] # minlon, maxlon, minlat, maxlat
# 45.48592, 9.92821 47.35537, 12.78054
#quicklook browser output path
quicklook_browser_output_path = '/data/obs/campaigns/teamx/quicklooks/mtg_fci/'
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
