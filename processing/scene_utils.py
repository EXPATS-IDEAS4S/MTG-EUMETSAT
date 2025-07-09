"""
Scene utility functions using Satpy for reading, loading, and cropping satellite imagery.

Includes:
- Scene creation from MTG and optional CTH data.
- Loading specified channels and cropping to a region of interest (ROI).
"""

from satpy import Scene
import xarray as xr

def has_corrupted_files(mtg_files):
    """
    Check if any of the given MTG NetCDF files are corrupted by attempting to open them with xarray.

    Parameters:
        mtg_files (list): List of file paths.

    Returns:
        bool: True if any file is corrupted (unreadable or empty), False otherwise.
    """
    for file in mtg_files:
        try:
            with xr.open_dataset(file) as ds:
                if not ds or len(ds.data_vars) == 0:
                    print(f"Empty dataset detected: {file}")
                    return True
        except Exception as e:
            print(f"Error opening {file}: {e}")
            return True
    
    return False



def make_scene(msg_file, cth_file, config):
    """
    Create a Satpy Scene from input files and config.

    Parameters:
        msg_file (Path or str): Path to the MTG input file.
        cth_file (Path or str): Path to the CTH file (used if parallax correction is enabled).
        config (dict): Configuration dictionary with reader settings.

    Returns:
        satpy.Scene: Initialized Scene with appropriate readers.
    """
    if config['parallax']:
        return Scene({
            config['mtg_reader']: [str(msg_file)],
            config['cth_reader']: [str(cth_file)]
        })
    else:
        return Scene(reader=config['mtg_reader'], filenames=msg_file)


def load_and_crop(scene: Scene, channels, roi):
    """
    Load specific channels and crop the Scene to the given geographic bounding box.

    Parameters:
        scene (satpy.Scene): A Satpy Scene object.
        channels (list of str): List of channels to load.
        roi (dict): Region of interest with keys 'lon_min', 'lat_min', 'lon_max', 'lat_max'.

    Returns:
        satpy.Scene: Cropped Scene with loaded channels.
    """
    scene.load(channels)
    crop = scene.crop(ll_bbox=(roi['lon_min'], roi['lat_min'], roi['lon_max'], roi['lat_max']))
    return crop
