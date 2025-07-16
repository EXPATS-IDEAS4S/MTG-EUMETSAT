"""
Scene utility functions using Satpy for reading, loading, and cropping satellite imagery.

Includes:
- Scene creation from MTG and optional CTH data.
- Loading specified channels and cropping to a region of interest (ROI).
"""

from satpy import Scene
import xarray as xr
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def has_corrupted_files(mtg_files, verbose=False):
    """
    Check if any of the given MTG NetCDF files are corrupted or unreadable.

    Parameters:
        mtg_files (list[str or Path]): List of file paths.
        verbose (bool): If True, logs additional debug information.

    Returns:
        bool: True if any file is corrupted (unreadable, empty, or missing), False otherwise.
    """
    if not mtg_files:
        if verbose:
            log.warning("No MTG files provided.")
        return True

    for file in mtg_files:
        path = Path(file)
        if not path.exists():
            log.error(f"File does not exist: {file}")
            return True

        try:
            with xr.open_dataset(path, engine='netcdf4') as ds:
                if ds is None or not ds.data_vars:
                    log.warning(f"Empty or invalid dataset: {file}")
                    return True
                # Optional: check if dimensions or variables are unexpectedly missing
                if any(dim_size == 0 for dim_size in ds.sizes.values()):
                    log.warning(f"Dataset contains empty dimensions: {file}")
                    return True
        except Exception as e:
            log.exception(f"Error opening file {file}: {e}")
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
