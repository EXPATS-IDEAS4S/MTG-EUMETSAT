"""
Scene utility functions using Satpy for reading, loading, and cropping satellite imagery.

Copied from `processing/scene_utils.py`.
"""

import xarray as xr
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def has_corrupted_files(mtg_files, verbose=False):
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
                if any(dim_size == 0 for dim_size in ds.sizes.values()):
                    log.warning(f"Dataset contains empty dimensions: {file}")
                    return True
        except Exception as e:
            log.exception(f"Error opening file {file}: {e}")
            return True

    return False


def get_channel(channels, parallax):
    new_channels = [None] * len(channels)
    for idx, channel in enumerate(channels):
        if parallax:
            new_channels[idx] = 'parallax_corrected_'+channel
        else:
            new_channels[idx] = channel
    return new_channels


def make_scene(msg_file, cth_file, config):
    from satpy import Scene

    if config['parallax']:
        return Scene({config['mtg_reader']: msg_file, config['cth_reader']: cth_file})
    else:
        return Scene(reader=config['mtg_reader'], filenames=msg_file)


def load_and_crop(scene, channels, roi, parallax=False):
    channels = get_channel(channels, parallax)
    scene.load(channels)
    crop = scene.crop(ll_bbox=(roi['lon_min'], roi['lat_min'], roi['lon_max'], roi['lat_max']))
    return crop
