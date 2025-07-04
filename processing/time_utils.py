"""
Time utility functions for generating timestamp sequences and extracting datetime information
from filenames of satellite data products (e.g. MTG, CTH).
"""

from datetime import datetime, timedelta
from pathlib import Path


def compute_timestamps(start_date: str, end_date: str, step: int):
    """
    Generate a list of datetime objects between start and end dates at fixed intervals.

    Parameters:
        start_date (str): Start date in 'YYYY.MM.DD' format.
        end_date (str): End date in 'YYYY.MM.DD' format.
        step (int): Interval in minutes between timestamps.

    Returns:
        list[datetime]: List of datetime timestamps.
    """
    start = datetime.strptime(start_date, '%Y.%m.%d')
    end = datetime.strptime(end_date, '%Y.%m.%d')
    ts = []
    while start <= end:
        ts.append(start)
        start += timedelta(minutes=step)
    return ts


def extract_mtg_time(fname: Path, interval: int = 10) -> datetime:
    """
    Extract the rounded start time from an MTG filename.

    Assumes filename contains a segment like: '_<beginYYYYMMDDHHMMSS>_' at position 8 when split by '_'.

    Parameters:
        fname (Path): Path to the MTG file.
        interval (int): Minute-rounding interval (e.g., 10 for every 10 minutes).

    Returns:
        datetime: Rounded-down datetime object corresponding to the file's start time.
    """
    parts = fname.name.split('_')
    if len(parts) <= 8:
        raise ValueError(f"Unexpected MTG filename format: {fname.name}")

    begin_tag = parts[8]
    dt = datetime.strptime(begin_tag, "%Y%m%d%H%M%S")
    rounded_min = dt.minute - (dt.minute % interval)
    return dt.replace(minute=rounded_min, second=0, microsecond=0)


def extract_cth_time(fname: Path):
    """
    Extract timestamp from a CTH filename using a fixed string slice.

    Assumes timestamp is in the form: 'cth_YYYYMMDDHHMM.nc'

    Parameters:
        fname (Path): Path to the CTH file.

    Returns:
        datetime: Parsed datetime from filename.
    """
    tstr = fname.name.split('.')[0][5:17]
    return datetime.strptime(tstr, '%Y%m%d%H%M')
