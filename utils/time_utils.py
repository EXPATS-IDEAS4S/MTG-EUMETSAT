"""
Time utility functions for generating timestamp sequences and extracting datetime information
from filenames of satellite data products.

Copied from `processing/time_utils.py`.
"""

from datetime import datetime, timedelta
from pathlib import Path


def compute_timestamps(start_date: str, end_date: str, step: int):
    start = datetime.strptime(start_date, '%Y.%m.%d')
    end = datetime.strptime(end_date, '%Y.%m.%d')
    ts = []
    while start <= end:
        ts.append(start)
        start += timedelta(minutes=step)
    return ts


def extract_mtg_time(fname: Path, interval: int = 10) -> datetime:
    parts = fname.name.split('_')
    if len(parts) <= 8:
        raise ValueError(f"Unexpected MTG filename format: {fname.name}")

    begin_tag = parts[8]
    dt = datetime.strptime(begin_tag, "%Y%m%d%H%M%S")
    rounded_min = dt.minute - (dt.minute % interval)
    return dt.replace(minute=rounded_min, second=0, microsecond=0)


def extract_cth_time(fname: Path) -> datetime:
    time_str = fname.name.split('_')[7]
    return datetime.strptime(time_str, '%Y%m%d%H%M%S')
