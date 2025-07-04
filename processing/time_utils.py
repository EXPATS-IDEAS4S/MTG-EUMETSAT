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
    """
    Extracts the start timestamp from an MTG FCI filename and rounds it
    down to the nearest `interval` minutes.

    Assumes the filename is of the form:
        ..._<beginYYYYMMDDHHMMSS>_<endYYYYMMDDHHMMSS>_...
    where splitting on '_' gives the begin timestamp at index 8.

    Args:
        fname: Path to the MTG chunk file.
        interval: Minute‑rounding interval (e.g. 10, 5, 1).

    Returns:
        A datetime object corresponding to the begin time, with:
          - seconds set to zero
          - minutes rounded down to nearest `interval`
    """
    parts = fname.name.split('_')
    if len(parts) <= 8:
        raise ValueError(f"Unexpected MTG filename format: {fname.name}")

    # parts[8] is the “begin” tag, e.g. "20250617083000"
    begin_tag = parts[8]
    dt = datetime.strptime(begin_tag, "%Y%m%d%H%M%S")

    # round minutes down to nearest `interval`
    rounded_min = dt.minute - (dt.minute % interval)
    return dt.replace(minute=rounded_min, second=0, microsecond=0)


def extract_cth_time(fname: Path):
    tstr = fname.name.split('.')[0][5:17]
    return datetime.strptime(tstr, '%Y%m%d%H%M')