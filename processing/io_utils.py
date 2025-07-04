"""
I/O utilities for listing satellite input files and organizing them by timestamp.

Functions:
- list_mtg_files: Lists MTG files for specific dates under a structured directory tree.
- list_cth_files: Lists all matching CTH files in a directory.
- build_time_map: Groups files by datetime using a custom extractor.
"""

from pathlib import Path
from datetime import datetime
from typing import Sequence, Callable, Iterable, Dict, List
from collections import defaultdict


def list_mtg_files(base: Path,
                   timestamps: Sequence[datetime],
                   pattern: str = "*.nat") -> list[Path]:
    """
    Lists MTG files from subdirectories matching dates in the timestamps list.

    Each date is expected in the structure: base/YYYY/MM/DD/.

    Parameters:
        base (Path): Root directory of MTG files.
        timestamps (Sequence[datetime]): List of timestamps to extract unique dates.
        pattern (str): Glob pattern for matching file names.

    Returns:
        list[Path]: Sorted list of matching file paths.
    """
    paths = []
    dates = {ts.date() for ts in timestamps}
    for d in sorted(dates):
        subdir = base / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}"
        if subdir.is_dir():
            paths.extend(subdir.rglob(pattern))
    return sorted(paths)


def list_cth_files(base: Path, pattern: str = "*.nc") -> list[Path]:
    """
    Lists all CTH files in a directory tree matching a given pattern.

    Parameters:
        base (Path): Root directory of CTH files.
        pattern (str): Glob pattern for matching file names.

    Returns:
        list[Path]: Sorted list of CTH files.

    Raises:
        ValueError: If base is not a directory.
    """
    if base is None:
        return []
    if not base.is_dir():
        raise ValueError(f"CTH base path {base} is not a directory.")
    return sorted(base.rglob(pattern))


def build_time_map(
    files: Iterable[Path],
    time_extractor: Callable[[Path], datetime],
) -> Dict[datetime, List[str]]:
    """
    Maps files to timestamps using a custom time extractor.

    Parameters:
        files (Iterable[Path]): Paths to input files.
        time_extractor (Callable): Function to extract datetime from each file path.

    Returns:
        Dict[datetime, List[str]]: Dictionary mapping each timestamp to file paths (as strings).
    """
    if not files:
        return {}

    grouped = defaultdict(list)
    for f in files:
        t = time_extractor(f)
        grouped[t].append(str(f))  # Use string paths for serialization or logging

    return dict(grouped)
