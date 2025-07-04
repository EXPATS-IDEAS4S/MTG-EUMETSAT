from pathlib import Path
from datetime import datetime
from typing import Sequence
from collections import defaultdict
from typing import Callable, Iterable, Dict, List


def list_mtg_files(base: Path,
                   timestamps: Sequence[datetime],
                   pattern: str = "*.nat") -> list[Path]:
    """
    For each unique date in `timestamps`, looks under
      base/YYYY/MM/DD
    and returns all files matching `pattern`, sorted.
    """
    paths = []
    # Extract the unique date strings we need
    dates = {ts.date() for ts in timestamps}
    for d in sorted(dates):
        subdir = base / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}"
        if subdir.is_dir():
            paths.extend(subdir.rglob(pattern))
    return sorted(paths)

def list_cth_files(base: Path, pattern: str = "*.nc") -> list[Path]:
    """
    Lists all CTH files in the given base directory matching the pattern.
    TODO: complete this based on the actual CTH file structure.
    """
    #retunr empty list if base is None
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
    Groups files by the datetime returned from time_extractor.
    Converts each Path to str.

    Args:
        files: An iterable of Path objects.
        time_extractor: A function that maps a Path -> datetime.

    Returns:
        A dict mapping each datetime to a list of file paths (as str).
        Returns an empty dict if no files are provided.
    """
    if not files:
        return {}

    grouped = defaultdict(list)
    for f in files:
        t = time_extractor(f)
        grouped[t].append(str(f))  # Convert Path to str here

    return dict(grouped)