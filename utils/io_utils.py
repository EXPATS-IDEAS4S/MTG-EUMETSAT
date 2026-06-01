"""
I/O utilities for listing satellite input files and organizing them by timestamp.

Copied from `processing/io_utils.py`.
"""

from pathlib import Path
from datetime import datetime
from typing import Sequence, Callable, Iterable, Dict, List
from collections import defaultdict


def list_mtg_files(base: Path,
                   timestamps: Sequence[datetime],
                   pattern: str = "*.nat") -> list[Path]:
    paths = []
    dates = {ts.date() for ts in timestamps}
    for d in sorted(dates):
        subdir = f"{base}/{d.year:04d}/{d.month:02d}/{d.day:02d}"
        if Path(subdir).is_dir():
            paths.extend(Path(subdir).rglob(pattern))
    return sorted(paths)


def list_cth_files(base: Path, pattern: str = "*.nc") -> list[Path]:
    if base is None:
        return []
    if not Path(base).is_dir():
        raise ValueError(f"CTH base path {base} is not a directory.")
    return sorted(Path(base).rglob(pattern))


def build_time_map(
    files: Iterable[Path],
    time_extractor: Callable[[Path], datetime],
) -> Dict[datetime, List[str]]:
    if not files:
        return {}

    grouped = defaultdict(list)
    for f in files:
        t = time_extractor(f)
        grouped[t].append(str(f))

    return dict(grouped)
