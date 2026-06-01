"""Load YAML configuration and expose a `CONFIG` dictionary.

This module centralizes configuration loading so other scripts can import
`from configs.config_loader import CONFIG as cfg`.

If `config.yaml` is missing or PyYAML is not installed, it will raise a
clear error instructing the user to install dependencies or create the file.
"""
from pathlib import Path
import sys

CONFIG_PATH = Path(__file__).parent / "config.yaml"

try:
    import yaml
except Exception as e:
    raise ImportError("PyYAML is required to load configs/config.yaml. Install with `pip install pyyaml`." ) from e

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, 'r') as fh:
    CONFIG = yaml.safe_load(fh)
