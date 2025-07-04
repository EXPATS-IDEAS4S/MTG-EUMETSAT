from pathlib import Path

CONFIG = {
    "output_base": Path("/data/sat/msg/mtg/fci/"),
    "mtg_base": Path("/data/trade_pc/mtg/fci/"),
    "cth_base": None,
    "file_extension": "*.nc",
    "mtg_reader": "fci_l1c_nc",
    "cth_reader": None,
    "channels": ["vis_06", "ir_105"],
    "roi": {
        "lon_min": 5,
        "lat_min": 42,
        "lon_max": 16,
        "lat_max": 52
    },
    "parallax": False,
    "time_interval_min": 10,
    "start_date": "2025.06.30",
    "end_date": "2025.07.01",
    "regular_grid": False,
    "interp_method": "nearest",
    "grid_step_deg": [(0.015, 0.010), (0.0075, 0.005)],
    "compress_level": 9,
    "delete_chunks": False
}
