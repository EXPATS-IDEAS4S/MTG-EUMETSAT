#!/usr/bin/env python3

"""Daily normal-resolution MTG pipeline with CMA attachment and optional parallax."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append("/home/Daniele/codes/MTG-EUMETSAT/")

from configs.config_loader import CONFIG as cfg

from pipelines.common_pipeline import run_pipeline_for_day
from pipelines.pipeline_utils import iter_dates, make_datastore, parse_date


def main():
    datastore = make_datastore(cfg)
    start = parse_date(cfg["start_date"])
    end = parse_date(cfg["end_date"])
    variables = cfg.get("normal_res_channels", cfg.get("channels", []))
    if not variables:
        raise ValueError("No normal-resolution channels configured")

    for day in iter_dates(start, end):
        run_pipeline_for_day(
            config=cfg,
            datastore=datastore,
            day=day,
            primary_spec_name="fci_normal_res",
            primary_reader=cfg.get("mtg_reader", "fci_l1c_nc"),
            primary_variables=variables,
            output_prefix="normal_res",
            secondary_spec_name="cloud_mask",
            secondary_variables=cfg.get("cma_variables", [cfg.get("cma_variable", "cloud_state"), "cma"]),
            secondary_output_name=cfg.get("cma_output_name", "cma"),
            use_secondary=True,
            use_parallax=bool(cfg.get("parallax", False)),
            cth_source_base=cfg.get("oca_source_base", cfg.get("cth_source_base", cfg.get("download_oca_base", cfg.get("cth_base")))),
        )


if __name__ == "__main__":
    main()
