#!/usr/bin/env python3

"""Daily high-resolution MTG pipeline without CMA or parallax."""

from __future__ import annotations

import sys

sys.path.append("/home/Daniele/codes/MTG-EUMETSAT/")

from configs.config_loader import CONFIG as cfg

from pipelines.common_pipeline import run_pipeline_for_day
from pipelines.pipeline_utils import iter_dates, make_datastore, parse_date


def main():
    datastore = make_datastore(cfg)
    start = parse_date(cfg["start_date"])
    end = parse_date(cfg["end_date"])
    variables = cfg.get("high_res_channels", cfg.get("channels", []))
    if not variables:
        raise ValueError("No high-resolution channels configured")

    for day in iter_dates(start, end):
        run_pipeline_for_day(
            config=cfg,
            datastore=datastore,
            day=day,
            primary_spec_name="fci_high_res",
            primary_reader=cfg.get("mtg_reader", "fci_l1c_nc"),
            primary_variables=variables,
            output_prefix="high_res",
            use_secondary=False,
            use_parallax=False,
        )


if __name__ == "__main__":
    main()
