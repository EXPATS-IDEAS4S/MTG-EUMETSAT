#!/usr/bin/env python3

"""Daily OCA pipeline that saves selected OCA variables and uploads the result."""

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
    variables = cfg.get("oca_variables", [cfg.get("cth_variable", "retrieved_cloud_top_height")])
    if not variables:
        raise ValueError("No OCA variables configured")

    for day in iter_dates(start, end):
        run_pipeline_for_day(
            config=cfg,
            datastore=datastore,
            day=day,
            primary_spec_name="oca",
            primary_reader=cfg.get("cth_reader", "fci_l2_nc"),
            primary_variables=variables,
            output_prefix="oca",
            use_secondary=False,
            use_parallax=False,
        )


if __name__ == "__main__":
    main()
