from datetime import datetime
from typing import Tuple

import ee
import pandas as pd
from loguru import logger


def get_start_end_date_phenology_for_ecoregion(
    ecoregion_id: int, year: int
) -> Tuple[str, str, int]:
    # ecoregion phenology table in data/phenology_pipeline/outputs/artificial_masked_w_amplitude_singleeco.csv
    phenology_table = pd.read_csv(
        "data/phenology_pipeline/outputs/artificial_masked_w_amplitude_singleeco.csv"
    )

    # get start and end data from phenology table
    phenology_entry_eco = phenology_table[phenology_table["ECO_ID"] == ecoregion_id]
    start_date_str = phenology_entry_eco["start_season"].values[0]
    end_date_str = phenology_entry_eco["end_season"].values[0]

    start_date = datetime.strptime(f"{year}-{start_date_str}", "%Y-%m-%d")
    end_date = datetime.strptime(f"{year}-{end_date_str}", "%Y-%m-%d")

    if end_date < start_date:
        logger.debug(
            f"Season spans year boundary for ecoregion {ecoregion_id}. New start date is {year-1}-{start_date_str}"
        )
        start_date = datetime.strptime(f"{year-1}-{start_date_str}", "%Y-%m-%d")
        end_date = datetime.strptime(f"{year}-{end_date_str}", "%Y-%m-%d")

    days_vegetative_period = int(
        phenology_entry_eco["days_vegetative_period"].values[0]
    )

    # if end_date is smaller than start_date, the season spans the year boundary, subtract 1 from year for start_date

    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    return start_date, end_date, days_vegetative_period


# Calculate weights based on linear distance from the midpoint, allowing weights to reach 0
def add_linear_weight(
    image: ee.Image, start_date: ee.Date, end_date: ee.Date, total_days: ee.Number
):
    # Calculate the difference in days from the start date.
    days_from_start = image.date().difference(start_date, "day").abs()
    days_to_end = image.date().difference(end_date, "day").abs()

    min_days_to_start_or_end = days_from_start.min(days_to_end)

    weight = min_days_to_start_or_end.divide(total_days.divide(2))

    # Design choice: cloudy pixel percentage is twice as important as the phenology weight.
    # cloud_percentage / 100 + (1 - weight) / 2
    # weight is 1 at the midpoint, 0 at the start and end
    # like that: an image in the exact center of the pheno period with cloud cover of 0.5 will have the same weight as an image at the start or end of the pheno period with cloud cover of 0
    cloud_pheno_weight_combined = (
        image.getNumber("CLOUDY_PIXEL_PERCENTAGE")
        .divide(100)
        .add((ee.Number(1).subtract(weight)).divide(2))
    )

    return image.set("cloud_pheno_image_weight", cloud_pheno_weight_combined)
