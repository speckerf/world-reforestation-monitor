import glob
import os
import uuid

import ee
import pandas as pd
from loguru import logger


def merge_fcover_files(directory, output_filename):
    # List all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    # Process all CSV files and combine them
    combined_data = pd.concat(
        [
            process_fcover_csv_file(f)
            for f in csv_files
            if process_fcover_csv_file(f) is not None
        ],
        ignore_index=True,
    )

    # only keep up_flag or down_flag equal to 0 or is NA
    combined_data = combined_data[
        (combined_data["up_flag"] == 0) | (combined_data["up_flag"].isna())
    ]
    combined_data = combined_data[
        (combined_data["down_flag"] == 0) | (combined_data["down_flag"].isna())
    ]

    # only keep data where percentage_valid is >0.6 or NA
    combined_data = combined_data[
        (combined_data["percentage_valid"] >= 100)
        | (combined_data["percentage_valid"].isna())
    ]

    # get df with missing lat or lon
    missing_lat_lon = combined_data[
        combined_data["Lat_IS"].isna() | combined_data["Lon_IS"].isna()
    ]

    # log sites with missing lat or lon
    if not missing_lat_lon.empty:
        logger.warning(
            f"Sites with missing Lat_IS or Lon_IS: {missing_lat_lon['Site'].unique()}"
        )
        logger.warning(
            f"Discarding {len(missing_lat_lon)} rows with missing Lat_IS or Lon_IS"
        )

    # filter rows with missing Lat_IS, Lon_IS or TIME_IS
    combined_data = combined_data.dropna(subset=["Lat_IS", "Lon_IS", "TIME_IS"])

    combined_data["system:time_start"] = (
        combined_data["TIME_IS"]
        .apply(lambda x: pd.Timestamp(x).timestamp() * 1000)
        .astype("int")
    )

    combined_data = combined_data.drop_duplicates()

    combined_data["uuid"] = [uuid.uuid4() for _ in range(len(combined_data.index))]

    # Write to disk
    combined_data.to_csv(output_filename, index=False)
    logger.info(f"Data merged and saved to {output_filename}")


def process_fcover_csv_file(file_path):
    try:

        # Read CSV file with specific delimiter and NA values
        df = pd.read_csv(file_path, delimiter=";", na_values=["", "NA", -999])

        # # if Site == 'Hainich': enter edebug model
        # if df["Site"].unique() == "Hainich":
        #     logger.debug(f"Processing file: {file_path}")

        # Convert 'TIME_IS' to datetime, extract year, and form a new column 'PLOT_ID_YEAR'
        if not all(col in df.columns for col in ["TIME_IS", "PLOT_ID", "Network"]):
            logger.error(f"Missing required columns in file: {file_path}")
        df["TIME_IS"] = pd.to_datetime(df["TIME_IS"].str.replace("T|Z", "", regex=True))
        df["YEAR"] = df["TIME_IS"].dt.year
        df["PLOT_ID_YEAR"] = df["PLOT_ID"].astype(str) + "_" + df["YEAR"].astype(str)

        if df["Network"].unique().size > 1:
            logger.error(f"Multiple networks found in file: {file_path}")
            return None  # Return None for problematic files

        # Define columns that should be zero-filled if missing
        na_to_zero_columns = [
            "FCOVER_up",
            "FCOVER_up_err",
            "FCOVER_down",
            "FCOVER_down_err",
        ]

        # assert that either contains only both up columns, both down columns, or all columns
        if not (
            all(col in df.columns for col in ["FCOVER_up", "FCOVER_up_err"])
            or all(col in df.columns for col in ["FCOVER_down", "FCOVER_down_err"])
            or all(col in df.columns for col in na_to_zero_columns)
        ):
            logger.error(f"Missing FCOVER columns in file: {file_path}")
            return None  # Return None for problematic files

        # check if any column contains conflicting data types / replaces with NaN /
        # contains values like "1.7)e-5" instead or correct numbers
        df = df.apply(
            lambda x: (
                pd.to_numeric(x, errors="coerce") if x.name in na_to_zero_columns else x
            )
        )

        # Fill NA with zero for existing columns, and add missing columns with zeros
        for col in [col for col in na_to_zero_columns if col in df.columns]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
            else:
                df[col] = 0

        if all(col in df.columns for col in ["FCOVER_total", "FCOVER_total_err"]):
            logger.trace(
                f"FCOVER_total and FCOVER_total_err already present in file: {file_path}"
            )
            return df
        else:
            logger.trace(
                f"Calculating FCOVER_total and FCOVER_total_err for file: {file_path}"
            )
            if all(
                col in df.columns
                for col in [
                    "FCOVER_up",
                    "FCOVER_up_err",
                    "FCOVER_down",
                    "FCOVER_down_err",
                ]
            ):
                # Calculate total FCover: FCover = FCover_{up} + (1 - FCover_{up}) FCover_{down}
                df["FCOVER_total"] = (
                    df["FCOVER_up"] + (1 - df["FCOVER_up"]) * df["FCOVER_down"]
                )

                # Calculate total FCover error: FCover_err = /sqrt{(1 - FCover_{down})^2 FCover_{up_err}^2 + (1 - FCover_{up})^2 FCover_{down_err}^2}
                df["FCOVER_total_err"] = (
                    ((1 - df["FCOVER_down"]) ** 2 * df["FCOVER_up_err"] ** 2)
                    + ((1 - df["FCOVER_up"]) ** 2 * df["FCOVER_down_err"] ** 2)
                ) ** 0.5

                # raise warning when: FCover_upp, FCover_up_err, Fcover_down > 0, and FCover_down_err == 0
                if (
                    (df["FCOVER_up"] > 0)
                    & (df["FCOVER_up_err"] > 0)
                    & (df["FCOVER_down"] > 0)
                    & (df["FCOVER_down_err"] == 0)
                ).any():
                    logger.warning(
                        f"FCOVER_down_err is zero for non-zero FCOVER_down in file: {file_path}"
                    )

            elif all(col in df.columns for col in ["FCOVER_up", "FCOVER_up_err"]):
                df["FCOVER_total"] = df["FCOVER_up"]
                df["FCOVER_total_err"] = df["FCOVER_up_err"]
            elif all(col in df.columns for col in ["FCOVER_down", "FCOVER_down_err"]):
                df["FCOVER_total"] = df["FCOVER_down"]
                df["FCOVER_total_err"] = df["FCOVER_down_err"]
            else:
                logger.error(f"Missing FCOVER columns in file: {file_path}")
                return None

        return df

    except Exception as e:
        logger.error(f"Error processing file: {file_path}")
        logger.error(f"Error details: {e}")
        return None  # Return None for problematic files
