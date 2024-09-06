import glob
import os
import uuid

import ee
import pandas as pd
from loguru import logger

# def convert_lai_table_to_gee_featurecollection(df: pd.DataFrame, lat_col: str = 'lat', lon_col: str = 'lon', date_col: str = 'date') -> ee.FeatureCollection:
#     df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
#     df['year'] = df['date'].dt.year

#     # Convert DataFrame to FeatureCollection
#     features = []
#     for i, row in df.iterrows():
#         feature = ee.Feature(ee.Geometry.Point([row[lon_col], row[lat_col]]), {
#             'GBOV_ID': row['GBOV_ID'],
#             'PLOT_ID': row['PLOT_ID'],
#             'Site': row['Site'],
#             'year': row['year']
#         })
#         ee_date = ee.Date(row[date_col].strftime('%Y-%m-%d'))
#         feature = feature.set('system:time_start', ee_date)
#         features.append(feature)
#     fc = ee.FeatureCollection(features)
#     return fc


def merge_lai_files(directory, output_filename):
    # List all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    # Process all CSV files and combine them
    combined_data = pd.concat(
        [
            process_lai_csv_file(f)
            for f in csv_files
            if process_lai_csv_file(f) is not None
        ],
        ignore_index=True,
    )

    # filter out if up_flag or down_flag not equal to 0 / NA should be treated like 0
    combined_data = combined_data[
        (combined_data["up_flag"] == 0) | (combined_data["up_flag"].isna())
    ]
    combined_data = combined_data[
        (combined_data["down_flag"] == 0) | (combined_data["down_flag"].isna())
    ]

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


def process_lai_csv_file(file_path):
    try:
        # Read CSV file with specific delimiter and NA values
        df = pd.read_csv(file_path, delimiter=";", na_values=["", "NA", -999])

        # Convert 'TIME_IS' to datetime, extract year, and form a new column 'PLOT_ID_YEAR'
        if not all(col in df.columns for col in ["TIME_IS", "PLOT_ID", "Network"]):
            logger.error(f"Missing required columns in file: {file_path}")
        df["TIME_IS"] = pd.to_datetime(df["TIME_IS"].str.replace("T|Z", "", regex=True))
        df["YEAR"] = df["TIME_IS"].dt.year
        df["PLOT_ID_YEAR"] = df["PLOT_ID"].astype(str) + "_" + df["YEAR"].astype(str)

        if df["Network"].unique().size > 1:
            logger.error(f"Multiple networks found in file: {file_path}")

        if (
            df["Network"].unique()[0] in ["NEON", "FluxNet", "TERN"]
            and df["PLOT_ID"].unique()[0].find("BE-Bra") == -1
        ) or df["PLOT_ID"].unique()[0].find("VALE") == 0:

            ########
            # NEON / FluxNet measurements contain either both upward and downward measurements, or only one of both
            ########
            # assert that any of the columns contain either "_up" or "_down"
            if not any("_up" in col for col in df.columns) and not any(
                "_down" in col for col in df.columns
            ):
                logger.error(f"No 'up' or 'down' columns found in file: {file_path}")

            # Warn about missing 'up' or 'down' columns
            if not any("_down" in col for col in df.columns):
                logger.warning(
                    f"No 'down' columns found in file: {file_path}; Assuming 'up' only."
                )
            if not any("_up" in col for col in df.columns):
                logger.warning(
                    f"No 'up' columns found in file: {file_path}; Assuming 'down' only."
                )

            # Define columns that should be zero-filled if missing
            na_to_zero_columns = [
                "LAI_Warren_up",
                "LAI_Warren_down",
                "LAIe_Warren_up",
                "LAIe_Warren_down",
                "LAI_Miller_up",
                "LAI_Miller_down",
                "LAIe_Miller_up",
                "LAIe_Miller_down",
                "LAI_Warren_up_err",
                "LAI_Warren_down_err",
                "LAIe_Warren_up_err",
                "LAIe_Warren_down_err",
                "LAI_Miller_up_err",
                "LAI_Miller_down_err",
                "LAIe_Miller_up_err",
                "LAIe_Miller_down_err",
            ]

            # check if any column contains conflicting data types / replaces with NaN /
            # e.g. file COPERNICUS_GBOV_RM6,7_20240620120826/RM07/GBOV_RM07_OnaquiAult_069_20180321T000000Z_20220913T103000Z_284_ACR_V2.0.csv
            # contains values like "1.7)e-5" instead or correct numbers
            df = df.apply(
                lambda x: (
                    pd.to_numeric(x, errors="coerce")
                    if x.name in na_to_zero_columns
                    else x
                )
            )

            # Fill NA with zero for existing columns, and add missing columns with zeros
            for col in na_to_zero_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = 0

            # Calculate combined LAI and LAIe for Warren and Miller, handling potentially missing columns
            for measure in ["LAI", "LAIe"]:
                for author in ["Warren", "Miller"]:
                    up_col = f"{measure}_{author}_up"
                    down_col = f"{measure}_{author}_down"
                    df[f"{measure}_{author}"] = df.get(up_col, 0) + df.get(down_col, 0)

            # Calculate standard errors if error columns exist
            for prefix in ["LAI_Warren", "LAIe_Warren", "LAI_Miller", "LAIe_Miller"]:
                up_err = f"{prefix}_up_err"
                down_err = f"{prefix}_down_err"
                if up_err in df.columns and down_err in df.columns:
                    df[prefix + "_err"] = (df[up_err] ** 2 + df[down_err] ** 2) ** 0.5

                elif up_err in df.columns and not down_err in df.columns:
                    df[prefix + "_err"] = df[up_err]
                elif not up_err in df.columns and down_err in df.columns:
                    df[prefix + "_err"] = df[down_err]
                else:
                    logger.warning(
                        f"No error columns found for {prefix} in file: {file_path}"
                    )

            # Drop rows based on 'up_flag' or 'down_flag' conditions
            flags_exist = [
                "up_flag" if "up_flag" in df.columns else None,
                "down_flag" if "down_flag" in df.columns else None,
            ]
            flags_exist = [flag for flag in flags_exist if flag is not None]

        elif (
            df["Network"].unique()[0] in ["ICOS", "SM"]
            or df["PLOT_ID"].unique()[0].find("BE-Bra") == 0
        ):

            ########
            # ICOS and SM measurements contain only upward measurements, downward (understory) LAI is approximated by some model
            # so LAI_total_Miller = PAI_Miller + LAI_down
            ########
            drop_if_na_in_columns = [
                "PAIe_Miller",
                "PAIe_Miller_err",
                "PAI_Miller",
                "PAI_Miller_err",
                "Clumping_Miller",
                "Clumping_Miller_err",
                "PAIe_Warren",
                "PAIe_Warren_err",
                "PAI_Warren",
                "PAI_Warren_err",
                "Clumping_Warren",
                "Clumping_Warren_err",
                "LAI_down",
                "LAI_total_Miller",
                "LAI_total_Warren",
            ]
            df = df.dropna(subset=drop_if_na_in_columns)

            # add LAIe_total_Miller and LAIe_total_Warren
            # assuming that the understory vegetation has the same clumping factor as the overstory vegetation
            df["LAIe_Miller"] = (
                df["PAIe_Miller"] + df["LAI_down"] * df["Clumping_Miller"]
            )
            df["LAIe_Warren"] = (
                df["PAIe_Warren"] + df["LAI_down"] * df["Clumping_Warren"]
            )

            # rename LAI_total_Miller and LAI_total_Warren to LAI_Miller and LAI_Warren
            df = df.rename(
                columns={
                    "LAI_total_Miller": "LAI_Miller",
                    "LAI_total_Warren": "LAI_Warren",
                }
            )
        else:
            logger.error(f"Unknown network in file: {file_path}")

        return df

    except Exception as e:
        logger.error(f"Error processing file: {file_path}")
        logger.error(f"Error details: {e}")
        return None  # Return None for problematic files
