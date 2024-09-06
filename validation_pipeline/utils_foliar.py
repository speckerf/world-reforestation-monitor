import glob
import uuid

import pandas as pd
from loguru import logger


def merge_foliar_files(directories) -> pd.DataFrame:
    intermediate = []
    results = []
    for directory in directories:
        file_pattern = f"{directory}/*cfc_chlorophyll*.csv"
        file_pattern_sample = f"{directory}/*cfc_fieldData*.csv"
        file_pattern_lma = f"{directory}/*cfc_LMA*.csv"

        chl = glob.glob(file_pattern)
        sample = glob.glob(file_pattern_sample)
        lma = glob.glob(file_pattern_lma)

        # assert only one file is found for both patterns
        if len(chl) != 1 or len(sample) != 1 or len(lma) != 1:
            logger.warning(
                f"Expected 1 file for each pattern, found {len(chl)} files for  and {len(sample)} files for sample in folder {directory}"
            )
            continue

        cfc_chl_fields = [
            "sampleID",
            "extractChlAConc",
            "freshMass",
            "extractChlBConc",
            "extractCarotConc",
            "solventVolume",
            "sampleCondition",
            "dataQF",
            "handlingQF",
            "measurementQF",
        ]
        cfc_fieldData_fields = [
            "plotID",
            "sampleID",
            "nlcdClass",
            "decimalLatitude",
            "decimalLongitude",
            "sampleType",
            "scientificName",
            "crownPolygonID",
            "collectDate",
        ]
        cfc_lma_fields = [
            "sampleID",
            "lmaSampleCondition",
            "freshMass",
            "dryMass",
            "leafArea",
            "percentGreen",
            "leafMassPerArea",
            "dryMassFraction",
        ]
        df_chl = pd.read_csv(chl[0], usecols=cfc_chl_fields)
        df_sample = pd.read_csv(sample[0], usecols=cfc_fieldData_fields)
        df_lma = pd.read_csv(lma[0], usecols=cfc_lma_fields)

        # convert collectDate to datetime
        df_sample["collectDate"] = pd.to_datetime(df_sample["collectDate"])

        joined_df = pd.merge(df_chl, df_sample, on="sampleID", how="inner").merge(
            df_lma, on="sampleID", how="inner", suffixes=("_chl", "_lma")
        )

        # remove all columns with dataQF not NAN, and sampleCondition not 'OK'
        filtered_df = joined_df[
            (joined_df["dataQF"].isna())
            & (joined_df["sampleCondition"] == "OK")
            & (joined_df["handlingQF"] == "OK")
            & (joined_df["measurementQF"] == "OK")
            & (joined_df["lmaSampleCondition"] == "OK")
        ]

        intermediate.append(filtered_df)

        # group by plotID
        # filter out all groups with more than one unique decimalLatitude
        # check that dates are not more than 10 days apart
        # TODO: Optionally, we could join the cover the the corwn Shapefile, and do a weighted average based on crown size
        grouped_df = filtered_df.groupby("plotID")

        # filterings
        group_filtered_df = grouped_df.filter(
            lambda x: len(x["decimalLatitude"].unique()) == 1
            and ((x["collectDate"].max() - x["collectDate"].min()).days <= 10)
        )

        # calculate group_mean values of traits:
        mean_df = group_filtered_df.groupby("plotID").agg(
            {
                "extractChlAConc": "mean",
                "extractChlBConc": "mean",
                "extractCarotConc": "mean",
                "decimalLatitude": "first",
                "decimalLongitude": "first",
                "collectDate": "median",
                "freshMass_chl": "mean",
                "freshMass_lma": "mean",
                "leafMassPerArea": "mean",
                "dryMass": "mean",
                "leafArea": "mean",
                "solventVolume": "mean",
                "dryMassFraction": "mean",
                "percentGreen": "mean",
            }
        )

        # append to results
        results.append(mean_df)

    combined_data = pd.concat(results).reset_index()

    combined_data = combined_data.drop_duplicates()

    combined_data["uuid"] = [uuid.uuid4() for _ in range(len(combined_data.index))]

    return combined_data
