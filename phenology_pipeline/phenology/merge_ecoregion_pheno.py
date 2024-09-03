import datetime
import os
import subprocess
import tempfile

import ee
import numpy as np
import pandas as pd
from loguru import logger

# TODO: CHECK THAT CODE RUNS


def download_gcs_storage(folder_name: str) -> pd.DataFrame:
    # download all the csv files from google cloud storage gs://felixspecker/open-earth/phenology and save into tmp folder (that will be deleted after shutdown)
    # use gsutils to download all files from the bucket
    # gsutil cp -r gs://felixspecker/open-earth/phenology tmp
    with tempfile.TemporaryDirectory() as tmpdirname:
        logger.info("created temporary directory", tmpdirname)

        # folder_name = "artifical_masked"
        command = f"gsutil -m cp -r gs://felixspecker/open-earth/phenology/{folder_name} {tmpdirname}"
        # wait until the command is finished

        subprocess.run(command, shell=True, check=True)

        # list all the files in the tmp folder
        files = os.listdir(os.path.join(tmpdirname, folder_name))

        # load all the files into pandas dataframe
        dataframes = []
        for file in files:
            dataframes.append(
                pd.read_csv(
                    os.path.join(tmpdirname, folder_name, file),
                    header=0,
                )
            )

    # return pd.DataFrame
    return pd.concat(dataframes)


def main():
    MIN_DAYS = 45

    folder_name = "artificial_masked_w_amplitude_singleeco"

    df = download_gcs_storage(folder_name=folder_name)
    df_raw = df.copy()
    # all column swith '_1' or '_2' should be parsed as pd.to_datetime("2021-01-01") + pd.to_timedelta(df[col], unit="D")
    # and then converted to string: %m-%d
    tmp_columns = [
        col for col in df_raw.columns if col.endswith("_1") or col.endswith("_2")
    ]
    df_raw[tmp_columns] = df_raw[tmp_columns].apply(
        lambda x: pd.to_datetime("2021-01-01") + pd.to_timedelta(x, unit="D")
    )
    df_raw[tmp_columns] = df_raw[tmp_columns].apply(lambda x: x.dt.strftime("%m-%d"))
    # save the raw data
    df_raw.to_csv(
        os.path.join("data", "phenology_pipeline", "outputs", f"{folder_name}_raw.csv"),
        index=False,
    )

    eco_cols = ["ECO_ID", "ECO_NAME", "BIOME_NUM", "BIOME_NAME"]
    pheno_cols = [
        "MidGreenup_1",
        "MidGreendown_1",
        "Maturity_1",
        "Senescence_1",
        "Peak_1",
        "Greenup_1",
        "Dormancy_1",
    ]
    evi_cols = [
        "EVI_Amplitude_1",
        "EVI_Area_1",
        "EVI_Minimum_1",
    ]
    # evi_cols = []

    start_period_1 = "MidGreenup_1"
    end_period_1 = "MidGreendown_1"

    # only keep the columns that are needed
    df = df[[*eco_cols, *pheno_cols, *evi_cols]]

    # the columns are in day of the year format, so we need to convert them to a date within the year
    for col in pheno_cols:
        df[f"{col}"] = pd.to_datetime("2021-01-01") + pd.to_timedelta(df[col], unit="D")

    # if start_period_1 is after end_period_1, then subtract 365 days from start_period_1
    df.loc[df[start_period_1] > df[end_period_1], start_period_1] = df.loc[
        df[start_period_1] > df[end_period_1], start_period_1
    ] - pd.Timedelta(days=365)

    # save season length in days
    df["season_length_1"] = (df[end_period_1] - df[start_period_1]).dt.days
    # df["season_length_2"] = (df[end_period_2] - df[start_period_2]).dt.days

    df["start_season"] = df[start_period_1]
    df["end_season"] = df[end_period_1]

    # Decision:
    # - If ecoregions in Biome 1 (Tropical and Subtropical Moist Broadleaf Forests) or Biome 14 (Mangroves)
    # - And: if the minimum EVI is above 4000:
    # - And and the ratio of EVI_Minimum_1 / EVI_Amplitude_1 is above 2.5,
    # - And season_length_1 is below 180 days
    # --> then the season length should be extended to 360 days
    # add in code above
    df.loc[
        (df["BIOME_NUM"].isin([1, 14]))
        & (df["EVI_Minimum_1"] > 3900)
        & (df["EVI_Minimum_1"] / df["EVI_Amplitude_1"] > 2.5),
        "start_season",
    ] = pd.to_datetime("2021-01-01")
    df.loc[
        (df["BIOME_NUM"].isin([1, 14]))
        & (df["EVI_Minimum_1"] > 3900)
        & (df["EVI_Minimum_1"] / df["EVI_Amplitude_1"] > 2.5),
        "end_season",
    ] = pd.to_datetime("2021-12-31")

    # Manual decision:
    # set the following ecoregions to full year period:
    # 460, 493, 481, 518, 442,
    # 491, 492, 508, 486, 516,
    # 471, 451, 494, 501, 30,
    # 11, 23, 22, 6, 111, 21,
    # 7, 5, 26, 223, 284, 218,
    # 252, 288, 289, 230, 229
    # 241, 240, 303, 248, 231
    # 188 # parse from yaml

    # read pheno set manually:
    ecoregions_pheno_manually = pd.read_csv(
        os.path.join(
            "data",
            "phenology_pipeline",
            "outputs",
            "ecoregions_pheno_manual_full_year.csv",
        )
    )

    # temp season length
    df["season_length_temp"] = (df["end_season"] - df["start_season"]).dt.days

    # if season_length_temp is smaller than MIN_DAYS days, then extent the season to 60 days. to do so, first save the days_diff, and the extent in both directions by half of the days_diff
    df["days_diff"] = MIN_DAYS - df["season_length_temp"]

    # change format of days_diff to timedelta
    df["days_diff"] = pd.to_timedelta(df["days_diff"], unit="D")

    df.loc[df["season_length_temp"] < MIN_DAYS, "start_season"] = (
        df.loc[df["season_length_temp"] < MIN_DAYS, "start_season"]
        - df.loc[df["season_length_temp"] < MIN_DAYS, "days_diff"] / 2
    )
    df.loc[df["season_length_temp"] < MIN_DAYS, "end_season"] = (
        df.loc[df["season_length_temp"] < MIN_DAYS, "end_season"]
        + df.loc[df["season_length_temp"] < MIN_DAYS, "days_diff"] / 2
    )

    df["days_vegetative_period"] = (df["end_season"] - df["start_season"]).dt.days

    # convert MidGreenup_1 and Senescence_1 to string format %m-%d
    df["start_season"] = df["start_season"].dt.strftime("%m-%d")
    df["end_season"] = df["end_season"].dt.strftime("%m-%d")

    for col in pheno_cols:
        df[col] = df[col].dt.strftime("%m-%d")

    # drop columns that are not needed
    df.drop(["season_length_temp", "days_diff"], axis=1, inplace=True)

    # overwrite the start_season and end_season with the manually set values
    df.loc[
        df["ECO_ID"].isin(ecoregions_pheno_manually["ECO_ID"]),
        ["start_season"],
    ] = "01-01"
    df.loc[
        df["ECO_ID"].isin(ecoregions_pheno_manually["ECO_ID"]),
        ["end_season"],
    ] = "12-31"
    df.loc[
        df["ECO_ID"].isin(ecoregions_pheno_manually["ECO_ID"]),
        ["days_vegetative_period"],
    ] = 364

    # set reason column for changed ecoregions
    df.loc[df["ECO_ID"].isin(ecoregions_pheno_manually["ECO_ID"]), "reason"] = (
        "Manually set to full year due to high cloud cover"
    )

    # save the final data
    df.to_csv(
        os.path.join(
            "data",
            "phenology_pipeline/outputs/",
            f"{folder_name}.csv",
        ),
        index=False,
    )


def plotting():
    filename = "artificial_masked_w_amplitude_singleeco.csv"

    read_path = os.path.join("data", "phenology", filename)
    df = pd.read_csv(read_path)

    # plot minimum EVI
    df["EVI_Minimum_1"] = df["EVI_Minimum_1"].round(2)
    df["EVI_Amplitude_1"] = df["EVI_Amplitude_1"].round(2)
    df["EVI_Area_1"] = df["EVI_Area_1"].round(2)
    df["BIOME_NUM"] = df["BIOME_NUM"].astype(str)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # plot histogram of EVI_Minimum_1, EVI_Amplitude_1, EVI_Area_1
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    sns.histplot(data=df, x="EVI_Minimum_1", ax=ax[0])
    sns.histplot(data=df, x="EVI_Amplitude_1", ax=ax[1])
    sns.histplot(data=df, x="EVI_Area_1", ax=ax[2])
    plt.tight_layout()
    plt.show()

    # plt.show()

    # clear plot:

    # plot scatter plot: EVI_minimum vs EVI_Amplitude
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.scatterplot(
        data=df, x="EVI_Minimum_1", y="EVI_Amplitude_1", hue="BIOME_NUM", ax=ax
    )

    # plot seasonal length vs EVI_Minimum_1
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.scatterplot(
        data=df, x="season_length_1", y="EVI_Minimum_1", hue="BIOME_NUM", ax=ax
    )

    # plot seasonal length vs EVI_Amplitude_1
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.scatterplot(
        data=df, x="season_length_1", y="EVI_Amplitude_1", hue="BIOME_NUM", ax=ax
    )

    # plot EVI maximum vs EVI_Amplitude_1
    df["EVI_Maximum_1"] = df["EVI_Minimum_1"] + df["EVI_Amplitude_1"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.scatterplot(
        data=df, x="EVI_Maximum_1", y="EVI_Amplitude_1", hue="BIOME_NUM", ax=ax
    )

    # plot EVI ratio vs season length
    df["EVI_Ratio_1"] = df["EVI_Minimum_1"] / df["EVI_Amplitude_1"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.scatterplot(
        data=df, x="season_length_1", y="EVI_Ratio_1", hue="BIOME_NUM", ax=ax
    )

    # plot EVI ratio vs season length
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.scatterplot(data=df, x="EVI_Minimum_1", y="EVI_Ratio_1", hue="BIOME_NUM", ax=ax)

    # Decision:
    # - If ecoregions in Biome 1
    # - And: if the minimum EVI is above 4000:
    # - And and the ratio of EVI_Minimum_1 / EVI_Amplitude_1 is above 2.5,
    # - And season_length_1 is below 180 days
    # --> then the season length should be extended to 360 days
    # add in code above


if __name__ == "__main__":
    main()
    # plotting()
