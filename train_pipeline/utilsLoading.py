import os

import pandas as pd


def create_validation_set(df, columns, rename_dict={}, return_site=True):
    if return_site:
        columns_to_select = [*columns, "Site", "Plot", "ECO_ID"]
        # ensure ECO_ID is integer
        df["ECO_ID"] = df["ECO_ID"].astype("Int32")
    else:
        columns_to_select = columns
    return df[columns_to_select].rename(columns=rename_dict)


def load_grounded_eo_validation_data() -> pd.DataFrame:
    """
    Load GROUNDED-EO validation data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing GROUNDED-EO validation data.
    """
    filename = os.path.join(
        "data",
        "validation_pipeline",
        "output",
        "grounded-eo",
        "EXPORT_GROUNDED-EO_all_s2_matchup.csv",
    )

    cols = [
        "Network",
        "Site",
        "Plot",
        "ECO_ID",
        "NLCD",
        "Latitude",
        "Longitude",
        "Date",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
        "sza",
        "vza",
        "phi",
        "LAI",
        "LAI_u",
        "LAIe",
        "LAIe_u",
        "FAPAR",
        "FAPAR_u",
        "FCOVER",
        "FCOVER_u",
        "date_difference",
        "uuid",
    ]
    df = pd.read_csv(
        filename,
        usecols=cols,
    )

    # if ECO_ID is NA, fill with values from the same Site
    df["ECO_ID"] = df.groupby("Site")["ECO_ID"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else -1)
    )

    # remove -1 ECO_ID rows
    df = df[df["ECO_ID"] != -1]

    df["ECO_ID"] = df["ECO_ID"].astype("Int32")

    # merge land cover classes to ensure consistency with Grounded-EO:
    # change dwarfScrub to shrubScrub
    # change woodyWetlands and emergentHerbaceousWetlands to wetlands
    # change sedgeHerbaceous to grasslandHerbaceous
    df["NLCD"] = df["NLCD"].replace(
        {
            "dwarfScrub": "shrubScrub",
            "woodyWetlands": "wetlands",
            "emergentHerbaceousWetlands": "wetlands",
            "sedgeHerbaceous": "grasslandHerbaceous",
        }
    )

    return df.rename(
        columns={
            "FAPAR": "fapar",
            "FAPAR_u": "fapar_std",
            "FCOVER": "fcover",
            "FCOVER_u": "fcover_std",
            "LAI": "lai",
            "LAI_u": "lai_std",
            "LAIe": "laie",
            "LAIe_u": "laie_std",
        }
    ).dropna()


def load_foliar_validation_data() -> pd.DataFrame:
    """
    Load foliar validation data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing foliar validation data.
    """
    filename = os.path.join(
        "data",
        "validation_pipeline",
        "output",
        "foliar",
        "EXPORT_NEON_foliar_reflectances_with_angles.csv",
    )

    cols = [
        "plotID",
        "ECO_ID",
        "uuid",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
        "sza",
        "vza",
        "phi",
        "chlorophyll_ab_mug_cm2",
        "carotenoid_mug_cm2",
        "ewt_cm",
        "leafMassPerArea_g_cm2",
        "date_difference",
    ]
    df = pd.read_csv(
        filename,
        usecols=cols,
    )

    # rename chlorophyll_ab_mug_cm2 to CHL, carotenoid_mug_cm2 to CAR, ewt_cm to EWT, leafMassPerArea_g_cm2 to LMA
    df = df.rename(
        columns={
            "plotID": "Plot",
            "chlorophyll_ab_mug_cm2": "CHL",
            "carotenoid_mug_cm2": "CAR",
            "ewt_cm": "EWT",
            "leafMassPerArea_g_cm2": "LMA",
            "sza": "tts",
            "vza": "tto",
            "phi": "psi",
        }
    )
    df["ECO_ID"] = df["ECO_ID"].astype("Int32")

    # remember to limit the min max:
    # trait-specific thresholds
    # if "EWT" in validation_sets:
    #     validation_sets["EWT"] = validation_sets["EWT"][
    #         validation_sets["EWT"]["EWT"] <= 0.1
    #     ]
    # if "CHL" in validation_sets:
    #     validation_sets["CHL"] = validation_sets["CHL"][
    #         validation_sets["CHL"]["CHL"] <= 100
    #     ]
    # if "LMA" in validation_sets:
    #     validation_sets["LMA"] = validation_sets["LMA"][
    #         validation_sets["LMA"]["LMA"] <= 0.05
    #     ]
    # if "lai" in validation_sets:
    #     validation_sets["lai"] = validation_sets["lai"][
    #         validation_sets["lai"]["lai"] <= 10
    #     ]
    return df


if __name__ == "__main__":
    grounded = load_grounded_eo_validation_data()
    foliar = load_foliar_validation_data()
    print(list(grounded.keys()))

    print(len(grounded["Plot"].unique()))
    print(len(grounded["Site"].unique()))
    print(len(grounded["ECO_ID"].unique()))
    print(len(grounded["uuid"].unique()))
