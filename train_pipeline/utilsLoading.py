import os

import pandas as pd


def create_validation_set(df, columns, rename_dict, return_site):
    if return_site:
        columns_to_select = [*columns, "PLOT_ID", "ECO_ID"]
        rename_dict["PLOT_ID"] = "site"
    return df[columns_to_select].rename(columns=rename_dict)


def load_validation_data(return_site=False) -> dict:

    path_to_new_repo = "/Users/felix/Projects/OEMC/world-reforestation-monitor"
    foliar_traits = pd.read_csv(
        os.path.join(
            path_to_new_repo,
            "data",
            "validation_pipeline",
            "output",
            "EXPORT_NEON_foliar_reflectances_with_angles.csv",
        )
    )
    canopy_traits = pd.read_csv(
        os.path.join(
            path_to_new_repo,
            "data",
            "validation_pipeline",
            "output",
            # "EXPORT_COPERNICUS_GBOV_RM6,7_20240620120826_reflectances_with_angles.csv",
            "EXPORT_GBOV_RM6,7_20240620120826_all_reflectances_with_angles.csv",
        )
    )

    # rename columns sza, vza, phi to tts, tto, psi
    foliar_traits = foliar_traits.rename(
        columns={"sza": "tts", "vza": "tto", "phi": "psi", "plotID": "PLOT_ID"}
    )
    canopy_traits = canopy_traits.rename(
        columns={"sza": "tts", "vza": "tto", "phi": "psi"}
    )

    # traits = config["PIPELINE_PARAMS"]["TRAITS_TO_PREDICT"]
    traits = ["lai", "CHL", "CAR", "EWT", "LMA"]

    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    angles = ["tts", "tto", "psi"]

    bands_angles = [*bands, *angles]

    validation_sets = {
        "lai": create_validation_set(
            canopy_traits,
            bands_angles + ["LAIe_Warren"],
            {"LAIe_Warren": "lai"},
            return_site,
        ),
        "CHL": create_validation_set(
            foliar_traits,
            bands_angles + ["chlorophyll_ab_mug_cm2"],
            {"chlorophyll_ab_mug_cm2": "CHL"},
            return_site,
        ),
        "CAR": create_validation_set(
            foliar_traits,
            bands_angles + ["carotenoid_mug_cm2"],
            {"carotenoid_mug_cm2": "CAR"},
            return_site,
        ),
        "EWT": create_validation_set(
            foliar_traits, bands_angles + ["ewt_cm"], {"ewt_cm": "EWT"}, return_site
        ),
        "LMA": create_validation_set(
            foliar_traits,
            bands_angles + ["leafMassPerArea_g_cm2"],
            {"leafMassPerArea_g_cm2": "LMA"},
            return_site,
        ),
    }

    # divide reflectances by 10000 to get reflectances in the range [0, 1]
    for trait in traits:
        validation_sets[trait][bands] = validation_sets[trait][bands] / 10000
        # validation_sets[trait][angles] = convert_angles_to_cosines(
        #     validation_sets[trait][angles], angle_colnames=angles
        # )

    # set maximum for inclusion in validation set:
    # ewt: 0.1
    # chl: 100
    # lma: 0.05
    # lai: 10
    # validation_sets["EWT"]["EWT"] = np.minimum(validation_sets["EWT"]["EWT"], 0.1)
    # validation_sets["CHL"]["CHL"] = np.minimum(validation_sets["CHL"]["CHL"], 100)
    # validation_sets["LMA"]["LMA"] = np.minimum(validation_sets["LMA"]["LMA"], 0.05)

    # instead: discard values above these thresholds
    validation_sets["EWT"] = validation_sets["EWT"][
        validation_sets["EWT"]["EWT"] <= 0.1
    ]
    validation_sets["CHL"] = validation_sets["CHL"][
        validation_sets["CHL"]["CHL"] <= 100
    ]
    validation_sets["LMA"] = validation_sets["LMA"][
        validation_sets["LMA"]["LMA"] <= 0.05
    ]
    validation_sets["lai"] = validation_sets["lai"][validation_sets["lai"]["lai"] <= 10]

    return validation_sets


# import os

# import numpy as np
# import pandas as pd
# from loguru import logger


# def convert_angles_to_cosines(df: pd.DataFrame, angle_colnames: list) -> pd.DataFrame:
#     """
#     Convert angles to cosines
#     """
#     # check if columns are already in cosine form (range [-1, 1])
#     if all([df[col].between(-1, 1).all() for col in angle_colnames]):
#         logger.warning(
#             f"Converting angles to cosines: {angle_colnames} already in range [-1, 1]"
#         )
#         return df
#     else:
#         logger.debug(f"Converting angles to cosines: {angle_colnames}")
#         for angle in angle_colnames:
#             df[angle] = df[angle].apply(lambda x: np.cos(np.deg2rad(x)))
#         return df


# # def prepare_lut_data(df: pd.DataFrame) -> pd.DataFrame:

# #     # check if columns 'tto', 'tts', 'psi' are present in the dataframe
# #     assert all(
# #         [col in df.columns for col in ["tto", "tts", "psi"]]
# #     ), "Columns 'tto', 'tts', 'psi' are not present in the dataframe"

# #     # check that range of angles is correct (-1, 1), otherwise convert angles to cosines
# #     if not all([df[col].between(-1, 1).all() for col in ["tto", "tts", "psi"]]):
# #         df = convert_angles_to_cosines(df, angle_colnames=["tto", "tts", "psi"])


# def load_lut_data(path=None) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     # return convert_angles_to_cosines(df, angle_colnames=["tto", "tts", "psi"])
#     return df


# def load_validation_data(eco_to_filter=None) -> dict:
#     foliar_traits = pd.read_csv(
#         os.path.join(
#             "data",
#             "validation_pipeline",
#             "output",
#             "EXPORT_NEON_foliar_reflectances_with_angles.csv",
#         )
#     )
#     canopy_traits = pd.read_csv(
#         os.path.join(
#             "data",
#             "validation_pipeline",
#             "output",
#             "EXPORT_COPERNICUS_GBOV_RM6,7_20240620120826_reflectances_with_angles.csv",
#         )
#     )

#     if eco_to_filter is not None:
#         foliar_traits = foliar_traits[foliar_traits["ECO_ID"] == eco_to_filter]
#         canopy_traits = canopy_traits[canopy_traits["ECO_ID"] == eco_to_filter]

#         if len(foliar_traits) == 0:
#             logger.warning(f"No foliar validation data for eco {eco_to_filter}")
#         if len(canopy_traits) == 0:
#             logger.warning(f"No canopy validation data for eco {eco_to_filter}")

#     # rename columns sza, vza, phi to tts, tto, psi
#     foliar_traits = foliar_traits.rename(
#         columns={"sza": "tts", "vza": "tto", "phi": "psi"}
#     )
#     canopy_traits = canopy_traits.rename(
#         columns={"sza": "tts", "vza": "tto", "phi": "psi"}
#     )

#     # convert angles to cosines
#     # foliar_traits = convert_angles_to_cosines(
#     #     foliar_traits, angle_colnames=["tts", "tto", "psi"]
#     # )
#     # canopy_traits = convert_angles_to_cosines(
#     #     canopy_traits, angle_colnames=["tts", "tto", "psi"]
#     # )

#     # traits = config["PIPELINE_PARAMS"]["TRAITS_TO_PREDICT"]
#     traits = ["lai", "CHL", "CAR", "EWT", "LMA"]

#     bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
#     angles = ["tts", "tto", "psi"]
#     validation_sets = {
#         "lai": canopy_traits[[*bands, *angles, "LAIe_Warren"]].rename(
#             columns={"LAIe_Warren": "lai"}
#         ),
#         "CHL": foliar_traits[[*bands, *angles, "chlorophyll_ab_mug_cm2"]].rename(
#             columns={"chlorophyll_ab_mug_cm2": "CHL"}
#         ),
#         "CAR": foliar_traits[[*bands, *angles, "carotenoid_mug_cm2"]].rename(
#             columns={"carotenoid_mug_cm2": "CAR"}
#         ),
#         "EWT": foliar_traits[[*bands, *angles, "ewt_cm"]].rename(
#             columns={"ewt_cm": "EWT"}
#         ),
#         "LMA": foliar_traits[[*bands, *angles, "leafMassPerArea_g_cm2"]].rename(
#             columns={"leafMassPerArea_g_cm2": "LMA"}
#         ),
#     }

#     # divide reflectances by 10000 to get reflectances in the range [0, 1]
#     for trait in traits:
#         validation_sets[trait][bands] = validation_sets[trait][bands] / 10000
#         # validation_sets[trait][angles] = convert_angles_to_cosines(
#         #     validation_sets[trait][angles], angle_colnames=angles
#         # )

#     # set maximum for inclusion in validation set:
#     # ewt: 0.1
#     # chl: 100
#     # lma: 0.05
#     # validation_sets["EWT"]["EWT"] = np.minimum(validation_sets["EWT"]["EWT"], 0.1)
#     # validation_sets["CHL"]["CHL"] = np.minimum(validation_sets["CHL"]["CHL"], 100)
#     # validation_sets["LMA"]["LMA"] = np.minimum(validation_sets["LMA"]["LMA"], 0.05)

#     # instead: discard values above these thresholds
#     validation_sets["EWT"] = validation_sets["EWT"][
#         validation_sets["EWT"]["EWT"] <= 0.1
#     ]
#     validation_sets["CHL"] = validation_sets["CHL"][
#         validation_sets["CHL"]["CHL"] <= 100
#     ]
#     validation_sets["LMA"] = validation_sets["LMA"][
#         validation_sets["LMA"]["LMA"] <= 0.05
#     ]

#     return validation_sets


# if __name__ == "__main__":
#     a = load_validation_data()
#     load_lut_data()
#     load_lut_data()
