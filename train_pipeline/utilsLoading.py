import os

import pandas as pd


def create_validation_set(df, columns, rename_dict={}, return_site=True):
    if return_site:
        columns_to_select = [*columns, "PLOT_ID", "ECO_ID"]
        rename_dict["PLOT_ID"] = "site"
    else:
        columns_to_select = columns
    return df[columns_to_select].rename(columns=rename_dict)


def load_validation_data(return_site=False) -> dict:

    foliar_traits = pd.read_csv(
        os.path.join(
            "data",
            "validation_pipeline",
            "output",
            "foliar",
            "EXPORT_NEON_foliar_reflectances_with_angles.csv",
        )
    )
    canopy_traits_lai = pd.read_csv(
        os.path.join(
            "data",
            "validation_pipeline",
            "output",
            "lai",
            # "EXPORT_COPERNICUS_GBOV_RM6,7_20240620120826_reflectances_with_angles.csv",
            "EXPORT_GBOV_RM6,7_20240620120826_all_reflectances_with_angles.csv",
        )
    )

    canopy_traits_fapar = pd.read_csv(
        os.path.join(
            "data",
            "validation_pipeline",
            "output",
            "fapar",
            # "EXPORT_COPERNICUS_GBOV_RM6,7_20240620120826_reflectances_with_angles.csv",
            "EXPORT_GBOV_RM6,7_20240620120826_all_reflectances_with_angles.csv",
        )
    )

    # rename columns sza, vza, phi to tts, tto, psi
    foliar_traits = foliar_traits.rename(
        columns={"sza": "tts", "vza": "tto", "phi": "psi", "plotID": "PLOT_ID"}
    )
    canopy_traits_lai = canopy_traits_lai.rename(
        columns={"sza": "tts", "vza": "tto", "phi": "psi"}
    )
    canopy_traits_fapar = canopy_traits_fapar.rename(
        columns={"sza": "tts", "vza": "tto", "phi": "psi"}
    )

    # traits = config["PIPELINE_PARAMS"]["TRAITS_TO_PREDICT"]
    traits = ["lai", "CHL", "CAR", "EWT", "LMA"]

    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    angles = ["tts", "tto", "psi"]

    bands_angles = [*bands, *angles]

    validation_sets = {
        "lai": create_validation_set(
            canopy_traits_lai,
            bands_angles + ["LAIe_Warren"],
            {"LAIe_Warren": "lai"},
            return_site,
        ),
        "fapar": create_validation_set(
            canopy_traits_fapar,
            bands_angles + ["FIPAR_total"],
            {"FIPAR_total": "fapar"},
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


if __name__ == "__main__":
    a = load_validation_data()
    print(list(a.keys()))
