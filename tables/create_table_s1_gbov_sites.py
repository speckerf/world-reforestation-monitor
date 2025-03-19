import pandas as pd

from train_pipeline.utilsLoading import load_validation_data

# SITE_DICT = {
#     "BART": "BartlettExperimentalForest",
#     "BE-Bra": "Brasschaat",
#     "BE-Vie": "Vielsalm",
#     "BLAN": "BlandyExperimentalFarm",
#     "CPER": "CentralPlainsExperimentalRange",
#     "DELA": "DeadLake",
#     "DSNY": "DisneyWildernessPreserve",
#     "FR-Fon": "Fontainebleau-Barbeau",
#     "GUAN": "GuanicaForest",
#     "HAIN": "Hainich",
#     "HARV": "HarvardForest",
#     "DE-HoH": "HohesHolz",
#     "JERC": "JonesEcologicalResearchCenter",
#     "JORN": "Jornada",
#     "KONA": "KonzaPrairieBiologicalStation",
#     "LAJA": "LajasExperimentalStation",
#     "LITC": "LitchfieldSavanna",
#     "MOAB": "Moab",
#     "NIWO": "NiwotRidgeMountainResearchStation",
#     "ONAQ": "OnaquiAult",
#     "ORNL": "OakRidge",
#     "OSBS": "OrdwaySwisherBiologicalStation",
#     "SCBI": "SmithsonianConservationBiologyInstitute",
#     "SERC": "SmithsonianEnvironmentalResearchCenter",
#     "SRER": "SantaRita",
#     "STEI": "SteigerwaldtLandServices",
#     "STER": "NorthSterling",
#     "TALL": "TalladegaNationalForest",
#     "TUMB": "Tumbarumba",
#     "UNDE": "Underc",
#     "VALE": "ValenciaAnchorStation",
#     "WOMB": "WombatStringbarkEucalypt",
#     "WOOD": "Woodworth",
# }

# NETWORK_DCIT = {
#     "BART": "NEON",
#     "BE-Bra": "FluxNet",
#     "BE-Vie": "ICOS",
#     "BLAN": "NEON",
#     "CPER": "NEON",
#     "DELA": "NEON",
#     "DSNY": "NEON",
#     "FR-Fon": "ICOS",
#     "GUAN": "NEON",
#     "HAIN": "FluxNet",
#     "HARV": "NEON",
#     "DE-HoH": "ICOS",
#     "JERC": "NEON",
#     "JORN": "NEON",
#     "KONA": "NEON",
#     "LAJA": "NEON",
#     "LITC": "NEON",
#     "MOAB": "NEON",
#     "NIWO": "NEON",
#     "ONAQ": "NEON",
#     "ORNL": "NEON",
#     "OSBS": "NEON",
#     "SCBI": "NEON",
#     "SERC": "NEON",
#     "SRER": "NEON",
#     "STEI": "NEON",
#     "STER": "NEON",
#     "TALL": "NEON",
#     "TUMB": "NEON",
#     "UNDE": "NEON",
#     "VALE": "NEON",
#     "WOMB": "NEON",
#     "WOOD": "NEON",
# }

# "IGBP": "NEON",

IGBP_ABBREVIATIONS = {
    "ENF": "Evergreen Needleleaf Forests",
    "EBF": "Evergreen Broadleaf Forests",
    "DBF": "Deciduous Broadleaf Forests",
    "DNF": "Deciduous Needleleaf Forests",
    "MF": "Mixed Forests",
    "OSH": "Open Shrublands",
    "CSH": "Closed Shrublands",
    "WSA": "Woody Savannas",
    "SAV": "Savannas",
    "GRA": "Grasslands",
    "WET": "Wetlands",
    "CRO": "Croplands",
    "URB": "Urban",
}

SITE_DICT = {
    "BART": {"Site": "BartlettExperimentalForest", "Network": "NEON", "IGBP": "MF"},
    "FR-Bil": {"Site": "Bilos", "Network": "ICOS", "IGBP": "ENF"},
    "BLAN": {"Site": "BlandyExperimentalFarm", "Network": "NEON", "IGBP": "DBF"},
    "BE-Bra": {"Site": "Brasschaat", "Network": "FluxNet", "IGBP": "MF"},
    "BE-Vie": {"Site": "Vielsalm", "Network": "ICOS", "IGBP": "MF"},
    "CPER": {
        "Site": "CentralPlainsExperimentalRange",
        "Network": "NEON",
        "IGBP": "GRA",
    },
    "DE-HoH": {"Site": "HohesHolz", "Network": "ICOS", "IGBP": "DBF"},
    "DELA": {"Site": "DeadLake", "Network": "NEON", "IGBP": "DBF"},
    "DSNY": {"Site": "DisneyWildernessPreserve", "Network": "NEON", "IGBP": "OSH"},
    "FI-Hyy": {"Site": "Hyytiala", "Network": "SM", "IGBP": "ENF"},
    "FR-Fon": {"Site": "Fontainebleau-Barbeau", "Network": "ICOS", "IGBP": "DBF"},
    "GUAN": {"Site": "GuanicaForest", "Network": "NEON", "IGBP": "EBF"},
    "HAIN": {"Site": "Hainich", "Network": "FluxNet", "IGBP": "MF"},
    "HARV": {"Site": "HarvardForest", "Network": "NEON", "IGBP": "MF"},
    "IT-SR2": {"Site": "SanRossore2", "Network": "ICOS", "IGBP": "ENF"},
    "JERC": {"Site": "JonesEcologicalResearchCenter", "Network": "NEON", "IGBP": "ENF"},
    "JORN": {"Site": "Jornada", "Network": "NEON", "IGBP": "OSH"},
    "KONA": {"Site": "KonzaPrairieBiologicalStation", "Network": "NEON", "IGBP": "CRO"},
    "LAJA": {"Site": "LajasExperimentalStation", "Network": "NEON", "IGBP": "GRA"},
    "LITC": {"Site": "LitchfieldSavanna", "Network": "TERN", "IGBP": "WSA"},
    "TEAK": {"Site": "LowerTeakettle", "Network": "NEON", "IGBP": "ENF"},
    "MOAB": {"Site": "Moab", "Network": "NEON", "IGBP": "OSH"},
    "NIWO": {
        "Site": "NiwotRidgeMountainResearchStation",
        "Network": "NEON",
        "IGBP": "ENF",
    },
    "ONAQ": {"Site": "OnaquiAult", "Network": "NEON", "IGBP": "OSH"},
    "ORNL": {"Site": "OakRidge", "Network": "NEON", "IGBP": "MF"},
    "OSBS": {
        "Site": "OrdwaySwisherBiologicalStation",
        "Network": "NEON",
        "IGBP": "ENF",
    },
    "SCBI": {
        "Site": "SmithsonianConservationBiologyInstitute",
        "Network": "NEON",
        "IGBP": "MF",
    },
    "SERC": {
        "Site": "SmithsonianEnvironmentalResearchCenter",
        "Network": "NEON",
        "IGBP": "CRO",
    },
    "SE-Htm": {"Site": "Hyltemossa", "Network": "ICOS", "IGBP": "ENF"},
    "SOAP": {"Site": "SoaprootSaddle", "Network": "NEON", "IGBP": "ENF"},
    "SRER": {"Site": "SantaRita", "Network": "NEON", "IGBP": "CSH"},
    "STEI": {"Site": "SteigerwaldtLandServices", "Network": "NEON", "IGBP": "DBF"},
    "STER": {"Site": "NorthSterling", "Network": "NEON", "IGBP": "GRA"},
    "TALL": {"Site": "TalladegaNationalForest", "Network": "NEON", "IGBP": "ENF"},
    "TUMB": {"Site": "Tumbarumba", "Network": "FluxNet", "IGBP": "EBF"},
    "UNDE": {"Site": "Underc", "Network": "NEON", "IGBP": "MF"},
    "VALE": {"Site": "ValenciaAnchorStation", "Network": "SM", "IGBP": "CRO"},
    "WOMB": {"Site": "WombatStringbarkEucalypt", "Network": "TERN", "IGBP": "EBF"},
    "WOOD": {"Site": "Woodworth", "Network": "NEON", "IGBP": "GRA"},
}

BIOME_ABBREVIATIONS = {
    "Temperate Broadleaf & Mixed Forests": "Temp. Br. & Mix. For.",
    "Tundra": "Tundra",
    "Temperate Grasslands, Savannas & Shrublands": "Temp. Grassl., Sav. & Shrub.",
    "Boreal Forests/Taiga": "Boreal For./Taiga",
    "Tropical & Subtropical Dry Broadleaf Forests": "Tr. & Subtr. Dry Br. For.",
    "Mediterranean Forests, Woodlands & Scrub": "Med. For., Woodl. & Scrub",
    "Deserts & Xeric Shrublands": "Deserts",
    "Temperate Conifer Forests": "Temp. Conif. For.",
    "Montane Grasslands & Shrublands": "Montane Grassl. & Shrub.",
    "Flooded Grasslands & Savannas": "Flooded Grassl. & Sav.",
    "Mangroves": "Mangroves",
}


def main():
    lai_val = load_validation_data(return_site=True)["lai"]
    fapar_val = load_validation_data(return_site=True)["fapar"]
    fcover_val = load_validation_data(return_site=True)["fcover"]

    # add Site:
    lai_val["Abbr"] = lai_val["site"].str.split("_").str[0]
    fapar_val["Abbr"] = fapar_val["site"].str.split("_").str[0]
    fcover_val["Abbr"] = fcover_val["site"].str.split("_").str[0]

    # create new table: with Site, Abbr., Network, Ecoregion, Biome, Latitude, Longitude, # Plots, # LAI, # FAPAR, # FCOVER

    # site id
    columns = ["site", "ECO_ID"]
    base_site = lai_val[columns].drop_duplicates()

    # rename site to PLOT_ID
    base_site = base_site.rename(columns={"site": "PLOT_ID"})

    # extract Abbr.: everything before '_'
    base_site["Abbr"] = base_site["PLOT_ID"].str.split("_").str[0]

    # Add 'Site' column
    base_site["Site"] = base_site["Abbr"].map(
        {abbr: SITE_DICT[abbr]["Site"] for abbr in SITE_DICT}
    )

    # Add 'Network' column
    base_site["Network"] = base_site["Abbr"].map(
        {abbr: SITE_DICT[abbr]["Network"] for abbr in SITE_DICT}
    )

    # add number of plots: count per Site / only unique plots
    base_site["# Plots"] = base_site["Abbr"].map(base_site["Abbr"].value_counts())

    # drop PLOT_ID and duplicate rows
    base_site = base_site.drop(columns=["PLOT_ID"]).drop_duplicates()

    # add number of LAI, FAPAR, FCOVER
    base_site["# LAI"] = base_site["Abbr"].map(lai_val["Abbr"].value_counts())
    base_site["# FAPAR"] = base_site["Abbr"].map(fapar_val["Abbr"].value_counts())
    base_site["# FCOVER"] = base_site["Abbr"].map(fcover_val["Abbr"].value_counts())

    # load data/misc/ecoregion_biome_table.csv
    ecoregion_biome = pd.read_csv("data/misc/ecoregion_biome_table.csv")[
        ["ECO_ID", "ECO_NAME", "BIOME_NAME", "BIOME_NUM"]
    ]
    # rename to Biome, Ecoregion
    ecoregion_biome = ecoregion_biome.rename(
        columns={
            "ECO_NAME": "Ecoregion",
            "BIOME_NAME": "Biome",
            "BIOME_NUM": "Biome_Num",
        }
    )

    # add Ecoregion, Biome:
    base_site = base_site.merge(ecoregion_biome, on="ECO_ID", how="left")

    # discard ECO_DI
    # base_site = base_site.drop(columns=["ECO_ID"])
    # rename ECO_ID to Eco_ID
    base_site = base_site.rename(columns={"ECO_ID": "Eco_ID"})

    # Rename and create new columns
    base_site.rename(columns={"Biome": "Biome_Long"}, inplace=True)
    base_site["Biome"] = base_site["Biome_Long"].map(BIOME_ABBREVIATIONS)

    print(base_site["Biome"].unique())

    # Biome_Num integer
    base_site["Biome_Num"] = base_site["Biome_Num"].astype(int)

    # save table: columns: Site, Abbr., Network, Eco_ID, Biome_Num, Latitude, Longitude, # Plots, # LAI, # FAPAR, # FCOVER
    base_site = base_site[
        [
            "Site",
            "Abbr",
            "Network",
            "Eco_ID",
            "Biome_Num",
            "# Plots",
            # "# LAI",
            # "# FAPAR",
            # "# FCOVER",
        ]
    ]

    # sort by Site
    base_site = base_site.sort_values(by="Site")

    # drop duplicate Site entries / (when have two rows / choose first)
    base_site = base_site.drop_duplicates(subset=["Site"], keep="first")

    # rename Biome_Num to Biome
    base_site = base_site.rename(columns={"Biome_Num": "Biome"})

    # in site columns: add spaces before capital letters (except first letter)
    base_site["Site"] = base_site["Site"].str.replace(
        r"(\w)([A-Z])", r"\1 \2", regex=True
    )

    base_site.to_csv("tables/gbov_sites.csv", index=False)


if __name__ == "__main__":
    main()
