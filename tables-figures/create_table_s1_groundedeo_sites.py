import pandas as pd

from train_pipeline.utilsLoading import load_grounded_eo_validation_data

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
    "ABBY": {"Site": "Abbey Road", "Network": "NEON", "IGBP": "ENF"},
    "BARR": {"Site": "Utqiaġvik", "Network": "NEON", "IGBP": "Tundra"},
    "BART": {"Site": "Bartlett Experimental Forest", "Network": "NEON", "IGBP": "MF"},
    "BONA": {
        "Site": "Caribou-Poker Creeks Research Watershed",
        "Network": "NEON",
        "IGBP": "Taiga",
    },
    "Boyagin": {
        "Site": "Boyagin",
        "Network": "NEON",
    },
    "CH-Dav": {"Site": "Davos", "Network": "ICOS", "IGBP": "MF"},
    "CLBJ": {
        "Site": "Lyndon B. Johnson National Grassland",
        "Network": "NEON",
        "IGBP": "UNKNOWN",
    },
    "DCFS": {"Site": "Dakota Field Site", "Network": "NEON", "IGBP": "UNKNOWN"},
    "DE-Tha": {"Site": "Tharandt", "Network": "ICOS", "IGBP": "ENF"},
    "DEJU": {"Site": "Delta Junction", "Network": "NEON", "IGBP": "UNKNOWN"},
    "DK-Sor": {"Site": "Soroe", "Network": "ICOS", "IGBP": "UNKNOWN"},
    "FR-Bil": {"Site": "Bilos", "Network": "ICOS", "IGBP": "ENF"},
    "FI-Sod": {"Site": "Sodankyla", "Network": "ICOS", "IGBP": "UNKNOWN"},
    "FR-FBn": {"Site": "Font-Blanche", "Network": "ICOS", "IGBP": "UNKNOWN"},
    "FR-Hes": {"Site": "Hesse", "Network": "ICOS", "IGBP": "UNKNOWN"},
    "FR-Pue": {"Site": "Puechabon", "Network": "ICOS", "IGBP": "UNKNOWN"},
    "BLAN": {"Site": "Blandy Experimental Farm", "Network": "NEON", "IGBP": "DBF"},
    "BE-Bra": {"Site": "Brasschaat", "Network": "FluxNet", "IGBP": "MF"},
    "BE-Vie": {"Site": "Vielsalm", "Network": "ICOS", "IGBP": "MF"},
    "CPER": {
        "Site": "Central Plains Experimental Range",
        "Network": "NEON",
        "IGBP": "GRA",
    },
    "DE-HoH": {"Site": "Hohes Holz", "Network": "ICOS", "IGBP": "DBF"},
    "DELA": {"Site": "Dead Lake", "Network": "NEON", "IGBP": "DBF"},
    "DSNY": {"Site": "Disney Wilderness Preserve", "Network": "NEON", "IGBP": "OSH"},
    "FI-Hyy": {"Site": "Hyytiala", "Network": "SM", "IGBP": "ENF"},
    "FR-Fon": {"Site": "Fontainebleau-Barbeau", "Network": "ICOS", "IGBP": "DBF"},
    "GRSM": {
        "Site": "Great Smoky Mountains National Park",
        "Network": "NEON",
        "IGBP": "UNKNOWN",
    },
    "GUAN": {"Site": "Guanica Forest", "Network": "NEON", "IGBP": "EBF"},
    "Gingin": {
        "Site": "Gingin",
        "Network": "TERN",
        "IGBP": "UNKNOWN",
    },
    "HAIN": {"Site": "Hainich", "Network": "FluxNet", "IGBP": "MF"},
    "HARV": {"Site": "Harvard Forest", "Network": "NEON", "IGBP": "MF"},
    "HEAL": {"Site": "Healy", "Network": "NEON", "IGBP": "UNKNOWN"},
    "IT-SR2": {"Site": "San Rossore 2", "Network": "ICOS", "IGBP": "ENF"},
    "IT-Cp2": {"Site": "Castelporziano 2", "Network": "ICOS", "IGBP": "UNKNOWN"},
    "IT-Ren": {"Site": "Renon", "Network": "ICOS", "IGBP": "UNKNOWN"},
    "JERC": {
        "Site": "Jones Ecological Research Center",
        "Network": "NEON",
        "IGBP": "ENF",
    },
    "JORN": {"Site": "Jornada", "Network": "NEON", "IGBP": "OSH"},
    "KONZ": {
        "Site": "Konza Prairie Biological Station",
        "Network": "NEON",
        "IGBP": "CRO",
    },
    "KONA": {
        "Site": "Konza Prairie Agroexosystem",
        "Network": "NEON",
        "IGBP": "UNKNOWN",
    },
    "LAJA": {"Site": "Lajas Experimental Station", "Network": "NEON", "IGBP": "GRA"},
    "LENO": {"Site": "Lenoir Landing Site", "Network": "NEON", "IGBP": "UNKNOWN"},
    "LITC": {"Site": "Litchfield Savanna", "Network": "TERN", "IGBP": "WSA"},
    "TEAK": {"Site": "Lower Teakettle", "Network": "NEON", "IGBP": "ENF"},
    "MOAB": {"Site": "Moab", "Network": "NEON", "IGBP": "OSH"},
    "MLBS": {
        "Site": "Mountain Lake Biological Station",
        "Network": "NEON",
        "IGBP": "UNKNOWN",
    },
    "NIWO": {
        "Site": "Niwot Ridge Mountain Research Station",
        "Network": "NEON",
        "IGBP": "ENF",
    },
    "NO-Hur": {"Site": "Hurdal", "Network": "ICOS", "IGBP": "UNKNOWN"},
    "NOGP": {
        "Site": "Northern Great Plains Research Laboratory",
        "Network": "NEON",
        "IGBP": "UNKNOWN",
    },
    "ONAQ": {"Site": "Onaqui Ault", "Network": "NEON", "IGBP": "OSH"},
    "ORNL": {"Site": "Oak Ridge", "Network": "NEON", "IGBP": "MF"},
    "OSBS": {
        "Site": "Ordway Swisher Biological Station",
        "Network": "NEON",
        "IGBP": "ENF",
    },
    "OAES": {
        "Site": "Marvin Klemme Range Research Station",
        "Network": "NEON",
        "IGBP": "UNKNOWN",
    },
    "PUUM": {
        "Site": "Pu'u Maka'ala Natural Area Reserve",
        "Network": "NEON",
        "IGBP": "UNKNOWN",
    },
    "RMNP": {
        "Site": "Rocky Mountain National Park",
        "Network": "NEON",
        "IGBP": "UNKNOWN",
    },
    "Robson_Creek": {
        "Site": "Robson Creek Rainforest",
        "Network": "TERN",
        "IGBP": "UNKNOWN",
    },
    "SCBI": {
        "Site": "Smithsonian Conservation Biology Institute",
        "Network": "NEON",
        "IGBP": "MF",
    },
    "SERC": {
        "Site": "Smithsonian Environmental Research Center",
        "Network": "NEON",
        "IGBP": "CRO",
    },
    "SE-Htm": {"Site": "Hyltemossa", "Network": "ICOS", "IGBP": "ENF"},
    "SE-Nor": {"Site": "Norunda", "Network": "ICOS", "IGBP": "UNKNOWN"},
    "SE-Svb": {"Site": "Svartberget", "Network": "ICOS", "IGBP": "UNKNOWN"},
    "SJER": {
        "Site": "San Joaquin Experimental Range",
        "Network": "NEON",
        "IGBP": "UNKNOWN",
    },
    "SOAP": {"Site": "Soaproot Saddle", "Network": "NEON", "IGBP": "ENF"},
    "SRER": {"Site": "Santa Rita", "Network": "NEON", "IGBP": "CSH"},
    "STEI": {"Site": "Steigerwaldt Land Services", "Network": "NEON", "IGBP": "DBF"},
    "STER": {"Site": "North Sterling", "Network": "NEON", "IGBP": "GRA"},
    "TALL": {"Site": "Talladega National Forest", "Network": "NEON", "IGBP": "ENF"},
    "TOOL": {
        "Site": "Toolik Field Station",
        "Network": "NEON",
        "IGBP": "UNKNOWN",
    },
    "TREE": {
        "Site": "Treehaven",
        "Network": "NEON",
        "IGBP": "UNKNOWN",
    },
    "TUMB": {"Site": "Tumbarumba", "Network": "FluxNet", "IGBP": "EBF"},
    "UNDE": {"Site": "Underc", "Network": "NEON", "IGBP": "MF"},
    "UKFS": {
        "Site": "University of Kansas Field Station",
        "Network": "NEON",
        "IGBP": "UNKNOWN",
    },
    "VALE": {"Site": "Valencia Anchor Station", "Network": "SM", "IGBP": "CRO"},
    "WOMB": {"Site": "Wombat Stringbark Eucalypt", "Network": "TERN", "IGBP": "EBF"},
    "WREF": {
        "Site": "Wind River Experimental Forest",
        "Network": "NEON",
        "IGBP": "UNKNOWN",
    },
    "Warra": {
        "Site": "Warra Tall Eucalypt",
        "Network": "TERN",
        "IGBP": "UNKNOWN",
    },
    "WOOD": {"Site": "Woodworth", "Network": "NEON", "IGBP": "GRA"},
    "YELL": {"Site": "Yellowstone National Park", "Network": "NEON", "IGBP": "UNKNOWN"},
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


def create_site_table_supplement():
    validation_data = load_grounded_eo_validation_data()

    cols = ["Site", "ECO_ID", "NLCD", "Network", "Plot"]

    site_table = validation_data[cols].drop_duplicates().reset_index(drop=True)

    # count N per site
    site_counts = (
        validation_data.groupby("Site")["uuid"]
        .count()
        .reset_index()
        .rename(columns={"uuid": "N"})
    )

    # count plots per site
    plot_counts = (
        validation_data.groupby("Site")["Plot"]
        .nunique()
        .reset_index()
        .rename(columns={"Plot": "N_Plots"})
    )

    modal_nlcd = (
        validation_data.groupby(["Site", "NLCD"])
        .count()
        .reset_index()
        .sort_values(["Site", "uuid"], ascending=[True, False])
        .drop_duplicates(subset=["Site"], keep="first")
        .rename(columns={"NLCD": "Modal_NLCD"})
    )

    model_ecoid = (
        validation_data.groupby(["Site", "ECO_ID"])
        .count()
        .reset_index()
        .sort_values(["Site", "uuid"], ascending=[True, False])
        .drop_duplicates(subset=["Site"], keep="first")
        .rename(columns={"ECO_ID": "Modal_ECO_ID"})
    )

    site_table = site_table.merge(site_counts, on="Site", how="left")
    site_table = site_table.merge(plot_counts, on="Site", how="left")
    site_table = site_table.merge(
        modal_nlcd[["Site", "Modal_NLCD"]], on="Site", how="left"
    )
    site_table = site_table.merge(
        model_ecoid[["Site", "Modal_ECO_ID"]], on="Site", how="left"
    )

    # drop uuid, drop duplicates again
    site_table = (
        site_table.drop(columns=["Plot", "NLCD", "ECO_ID"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # rename Modal_ECO_ID to ECO_ID, Modal_NLCD to NLCD
    site_table = site_table.rename(
        columns={"Modal_ECO_ID": "ECO_ID", "Modal_NLCD": "NLCD"}
    )

    # load biome ecoregion lookup table and add Biome column
    ecoregion_lookup = pd.read_csv(
        "data/misc/ecoregion_biome_table.csv", usecols=["ECO_ID", "BIOME_NUM"]
    )

    site_table = site_table.merge(ecoregion_lookup, on="ECO_ID", how="left")

    # colname BIOME_NUM to Biome  and to int
    site_table = site_table.rename(columns={"BIOME_NUM": "Biome"})
    site_table["Biome"] = site_table["Biome"].astype(pd.Int64Dtype())

    print("Site Table Supplement:")

    # from SITE_DICT add station name
    site_table["Name"] = site_table["Site"].map(
        {abbr: SITE_DICT[abbr]["Site"] for abbr in SITE_DICT}
    )

    # relocate Name to first column
    site_table = site_table[
        ["Name"] + [col for col in site_table.columns if col != "Name"]
    ]

    # column order: Name, Site, Network, ECO_ID, Biome, NLCD, # plots, # measurements
    site_table = site_table[
        [
            "Name",
            "Site",
            "Network",
            "ECO_ID",
            "Biome",
            "NLCD",
            "N_Plots",
            "N",
        ]
    ]

    # rename N to # Measurements
    site_table = site_table.rename(columns={"N": "# Measurements"})
    site_table = site_table.rename(columns={"N_Plots": "# Plots"})

    # rename NLCD to land cover
    site_table = site_table.rename(columns={"NLCD": "Land Cover"})

    # drop Site column
    site_table = site_table.drop(columns=["Site"])

    print(site_table)

    # forst sort by network, then by Name
    site_table = site_table.sort_values(by=["Network", "Name"])

    print(f"All Biomes: {site_table['Biome'].unique()}")

    print(f"N Biomes: {site_table['Biome'].nunique()}")
    print(f"N Ecoregions: {site_table['ECO_ID'].nunique()}")
    print(f"N Sites: {site_table['Name'].nunique()}")
    print(f"N Networks: {site_table['Network'].nunique()}")
    print(f"Total Measurements: {site_table['# Measurements'].sum()}")
    print(f"Total Plots: {site_table['# Plots'].sum()}")

    # save as latex table
    with open("tables/groundedeo_sites_supplement.tex", "w") as f:
        f.write(site_table.to_latex(index=False))


def old_main():
    raise DeprecationWarning("Use other function instead.")
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
    create_site_table_supplement()
