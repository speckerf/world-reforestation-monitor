import json

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from train_pipeline.utilsLoading import load_grounded_eo_validation_data

BACKGROUND = "land_ocean"  # options: "stock", "land_ocean", "plain"
POINT_SIZE = 26
POINT_EDGEWIDTH = 0.6
SAVE_PATH = "revision-figures/figure-s2/figure_s2.png"
FIG_DPI = 300


def make_background(ax, style: str = "land_ocean"):
    """
    Apply a background to the given Cartopy axes.
    """
    style = style.lower().strip()
    if style == "stock":
        # Low-res Blue Marble-like image
        ax.stock_img()
        # Add thin coastlines on top to keep borders crisp
        ax.coastlines(linewidth=0.6, color="black")
    elif style == "land_ocean":
        # Simple, clean land/ocean + coastlines
        ax.add_feature(cfeature.OCEAN, facecolor="#dde7f0")
        ax.add_feature(cfeature.LAND, facecolor="#efefe7")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#666666")
        ax.coastlines(linewidth=0.6, color="#333333")
    elif style == "plain":
        # Just coastlines; let the facecolor show through
        ax.set_facecolor("white")
        ax.coastlines(linewidth=0.6, color="black")
    else:
        raise ValueError(f"Unknown BACKGROUND style: {style!r}")


def get_ecoregion_model_ensemble_splits():
    """
    Load the ecoregion model ensemble splits.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the ecoregion model ensemble splits.
    """
    filename_regex = "data/train_pipeline/output/models/laie/model_optuna-v2-laie-mlp-split-*_split.json"
    # this is json looking like: {"val_ecos_train": [352, 338, 664, 331, 416, 205, 689, 390, 402, 396, 386, 654, 375, 717, 648, 799, 543, 206, 339, 405, 795, 428, 392, 429, 353, 389, 430, 636, 150, 679, 393, 423, 366, 435, 344, 388, 179, 367], "val_ecos_test": [407, 686, 399, 647, 623, 411]}
    splits = {}
    for split in range(5):
        filename = filename_regex.replace("*", str(split))
        json_data = json.load(open(filename, "r"))
        splits[split] = json_data["val_ecos_test"]

    return splits


def main():
    # --------------------------------
    # Load + prepare sites
    # --------------------------------
    df = load_grounded_eo_validation_data()

    # Select columns, deduplicate (if needed), then aggregate to 1 row per Site
    sites = (
        df[["Site", "Latitude", "Longitude", "ECO_ID"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .groupby("Site", as_index=False)
        .agg(
            {
                "Latitude": "mean",
                "Longitude": "mean",
                "ECO_ID": "first",
            }
        )
    )

    splits = get_ecoregion_model_ensemble_splits()

    def assign_split(eco_id):
        for split, eco_ids in splits.items():
            if eco_id in eco_ids:
                return split
        return -1  # not found

    sites["split"] = sites["ECO_ID"].apply(assign_split)

    # load biome ecoregion mapping
    path_biome_eco = "data/misc/ecoregion_biome_table.csv"
    df_biome_eco = pd.read_csv(path_biome_eco)
    eco_to_biome = dict(
        zip(
            df_biome_eco["ECO_ID"].astype("Int32"),
            df_biome_eco["BIOME_NAME"].astype("string"),
        )
    )
    sites["BIOME"] = sites["ECO_ID"].map(eco_to_biome)

    # --------------------------------
    biome_colors = {
        "Tropical & Subtropical Moist Broadleaf Forests": "#1b9e77",  # deep green – tropical forests
        "Tropical & Subtropical Dry Broadleaf Forests": "#d95f02",  # orange – dry forests
        "Temperate Broadleaf & Mixed Forests": "#66a61e",  # green – temperate forests
        "Temperate Conifer Forests": "#1f78b4",  # blue – conifers
        "Boreal Forests/Taiga": "#7570b3",  # violet – boreal
        "Mediterranean Forests, Woodlands & Scrub": "#e6ab02",  # gold – Mediterranean
        "Temperate Grasslands, Savannas & Shrublands": "#a6761d",  # brown – temperate grasslands
        "Deserts & Xeric Shrublands": "#e7298a",  # magenta – deserts
        "Tundra": "#66c2a5",  # light cyan – tundra
    }

    sites["BIOME_COLOR"] = sites["BIOME"].map(biome_colors)

    # --------------------------------
    # Build the plot (Robinson)
    # --------------------------------
    proj = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()  # the CRS of our lat/lon points

    fig = plt.figure(figsize=(12, 6.6))
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_global()

    # Background
    make_background(ax, BACKGROUND)

    # Optional: nice graticules
    gl = ax.gridlines(
        draw_labels=False,
        linewidth=0.3,
        color="#888888",
        alpha=0.4,
        linestyle="--",
    )
    # Plot sites, colored by ECO_ID (categorical)
    # Scatter plot of site locations
    n_splits = sites["split"].nunique()
    cmap = plt.get_cmap("tab10")

    color_map = {i: cmap(i) for i in range(n_splits)}

    point_colors = sites["split"].astype("category").cat.codes.map(color_map).tolist()

    sc = ax.scatter(
        sites["Longitude"],
        sites["Latitude"],
        c=sites["BIOME_COLOR"],
        s=60,
        edgecolor="black",
        linewidth=0.4,
        transform=data_crs,
        zorder=3,
    )

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=biome,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
        )
        for biome, color in biome_colors.items()
    ]

    ax.legend(
        handles=legend_elements,
        title="Terrestrial Biomes",
        loc="lower left",
        frameon=True,
        fontsize=10,
        title_fontsize=12,
        ncol=1,
    )

    # save figure
    plt.savefig(SAVE_PATH, dpi=FIG_DPI, bbox_inches="tight")


if __name__ == "__main__":
    ee.Initialize(project="ee-speckerfelix")
    main()
    # main()
