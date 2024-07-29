import os

import geopandas as gpd
import hvplot.pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from validation_pipeline.utils import load_ecoregion_shapefile


def load_data(data_dir):
    visnir = pd.read_parquet(os.path.join(data_dir, "ossl_visnir_L0_v1.2.parquet"))
    soilsite = pd.read_parquet(os.path.join(data_dir, "ossl_soilsite_L0_v1.2.parquet"))
    soillab = pd.read_parquet(os.path.join(data_dir, "ossl_soillab_L0_v1.2.parquet"))
    return visnir, soilsite, soillab


def filter_visnir_data(visnir):
    cols = [f"scan_visnir.{i}_ref" for i in range(400, 2301, 2)]
    filtered_data = visnir.dropna(subset=cols)
    return filtered_data


def join_data(filtered_data, soilsite):
    full_df = filtered_data.merge(soilsite, on="id.layer_uuid_txt", how="left")
    return full_df


def select_columns(full_df):
    df_locations = full_df[
        ["id.layer_uuid_txt", "longitude.point_wgs84_dd", "latitude.point_wgs84_dd"]
    ]
    df_reflectance = full_df.filter(regex="^id.layer_uuid_txt$|scan_visnir.*_ref")
    return df_locations, df_reflectance


def remove_missing_coordinates(df_locations):
    df_locations_nona = df_locations.dropna()
    df_locations_na = df_locations.loc[
        ~df_locations.index.isin(df_locations_nona.index)
    ]
    return df_locations_nona, df_locations_na


def convert_to_spatial(df_locations_nona):
    gdf = gpd.GeoDataFrame(
        df_locations_nona,
        geometry=gpd.points_from_xy(
            df_locations_nona["longitude.point_wgs84_dd"],
            df_locations_nona["latitude.point_wgs84_dd"],
        ),
        crs="EPSG:4326",
    ).drop(columns=["longitude.point_wgs84_dd", "latitude.point_wgs84_dd"])
    return gdf


def spatial_join(df_locations_fc, resolve_ecoregions):
    df_locations_fc_with_eco_id = gpd.sjoin(
        df_locations_fc, resolve_ecoregions, how="inner", op="within"
    )
    df_locations_fc_with_eco_id = df_locations_fc_with_eco_id.dropna().drop(
        columns="index_right"
    )
    return df_locations_fc_with_eco_id


def interpolate_spectra(spectral_data, current_wavelengths, new_wavelengths):
    interpolated_data = spectral_data.apply(
        lambda row: interp1d(current_wavelengths, row, kind="linear")(new_wavelengths),
        axis=1,
    )
    interpolated_df = pd.DataFrame(
        interpolated_data.tolist(),
        columns=[f"scan_visnir.{w}_ref" for w in new_wavelengths],
    )
    return interpolated_df


def convert_hyperspectral_to_sentinel2(df):
    # Extract column names corresponding to the wavelengths (scan_visnir.400_ref to scan_visnir.2500_ref)
    wavelength_columns = [f"scan_visnir.{i}_ref" for i in range(400, 2501)]

    # Assert that the required wavelength columns are present
    assert all([col in df.columns for col in wavelength_columns])

    # load Sentinel-2 SRF
    s2_srf = pd.read_csv(
        "data/rtm_pipeline/input/Sentinel_2_Spectral_Response.csv",
        sep="\t",
        index_col="Wavelength",
    )
    s2_srf_matrix = s2_srf.loc[400:2500].values

    # normalize the SRF matrix, such that each column sums to 1
    s2_srf_matrix = s2_srf_matrix / s2_srf_matrix.sum(axis=0)

    # check that columns sum to 1
    assert all(np.isclose(s2_srf_matrix.sum(axis=0), 1))

    # format is rows are wavelengths 300 to 2600, columns are senintle-2 bands B2, B3, ..., B12 (surface reflectance)
    # num_spectra: n, num_bands: m (10), num_wavelengths: p (2101)
    # matrix multiplication: (n x p) * (p x m) = (n x m)

    # Select all columns from scan_visnir.400_ref to scan_visnir.2500_ref
    df_metadata = df.loc[:, ["id.layer_uuid_txt", "geometry", "BIOME_NUM", "ECO_ID"]]
    df_matrix = df.loc[:, wavelength_columns].values

    # matrix multiplication
    df_matrix_s2 = df_matrix @ s2_srf_matrix

    # return a dataframe with all the wavelengths replaced by the sentinel-2 bands
    # add the metadata columns
    df_to_return = pd.DataFrame(df_matrix_s2, columns=s2_srf.columns).set_index(
        df_metadata.index
    )
    df_to_return = pd.concat([df_metadata, df_to_return], axis=1)

    return df_to_return


def plot_s2_reflectance(df, file_path):
    # Extract the columns for Sentinel-2 bands B2 through B12
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    df_bands = df[bands]

    # Plot the reflectance spectra
    plt.figure(figsize=(12, 8))
    for index, row in df_bands.iterrows():
        plt.plot(bands, row, label=f"Row {index}")

    plt.xlabel("Bands")
    plt.ylabel("Reflectance")
    plt.title("Reflectance Spectra for Sentinel-2 Bands B2 to B12")
    plt.legend(loc="best", bbox_to_anchor=(1.05, 1), fontsize="small", ncol=1)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to disk
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


def plot_hyperspectral_data(df):
    # Extract the columns for hyperspectral data
    wavelength_columns = [
        col
        for col in df.columns
        if col.startswith("scan_visnir.") and col.endswith("_ref")
    ]

    # Extract the wavelengths from column names
    wavelengths = [int(col.split(".")[1].split("_")[0]) for col in wavelength_columns]

    # Create a DataFrame with the wavelengths as columns
    df_wavelengths = df[wavelength_columns]
    df_wavelengths.columns = wavelengths

    # Melt the DataFrame for hvplot
    df_melted = df_wavelengths.reset_index().melt(
        id_vars=["index"], var_name="Wavelength", value_name="Reflectance"
    )

    # Plot using hvplot
    plot = df_melted.hvplot.line(
        x="Wavelength",
        y="Reflectance",
        by="index",
        legend=False,
        title="Insitu hyperspectral soil samples",
    )

    hvplot.save(
        plot,
        "rtm_pipeline_python/preprocessing/plots/insitu_samples_global_hyperspectral.html",
    )

    return None


def main():
    # Set data directory
    DATA_DIR = os.path.join("data", "rtm_pipeline", "input", "insitu_soil_database")

    # Load data files
    visnir, soilsite, soillab = load_data(DATA_DIR)

    # Filter VIS-NIR data
    filtered_data = filter_visnir_data(visnir)

    # Join data
    full_df = join_data(filtered_data, soilsite)

    # Select relevant columns
    df_locations, df_reflectance = select_columns(full_df)

    # Remove rows with missing geographic coordinates
    df_locations_nona, df_locations_na = remove_missing_coordinates(df_locations)

    # Convert to spatial feature collection
    df_locations_fc = convert_to_spatial(df_locations_nona)

    # Load and prepare ecoregions feature collection
    resolve_ecoregions = load_ecoregion_shapefile()[["ECO_ID", "geometry", "BIOME_NUM"]]

    # Spatial join to assign ecoregion IDs to locations
    df_locations_fc_with_eco_id = spatial_join(df_locations_fc, resolve_ecoregions)

    # Prepare reflectance spectra to be in the format needed by prosail
    current_wavelengths = list(range(350, 2501, 2))
    new_wavelengths = list(range(400, 2501, 1))

    # Interpolate spectral data
    interpolated_df = interpolate_spectra(
        df_reflectance.drop(columns="id.layer_uuid_txt"),
        current_wavelengths,
        new_wavelengths,
    )

    # Bind with the original id column
    interpolated_id_df = pd.concat(
        [df_reflectance[["id.layer_uuid_txt"]], interpolated_df], axis=1
    )

    # Filter spectra by df_locations_fc_with_eco_id
    interpolated_location_df = interpolated_id_df[
        interpolated_id_df["id.layer_uuid_txt"].isin(
            df_locations_fc_with_eco_id["id.layer_uuid_txt"]
        )
    ]

    soil_spectra_with_eco_id = df_locations_fc_with_eco_id.merge(
        interpolated_location_df, on="id.layer_uuid_txt", how="inner"
    )

    # Optionally, write the results to file, e.g., CSV, Shapefile, etc.
    soil_spectra_with_eco_id.to_csv(
        f"data/rtm_pipeline/output/insitu_soil_database/insitu_soil_spectra_hyperspectral.csv",
        index=False,
    )

    # write a single file per ecoregion
    for eco_id in soil_spectra_with_eco_id["ECO_ID"].unique():
        soil_spectra_with_eco_id[soil_spectra_with_eco_id["ECO_ID"] == eco_id].to_csv(
            f"data/rtm_pipeline/output/insitu_soil_database/insitu_soil_spectra_hyperspectral_eco_{int(eco_id)}.csv",
            index=False,
        )

    for biome in soil_spectra_with_eco_id["BIOME_NUM"].unique():
        soil_spectra_with_eco_id[soil_spectra_with_eco_id["BIOME_NUM"] == biome].to_csv(
            f"data/rtm_pipeline/output/insitu_soil_database/insitu_soil_spectra_hyperspectral_biome_{int(biome)}.csv",
            index=False,
        )

    soil_spectra_with_sentinel2bands = convert_hyperspectral_to_sentinel2(
        soil_spectra_with_eco_id
    )

    # Optionally, write the results to file, e.g., CSV, Shapefile, etc.
    soil_spectra_with_sentinel2bands.to_csv(
        f"data/rtm_pipeline/output/insitu_soil_database/insitu_soil_spectra_sentinel2bands.csv",
        index=False,
    )

    # write a single file per ecoregion
    for eco_id in soil_spectra_with_sentinel2bands["ECO_ID"].unique():
        soil_spectra_with_sentinel2bands[
            soil_spectra_with_sentinel2bands["ECO_ID"] == eco_id
        ].to_csv(
            f"data/rtm_pipeline/output/insitu_soil_database/insitu_soil_spectra_sentinel2bands_eco_{int(eco_id)}.csv",
            index=False,
        )

    for biome in soil_spectra_with_sentinel2bands["BIOME_NUM"].unique():
        soil_spectra_with_sentinel2bands[
            soil_spectra_with_sentinel2bands["BIOME_NUM"] == biome
        ].to_csv(
            f"data/rtm_pipeline/output/insitu_soil_database/insitu_soil_spectra_sentinel2bands_biome_{int(biome)}.csv",
            index=False,
        )

    # Plot the reflectance spectra for Sentinel-2 bands B2 to B12
    plot_s2_reflectance(
        soil_spectra_with_sentinel2bands.sample(n=20),
        file_path="rtm_pipeline_python/preprocessing/plots/insitu_samples_global_sentinel2",
    )

    plot_hyperspectral_data(soil_spectra_with_eco_id.sample(n=20)).opts(
        width=800, height=600
    ).save(
        "rtm_pipeline_python/preprocessing/plots/insitu_samples_global_hyperspectral.html"
    )


if __name__ == "__main__":
    main()
