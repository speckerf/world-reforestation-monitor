import os

import geopandas as gpd
import hvplot.pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

from validation_pipeline.utils import load_ecoregion_shapefile


def load_data(data_folder, emit_folder_name, base_filename):
    reflectance_filename_ending = "-EMIT-L2A-RFL-001-results.csv"
    uncertainty_filename_ending = "-EMIT-L2A-RFLUNCERT-001-results.csv"
    mask_filename_ending = "-EMIT-L2A-MASK-001-results.csv"

    df_refl = pd.read_csv(
        os.path.join(
            data_folder, emit_folder_name, base_filename + reflectance_filename_ending
        )
    )
    df_uncert = pd.read_csv(
        os.path.join(
            data_folder, emit_folder_name, base_filename + uncertainty_filename_ending
        )
    )
    df_mask = pd.read_csv(
        os.path.join(
            data_folder, emit_folder_name, base_filename + mask_filename_ending
        )
    )

    return df_refl, df_uncert, df_mask


# Define the interpolation function to be applied row-wise
def interpolate_row(row, new_wavelengths):
    # Extract current wavelengths and values
    original_wavelengths = np.array([float(w) for w in row.index])
    values = row.values

    # Create interpolation function
    interp_func = interp1d(
        original_wavelengths,
        values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    # Interpolate to new wavelength values
    new_values = interp_func(new_wavelengths)

    return pd.Series(new_values, index=new_wavelengths)


def process_and_interpolate_data(df_refl, df_uncert, df_mask):
    # join all three dataframes by Latitude, Longitude, and Date
    assert (
        df_refl[["Latitude", "Longitude", "Date"]].values
        == df_uncert[["Latitude", "Longitude", "Date"]].values
    ).all()
    assert len(df_refl) == len(df_uncert) == len(df_mask)

    df = df_refl.merge(
        df_uncert,
        on=["Latitude", "Longitude", "Date", "Band", "elev", "fwhm", "wavelength"],
    ).merge(
        df_mask,
        on=["Latitude", "Longitude", "Date", "Band", "elev", "fwhm", "wavelength"],
    )

    # create unique observation_id column (index) by unique combination of ID and Date
    df["id_observation_temp"] = (
        df["Latitude"].astype(str) + df["Longitude"].astype(str) + df["Date"]
    )
    df["observation_id"] = df["id_observation_temp"].astype("category").cat.codes
    df = df.drop(columns=["id_observation_temp"])
    df = df.set_index(["observation_id"])

    ## set bad values to NaN
    df["reflectance"] = df["reflectance"].apply(lambda x: np.nan if x < 0 else x)
    df_filtered = df.copy()
    df_filtered["reflectance"] = df_filtered["reflectance"].where(
        df_filtered["cirrus_flag"] == 0.0, np.nan
    )
    df_filtered["reflectance"] = df_filtered["reflectance"].where(
        df_filtered["cloud_flag"] == 0.0, np.nan
    )
    df_filtered["reflectance"] = df_filtered["reflectance"].where(
        df_filtered["dilated_cloud_flag"] == 0.0, np.nan
    )
    df_filtered["reflectance"] = df_filtered["reflectance"].where(
        df_filtered["spacecraft_flag"] == 0.0, np.nan
    )
    df_filtered["reflectance"] = df_filtered["reflectance"].where(
        df_filtered["water_flag"] == 0.0, np.nan
    )
    df_filtered["reflectance"] = df_filtered["reflectance"].where(
        df_filtered["good_wavelengths"] == 1.0, np.nan
    )

    # replace wavelengths between 1430 and 1460 with nan
    df_filtered["reflectance"] = df_filtered["reflectance"].where(
        (df_filtered["wavelength"] < 1430) | (df_filtered["wavelength"] > 1470), np.nan
    )

    # Step 2: Interpolate missing values in 'wavelength' for each 'observation_id'
    # The loop goes through each group, interpolates the missing values, and then updates the DataFrame.
    for observation_id, group in df_filtered.groupby(level="observation_id"):
        # print(group)
        interpolated_values = (
            group["reflectance"].infer_objects(copy=False).interpolate(method="linear")
        )  # Default is linear interpolation / cubicspline for some reason does not work????
        df_filtered.loc[(observation_id,), "reflectance"] = interpolated_values

    df_interpolated = df_filtered.copy()

    # pivot table to have rows as spectra and columns as wavelengths /
    df_pivot = df_interpolated.pivot_table(
        index=["observation_id", "Latitude", "Longitude", "Date"],
        columns="wavelength",
        values="reflectance",
    )
    # somehow a single missing value is remaining: fill na with nearest value in row: bbfill
    df_pivot = df_pivot.apply(lambda row: row.bfill().ffill(), axis=1)

    # save as 2 csv]s" one with observation_id, Latitude, Longitude, Date, and the other with the reflectance values
    df_id = df_pivot.reset_index()[
        ["observation_id", "Latitude", "Longitude", "Date"]
    ].set_index("observation_id")
    df_reflectance = (
        df_pivot.reset_index()
        .drop(columns=["Latitude", "Longitude", "Date"])
        .set_index("observation_id")
    )
    df_id.columns.name = None

    # Step 1: Define the new wavelength range
    new_wavelengths = np.arange(400, 2501, 1)

    # step 3: Apply the function to each row of the DataFrame
    df_resampled = df_reflectance.apply(
        interpolate_row, axis=1, args=(new_wavelengths,)
    )

    df_id_gpd = gpd.GeoDataFrame(
        df_id, geometry=gpd.points_from_xy(df_id.Longitude, df_id.Latitude)
    )
    # join ECO_ID from ecoregions gdf
    ecoregions = load_ecoregion_shapefile()
    df_with_eco = gpd.sjoin(df_id_gpd, ecoregions, how="left", op="within").drop(
        columns="index_right"
    )

    df_with_eco_to_save = df_with_eco[["geometry", "BIOME_NUM", "ECO_ID"]]
    df_to_save = df_with_eco_to_save.join(df_resampled)
    return df_to_save


def convert_hyperspectral_to_sentinel2(df):
    # assert that colmumns 400 to 2500 are present: these are wavelengths
    assert all([i in df.columns for i in range(400, 2501)])

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

    # select all columns from 400 to 2500
    df_metadata = df.loc[:, ["geometry", "BIOME_NUM", "ECO_ID"]]
    df_matrix = df.loc[:, 400:2500].values

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

    return None


def plot_hyperspectral_reflectance(df, file_path):
    # Extract the columns for hyperspectral bands
    wavelength_columns = list(range(400, 2501))

    # Create a DataFrame with the wavelengths as columns
    df_wavelengths = df[wavelength_columns]

    # Melt the DataFrame for hvplot
    df_melted = df_wavelengths.reset_index().melt(
        id_vars=["observation_id"], var_name="Wavelength", value_name="Reflectance"
    )

    # Plot using hvplot
    plot = df_melted.hvplot.line(
        x="Wavelength",
        y="Reflectance",
        by="observation_id",
        legend=False,
        title="Emit hyperspectral soil samples",
    )

    hvplot.save(
        plot,
        filename=file_path,
    )
    return None


def main():
    data = {
        "global-baresoil-radom-points-1": "1d55e9e3-8e05-4952-bffc-abf492647c40",
        "global-baresoil-radom-points-2": "76a76fbe-21be-4b6d-9c3f-8b164885f09a",
        "global-baresoil-radom-points-3": "4098a34a-6855-4082-8e70-1f04cd3a2f18",
        "global-baresoil-radom-points-4": "aa3c0357-b8cb-4c6c-889b-1983854709a7",
        "global-baresoil-radom-points-5": "545eb0fa-fcef-46de-8181-eaaef218fada",
    }

    # load all data and store in a dictionary
    data_folder = "data/rtm_pipeline/input/emit_hyperspectral/point_data"
    data_loaded = {}
    for base_filename, emit_folder_name in data.items():
        data_loaded[base_filename] = load_data(
            data_folder, emit_folder_name, base_filename
        )

    # process and interpolate data
    for base_filename, (df_refl, df_uncert, df_mask) in tqdm(data_loaded.items()):
        df_hyper = process_and_interpolate_data(df_refl, df_uncert, df_mask)

        df_hyper.to_csv(
            f"data/rtm_pipeline/output/emit_hyperspectral/point_data/{base_filename}_hyperspectral.csv"
        )

        df_s2 = convert_hyperspectral_to_sentinel2(df_hyper)
        df_s2.to_csv(
            f"data/rtm_pipeline/output/emit_hyperspectral/point_data/{base_filename}_sentinel2bands.csv"
        )

    # plot hyperspectral and sentinel-2 reflectances
    plot_s2_reflectance(
        df_s2.sample(n=20),
        "rtm_pipeline_python/preprocessing/plots/emit_reflectance_Sentinel2bands.png",
    )

    plot_hyperspectral_reflectance(
        df_hyper.sample(n=20),
        "rtm_pipeline_python/preprocessing/plots/emit_reflectance_hyperspectral.html",
    )


def merge_runs():
    data_folder = "data/rtm_pipeline/output/emit_hyperspectral/point_data"
    dfs = []
    for filename in os.listdir(data_folder):
        if filename.endswith("_hyperspectral.csv"):
            df = pd.read_csv(os.path.join(data_folder, filename), index_col=0)
            dfs.append(df)
    df_merged = pd.concat(dfs).reset_index(drop=True)
    df_merged.to_csv(
        f"data/rtm_pipeline/output/emit_hyperspectral/point_data/global-baresoil-random-points-all_hyperspectral.csv",
        index=False,
    )

    # save per ecoregion and per biome
    for eco_id, df_eco in df_merged.groupby("ECO_ID"):
        df_eco.to_csv(
            f"data/rtm_pipeline/output/emit_hyperspectral/point_data/global-baresoil-random-points-eco_{int(eco_id)}_hyperspectral.csv",
            index=False,
        )

    for biome_num, df_biome in df_merged.groupby("BIOME_NUM"):
        df_biome.to_csv(
            f"data/rtm_pipeline/output/emit_hyperspectral/point_data/global-baresoil-random-points-biome_{int(biome_num)}_hyperspectral.csv",
            index=False,
        )

    dfs2 = []
    for filename in os.listdir(data_folder):
        if filename.endswith("_sentinel2bands.csv"):
            df = pd.read_csv(os.path.join(data_folder, filename), index_col=0)
            dfs2.append(df)
    df_merged_s2 = pd.concat(dfs2).reset_index(drop=True)
    df_merged_s2.to_csv(
        f"data/rtm_pipeline/output/emit_hyperspectral/point_data/global-baresoil-random-points-all_sentinel2bands.csv",
        index=False,
    )

    for eco_id, df_eco in df_merged_s2.groupby("ECO_ID"):
        df_eco.to_csv(
            f"data/rtm_pipeline/output/emit_hyperspectral/point_data/global-baresoil-random-points-eco_{int(eco_id)}_sentinel2bands.csv",
            index=False,
        )

    for biome_num, df_biome in df_merged_s2.groupby("BIOME_NUM"):
        df_biome.to_csv(
            f"data/rtm_pipeline/output/emit_hyperspectral/point_data/global-baresoil-random-points-biome_{int(biome_num)}_sentinel2bands.csv",
            index=False,
        )

    return None


if __name__ == "__main__":
    # main()
    merge_runs()
