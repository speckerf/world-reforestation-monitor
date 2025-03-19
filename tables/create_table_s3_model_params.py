import glob
import json
import os
import pickle

import pandas as pd

# Define the path to the JSON file: dynamically:
BASE_PATH = "data/train_pipeline/output/models/{TRAIT}/model_optuna-v11-{TRAIT}-mlp-split-{ENSEMBLE_MEMBER}_config.json"

ALL_TRAITS = [
    "lai",
    "fcover",
    "fapar",
]

ALL_ENSEMBLE_MEMBERS = [
    "0",
    "1",
    "2",
    "3",
    "4",
]


def get_file_paths():
    # Get all the JSON
    file_paths = []
    for trait in ALL_TRAITS:
        for ensemble_member in ALL_ENSEMBLE_MEMBERS:
            file_path = BASE_PATH.format(TRAIT=trait, ENSEMBLE_MEMBER=ensemble_member)
            file_paths.extend(glob.glob(file_path))
    return file_paths


# Function to extract information from the JSON file
def extract_trait_info(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, "r") as file:
        data = json.load(file)

    # Extract relevant information
    trait_info = {
        "num_spectra": data.get("num_spectra"),
        "parameter_setup": data.get("parameter_setup"),
        "lai_mean": data.get("lai_mean"),
        "lai_std": data.get("lai_std"),
        "rsoil_emit_insitu": data.get("rsoil_emit_insitu"),
        "rsoil_fraction": data.get("rsoil_fraction"),
        "additive_noise": data.get("additive_noise"),
        "multiplicative_noise": data.get("multiplicative_noise"),
        "p_baresoil_insitu": data.get("p_baresoil_insitu"),
        "p_baresoil_s2": data.get("p_baresoil_s2"),
        "p_baresoil_emit": data.get("p_baresoil_emit"),
        "p_urban_s2": data.get("p_urban_s2"),
        "p_snowice_s2": data.get("p_snowice_s2"),
        "use_observation_angles": data.get("use_angles_for_prediction"),
        "nirv_norm": data.get("nirv_norm"),
        "transform_target": data.get("transform_target"),
        # "hidden_layer_sizes": ", ".join(
        #     data.get("mlp_grid", {}).get("hidden_layer_sizes", [])
        # ),
        # "activation": ", ".join(data.get("mlp_grid", {}).get("activation", [])),
        # "alpha": ", ".join(map(str, data.get("mlp_grid", {}).get("alpha", []))),
        # "learning_rate": ", ".join(data.get("mlp_grid", {}).get("learning_rate", [])),
    }

    return trait_info


def main():

    # Get all the JSON files
    file_paths = get_file_paths()

    # Extract the data
    trait_data = {}
    for file_path in file_paths:
        data = None
        ensemble_number = file_path.split("-")[-1].split("_")[0]
        trait = file_path.split("-")[-4]
        data = extract_trait_info(file_path)
        if data:
            trait_data[f"{trait}_ensemble_{ensemble_number}"] = data

    # convert the dictionary to a dataframe
    df = pd.DataFrame(trait_data)
    # save to csv but with rounding
    output_path = "tables/ensemble_params.csv"
    df.to_csv(output_path, index=True)


if __name__ == "__main__":
    main()
