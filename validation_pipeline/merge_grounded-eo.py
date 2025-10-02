import os
import subprocess
import uuid

import ee
import pandas as pd

ee.Initialize(project="ee-speckerfelix")


def main():
    # list all files in google cloud storage matching the pattern: open-earth/validation/traits_GROUNDED-EO/*_s2_matchup.csv
    command = "gsutil ls gs://felixspecker/open-earth/validation/traits_GROUNDED-EO/*_s2_matchup.csv"
    files_names = (
        subprocess.check_output(command, shell=True).decode("utf-8").split("\n")
    )
    # remove empty strings
    files_names = [file_name for file_name in files_names if file_name]

    # empy matchups:
    empty_sites = ["Alice_Mulga", "Karawatha", "Whroo", "Wombat"]

    # check for substrings in filenames and remove them from the list
    files_names = [
        file_name
        for file_name in files_names
        if not any(empty_site in file_name for empty_site in empty_sites)
    ]

    for file in files_names:
        command = f"gsutil cp {file} data/validation_pipeline/output/grounded-eo/"
        subprocess.run(command, shell=True)

    # download all files to data/validation_pipeline/output/lai/EXPORT_GBOV_RM6,7_20240620120826_{site}_reflectances_with_angles.csv
    local_filenames = [
        os.path.join(
            "data",
            "validation_pipeline",
            "output",
            "grounded-eo",
            os.path.basename(file_name),
        )
        for file_name in files_names
    ]

    # merge all files to data/validation_pipeline/output/EXPORT_GBOV_RM6,7_20240620120826_all_reflectances_with_angles.csv
    df = pd.concat(
        [pd.read_csv(file_name) for file_name in local_filenames], ignore_index=True
    )

    # create a UUID column
    df["uuid"] = [uuid.uuid4() for _ in range(len(df))]

    df.to_csv(
        os.path.join(
            "data",
            "validation_pipeline",
            "output",
            "grounded-eo",
            "EXPORT_GROUNDED-EO_all_s2_matchup.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    main()
