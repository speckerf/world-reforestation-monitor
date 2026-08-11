"""Create reproducible cross-validation splits."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

# add base directory to sys.path to import train_pipeline.utilsLoading
BASE_DIR_ALL = Path(__file__).parent.parent.parent.parent
sys.path.append(str(BASE_DIR_ALL))

BASE_DIR_ANALYSIS = (
    Path(__file__).resolve().parents[1]
)  # analysis/05_biome_vegetation_specific_models


def main() -> None:
    # based on manual land cover / biome recoding: create cross-validation splits for each recoded group (ecoregion-based 3-fold crossvalidation)
    recoded_groups = pd.read_csv(BASE_DIR_ANALYSIS / "data/validation_groups.csv")

    # drop rows with RECODED_GROUP == -1 (not assigned to any group)
    print(
        f"Dropping {len(recoded_groups[recoded_groups['RECODED_GROUP'] == -1])} rows not assigned to any recoded group."
    )
    recoded_groups = recoded_groups[recoded_groups["RECODED_GROUP"] != -1]

    # create a table with uuid, eco_id, recoded_group, and fold (3-fold cross-validation)
    df = recoded_groups[["uuid", "ECO_ID", "RECODED_GROUP"]].copy()

    for group in df["RECODED_GROUP"].unique():
        group_df = df[df["RECODED_GROUP"] == group]

        # perform stratified 3-fold cross-validation: GroupKFold by ECO_ID
        skf = GroupKFold(n_splits=3)
        for fold_id, (_, test_index) in enumerate(
            skf.split(group_df, groups=group_df["ECO_ID"])
        ):
            df.loc[group_df.index[test_index], "test_fold"] = fold_id

    df["test_fold"] = df["test_fold"].fillna(-1).astype(int)

    # print number of observations per fold per recoded group / add n unique ECO_IDs per fold per recoded group
    fold_counts = (
        df.groupby(["RECODED_GROUP", "test_fold"])
        .agg(n_obs=("uuid", "size"), n_unique_eco_ids=("ECO_ID", "nunique"))
        .reset_index()
        .sort_values(["RECODED_GROUP", "test_fold"])
    )
    print(fold_counts)

    # save fold assignments

    df.to_csv(BASE_DIR_ANALYSIS / "data/cv_splits.csv", index=False)


if __name__ == "__main__":
    main()
