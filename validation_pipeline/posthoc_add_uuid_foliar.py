import os
import uuid

import pandas as pd

"""
Add a uuid column to the foliar reflectance csv file.
"""


def main():
    path = os.path.join(
        "data",
        "validation_pipeline",
        "output",
        "foliar",
        "EXPORT_NEON_foliar_reflectances_with_angles.csv",
    )
    df = pd.read_csv(path)
    df["uuid"] = [uuid.uuid4() for _ in range(len(df))]
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
