import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv(
        "data/phenology_pipeline/outputs/artificial_masked_w_amplitude_singleeco.csv"
    )

    # plot minimum EVI versus evi amplitude
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.scatter(df["EVI_Minimum_1"], df["EVI_Amplitude_1"])
    ax.set_xlabel("EVI Minimum")
    ax.set_ylabel("EVI Amplitude")
    ax.set_title("EVI Minimum vs Amplitude")

    plt.show()


if __name__ == "__main__":
    main()
