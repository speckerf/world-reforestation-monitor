import numpy as np
import pandas as pd


def load_datasets():
    """
    Load the validation datasets
    """
    path_all = "data/validation_pipeline/output/"
    lai_path = (
        path_all
        + "lai/EXPORT_GBOV_RM6,7_20240620120826_all_reflectances_with_angles.csv"
    )
    lai = pd.read_csv(lai_path)

    fapar_path = (
        path_all
        + "fapar/EXPORT_GBOV_RM6,7_20240620120826_all_reflectances_with_angles.csv"
    )
    fapar = pd.read_csv(fapar_path)

    fcover_path = (
        path_all
        + "fcover/EXPORT_COPERNICUS_GBOV_RM4_20240816101306_all_reflectances_with_angles.csv"
    )
    fcover = pd.read_csv(fcover_path)

    return {"lai": lai, "fapar": fapar, "fcover": fcover}


def sample_from_distribution(row, value_col, error_col):
    return np.random.normal(row[value_col], row[error_col])


# Function to calculate metrics
def calculate_metrics(data, value_col, error_col, n_bootstrap=1000, frac=1.0):
    bootstrap_results = {"MAE": [], "RMSE": [], "R2": []}

    for _ in range(n_bootstrap):
        # Sample data with replacement
        bootstrap_sample = data.sample(frac=frac, replace=True)

        # Generate new samples for true and predicted values
        bootstrap_sample["true_sample"] = bootstrap_sample.apply(
            sample_from_distribution, axis=1, value_col=value_col, error_col=error_col
        )

        # Calculate metrics
        true = bootstrap_sample["true_sample"]
        pred = bootstrap_sample[value_col]

        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)

        bootstrap_results["MAE"].append(mae)
        bootstrap_results["RMSE"].append(rmse)
        bootstrap_results["R2"].append(r2)

    return {
        "MAE_mean": np.mean(bootstrap_results["MAE"]),
        "MAE_std": np.std(bootstrap_results["MAE"]),
        "MAE_ci": (
            np.percentile(bootstrap_results["MAE"], 2.5),
            np.percentile(bootstrap_results["MAE"], 97.5),
        ),
        "RMSE_mean": np.mean(bootstrap_results["RMSE"]),
        "RMSE_std": np.std(bootstrap_results["RMSE"]),
        "RMSE_ci": (
            np.percentile(bootstrap_results["RMSE"], 2.5),
            np.percentile(bootstrap_results["RMSE"], 97.5),
        ),
        "R2_mean": np.mean(bootstrap_results["R2"]),
        "R2_std": np.std(bootstrap_results["R2"]),
        "R2_ci": (
            np.percentile(bootstrap_results["R2"], 2.5),
            np.percentile(bootstrap_results["R2"], 97.5),
        ),
    }


def main():
    dfs = load_datasets()

    # bootstrap lai
    lai_metrics = calculate_metrics(dfs["lai"], "LAIe_Warren", "LAIe_Warren_err")

    # bootstrap fapar
    fapar_metrics = calculate_metrics(dfs["fapar"], "FIPAR_total", "FIPAR_total_err")

    # bootstrap fcover
    fcover_metrics = calculate_metrics(
        dfs["fcover"], "FCOVER_total", "FCOVER_total_err"
    )

    # write all results to a file in data/validation_pipeline/output/
    #
    results = pd.DataFrame(
        {
            "metric": ["MAE", "RMSE", "R2"],
            "lai_mean": [
                lai_metrics["MAE_mean"],
                lai_metrics["RMSE_mean"],
                lai_metrics["R2_mean"],
            ],
            "lai_ci": [
                lai_metrics["MAE_ci"],
                lai_metrics["RMSE_ci"],
                lai_metrics["R2_ci"],
            ],
            "fapar_mean": [
                fapar_metrics["MAE_mean"],
                fapar_metrics["RMSE_mean"],
                fapar_metrics["R2_mean"],
            ],
            "fapar_ci": [
                fapar_metrics["MAE_ci"],
                fapar_metrics["RMSE_ci"],
                fapar_metrics["R2_ci"],
            ],
            "fcover_mean": [
                fcover_metrics["MAE_mean"],
                fcover_metrics["RMSE_mean"],
                fcover_metrics["R2_mean"],
            ],
            "fcover_ci": [
                fcover_metrics["MAE_ci"],
                fcover_metrics["RMSE_ci"],
                fcover_metrics["R2_ci"],
            ],
        }
    )

    results.to_csv("data/validation_pipeline/output/bootstrap_metrics.csv")


if __name__ == "__main__":
    main()
