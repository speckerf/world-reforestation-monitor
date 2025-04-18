import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error)


def plot_predicted_vs_true(
    y_true: np.array,
    y_pred: np.array,
    plot_type="scatter",
    save_plot_filename: str = None,
    x_label: str = "True Values",
    y_label: str = "Predicted Values",
):
    plt.close("all")
    # Convert inputs to numpy arrays
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    # Set plot size and style for publication quality
    plt.figure(figsize=(8, 6))
    sns.set_context("talk")  # Larger font sizes

    if plot_type == "scatter":
        # Create scatter plot
        data = pd.DataFrame({"True": y_true, "Predicted": y_pred})
        # limit the number of points to 3000
        if len(data) > 3000:
            data = data.sample(3000)
        ax = sns.scatterplot(
            data=data,
            x="True",
            y="Predicted",
            color="grey",
            s=8,  # Smaller size for points
            alpha=0.2,  # Increased transparency to handle overplotting
            thresh=0,
        )
        # add x, y labels

    elif plot_type == "hexbin":
        # Hexbin plot for large datasets
        plt.hexbin(
            y_true,
            y_pred,
            gridsize=20,
            cmap="Blues",
            mincnt=1,
            linewidths=0.5,
            norm=LogNorm(),
        )
        plt.colorbar(label="Count in Bin")
    elif plot_type == "density":
        # Density plot for large datasets
        ax = sns.kdeplot(
            data=pd.DataFrame({"True": y_true, "Predicted": y_pred}),
            x="True",
            y="Predicted",
            cmap="Blues",
            levels=100,
            fill=True,
        )
    elif plot_type == "density_scatter":
        # Density plot for large datasets
        data = pd.DataFrame({"True": y_true, "Predicted": y_pred})
        ax = sns.kdeplot(
            data=data,
            x="True",
            y="Predicted",
            cmap="Blues",
            levels=100,
            fill=True,
            thresh=0,
        )
        # limit the number of points to 3000
        if len(data) > 3000:
            data = data.sample(3000)
        sns.scatterplot(
            data=data,
            x="True",
            y="Predicted",
            color="grey",
            s=8,  # Smaller size for points
            alpha=0.2,  # Increased transparency to handle overplotting
        )
    else:
        raise ValueError(
            f"Invalid plot type: {plot_type}. Choose from 'scatter', 'hexbin', or 'density'."
        )

    # Plot a reference line for perfect prediction
    if "-lai-" in save_plot_filename:
        max_val = 6.0
    else:
        max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot(
        [0, max_val],
        [0, max_val],
        color="blue",
        linestyle="--",
        linewidth=1,
        label="1:1 line",
    )

    if save_plot_filename is not None:
        # add number of points n, rmse, mae, r2
        n = len(y_true)
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        me = np.mean(y_pred - y_true)
        plt.text(
            0.95,
            0.05,
            f"n={n}\nRMSE={rmse:.2f}\nMAE={mae:.2f}\nME={me:.2f}\nR2={r2:.2f}",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
            fontsize=14,
        )

        if "-lai-" in save_plot_filename:
            plt.xlim(-0.05, max_val + 0.05)
            plt.ylim(-0.05, max_val + 0.05)
        elif "-fcover-" in save_plot_filename:
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
        elif "-fapar-" in save_plot_filename:
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
        elif "-CHL-" in save_plot_filename:
            plt.xlim(0, max_val)
            plt.ylim(0, max_val)
        elif "-EWT-" in save_plot_filename:
            plt.xlim(0, max_val)
            plt.ylim(0, max_val)
        elif "-LMA-" in save_plot_filename:
            plt.xlim(0, max_val)
            plt.ylim(0, max_val)

    # Set plot labels and title
    plt.xlabel(f"{x_label}", fontsize=14, weight="bold")
    plt.ylabel(f"{y_label}", fontsize=14, weight="bold")

    # Improve plot aesthetics
    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if save_plot_filename:
        plt.savefig(save_plot_filename.replace("models", "plots"), dpi=800)
        plt.close()
    else:
        plt.show()
