"""Compare temporally matched S2BIOPHYS and CLMS Sentinel-3 at 300 m."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from helpers import Region, make_square_window, run_region_comparison
from helpers_sentinel3_clms import load_clms_s3_xarray
from matplotlib.colors import LinearSegmentedColormap
from rasterio.enums import Resampling
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "data" / "cache"
FIGURE_DIR = SCRIPT_DIR / "figures"
CLMS_REPROJECTION_BUFFER_M = 600


REGIONS_FIGURE = [
    Region(
        name="borealRussia",
        lon=125.36,
        lat=60.0,
        window_m=5_000,
        time_start="2022-07-16",
        time_end="2022-08-16",
        ecosystem="Boreal forest",
    ),
    Region(
        "hainichDeciduousForest",
        10.4521,
        51.0794,
        5_000,
        "2022-06-01",
        "2022-07-01",
        "Deciduous forest",
    ),
    Region(
        "beauceCropland",
        1.70,
        48.30,
        5_000,
        "2022-06-01",
        "2022-07-01",
        "Cultivated cropland",
    ),
    Region(
        "sierraMorenaShrubland",
        -3.80,
        38.30,
        5_000,
        "2022-04-01",
        "2022-05-01",
        "Mediterranean shrubland",
    ),
]

TRAITS = {
    "fapar": {
        "s2_variable": "fapar_s2biophys_mean",
        "label": "FAPAR",
        "limits": (0.0, 1.0),
        "cmap": LinearSegmentedColormap.from_list(
            "fapar", ["#ffffdd", "#e6ad12", "#c53859", "#3a26a1", "#000000"]
        ),
    },
    "lai": {
        "s2_variable": "laie_s2biophys_mean",
        "label": "LAIe / LAI",
        "limits": (0.0, 6.0),
        "cmap": LinearSegmentedColormap.from_list(
            "lai",
            [
                "#fffdcd",
                "#e1cd73",
                "#aaac20",
                "#5f920c",
                "#187328",
                "#144b2a",
                "#172313",
            ],
        ),
    },
}


def load_clms_temporal_mean(
    region: Region, product_name: Literal["fapar", "lai"], *, use_cache: bool = True
) -> xr.DataArray:
    """Load every CLMS dekad in the S2 interval and return their pixel mean."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / (
        f"clms_s3_{product_name}_{region.name.lower()}_"
        f"{region.time_start}_{region.time_end}_"
        f"buffer{CLMS_REPROJECTION_BUFFER_M}m.nc"
    )
    if use_cache and cache.exists():
        return xr.open_dataarray(cache).load()

    # Bilinear reprojection requires source pixels beyond the exact output
    # boundary. Load a two-pixel halo, which is especially important at high
    # latitudes where a geographic CLMS grid is strongly distorted in UTM.
    load_window_m = region.window_m + 2 * CLMS_REPROJECTION_BUFFER_M
    bbox = make_square_window(region.lon, region.lat, load_window_m)["bbox_wgs84"]
    da = load_clms_s3_xarray(
        bbox=bbox,
        time_from=f"{region.time_start}T00:00:00Z",
        time_to=f"{region.time_end}T00:00:00Z",
        product_name=product_name,
    )

    if da is None:
        raise ValueError(f"No CLMS {product_name} dekads found for {region.name}.")
    mean = da.mean("time", skipna=True, keep_attrs=True)
    mean.attrs["dekad_count"] = da.sizes["time"]
    mean.attrs["time_start"] = region.time_start
    mean.attrs["time_end_exclusive"] = region.time_end
    mean.attrs["reprojection_buffer_m"] = CLMS_REPROJECTION_BUFFER_M
    if use_cache:
        mean.to_netcdf(cache)
    return mean


def align_at_clms_resolution(
    s2_20m: xr.DataArray, clms: xr.DataArray, *, utm_epsg: int
) -> tuple[xr.DataArray, xr.DataArray]:
    """Aggregate S2 to 300 m and project CLMS onto that exact UTM grid."""
    source = s2_20m.rio.write_crs(f"EPSG:{utm_epsg}", inplace=False)

    # Define the comparison grid from the exact Sentinel-2 analysis window.
    # A grid derived from the enclosing lon/lat CLMS bbox can extend beyond
    # this window after reprojection, producing an empty edge row or column.
    aggregated = source.rio.reproject(
        f"EPSG:{utm_epsg}",
        resolution=300,
        resampling=Resampling.average,
    )
    aggregated.name = "s2biophys_300m"

    clms_geographic = clms.rename({"longitude": "x", "latitude": "y"})
    clms_geographic = clms_geographic.rio.set_spatial_dims(x_dim="x", y_dim="y")
    clms_geographic = clms_geographic.rio.write_crs("EPSG:4326", inplace=False)

    # CLMS is distributed on a geographic grid. Plotting that grid with an
    # equal x/y aspect stretches high-latitude regions because one longitude
    # degree is physically shorter than one latitude degree. Reproject first
    # so both map axes are expressed in metres.
    target = clms_geographic.rio.reproject_match(
        aggregated,
        resampling=Resampling.bilinear,
    )
    target.name = "sentinel3_clms_300m"

    # Restrict the result to the common raster footprints. The geographic
    # CLMS crop can fall just short of a projected S2 edge by one coarse cell.
    # Use a footprint mask rather than product validity, so genuine internal
    # CLMS nodata pixels do not change the spatial comparison window.
    footprint = xr.ones_like(clms_geographic, dtype=np.uint8).rio.reproject_match(
        aggregated,
        resampling=Resampling.nearest,
        nodata=0,
    )
    covered = footprint.values.astype(bool)
    covered_rows = np.flatnonzero(covered.any(axis=1))
    covered_columns = np.flatnonzero(covered.any(axis=0))
    if not covered_rows.size or not covered_columns.size:
        raise ValueError("Sentinel-2 and CLMS grids do not overlap.")

    selection = {
        "y": slice(covered_rows[0], covered_rows[-1] + 1),
        "x": slice(covered_columns[0], covered_columns[-1] + 1),
    }
    aggregated = aggregated.isel(selection)
    target = target.isel(selection)
    return aggregated, target


def plot_comparison(rows: list[dict], *, trait: Literal["fapar", "lai"]):
    """Plot RGB, S2BIOPHYS at 20/300 m, CLMS at 300 m, and pixel hexbin."""
    cfg = TRAITS[trait]
    vmin, vmax = cfg["limits"]
    fig, axes = plt.subplots(
        len(rows), 5, figsize=(13, 2.45 * len(rows) + 0.8), squeeze=False
    )
    fig.subplots_adjust(
        left=0.09, right=0.96, top=0.90, bottom=0.10, wspace=0.16, hspace=0.12
    )
    image = None

    for row_index, row in enumerate(rows):
        row_axes = axes[row_index]
        retrieval = row["retrieval"]
        rgb_ds = row["rgb"]
        s2_20m = row["s2_20m"]
        s2_300m = row["s2_300m"]
        clms = row["clms"]

        rgb = np.stack([rgb_ds["B4"], rgb_ds["B3"], rgb_ds["B2"]], axis=-1)
        rgb = np.clip(rgb.astype(float) / 0.3, 0, 1)
        extent_20m = [
            float(retrieval.x.min()),
            float(retrieval.x.max()),
            float(retrieval.y.min()),
            float(retrieval.y.max()),
        ]
        extent_300m = [
            float(clms.x.min()),
            float(clms.x.max()),
            float(clms.y.min()),
            float(clms.y.max()),
        ]
        row_axes[0].imshow(rgb, extent=extent_20m, origin="upper")
        image = row_axes[1].imshow(
            s2_20m,
            extent=extent_20m,
            origin="upper",
            cmap=cfg["cmap"],
            vmin=vmin,
            vmax=vmax,
        )
        row_axes[2].imshow(
            s2_300m,
            extent=extent_300m,
            origin="upper",
            cmap=cfg["cmap"],
            vmin=vmin,
            vmax=vmax,
        )
        row_axes[3].imshow(
            clms,
            extent=extent_300m,
            origin="upper",
            cmap=cfg["cmap"],
            vmin=vmin,
            vmax=vmax,
        )

        x, y = s2_300m.values.ravel(), clms.values.ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if x.size:
            row_axes[4].hexbin(x, y, gridsize=30, mincnt=1, bins="log", cmap="viridis")
            rho = spearmanr(x, y).statistic
            metrics = (
                rf"$r_s$ = {rho:.2f}"
                + "\n"
                + rf"median $\Delta$ = {np.median(x - y):.2f}"
                + "\n"
                + f"n = {x.size}"
            )
        else:
            metrics = "No overlapping valid pixels"
        row_axes[4].plot([vmin, vmax], [vmin, vmax], "--", color="black", linewidth=0.8)
        row_axes[4].set(xlim=(vmin, vmax), ylim=(vmin, vmax), aspect="equal")
        row_axes[4].text(
            0.05,
            0.95,
            metrics,
            transform=row_axes[4].transAxes,
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        row_axes[4].set_ylabel("Sentinel-3 CLMS", fontsize=7)
        row_axes[4].tick_params(labelsize=7, length=2)
        row_axes[0].set_ylabel(row["region"].ecosystem, fontsize=9, labelpad=10)
        for map_ax in row_axes[:4]:
            map_ax.set_xticks([])
            map_ax.set_yticks([])
            map_ax.set_aspect("equal")
        if row_index == len(rows) - 1:
            row_axes[4].set_xlabel("S2BIOPHYS aggregated to 300 m", fontsize=7)
        else:
            row_axes[4].tick_params(axis="x", labelbottom=False)

    titles = [
        "Sentinel-2 RGB",
        "S2BIOPHYS 20 m",
        "S2BIOPHYS 300 m\n(mean)",
        "Sentinel-3 CLMS 300 m\n(dekadal mean)",
        "Pixel relationship",
    ]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=9)
    if trait == "lai":
        fig.suptitle("S2BIOPHYS effective LAI (LAIe) vs CLMS LAI", fontsize=11)
    cbar = fig.colorbar(
        image,
        ax=axes[:, :4],
        orientation="horizontal",
        fraction=0.025,
        pad=0.04,
        aspect=50,
    )
    cbar.set_label(cfg["label"], fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return fig, axes


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = {}
        for region in REGIONS_FIGURE:
            print(f"Loading Sentinel-2 and S2BIOPHYS: {region.name}")
            base[region.name] = run_region_comparison(
                region,
                gee_project="ee-speckerfelix",
                use_cache=True,
                cache_dir=CACHE_DIR,
            )

        for trait in ("fapar", "lai"):
            rows = []
            for region in REGIONS_FIGURE:
                retrieval, rgb = base[region.name]
                s2_20m = retrieval[TRAITS[trait]["s2_variable"]]
                clms_mean = load_clms_temporal_mean(region, trait)
                s2_300m, clms_aligned = align_at_clms_resolution(
                    s2_20m, clms_mean, utm_epsg=int(retrieval.attrs["utm_epsg"])
                )
                rows.append(
                    {
                        "region": region,
                        "retrieval": retrieval,
                        "rgb": rgb,
                        "s2_20m": s2_20m,
                        "s2_300m": s2_300m,
                        "clms": clms_aligned,
                    }
                )

            fig, _ = plot_comparison(rows, trait=trait)
            output = FIGURE_DIR / f"compare_s2biophys_and_sentinel3-clms_{trait}.png"
            fig.savefig(output, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
