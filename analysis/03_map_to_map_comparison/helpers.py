from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401
import shapely
import xarray as xr
from gee_biophys.config import ConfigParams
from gee_biophys.s2_input import convert_s2_input_to_xarray, load_s2_input
from gee_biophys.s2_predict import biophys_predict_local
from loguru import logger
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from xee.helpers import fit_geometry

# ---------------------------------------------------------------------
# Region specification
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    name: str
    lon: float
    lat: float
    window_m: float
    time_start: str
    time_end: str
    ecosystem: str | None = None


# ---------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------


def get_utm_crs(lon: float, lat: float) -> CRS:
    """Return the most appropriate UTM CRS for a lon/lat point."""
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )

    if not utm_crs_list:
        raise ValueError(f"No UTM CRS found for lon={lon}, lat={lat}.")

    return CRS.from_epsg(utm_crs_list[0].code)


def make_square_window(
    lon: float,
    lat: float,
    window_m: float,
) -> dict:
    """Create an exact square analysis window in the local UTM CRS.

    Parameters
    ----------
    lon, lat
        Center coordinates in WGS84.
    window_m
        Side length of the square in metres.

    Returns
    -------
    dict
        UTM CRS, projected square bounds, and an enclosing WGS84 bbox
        suitable for loading the data from GEE.
    """
    utm_crs = get_utm_crs(lon, lat)

    to_utm = Transformer.from_crs(
        "EPSG:4326",
        utm_crs,
        always_xy=True,
    )
    to_wgs84 = Transformer.from_crs(
        utm_crs,
        "EPSG:4326",
        always_xy=True,
    )

    center_x, center_y = to_utm.transform(lon, lat)

    half = window_m / 2

    xmin = center_x - half
    xmax = center_x + half
    ymin = center_y - half
    ymax = center_y + half

    # Transform all four UTM-square corners back to WGS84.
    corners_utm = [
        (xmin, ymin),
        (xmin, ymax),
        (xmax, ymin),
        (xmax, ymax),
    ]

    corners_wgs84 = [to_wgs84.transform(x, y) for x, y in corners_utm]

    lons = [p[0] for p in corners_wgs84]
    lats = [p[1] for p in corners_wgs84]

    bbox_wgs84 = [
        min(lons),
        min(lats),
        max(lons),
        max(lats),
    ]

    return {
        "crs": utm_crs,
        "epsg": utm_crs.to_epsg(),
        "bounds_utm": (xmin, ymin, xmax, ymax),
        "bbox_wgs84": bbox_wgs84,
    }


def crop_to_utm_window(
    ds: xr.Dataset,
    bounds_utm: tuple[float, float, float, float],
) -> xr.Dataset:
    """Crop an xarray Dataset to the exact requested UTM square."""
    xmin, ymin, xmax, ymax = bounds_utm

    # Most raster datasets have descending y coordinates.
    if ds.y[0] > ds.y[-1]:
        y_slice = slice(ymax, ymin)
    else:
        y_slice = slice(ymin, ymax)

    return ds.sel(
        x=slice(xmin, xmax),
        y=y_slice,
    )


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


def make_region_cfg(
    region: Region,
    *,
    gee_project: str,
    scale_m: int = 20,
    max_cloud_cover: int = 40,
) -> tuple[ConfigParams, dict]:
    """Create gee-biophys configuration for one analysis region."""
    spatial = make_square_window(
        lon=region.lon,
        lat=region.lat,
        window_m=region.window_m,
    )

    cfg = ConfigParams(
        **{
            "spatial": {
                "type": "bbox",
                "bbox": spatial["bbox_wgs84"],
                "region_name": region.name,
            },
            "temporal": {
                "start": region.time_start,
                "end": region.time_end,
                "cadence": {
                    "type": "fixed",
                    "interval": "monthly",
                },
            },
            "variables": {
                # "model": "s2biophys",
                "variable": "fapar",
            },
            "export": {
                "destination": "asset",
                "collection_path": (
                    f"projects/{gee_project}/assets/map-comparison/{region.name}"
                ),
                "project_id": gee_project,
                "crs": f"EPSG:{spatial['epsg']}",
                "scale": scale_m,
                "max_pixels": 100_000_000,
            },
            "options": {
                "max_cloud_cover": max_cloud_cover,
                "csplus_band": "cs",
                "cs_plus_threshold": 0.65,
                "clip_min_max": True,
                "prediction_mode": "predict_then_aggregate",
            },
            "version": "v02",
        }
    )

    return cfg, spatial


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

MODEL_RUNS = [
    ("s2biophys", "fapar"),
    ("sl2p", "fapar"),
    ("groundedeo", "fapar"),
    ("s2biophys", "laie"),
    ("sl2p", "laie"),
    ("groundedeo", "lai"),
    ("s2biophys", "fcover"),
    ("sl2p", "fcover"),
]


def load_and_save_esa_worldcover(
    s2_ds: xr.Dataset,
    *,
    cache_dir: Path,
    use_cache: bool = True,
) -> xr.DataArray:
    """
    Load ESA WorldCover 2021 with xee and align exactly to the S2 grid.

    Returns
    -------
    xr.DataArray
        ESA WorldCover classes with dimensions (y, x), exactly matching s2_ds.
    """

    region = s2_ds.attrs["region"]

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = cache_dir / f"worldcover_{region}.zarr"

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    if use_cache and cache_path.exists():
        print(f"  Loading cached WorldCover: {cache_path}")

        return xr.open_zarr(cache_path)["landcover"]

    # ------------------------------------------------------------------
    # Target S2 grid
    # ------------------------------------------------------------------

    if s2_ds.rio.crs is None:
        # Fall back to stored metadata
        epsg = int(s2_ds.attrs["utm_epsg"])
        s2_ds = s2_ds.rio.write_crs(f"EPSG:{epsg}")

    target_crs = s2_ds.rio.crs

    xmin, ymin, xmax, ymax = s2_ds.rio.bounds()

    aoi = shapely.geometry.box(xmin, ymin, xmax, ymax)

    grid = fit_geometry(
        geometry=aoi,
        geometry_crs=target_crs.to_string(),
        grid_crs=target_crs.to_string(),
        grid_shape=(
            s2_ds.sizes["x"],
            s2_ds.sizes["y"],
        ),
    )

    # ------------------------------------------------------------------
    # ESA WorldCover
    # ------------------------------------------------------------------

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

    ds = xr.open_dataset(
        worldcover,
        engine="ee",
        **grid,
    ).load()

    # squeeze time dimension
    if "time" in ds.dims:
        assert ds.sizes["time"] == 1, "Expected single time dimension for WorldCover"
        ds = ds.isel(time=0, drop=True)

    ds = ds.rename({"Map": "landcover"})

    ds.attrs.update(
        {
            "source": "ESA WorldCover 2021 v200",
            "region": region,
        }
    )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    if use_cache:
        ds.to_zarr(
            cache_path,
            mode="w",
        )

        print(f"  Cached WorldCover: {cache_path}")

    return ds


def _load_s2_stack(
    region: Region,
    cfg,
    interval_start: datetime,
    interval_end: datetime,
    *,
    cache_path: Path,
    use_cache: bool,
) -> xr.Dataset:

    if use_cache and cache_path.exists():
        print(f"  Loading cached S2 stack: {cache_path}")
        return xr.open_zarr(cache_path).load()

    print("  Fetching S2 stack")

    s2_imgc = load_s2_input(
        cfg,
        interval_start,
        interval_end,
    )

    if s2_imgc.size().getInfo() == 0:
        raise ValueError(
            f"No Sentinel-2 images found for {region.name} "
            f"({region.time_start} to {region.time_end})"
        )

    s2_ds = convert_s2_input_to_xarray(
        cfg,
        s2_imgc,
    ).load()

    if s2_ds.sizes["x"] == 0 or s2_ds.sizes["y"] == 0:
        raise ValueError(
            f"Empty S2 grid for {region.name}: "
            f"x={s2_ds.sizes['x']}, y={s2_ds.sizes['y']}"
        )

    if s2_ds.sizes["x"] != s2_ds.sizes["y"]:
        logger.warning(
            f"S2 grid for {region.name} is not square: "
            f"x={s2_ds.sizes['x']}, y={s2_ds.sizes['y']}"
        )

    if use_cache:
        s2_ds.to_zarr(cache_path, mode="w")
        print(f"  Cached S2 stack: {cache_path}")

    return s2_ds


def _get_region_attrs(
    region: Region,
    spatial: dict,
) -> dict:
    return {
        "region": region.name,
        "ecosystem": region.ecosystem or "",
        "center_lon": region.lon,
        "center_lat": region.lat,
        "window_m": region.window_m,
        "utm_epsg": spatial["epsg"],
        "time_start": region.time_start,
        "time_end": region.time_end,
    }


def _run_retrievals(
    s2_ds: xr.Dataset,
    *,
    cache_path: Path,
    use_cache: bool,
) -> xr.Dataset:

    if use_cache and cache_path.exists():
        print(f"  Loading cached retrievals: {cache_path}")
        return xr.open_zarr(cache_path).load()

    retrieval_datasets = []

    for model, variable in MODEL_RUNS:
        print(f"  Predicting {model} / {variable}")

        pred = biophys_predict_local(
            input_ds=s2_ds,
            variable=variable,
            model=model,
        )

        pred = pred.rename(
            {
                name: (f"{variable}_{model}_{name.removeprefix(f'{variable}_')}")
                for name in pred.data_vars
            }
        )

        retrieval_datasets.append(pred)

    retrievals = xr.merge(
        retrieval_datasets,
        join="exact",
        compat="no_conflicts",
    )

    if use_cache:
        retrievals.to_zarr(cache_path, mode="w")
        print(f"  Cached retrievals: {cache_path}")

    return retrievals


def run_region_comparison(
    region: Region,
    *,
    gee_project: str,
    scale_m: int = 20,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Return retrievals and median Sentinel-2 composite."""

    cache_dir = cache_dir or Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_name = (
        f"{region.name.lower().replace(' ', '_')}_"
        f"{region.time_start}_{region.time_end}_"
        f"{scale_m}m.zarr"
    )

    s2_cache_path = cache_dir / f"s2_{cache_name}"
    retrievals_cache_path = cache_dir / f"retrievals_{cache_name}"

    cfg, spatial = make_region_cfg(
        region,
        gee_project=gee_project,
        scale_m=scale_m,
    )

    interval_start = datetime.fromisoformat(region.time_start).replace(tzinfo=UTC)

    interval_end = datetime.fromisoformat(region.time_end).replace(tzinfo=UTC)

    print(
        f"{region.name}: {region.window_m / 1000:g} km square, EPSG:{spatial['epsg']}"
    )

    # Sentinel-2
    s2_ds = _load_s2_stack(
        region,
        cfg,
        interval_start,
        interval_end,
        cache_path=s2_cache_path,
        use_cache=use_cache,
    )

    # Retrievals
    retrievals = _run_retrievals(
        s2_ds,
        cache_path=retrievals_cache_path,
        use_cache=use_cache,
    )

    # Common metadata
    attrs = _get_region_attrs(
        region,
        spatial,
    )

    retrievals.attrs.update(attrs)

    # Median RGB/reference composite
    s2_median = s2_ds.median(
        dim="time",
        skipna=True,
    )
    s2_median.attrs.update(attrs)

    return retrievals, s2_median


# ---------------------------------------------------------------------
# Plot maps
# ---------------------------------------------------------------------

ALGORITHMS = [
    "s2biophys",
    "sl2p",
    "groundedeo",
]

LABELS = {
    "s2biophys": "S2BIOPHYS",
    "sl2p": "SL2P",
    "groundedeo": "GROUNDED-EO GPR",
}


def plot_region_comparison(
    comparisons: xr.Dataset | list[xr.Dataset],
    s2_medians: xr.Dataset | list[xr.Dataset],
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    diff_max: float = 0.3,
    figsize_per_row: tuple[float, float] = (15, 2.5),
    row_labels: dict[str, str] | None = None,
):
    """
    Plot one or more regional FAPAR comparisons.

    Columns
    -------
    1. Sentinel-2 median RGB
    2. S2BIOPHYS
    3. SL2P
    4. GROUNDED-EO
    5. S2BIOPHYS - SL2P
    6. S2BIOPHYS - GROUNDED-EO

    Each region is shown as one row.
    """

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    if isinstance(comparisons, xr.Dataset):
        comparisons = [comparisons]

    if isinstance(s2_medians, xr.Dataset):
        s2_medians = [s2_medians]

    if len(comparisons) != len(s2_medians):
        raise ValueError(
            "comparisons and s2_medians must contain the same number of datasets."
        )

    if row_labels is None:
        row_labels = {}

    n_regions = len(comparisons)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        n_regions,
        6,
        figsize=(
            figsize_per_row[0],
            figsize_per_row[1] * n_regions,
        ),
        squeeze=False,
        constrained_layout=False,
    )

    # Leave fixed space at bottom for colorbars.
    # Because the colorbars use separate axes, they do not resize
    # any of the map panels.
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.94,
        bottom=0.10,
        wspace=0.04,
        hspace=0.08,
    )

    algorithms = [
        "s2biophys",
        "sl2p",
        "groundedeo",
    ]

    im = None
    im_diff = None

    # ------------------------------------------------------------------
    # Plot rows
    # ------------------------------------------------------------------

    for row, (comparison, s2_median) in enumerate(zip(comparisons, s2_medians)):
        row_axes = axes[row]

        # Optional sanity check: S2 and retrieval grids should match
        if (
            comparison.sizes["x"] != s2_median.sizes["x"]
            or comparison.sizes["y"] != s2_median.sizes["y"]
        ):
            raise ValueError(
                f"Grid size mismatch for "
                f"{comparison.attrs.get('region', f'row {row}')}: "
                f"comparison={comparison.sizes['y']}x{comparison.sizes['x']}, "
                f"S2={s2_median.sizes['y']}x{s2_median.sizes['x']}."
            )

        extent = [
            float(comparison.x.min()),
            float(comparison.x.max()),
            float(comparison.y.min()),
            float(comparison.y.max()),
        ]

        # --------------------------------------------------------------
        # Sentinel-2 RGB
        # --------------------------------------------------------------

        rgb = np.stack(
            [
                s2_median["B4"].values,  # red
                s2_median["B3"].values,  # green
                s2_median["B2"].values,  # blue
            ],
            axis=-1,
        ).astype(float)

        # Fixed reflectance visualization range
        rgb_vmin = 0.0
        rgb_vmax = 0.3

        rgb = (rgb - rgb_vmin) / (rgb_vmax - rgb_vmin)
        rgb = np.clip(rgb, 0, 1)

        row_axes[0].imshow(
            rgb,
            extent=extent,
            origin="upper",
        )

        # --------------------------------------------------------------
        # Retrieval maps
        # --------------------------------------------------------------

        for ax, algorithm in zip(
            row_axes[1:4],
            algorithms,
        ):
            da = comparison[f"fapar_{algorithm}_mean"]  # .sel(algorithm=algorithm)

            im = ax.imshow(
                da.values,
                extent=extent,
                origin="upper",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

        # --------------------------------------------------------------
        # Difference maps
        # --------------------------------------------------------------

        s2biophys = comparison["fapar_s2biophys_mean"]

        differences = [
            s2biophys - comparison["fapar_sl2p_mean"],
            s2biophys - comparison["fapar_groundedeo_mean"],
        ]

        for ax, da in zip(
            row_axes[4:],
            differences,
        ):
            im_diff = ax.imshow(
                da.values,
                extent=extent,
                origin="upper",
                cmap="RdBu_r",
                vmin=-diff_max,
                vmax=diff_max,
            )

        # --------------------------------------------------------------
        # Row formatting
        # --------------------------------------------------------------

        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])

            # Keep the physical plotting areas identical
            ax.set_aspect("equal")

        region = comparison.attrs.get(
            "region",
            "",
        )

        ecosystem = comparison.attrs.get(
            "ecosystem",
            "",
        )

        label = row_labels.get(
            region,
            ecosystem if ecosystem else region,
        )

        row_axes[0].set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
            labelpad=12,
        )

    # ------------------------------------------------------------------
    # Column titles
    # ------------------------------------------------------------------

    column_titles = [
        "Sentinel-2",
        "S2BIOPHYS",
        "SL2P",
        "GROUNDED-EO GPR",
        r"$\Delta$ SL2P",
        r"$\Delta$ GROUNDED-EO GPR",
    ]

    for ax, title in zip(
        axes[0],
        column_titles,
    ):
        ax.set_title(
            title,
            fontsize=10,
            pad=6,
        )

    # ------------------------------------------------------------------
    # Dedicated colorbar axes
    # ------------------------------------------------------------------

    # These do NOT resize the subplot axes.

    # FAPAR bar underneath columns 2–4
    cax_fapar = fig.add_axes([0.285, 0.035, 0.36, 0.012])

    cbar = fig.colorbar(
        im,
        cax=cax_fapar,
        orientation="horizontal",
    )

    cbar.set_label(
        "FAPAR",
        fontsize=9,
    )

    cbar.ax.tick_params(
        labelsize=8,
    )

    # Difference bar underneath columns 5–6
    cax_diff = fig.add_axes([0.70, 0.035, 0.25, 0.012])

    cbar_diff = fig.colorbar(
        im_diff,
        cax=cax_diff,
        orientation="horizontal",
    )

    cbar_diff.set_label(
        r"$\Delta$ FAPAR (S2BIOPHYS − comparison)",
        fontsize=9,
    )

    cbar_diff.ax.tick_params(
        labelsize=8,
    )

    return fig, axes


# ---------------------------------------------------------------------
# Pairwise scatterplots
# ---------------------------------------------------------------------


def plot_pairwise_comparison(
    comparison: xr.Dataset,
):
    """Plot pairwise pixel-level FAPAR comparisons."""
    pairs = list(combinations(ALGORITHMS, 2))

    fig, axes = plt.subplots(
        1,
        len(pairs),
        figsize=(15, 5),
        constrained_layout=True,
    )

    for ax, (alg_x, alg_y) in zip(
        axes,
        pairs,
    ):
        x = comparison["mean"].sel(algorithm=alg_x).values.ravel()

        y = comparison["mean"].sel(algorithm=alg_y).values.ravel()

        valid = np.isfinite(x) & np.isfinite(y)

        x = x[valid]
        y = y[valid]

        ax.hexbin(
            x,
            y,
            gridsize=60,
            mincnt=1,
            bins="log",
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1,
        )

        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel=LABELS[alg_x],
            ylabel=LABELS[alg_y],
        )
        ax.set_aspect("equal")

        r = np.corrcoef(x, y)[0, 1]
        rmse = np.sqrt(np.mean((x - y) ** 2))
        bias = np.mean(y - x)

        ax.text(
            0.05,
            0.95,
            (f"n = {len(x):,}\nr = {r:.2f}\nRMSE = {rmse:.3f}\nBias = {bias:.3f}"),
            transform=ax.transAxes,
            va="top",
        )

    region_name = comparison.attrs.get(
        "region",
        "Region",
    )

    fig.suptitle(f"{region_name} – pairwise FAPAR comparison")

    return fig
