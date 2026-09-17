import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from exactextract import exact_extract
from shapely.geometry import box

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ANALYSIS_DIR / "data"
OUTPUT_DIR = ANALYSIS_DIR / "results"
FCOVER_DIR = DATA_DIR / "fcover_sweden"
OUTPUT_PATH = OUTPUT_DIR / "sweden_fcover_pre_post_intervention_zonal_means.csv"

FCOVER_FILENAME = re.compile(
    r"sweden_fcover_(?P<year>\d{4})_"
    r"(?P<resolution_m>\d+)m_"
    r"epsg-(?P<epsg>\d+)_"
    r"(?P<tile>[0-9]{2}[A-Z]{1,3})\.tif$"
)
INCLUDE_COLUMNS = [
    "disturbance_id",
    "area_ha",
    "disturbance_type",
    "disturbance_year",
]


def raster_catalog(
    raster_dir: Path,
) -> dict[tuple[int, int], list[Path]]:
    """Index FCOVER tiles by (year, resolution_m)."""
    catalog = {}
    seen = set()

    for raster_path in sorted(raster_dir.glob("*.tif")):
        match = FCOVER_FILENAME.fullmatch(raster_path.name)

        if match is None:
            print(f"Skipping unrecognised raster filename: {raster_path.name}")
            continue

        year = int(match["year"])
        resolution_m = int(match["resolution_m"])

        if not 2019 <= year <= 2025:
            continue

        tile_key = (
            year,
            resolution_m,
            int(match["epsg"]),
            match["tile"],
        )

        if tile_key in seen:
            raise ValueError(f"Duplicate FCOVER tile: {tile_key}")

        seen.add(tile_key)
        catalog.setdefault((year, resolution_m), []).append(raster_path)

    if not catalog:
        raise FileNotFoundError(f"No FCOVER rasters found in {raster_dir}")

    return catalog


def polygons_intersecting_raster(
    polygons: gpd.GeoDataFrame,
    raster_path: Path,
) -> gpd.GeoDataFrame:
    """Return polygons that intersect a raster footprint, in the raster CRS."""

    with rasterio.open(raster_path) as raster:
        if raster.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        raster_crs = raster.crs
        raster_bounds = box(*raster.bounds)

    polygons_in_raster_crs = polygons.to_crs(raster_crs)
    return polygons_in_raster_crs[
        polygons_in_raster_crs.geometry.intersects(raster_bounds)
    ].copy()


def zonal_extractions(
    polygons: gpd.GeoDataFrame,
    raster_path: Path,
) -> pd.DataFrame:
    """Extract mean FCOVER for every polygon intersecting a raster."""

    intersecting = polygons_intersecting_raster(polygons, raster_path)
    if intersecting.empty:
        return pd.DataFrame(columns=[*INCLUDE_COLUMNS, "mean"])

    return exact_extract(
        raster_path,
        intersecting,
        ops=["mean"],
        include_cols=INCLUDE_COLUMNS,
        output="pandas",
    )


def main() -> pd.DataFrame:
    polygon_path = DATA_DIR / "sweden_completed_harvests_2020_2024.gpkg"
    polygons = gpd.read_file(polygon_path)
    polygons = polygons[
        polygons.geometry.notna()
        & ~polygons.geometry.is_empty
        & ~polygons.geometry.type.isin(["GeometryCollection"])
    ].copy()

    print(f"Sweden: {len(polygons)} polygons")

    catalog = raster_catalog(FCOVER_DIR)
    resolutions = sorted({resolution for _, resolution in catalog})
    disturbance_years = sorted(polygons["disturbance_year"].dropna().unique())
    extractions = []

    for disturbance_year_value in disturbance_years:
        disturbance_year = int(disturbance_year_value)
        polygons_for_year = polygons[
            polygons["disturbance_year"] == disturbance_year_value
        ]
        print(f"Disturbance year {disturbance_year}: {len(polygons_for_year)} polygons")

        for period, fcover_year in (
            ("pre", disturbance_year - 1),
            ("post", disturbance_year + 1),
        ):
            for resolution_m in resolutions:
                raster_paths = catalog.get((fcover_year, resolution_m), [])

                if not raster_paths:
                    print(
                        "Skipping missing rasters: "
                        f"year={fcover_year}, resolution={resolution_m} m"
                    )
                    continue

                for raster_path in raster_paths:
                    extracted = zonal_extractions(
                        polygons_for_year,
                        raster_path,
                    )

                    if extracted.empty:
                        continue

                    match = FCOVER_FILENAME.fullmatch(raster_path.name)

                    extracted = extracted.rename(columns={"mean": "fcover_mean"})
                    extracted["fcover_year"] = fcover_year
                    extracted["intervention_period"] = period
                    extracted["resolution_m"] = resolution_m
                    extracted["raster_file"] = raster_path.name
                    extracted["raster_epsg"] = int(match["epsg"])
                    extracted["tile"] = match["tile"]

                    extractions.append(extracted)

                    print(
                        f"  {period}: {fcover_year}, {resolution_m} m, "
                        f"{match['tile']} -> "
                        f"{len(extracted)} intersecting polygons"
                    )

    if not extractions:
        raise RuntimeError("No polygon/raster intersections were extracted")

    result = pd.concat(extractions, ignore_index=True)
    result = result.sort_values(
        ["disturbance_year", "disturbance_id", "intervention_period", "resolution_m"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(result)} rows to {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    main()
