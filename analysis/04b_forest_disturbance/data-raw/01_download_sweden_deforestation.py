#!/usr/bin/env python3
"""Download and clean Skogsstyrelsen completed-harvest polygons.

The product documentation defines ``Avvdatum`` as the harvest date. In
contrast, ``Arendear`` is the year in which the notification/application was
registered, so the date filter below intentionally uses ``Avvdatum``.
"""

from __future__ import annotations

import argparse
from datetime import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

LAYER_URL = (
    "https://geodpags.skogsstyrelsen.se/arcgis/rest/services/"
    "Geodataportal/GeodataportalVisaSkogsbruk/MapServer/6"
)
QUERY_URL = f"{LAYER_URL}/query"
SOURCE_CRS = "EPSG:3006"
OUT_FIELDS = [
    "OBJECTID",
    "Arendear",
    "Avvdatum",
    "Arealha",
    "Avverktyp",
    "Skogstyp",
    "Kalladatum",
    "Kallaareal",
    "Forebild",
    "Efterbild",
]
AREA_LABELS = ["<1", "1-2", "2-5", "5-10", "10-25", "25-50", "50-100", ">100"]
AREA_BINS = [-float("inf"), 1, 2, 5, 10, 25, 50, 100, float("inf")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "data"
        / "reference"
        / "sweden_completed_harvests_2020_2024.gpkg",
    )
    return parser.parse_args()


def get_layer_metadata(session: requests.Session) -> dict:
    response = session.get(LAYER_URL, params={"f": "json"}, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    if "error" in metadata:
        raise RuntimeError(f"ArcGIS metadata error: {metadata['error']}")
    return metadata


def download_harvests(
    session: requests.Session,
    start_year: int,
    end_year: int,
    batch_size: int,
    max_retries: int = 5,
) -> gpd.GeoDataFrame:
    where = (
        f"Avvdatum >= TIMESTAMP '{start_year}-01-01 00:00:00' AND "
        f"Avvdatum < TIMESTAMP '{end_year + 1}-01-01 00:00:00'"
    )

    pages: list[gpd.GeoDataFrame] = []
    offset = 0

    while True:
        params = {
            "where": where,
            "outFields": ",".join(OUT_FIELDS),
            "returnGeometry": "true",
            "outSR": 3006,
            # Sub-pixel simplification: the finest downstream grid is 20 m.
            "maxAllowableOffset": 1,
            "orderByFields": "OBJECTID ASC",
            "resultOffset": offset,
            "resultRecordCount": batch_size,
            "f": "geojson",
        }

        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(
                    QUERY_URL,
                    params=params,
                    timeout=120,
                )
                response.raise_for_status()

                payload = response.json()

                if "error" in payload:
                    raise RuntimeError(f"ArcGIS query error: {payload['error']}")

                features = payload.get("features", [])
                break

            except (
                requests.RequestException,
                ValueError,  # Invalid JSON response
                RuntimeError,
            ) as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to download page at offset {offset:,} "
                        f"after {max_retries} attempts"
                    ) from exc

                wait_seconds = 2 ** (attempt - 1)

                print(
                    f"Attempt {attempt}/{max_retries} failed at "
                    f"offset {offset:,}: {exc}. "
                    f"Retrying in {wait_seconds}s...",
                    flush=True,
                )
                time.sleep(wait_seconds)

        if not features:
            break

        page = gpd.GeoDataFrame.from_features(
            features,
            crs=SOURCE_CRS,
        )
        pages.append(page)

        offset += len(features)
        print(f"Downloaded {offset:,} polygons", flush=True)

        if len(features) < batch_size:
            break

    if not pages:
        raise RuntimeError(
            f"The query returned no harvest polygons for {start_year}–{end_year}"
        )

    return gpd.GeoDataFrame(
        pd.concat(pages, ignore_index=True),
        crs=SOURCE_CRS,
    )


def clean_harvests(raw: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    gdf = raw.copy()
    # ArcGIS GeoJSON serializes date fields as Unix epoch milliseconds.
    date_values = pd.to_numeric(gdf["Avvdatum"], errors="coerce")
    gdf["harvest_date"] = pd.to_datetime(
        date_values, unit="ms", errors="coerce", utc=True
    ).dt.tz_localize(None)
    gdf["disturbance_year"] = gdf["harvest_date"].dt.year.astype("Int64")
    gdf["area_ha"] = pd.to_numeric(gdf["Arealha"], errors="coerce")
    gdf["disturbance_type"] = gdf["Avverktyp"].astype("string")
    gdf["disturbance_id"] = "skogsstyrelsen_" + gdf["OBJECTID"].astype("Int64").astype(
        str
    )

    empty = gdf.geometry.isna() | gdf.geometry.is_empty
    invalid = ~gdf.geometry.is_valid & ~empty
    duplicate_ids = gdf.duplicated("OBJECTID", keep="first")
    duplicate_geometries = gdf.geometry.duplicated(keep="first") & ~empty
    missing_dates = gdf["harvest_date"].isna()
    nonpositive_area = gdf["area_ha"].isna() | (gdf["area_ha"] <= 0)
    checks = {
        "input": len(gdf),
        "missing_dates": int(missing_dates.sum()),
        "empty_geometries": int(empty.sum()),
        "invalid_geometries": int(invalid.sum()),
        "duplicate_objectids": int(duplicate_ids.sum()),
        "duplicate_geometries": int(duplicate_geometries.sum()),
        "missing_or_nonpositive_area": int(nonpositive_area.sum()),
    }

    gdf = gdf.loc[~(empty | missing_dates | nonpositive_area | duplicate_ids)].copy()
    if (~gdf.geometry.is_valid).any():
        gdf.geometry = gdf.geometry.make_valid()
        gdf = gdf.loc[~(gdf.geometry.isna() | gdf.geometry.is_empty)].copy()
    gdf["area_bin"] = pd.cut(
        gdf["area_ha"], bins=AREA_BINS, labels=AREA_LABELS, right=False
    ).astype("string")
    gdf = (
        gdf[
            [
                "disturbance_id",
                "harvest_date",
                "disturbance_year",
                "area_ha",
                "area_bin",
                "disturbance_type",
                "Skogstyp",
                "Kalladatum",
                "Kallaareal",
                "Forebild",
                "Efterbild",
                "Arendear",
                "OBJECTID",
                "geometry",
            ]
        ]
        .sort_values(["harvest_date", "OBJECTID"])
        .reset_index(drop=True)
    )
    checks["output"] = len(gdf)
    return gdf, checks


def print_summary(gdf: gpd.GeoDataFrame, checks: dict[str, int], output: Path) -> None:
    print("\nCleaning checks")
    for name, value in checks.items():
        print(f"  {name.replace('_', ' ')}: {value:,}")
    print(
        f"  harvest date range: {gdf.harvest_date.min().date()} to {gdf.harvest_date.max().date()}"
    )
    print(
        f"  disturbance years: {gdf.disturbance_year.min()} to {gdf.disturbance_year.max()}"
    )
    print("\nArea (ha) distribution")
    print(
        gdf.area_ha.describe(
            percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        ).to_string()
    )
    print("\nSuggested area bins (left-inclusive)")
    print(gdf.area_bin.value_counts().reindex(AREA_LABELS, fill_value=0).to_string())
    print(f"\nSaved {len(gdf):,} polygons in {gdf.crs} to {output}")


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start-year must not be later than end-year")
    with requests.Session() as session:
        metadata = get_layer_metadata(session)
        batch_size = int(metadata.get("maxRecordCount", 2000))
        raw = download_harvests(session, args.start_year, args.end_year, batch_size)
    cleaned, checks = clean_harvests(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_file(args.output, layer="completed_harvests", driver="GPKG", index=False)
    print_summary(cleaned, checks, args.output)


if __name__ == "__main__":
    main()
