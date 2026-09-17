#!/usr/bin/env python3
"""Rasterize OSM roads and buildings onto an existing reference raster grid."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom

DEFAULT_LAYERS = (
    "gis_osm_roads_free",
    "gis_osm_buildings_a_free",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rasterize OSM features to the exact CRS, extent, resolution, and "
            "alignment of a reference raster. Output is 1 where any selected "
            "feature touches a pixel and 0 elsewhere."
        )
    )
    parser.add_argument("--osm", required=True, type=Path, help="Input OSM GeoPackage.")
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Raster defining the target grid, for example the 20 m LAIe raster.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output binary COG.")
    parser.add_argument(
        "--layers",
        nargs="+",
        default=list(DEFAULT_LAYERS),
        help="GeoPackage layers to rasterize.",
    )
    parser.add_argument(
        "--no-all-touched",
        action="store_true",
        help="Burn only features intersecting pixel centres (not recommended for roads).",
    )
    return parser.parse_args()


def transformed_geometries(
    gpkg: Path,
    layer: str,
    target_crs: rasterio.crs.CRS,
) -> Iterator[tuple[dict, int]]:
    """Yield valid layer geometries transformed to the target CRS."""
    features = gpd.read_file(gpkg, layer=layer)
    if features.crs is None:
        raise ValueError(f"Layer {layer!r} has no CRS.")

    source_crs = features.crs
    for geometry in features.geometry:
        if geometry is None or geometry.is_empty:
            continue
        yield transform_geom(source_crs, target_crs, geometry.__geo_interface__), 1


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    available_layers = set(gpd.list_layers(args.osm)["name"])
    missing_layers = set(args.layers) - available_layers
    if missing_layers:
        raise ValueError(
            f"Missing GeoPackage layers: {sorted(missing_layers)}. "
            f"Available layers: {sorted(available_layers)}"
        )

    with rasterio.open(args.reference) as reference:
        if reference.crs is None:
            raise ValueError("Reference raster has no CRS.")

        profile = reference.profile.copy()
        profile.update(
            driver="COG",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="DEFLATE",
            blocksize=512,
            overview_resampling="nearest",
        )

        with rasterio.open(args.output, "w", **profile) as destination:
            mask = rasterize(
                (
                    item
                    for layer in args.layers
                    for item in transformed_geometries(args.osm, layer, reference.crs)
                ),
                out_shape=reference.shape,
                transform=reference.transform,
                fill=0,
                default_value=1,
                dtype="uint8",
                all_touched=not args.no_all_touched,
            )
            destination.write(mask, 1)
            destination.set_band_description(1, "OSM roads or buildings")
            destination.update_tags(
                values="0=no mapped feature; 1=road or building",
                source=str(args.osm),
                layers=",".join(args.layers),
                all_touched=str(not args.no_all_touched).lower(),
                reference_grid=str(args.reference),
            )

    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
