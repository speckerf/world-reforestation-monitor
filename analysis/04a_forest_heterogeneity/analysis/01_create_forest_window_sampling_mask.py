#!/usr/bin/env python3
"""Create valid forest-window centre masks on a shared raster grid.

An output pixel retains its JRC forest-type code only when every pixel in the
centred neighborhood:

1. belongs to that same JRC forest class;
2. is ESA WorldCover tree cover; and
3. contains no rasterized OSM road or building.

All other output pixels are set to 0. Processing is blockwise with a halo, so
the script can cover Tasmania without loading the full rasters into memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import maximum_filter, minimum_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jrc", required=True, type=Path, help="Aligned JRC forest-type raster.")
    parser.add_argument(
        "--worldcover",
        required=True,
        type=Path,
        help="Aligned ESA WorldCover raster.",
    )
    parser.add_argument(
        "--osm",
        required=True,
        type=Path,
        help="Aligned binary OSM mask: 1=road/building, 0=clear.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output sampling-mask COG.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=25,
        help="Odd window width in pixels; 25 pixels equals 500 m on a 20 m grid.",
    )
    parser.add_argument(
        "--jrc-classes",
        type=int,
        nargs="+",
        default=[1, 10, 20],
        help="JRC classes retained in the output (default: 1 10 20).",
    )
    parser.add_argument(
        "--worldcover-tree-class",
        type=int,
        default=10,
        help="ESA WorldCover tree-cover class (default: 10).",
    )
    parser.add_argument(
        "--processing-block-size",
        type=int,
        default=1024,
        help="Core processing block width and height in pixels.",
    )
    return parser.parse_args()


def assert_same_grid(reference: rasterio.DatasetReader, other: rasterio.DatasetReader) -> None:
    checks = {
        "shape": reference.shape == other.shape,
        "transform": reference.transform.almost_equals(other.transform),
        "CRS": reference.crs == other.crs,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(
            f"{other.name} does not match the JRC grid: {', '.join(failures)}"
        )


def core_windows(width: int, height: int, block_size: int):
    for row_off in range(0, height, block_size):
        height_here = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            width_here = min(block_size, width - col_off)
            yield Window(col_off, row_off, width_here, height_here)


def expanded_window(core: Window, radius: int) -> Window:
    return Window(
        core.col_off - radius,
        core.row_off - radius,
        core.width + 2 * radius,
        core.height + 2 * radius,
    )


def main() -> None:
    args = parse_args()

    if args.window_size <= 0 or args.window_size % 2 == 0:
        raise ValueError("--window-size must be a positive odd integer.")
    if args.processing_block_size <= 0:
        raise ValueError("--processing-block-size must be positive.")
    if not args.jrc_classes:
        raise ValueError("At least one JRC class is required.")
    if 0 in args.jrc_classes:
        raise ValueError("JRC class 0 is reserved for ineligible output pixels.")
    if min(args.jrc_classes) < 0 or max(args.jrc_classes) > 255:
        raise ValueError("JRC output classes must fit in uint8 (1-255).")

    radius = args.window_size // 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with (
        rasterio.open(args.jrc) as jrc_src,
        rasterio.open(args.worldcover) as worldcover_src,
        rasterio.open(args.osm) as osm_src,
    ):
        assert_same_grid(jrc_src, worldcover_src)
        assert_same_grid(jrc_src, osm_src)

        profile = dict(
            driver="COG",
            width=jrc_src.width,
            height=jrc_src.height,
            crs=jrc_src.crs,
            transform=jrc_src.transform,
            dtype="uint8",
            count=1,
            nodata=0,
            compress="DEFLATE",
            blocksize=512,
            overview_resampling="nearest",
            bigtiff="IF_SAFER",
        )

        counts = {code: 0 for code in args.jrc_classes}

        with rasterio.open(args.output, "w", **profile) as dst:
            for core in core_windows(jrc_src.width, jrc_src.height, args.processing_block_size):
                expanded = expanded_window(core, radius)

                # Boundless fill values deliberately invalidate neighborhoods
                # extending beyond the Tasmania raster grid.
                jrc = jrc_src.read(1, window=expanded, boundless=True, fill_value=0)
                worldcover = worldcover_src.read(
                    1, window=expanded, boundless=True, fill_value=0
                )
                osm = osm_src.read(1, window=expanded, boundless=True, fill_value=1)

                all_worldcover_tree = minimum_filter(
                    (worldcover == args.worldcover_tree_class).astype(np.uint8),
                    size=args.window_size,
                    mode="constant",
                    cval=0,
                ).astype(bool)

                no_osm_feature = ~maximum_filter(
                    (osm != 0).astype(np.uint8),
                    size=args.window_size,
                    mode="constant",
                    cval=1,
                ).astype(bool)

                result = np.zeros(jrc.shape, dtype=np.uint8)

                for code in args.jrc_classes:
                    all_same_jrc_class = minimum_filter(
                        (jrc == code).astype(np.uint8),
                        size=args.window_size,
                        mode="constant",
                        cval=0,
                    ).astype(bool)

                    valid = all_same_jrc_class & all_worldcover_tree & no_osm_feature
                    result[valid] = code

                # Remove the halo and write only the core block.
                row_slice = slice(radius, radius + int(core.height))
                col_slice = slice(radius, radius + int(core.width))
                result_core = result[row_slice, col_slice]
                dst.write(result_core, 1, window=core)

                for code in args.jrc_classes:
                    counts[code] += int(np.count_nonzero(result_core == code))

            window_metres = args.window_size * abs(jrc_src.transform.a)
            dst.set_band_description(
                1, f"eligible {window_metres:g} m forest-window centres"
            )
            dst.update_tags(
                value_0="ineligible window centre",
                retained_jrc_codes=",".join(map(str, args.jrc_classes)),
                window_pixels=str(args.window_size),
                criteria=(
                    "all pixels same retained JRC class; all pixels ESA WorldCover "
                    "tree cover; no pixel intersects rasterized OSM road/building"
                ),
                jrc_source=str(args.jrc),
                worldcover_source=str(args.worldcover),
                osm_source=str(args.osm),
            )

    print(f"Created {args.output}")
    for code, count in counts.items():
        print(f"JRC class {code}: {count:,} eligible centre pixels")


if __name__ == "__main__":
    main()

