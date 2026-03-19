#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-.}"
OUTPUT_DIR="${2:-./output}"
SEEDS="1 2 3 4 5"

find "$INPUT_DIR" -type d -print | while IFS= read -r dir; do
    echo "Checking: $dir"

    # Build corresponding output directory
    rel="${dir#$INPUT_DIR/}"
    out_dir="$OUTPUT_DIR/"
    mkdir -p "$out_dir"

    found_any=0
    tiles_tmp="$(mktemp)"

    for f in "$dir"/knn-distance_all-traits_*_seed-*.tif; do
        if [ ! -f "$f" ]; then
            continue
        fi

        found_any=1

        base=$(basename "$f")
        tile=$(printf '%s\n' "$base" | sed -E 's/^knn-distance_all-traits_(.*)_seed-[0-9]+\.tif$/\1/')
        printf '%s\n' "$tile" >> "$tiles_tmp"
    done

    if [ "$found_any" -eq 0 ]; then
        rm -f "$tiles_tmp"
        continue
    fi

    sort -u "$tiles_tmp" | while IFS= read -r tile; do
        [ -n "$tile" ] || continue

        echo "  Tile: $tile"

        seed_files=()
        missing=0

        for seed in $SEEDS; do
            f="$dir/knn-distance_all-traits_${tile}_seed-${seed}.tif"
            if [ -f "$f" ]; then
                seed_files+=("$f")
            else
                echo "    Missing seed $seed: $(basename "$f")"
                missing=1
            fi
        done

        if [ "$missing" -eq 0 ]; then
            out="$out_dir/knn-distance_all-traits_${tile}_mean.tif"
            echo "    Writing to: $out"

            input_args=()
            for f in "${seed_files[@]}"; do
                input_args+=(-i "$f")
            done

            gdal raster calc \
                "${input_args[@]}" \
                -o "$out" \
                --dialect builtin \
                --flatten \
                --calc mean \
                --ot Byte \
                --nodata 255 \
                --overwrite \
                --co COMPRESS=DEFLATE \
                --co TILED=YES \
                --co BLOCKXSIZE=512 \
                --co BLOCKYSIZE=512

            gdal_edit.py "$out" -offset 0 -scale 0.04

            echo "    Done."
        else
            echo "    Skipping tile $tile: incomplete seed set"
        fi
    done

    rm -f "$tiles_tmp"
done
