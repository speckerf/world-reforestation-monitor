#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

INPUT_DIR="${1:-data/fcover_sweden}"
RESOLUTIONS=(100 300 500)

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "Input directory not found: ${INPUT_DIR}" >&2
    exit 1
fi

for year in {2019..2025}; do
    inputs=("${INPUT_DIR}"/sweden_fcover_"${year}"_20m_*.tif)

    if (( ${#inputs[@]} == 0 )); then
        echo "No inputs found for ${year}"
        continue
    fi

    for input_path in "${inputs[@]}"; do
        filename="${input_path##*/}"

        for resolution in "${RESOLUTIONS[@]}"; do
            output_name="${filename/_20m_/_${resolution}m_}"
            output_path="${INPUT_DIR}/${output_name}"

            if [[ -f "${output_path}" ]]; then
                echo "Skipping existing file: ${output_path}"
                continue
            fi

            echo "Aggregating ${filename} to ${resolution} m"

            gdalwarp \
                -r average \
                -tr "${resolution}" "${resolution}" \
                -tap \
                -srcnodata -9999 \
                -dstnodata -9999 \
                -ot Int16 \
                -of GTiff \
                -co COMPRESS=DEFLATE \
                -co PREDICTOR=2 \
                -co TILED=YES \
                -co BIGTIFF=IF_SAFER \
                -co NUM_THREADS=ALL_CPUS \
                -wo NUM_THREADS=ALL_CPUS \
                "${input_path}" \
                "${output_path}"

            echo "Saved: ${output_path}"
        done
    done
done

for tif_path in "${INPUT_DIR}"/*.tif; do
    echo "Setting scale factor 0.0001: ${tif_path}"
    gdal_edit.py -scale 0.0001 "${tif_path}"
done
