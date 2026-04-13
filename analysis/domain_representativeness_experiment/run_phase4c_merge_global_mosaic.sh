#!/usr/bin/env bash

set -euo pipefail

# -------------------------------
# Usage check
# -------------------------------
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/tiles/*.tif"
  exit 1
fi

# -------------------------------
# Input files (expand glob safely)
# -------------------------------
INPUT_FILES=("$@")

# -------------------------------
# Settings
# -------------------------------
OUT_DIR="ood-merged-epsg4326"
VRT="knn-distance_all-traits_global_epsg4326.vrt"
OUT_TIF="knn-distance_all-traits_global_epsg4326.tif"

# 3 arcseconds in degrees / ~ 90 meters
RES="0.000833333333333"

# Number of parallel jobs = all CPUs
NPROC=12

mkdir -p "$OUT_DIR"

echo "Using $NPROC parallel jobs"

# -------------------------------
# Function to warp one file
# -------------------------------
warp_one() {
  local f="$1"
  local base
  base=$(basename "${f%.tif}")
  local out="${OUT_DIR}/${base}_4326.tif"

  gdalwarp \
    -t_srs EPSG:4326 \
    -tr $RES $RES \
    -tap \
    -r bilinear \
    -multi \
    -overwrite \
    -co TILED=YES \
    -co COMPRESS=DEFLATE \
    -co BIGTIFF=YES \
    "$f" "$out"
}

export -f warp_one
export OUT_DIR RES

# -------------------------------
# Run warping in parallel
# -------------------------------
printf "%s\n" "${INPUT_FILES[@]}" | \
  xargs -P "$NPROC" -I {} bash -c 'warp_one "$@"' _ {}

echo "Warping done"

# -------------------------------
# Build VRT
# -------------------------------
echo "Building VRT..."
gdalbuildvrt "$VRT" ${OUT_DIR}/*.tif

# -------------------------------
# Final GeoTIFF
# -------------------------------
echo "Translating to final GeoTIFF..."
gdal_translate \
  -co TILED=YES \
  -co COMPRESS=DEFLATE \
  -co BIGTIFF=YES \
  "$VRT" \
  "$OUT_TIF"

echo "Done: $OUT_TIF"