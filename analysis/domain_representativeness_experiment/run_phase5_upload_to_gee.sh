#!/usr/bin/env bash

set -euo pipefail

BUCKET="gs://felixspecker/open-earth/ood-merged-100m"
ASSET_ROOT="projects/ee-speckerfelix/assets/open-earth/knn-distance_all-traits_100m"
OUTFILE="list.txt"

# previous list
> "$OUTFILE"

echo "Starting uploads..."

gsutil ls ${BUCKET}/*.tif | while read file; do
  
  filename=$(basename "$file")
  tile=$(echo "$filename" | sed 's/knn-distance_all-traits_//; s/_mean.tif//')

  asset_id="${ASSET_ROOT}/${tile}"

  echo "Uploading $tile..."

  # run upload and capture output
  output=$(earthengine upload image \
    --asset_id="$asset_id" \
    --pyramiding_policy=mean \
    --property mgrs_tile=$tile \
    --nodata_value=255 \
    "$file" 2>&1)

  echo "$output"

  # extract task ID
  task_id=$(echo "$output" | grep -oE '[A-Z0-9]{20,}')

  if [[ -n "$task_id" ]]; then
    echo "$task_id" >> "$OUTFILE"
  else
    echo "⚠️ Failed to extract task ID for $tile"
  fi

done

echo "All tasks started. Task IDs saved to $OUTFILE"
