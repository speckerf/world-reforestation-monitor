#!/bin/bash

# Usage: ./process_lai.sh <gcs_repo_path> <local_temp_folder> <local_output_folder> <scaling_factor>

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <gcs_repo_path> <filename> <scaling_factor>"
    exit 1
fi

# Set parameters
GCS_REPO_PATH=$1
FILENAME=$2
SCALING_FACTOR=$3
LOCAL_TEMP_FOLDER="data-local/trait_maps/intermediate"
LOCAL_PROCESSING_FOLDER="data-local/trait_maps/processing"
LOCAL_OUTPUT_FOLDER="data-local/trait_maps/output"

# Logging function for cleaner logs
log() {
    echo "$(date +"%Y-%m-%d %T") - $1"
}

# Sync GCS to local
sync_gcs_to_local() {
    # Create local directories if they do not exist
    mkdir -p "$LOCAL_TEMP_FOLDER/$FILENAME"
    
    # Prompt the user for confirmation before syncing
    log "Ready to sync from GCS: $GCS_REPO_PATH/ to $LOCAL_TEMP_FOLDER/$FILENAME"
    read -p "Do you want to continue with the sync (yes/no)? " CONFIRMATION

    if [[ "$CONFIRMATION" != "yes" ]]; then
        log "Sync aborted by user."
        exit 1
    fi

    log "Syncing from GCS: $GCS_REPO_PATH/ to $LOCAL_TEMP_FOLDER"
    
    # Perform rsync from GCS to local, -m enables parallel transfer, -d removes extraneous files from local
    gcloud storage rsync --delete-unmatched-destination-objects -r --dry-run "$GCS_REPO_PATH/" "$LOCAL_TEMP_FOLDER"
    # gcloud storage rsync --delete-unmatched-destination-objects -r "$GCS_REPO_PATH/" "$LOCAL_TEMP_FOLDER"
    
    log "Sync completed successfully."
}

# Merge TIFF files
merge_tifs() {
    log "Merging TIFF files..."
    # Find all the TIFF files in the folder
    FILES=$(find "$LOCAL_TEMP_FOLDER/$FILENAME" -name "*.tif")
    
    if [ -z "$FILES" ]; then
        log "No TIFF files found in $LOCAL_TEMP_FOLDER/$FILENAME."
        exit 1
    fi
    
    # Merge the files into a single output TIFF
    gdal_merge.py -co BIGTIFF=IF_SAFER  -co COMPRESS=DEFLATE -o "$LOCAL_PROCESSING_FOLDER/$FILENAME.tif" $FILES
    
    log "TIFF merge completed."
}

# Add metadata (scale and offset)
edit_metadata() {
    SCALE=$(echo "1 / $SCALING_FACTOR" | bc -l)
    OFFSET=0
    log "Adding metadata with scale: $SCALE and offset: $OFFSET"
    
    # Edit the metadata of the merged file
    gdal_edit.py -scale $SCALE -offset $OFFSET -a_nodata 0 "$LOCAL_PROCESSING_FOLDER/$FILENAME.tif"
    
    log "Metadata added successfully."
}

# Convert to Cloud Optimized GeoTIFF
convert_to_cog() {
    log "Converting to Cloud Optimized GeoTIFF..."
    
    # Use gdal_translate to create a COG
    gdal_translate -of COG -co COMPRESS=DEFLATE -co BIGTIFF=IF_SAFER \
        "$LOCAL_PROCESSING_FOLDER/$FILENAME.tif" "$LOCAL_OUTPUT_FOLDER/$FILENAME.tif"
    
    log "Conversion to COG completed."
    
    # Remove the non-COG file
    rm "$LOCAL_PROCESSING_FOLDER/$FILENAME.tif"
    log "Non-COG file removed."
}

# Main script
log "Process started."

sync_gcs_to_local
merge_tifs
edit_metadata
convert_to_cog

log "Process completed successfully."
