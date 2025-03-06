#!/bin/bash

# Set the target directory where Google Drive folders will be synced locally
TARGET_DIR="/Volumes/RAID/felix_oemc/world-reforestation-monitor"

# Define Google Drive remote name (set in `rclone config`)
REMOTE_NAME="gdrive_felix"

# Define a filter string for selecting folders (modify as needed)
FILTER_STRING="predictions-mlp_1000m_v01"

# List all folders in Google Drive that match the filter
echo "Fetching list of folders matching filter: $FILTER_STRING"
FOLDERS=$(rclone lsf "$REMOTE_NAME:" --dirs-only --filter "+ $FILTER_STRING/**")

# Check if any folders were found
if [[ -z "$FOLDERS" ]]; then
    echo "No folders found matching the filter '$FILTER_STRING'. Exiting."
    exit 1
fi

# Display the list of folders to be synced
echo -e "\nThe following folders will be synced:\n"
echo "$FOLDERS" | nl
echo -e "\n"

# Ask for confirmation once
read -p "Do you want to sync all these folders? (y/n) " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Sync cancelled."
    exit 0
fi

# Sync each folder
for FOLDER in $FOLDERS; do
    LOCAL_PATH="$TARGET_DIR/$FOLDER"
    REMOTE_PATH="$REMOTE_NAME:$FOLDER"

    echo "Syncing: $REMOTE_PATH --> $LOCAL_PATH"
    rclone sync --progress "$REMOTE_PATH" "$LOCAL_PATH"
done

echo "All folders have been synced successfully."
