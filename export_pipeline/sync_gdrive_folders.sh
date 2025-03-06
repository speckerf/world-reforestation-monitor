# Description: Syncs all folders from a Google Drive remote that match a filter pattern to a local directory.
#!/bin/bash

# Set the target directory where Google Drive folders will be synced locally
# TARGET_DIR="/Users/felix/Coding_Playground/gdrive_tutorial" # Local
TARGET_DIR="/Volumes/RAID/felix_oemc/world-reforestation-monitor"

# Define Google Drive remote name (set in `rclone config`)
# REMOTE_NAME="drive" # Local
REMOTE_NAME="gdrive_felix"

# Define a filter pattern for selecting folders
FILTER_PATTERN="predictions-mlp_1000m_v01"

# List all matching folders and extract only their names
echo "Fetching list of folders matching filter: $FILTER_PATTERN"
FOLDERS=$(rclone lsd "$REMOTE_NAME:" | awk -v pattern="$FILTER_PATTERN" '$NF ~ pattern {print $NF}')

# Check if any folders were found
if [[ -z "$FOLDERS" ]]; then
    echo "No folders found matching the filter '$FILTER_PATTERN'. Exiting."
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
while read -r FOLDER; do
    LOCAL_PATH="$TARGET_DIR/$FOLDER"
    REMOTE_PATH="$REMOTE_NAME:$FOLDER"

    echo "Syncing: $REMOTE_PATH --> $LOCAL_PATH"
    rclone sync --progress "$REMOTE_PATH" "$LOCAL_PATH"
done <<< "$FOLDERS"

echo "All selected folders have been synced successfully."
