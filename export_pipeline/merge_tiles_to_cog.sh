#!/bin/bash

# Define paths
SOURCE="/Users/felix/Coding_Playground/gdrive_tutorial/exports_gdrive" # local
WORKDIR="/Users/felix/Coding_Playground/gdrive_tutorial/workdir" # local
DESTINATION="/Users/felix/Coding_Playground/gdrive_tutorial/results" # local

# Alternative remote paths (commented out)
# SOURCE="/Volumes/RAID/felix_oemc/world-reforestation-monitor" # remote
# WORKDIR="/Users/remotelogin3/Documents/felix/temp-workdir" # remote
# DESTINATION="/Volumes/RAID/felix_oemc/results" # remote

# Ensure the work directory is clean
rm -rf "$WORKDIR"/*

# Ensure the destination directory exists
mkdir -p "$DESTINATION"

# List all directories inside SOURCE
echo "Finding directories in $SOURCE..."
DIRS=$(find "$SOURCE" -mindepth 1 -maxdepth 1 -type d)

# Define colors using `tput`
GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
YELLOW=$(tput setaf 3)
RESET=$(tput sgr0)

# Loop through each directory
for DIR in $DIRS; do
    echo -e "\nProcessing directory: $DIR"

    # Find all TIFF files in the directory
    TIFF_FILES=($(ls "$DIR"/*.tif 2>/dev/null))

    # Skip processing if no .tif files are found
    if [[ ${#TIFF_FILES[@]} -eq 0 ]]; then
        echo "No TIFF files found in $DIR. Skipping..."
        continue
    fi

    # Find unique year ranges
    YEAR_RANGES=$(ls "$DIR"/*.tif | grep -o 's_[0-9]\{8\}_[0-9]\{8\}' | sort -u)
    echo "Found year ranges: $YEAR_RANGES"

    # Loop through each unique year range
    for YEAR in $YEAR_RANGES; do
        echo "Processing tiles for: $YEAR in $DIR"

        # Clean and create temporary working directory
        rm -rf "$WORKDIR"/*
        mkdir -p "$WORKDIR"

        # Copy relevant files for the current year range
        echo "Copying relevant files for $YEAR to $WORKDIR..."
        scp "$DIR"/*"$YEAR"*.tif "$WORKDIR"/

        # Change to working directory
        cd "$WORKDIR" || exit

        # Extract base name from the first matching file
        BASE_NAME=$(ls *"$YEAR"*.tif | head -n 1 | sed -E 's/-[0-9]+-[0-9]+.tif/.tif/')

        # Define the output filename
        OUTPUT_FILE="${BASE_NAME}"

        echo "Merging files for: $YEAR"

        # Determine scale factor based on filename
        if [[ "$OUTPUT_FILE" == *"count"* ]]; then
            # Merge the tiles into a single GeoTIFF
            gdal_merge.py -o "$OUTPUT_FILE" -of GTiff -ot UInt8 \
                -co COMPRESS=DEFLATE -co TILED=YES -co BIGTIFF=IF_SAFER \
                -co BIGTIFF=YES --config GDAL_NUM_THREADS ALL_CPUS -a_nodata 255 *"$YEAR"*.tif
        else
            # Merge the tiles into a single GeoTIFF
            gdal_merge.py -o "$OUTPUT_FILE" -of GTiff -ot Int16 \
                -co COMPRESS=DEFLATE -co TILED=YES -co BIGTIFF=IF_SAFER \
                -co BIGTIFF=YES --config GDAL_NUM_THREADS ALL_CPUS -a_nodata -29999 *"$YEAR"*.tif
        fi

        # Determine scale factor based on filename
        if [[ "$OUTPUT_FILE" == *"lai_rtm.mlp_mean_"* || "$OUTPUT_FILE" == *"lai_rtm.mlp_std_"* ]]; then
            SCALE=0.001
        elif [[ "$OUTPUT_FILE" == *"fapar_rtm.mlp_mean_"* || "$OUTPUT_FILE" == *"fcover_rtm.mlp_mean_"* || \
                "$OUTPUT_FILE" == *"fapar_rtm.mlp_std_"* || "$OUTPUT_FILE" == *"fcover_rtm.mlp_std_"* ]]; then
            SCALE=0.0001
        else
            SCALE=1
        fi

        echo "Applying scale factor: $SCALE to $OUTPUT_FILE"
        gdal_edit.py -oo IGNORE_COG_LAYOUT_BREAK=YES -scale "$SCALE" "$OUTPUT_FILE"

        # Convert to Cloud-Optimized GeoTIFF
        echo "Finalizing as Cloud-Optimized GeoTIFF..."
        rio cogeo create "$OUTPUT_FILE" "$OUTPUT_FILE"

        # Run validation and capture output
        VALIDATION_OUTPUT=$(rio cogeo validate "$OUTPUT_FILE")

        # Print output with colors
        if [[ $VALIDATION_OUTPUT == *"is a valid cloud optimized GeoTIFF"* ]]; then
            printf "${GREEN}%s${RESET}\n" "$VALIDATION_OUTPUT"
        elif [[ $VALIDATION_OUTPUT == *"Warning"* ]]; then
            printf "${YELLOW}%s${RESET}\n" "$VALIDATION_OUTPUT"
        else
            printf "${RED}%s${RESET}\n" "$VALIDATION_OUTPUT"
        fi

        # Move the final file to the destination folder
        mv "$OUTPUT_FILE" "$DESTINATION"/

        echo "Completed processing for: $YEAR in $DIR"
    done

done

echo "All yearly mosaics have been processed successfully!"
# #!/bin/bash

# # Define paths
# SOURCE="/Users/felix/Coding_Playground/gdrive_tutorial/exports_gdrive" # local
# WORKDIR="/Users/felix/Coding_Playground/gdrive_tutorial/workdir" # local
# DESTINATION="/Users/felix/Coding_Playground/gdrive_tutorial/results" # local

# # SOURCE="/Volumes/RAID/felix_oemc/world-reforestation-monitor" # remote
# # WORKDIR="/Users/remotelogin3/Documents/felix/temp-workdir" # remote
# # DESTINATION="/Volumes/RAID/felix_oemc/results" # remote

# rm -rf "$WORKDIR"/*


# # Ensure the destination directory exists
# mkdir -p "$DESTINATION"

# # List all directories inside SOURCE
# echo "Finding directories in $SOURCE..."
# DIRS=$(find "$SOURCE" -mindepth 1 -maxdepth 1 -type d)

# # Define colors using `tput`
# GREEN=$(tput setaf 2)
# RED=$(tput setaf 1)
# YELLOW=$(tput setaf 3)
# RESET=$(tput sgr0)


# # Loop through each directory
# for DIR in $DIRS; do
#     echo -e "\nProcessing directory: $DIR"

#     # Find all TIFF files in the directory
#     TIFF_FILES=($(ls "$DIR"/*.tif 2>/dev/null))

#     # Skip processing if no .tif files are found
#     if [[ ${#TIFF_FILES[@]} -eq 0 ]]; then
#         echo "No TIFF files found in $DIR. Skipping..."
#         continue
#     fi

#     # Determine scale factor based on filename
#     if [[ "$OUTPUT_FILE" == *"lai-mean"* || "$OUTPUT_FILE" == *"lai-std"* ]]; then
#         SCALE=0.001
#     elif [[ "$OUTPUT_FILE" == *"fapar-mean"* || "$OUTPUT_FILE" == *"fcover-mean"* || "$OUTPUT_FILE" == *"fapar-std"* || "$OUTPUT_FILE" == *"fcover-std"* ]]; then
#         SCALE=0.0001
#     else
#         SCALE=1
#     fi

#     YEAR_RANGES=$(ls "$DIR"/*.tif | grep -o 's_[0-9]\{8\}_[0-9]\{8\}' | sort -u)
#     echo "Found year ranges: $YEAR_RANGES"
#             # Find unique year ranges
#     # Loop through each unique year range
#     for YEAR in $YEAR_RANGES; do

#         # Check if multiple tiles exist (by searching for coordinates in filenames) for this year: ls "$DIR"/*.tif | grep -q -- '-[0-9]\+-[0-9]\+.tif'


#         # if ls "$DIR"/*.tif 2>/dev/null | grep -q -- '-[0-9]\+-[0-9]\+.tif'; then
#             echo "Detected multiple tiles. Processing by year range..."
        
#             echo "Processing tiles for: $YEAR in $DIR"

#             # Clear the temporary working directory before each iteration
#             echo "Cleaning up temporary directory..."
#             mkdir -p "$WORKDIR"

#             # Copy only relevant files for the current year range
#             echo "Copying relevant files for $YEAR to $WORKDIR..."
#             scp "$DIR"/*"$YEAR"*.tif "$WORKDIR"

#             # Change to working directory
#             cd "$WORKDIR" || exit

#             # Extract base name from first matching file
#             BASE_NAME=$(ls *"$YEAR"*.tif | head -n 1 | sed -E 's/-[0-9]+-[0-9]+.tif//')

#             # Define the output filename
#             OUTPUT_FILE="${BASE_NAME}.tif"

#             echo "Merging files for: $YEAR"

#             # Merge the tiles into a single GeoTIFF
#             gdal_merge.py -o "$OUTPUT_FILE" -of GTiff -ot Int16 -co COMPRESS=DEFLATE -co TILED=YES -co BIGTIFF=IF_SAFER -co BIGTIFF=YES --config GDAL_NUM_THREADS ALL_CPUS -a_nodata -29999 *"$YEAR"*.tif

#             echo "Applying scale factor: $SCALE to $OUTPUT_FILE"
#             gdal_edit.py -scale "$SCALE" "$OUTPUT_FILE"

#             # Convert to Cloud-Optimized GeoTIFF
#             echo "Finalizing as Cloud-Optimized GeoTIFF..."
#             rio cogeo create "$OUTPUT_FILE" "$OUTPUT_FILE"
            
#             # Run validation and capture output
#             VALIDATION_OUTPUT=$(rio cogeo validate "$OUTPUT_FILE")

#             # Print output with colors
#             if [[ $VALIDATION_OUTPUT == *"is a valid cloud optimized GeoTIFF"* ]]; then
#                 printf "${GREEN}%s${RESET}\n" "$VALIDATION_OUTPUT"  # Green for success
#             elif [[ $VALIDATION_OUTPUT == *"Warning"* ]]; then
#                 printf "${YELLOW}%s${RESET}\n" "$VALIDATION_OUTPUT"  # Yellow for warnings
#             else
#                 printf "${RED}%s${RESET}\n" "$VALIDATION_OUTPUT"  # Red for errors
#             fi
#             # Move the final file to the destination folder
#             scp "$OUTPUT_FILE" "$DESTINATION"

#             echo "Completed processing for: $YEAR in $DIR"
        
#         else
#             echo "Detected single file format. Moving directly to results."

#             # Move single file directly to destination without merging
#             SINGLE_FILE="${TIFF_FILES[0]}"  # Use the first found TIFF file
#             BASENAME=$(basename "$SINGLE_FILE")

#             echo "Applying scale factor: $SCALE to $SINGLE_FILE"
#             gdal_edit.py -oo IGNORE_COG_LAYOUT_BREAK=YES -scale "$SCALE" "$SINGLE_FILE"

#             # Convert to Cloud-Optimized GeoTIFF
#             echo "Finalizing as Cloud-Optimized GeoTIFF..."
#             rio cogeo create "$SINGLE_FILE" "$SINGLE_FILE"
            
#             # Run validation and capture output
#             VALIDATION_OUTPUT=$(rio cogeo validate "$SINGLE_FILE")

#             # Print output with colors
#             if [[ $VALIDATION_OUTPUT == *"is a valid cloud optimized GeoTIFF"* ]]; then
#                 printf "${GREEN}%s${RESET}\n" "$VALIDATION_OUTPUT"  # Green for success
#             elif [[ $VALIDATION_OUTPUT == *"Warning"* ]]; then
#                 printf "${YELLOW}%s${RESET}\n" "$VALIDATION_OUTPUT"  # Yellow for warnings
#             else
#                 printf "${RED}%s${RESET}\n" "$VALIDATION_OUTPUT"  # Red for errors
#             fi

#             # Move to destination
#             scp "$SINGLE_FILE" "$DESTINATION"

#             echo "Completed processing for single file: $BASENAME"
#         fi

#         rm -rf "$WORKDIR"/*
#     done
# done

# echo "All yearly mosaics and single files have been processed successfully!"