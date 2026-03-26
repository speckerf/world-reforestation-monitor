#!/bin/bash

# Define paths
# SOURCE="/Users/felix/Coding_Playground/gdrive_tutorial/exports_gdrive" # local
# WORKDIR="/Users/felix/Coding_Playground/gdrive_tutorial/workdir" # local
# DESTINATION="/Users/felix/Coding_Playground/gdrive_tutorial/results" # local

# Alternative remote paths (commented out)
SOURCE="data-local/download-v03" # remote
WORKDIR="data-local/tmp" # remote
DESTINATION="data-local/merged-v03" # remote

set -e

# Check for required tools
command -v gdalbuildvrt >/dev/null 2>&1 || { echo >&2 "Error: gdalbuildvrt is required but not installed. Aborting."; exit 1; }
command -v gdal_translate >/dev/null 2>&1 || { echo >&2 "Error: gdal_translate is required but not installed. Aborting."; exit 1; }
command -v rio >/dev/null 2>&1 || { echo >&2 "Error: rasterio (rio) is required but not installed. Aborting."; exit 1; }

# Ensure the work directory is clean
rm -rf "$WORKDIR"/*

# Ensure the destination directory exists
mkdir -p "$DESTINATION"

# Find directories containing "_100m_"
echo "Finding directories in $SOURCE..."
DIRS=$(find "$SOURCE" -mindepth 1 -maxdepth 1 -type d -name '*_100m_*')

# If no directories found, exit
if [[ -z "$DIRS" ]]; then
    echo "No directories matching '_100m_' found. Exiting."
    exit 1
fi

# Display matched directories
echo "Matched directories:"
echo "$DIRS"

# Ask for confirmation
# read -p "Do you want to proceed with processing these directories? (y/n): " CONFIRM
# if [[ "$CONFIRM" != "y" ]]; then
#     echo "Operation canceled."
#     exit 0
# fi

echo "Proceeding with processing..."

# Define colors using `tput`
GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
YELLOW=$(tput setaf 3)
BLUE=$(tput setaf 4)
RESET=$(tput sgr0)

# Loop through each directory
for DIR in $DIRS; do
    echo -e "\nProcessing directory: $DIR"

    # Find all TIFF files in the directory
    # TIFF_FILES=($(ls "$DIR"/*.tif 2>/dev/null))
    shopt -s nullglob
    TIFF_FILES=("$DIR"/*.tif)
    shopt -u nullglob

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

        # Extract base name from the first matching file
        FIRST_FILE=$(ls "$DIR"/*"$YEAR"*.tif | head -n 1)
        BASE_NAME=$(basename "$FIRST_FILE" | sed -E 's/-[0-9]+-[0-9]+\.tif/.tif/')

        # Define the output filename in the destination directory
        OUTPUT_FILE="${DESTINATION}/${BASE_NAME}"
        
        # Define temporary VRT file in work directory
        TEMP_VRT="${WORKDIR}/temp_${YEAR}.vrt"

        echo "Merging files for: $YEAR"
        echo "Output will be: $OUTPUT_FILE"
        

        # Determine scale factor based on filename
        if [[ "$OUTPUT_FILE" == *"laie_rtm.mlp.v02_mean_"* || "$OUTPUT_FILE" == *"laie_rtm.mlp.v02_std_"* ]]; then
            SCALE=0.001
            echo "Using scale factor: $SCALE for LAIe data"
        elif [[ "$OUTPUT_FILE" == *"fapar_rtm.mlp.v02_mean_"* || "$OUTPUT_FILE" == *"fcover_rtm.mlp.v02_mean_"* || \
                "$OUTPUT_FILE" == *"fapar_rtm.mlp.v02_std_"* || "$OUTPUT_FILE" == *"fcover_rtm.mlp.v02_std_"* ]]; then
            SCALE=0.0001
            echo "Using scale factor: $SCALE for FAPAR/FCOVER data"
        else
            # raise warning and skip scaling
            echo "${YELLOW}Warning: No specific scale factor defined for $OUTPUT_FILE. Using scale factor: 1${RESET}"
            SCALE=1
        fi

        # continue
        # Find all matching .tif files for the given YEAR
        TIFF_FILES=("$DIR"/*"$YEAR"*.tif)
        NUM_TIFF_FILES=${#TIFF_FILES[@]}

        if [[ $NUM_TIFF_FILES -gt 1 ]]; then
            # continue
            echo "Multiple TIFF files found for $YEAR. Merging "${TIFF_FILES[@]}""

            temp_list=$(mktemp)
            printf '%s\n' "${TIFF_FILES[@]}" > "$temp_list"

            if ! gdalbuildvrt \
                -srcnodata -9999 \
                -vrtnodata -9999 \
                -overwrite \
                -input_file_list "$temp_list" \
                "$TEMP_VRT"
            then
                echo "${RED}Error: Failed to create VRT for $YEAR${RESET}"
                rm -f "$temp_list"
                ((FAILED_FILES++))
                continue
            fi
            rm -f "$temp_list"cd

            # Convert to COG with proper error handling
            if ! gdal_translate -of COG -ot Int16 -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BIGTIFF=IF_SAFER -co NUM_THREADS=ALL_CPUS \
                -a_nodata -9999 -a_scale $SCALE "$TEMP_VRT" "$OUTPUT_FILE"; then
                echo "${RED}Error: Failed to create COG for $YEAR${RESET}"
                rm -f "$TEMP_VRT"
                ((FAILED_FILES++))
                continue
            fi
            
            # Clean up temporary VRT file
            rm -f "$TEMP_VRT"

        elif [[ $NUM_TIFF_FILES -eq 1 ]]; then
            # continue
            echo "Only one TIFF file found for $YEAR. Using gdal_translate..."
            
            # Use gdal_translate for a single file with proper error handling
            if ! gdal_translate -of COG -ot Int16 -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BIGTIFF=IF_SAFER -co NUM_THREADS=ALL_CPUS \
                -a_nodata -9999 -a_scale $SCALE "${TIFF_FILES[0]}" "$OUTPUT_FILE"; then
                echo "${RED}Error: Failed to create COG for $YEAR${RESET}"
                ((FAILED_FILES++))
                continue
            fi
        else
            echo "No TIFF files found for $YEAR in $DIR. Skipping..."
            continue
        fi

        # Run validation only if output file exists and capture output
        if [[ -f "$OUTPUT_FILE" ]]; then
            echo "Validating COG: $OUTPUT_FILE"
            VALIDATION_OUTPUT=$(rio cogeo validate "$OUTPUT_FILE" 2>&1)

            # Print output with colors
            if [[ $VALIDATION_OUTPUT == *"is a valid cloud optimized GeoTIFF"* ]]; then
                printf "${GREEN}%s${RESET}\n" "$VALIDATION_OUTPUT"
            elif [[ $VALIDATION_OUTPUT == *"Warning"* ]]; then
                printf "${YELLOW}%s${RESET}\n" "$VALIDATION_OUTPUT"
            else
                printf "${RED}%s${RESET}\n" "$VALIDATION_OUTPUT"
            fi
            echo "Completed processing for: $YEAR in $DIR"
        else
            echo "${RED}Error: Output file $OUTPUT_FILE was not created${RESET}"
        fi
    done

done

# Final cleanup of work directory
rm -rf "$WORKDIR"/*

echo -e "\n${GREEN}Processing Summary:${RESET}"
echo "Successfully processed: $PROCESSED_FILES files"
if [[ $FAILED_FILES -gt 0 ]]; then
    echo "${RED}Failed to process: $FAILED_FILES files${RESET}"
    exit 1
else
    echo "All yearly mosaics have been processed successfully!"
fi