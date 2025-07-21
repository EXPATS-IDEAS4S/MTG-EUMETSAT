#!/bin/bash

# Base directories
BASE_DIR="/data/sat/mtg/fci/oca"
NETCDF_DIR="$BASE_DIR/netcdf"
QUICKLOOKS_DIR="$BASE_DIR/quicklooks"

# ✅ Control whether to extract images
EXTRACT_IMAGES=true

# Define quicklook types
TYPES=("CER" "COT" "CTH" "CTPH" "CTP" "CTT")

# Ensure root folders exist
mkdir -p "$NETCDF_DIR"
mkdir -p "$QUICKLOOKS_DIR"

# Find all .zip files recursively
find "$BASE_DIR" -type f -name "*.zip" | while read -r zipfile; do
    echo "📦 Processing $zipfile"

    # Extract date from filename (2nd 14-digit timestamp)
    filename=$(basename "$zipfile")
    date_str=$(echo "$filename" | grep -oP '\d{14}' | tail -n 2 | head -n 1)
    year=${date_str:0:4}
    month=${date_str:4:2}
    day=${date_str:6:2}

    echo "📅 Date extracted: $year-$month-$day"

    # Define target paths
    netcdf_target="$NETCDF_DIR/$year/$month/$day"
    quicklook_base="$QUICKLOOKS_DIR/$year/$month/$day"

    # Create necessary directories
    mkdir -p "$netcdf_target"
    if [ "$EXTRACT_IMAGES" = true ]; then
        for type in "${TYPES[@]}"; do
            mkdir -p "$quicklook_base/$type"
        done
    fi

    # Create temporary extraction directory
    TEMP_DIR=$(mktemp -d)

    # Extract zip contents to temp dir
    unzip -q "$zipfile" -d "$TEMP_DIR"

    # Move .nc files
    find "$TEMP_DIR" -type f -name "*.nc" -exec mv -v {} "$netcdf_target/" \;

    # Conditionally extract images
    if [ "$EXTRACT_IMAGES" = true ]; then
        find "$TEMP_DIR" -type f -name "*.jpg" | while read -r img; do
            for type in "${TYPES[@]}"; do
                if [[ "$img" == *"$type"* ]]; then
                    mv -v "$img" "$quicklook_base/$type/"
                    break
                fi
            done
        done
    fi

    # Cleanup
    rm -rf "$TEMP_DIR"
done

echo "✅ Done. NetCDF files organized${EXTRACT_IMAGES:+ with quicklooks extracted}."


# 211127