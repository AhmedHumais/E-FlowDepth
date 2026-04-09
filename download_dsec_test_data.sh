#!/bin/bash

# --- Configuration ---
dataset_root="data/dsec/test"

# Define the source URLs
URL_EVENTS="https://download.ifi.uzh.ch/rpg/DSEC/test_coarse/test_events.zip"
URL_OTHER=(
    "https://download.ifi.uzh.ch/rpg/DSEC/test_disparity_timestamps.zip"
    "https://download.ifi.uzh.ch/rpg/DSEC/test_forward_optical_flow_timestamps.zip"
)

# --- Execution ---
mkdir -p "$dataset_root"

# 1. Handle the Events zip specifically
echo "--- Processing: test_events.zip ---"
target_event_dir="$dataset_root/event_data"
mkdir -p "$target_event_dir"

# Wget Robustness Strategy:
# -c: Resumes partial downloads
# --tries=0: Retries infinitely
# --read-timeout=30: Restarts if the server hangs without sending data
# --retry-connrefused: Retries if the server is temporarily overwhelmed
wget -c --tries=0 --read-timeout=30 --retry-connrefused "$URL_EVENTS"

echo "Extracting events..."
unzip -q test_events.zip -d "$target_event_dir"
rm test_events.zip

# --- NEW STEP: Download image_timestamps.txt for each sequence ---
echo "--- Downloading image timestamps for sequences ---"
for seq_path in "$target_event_dir"/*/; do
    # Remove trailing slash to get the sequence name
    seq_name=$(basename "$seq_path")
    
    echo "Processing timestamps for: $seq_name"
    
    # Construct the specific URL for this sequence
    timestamp_url="https://download.ifi.uzh.ch/rpg/DSEC/test/${seq_name}/${seq_name}_image_timestamps.txt"
    
    # Download and rename to image_timestamps.txt in one go
    curl -L --retry 5 "$timestamp_url" -o "${seq_path}image_timestamps.txt"
done

# 2. Handle the Timestamps zips
for url in "${URL_OTHER[@]}"; do
    zip_name=$(basename "$url")
    folder_name="${zip_name%.*}"
    target_dir="$dataset_root/$folder_name"
    
    echo "--- Processing: $zip_name ---"
    
    mkdir -p "$target_dir"
    
    # Using curl for smaller files, but adding --retry for safety
    curl -L -O --retry 5 "$url"
    unzip -q "$zip_name" -d "$target_dir"
    rm "$zip_name"
done

echo "--- All downloads complete! Data is located in: $dataset_root ---"