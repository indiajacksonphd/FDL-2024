#!/bin/bash

# Define base URL for the raw GitHub content
BASE_URL="https://raw.githubusercontent.com/indiajacksonphd/FDL-2024/main"

# Download each script using curl
echo "Downloading supermag_UQ_OI_IJ.py..."
curl -o supermag_UQ_OI_IJ.py $BASE_URL/supermag_UQ_OI_IJ.py
chmod +x supermag_UQ_OI_IJ.py

echo "Downloading sec.py..."
curl -o sec.py $BASE_URL/sec.py
chmod +x sec.py

echo "Downloading train_model.py..."
curl -o supermag_api.py $BASE_URL/supermag_api.py
chmod +x supermag_api.py

echo "Downloading save_outputs.py..."
curl -o run_supermag_5m.sh $BASE_URL/run_supermag_5m.sh
chmod +x run_supermag_5m.sh

echo "Download complete."
