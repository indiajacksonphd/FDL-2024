#!/bin/bash

# Define base URL for the raw GitHub content
BASE_URL="https://raw.githubusercontent.com/indiajacksonphd/FDL-2024/main"

# Download each script using curl
echo "Downloading train_model.py..."
curl -o train_model.py $BASE_URL/train_model.py
chmod +x train_model.py

echo "Downloading save_outputs.py..."
curl -o save_outputs.py $BASE_URL/save_outputs.py
chmod +x save_outputs.py

echo "Downloading predict_save.py..."
curl -o predict_save.py $BASE_URL/predict_save.py
chmod +x predict_save.py

echo "Downloading generate_data.py..."
curl -o generate_data.py $BASE_URL/generate_data.py
chmod +x generate_data.py

echo "Download complete."
