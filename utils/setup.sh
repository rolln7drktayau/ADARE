#!/bin/bash

# Create virtual environment
python -m venv adare_env

# Activate virtual environment
source adare_env/bin/activate

# Install dependencies
pip install -r ADARE/utils/requirements.txt

# Create necessary output directories
mkdir -p output/plots
mkdir -p output/reports

echo "Setup completed successfully!"
