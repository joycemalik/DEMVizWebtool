#!/usr/bin/env bash
set -e

# Install system dependencies for GDAL/rasterio
apt-get update
apt-get install -y gdal-bin libgdal-dev libexpat1

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
