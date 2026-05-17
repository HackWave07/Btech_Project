#!/bin/bash
# setup.sh – Bootstrap script for Streamlit Community Cloud
# Runs before the app starts; installs Python dependencies.

set -e

echo "=== SPC-GAN Setup ==="
echo "Python: $(python --version)"

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Setup complete ==="
