#!/bin/bash
# Setup script for SmartPlay benchmark
# This script creates a conda environment and installs all dependencies

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SMARTPLAY_DIR="$ROOT_DIR/smartplay"

echo "========================================"
echo "  SmartPlay Benchmark Installation"
echo "========================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment
echo ""
echo "[1/4] Creating conda environment 'smartplay'..."
conda create -n smartplay python=3.10 -y || {
    echo "Environment 'smartplay' may already exist. Continuing..."
}

# Activate environment
echo ""
echo "[2/4] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate smartplay

# Install SmartPlay package
echo ""
echo "[3/4] Installing SmartPlay package..."
cd "$SMARTPLAY_DIR/libs/smartplay"

# Downgrade pip for gym compatibility
python -m pip install --upgrade "pip==23.*"

# Install smartplay
pip install -e .

# Install project requirements
echo ""
echo "[4/4] Installing project requirements..."
cd "$SMARTPLAY_DIR"
pip install -r requirements.txt

echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "To use SmartPlay:"
echo "  1. Activate the environment:"
echo "     conda activate smartplay"
echo ""
echo "  2. Navigate to the smartplay directory:"
echo "     cd $SMARTPLAY_DIR"
echo ""
echo "  3. Set the Python path:"
echo "     export PYTHONPATH=\$PYTHONPATH:\$(pwd)/src"
echo ""
echo "  4. Run experiments:"
echo "     python -m scripts.main"
echo ""
echo "  5. (Optional) Create .env file with API keys:"
echo "     echo 'OPENAI_API_KEY=your_key' > .env"
echo ""
