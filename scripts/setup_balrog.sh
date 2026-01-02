#!/bin/bash
# Setup script for BALROG benchmark
# This script creates a conda environment and installs all dependencies

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BALROG_DIR="$ROOT_DIR/BALROG"

echo "========================================"
echo "  BALROG Benchmark Installation"
echo "========================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment
echo ""
echo "[1/3] Creating conda environment 'balrog'..."
conda create -n balrog python=3.10 -y || {
    echo "Environment 'balrog' may already exist. Continuing..."
}

# Activate environment
echo ""
echo "[2/3] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate balrog

# Install BALROG package
echo ""
echo "[3/3] Installing BALROG package..."
cd "$BALROG_DIR"

# Install in editable mode
pip install -e .

echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "To use BALROG:"
echo "  1. Activate the environment:"
echo "     conda activate balrog"
echo ""
echo "  2. Navigate to the BALROG directory:"
echo "     cd $BALROG_DIR"
echo ""
echo "  3. Run evaluations:"
echo "     python eval.py --agent NaiveAgent --env_name babyai"
echo ""
echo "  4. (Optional) Set API keys as environment variables:"
echo "     export OPENAI_API_KEY=your_key"
echo ""
echo "Available environments: babyai, nethack, textworld"
echo ""
