#!/bin/bash

# Stop on error
set -e

# Check for Python 3.11 and install if necessary (this step might need modification based on your OS)
# apt-get install python3.11  # Example for Debian/Ubuntu

# Create a Python virtual environment
if [ ! -d "env" ]; then
    python3 -m venv env
else
    echo "Virtual environment already exists."
fi

# activate the virtual environment
source env/bin/activate

# Upgrade pip and install required dependencies
pip install --upgrade pip
pip install numpy librosa matplotlib scikit-learn midi2audio pretty_midi tensorflow sounddevice pynput scipy argparse tqdm simpleaudio

echo "All project dependencies installed."