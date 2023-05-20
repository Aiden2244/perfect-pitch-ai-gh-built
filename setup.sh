#!/bin/bash

# Create a Python virtual environment
python3.11 -m venv env

# activate the virtual environment
source env/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install numpy
pip install librosa
pip install matplotlib
pip install sklearn
pip install midi2audio
pip install pretty_midi
pip install tensorflow
pip install sounddevice
pip install keyboard
pip install scipy
pip install argparse
pip install tqdm

echo "All project dependencies installed."

# activate the virtual environment
source env/bin/activate
