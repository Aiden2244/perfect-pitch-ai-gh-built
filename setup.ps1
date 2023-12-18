# Stop on error
$ErrorActionPreference = "Stop"

# Check for Python 3.11 and install if necessary (this step might need modification based on your OS)
# Install-Package -Name python3.11 -ProviderName Chocolatey -Force # Example for Windows using Chocolatey

# Create a Python virtual environment
if (-Not (Test-Path "env")) {
    python -m venv env
} else {
    Write-Host "Virtual environment already exists."
}

# Activate the virtual environment
env\Scripts\Activate.ps1

# Upgrade pip and install required dependencies
python -m pip install --upgrade pip
pip install numpy librosa matplotlib scikit-learn midi2audio pretty_midi tensorflow sounddevice pynput scipy argparse tqdm simpleaudio

Write-Host "All project dependencies installed."
