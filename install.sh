#!/bin/bash

# install.sh
# Installs required Python packages for the demo.
# Assumes Python 3, pybullet, numpy, pytorch, and matplotlib are already installed.

echo "Installing required packages..."

# Install packages identified from the notebook and Python scripts
pip install tqdm gym numpngw torch torchvision matplotlib pybullet

echo "Installation complete."

