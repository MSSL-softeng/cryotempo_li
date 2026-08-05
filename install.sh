#!/usr/bin/env bash

# Installation script for CryoTEMPO LI (CLEV2ER framework) on linux/macos

set -e  # Exit on any error

# The activation script is now version controlled as ./activate.sh - it is
# no longer generated here. It resolves CLEV2ER_BASE_DIR from its own
# location and holds the per-host data path defaults (see its section 3).
setup_and_run_file=./activate.sh

export CLEV2ER_BASE_DIR=$PWD

if [ ! -f "$setup_and_run_file" ]; then
    echo "ERROR: $setup_and_run_file is missing from the checkout." >&2
    exit 1
fi
chmod +x $setup_and_run_file

# Install Python and dependencies
conda_used=0

if command -v python3.12 &>/dev/null; then
    echo "Python 3.12 is already installed."
else
    if command -v conda &>/dev/null; then
        echo "Conda is available, creating Python 3.12 environment..."
        conda create -n py312 python=3.12 -y
        conda_used=1
    else
        echo "Installing Miniconda..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda
        rm miniconda.sh
        export PATH=$HOME/miniconda/bin:$PATH
        conda init
        conda create -n py312 python=3.12 -y
        conda_used=1
    fi
fi

if [ $conda_used -eq 1 ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate py312
fi

curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.create true
poetry env use python3.12
poetry lock
poetry install

# Get the version of git
git_version=$(git --version | awk '{print $3}')

# Convert the version to a comparable number
# Major version gets padded normally, while minor and patch versions are three digits
version_number=$(echo "$git_version" | awk -F. '{printf "%d%03d%03d", $1, $2, $3}')

# Define the required version (2.20.0)
required_version=$(echo "2.20.0" | awk -F. '{printf "%d%03d%03d", $1, $2, $3}')

# Debugging output to ensure proper values
echo "Detected Git version: $git_version "
echo "Required Git version: >= 2.20.0"

# Compare the versions
if [[ "$version_number" -gt "$required_version" ]]; then
    echo "Git version is greater than 2.20. Performing the task..."
    # Place your task commands here
    # Install pre-commit if not already installed
    if ! command -v pre-commit &>/dev/null; then
        echo "Installing pre-commit..."
        pip install pre-commit
    fi

    pre-commit install
    pre-commit autoupdate
else
    echo "Git version is not greater than 2.20. Skipping the task."
    echo "WARNING: git version is < 2.20. Can not install pre-commit hooks"
    echo "Please upgrade git on your system and re-run the install"
fi


echo ""
echo "-----------------------"
echo "Installation complete. "
echo "-----------------------"
echo "Use \"source $setup_and_run_file\" to set up and activate the environment."
