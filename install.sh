#!/usr/bin/env bash

# Installation script for CryoTEMPO LI (CLEV2ER framework) on linux/macos

set -e  # Exit on any error

setup_file=./setup_env.sh

export CLEV2ER_BASE_DIR=$PWD

echo "export CLEV2ER_BASE_DIR=$PWD" > $setup_file

if [ -z "${PYTHONPATH}" ]; then
    echo "export PYTHONPATH=$CLEV2ER_BASE_DIR/src" >> $setup_file
else
    echo "export PYTHONPATH=$CLEV2ER_BASE_DIR/src:$PYTHONPATH" >> $setup_file
fi

conda_used=0

# Check if python3.12 is installed
if command -v python3.12 &>/dev/null; then
    echo "Python 3.12 is installed at:"
    command -v python3.12
    python3.12 -V
else
    echo "Python 3.12 is not installed"
    # Check if conda is installed
    if command -v conda &>/dev/null; then
        echo "Conda is already installed, creating a new environment with Python 3.12..."
        conda create -n py312 python=3.12 -y
        echo "Python 3.12 environment 'py312' created."
        source "$(conda info --base)/etc/profile.d/conda.sh"  # Ensure conda is initialized
        conda activate py312
        conda_used=1
    else
        echo "Conda is not installed, installing Miniconda..."
        
        # Download Miniconda installer for Linux or macOS (adjust URL for your platform)
        MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
        
        # For macOS, use the following line:
        # MINICONDA_INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
        
        # Download Miniconda installer
        wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER
        
        # Make the installer executable
        chmod +x $MINICONDA_INSTALLER
        
        # Install Miniconda (non-interactively)
        ./$MINICONDA_INSTALLER -b -p $HOME/miniconda
        
        # Clean up the installer
        rm $MINICONDA_INSTALLER
        
        echo "Miniconda installed successfully."

        # Initialize Conda
        $HOME/miniconda/bin/conda init

        # Reload shell configuration without exiting the script
        source $HOME/.bashrc || source $HOME/.zshrc || true

        # Create the environment with Python 3.12
        $HOME/miniconda/bin/conda create -n py312 python=3.12 -y
        echo "Python 3.12 environment 'py312' created."

        source $HOME/miniconda/etc/profile.d/conda.sh
        conda activate py312
        conda_used=1
    fi
fi

# Install/reinstall Poetry using the official installer
curl -sSL https://install.python-poetry.org | python3 -

# Make sure that Poetry creates its own venv and doesn't reuse Conda
poetry config virtualenvs.create true

if [ $conda_used -eq 1 ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate py312
fi

# Set Poetry to use Python 3.12
poetry env use python3.12

# Lock and install dependencies
poetry lock
poetry install

export ppath=$(poetry env info --path)

echo "export PATH=$CLEV2ER_BASE_DIR/src/clev2er/tools:${ppath}/bin:\$PATH" >> $setup_file

export PATH=$CLEV2ER_BASE_DIR/src/clev2er/tools:${ppath}/bin:$PATH

echo "# set the dir where products will be written within the following" >> $setup_file
echo "# sub-directories:" >> $setup_file
echo "# $CT_PRODUCT_BASEDIR/<baseline>/<version:03>/LAND_ICE/<ANTARC,GREENL>/<YYYY>/<MM>/" >> $setup_file
echo "export CT_PRODUCT_BASEDIR=/raid6/cryo-tempo/product_baselines" >> $setup_file

echo "# set the dir where logs will be written to" >> $setup_file
echo "export CT_LOG_DIR=/tmp" >> $setup_file

echo "# location of CPOM data archive base directory (CPDATA_DIR)" >> $setup_file
echo "export CPDATA_DIR=/cpdata" >> $setup_file

echo "# L1b location: should contain LRM,SIN/<YYYY>/<MM>/" >> $setup_file
echo "export L1B_BASE_DIR=${CPDATA_DIR}/SATS/RA/CRY/L1B" >> $setup_file

echo "# FES2014b tides base directory containing LRM,SIN/<YYYY>/<MM>/CS*.fes2014b.nc" >> $setup_file
echo "export FES2014B_BASE_DIR=/raid6/cpdata/SATS/RA/CRY/L1B/FES2014" >> $setup_file

echo "# CATS2008a base dir, containing <YYYY>/<MM>/CS*_cats2008a_tides.nc" >> $setup_file
echo "export CATS2008A_BASE_DIR=/raid6/cpdata/SATS/RA/CRY/L2I/SIN/CATS_tides" >> $setup_file

echo "# Base directory to find uncertainty LUTs" >> $setup_file
echo "export CS2_UNCERTAINTY_BASE_DIR=/raid6/cryo-tempo/land_ice/uncertainty" >> $setup_file


echo "# for multi-processing/shared mem support set ulimit" >> $setup_file
echo "# to make sure you have enough file descriptors available" >> $setup_file
echo "ulimit -n 8192" >> $setup_file

pre-commit install
pre-commit autoupdate

echo "Installation complete!"
echo "To set up to use the CryoTEMPO Land Ice software:"
echo "-------------------------------------"
echo "cd $PWD"
echo "poetry shell"
echo ". setup_env.sh"
echo "-------------------------------------"
echo "Note, you may need to edit environment variables in setup_env.sh for paths used in "
echo "your particular system"
