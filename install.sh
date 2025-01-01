#!/usr/bin/env bash

# Installation script for CryoTEMPO LI (CLEV2ER framework) on linux/macos

set -e  # Exit on any error

setup_and_run_file=./ct_activate.sh

export CLEV2ER_BASE_DIR=$PWD

echo "#!/usr/bin/env bash" > $setup_and_run_file
echo "" >> $setup_and_run_file
echo "# Combined setup and run script for CryoTEMPO LI" >> $setup_and_run_file
echo "set -e" >> $setup_and_run_file
echo "" >> $setup_and_run_file
echo "# Activate Poetry virtual environment" >> $setup_and_run_file
echo "VENV_PATH=\$(poetry env info --path)" >> $setup_and_run_file
echo "if [ -z \"\$VENV_PATH\" ]; then" >> $setup_and_run_file
echo "    echo \"Poetry virtual environment not found. Did you set it up?\"" >> $setup_and_run_file
echo "    exit 1" >> $setup_and_run_file
echo "fi" >> $setup_and_run_file
echo "source \"\$VENV_PATH/bin/activate\"" >> $setup_and_run_file
echo "" >> $setup_and_run_file

# Export environment variables to the script
echo "export CLEV2ER_BASE_DIR=$PWD" >> $setup_and_run_file
echo "export PYTHONPATH=$CLEV2ER_BASE_DIR/src" >> $setup_and_run_file
echo "export PATH=${CLEV2ER_BASE_DIR}/src/clev2er/tools:\${PATH}" >> $setup_and_run_file

echo "export CT_PRODUCT_BASEDIR=/raid6/cryo-tempo/product_baselines" >> $setup_and_run_file
echo "export CT_LOG_DIR=/tmp" >> $setup_and_run_file
echo "export CPDATA_DIR=/cpdata" >> $setup_and_run_file
echo "export L1B_BASE_DIR=\${CPDATA_DIR}/SATS/RA/CRY/L1B" >> $setup_and_run_file
echo "export FES2014B_BASE_DIR=/raid6/cpdata/SATS/RA/CRY/L1B/FES2014" >> $setup_and_run_file
echo "export CATS2008A_BASE_DIR=/raid6/cpdata/SATS/RA/CRY/L1B/CATS2008/SIN" >> $setup_and_run_file
echo "export CS2_UNCERTAINTY_BASE_DIR=/raid6/cryo-tempo/land_ice/uncertainty" >> $setup_and_run_file

# Special handling for hostname "lec-cpom"
current_hostname=$(hostname)
if [[ "$current_hostname" == "lec-cpom" ]]; then
    echo "export CT_PRODUCT_BASEDIR=~/cryotempo/products" >> $setup_and_run_file
    echo "export CPDATA_DIR=/media/luna/archive" >> $setup_and_run_file
    echo "export L1B_BASE_DIR=\${CPDATA_DIR}/SATS/RA/CRY/L1B" >> $setup_and_run_file
    echo "export FES2014B_BASE_DIR=/media/luna/archive/SATS/RA/CRY/L1B/FES2014" >> $setup_and_run_file
    echo "export CATS2008A_BASE_DIR=/media/luna/archive/SATS/RA/CRY/L1B/CATS2008/SIN" >> $setup_and_run_file
    echo "export CS2_UNCERTAINTY_BASE_DIR=/media/luna/archive/RESOURCES/ct_uncertainty" >> $setup_and_run_file
    echo "export CPOM_SOFTWARE_DIR=/media/luna/shared/software/cpom_software" >> $setup_and_run_file
fi

# Set ulimit
echo "ulimit -n 8192" >> $setup_and_run_file

# Notify user the environment is ready
echo "echo \"CryoTEMPO Environment setup complete. You are now in the Poetry virtual environment for CryoTEMPO.\"" >> $setup_and_run_file
echo "bash" >> $setup_and_run_file  # Open a subshell to keep the environment active

# Ensure the output script is executable
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

pre-commit install
pre-commit autoupdate

echo "Installation complete. Use ./setup_and_run.sh to set up and activate the environment."
