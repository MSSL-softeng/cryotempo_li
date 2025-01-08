#!/usr/bin/env bash

# Check if the script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced, not executed!"
    echo "Please run: source ./ct_activate.sh"
    exit 1
fi
# Combined setup and run script for CryoTEMPO LI
set -e

# Activate Poetry virtual environment
VENV_PATH=$(poetry env info --path)
if [ -z "$VENV_PATH" ]; then
    echo "Poetry virtual environment not found. Did you set it up?"
    exit 1
fi
source "$VENV_PATH/bin/activate"

export CLEV2ER_BASE_DIR=/media/luna/boxallk/cryotempo_li
export PYTHONPATH=/media/luna/boxallk/cryotempo_li/src
export PATH=/media/luna/boxallk/cryotempo_li/src/clev2er/tools:${PATH}
export CT_PRODUCT_BASEDIR=/raid6/cryo-tempo/product_baselines
export CT_LOG_DIR=/media/luna/boxallk/cryoTEMPO/products/logs
export CPDATA_DIR=/cpdata
export L1B_BASE_DIR=${CPDATA_DIR}/SATS/RA/CRY/L1B
export FES2014B_BASE_DIR=/cpdata/SATS/RA/CRY/L1B/FES2014
export CATS2008A_BASE_DIR=/cpdata/SATS/RA/CRY/L1B/CATS2008/SIN
export CS2_UNCERTAINTY_BASE_DIR=/raid6/cryo-tempo/land_ice/uncertainty
export CT_PRODUCT_BASEDIR=/media/luna/boxallk/cryoTEMPO/products
export CPDATA_DIR=/media/luna/archive
export L1B_BASE_DIR=${CPDATA_DIR}/SATS/RA/CRY/L1B
export FES2014B_BASE_DIR=/media/luna/archive/SATS/RA/CRY/L1B/FES2014
export CATS2008A_BASE_DIR=/media/luna/archive/SATS/RA/CRY/L1B/CATS2008/SIN
export CS2_UNCERTAINTY_BASE_DIR=/media/luna/archive/RESOURCES/ct_uncertainty
export CPOM_SOFTWARE_DIR=/media/luna/shared/software/cpom_software

# Check if specified paths exist
declare -A path_env_map=(
    [$CT_PRODUCT_BASEDIR]=CT_PRODUCT_BASEDIR
    [$CT_LOG_DIR]=CT_LOG_DIR
    [$CPDATA_DIR]=CPDATA_DIR
    [$L1B_BASE_DIR]=L1B_BASE_DIR
    [$FES2014B_BASE_DIR]=FES2014B_BASE_DIR
    [$CATS2008A_BASE_DIR]=CATS2008A_BASE_DIR
    [$CS2_UNCERTAINTY_BASE_DIR]=CS2_UNCERTAINTY_BASE_DIR
)

missing_paths=()
for path in "${!path_env_map[@]}"; do
    if [ ! -d "$path" ]; then
        missing_paths+=("${path_env_map[$path]}: $path")
    fi
done

if [ ${#missing_paths[@]} -gt 0 ]; then
    echo "WARNING: The following environment variables have paths that do not exist:" >&2
    for missing_path in "${missing_paths[@]}"; do
        echo "  - $missing_path" >&2
    done
fi

ulimit -n 8192

echo "Environment setup complete. You are now in the CryoTEMPO Land Ice Poetry virtual environment."
