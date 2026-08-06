#!/usr/bin/env bash
# ----------------------------------------------------------------
# CryoTEMPO Land Ice user activation script
#     source this to setup the environment prior to use of the chain
#     source ./activate.sh
#
# Portable across Linux (bash 4/5) and macOS (bash 3.2 or zsh).
# Avoids bash-4-only features (associative arrays, ${VAR,,} etc).
#
# Data directories default to the CPOM production server layout (with
# per-host overrides in section 3) but any of them may be overridden by
# exporting the variable before sourcing, eg:
#     export CPDATA_DIR=/Volumes/cpdata ; source ./activate.sh
# ----------------------------------------------------------------

# --- 1. SOURCED VS EXECUTED CHECK, AND LOCATE THIS SCRIPT --------
# Works under both bash and zsh. The zsh-only syntax is hidden inside
# eval so that bash never has to parse it.
_ct_script=""
if [ -n "${ZSH_VERSION:-}" ]; then
    eval '_ct_script="${(%):-%x}"'
    case "${ZSH_EVAL_CONTEXT:-}" in
        *:file*) : ;;
        *) echo "ERROR: This script must be sourced, not executed!"
           echo "Please run: source ./activate.sh"
           exit 1 ;;
    esac
elif [ -n "${BASH_VERSION:-}" ]; then
    _ct_script="${BASH_SOURCE[0]}"
    if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
        echo "ERROR: This script must be sourced, not executed!"
        echo "Please run: source ./activate.sh"
        exit 1
    fi
else
    echo "ERROR: unsupported shell - please use bash or zsh" >&2
    return 1 2>/dev/null || exit 1
fi

# Resolve the repository root from the script location rather than
# hardwiring it, so the same script works on every machine/checkout.
CLEV2ER_BASE_DIR="$(cd "$(dirname "$_ct_script")" && pwd)"
unset _ct_script
export CLEV2ER_BASE_DIR

# --- 2. ACTIVATE THE VIRTUAL ENVIRONMENT -------------------------
# Prefer an in-project .venv, otherwise ask poetry where it lives.
if [ -f "$CLEV2ER_BASE_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$CLEV2ER_BASE_DIR/.venv/bin/activate"
else
    VENV_PATH=$(cd "$CLEV2ER_BASE_DIR" && poetry env info --path 2>/dev/null)
    if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
        # shellcheck disable=SC1091
        . "$VENV_PATH/bin/activate"
    else
        echo "ERROR: no virtual environment found (.venv or via poetry)." >&2
        echo "       Did you run the install? Try: poetry install" >&2
        return 1
    fi
fi

# --- 3. ENVIRONMENT VARIABLES ------------------------------------
export PYTHONPATH="$CLEV2ER_BASE_DIR/src"
export PATH="$CLEV2ER_BASE_DIR/src/clev2er/tools:${PATH}"

# Data locations. Precedence, highest first:
#   1. values already exported in your environment before sourcing
#   2. the per-host block below, for machines with a non-standard layout
#   3. the generic CPOM production server defaults
# Host-specific settings only need to name the paths that differ; the
# rest are derived from CPDATA_DIR by the generic defaults.
#
# Note on host detection: on macOS `hostname` returns UNKNOWN whenever the
# network has not settled (eg straight after a reboot, or off the network),
# which silently drops the per-host settings and falls back to the production
# Linux paths. scutil's LocalHostName is stable, so prefer it where available.
_ct_host=$(scutil --get LocalHostName 2>/dev/null || true)
if [ -z "$_ct_host" ] || [ "$_ct_host" = "UNKNOWN" ]; then
    _ct_host=$(hostname -s 2>/dev/null || hostname)
fi

unset CS2_UNCERTAINTY_BASE_DIR

case "$_ct_host" in
    lec-cpom)
        export CT_PRODUCT_BASEDIR="${CT_PRODUCT_BASEDIR:-$HOME/cryotempo/products}"
        export CPDATA_DIR="${CPDATA_DIR:-/media/luna/archive}"
        export CS2_UNCERTAINTY_BASE_DIR="${CS2_UNCERTAINTY_BASE_DIR:-/media/luna/archive/RESOURCES/ct_uncertainty}"
        export CPOM_SOFTWARE_DIR="${CPOM_SOFTWARE_DIR:-/media/luna/shared/software/cpom_software}"
        ;;
    Macbook-Pro*)
        # macOS: slope/roughness data live under the local clevops checkout
        export TESTDATA_EXTERNAL_DIR="${TESTDATA_EXTERNAL_DIR:-$HOME/software/clevops/testdata_external}"
        # macOS: the only local copy of the pyTMD tide models (CATS2008_v2023)
        # is the one held by the CLEV2ER LIIW project
        export PYTMD_TIDE_MODELS_DIR="${PYTMD_TIDE_MODELS_DIR:-\
$HOME/software/clev2er_liiw/testdata_external/adf/landice_swath/pytmd_tide_models}"
        ;;
esac
unset _ct_host

export CT_PRODUCT_BASEDIR="${CT_PRODUCT_BASEDIR:-/raid6/cryo-tempo/product_baselines}"
export CT_LOG_DIR="${CT_LOG_DIR:-/tmp}"
export CPDATA_DIR="${CPDATA_DIR:-/cpdata}"
export L1B_BASE_DIR="${L1B_BASE_DIR:-${CPDATA_DIR}/SATS/RA/CRY/L1B}"
export FES2014B_BASE_DIR="${FES2014B_BASE_DIR:-${CPDATA_DIR}/SATS/RA/CRY/L1B/FES2014}"
export CATS2008A_BASE_DIR="${CATS2008A_BASE_DIR:-${CPDATA_DIR}/SATS/RA/CRY/L1B/CATS2008/SIN}"
export CS2_UNCERTAINTY_BASE_DIR="${CS2_UNCERTAINTY_BASE_DIR:-/raid6/cryo-tempo/land_ice/uncertainty}"
export TESTDATA_EXTERNAL_DIR="${TESTDATA_EXTERNAL_DIR:-/home/clopr/software/clevops/testdata_external}"
# pyTMD tide model directory, holding CATS2008_v2023/ and fes2022b/ (baseline-F onwards)
export PYTMD_TIDE_MODELS_DIR="${PYTMD_TIDE_MODELS_DIR:-${CPDATA_DIR}/MODELS/tides}"
# pre-computed FES2022 tide files (baseline-F onwards), containing LRM,SIN/<YYYY>/<MM>/
export FES2022_BASE_DIR="${FES2022_BASE_DIR:-${CPDATA_DIR}/SATS/RA/CRY/L1B/FES2022b}"

echo "env vars setup"

# --- 4. CHECK THE DATA PATHS EXIST -------------------------------
# Plain loop over variable *names* with indirect expansion via eval,
# so no bash-4 associative array is needed. Variables that are unset
# (eg CPOM_SOFTWARE_DIR off lec-cpom) are skipped rather than reported.
missing_paths=""
for _var in CT_PRODUCT_BASEDIR CT_LOG_DIR CPDATA_DIR L1B_BASE_DIR \
            FES2014B_BASE_DIR CATS2008A_BASE_DIR CS2_UNCERTAINTY_BASE_DIR \
            TESTDATA_EXTERNAL_DIR PYTMD_TIDE_MODELS_DIR FES2022_BASE_DIR \
            CPOM_SOFTWARE_DIR; do
    eval "_path=\$$_var"
    if [ -n "$_path" ] && [ ! -d "$_path" ]; then
        missing_paths="${missing_paths}  - ${_var}: ${_path}
"
    fi
done
unset _var _path

if [ -n "$missing_paths" ]; then
    echo "WARNING: The following environment variables have paths that do not exist:" >&2
    printf '%s' "$missing_paths" >&2
fi
unset missing_paths

# --- 5. FILE DESCRIPTOR LIMIT ------------------------------------
# multi-processing/shared mem support needs plenty of file descriptors.
# Only raise the limit, never lower an already generous one.
_ct_nofile=$(ulimit -n)
if [ "$_ct_nofile" != "unlimited" ] && [ "$_ct_nofile" -lt 8192 ] 2>/dev/null; then
    ulimit -n 8192 2>/dev/null || echo "WARNING: could not raise open file limit to 8192" >&2
fi
unset _ct_nofile

echo "CLEV2ER_BASE_DIR=$CLEV2ER_BASE_DIR"
echo "Environment setup complete. You are now in the CryoTEMPO Land Ice Poetry virtual environment."
