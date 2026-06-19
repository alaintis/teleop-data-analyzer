#!/usr/bin/env bash
#
# install.sh — set up the Tele-op Data Analyzer environment.
#
# 1. Creates a Python virtual environment (.venv) with MuJoCo and the other
#    project dependencies (see requirements.txt).
# 2. Downloads the LeRobot dataset `alaintis/red_cube_cardbox_all_cleaned_01`
#    from the Hugging Face Hub.
#
# Usage:
#   ./install.sh                 # full setup into ./.venv and ./data
#   VENV_DIR=myenv ./install.sh  # override venv location
#   DATA_DIR=/path ./install.sh  # override dataset download location
#   SKIP_DATASET=1 ./install.sh  # only build the venv, skip the download
#
set -euo pipefail

# --- Configuration -----------------------------------------------------------
HF_REPO="alaintis/red_cube_cardbox_all_cleaned_01"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
PYTHON="${PYTHON:-python3}"

echo "==> Project:  $PROJECT_DIR"
echo "==> Venv:     $VENV_DIR"
echo "==> Dataset:  $HF_REPO -> $DATA_DIR/$(basename "$HF_REPO")"
echo

# --- 1. Python virtual environment ------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtual environment with $($PYTHON --version)"
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "==> Reusing existing virtual environment"
fi

# Use the venv's interpreter directly (no need to 'source activate').
VENV_PY="$VENV_DIR/bin/python"

echo "==> Upgrading pip"
"$VENV_PY" -m pip install --upgrade pip >/dev/null

echo "==> Installing dependencies from requirements.txt"
"$VENV_PY" -m pip install -r "$PROJECT_DIR/requirements.txt"

echo "==> Verifying MuJoCo import"
"$VENV_PY" - <<'PY'
import mujoco
print(f"    MuJoCo {mujoco.__version__} OK")
PY

# --- 2. Dataset download -----------------------------------------------------
if [ "${SKIP_DATASET:-0}" = "1" ]; then
    echo "==> SKIP_DATASET=1 set; skipping dataset download"
else
    echo "==> Downloading dataset from Hugging Face"
    mkdir -p "$DATA_DIR"
    HF_REPO="$HF_REPO" DATA_DIR="$DATA_DIR" "$VENV_PY" - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = os.environ["HF_REPO"]
out = os.path.join(os.environ["DATA_DIR"], repo.split("/")[-1])
path = snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    local_dir=out,
)
print(f"    Dataset downloaded to: {path}")
PY
fi

echo
echo "==> Done."
echo "    Activate the environment with:  source \"$VENV_DIR/bin/activate\""
