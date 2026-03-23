#!/bin/bash
# Unified environment setup for Pick-and-Place pipeline.
# Run from repo root (so101_ws): ./scripts/setup_env.sh
#
# Prerequisites: uv (pip install uv), Python 3.10, ROS 2 Humble
# Optional: CUDA for GPU acceleration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${REPO_ROOT}/env_pickplace"
DEPS_DIR="${REPO_ROOT}/src/object_detection/deps"

echo "=== Pick-and-Place Environment Setup ==="
echo "Repo root: $REPO_ROOT"
echo "Venv: $VENV_DIR"
echo ""

# Check for uv
if ! command -v uv &>/dev/null; then
    echo "uv not found. Install with: pip install uv"
    exit 1
fi

# Remove existing venv if present (fresh install)
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing env_pickplace..."
    rm -rf "$VENV_DIR"
fi

# Create venv with Python 3.10
echo "Creating virtual environment (Python 3.10)..."
uv venv "$VENV_DIR" --python 3.10
source "$VENV_DIR/bin/activate"

# Upgrade pip, setuptools, wheel
uv pip install --python "$VENV_DIR/bin/python" pip setuptools wheel

# Install requirements.txt
echo ""
echo "Installing base requirements..."
uv pip install --python "$VENV_DIR/bin/python" -r "$REPO_ROOT/requirements.txt"

# Clone and install GroundingDINO
echo ""
echo "=== GroundingDINO ==="
if [ ! -d "$DEPS_DIR/GroundingDINO" ]; then
    mkdir -p "$DEPS_DIR"
    git clone https://github.com/IDEA-Research/GroundingDINO.git "$DEPS_DIR/GroundingDINO"
fi
uv pip install --python "$VENV_DIR/bin/python" --no-build-isolation -e "$DEPS_DIR/GroundingDINO"

# CLIP (GroundingDINO dep) needs setuptools < 70 for pkg_resources
echo ""
echo "Installing CLIP (setuptools pin)..."
uv pip install --python "$VENV_DIR/bin/python" "setuptools>=58,<70"
uv pip install --python "$VENV_DIR/bin/python" --no-build-isolation git+https://github.com/openai/CLIP.git

# FastSAM (also needs CLIP; keep setuptools<70 for pkg_resources at build and runtime)
echo ""
echo "=== FastSAM ==="
if [ ! -d "$DEPS_DIR/FastSAM" ]; then
    mkdir -p "$DEPS_DIR"
    git clone https://github.com/CASIA-IVA-Lab/FastSAM.git "$DEPS_DIR/FastSAM"
fi
uv pip install --python "$VENV_DIR/bin/python" --no-build-isolation -e "$DEPS_DIR/FastSAM"

# Freeze lock file
echo ""
echo "Freezing requirements to requirements.lock..."
"$VENV_DIR/bin/pip" freeze > "$REPO_ROOT/requirements.lock"

echo ""
echo "=== Setup complete ==="
echo ""
echo "If you hit build errors (e.g. CUDA mismatch), refer to the upstream repos:"
echo "  GroundingDINO: https://github.com/IDEA-Research/GroundingDINO"
echo "  FastSAM: https://github.com/CASIA-IVA-Lab/FastSAM"
echo ""
echo "Next steps:"
echo "  1. Download weights to weights/:"
echo "     - GroundingDINO: groundingdino_swint_ogc.pth, GroundingDINO_SwinT_OGC.py"
echo "     - FastSAM: FastSAM-x.pt"
echo "     - GGCNN: ggcnn_epoch_23_cornell_statedict.pt (in ggcnn_weights_cornell/)"
echo ""
echo "  2. Build: ./scripts/build_with_venv.sh"
echo ""
echo "  3. Run: ./scripts/launch_pipeline.sh"
echo ""
