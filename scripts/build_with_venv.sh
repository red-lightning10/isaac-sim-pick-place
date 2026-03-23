#!/bin/bash
# Build ROS 2 workspace with venv activated
# Run from repo root (so101_ws): ./scripts/build_with_venv.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
source "$REPO_ROOT/env_pickplace/bin/activate"

# Build workspace
cd "$REPO_ROOT"
colcon build --packages-select so101_interfaces so101_description so101_moveit_config so101_bringup so101_moveit_interface object_detection grasping so101_state_machine --symlink-install

# Fix shebangs to use env instead of absolute path
echo ""
echo "Fixing shebangs to use venv..."
find install -type f \( -name "bt_node" -o -name "ggcnn_service" -o -name "detection_service" \) -exec sed -i '1s|#!/usr/bin/python3|#!/usr/bin/env python3|' {} \;

echo ""
echo "Build complete! The installed scripts will use:"
which python3
echo ""
echo "Shebangs fixed for venv compatibility."
