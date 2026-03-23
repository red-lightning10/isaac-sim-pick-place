#!/bin/bash
# Launch the complete pick-and-place pipeline with proper environment.
# Run from repo root (so101_ws): ./scripts/launch_pipeline.sh [debug:=true]
# Requires: bringup_moveit.launch.py (move_group + controllers) running first.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
source "$REPO_ROOT/env_pickplace/bin/activate"

# Ensure venv Python is first in PATH (for scripts with #!/usr/bin/python3 shebang)
export PATH="$REPO_ROOT/env_pickplace/bin:$PATH"

# Source ROS 2 workspace
source "$REPO_ROOT/install/setup.bash"

# Export for config path expansion ($REPO_ROOT in default.yaml, ggcnn_service.yaml)
export REPO_ROOT="$REPO_ROOT"

# Launch the pipeline
ros2 launch so101_state_machine pick_and_place.launch.py "$@"
