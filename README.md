# Pick & Place — Isaac Sim

## Overview

Complete pick-and-place pipeline for the SO101 robot in Isaac Sim:

- **object_detection** — GroundingDINO + FastSAM for text-prompted object segmentation and 3D point clouds
- **grasping** — GGCNN for 6-DOF grasp pose computation from depth + bbox
- **so101_state_machine** — py_trees behavior tree (detect → grasp → attach → move to tray → detach)
- **so101_moveit_interface** — MoveIt services for arm and gripper

---

## Setup & Build

This workspace is **self-contained**. Run everything from the `so101_ws` directory:

1. **Setup env:** `./scripts/setup_env.sh` — Creates venv (`env_pickplace/`), clones GroundingDINO/FastSAM into `src/object_detection/deps/`, installs deps
2. **Download weights** to `weights/` in this directory
3. **Build:** `./scripts/build_with_venv.sh` — Colcon build with shebang fix
4. **Run:** `./scripts/launch_pipeline.sh` — Launches pick-and-place

### Quick start

```bash
cd so101_ws
./scripts/setup_env.sh
# Download weights to weights/
./scripts/build_with_venv.sh
```

---

## Running the Pipeline

### Prerequisites

1. **Isaac Sim** — Start Isaac Sim and load your environment (scene with SO101 robot, cameras, etc.).

2. **Bringup MoveIt** — In a separate terminal:
   ```bash
   cd so101_ws
   source /opt/ros/humble/setup.bash
   source install/setup.bash
   ros2 launch so101_bringup bringup_moveit.launch.py use_fake_hardware:=true use_sim_time:=true
   ```

3. **Pipeline** — From `so101_ws`:
   ```bash
   ./scripts/launch_pipeline.sh
   ```

---

## Packages

| Package | Description |
|---------|-------------|
| so101_interfaces | Service definitions (DetectObject, ComputeGrasp, MoveToPose, etc.) |
| so101_description | URDF/xacro |
| so101_moveit_config | MoveIt config |
| so101_bringup | bringup_moveit.launch.py |
| so101_moveit_interface | move_to_pose, move_gripper, spawn_cube servers |
| object_detection | GroundingDINO + FastSAM + pointcloud |
| grasping | GGCNN grasp service |
| so101_state_machine | Behavior tree (bt_node), pick_and_place.launch.py |

## Package structure

```
src/
├── so101_interfaces/
├── so101_description/
├── so101_moveit_config/
├── so101_bringup/
├── so101_moveit_interface/
├── object_detection/
├── grasping/
└── so101_state_machine/
```
