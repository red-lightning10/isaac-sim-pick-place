#!/usr/bin/env python3
"""
Launch file for complete pick-and-place pipeline.

Starts: move_to_pose, move_gripper, spawn_cube, detection, GGCNN grasp, BT node.
Requires bringup_moveit.launch.py (move_group + controllers) to be running first.

Launch args: debug, grasp_debug, detection_debug, state_machine_debug.
When debug:=true, all debug params and INFO log level are enabled.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def load_yaml(package_name, file_path):
    """Load YAML config from package share directory."""
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    with open(absolute_file_path, 'r') as f:
        return yaml.safe_load(f)


def get_robot_description_semantic():
    """Load SRDF for MoveIt semantic model."""
    package_path = get_package_share_directory('so101_moveit_config')
    srdf_path = os.path.join(package_path, 'config', 'so101_new_calib.srdf')
    with open(srdf_path, 'r') as f:
        content = f.read()
    return {'robot_description_semantic': ParameterValue(content, value_type=str)}


def generate_launch_description():
    detection_pkg = get_package_share_directory('object_detection')
    grasp_pkg = get_package_share_directory('grasping')
    moveit_interface_pkg = get_package_share_directory('so101_moveit_interface')

    detection_config = os.path.join(detection_pkg, 'config', 'default.yaml')
    ggcnn_config = os.path.join(grasp_pkg, 'config', 'ggcnn_service.yaml')
    moveit_service_config = os.path.join(moveit_interface_pkg, 'config', 'move_to_pose.yaml')
    move_gripper_config = os.path.join(moveit_interface_pkg, 'config', 'move_gripper.yaml')
    spawn_cube_config = os.path.join(moveit_interface_pkg, 'config', 'spawn_cube.yaml')

    robot_description_semantic = get_robot_description_semantic()
    robot_description_kinematics = {
        'robot_description_kinematics': load_yaml('so101_moveit_config', 'config/kinematics.yaml')
    }

    log_level = PythonExpression([
        "'info' if '", LaunchConfiguration('debug'), "' == 'true' else 'warn'"
    ])
    node_args = ['--ros-args', '--log-level', log_level]
    # When debug:=true, enable all debug params
    grasp_debug = PythonExpression([
        "'true' if '", LaunchConfiguration('debug'), "' == 'true' else '", LaunchConfiguration('grasp_debug'), "'"
    ])
    detection_debug = PythonExpression([
        "'true' if '", LaunchConfiguration('debug'), "' == 'true' else '", LaunchConfiguration('detection_debug'), "'"
    ])
    state_machine_debug = PythonExpression([
        "'true' if '", LaunchConfiguration('debug'), "' == 'true' else '", LaunchConfiguration('state_machine_debug'), "'"
    ])

    return LaunchDescription([
        DeclareLaunchArgument('debug', default_value='false', description='Enable verbose output (INFO log level, YOLO/Python warnings)'),
        DeclareLaunchArgument('grasp_debug', default_value='false', description='Enable GGCNN debug (verbose logs, save visualization)'),
        DeclareLaunchArgument('detection_debug', default_value='false', description='Enable detection debug (verbose logs, save visualization)'),
        DeclareLaunchArgument('state_machine_debug', default_value='false', description='Enable BT debug (verbose logs)'),
        SetEnvironmentVariable('YOLO_VERBOSE', 'false', condition=UnlessCondition(LaunchConfiguration('debug'))),
        SetEnvironmentVariable('PYTHONWARNINGS', 'ignore', condition=UnlessCondition(LaunchConfiguration('debug'))),
        
        # MoveIt C++ service (loads SRDF + kinematics directly, connects to running move_group)
        Node(
            package='so101_moveit_interface',
            executable='move_to_pose_server',
            name='move_to_pose_server',
            parameters=[
                moveit_service_config,
                robot_description_semantic,
                robot_description_kinematics,
                {'debug': LaunchConfiguration('debug')},
            ],
            arguments=node_args,
            output='screen'
        ),

        # MoveIt gripper service
        Node(
            package='so101_moveit_interface',
            executable='move_gripper_server',
            name='move_gripper_server',
            parameters=[
                move_gripper_config,
                robot_description_semantic,
                robot_description_kinematics,
                {'debug': LaunchConfiguration('debug')},
            ],
            arguments=node_args,
            output='screen'
        ),

        # Spawn/delete cubes in planning scene
        Node(
            package='so101_moveit_interface',
            executable='spawn_cube_server',
            name='spawn_cube_server',
            parameters=[spawn_cube_config, {'debug': LaunchConfiguration('debug')}],
            arguments=node_args,
            output='screen'
        ),

        # Detection service
        Node(
            package='object_detection',
            executable='detection_service',
            name='detection_service_node',
            parameters=[detection_config, {'detection_debug': detection_debug}],
            arguments=node_args,
            output='screen'
        ),

        # GGCNN grasp service
        Node(
            package='grasping',
            executable='ggcnn_service',
            name='ggcnn_service_node',
            parameters=[ggcnn_config, {'grasp_debug': grasp_debug}],
            arguments=node_args,
            output='screen'
        ),

        # Behavior tree
        Node(
            package='so101_state_machine',
            executable='bt_node',
            name='bt_node',
            parameters=[
                {'grasp_debug': grasp_debug},
                {'detection_debug': detection_debug},
                {'state_machine_debug': state_machine_debug},
            ],
            arguments=node_args,
            output='screen'
        ),
    ])
