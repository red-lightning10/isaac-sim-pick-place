import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("object_detection"), "config", "default.yaml"
    )
    return LaunchDescription([
        Node(
            package="object_detection",
            executable="detection_service",
            name="detection_service_node",
            parameters=[config],
            output="screen",
        ),
    ])
