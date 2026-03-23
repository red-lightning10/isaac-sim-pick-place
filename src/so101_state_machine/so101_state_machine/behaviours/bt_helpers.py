"""Shared helpers for BT behaviours: debug params, service request builders, pose utils."""

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from so101_interfaces.srv import ComputeGrasp, DetectObject


def parse_debug_param(node: Node, param_name: str) -> bool:
    """Return True if param is truthy (bool True, 'true', '1', 'yes')."""
    val = node.get_parameter(param_name).value
    return val if isinstance(val, bool) else str(val).lower() in ('true', '1', 'yes')


def build_detect_request(
    node: Node,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> DetectObject.Request:
    """Build DetectObject request with detection_debug from node params."""
    req = DetectObject.Request()
    req.text_prompt = text_prompt
    req.box_threshold = box_threshold
    req.text_threshold = text_threshold
    req.debug = parse_debug_param(node, 'detection_debug')
    return req


def build_grasp_request(node: Node, pointcloud, bbox) -> ComputeGrasp.Request:
    """Build ComputeGrasp request with grasp_debug from node params."""
    req = ComputeGrasp.Request()
    req.pointcloud = pointcloud
    req.bbox = bbox
    req.debug = parse_debug_param(node, 'grasp_debug')
    return req


def build_pose_stamped(
    node: Node,
    position: tuple,
    orientation,
    frame_id: str = 'base_link',
) -> PoseStamped:
    """Build PoseStamped. orientation: tuple (x,y,z,w) or geometry_msgs Quaternion."""
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = node.get_clock().now().to_msg()
    pose.pose.position.x = float(position[0])
    pose.pose.position.y = float(position[1])
    pose.pose.position.z = float(position[2])
    if hasattr(orientation, 'x'):
        pose.pose.orientation = orientation
    else:
        pose.pose.orientation.x = float(orientation[0])
        pose.pose.orientation.y = float(orientation[1])
        pose.pose.orientation.z = float(orientation[2])
        pose.pose.orientation.w = float(orientation[3])
    return pose
