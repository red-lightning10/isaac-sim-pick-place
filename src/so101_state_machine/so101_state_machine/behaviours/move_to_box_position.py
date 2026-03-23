"""Move to box position behaviour: detect tray (gray-container) -> lift -> move to centroid -> lower.

Placement phase: gripper moves to tray centroid at (lift_z - lower_height).
"""

import rclpy
import py_trees
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from so101_interfaces.srv import DetectObject, MoveToPose

from .bt_helpers import build_detect_request, build_pose_stamped, parse_debug_param


class MoveToBoxPosition(py_trees.behaviour.Behaviour):
    """State machine: CALL_DETECTION -> MOVE_TO_LIFT_POSE -> MOVE_TO_CENTROID."""

    LIFT_POSE = {
        'position': (0.186, -0.000, 0.300),
        'orientation': (0.563, 0.590, -0.399, -0.420),
    }

    def __init__(self, name: str, node: Node):
        super().__init__(name)
        self.node = node
        self.callback_group = ReentrantCallbackGroup()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        self._detection_client = node.create_client(
            DetectObject, 'detect_object', callback_group=self.callback_group
        )
        self._moveit_client = node.create_client(
            MoveToPose, '/move_to_pose', callback_group=self.callback_group
        )
        self._state = 'CALL_DETECTION'
        self._service_future = None
        self._centroid = None
        self._lifted_z = None
        self._lower_height = 0.08  # m above tray centroid for drop
        self._wait_logged = {}

    def initialise(self):
        self._state = 'CALL_DETECTION'
        self._service_future = None
        self._centroid = None
        self._lifted_z = None
        self._wait_logged = {}
        if parse_debug_param(self.node, 'state_machine_debug'):
            self.node.get_logger().info(f"[{self.name}] Detect tray, move to lift pose, move to centroid, lower {self._lower_height}m...")

    def update(self) -> py_trees.common.Status:
        if self._state == 'CALL_DETECTION':
            if not self._detection_client.wait_for_service(timeout_sec=0.1):
                if not self._wait_logged.get('detection'):
                    self._wait_logged['detection'] = True
                    self.node.get_logger().warn(f"[{self.name}] Waiting for detection...")
                return py_trees.common.Status.RUNNING
            if self._service_future is None:
                req = build_detect_request(self.node, "gray-container", 0.40, 0.25)
                self._service_future = self._detection_client.call_async(req)
                if parse_debug_param(self.node, 'state_machine_debug'):
                    self.node.get_logger().info(f"[{self.name}] Detecting Box (FastSAM + pointcloud centroid)...")
                return py_trees.common.Status.RUNNING
            if self._service_future.done():
                try:
                    res = self._service_future.result()
                    if not res.success:
                        self.node.get_logger().error(f"[{self.name}] Detection failed: {res.message}")
                        return py_trees.common.Status.FAILURE
                    self._centroid = (res.centroid.x, res.centroid.y, res.centroid.z)
                    if parse_debug_param(self.node, 'state_machine_debug'):
                        self.node.get_logger().info(
                            f"[{self.name}] Tray centroid: [{res.centroid.x:.3f}, {res.centroid.y:.3f}, {res.centroid.z:.3f}]"
                        )
                    self._service_future = None
                    self._state = 'MOVE_TO_LIFT_POSE'
                except Exception as e:
                    self.node.get_logger().error(f"[{self.name}] Detection error: {e}")
                    return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING

        if self._state == 'MOVE_TO_LIFT_POSE':
            if not self._moveit_client.wait_for_service(timeout_sec=0.1):
                if not self._wait_logged.get('moveit'):
                    self._wait_logged['moveit'] = True
                    self.node.get_logger().warn(f"[{self.name}] Waiting for move_to_pose service...")
                return py_trees.common.Status.RUNNING
            if self._service_future is None:
                pos, ori = self.LIFT_POSE['position'], self.LIFT_POSE['orientation']
                self._lifted_z = pos[2]
                target = build_pose_stamped(self.node, pos, ori)
                req = MoveToPose.Request()
                req.target_pose = target
                req.use_cartesian_fraction = False
                self._service_future = self._moveit_client.call_async(req)
                if parse_debug_param(self.node, 'state_machine_debug'):
                    self.node.get_logger().info(
                        f"[{self.name}] Moving to lift pose: gripper_frame_link [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
                    )
            elif self._service_future.done():
                try:
                    res = self._service_future.result()
                    if not res.success:
                        self.node.get_logger().error(f"[{self.name}] Lift failed: {res.message}")
                        return py_trees.common.Status.FAILURE
                    self._service_future = None
                    self._state = 'MOVE_TO_CENTROID'
                except Exception as e:
                    self.node.get_logger().error(f"[{self.name}] Move error: {e}")
                    return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING

        if self._state == 'MOVE_TO_CENTROID':
            if self._service_future is None:
                cx, cy, cz = self._centroid
                try:
                    T = self._tf_buffer.lookup_transform(
                        'base_link', 'gripper_frame_link', rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.5)
                    )
                    drop_z = self._lifted_z - self._lower_height
                    target = build_pose_stamped(
                        self.node, (cx, cy, drop_z), T.transform.rotation
                    )
                    req = MoveToPose.Request()
                    req.target_pose = target
                    req.use_cartesian_fraction = False
                    self._service_future = self._moveit_client.call_async(req)
                    if parse_debug_param(self.node, 'state_machine_debug'):
                        self.node.get_logger().info(
                            f"[{self.name}] Moving gripper_frame_link to drop position: [{cx:.3f}, {cy:.3f}, {drop_z:.3f}]"
                        )
                except Exception as e:
                    self.node.get_logger().warn(f"[{self.name}] TF failed: {e}")
                    return py_trees.common.Status.RUNNING
            elif self._service_future.done():
                try:
                    res = self._service_future.result()
                    if res.success:
                        if parse_debug_param(self.node, 'state_machine_debug'):
                            self.node.get_logger().info(f"[{self.name}] Placement complete (ready for detach): {res.message}")
                        return py_trees.common.Status.SUCCESS
                    self.node.get_logger().error(f"[{self.name}] Move to drop position failed: {res.message}")
                    return py_trees.common.Status.FAILURE
                except Exception as e:
                    self.node.get_logger().error(f"[{self.name}] Service exception: {e}")
                    return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING

        return py_trees.common.Status.RUNNING
