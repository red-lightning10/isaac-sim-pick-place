"""Grabbing behaviour: detect object (red mug) -> compute grasp (GGCNN) -> move to pre-grasp pose.

Uses cartesian_fraction to stop short of target for safer grasping. Logs gripper
transform after move when state_machine_debug is enabled.
"""

import py_trees
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from so101_interfaces.srv import ComputeGrasp, DetectObject, MoveToPose, MoveGripper

from .bt_helpers import build_detect_request, build_grasp_request, parse_debug_param


class Grabbing(py_trees.behaviour.Behaviour):
    """State machine: CALL_DETECTION -> CALL_GRASP -> MOVE_TO_PREGRASP -> DONE."""

    def __init__(self, name: str, node: Node):
        super().__init__(name)
        self.node = node
        self.callback_group = ReentrantCallbackGroup()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        self._detection_client = self.node.create_client(
            DetectObject, 'detect_object', callback_group=self.callback_group
        )
        self._grasp_client = self.node.create_client(
            ComputeGrasp, 'compute_grasp', callback_group=self.callback_group
        )
        self._moveit_client = self.node.create_client(
            MoveToPose, '/move_to_pose', callback_group=self.callback_group
        )
        self._gripper_client = self.node.create_client(
            MoveGripper, '/move_gripper', callback_group=self.callback_group
        )
        self._grasp_pose = None
        self._pointcloud = None
        self._bbox = None
        self._state = 'CALL_DETECTION'
        self._service_future = None
        self._wait_ticks = 0
        self._wait_logged = {}

    def initialise(self):
        self._state = 'CALL_DETECTION'
        self._service_future = None
        self._grasp_pose = None
        self._pointcloud = None
        self._bbox = None
        self._wait_ticks = 0
        self._wait_logged = {}
        if parse_debug_param(self.node, 'state_machine_debug'):
            self.node.get_logger().info(f"[{self.name}] Starting pick sequence (detection -> grasp -> move to pose)...")

    def update(self) -> py_trees.common.Status:
        if self._state == 'CALL_DETECTION':
            if not self._detection_client.wait_for_service(timeout_sec=0.1):
                if not self._wait_logged.get('detection'):
                    self._wait_logged['detection'] = True
                    self.node.get_logger().warn(f"[{self.name}] Waiting for detection service...")
                return py_trees.common.Status.RUNNING

            if self._service_future is None:
                request = build_detect_request(self.node, "red mug", 0.35, 0.25)
                self._service_future = self._detection_client.call_async(request)
                self.node.get_logger().info(f"[{self.name}] Calling detection service...")
                return py_trees.common.Status.RUNNING

            self._wait_ticks += 1
            if self._wait_ticks % 20 == 0 and self._wait_ticks > 0:
                self.node.get_logger().info(f"[{self.name}] Still waiting for detection... (GroundingDINO can take 5-10s)")
            if self._service_future.done():
                try:
                    result = self._service_future.result()
                    if not result.success:
                        self.node.get_logger().error(f"[{self.name}] Detection failed: {result.message}")
                        return py_trees.common.Status.FAILURE
                    self._pointcloud = result.pointcloud
                    self._bbox = result.bbox
                    if parse_debug_param(self.node, 'state_machine_debug'):
                        self.node.get_logger().info(
                            f"[{self.name}] Detection successful: {result.message}, "
                            f"pointcloud: {self._pointcloud.width} points"
                        )
                    self._state = 'CALL_GRASP'
                    self._service_future = None
                    self._wait_ticks = 0
                    return py_trees.common.Status.RUNNING
                except Exception as e:
                    self.node.get_logger().error(f"[{self.name}] Detection result error: {e}")
                    return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING

        elif self._state == 'CALL_GRASP':
            if not self._grasp_client.wait_for_service(timeout_sec=0.1):
                if not self._wait_logged.get('grasp'):
                    self._wait_logged['grasp'] = True
                    self.node.get_logger().warn(f"[{self.name}] Waiting for grasp service...")
                return py_trees.common.Status.RUNNING

            if self._service_future is None:
                request = build_grasp_request(self.node, self._pointcloud, self._bbox)
                self._service_future = self._grasp_client.call_async(request)
                if parse_debug_param(self.node, 'state_machine_debug'):
                    self.node.get_logger().info(f"[{self.name}] Calling grasp service...")
                return py_trees.common.Status.RUNNING

            self._wait_ticks += 1
            if parse_debug_param(self.node, 'state_machine_debug') and self._wait_ticks % 20 == 0 and self._wait_ticks > 0:
                self.node.get_logger().info(f"[{self.name}] Still waiting for grasp (GGCNN inference)...")
            if self._service_future.done():
                try:
                    result = self._service_future.result()
                    if not result.success:
                        self.node.get_logger().error(f"[{self.name}] Grasp computation failed: {result.message}")
                        return py_trees.common.Status.FAILURE
                    if not result.grasp_poses.poses:
                        self.node.get_logger().error(f"[{self.name}] Grasp result empty")
                        return py_trees.common.Status.FAILURE
                    self._grasp_pose = result.grasp_poses.poses[0]
                    if parse_debug_param(self.node, 'state_machine_debug'):
                        self.node.get_logger().info(
                            f"[{self.name}] Received grasp in frame '{result.grasp_poses.header.frame_id}' "
                            f"at position [{self._grasp_pose.position.x:.3f}, "
                            f"{self._grasp_pose.position.y:.3f}, {self._grasp_pose.position.z:.3f}]"
                        )
                    self._state = 'MOVE_TO_PREGRASP'
                    self._service_future = None
                    self._wait_ticks = 0
                    return py_trees.common.Status.RUNNING
                except Exception as e:
                    self.node.get_logger().error(f"[{self.name}] Grasp result error: {e}")
                    return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING

        elif self._state == 'MOVE_TO_PREGRASP':
            if self._service_future is None:
                grasp = PoseStamped()
                grasp.header.frame_id = 'base_link'
                grasp.header.stamp = self.node.get_clock().now().to_msg()
                grasp.pose = self._grasp_pose
                return self._send_moveit_request(grasp, 'WAIT_FOR_MOVE', use_cartesian_fraction=True)
            return self._check_moveit_response('DONE')

        elif self._state == 'WAIT_FOR_MOVE':
            return self._check_moveit_response('DONE')

        elif self._state == 'DONE':
            self.node.get_logger().info(f"[{self.name}] Pick complete!")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

    def _send_moveit_request(self, target_pose: PoseStamped, next_state: str, use_cartesian_fraction: bool = False):
        """Send MoveToPose request and transition to next_state."""
        if not self._moveit_client.wait_for_service(timeout_sec=0.1):
            if not self._wait_logged.get('moveit'):
                self._wait_logged['moveit'] = True
                self.node.get_logger().warn(f"[{self.name}] Waiting for MoveToPose service...")
            return py_trees.common.Status.RUNNING
        if parse_debug_param(self.node, 'state_machine_debug'):
            self.node.get_logger().info(
                f"[{self.name}] MoveIt goal: frame='{target_pose.header.frame_id}', "
                f"pos=[{target_pose.pose.position.x:.3f}, {target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f}]"
            )
        request = MoveToPose.Request()
        request.target_pose = target_pose
        request.use_cartesian_fraction = use_cartesian_fraction
        self._service_future = self._moveit_client.call_async(request)
        if parse_debug_param(self.node, 'state_machine_debug'):
            self.node.get_logger().info(f"[{self.name}] Moving to target...")
        self._state = next_state
        return py_trees.common.Status.RUNNING

    def _check_moveit_response(self, next_state: str):
        """Poll move result; on success, optionally log gripper transform when next_state is DONE."""
        if self._service_future is not None and self._service_future.done():
            try:
                result = self._service_future.result()
                if result.success:
                    if parse_debug_param(self.node, 'state_machine_debug'):
                        self.node.get_logger().info(f"[{self.name}] Movement succeeded: {result.message}")
                        if next_state == 'DONE':
                            try:
                                T = self._tf_buffer.lookup_transform(
                                    'base_link', 'gripper_frame_link', rclpy.time.Time(),
                                    timeout=rclpy.duration.Duration(seconds=0.5)
                                )
                                t = T.transform.translation
                                q = T.transform.rotation
                                self.node.get_logger().info(
                                    f"[{self.name}] gripper_frame_link w.r.t. base_link: "
                                    f"pos=[{t.x:.3f}, {t.y:.3f}, {t.z:.3f}], "
                                    f"ori=[{q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f}]"
                                )
                            except Exception as e:
                                self.node.get_logger().warn(f"[{self.name}] TF lookup failed: {e}")
                    self._service_future = None
                    self._state = next_state
                    return py_trees.common.Status.RUNNING
                self.node.get_logger().error(f"[{self.name}] Movement failed: {result.message}")
                return py_trees.common.Status.FAILURE
            except Exception as e:
                self.node.get_logger().error(f"[{self.name}] Service exception: {e}")
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING
