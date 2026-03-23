#!/usr/bin/env python3
"""
GGCNN Grasp Service Node - grasp pose computation from depth + bbox.

Subscribes to depth and camera_info; provides compute_grasp service.
Transforms grasp from camera frame to base_link; applies z_offset.
"""
import os
import traceback

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseArray
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from so101_interfaces.srv import ComputeGrasp
from tf2_ros import Buffer, TransformListener
import threading

from .util.inference import GraspComputer


class GGCNNServiceNode(Node):
    """ROS 2 node: depth/camera_info -> GGCNN inference -> grasp PoseStamped in base_link."""

    def __init__(self):
        super().__init__('ggcnn_service_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('depth_topic', '/front_camera/depth')
        self.declare_parameter('camera_info_topic', '/front_camera/camera_info')
        self.declare_parameter('camera_frame', 'Camera_Pseudo_Depth')
        self.declare_parameter('z_offset', 0.1)
        self.declare_parameter('grasp_debug', False)

        raw_path = self.get_parameter('model_path').value
        model_path = os.path.expandvars(os.path.expanduser(raw_path or ""))
        os.makedirs(GraspComputer.get_output_dir(), exist_ok=True)
        self._grasp_computer = None
        if model_path and os.path.exists(model_path):
            self._grasp_computer = GraspComputer(model_path)
            self.get_logger().info(f"Loaded GGCNN model from {model_path}")
        else:
            self.get_logger().error(f"Model path not found: {model_path}")

        self._bridge = CvBridge()
        self._depth_image = None
        self._camera_info = None
        self._lock = threading.Lock()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._cb_group = ReentrantCallbackGroup()

        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        self.create_subscription(Image, depth_topic, self._depth_cb, 1)
        self.create_subscription(CameraInfo, info_topic, self._info_cb, 1)

        self.create_service(
            ComputeGrasp,
            'compute_grasp',
            self._grasp_callback,
            callback_group=self._cb_group
        )

        self.get_logger().warn("Grasp Initiated.")

    def _depth_cb(self, msg: Image):
        with self._lock:
            self._depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _info_cb(self, msg: CameraInfo):
        with self._lock:
            if self._camera_info is None:
                debug_val = self.get_parameter('grasp_debug').value
                if debug_val if isinstance(debug_val, bool) else str(debug_val).lower() in ('true', '1', 'yes'):
                    self.get_logger().info("Camera intrinsics initialized")
            self._camera_info = msg

    def _grasp_callback(self, request, response):
        debug = getattr(request, 'debug', False)

        if debug:
            self.get_logger().info(f"Grasp request with bbox: {request.bbox}")

        with self._lock:
            missing = []
            if self._depth_image is None:
                missing.append("depth image")
            if self._camera_info is None:
                missing.append("camera_info")
            if missing:
                response.success = False
                response.message = f"Missing: {', '.join(missing)}"
                return response
            depth = self._depth_image.copy()
            camera_info = self._camera_info

        if len(request.bbox) < 4:
            response.success = False
            response.message = "Invalid bbox"
            return response

        if self._grasp_computer is None:
            response.success = False
            response.message = "GGCNN model not configured"
            return response

        bbox = request.bbox[:4]
        if debug:
            self.get_logger().info(f"Using bbox: {bbox}")

        try:
            result = self._grasp_computer.run(depth, bbox)
            if result is None:
                response.success = False
                response.message = "Grasp inference failed"
                return response

            grasp_x, grasp_y, angle, grasp_width = result

            if debug:
                path = GraspComputer.save_visualization(depth, grasp_x, grasp_y, angle, grasp_width)
                if path:
                    self.get_logger().info(f"Saved visualization to {path}")
                else:
                    self.get_logger().warn("Visualization failed")

            point_depth = depth[grasp_y, grasp_x]
            if point_depth == 0 or not np.isfinite(point_depth):
                response.success = False
                response.message = "Invalid depth at grasp point"
                return response

            fx, fy = camera_info.k[0], camera_info.k[4]
            cx, cy = camera_info.k[2], camera_info.k[5]
            # Unproject pixel to 3D in camera frame
            x = (grasp_x - cx) / fx * point_depth
            y = (grasp_y - cy) / fy * point_depth
            z = point_depth

            if debug:
                self.get_logger().info(f"Camera frame: pos=[{x:.3f}, {y:.3f}, {z:.3f}]")

            camera_frame = self.get_parameter('camera_frame').value
            target_frame = 'base_link'

            try:
                transform = self._tf_buffer.lookup_transform(
                    target_frame,
                    camera_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
                t = transform.transform.translation
                q = transform.transform.rotation
                translation = np.array([t.x, t.y, t.z])
                rotation = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                pos_camera = np.array([x, y, z])
                pos_world = rotation @ pos_camera + translation
                x, y, z = pos_world

                quat_base = np.array([0.0, np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2])  # gripper down
                rot_z = R.from_rotvec([0, 0, -angle])  # grasp angle from GGCNN
                rot = rot_z * R.from_quat(quat_base)
                quat = rot.as_quat()

                if debug:
                    self.get_logger().info(f"Transformed to {target_frame}: pos=[{x:.3f}, {y:.3f}, {z:.3f}]")

                z_offset = self.get_parameter('z_offset').value
                z += z_offset
                if debug:
                    self.get_logger().info(f"After z-offset (+{z_offset:.2f}m): pos=[{x:.3f}, {y:.3f}, {z:.3f}]")

            except Exception as e:
                response.success = False
                response.message = f"TF transform failed: {e}"
                return response

            response.success = True
            response.message = "Grasp computed successfully"
            response.grasp_poses = PoseArray()
            response.grasp_poses.header.frame_id = target_frame
            response.grasp_poses.header.stamp = self.get_clock().now().to_msg()

            pose = PoseStamped()
            pose.header = response.grasp_poses.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.x = float(quat[0])
            pose.pose.orientation.y = float(quat[1])
            pose.pose.orientation.z = float(quat[2])
            pose.pose.orientation.w = float(quat[3])
            response.grasp_poses.poses.append(pose.pose)

            response.widths = Float32MultiArray()
            response.widths.data = [float(grasp_width / 1000.0)]

            if debug:
                euler = R.from_quat(quat).as_euler('xyz', degrees=True)
                self.get_logger().info(
                    f"Grasp pose (base_link): pos=[{x:.3f}, {y:.3f}, {z:.3f}], "
                    f"ori=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}], "
                    f"euler=[{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]°"
                )

        except Exception as e:
            response.success = False
            response.message = f"GGCNN inference failed: {e}"
            self.get_logger().error(f"{response.message}\n{traceback.format_exc()}")

        return response


def main(args=None):
    rclpy.init(args=args)
    node = GGCNNServiceNode()

    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
