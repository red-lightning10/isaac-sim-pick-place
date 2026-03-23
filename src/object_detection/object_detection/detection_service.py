#!/usr/bin/env python3
"""
Detection Service Node - object detection and segmented pointcloud via service.

Uses GroundingDINO (text prompt) + FastSAM (segmentation). Subscribes to RGB,
depth, camera_info; provides detect_object service returning bbox and pointcloud.
"""
import os
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from so101_interfaces.srv import DetectObject
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
import tf2_py as tf2
import threading

import numpy as np
from scipy.spatial.transform import Rotation as R

import cv2
from .util.image_utils import ros_encoding_to_rgb
from .util.inference import ObjectDetector
from .util.pointcloud_utils import depth_mask_to_points_cam, create_pointcloud2_msg, centroid_from_points

from rclpy.executors import MultiThreadedExecutor

class DetectionServiceNode(Node):
    """ROS 2 node: RGB/depth -> GroundingDINO+FastSAM -> bbox + pointcloud in target_frame."""

    def __init__(self):
        super().__init__('detection_service_node')
        
        # Parameters
        self.declare_parameter('color_image_topic', '/front_camera/rgb')
        self.declare_parameter('depth_image_topic', '/front_camera/depth')
        self.declare_parameter('camera_info_topic', '/front_camera/camera_info')
        self.declare_parameter('grounding_dino_config', '')
        self.declare_parameter('grounding_dino_weights', '')
        self.declare_parameter('fastsam_weights', '')
        self.declare_parameter('box_threshold', 0.35)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('input_encoding', '')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('height_filter_z_min', 0.02)
        self.declare_parameter('detection_debug', False)

        self._bridge = CvBridge()
        self._latest_rgb = None
        self._latest_depth = None
        self._camera_info = None
        self._lock = threading.Lock()

        # TF2 for coordinate transformations
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Callback group for service
        self._cb_group = ReentrantCallbackGroup()
        
        # Subscribe to RGB, depth, camera_info
        color_topic = self.get_parameter('color_image_topic').value
        depth_topic = self.get_parameter('depth_image_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        
        self.create_subscription(Image, color_topic, self._rgb_cb, 1)
        self.create_subscription(Image, depth_topic, self._depth_cb, 1)
        self.create_subscription(CameraInfo, info_topic, self._info_cb, 1)
        
        # Create service
        self._service = self.create_service(
            DetectObject,
            'detect_object',
            self._detect_callback,
            callback_group=self._cb_group
        )

        params = self._get_inference_params()
        self._detector = None
        if params["gdino_config"] and params["gdino_weights"] and params["fastsam_weights"]:
            self._detector = ObjectDetector(
                params["gdino_config"],
                params["gdino_weights"],
                params["fastsam_weights"],
                params["device"],
            )
        
        self.get_logger().warn("Detection Initiated.")
    
    def _get_inference_params(self):
        """Get and expand inference params once per request."""
        return {
            "gdino_config": os.path.expandvars(os.path.expanduser(self.get_parameter("grounding_dino_config").value or "")),
            "gdino_weights": os.path.expandvars(os.path.expanduser(self.get_parameter("grounding_dino_weights").value or "")),
            "fastsam_weights": os.path.expandvars(os.path.expanduser(self.get_parameter("fastsam_weights").value or "")),
            "device": self.get_parameter("device").value,
        }

    def _rgb_cb(self, msg: Image):
        """Store latest RGB image."""
        with self._lock:
            self._latest_rgb = msg
    
    def _depth_cb(self, msg: Image):
        """Store latest depth image."""
        with self._lock:
            self._latest_depth = msg
    
    def _info_cb(self, msg: CameraInfo):
        """Store camera info."""
        with self._lock:
            if self._camera_info is None:
                debug_val = self.get_parameter('detection_debug').value
                if debug_val if isinstance(debug_val, bool) else str(debug_val).lower() in ('true', '1', 'yes'):
                    self.get_logger().info(f"Received camera_info: {msg.width}x{msg.height}")
            self._camera_info = msg
    
    def _detect_callback(self, request, response):
        """Service callback for object detection with pointcloud generation."""
        debug = getattr(request, 'debug', False)

        if debug:
            self.get_logger().info(f"Detection request: '{request.text_prompt}'")

        # Get latest sensor data
        missing = []
        with self._lock:    
            if self._latest_rgb is None:
                missing.append("RGB image")
            if self._latest_depth is None:
                missing.append("depth image")
            if self._camera_info is None:
                missing.append("camera_info")
            if missing:
                response.success = False
                response.message = f"Missing: {', '.join(missing)}"
                return response
            
            rgb_msg = self._latest_rgb
            depth_msg = self._latest_depth
            camera_info = self._camera_info
        
        # Decode RGB image
        try:
            cv_image = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough")
            enc_override = self.get_parameter("input_encoding").value or ""
            encoding = enc_override if enc_override else getattr(rgb_msg, "encoding", "")
            rgb = ros_encoding_to_rgb(cv_image, encoding)
        except Exception as e:
            response.success = False
            response.message = f"Image decode error: {e}"
            return response
        
        # Run GroundingDINO + FastSAM
        try:
            if self._detector is None:
                response.success = False
                response.message = "GroundingDINO/FastSAM weights not configured"
                return response

            result = self._detector.run(
                rgb_np=rgb,
                text_prompt=request.text_prompt,
                box_threshold=request.box_threshold,
                text_threshold=request.text_threshold,
            )
            
            if result is None:
                response.success = False
                response.message = f"No detection for '{request.text_prompt}'"
                return response

            if debug:
                self.get_logger().info(
                    f"Detected '{result.label}' score={result.score:.3f} "
                    f"box={result.box_xyxy.astype(int).tolist()} mask_area={result.mask.sum()}"
                )
                path = ObjectDetector.save_visualization(rgb, result)
                if path:
                    self.get_logger().info(f"Saved visualization to {path}")
                else:
                    self.get_logger().warn("Visualization failed")

        except Exception as e:
            response.success = False
            response.message = f"Inference error: {e}"
            self.get_logger().error(response.message)
            return response
        
        response.bbox = result.box_xyxy.astype(int).tolist()
        response.confidence = float(result.score)

        try:
            pc_msg, centroid = self._create_pointcloud(result.mask, depth_msg, camera_info, depth_msg.header, debug)
            response.pointcloud = pc_msg
            response.centroid.x = float(centroid[0])
            response.centroid.y = float(centroid[1])
            response.centroid.z = float(centroid[2])
            response.success = True
            response.message = f"Detected '{result.label}'"
            if debug:
                self.get_logger().info(f"Generated pointcloud: {pc_msg.width} points, centroid from height-filtered cloud")
        except Exception as e:
            response.success = False
            response.message = f"Pointcloud generation error: {e}"
            self.get_logger().error(response.message)

        return response

    def _create_pointcloud(self, mask: np.ndarray, depth_msg: Image, camera_info: CameraInfo, header, debug: bool = False):
        """
        Convert depth image + mask to a 3D point cloud.
        Uses FastSAM mask for segmentation; applies height filtering for robustness.

        Returns:
            Tuple of (PointCloud2 message, centroid as np.ndarray [x,y,z] in target_frame)
        """
        try:
            depth_cv = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            raise RuntimeError(f"Failed to decode depth image: {exc}")

        fx, fy = camera_info.k[0], camera_info.k[4]
        cx, cy = camera_info.k[2], camera_info.k[5]
        points_cam = depth_mask_to_points_cam(mask, depth_cv, fx, fy, cx, cy)

        # Transform to target frame; apply height filter (z in base_link)
        target_frame = self.get_parameter("target_frame").value
        z_min = self.get_parameter("height_filter_z_min").value

        if target_frame:
            source_frame = header.frame_id
            try:
                transform = self._tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
                t = transform.transform.translation
                q = transform.transform.rotation
                translation = np.array([t.x, t.y, t.z])
                rotation = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                points_cam = (rotation @ points_cam.T).T + translation
                header.frame_id = target_frame
                points_cam = points_cam[points_cam[:, 2] > z_min]
            except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as ex:
                self.get_logger().warn(f"TF lookup failed ({source_frame} -> {target_frame}): {ex}")

        if len(points_cam) == 0:
            raise ValueError("No points remaining after filtering")

        pc_msg = create_pointcloud2_msg(points_cam, header)
        centroid = centroid_from_points(points_cam)
        if debug:
            self.get_logger().info(
                f"Centroid from pointcloud (n={len(points_cam)}): [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]"
            )
        return pc_msg, centroid


def main(args=None):
    rclpy.init(args=args)
    node = DetectionServiceNode()

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
