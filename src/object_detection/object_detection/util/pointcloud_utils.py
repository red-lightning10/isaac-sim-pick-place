"""
Pointcloud utilities for object detection.
Converts depth + mask to 3D points and PointCloud2 messages.
"""
from __future__ import annotations

import struct

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


def depth_mask_to_points_cam(
    mask: np.ndarray,
    depth_cv: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Back-project masked depth pixels to 3D points in camera frame.

    Args:
        mask: Binary mask (H, W) bool or uint8
        depth_cv: Depth image (H, W) float32
        fx, fy, cx, cy: Camera intrinsics

    Returns:
        points_cam: (N, 3) float64 array in camera frame
    """
    if depth_cv.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Depth shape {depth_cv.shape[:2]} != mask shape {mask.shape[:2]}"
        )

    h, w = mask.shape
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

    mask_bool = mask.astype(bool)
    u_masked = u_coords[mask_bool]
    v_masked = v_coords[mask_bool]
    z_masked = depth_cv[mask_bool].astype(np.float32)

    valid = (z_masked > 0.0) & np.isfinite(z_masked)
    u_valid = u_masked[valid]
    v_valid = v_masked[valid]
    z_valid = z_masked[valid]

    x_cam = (u_valid - cx) * z_valid / fx
    y_cam = (v_valid - cy) * z_valid / fy
    z_cam = z_valid

    return np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float64)


def create_pointcloud2_msg(points: np.ndarray, header) -> PointCloud2:
    """Pack Nx3 points into a PointCloud2 message."""
    num_points = len(points)
    data = []
    for point in points:
        data.extend(struct.pack("fff", float(point[0]), float(point[1]), float(point[2])))

    pc_msg = PointCloud2()
    pc_msg.header = header
    pc_msg.height = 1
    pc_msg.width = num_points
    pc_msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    pc_msg.is_bigendian = False
    pc_msg.point_step = 12
    pc_msg.row_step = pc_msg.point_step * num_points
    pc_msg.data = bytes(data)
    pc_msg.is_dense = True
    return pc_msg


def centroid_from_points(points: np.ndarray) -> np.ndarray:
    """Compute centroid as median of points (more robust than mean)."""
    return np.median(points, axis=0).astype(np.float64)
