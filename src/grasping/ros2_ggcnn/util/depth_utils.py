"""Depth image processing for GGCNN grasp inference."""

import numpy as np
import cv2
import torch
from skimage.filters import gaussian


def get_patch(depth_image: np.ndarray, bbox, patch_size: int = 300):
    """Crop a patch around the bounding box center."""
    min_x, min_y, max_x, max_y = bbox
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    half = patch_size // 2

    top_left_x = max(0, min(center_x - half, depth_image.shape[1] - patch_size))
    top_left_y = max(0, min(center_y - half, depth_image.shape[0] - patch_size))

    depth_crop = depth_image[
        top_left_y : top_left_y + patch_size,
        top_left_x : top_left_x + patch_size,
    ]
    return depth_crop, (top_left_x, top_left_y)


def process_depth_image(depth: np.ndarray) -> np.ndarray:
    """Process depth image for model input."""
    depth = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = np.isnan(depth).astype(np.uint8)
    depth[depth_nan_mask == 1] = 0

    depth_scale = np.abs(depth).max()
    depth = depth.astype(np.float32) / depth_scale
    depth = cv2.inpaint(depth, depth_nan_mask, 1, cv2.INPAINT_NS)
    return depth[1:-1, 1:-1] * depth_scale


def post_process_output(pos_output, cos_output, sin_output, width_output, width_scale: float = 150.0):
    """Post-process model outputs."""
    q_img = pos_output.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_output, cos_output) / 2.0).cpu().numpy().squeeze()
    width_img = width_output.cpu().numpy().squeeze() * width_scale

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img


def calculate_grasp(q_img, ang_img, width_img, bounding_box):
    """Find best grasp in bounding box region. Returns (grasp_y, grasp_x, angle, grasp_width)."""
    q_img_temp = q_img[
        bounding_box[1] : bounding_box[1] + 2 * (bounding_box[3] - bounding_box[1]) // 3,
        bounding_box[0] : bounding_box[2],
    ]
    max_q_idx = np.unravel_index(np.argmax(q_img_temp), q_img_temp.shape)
    max_q_idx = (max_q_idx[0] + bounding_box[1], max_q_idx[1] + bounding_box[0])
    grasp_y, grasp_x = max_q_idx

    angle = ang_img[grasp_y, grasp_x]
    grasp_width = width_img[grasp_y, grasp_x]
    return grasp_y, grasp_x, angle, grasp_width
