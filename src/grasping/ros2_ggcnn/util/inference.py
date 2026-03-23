"""
GGCNN grasp inference wrapper.

Input: depth image (H, W), bbox [x_min, y_min, x_max, y_max]
Output: (grasp_x, grasp_y, angle, grasp_width) in full image coordinates
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from ..ggcnn2 import GGCNN
from .depth_utils import get_patch, process_depth_image, post_process_output, calculate_grasp


_torch_load_orig = torch.load


def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)


torch.load = _torch_load


def _plot_rectangle_on_image(image, grasp_x, grasp_y, angle, grasp_width):
    """Draw grasp rectangle on image."""
    grasp_display = image.copy()
    half_length = grasp_width / 2
    half_width = half_length / 2
    corners = np.array([
        [-half_length, -half_width],
        [-half_length, half_width],
        [half_length, half_width],
        [half_length, -half_width],
    ])
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    rotated_corners = np.dot(corners, rotation_matrix.T)
    rotated_corners[:, 0] += grasp_x
    rotated_corners[:, 1] += grasp_y
    pts = [(int(rotated_corners[i, 0]), int(rotated_corners[i, 1])) for i in range(4)]
    for i in range(4):
        cv2.line(grasp_display, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
    cv2.circle(grasp_display, (grasp_x, grasp_y), 3, (0, 255, 0), -1)
    return grasp_display


class GraspComputer:
    """GGCNN grasp predictor with cached model."""

    PATCH_SIZE = 300
    WIDTH_SCALE = 150.0

    def __init__(self, model_path: str, device: str = "cuda"):
        self._model_path = model_path
        self._device = "cpu" if (device == "cuda" and not torch.cuda.is_available()) else device

        self._model = GGCNN()
        self._model.load_state_dict(torch.load(model_path, weights_only=False))
        self._model.eval()

    def run(
        self, depth: np.ndarray, bbox
    ) -> Optional[Tuple[int, int, float, float]]:
        """
        Run GGCNN inference. Returns (grasp_x, grasp_y, angle, grasp_width) or None.
        """
        if len(bbox) < 4:
            return None

        bbox = bbox[:4]
        depth_crop, top_corner = get_patch(depth, bbox, self.PATCH_SIZE)
        depth_proc = process_depth_image(depth_crop)

        depth_tensor = torch.tensor(depth_proc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pos_out, cos_out, sin_out, width_out = self._model(depth_tensor)

        q_img, ang_img, width_img = post_process_output(
            pos_out, cos_out, sin_out, width_out, self.WIDTH_SCALE
        )

        new_bbox = (
            bbox[0] - top_corner[0],
            bbox[1] - top_corner[1],
            bbox[2] - top_corner[0],
            bbox[3] - top_corner[1],
        )
        grasp_y, grasp_x, angle, grasp_width = calculate_grasp(
            q_img, ang_img, width_img, new_bbox
        )

        grasp_x += top_corner[0]
        grasp_y += top_corner[1]

        return grasp_x, grasp_y, angle, grasp_width

    @staticmethod
    def get_output_dir() -> str:
        """Return output directory for grasp visualizations ($REPO_ROOT/output/ggcnn)."""
        repo_root = os.environ.get("REPO_ROOT", "")
        if repo_root:
            return os.path.join(os.path.expandvars(os.path.expanduser(repo_root)), "output", "ggcnn")
        pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.abspath(os.path.join(pkg_dir, "..", "..", "..", "output", "ggcnn"))

    @staticmethod
    def save_visualization(depth, grasp_x, grasp_y, angle, grasp_width, output_dir: str = None) -> Optional[str]:
        """Save grasp visualization to output_dir. Returns path or None on failure."""
        try:
            if output_dir is None:
                output_dir = GraspComputer.get_output_dir()
            os.makedirs(output_dir, exist_ok=True)
            depth_vis = depth.copy()
            depth_vis = (depth_vis / depth_vis.max() * 255).astype(np.uint8)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            vis_img = _plot_rectangle_on_image(depth_vis, grasp_x, grasp_y, angle, grasp_width)
            path = os.path.join(output_dir, f"ggcnn_grasp_{int(time.time())}.png")
            cv2.imwrite(path, vis_img)
            return path
        except Exception:
            return None
