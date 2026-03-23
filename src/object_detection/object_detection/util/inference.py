"""
GroundingDINO + FastSAM inference wrapper.

Channel / shape conventions:
  Input to ObjectDetector.run():  HxWxC RGB uint8 numpy
  GroundingDINO input:            PIL(RGB) → T.Compose → CHW float32 tensor
  GroundingDINO boxes out:        (N, 4) cxcywh NORMALISED [0..1]  → converted to pixel xyxy
  FastSAM input:                 HxWxC RGB uint8 numpy (passed as numpy array)
  FastSAM masks out:             (N, H, W) bool tensor → take [0] → (H, W) bool
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from groundingdino.util.inference import load_model as load_gdino_model, predict as gdino_predict
from fastsam import FastSAM
import groundingdino.datasets.transforms as T

# PyTorch 2.6+ defaults torch.load(..., weights_only=True); legacy checkpoints need weights_only=False
_torch_load_orig = torch.load


def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)


torch.load = _torch_load


@dataclass
class DetectionResult:
    mask: np.ndarray  
    box_xyxy: np.ndarray 
    score: float
    label: str


class ObjectDetector:
    """
    GroundingDINO + FastSAM object detector with cached models.
    """

    # GroundingDINO image transform
    GDINO_RESIZE_SIZES = [800]
    GDINO_RESIZE_MAX_SIZE = 1333
    GDINO_MEAN = [0.485, 0.456, 0.406]
    GDINO_STD = [0.229, 0.224, 0.225]

    # Default detection thresholds
    DEFAULT_BOX_THRESHOLD = 0.35
    DEFAULT_TEXT_THRESHOLD = 0.25

    # FastSAM
    FASTSAM_IMGSZ = 1024
    FASTSAM_CONF = 0.4
    FASTSAM_IOU = 0.9

    def __init__(
        self,
        gdino_config: str,
        gdino_weights: str,
        fastsam_weights: str,
        device: str = "cuda",
    ):
        self._gdino_config = gdino_config
        self._gdino_weights = gdino_weights
        self._fastsam_weights = fastsam_weights
        self._device = "cpu" if (device == "cuda" and not torch.cuda.is_available()) else device

        self._lock = threading.Lock()
        self._gdino_model = None
        self._fastsam_model = None
        self._gdino_weights_loaded: Optional[str] = None
        self._fastsam_weights_loaded: Optional[str] = None

        self._gdino_transform = None
        self._gdino_transform_lock = threading.Lock()

    def _get_models(self):
        with self._lock:
            if self._gdino_model is None or self._gdino_weights_loaded != self._gdino_weights:
                self._gdino_model = load_gdino_model(
                    self._gdino_config, self._gdino_weights, device=self._device
                )
                self._gdino_weights_loaded = self._gdino_weights
            if self._fastsam_model is None or self._fastsam_weights_loaded != self._fastsam_weights:
                self._fastsam_model = FastSAM(self._fastsam_weights)
                self._fastsam_weights_loaded = self._fastsam_weights
        return self._gdino_model, self._fastsam_model

    def _get_gdino_transform(self):
        with self._gdino_transform_lock:
            if self._gdino_transform is None:
                self._gdino_transform = T.Compose([
                    T.RandomResize(self.GDINO_RESIZE_SIZES, max_size=self.GDINO_RESIZE_MAX_SIZE),
                    T.ToTensor(),
                    T.Normalize(self.GDINO_MEAN, self.GDINO_STD),
                ])
        return self._gdino_transform

    def _rgb_to_gdino_tensor(self, rgb_np: np.ndarray) -> torch.Tensor:
        """HxWxC RGB uint8 → CHW float32 tensor via GroundingDINO's own transform."""
        pil = PILImage.fromarray(rgb_np)
        tensor, _ = self._get_gdino_transform()(pil, None)
        return tensor

    @staticmethod
    def _cxcywh_norm_to_xyxy_px(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """(N,4) normalised cxcywh → (N,4) pixel xyxy."""
        cx, cy, bw, bh = boxes.unbind(1)
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        return torch.stack([x1, y1, x2, y2], dim=1).clamp(min=0)

    def run(
        self,
        rgb_np: np.ndarray,
        text_prompt: str,
        box_threshold: float = None,
        text_threshold: float = None,
    ) -> Optional[DetectionResult]:
        """
        Run GroundingDINO (text → box) then FastSAM (box → mask).

        Returns
        -------
        DetectionResult or None if nothing detected.
        """
        if box_threshold is None:
            box_threshold = self.DEFAULT_BOX_THRESHOLD
        if text_threshold is None:
            text_threshold = self.DEFAULT_TEXT_THRESHOLD

        if rgb_np.ndim != 3 or rgb_np.shape[2] != 3:
            raise ValueError(f"Expected HxWxC RGB, got shape {rgb_np.shape}")

        img_h, img_w = rgb_np.shape[:2]
        gdino_model, fastsam_model = self._get_models()

        # Stage 1: GroundingDINO → boxes
        image_tensor = self._rgb_to_gdino_tensor(rgb_np).to(self._device)
        with torch.no_grad():
            boxes_cxcywh, logits, phrases = gdino_predict(
                model=gdino_model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self._device,
            )

        if boxes_cxcywh is None or boxes_cxcywh.shape[0] == 0:
            return None

        boxes_xyxy = self._cxcywh_norm_to_xyxy_px(boxes_cxcywh, img_h, img_w)
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(max=img_w)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(max=img_h)

        best_idx = int(logits.argmax())
        best_box = boxes_xyxy[best_idx].cpu().numpy().astype(np.float32)
        best_score = float(logits[best_idx])
        best_label = phrases[best_idx] if phrases else text_prompt

        # Stage 2: mask from FastSAM
        x1, y1, x2, y2 = best_box.astype(int)
        x1, x2 = max(0, x1), min(img_w, x2)
        y1, y2 = max(0, y1), min(img_h, y2)

        results = fastsam_model(
            rgb_np,
            device=self._device,
            retina_masks=True,
            imgsz=self.FASTSAM_IMGSZ,
            conf=self.FASTSAM_CONF,
            iou=self.FASTSAM_IOU,
        )
        if not results or results[0].masks is None:
            return None

        masks_tensor = results[0].masks.data.cpu()
        box_area = max((x2 - x1) * (y2 - y1), 1)
        best_mask_idx = 0
        best_iou = -1.0
        for i in range(masks_tensor.shape[0]):
            m = masks_tensor[i].numpy().astype(bool)
            if m.shape != (img_h, img_w):
                m = cv2.resize(
                    m.astype(np.uint8), (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            intersection = m[y1:y2, x1:x2].sum()
            mask_area = m.sum()
            union = mask_area + box_area - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_mask_idx = i

        final_mask = masks_tensor[best_mask_idx].numpy().astype(bool)
        if final_mask.shape != (img_h, img_w):
            final_mask = cv2.resize(
                final_mask.astype(np.uint8), (img_w, img_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        return DetectionResult(
            mask=final_mask,
            box_xyxy=best_box,
            score=best_score,
            label=best_label,
        )

    @staticmethod
    def get_output_dir() -> str:
        """Return output directory for detection visualizations ($REPO_ROOT/output/detection)."""
        repo_root = os.environ.get("REPO_ROOT", "")
        if repo_root:
            return os.path.join(os.path.expandvars(os.path.expanduser(repo_root)), "output", "detection")
        pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.abspath(os.path.join(pkg_dir, "..", "..", "..", "output", "detection"))

    @staticmethod
    def save_visualization(rgb: np.ndarray, result: DetectionResult, output_dir: str = None) -> Optional[str]:
        """Save detection visualization. Returns path or None on failure."""
        try:
            if output_dir is None:
                output_dir = ObjectDetector.get_output_dir()
            os.makedirs(output_dir, exist_ok=True)
            vis = rgb.copy()
            x1, y1, x2, y2 = result.box_xyxy.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            mask_overlay = np.zeros_like(vis)
            mask_overlay[result.mask] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 1.0, mask_overlay, 0.3, 0)
            cv2.putText(vis, f"{result.label} {result.score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            path = os.path.join(output_dir, f"detection_{int(time.time())}.png")
            cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            return path
        except Exception:
            return None
