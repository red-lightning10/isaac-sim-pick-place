"""
Image normalisation utilities.

All internal processing uses HxWxC RGB uint8 numpy arrays.
ROS image messages can arrive in various encodings — this module
normalises them to a single consistent format before inference.

Mask convention used throughout this package:
  - Masks are always (H, W) bool or uint8 — NO channel dimension.
"""

from __future__ import annotations

import cv2
import numpy as np


def ros_encoding_to_rgb(cv_image: np.ndarray, encoding: str) -> np.ndarray:
    """
    Convert a cv_bridge passthrough image to HxWxC RGB uint8.

    Parameters
    ----------
    cv_image : np.ndarray   Raw array from cv_bridge.
    encoding : str          ROS msg.encoding, e.g. "bgr8", "rgb8", "mono8".
                            Pass "" to assume RGB.

    Returns
    -------
    np.ndarray  shape (H, W, 3), dtype uint8, RGB order.
    """
    enc = (encoding or "").lower().strip()

    # Grayscale
    if cv_image.ndim == 2 or enc in ("mono8", "mono16", "32fc1"):
        gray = cv_image.astype(np.uint8) if cv_image.dtype == np.uint8 \
               else _to_uint8(cv_image)
        return np.stack([gray, gray, gray], axis=-1)

    img = cv_image.astype(np.uint8)

    if enc in ("bgr8", "bgra8"):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if enc == "rgba8":
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # rgb8, unknown, or empty → strip to 3 channels and assume RGB
    return img[..., :3]


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
