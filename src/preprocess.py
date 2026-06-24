"""Shared preprocessing helpers."""

import cv2
import numpy as np


def _make_odd(value, minimum=1):
    value = max(minimum, int(round(float(value))))
    return value if value % 2 == 1 else value + 1


def to_gray(image):
    """Convert an image to 8-bit grayscale."""
    if image is None:
        raise ValueError("Anh dau vao rong.")
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def apply_clahe(gray, clip_limit, tile_grid_size):
    """Apply CLAHE with a square tile size."""
    tile = max(1, int(round(float(tile_grid_size))))
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(clip_limit)),
        tileGridSize=(tile, tile),
    )
    return clahe.apply(gray)


def apply_blur(gray, params):
    """Apply one of the supported blur methods."""
    method = str(params.get("blur_method", "none")).lower()
    logs = []
    if method == "gaussian":
        kernel = _make_odd(params.get("gaussian_kernel", 5), minimum=1)
        logs.append("Blur: Gaussian k={}".format(kernel))
        return cv2.GaussianBlur(gray, (kernel, kernel), 0), logs
    if method == "median":
        kernel = _make_odd(params.get("median_kernel", 5), minimum=1)
        logs.append("Blur: Median k={}".format(kernel))
        return cv2.medianBlur(gray, kernel), logs
    if method == "bilateral":
        d = max(1, int(round(float(params.get("bilateral_d", 7)))))
        sigma_color = max(1.0, float(params.get("bilateral_sigma_color", 50)))
        sigma_space = max(1.0, float(params.get("bilateral_sigma_space", 50)))
        logs.append("Blur: Bilateral d={}".format(d))
        return cv2.bilateralFilter(gray, d, sigma_color, sigma_space), logs
    logs.append("Blur: None")
    return gray.copy(), logs


def preprocess_for_hough(image, params):
    """Prepare grayscale image for HoughCircle."""
    gray = to_gray(image)
    logs = ["Gray image ready"]
    output = gray.copy()
    preprocess_cfg = params.get("preprocess", {})
    images = {"gray": gray}

    if not preprocess_cfg.get("enabled", True):
        logs.append("Preprocess truoc Hough: TAT")
        images["preprocessed"] = gray.copy()
        return {"success": True, "data": output, "images": images, "logs": logs}

    if preprocess_cfg.get("use_clahe", True):
        output = apply_clahe(
            output,
            preprocess_cfg.get("clahe_clip_limit", 2.5),
            preprocess_cfg.get("clahe_tile_grid_size", 8),
        )
        logs.append("CLAHE enabled")
    if preprocess_cfg.get("use_gaussian", True):
        kernel = _make_odd(preprocess_cfg.get("gaussian_kernel", 5))
        output = cv2.GaussianBlur(output, (kernel, kernel), 0)
        logs.append("Gaussian blur k={}".format(kernel))

    images["preprocessed"] = output
    return {"success": True, "data": output, "images": images, "logs": logs}


def preprocess_roi_for_tab_edges(roi, params):
    """Prepare ROI for tab-edge extraction."""
    gray = to_gray(roi)
    output = gray.copy()
    logs = ["ROI gray ready"]
    images = {"roi_gray": gray}

    preprocess_cfg = params.get("preprocess", {})
    if preprocess_cfg.get("use_clahe", True):
        output = apply_clahe(
            output,
            preprocess_cfg.get("clahe_clip_limit", 2.0),
            preprocess_cfg.get("clahe_tile_grid_size", 8),
        )
        logs.append("CLAHE enabled")

    output, blur_logs = apply_blur(output, preprocess_cfg)
    logs.extend(blur_logs)
    images["roi_preprocessed"] = output
    return {"success": True, "data": output, "images": images, "logs": logs}

