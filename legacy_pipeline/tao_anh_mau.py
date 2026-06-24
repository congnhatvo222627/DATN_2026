# -*- coding: utf-8 -*-
import base64
import json
import math
import os
import re

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    import Tkinter as tk
    import ttk
    import tkFileDialog as filedialog
    import tkMessageBox as messagebox

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "images"))
HOUGH_PRESETS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Circle_Center", "presets"))
LEGACY_RADIAL_PRESETS_DIR = os.path.join(BASE_DIR, "presets")
TEMPLATE_SETTING_PRESETS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "presets", "template_settings"))
YOLO_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "best.pt"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MIN_PROCESS_SHORT_SIDE = 700
MIN_PROCESS_LONG_SIDE = 1000

DEFAULT_HOUGH_PARAMS = {
    "clahe_clip": 2.0,
    "blur_kernel": 7,
    "blur_sigma": 1.5,
    "canny_low": 60,
    "canny_high": 140,
    "roi_ratio": 0.72,
    "min_radius_ratio": 0.22,
    "max_radius_ratio": 0.44,
    "min_dist_ratio": 0.50,
    "hough_dp": 1.2,
    "hough_param1": 140,
    "hough_param2": 26,
    "band_width": 8,
}
TAB_BOX_PADDING_RATIO = 0.10
TAB_BOX_PADDING_MIN_PX = 12
TAB_MIN_CONTOUR_AREA = 24.0
TAB_MIN_CONTOUR_AREA_RATIO = 0.0015
TAB_CONTOUR_KEEP_DISTANCE_RATIO = 0.88
TAB_OUTER_PROFILE_BIN_DEG = 1.0
TAB_OUTER_PROFILE_MAX_POINT_GAP = 28.0
TAB_OUTER_PROFILE_MAX_ANGLE_GAP_DEG = 5.0
WHITE_DRAW_COLOR = (255, 255, 255)
RADIAL_LINE_COLOR = (0, 255, 0)
HOUGH_CIRCLE_COLOR = (255, 0, 255)
CENTER_MARKER_COLOR = (0, 0, 255)
RADIAL_HIT_POINT_COLOR = (0, 255, 255)
SIGNATURE_BG_COLOR = "#ffffff"
SIGNATURE_PLOT_BG_COLOR = "#ffffff"
SIGNATURE_GRID_COLOR = "#d7dde5"
SIGNATURE_AXIS_COLOR = "#64748b"
SIGNATURE_TEXT_COLOR = "#111827"
SIGNATURE_BASELINE_COLOR = "#ec4899"
SIGNATURE_HIT_COLOR = "#f59e0b"
SIGNATURE_FLAT_COLOR = "#22c55e"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def natural_sort_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def ensure_odd(value):
    value = int(round(float(value)))
    return value if value % 2 == 1 else value + 1


def clamp_value(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def smooth_circular_profile(values, kernel_size):
    if np is None:
        return []

    profile = np.asarray(values, dtype=np.float32)
    sample_count = int(profile.size)
    if sample_count == 0:
        return []

    kernel_size = int(max(1, kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size <= 1:
        return profile.astype(np.float32).tolist()

    if kernel_size > sample_count:
        kernel_size = sample_count if (sample_count % 2 == 1) else max(1, sample_count - 1)
        if kernel_size <= 1:
            return profile.astype(np.float32).tolist()

    pad = kernel_size // 2
    extended = np.concatenate([profile[-pad:], profile, profile[:pad]])
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    smoothed = np.convolve(extended, kernel, mode="same")
    return smoothed[pad : pad + sample_count].astype(np.float32).tolist()


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_edges(gray_image, params):
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(params["clahe_clip"])),
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(gray_image)
    blurred = cv2.GaussianBlur(
        enhanced,
        (ensure_odd(params["blur_kernel"]), ensure_odd(params["blur_kernel"])),
        max(0.1, float(params["blur_sigma"])),
    )
    return enhanced, blurred


def preprocess_for_circle(gray_image, params):
    return preprocess_for_edges(gray_image, params)


def upscale_small_image(image_bgr):
    height, width = image_bgr.shape[:2]
    short_side = float(min(width, height))
    long_side = float(max(width, height))
    scale_factor = max(
        float(MIN_PROCESS_SHORT_SIDE) / max(1.0, short_side),
        float(MIN_PROCESS_LONG_SIDE) / max(1.0, long_side),
        1.0,
    )
    if scale_factor <= 1.0:
        return image_bgr, 1.0
    new_width = max(1, int(round(width * scale_factor)))
    new_height = max(1, int(round(height * scale_factor)))
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized, scale_factor


def scale_hough_result(hough_result, scale_factor):
    if scale_factor == 1.0:
        return hough_result

    scaled_result = dict(hough_result)
    scaled_result["center_x"] = float(hough_result["center_x"]) * float(scale_factor)
    scaled_result["center_y"] = float(hough_result["center_y"]) * float(scale_factor)
    scaled_result["radius"] = float(hough_result["radius"]) * float(scale_factor)

    debug = dict(hough_result.get("debug", {}))
    roi = debug.get("roi")
    if roi is not None and len(roi) == 4:
        roi_x, roi_y, roi_w, roi_h = roi
        debug["roi"] = (
            int(round(float(roi_x) * float(scale_factor))),
            int(round(float(roi_y) * float(scale_factor))),
            int(round(float(roi_w) * float(scale_factor))),
            int(round(float(roi_h) * float(scale_factor))),
        )

    ring_points = debug.get("ring_points")
    if ring_points:
        debug["ring_points"] = [
            (
                float(point_x) * float(scale_factor),
                float(point_y) * float(scale_factor),
            )
            for point_x, point_y in ring_points
        ]

    scaled_result["debug"] = debug
    return scaled_result


# ---------------------------------------------------------------------------
# Hough circle detection
# ---------------------------------------------------------------------------

def build_roi(image_shape, roi_ratio):
    height, width = image_shape[:2]
    roi_ratio = max(0.3, min(1.0, float(roi_ratio)))
    roi_width = int(width * roi_ratio)
    roi_height = int(height * roi_ratio)
    offset_x = int((width - roi_width) / 2.0)
    offset_y = int((height - roi_height) / 2.0)
    return offset_x, offset_y, roi_width, roi_height


def fit_circle_least_squares(points):
    if len(points) < 3:
        return None
    points_array = np.asarray(points, dtype=np.float64)
    x_values = points_array[:, 0]
    y_values = points_array[:, 1]
    matrix_a = np.column_stack((x_values, y_values, np.ones(len(points_array))))
    matrix_b = -(x_values ** 2 + y_values ** 2)
    try:
        solution, _, _, _ = np.linalg.lstsq(matrix_a, matrix_b, rcond=None)
    except TypeError:
        solution, _, _, _ = np.linalg.lstsq(matrix_a, matrix_b)
    coeff_a, coeff_b, coeff_c = solution
    center_x = -coeff_a / 2.0
    center_y = -coeff_b / 2.0
    radius_squared = (center_x ** 2) + (center_y ** 2) - coeff_c
    if radius_squared <= 0:
        return None
    return center_x, center_y, math.sqrt(radius_squared)


def mask_edges_by_radius(edge_image, roi_center, min_radius, max_radius):
    masked = np.zeros_like(edge_image)
    edge_points = cv2.findNonZero(edge_image)
    if edge_points is None:
        return masked
    center_x, center_y = roi_center
    points = edge_points[:, 0, :].astype(np.float32)
    dx = points[:, 0] - float(center_x)
    dy = points[:, 1] - float(center_y)
    distances = np.sqrt((dx * dx) + (dy * dy))
    keep_mask = (distances >= float(min_radius)) & (distances <= float(max_radius))
    kept_points = points[keep_mask].astype(np.int32)
    if len(kept_points) > 0:
        masked[kept_points[:, 1], kept_points[:, 0]] = 255
    return masked


def score_circle(circle, edge_image, roi_center, target_radius):
    center_x, center_y, radius = circle
    samples = 360
    hit_count = 0
    for index in range(samples):
        angle = (2.0 * math.pi * index) / samples
        sample_x = int(round(center_x + radius * math.cos(angle)))
        sample_y = int(round(center_y + radius * math.sin(angle)))
        if sample_x < 2 or sample_y < 2 or sample_x >= edge_image.shape[1] - 2 or sample_y >= edge_image.shape[0] - 2:
            continue
        patch = edge_image[sample_y - 2 : sample_y + 3, sample_x - 2 : sample_x + 3]
        if np.count_nonzero(patch) > 0:
            hit_count += 1
    support_ratio = float(hit_count) / float(samples)
    center_distance = math.hypot(center_x - roi_center[0], center_y - roi_center[1])
    center_penalty = center_distance / max(1.0, radius)
    radius_penalty = abs(radius - target_radius) / max(1.0, target_radius)
    return support_ratio - (0.10 * center_penalty) - (0.20 * radius_penalty)


def collect_ring_points(edge_image, circle, band_width):
    center_x, center_y, radius = circle
    points = []
    edge_points = cv2.findNonZero(edge_image)
    if edge_points is None:
        return points
    point_array = edge_points[:, 0, :].astype(np.float32)
    dx = point_array[:, 0] - float(center_x)
    dy = point_array[:, 1] - float(center_y)
    distances = np.sqrt((dx * dx) + (dy * dy))
    keep_mask = np.abs(distances - float(radius)) <= float(band_width)
    kept_points = point_array[keep_mask]
    if len(kept_points) == 0:
        return points
    return [tuple(map(float, point)) for point in kept_points]


def detect_hough_reference_circle(image_bgr, params):
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    enhanced, blurred = preprocess_for_circle(gray_image, params)

    roi_x, roi_y, roi_width, roi_height = build_roi(image_bgr.shape, params["roi_ratio"])
    roi_enhanced = enhanced[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    roi_blurred = blurred[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

    min_dimension = min(roi_width, roi_height)
    min_radius = int(min_dimension * float(params["min_radius_ratio"]))
    max_radius = int(min_dimension * float(params["max_radius_ratio"]))
    if min_radius >= max_radius:
        raise ValueError("Preset Hough co min_radius_ratio lon hon hoac bang max_radius_ratio.")

    raw_edges = cv2.Canny(
        roi_blurred,
        int(params["canny_low"]),
        int(params["canny_high"]),
    )
    masked_edges = mask_edges_by_radius(
        raw_edges,
        (roi_width / 2.0, roi_height / 2.0),
        min_radius,
        max_radius,
    )

    circles = cv2.HoughCircles(
        roi_blurred,
        cv2.HOUGH_GRADIENT,
        dp=max(1.0, float(params["hough_dp"])),
        minDist=max(20.0, min_dimension * float(params["min_dist_ratio"])),
        param1=max(1, int(params["hough_param1"])),
        param2=max(1, int(params["hough_param2"])),
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        raise ValueError("Khong tim thay duoc ung vien Hough tu preset dang nap.")

    roi_center = (roi_width / 2.0, roi_height / 2.0)
    target_radius = (min_radius + max_radius) / 2.0
    best_circle = None
    best_score = -999.0

    for circle in np.round(circles[0, :]).astype("int"):
        candidate = (float(circle[0]), float(circle[1]), float(circle[2]))
        candidate_score = score_circle(candidate, masked_edges, roi_center, target_radius)
        if candidate_score > best_score:
            best_score = candidate_score
            best_circle = candidate

    if best_circle is None:
        raise ValueError("Khong chon duoc vong tron Hough phu hop.")

    ring_points = collect_ring_points(masked_edges, best_circle, float(params["band_width"]))
    refined_circle = fit_circle_least_squares(ring_points) if len(ring_points) >= 20 else None
    if refined_circle is None:
        refined_circle = best_circle

    refined_x, refined_y, refined_radius = refined_circle
    debug = {
        "roi": (roi_x, roi_y, roi_width, roi_height),
        "roi_enhanced": roi_enhanced,
        "roi_blurred": roi_blurred,
        "raw_edges": raw_edges,
        "masked_edges": masked_edges,
        "ring_points": ring_points,
        "score": best_score,
    }
    return {
        "center_x": refined_x + roi_x,
        "center_y": refined_y + roi_y,
        "radius": refined_radius,
        "debug": debug,
    }


# ---------------------------------------------------------------------------
# YOLO tab detection
# ---------------------------------------------------------------------------

def load_yolo_model(model_path):
    """Load YOLO model from .pt file."""
    if YOLO is None:
        raise ImportError(
            "Can cai dat ultralytics truoc khi chay.\n\n"
            "Lenh goi y:\npip install ultralytics"
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Khong tim thay file model YOLO:\n{}".format(model_path))
    return YOLO(model_path)


def detect_tabs_yolo(image_bgr, model, conf_threshold=0.25):
    """
    Run YOLO inference on image. Returns list of tab detections.
    Each detection: dict with keys x1, y1, x2, y2, confidence, class_id, class_name
    """
    results = model.predict(image_bgr, conf=conf_threshold, verbose=False)
    detections = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names.get(cls_id, str(cls_id)) if hasattr(result, 'names') else str(cls_id)
                detections.append({
                    "x1": int(round(x1)),
                    "y1": int(round(y1)),
                    "x2": int(round(x2)),
                    "y2": int(round(y2)),
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                })
    return detections


# ---------------------------------------------------------------------------
# Tab edge extraction & radial lines
# ---------------------------------------------------------------------------

def preprocess_tab_region(tab_bgr, params):
    """Apply preprocessing to a cropped tab region."""
    gray = cv2.cvtColor(tab_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(params.get("tab_clahe_clip", 2.0))),
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(gray)
    k = ensure_odd(params.get("tab_blur_kernel", 5))
    sigma = max(0.1, float(params.get("tab_blur_sigma", 1.0)))
    blurred = cv2.GaussianBlur(enhanced, (k, k), sigma)
    edges = cv2.Canny(
        blurred,
        int(params.get("tab_canny_low", 50)),
        int(params.get("tab_canny_high", 150)),
    )
    close_k = ensure_odd(params.get("tab_close_kernel", 3))
    kernel = np.ones((close_k, close_k), dtype=np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=int(params.get("tab_close_iterations", 1)))
    return {
        "gray": gray,
        "enhanced": enhanced,
        "blurred": blurred,
        "edges": edges,
        "closed": closed,
    }


def extract_tab_edge_contours(closed_edges):
    """Find external contours in a tab region."""
    contour_info = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contour_info) == 3:
        _, contours, _ = contour_info
    else:
        contours, _ = contour_info

    if not contours:
        return []
    return contours


def build_outer_profile_contours(edge_points, center_x, center_y):
    """
    Keep only the outer envelope seen from the Hough center:
    for each angular bin, retain the farthest edge point.
    """
    if not edge_points:
        return []

    bin_size_deg = max(0.5, float(TAB_OUTER_PROFILE_BIN_DEG))
    outer_points_by_bin = {}

    for px, py, dist in edge_points:
        angle_deg = (math.degrees(math.atan2(py - center_y, px - center_x)) + 360.0) % 360.0
        bin_index = int(round(angle_deg / bin_size_deg))
        current_best = outer_points_by_bin.get(bin_index)
        if current_best is None or dist > current_best[2]:
            outer_points_by_bin[bin_index] = (float(px), float(py), float(dist), float(angle_deg))

    if not outer_points_by_bin:
        return []

    sorted_points = [outer_points_by_bin[key] for key in sorted(outer_points_by_bin.keys())]
    segments = []
    current_segment = []
    previous_point = None

    for point in sorted_points:
        px, py, dist, angle_deg = point
        if previous_point is not None:
            prev_px, prev_py, _, prev_angle_deg = previous_point
            angle_gap = abs(angle_deg - prev_angle_deg)
            point_gap = math.hypot(px - prev_px, py - prev_py)
            if angle_gap > float(TAB_OUTER_PROFILE_MAX_ANGLE_GAP_DEG) or point_gap > float(TAB_OUTER_PROFILE_MAX_POINT_GAP):
                if len(current_segment) >= 2:
                    contour = np.asarray(current_segment, dtype=np.int32).reshape((-1, 1, 2))
                    segments.append(contour)
                current_segment = []

        current_segment.append((int(round(px)), int(round(py))))
        previous_point = point

    if len(current_segment) >= 2:
        contour = np.asarray(current_segment, dtype=np.int32).reshape((-1, 1, 2))
        segments.append(contour)

    return segments


def compute_tab_edge_points_on_full_image(detection, preprocessed, center_x, center_y, crop_x1, crop_y1):
    """
    Given a YOLO detection box and its preprocessed edge data,
    extract edge points in full-image coordinates, keep multiple valid
    contour fragments, and find the outermost point plus a centroid-guided
    radial edge point.
    """
    closed = preprocessed["closed"]

    contours = extract_tab_edge_contours(closed)
    if not contours:
        return None

    roi_height, roi_width = closed.shape[:2]
    roi_area = float(max(1, roi_width * roi_height))
    min_area = max(TAB_MIN_CONTOUR_AREA, roi_area * TAB_MIN_CONTOUR_AREA_RATIO)
    detection_center_x = (float(detection["x1"]) + float(detection["x2"])) / 2.0
    detection_center_y = (float(detection["y1"]) + float(detection["y2"])) / 2.0
    detection_center_dist = math.hypot(detection_center_x - center_x, detection_center_y - center_y)
    min_keep_distance = detection_center_dist * TAB_CONTOUR_KEEP_DISTANCE_RATIO

    filtered_contours = []
    outermost_point = None
    all_edge_pts = []
    weighted_x = 0.0
    weighted_y = 0.0
    total_weight = 0.0
    max_dist = 0.0

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < min_area:
            continue

        contour_full = contour.copy()
        contour_full[:, 0, 0] += crop_x1
        contour_full[:, 0, 1] += crop_y1

        contour_points = contour_full[:, 0, :].astype(np.float32)
        dx = contour_points[:, 0] - float(center_x)
        dy = contour_points[:, 1] - float(center_y)
        distances = np.sqrt((dx * dx) + (dy * dy))
        contour_max_dist = float(np.max(distances)) if len(distances) else 0.0
        if contour_max_dist < min_keep_distance:
            continue

        filtered_contours.append(contour_full)

        moments = cv2.moments(contour_full)
        if abs(moments["m00"]) > 1e-6:
            contour_cx = float(moments["m10"]) / float(moments["m00"])
            contour_cy = float(moments["m01"]) / float(moments["m00"])
        else:
            contour_cx = float(np.mean(contour_points[:, 0]))
            contour_cy = float(np.mean(contour_points[:, 1]))

        weight = max(contour_area, 1.0)
        weighted_x += contour_cx * weight
        weighted_y += contour_cy * weight
        total_weight += weight

        for point_xy, dist in zip(contour_points, distances):
            px = float(point_xy[0])
            py = float(point_xy[1])
            all_edge_pts.append((px, py, float(dist)))
            if dist > max_dist:
                max_dist = float(dist)
                outermost_point = (px, py)

    if not filtered_contours or not all_edge_pts:
        return None

    outer_profile_contours = build_outer_profile_contours(all_edge_pts, center_x, center_y)
    if outer_profile_contours:
        outer_edge_pts = []
        max_dist = 0.0
        outermost_point = None
        for contour in outer_profile_contours:
            contour_points = contour[:, 0, :].astype(np.float32)
            for point_xy in contour_points:
                px = float(point_xy[0])
                py = float(point_xy[1])
                dist = math.hypot(px - center_x, py - center_y)
                outer_edge_pts.append((px, py, dist))
                if dist > max_dist:
                    max_dist = float(dist)
                    outermost_point = (px, py)
    else:
        outer_edge_pts = all_edge_pts

    if total_weight > 0.0:
        tab_cx = weighted_x / total_weight
        tab_cy = weighted_y / total_weight
    else:
        tab_cx = detection_center_x
        tab_cy = detection_center_y

    angle_to_tab = math.atan2(tab_cy - center_y, tab_cx - center_x)
    radial_edge_point = None
    best_radial_dist = 0.0
    angle_tolerance = math.radians(18)

    for px, py, dist in outer_edge_pts:
        pt_angle = math.atan2(py - center_y, px - center_x)
        angle_diff = abs(math.atan2(math.sin(pt_angle - angle_to_tab), math.cos(pt_angle - angle_to_tab)))
        if angle_diff <= angle_tolerance and dist > best_radial_dist:
            best_radial_dist = dist
            radial_edge_point = (px, py)

    if radial_edge_point is None:
        radial_edge_point = outermost_point

    return {
        "raw_contours": filtered_contours,
        "contours": outer_profile_contours if outer_profile_contours else filtered_contours,
        "contour": (outer_profile_contours[0] if outer_profile_contours else filtered_contours[0]),
        "centroid": (tab_cx, tab_cy),
        "angle": angle_to_tab,
        "outermost": outermost_point,
        "radial_edge": radial_edge_point,
        "radial_dist": best_radial_dist,
        "max_dist": max_dist,
    }


def process_all_tabs(image_bgr, detections, center_x, center_y, params):
    """Process all detected tabs: preprocess, extract edges, compute radial points."""
    tab_results = []
    height, width = image_bgr.shape[:2]

    for det in detections:
        x1 = max(0, det["x1"])
        y1 = max(0, det["y1"])
        x2 = min(width, det["x2"])
        y2 = min(height, det["y2"])

        if x2 <= x1 or y2 <= y1:
            continue

        pad_x = max(TAB_BOX_PADDING_MIN_PX, int(round((x2 - x1) * TAB_BOX_PADDING_RATIO)))
        pad_y = max(TAB_BOX_PADDING_MIN_PX, int(round((y2 - y1) * TAB_BOX_PADDING_RATIO)))
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(width, x2 + pad_x)
        crop_y2 = min(height, y2 + pad_y)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue

        tab_crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        preprocessed = preprocess_tab_region(tab_crop, params)

        det_adjusted = dict(det)
        det_adjusted["x1"] = x1
        det_adjusted["y1"] = y1
        det_adjusted["x2"] = x2
        det_adjusted["y2"] = y2
        det_adjusted["crop_x1"] = crop_x1
        det_adjusted["crop_y1"] = crop_y1
        det_adjusted["crop_x2"] = crop_x2
        det_adjusted["crop_y2"] = crop_y2

        edge_info = compute_tab_edge_points_on_full_image(
            det_adjusted, preprocessed, center_x, center_y, crop_x1, crop_y1
        )

        tab_results.append({
            "detection": det_adjusted,
            "preprocessed": preprocessed,
            "edge_info": edge_info,
        })

    return tab_results


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------

def _cross_2d(vec_a, vec_b):
    return (vec_a[0] * vec_b[1]) - (vec_a[1] * vec_b[0])


def _ray_segment_intersection_distance(center_xy, direction_xy, seg_start_xy, seg_end_xy):
    """
    Return the distance t along the ray `center + t * direction` where it
    intersects the line segment, or None when there is no valid intersection.
    """
    origin = np.array(center_xy, dtype=np.float64)
    direction = np.array(direction_xy, dtype=np.float64)
    seg_start = np.array(seg_start_xy, dtype=np.float64)
    seg_end = np.array(seg_end_xy, dtype=np.float64)
    segment = seg_end - seg_start

    denom = _cross_2d(direction, segment)
    if abs(denom) < 1e-9:
        return None

    offset = seg_start - origin
    t = _cross_2d(offset, segment) / denom
    u = _cross_2d(offset, direction) / denom
    if t < 0.0 or u < 0.0 or u > 1.0:
        return None
    return float(t)


def build_radial_ray_segments(tab_results, center_x, center_y, base_radius, angle_step_deg):
    """
    Build 360-degree ray segments. Rays that intersect a tab contour outside
    the Hough circle are marked as hits; the rest stop at the Hough radius.
    """
    step_deg = max(0.1, float(angle_step_deg))
    step_deg = min(step_deg, 360.0)

    rays = []
    current_angle = 0.0
    while current_angle < 360.0 - 1e-9:
        angle_rad = math.radians(current_angle)
        direction = (math.cos(angle_rad), math.sin(angle_rad))
        best_distance = None
        best_tab_index = None

        for tab_index, tab in enumerate(tab_results):
            edge_info = tab.get("edge_info")
            if edge_info is None:
                continue

            contour_list = edge_info.get("contours") or []
            for contour in contour_list:
                contour_points = contour[:, 0, :]
                point_count = len(contour_points)
                if point_count < 2:
                    continue

                for idx in range(point_count):
                    pt1 = contour_points[idx]
                    pt2 = contour_points[(idx + 1) % point_count]
                    hit_distance = _ray_segment_intersection_distance(
                        (center_x, center_y), direction, pt1, pt2
                    )
                    if hit_distance is None:
                        continue
                    if best_distance is None or hit_distance > best_distance:
                        best_distance = hit_distance
                        best_tab_index = tab_index

        is_hit = best_distance is not None and best_distance > float(base_radius) + 1.0
        end_distance = float(best_distance) if is_hit else float(base_radius)
        end_point = (
            center_x + direction[0] * end_distance,
            center_y + direction[1] * end_distance,
        )

        rays.append({
            "angle_deg": current_angle,
            "distance": end_distance,
            "hit_distance": float(best_distance) if best_distance is not None else None,
            "is_hit": is_hit,
            "tab_index": best_tab_index if is_hit else None,
            "end_point": end_point,
        })
        current_angle += step_deg

    return rays


def draw_center_marker(image_bgr, center_xy):
    center = (int(round(center_xy[0])), int(round(center_xy[1])))
    cv2.circle(image_bgr, center, 10, WHITE_DRAW_COLOR, 2, cv2.LINE_AA)
    cv2.line(image_bgr, (center[0] - 20, center[1]), (center[0] + 20, center[1]), CENTER_MARKER_COLOR, 3, cv2.LINE_AA)
    cv2.line(image_bgr, (center[0], center[1] - 20), (center[0], center[1] + 20), CENTER_MARKER_COLOR, 3, cv2.LINE_AA)
    cv2.circle(image_bgr, center, 5, CENTER_MARKER_COLOR, -1, cv2.LINE_AA)


def draw_result(image_bgr, hough_result, tab_results, show_roi, show_circle,
                show_tab_edges, show_radial_lines,
                radial_step_deg=1.0, radial_segments=None):
    """Draw all detection results on the image."""
    output = image_bgr.copy()
    center_x = hough_result["center_x"]
    center_y = hough_result["center_y"]
    center = (int(round(center_x)), int(round(center_y)))
    radius = int(round(hough_result["radius"]))
    roi_x, roi_y, roi_w, roi_h = hough_result["debug"]["roi"]

    # Draw ROI rectangle
    if show_roi:
        cv2.rectangle(output, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)

    # Draw Hough circle and center
    if show_circle:
        cv2.circle(output, center, radius, HOUGH_CIRCLE_COLOR, 2)
    draw_center_marker(output, center)

    # Draw tab edge contours
    if show_tab_edges and tab_results:
        for tab in tab_results:
            edge_info = tab.get("edge_info")
            if edge_info is None:
                continue
            contour_list = edge_info.get("contours") or []
            if contour_list:
                cv2.drawContours(output, contour_list, -1, WHITE_DRAW_COLOR, 2)

    # Draw radial lines from center around the full 360-degree range
    if show_radial_lines and tab_results:
        ray_segments = radial_segments
        if ray_segments is None:
            ray_segments = build_radial_ray_segments(
                tab_results, center_x, center_y, hough_result["radius"], radial_step_deg
            )
        for ray in ray_segments:
            end_pt = (int(round(ray["end_point"][0])), int(round(ray["end_point"][1])))
            cv2.line(output, center, end_pt, RADIAL_LINE_COLOR, 1, cv2.LINE_AA)
            if ray["is_hit"]:
                cv2.circle(output, end_pt, 3, RADIAL_HIT_POINT_COLOR, -1)

        for tab in tab_results:
            edge_info = tab.get("edge_info")
            if edge_info is None:
                continue
            if edge_info["centroid"] is not None:
                cpt = (int(round(edge_info["centroid"][0])),
                       int(round(edge_info["centroid"][1])))
                cv2.drawMarker(output, cpt, WHITE_DRAW_COLOR, cv2.MARKER_CROSS, 10, 1)

    return output


def _tab_color(index):
    """Generate a distinct color for each tab index."""
    colors = [
        (0, 255, 0),     # Green
        (255, 100, 0),   # Blue-ish
        (0, 200, 255),   # Orange-yellow
        (255, 0, 200),   # Pink
        (100, 255, 100), # Light green
        (255, 200, 0),   # Cyan
        (0, 100, 255),   # Orange
        (200, 0, 255),   # Purple
        (255, 255, 0),   # Cyan
        (100, 0, 255),   # Red-purple
    ]
    return colors[index % len(colors)]


def draw_yolo_only(image_bgr, detections):
    """Draw only YOLO detection boxes on image."""
    output = image_bgr.copy()
    for i, det in enumerate(detections):
        color = _tab_color(i)
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["confidence"]
        cls_name = det.get("class_name", "tab")
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = "{} {:.0f}%".format(cls_name, conf * 100)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(output, (x1, y1 - label_size[1] - 8), (x1 + label_size[0] + 6, y1), color, -1)
        cv2.putText(output, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def draw_tab_edges_only(image_bgr, tab_results, center_x, center_y, hough_radius, radial_step_deg,
                        radial_segments=None, use_outer_profile=False):
    """Draw tab contours on a dark background for tuning or outer-profile review."""
    output = np.zeros_like(image_bgr)
    output[:] = (30, 30, 30)
    center = (int(round(center_x)), int(round(center_y)))

    for tab in tab_results:
        edge_info = tab.get("edge_info")
        if edge_info is None:
            continue
        contour_key = "contours" if use_outer_profile else "raw_contours"
        contour_list = edge_info.get(contour_key) or edge_info.get("contours") or []
        if contour_list:
            cv2.drawContours(output, contour_list, -1, WHITE_DRAW_COLOR, 2)

    if radial_segments:
        for ray in radial_segments:
            end_pt = (int(round(ray["end_point"][0])), int(round(ray["end_point"][1])))
            cv2.line(output, center, end_pt, RADIAL_LINE_COLOR, 1, cv2.LINE_AA)
            if ray["is_hit"]:
                cv2.circle(output, end_pt, 3, RADIAL_HIT_POINT_COLOR, -1)

    draw_center_marker(output, center)
    cv2.circle(output, center, int(round(hough_radius)), (80, 80, 80), 1, cv2.LINE_AA)
    return output


def draw_radial_lines_only(image_bgr, tab_results, center_x, center_y, hough_radius, radial_step_deg,
                           radial_segments=None):
    """Draw only the 360-degree radial rays from center to edge/circle."""
    output = image_bgr.copy()
    center = (int(round(center_x)), int(round(center_y)))
    ray_segments = radial_segments
    if ray_segments is None:
        ray_segments = build_radial_ray_segments(tab_results, center_x, center_y, hough_radius, radial_step_deg)

    for ray in ray_segments:
        end_pt = (int(round(ray["end_point"][0])), int(round(ray["end_point"][1])))
        cv2.line(output, center, end_pt, RADIAL_LINE_COLOR, 1, cv2.LINE_AA)
        if ray["is_hit"]:
            cv2.circle(output, end_pt, 3, RADIAL_HIT_POINT_COLOR, -1)

    draw_center_marker(output, center)
    cv2.circle(output, center, int(round(hough_radius)), (160, 160, 160), 1, cv2.LINE_AA)
    return output


def edge_to_bgr(edge_image):
    return cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)


def to_photo_image(image_bgr, max_width=None, max_height=None):
    preview = image_bgr.copy()
    height, width = preview.shape[:2]
    if max_width is not None and max_height is not None:
        scale = min(float(max_width) / float(width), float(max_height) / float(height), 1.0)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        preview = cv2.resize(preview, (new_width, new_height), interpolation=cv2.INTER_AREA)
    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    success, buffer_data = cv2.imencode(".png", preview_rgb)
    if not success:
        return None
    encoded = base64.b64encode(buffer_data.tobytes() if hasattr(buffer_data, "tobytes") else buffer_data.tostring())
    if not isinstance(encoded, str):
        encoded = encoded.decode("ascii")
    return tk.PhotoImage(data=encoded)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class RadialSignatureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Radial Signature")
        self.root.geometry("1780x920")

        # State variables
        self.input_dir_var = tk.StringVar(value=DEFAULT_IMAGE_DIR)
        self.image_var = tk.StringVar()
        self.center_var = tk.StringVar(value="Toa do tam Hough: -")
        self.radius_var = tk.StringVar(value="Ban kinh Hough: -")
        self.score_var = tk.StringVar(value="Hough score: -")
        self.index_var = tk.StringVar(value="Anh hien tai: -")
        self.status_var = tk.StringVar(value="Chon thu muc anh, load preset Hough va bam Run.")
        self.hough_preset_var = tk.StringVar(value="Preset Hough: dang dung mac dinh noi bo")
        self.radial_step_var = tk.StringVar(value="1")
        self.current_image_path = ""
        self.loaded_hough_preset_path = ""

        # Preview & display options
        self.preview_mode_var = tk.StringVar(value="result")
        self.show_roi_var = tk.IntVar(value=1)
        self.show_circle_var = tk.IntVar(value=1)
        self.show_tab_edges_var = tk.IntVar(value=1)
        self.show_radial_lines_var = tk.IntVar(value=0)
        self.auto_update_var = tk.IntVar(value=0)

        # Hough params
        self.hough_params = dict(DEFAULT_HOUGH_PARAMS)

        # YOLO model
        self.yolo_model = None

        # Preview canvas
        self.preview_image = None
        self.preview_canvas_image_id = None
        self.preview_canvas_text_id = None
        self.last_preview_source = None
        self.preview_resize_job = None
        self.signature_canvas = None
        self.signature_resize_job = None
        self.signature_popup = None
        self.signature_popup_canvas = None
        self.signature_popup_resize_job = None
        self.process_update_job = None

        # Scrollable panels
        self.left_canvas = None
        self.left_inner = None
        self.left_canvas_window = None
        self.params_canvas = None
        self.params_inner = None
        self.params_canvas_window = None
        self.active_scroll_canvas = None

        # Image list
        self.image_paths = []
        self.current_index = -1

        # Last processing results
        self.last_hough_result = None
        self.last_tab_results = None
        self.last_detections = None
        self.last_image_bgr = None
        self.last_scale_factor = 1.0
        self.radial_segments_cache = {}

        # Parameter controls
        self.param_vars = {}
        self.param_groups = [
            (
                "1. Tab Edge Preprocessing",
                [
                    ("tab_clahe_clip", "Tab CLAHE clip", 0.5, 5.0, 0.1, 2.0),
                    ("tab_blur_kernel", "Tab Blur kernel", 3, 15, 2, 5),
                    ("tab_blur_sigma", "Tab Blur sigma", 0.5, 5.0, 0.1, 1.0),
                    ("tab_canny_low", "Tab Canny low", 10, 200, 1, 50),
                    ("tab_canny_high", "Tab Canny high", 20, 255, 1, 150),
                    ("tab_close_kernel", "Tab Close kernel", 1, 15, 2, 3),
                    ("tab_close_iterations", "Tab Close iterations", 1, 5, 1, 1),
                ],
            ),
        ]

        self.build_ui()
        self.root.bind_all("<MouseWheel>", self.on_global_mousewheel)
        self.root.bind_all("<Button-4>", self.on_global_mousewheel)
        self.root.bind_all("<Button-5>", self.on_global_mousewheel)

    # -------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------

    def build_ui(self):
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        # Title
        ttk.Label(
            container,
            text="Radial Signature",
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        # Controls bar
        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 10))
        ttk.Entry(controls, textvariable=self.input_dir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(controls, text="Chon thu muc", command=self.select_input_dir).pack(side="left", padx=6)
        ttk.Button(controls, text="Run", command=self.run_detection).pack(side="left")
        ttk.Button(controls, text="Previous", command=self.show_previous_image).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="Next", command=self.show_next_image).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="Load preset Hough", command=self.load_hough_preset).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="Load preset anh mau", command=self.load_radial_preset).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="Save tao anh mau", command=self.save_radial_preset).pack(side="left", padx=(6, 0))

        # Info labels
        info_frame = ttk.Frame(container)
        info_frame.pack(fill="x", pady=(0, 6))
        ttk.Label(info_frame, textvariable=self.hough_preset_var, foreground="#555555").pack(anchor="w")

        current_image_frame = ttk.Frame(container)
        current_image_frame.pack(fill="x", pady=(0, 6))
        ttk.Label(current_image_frame, text="Anh dang xu ly:").pack(side="left")
        ttk.Entry(current_image_frame, textvariable=self.image_var, state="readonly").pack(side="left", fill="x", expand=True, padx=8)

        # Main content: PanedWindow with 3 sections
        content = ttk.Panedwindow(container, orient="horizontal")
        content.pack(fill="both", expand=True)

        # === LEFT PANEL: Results + Preview mode + Display options ===
        left_panel = ttk.Frame(content)
        left_pane = ttk.Panedwindow(left_panel, orient="horizontal")
        left_pane.pack(fill="both", expand=True)

        left_sidebar_frame = ttk.LabelFrame(left_pane, text="Ket qua va hien thi", padding=6)

        self.left_canvas = tk.Canvas(left_sidebar_frame, width=290, height=560, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_sidebar_frame, orient="vertical", command=self.left_canvas.yview)
        self.left_inner = ttk.Frame(self.left_canvas)
        self.left_inner.bind("<Configure>", lambda event: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all")))
        self.left_canvas_window = self.left_canvas.create_window((0, 0), window=self.left_inner, anchor="nw")
        self.left_canvas.configure(yscrollcommand=left_scroll.set)
        self.left_canvas.pack(side="left", fill="both", expand=True)
        left_scroll.pack(side="right", fill="y")
        self.left_canvas.bind("<Enter>", lambda event: self.bind_mousewheel(self.left_canvas))
        self.left_canvas.bind("<Leave>", lambda event: self.unbind_mousewheel(self.left_canvas))
        self.left_inner.bind("<Enter>", lambda event: self.bind_mousewheel(self.left_canvas))
        self.left_inner.bind("<Leave>", lambda event: self.unbind_mousewheel(self.left_canvas))
        self.left_canvas.bind("<Configure>", self.on_left_canvas_configure)

        # Results section
        result_frame = ttk.LabelFrame(self.left_inner, text="Ket qua", padding=8)
        result_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(result_frame, textvariable=self.index_var, font=("Segoe UI", 10), wraplength=260, justify="left").pack(anchor="w", pady=(2, 0))
        ttk.Label(result_frame, textvariable=self.center_var, font=("Segoe UI", 11, "bold"), wraplength=260, justify="left").pack(anchor="w", pady=(4, 0))
        ttk.Label(result_frame, textvariable=self.radius_var, font=("Segoe UI", 11, "bold"), wraplength=260, justify="left").pack(anchor="w", pady=(4, 0))
        ttk.Label(result_frame, textvariable=self.score_var, font=("Segoe UI", 10), wraplength=260, justify="left").pack(anchor="w", pady=(4, 0))

        # Preview mode
        mode_frame = ttk.LabelFrame(self.left_inner, text="Preview mode", padding=8)
        mode_frame.pack(fill="x", pady=(0, 10))
        for label, value in [
            ("Ket qua tong hop", "result"),
            ("1. Enhanced (ROI Hough)", "enhanced"),
            ("2. Blurred (ROI Hough)", "blurred"),
            ("3. Hough Edges", "edges"),
            ("4. Masked edges (Hough)", "masked_edges"),
            ("6. Tab edges (bien tai)", "tab_edges"),
            ("7. Radial lines chi", "radial_lines"),
        ]:
            ttk.Radiobutton(mode_frame, text=label, value=value, variable=self.preview_mode_var, command=self.on_preview_change).pack(anchor="w")

        # Display toggles
        toggle_frame = ttk.LabelFrame(self.left_inner, text="Tuy chon hien thi", padding=8)
        toggle_frame.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(toggle_frame, text="Hien ROI Hough", variable=self.show_roi_var, command=self.on_preview_change).pack(anchor="w")
        ttk.Checkbutton(toggle_frame, text="Hien vong tron Hough", variable=self.show_circle_var, command=self.on_preview_change).pack(anchor="w")
        ttk.Checkbutton(toggle_frame, text="Hien bien tai", variable=self.show_tab_edges_var, command=self.on_preview_change).pack(anchor="w")
        ttk.Checkbutton(toggle_frame, text="Hien radial lines", variable=self.show_radial_lines_var, command=self.on_preview_change).pack(anchor="w")
        ttk.Checkbutton(toggle_frame, text="Auto update khi doi tham so", variable=self.auto_update_var).pack(anchor="w", pady=(6, 0))

        radial_step_frame = ttk.LabelFrame(self.left_inner, text="Buoc goc radial", padding=8)
        radial_step_frame.pack(fill="x", pady=(0, 10))
        radial_step_row = ttk.Frame(radial_step_frame)
        radial_step_row.pack(fill="x")
        ttk.Label(radial_step_row, text="Moi tia cach nhau (deg):").pack(side="left")
        radial_step_entry = ttk.Entry(radial_step_row, textvariable=self.radial_step_var, width=8)
        radial_step_entry.pack(side="right")
        radial_step_entry.bind("<Return>", self.on_radial_step_changed)
        radial_step_entry.bind("<FocusOut>", self.on_radial_step_changed)
        ttk.Label(
            radial_step_frame,
            text="Vi du: 1 = 360 tia, 2 = 180 tia, 5 = 72 tia.",
            wraplength=240,
            justify="left",
            foreground="#555555",
        ).pack(anchor="w", pady=(6, 0))

        # === CENTER: Preview canvas ===
        preview_frame = ttk.LabelFrame(left_pane, text="Vung hien thi anh", padding=10)

        self.preview_canvas = tk.Canvas(preview_frame, width=760, height=560, highlightthickness=0, background="#202020")
        preview_scroll_y = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_canvas.yview)
        preview_scroll_x = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_canvas.xview)
        self.preview_canvas.configure(yscrollcommand=preview_scroll_y.set, xscrollcommand=preview_scroll_x.set)
        self.preview_canvas.pack(side="left", fill="both", expand=True)
        preview_scroll_y.pack(side="right", fill="y")
        preview_scroll_x.pack(side="bottom", fill="x")
        self.preview_canvas.bind("<Enter>", lambda event: self.bind_mousewheel(self.preview_canvas))
        self.preview_canvas.bind("<Leave>", lambda event: self.unbind_mousewheel(self.preview_canvas))
        self.preview_canvas.bind("<Configure>", self.on_preview_canvas_configure)
        self.preview_canvas_text_id = self.preview_canvas.create_text(
            30, 30, anchor="nw", fill="#f0f0f0",
            text="Preview se hien thi o day.\nLoad preset Hough va bam Run de bat dau.",
            font=("Segoe UI", 12, "bold"),
        )
        self.preview_canvas.configure(scrollregion=(0, 0, 1200, 900))

        left_pane.add(left_sidebar_frame, weight=1)
        left_pane.add(preview_frame, weight=4)

        # === RIGHT PANEL: Parameters ===
        right_panel = ttk.LabelFrame(content, text="Tham so", padding=10)

        self.params_canvas = tk.Canvas(right_panel, width=380, height=540, highlightthickness=0)
        params_scroll = ttk.Scrollbar(right_panel, orient="vertical", command=self.params_canvas.yview)
        self.params_inner = ttk.Frame(self.params_canvas)
        self.params_inner.bind("<Configure>", lambda event: self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all")))
        self.params_canvas_window = self.params_canvas.create_window((0, 0), window=self.params_inner, anchor="nw")
        self.params_canvas.configure(yscrollcommand=params_scroll.set)
        self.params_canvas.pack(side="left", fill="both", expand=True)
        params_scroll.pack(side="right", fill="y")
        self.params_canvas.bind("<Enter>", lambda event: self.bind_mousewheel(self.params_canvas))
        self.params_canvas.bind("<Leave>", lambda event: self.unbind_mousewheel(self.params_canvas))
        self.params_inner.bind("<Enter>", lambda event: self.bind_mousewheel(self.params_canvas))
        self.params_inner.bind("<Leave>", lambda event: self.unbind_mousewheel(self.params_canvas))
        self.params_canvas.bind("<Configure>", self.on_params_canvas_configure)

        for group_name, group_params in self.param_groups:
            group_frame = ttk.LabelFrame(self.params_inner, text=group_name, padding=8)
            group_frame.pack(fill="x", pady=(0, 10))
            for key, label, minimum, maximum, step, default in group_params:
                self.add_param_control(group_frame, key, label, minimum, maximum, step, default)

        signature_frame = ttk.LabelFrame(self.params_inner, text="Do thi R(alpha)", padding=8)
        signature_frame.pack(fill="x", pady=(0, 10))
        signature_header = ttk.Frame(signature_frame)
        signature_header.pack(fill="x", pady=(0, 6))
        ttk.Label(
            signature_header,
            text="R(alpha) tu tam Hough ra bien ngoai",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")
        ttk.Button(signature_header, text="Phong to", command=self.open_signature_popup).pack(side="right")
        ttk.Label(
            signature_frame,
            text="Doan vang = co tai | Doan xanh = khong co tai",
            foreground="#555555",
            wraplength=300,
            justify="left",
        ).pack(anchor="w", pady=(0, 6))
        self.signature_canvas = tk.Canvas(
            signature_frame,
            width=320,
            height=250,
            highlightthickness=1,
            highlightbackground="#d7dde5",
            background=SIGNATURE_BG_COLOR,
        )
        self.signature_canvas.pack(fill="x", expand=True)
        self.signature_canvas.bind("<Configure>", self.on_signature_canvas_configure)

        content.add(left_panel, weight=4)
        content.add(right_panel, weight=1)

        # Status bar
        status_bar = ttk.Frame(container)
        status_bar.pack(fill="x", pady=(8, 0))
        ttk.Label(status_bar, textvariable=self.status_var, font=("Segoe UI", 10), foreground="#333333").pack(anchor="w")

    def add_param_control(self, parent, key, label, minimum, maximum, step, default):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=4)
        frame.columnconfigure(0, weight=1)

        var = tk.DoubleVar(value=default)
        self.param_vars[key] = var

        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        value_label = ttk.Label(frame, text="{:.2f}".format(default), width=8, anchor="e")
        value_label.grid(row=0, column=1, sticky="e", padx=(8, 0))

        scale = tk.Scale(
            frame,
            from_=minimum,
            to=maximum,
            resolution=step,
            orient="horizontal",
            showvalue=0,
            variable=var,
            command=lambda value, name=key, widget=value_label: self.on_scale_change(name, value, widget),
            length=260,
        )
        scale.grid(row=1, column=0, columnspan=2, sticky="ew")

    # -------------------------------------------------------------------
    # Scrolling
    # -------------------------------------------------------------------

    def bind_mousewheel(self, canvas):
        self.active_scroll_canvas = canvas

    def unbind_mousewheel(self, canvas):
        if self.active_scroll_canvas is canvas:
            self.active_scroll_canvas = None

    def on_global_mousewheel(self, event):
        if self.active_scroll_canvas is None:
            return
        self.on_mousewheel(self.active_scroll_canvas, event)

    def on_mousewheel(self, canvas, event):
        if hasattr(event, "delta") and event.delta:
            step = -1 if event.delta > 0 else 1
            canvas.yview_scroll(step, "units")
        elif getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")

    def on_left_canvas_configure(self, event):
        if self.left_canvas_window is not None:
            self.left_canvas.itemconfigure(self.left_canvas_window, width=max(1, event.width - 2))

    def on_params_canvas_configure(self, event):
        if self.params_canvas_window is not None:
            self.params_canvas.itemconfigure(self.params_canvas_window, width=max(1, event.width - 2))

    def on_preview_canvas_configure(self, event):
        if self.preview_resize_job is not None:
            self.root.after_cancel(self.preview_resize_job)
        self.preview_resize_job = self.root.after(120, self.refresh_preview_canvas)

    def on_signature_canvas_configure(self, event):
        if self.signature_resize_job is not None:
            self.root.after_cancel(self.signature_resize_job)
        self.signature_resize_job = self.root.after(120, self.refresh_signature_plot)

    def on_signature_popup_configure(self, event):
        if self.signature_popup_resize_job is not None:
            self.root.after_cancel(self.signature_popup_resize_job)
        self.signature_popup_resize_job = self.root.after(120, self.refresh_signature_popup_plot)

    def open_signature_popup(self):
        if self.signature_popup is not None and self.signature_popup.winfo_exists():
            self.signature_popup.deiconify()
            self.signature_popup.lift()
            self.signature_popup.focus_force()
            self.refresh_signature_popup_plot()
            return

        popup = tk.Toplevel(self.root)
        popup.title("Do thi R(alpha)")
        popup.geometry("980x720")
        popup.minsize(720, 520)
        popup.protocol("WM_DELETE_WINDOW", self.close_signature_popup)

        popup_container = ttk.Frame(popup, padding=12)
        popup_container.pack(fill="both", expand=True)
        ttk.Label(
            popup_container,
            text="R(alpha) tu tam Hough ra bien ngoai",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            popup_container,
            text="Doan vang = co tai | Doan xanh = khong co tai",
            foreground="#555555",
        ).pack(anchor="w", pady=(2, 8))

        self.signature_popup_canvas = tk.Canvas(
            popup_container,
            highlightthickness=1,
            highlightbackground="#d7dde5",
            background=SIGNATURE_BG_COLOR,
        )
        self.signature_popup_canvas.pack(fill="both", expand=True)
        self.signature_popup_canvas.bind("<Configure>", self.on_signature_popup_configure)

        self.signature_popup = popup
        self.refresh_signature_popup_plot()

    def close_signature_popup(self):
        if self.signature_popup_resize_job is not None:
            self.root.after_cancel(self.signature_popup_resize_job)
            self.signature_popup_resize_job = None
        if self.signature_popup is not None and self.signature_popup.winfo_exists():
            self.signature_popup.destroy()
        self.signature_popup = None
        self.signature_popup_canvas = None

    def _draw_signature_placeholder(self, canvas, message):
        canvas_width = max(1, canvas.winfo_width())
        canvas_height = max(1, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_rectangle(
            0, 0, canvas_width, canvas_height,
            fill=SIGNATURE_BG_COLOR,
            outline=SIGNATURE_BG_COLOR,
        )
        canvas.create_text(
            canvas_width / 2.0,
            canvas_height / 2.0,
            text=message,
            fill=SIGNATURE_TEXT_COLOR,
            font=("Segoe UI", 11, "bold"),
            justify="center",
        )

    def draw_signature_plot(self, canvas):
        if canvas is None or not canvas.winfo_exists():
            return

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width <= 1:
            canvas_width = 320
        if canvas_height <= 1:
            canvas_height = 250

        canvas.delete("all")
        canvas.create_rectangle(
            0, 0, canvas_width, canvas_height,
            fill=SIGNATURE_BG_COLOR,
            outline=SIGNATURE_BG_COLOR,
        )

        if self.last_hough_result is None:
            self._draw_signature_placeholder(canvas, "Do thi R(alpha) se hien thi o day\nsau khi bam Run.")
            return

        radial_step_deg = self.get_radial_step_deg()
        ray_segments = self.get_cached_radial_segments(radial_step_deg)
        base_radius = float(self.last_hough_result["radius"])
        if not ray_segments:
            self._draw_signature_placeholder(canvas, "Khong co du lieu radial de ve do thi.")
            return

        left_margin = 54
        right_margin = 14
        top_margin = 18
        bottom_margin = 40
        plot_left = left_margin
        plot_top = top_margin
        plot_right = max(plot_left + 40, canvas_width - right_margin)
        plot_bottom = max(plot_top + 40, canvas_height - bottom_margin)
        plot_width = max(1.0, float(plot_right - plot_left))
        plot_height = max(1.0, float(plot_bottom - plot_top))

        canvas.create_rectangle(
            plot_left,
            plot_top,
            plot_right,
            plot_bottom,
            fill=SIGNATURE_PLOT_BG_COLOR,
            outline=SIGNATURE_GRID_COLOR,
            width=1,
        )

        distances = [float(ray.get("distance", base_radius)) for ray in ray_segments]
        peak_radius = max([base_radius] + distances)
        bump_height = max(0.0, peak_radius - base_radius)
        if bump_height < 1.0:
            y_min = max(0.0, base_radius - 12.0)
            y_max = base_radius + 12.0
        else:
            y_min = max(0.0, base_radius - max(8.0, bump_height * 0.35))
            y_max = peak_radius + max(8.0, bump_height * 0.20)

        if y_max <= y_min:
            y_max = y_min + 1.0

        def x_from_angle(angle_deg):
            clamped_angle = min(max(float(angle_deg), 0.0), 360.0)
            return plot_left + (clamped_angle / 360.0) * plot_width

        def y_from_radius(radius_value):
            normalized = (float(radius_value) - y_min) / (y_max - y_min)
            normalized = min(max(normalized, 0.0), 1.0)
            return plot_bottom - normalized * plot_height

        for grid_index in range(6):
            fraction = float(grid_index) / 5.0
            y_pos = plot_top + fraction * plot_height
            value = y_max - fraction * (y_max - y_min)
            canvas.create_line(
                plot_left,
                y_pos,
                plot_right,
                y_pos,
                fill=SIGNATURE_GRID_COLOR,
                width=1,
            )
            canvas.create_text(
                plot_left - 8,
                y_pos,
                anchor="e",
                text="{:.0f}".format(value),
                fill=SIGNATURE_AXIS_COLOR,
                font=("Segoe UI", 8),
            )

        for angle_tick in range(0, 361, 45):
            x_pos = x_from_angle(angle_tick)
            canvas.create_line(
                x_pos,
                plot_top,
                x_pos,
                plot_bottom,
                fill=SIGNATURE_GRID_COLOR,
                width=1,
            )
            canvas.create_text(
                x_pos,
                plot_bottom + 18,
                anchor="n",
                text=str(angle_tick),
                fill=SIGNATURE_AXIS_COLOR,
                font=("Segoe UI", 8),
            )

        base_line_y = y_from_radius(base_radius)
        canvas.create_line(
            plot_left,
            base_line_y,
            plot_right,
            base_line_y,
            fill=SIGNATURE_BASELINE_COLOR,
            width=2,
            dash=(6, 4),
        )
        canvas.create_text(
            plot_right - 6,
            max(plot_top + 10, base_line_y - 8),
            anchor="e",
            text="r = {:.1f}px".format(base_radius),
            fill=SIGNATURE_BASELINE_COLOR,
            font=("Segoe UI", 9, "bold"),
        )

        plot_samples = list(ray_segments)
        plot_samples.append({
            "angle_deg": 360.0,
            "distance": ray_segments[0]["distance"],
            "is_hit": ray_segments[0]["is_hit"],
        })

        hit_points = []
        for index in range(len(plot_samples) - 1):
            current_ray = plot_samples[index]
            next_ray = plot_samples[index + 1]
            x1 = x_from_angle(current_ray["angle_deg"])
            y1 = y_from_radius(current_ray["distance"])
            x2 = x_from_angle(next_ray["angle_deg"])
            y2 = y_from_radius(next_ray["distance"])
            is_hit_segment = bool(current_ray.get("is_hit"))
            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=SIGNATURE_HIT_COLOR if is_hit_segment else SIGNATURE_FLAT_COLOR,
                width=3 if is_hit_segment else 2,
                smooth=1,
            )
            if is_hit_segment:
                hit_points.append((x1, y1))

        for x_pos, y_pos in hit_points:
            canvas.create_oval(
                x_pos - 2,
                y_pos - 2,
                x_pos + 2,
                y_pos + 2,
                outline=SIGNATURE_HIT_COLOR,
                fill=SIGNATURE_HIT_COLOR,
            )

        canvas.create_text(
            (plot_left + plot_right) / 2.0,
            canvas_height - 12,
            anchor="s",
            text="alpha (deg)",
            fill=SIGNATURE_TEXT_COLOR,
            font=("Segoe UI", 9, "bold"),
        )
        canvas.create_text(
            8,
            plot_top - 2,
            anchor="w",
            text="R(alpha) [px]",
            fill=SIGNATURE_TEXT_COLOR,
            font=("Segoe UI", 9, "bold"),
        )

    # -------------------------------------------------------------------
    # Preview rendering
    # -------------------------------------------------------------------

    def refresh_preview_canvas(self):
        self.preview_resize_job = None
        if self.last_preview_source is None:
            return
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        if canvas_width <= 1:
            canvas_width = 760
        if canvas_height <= 1:
            canvas_height = 560
        preview = to_photo_image(
            self.last_preview_source,
            max_width=max(200, canvas_width - 24),
            max_height=max(200, canvas_height - 24),
        )
        self.update_preview_canvas(preview)

    def refresh_signature_plot(self):
        self.signature_resize_job = None
        self.draw_signature_plot(self.signature_canvas)
        self.refresh_signature_popup_plot()

    def refresh_signature_popup_plot(self):
        self.signature_popup_resize_job = None
        if self.signature_popup_canvas is None:
            return
        self.draw_signature_plot(self.signature_popup_canvas)

    def update_preview_canvas(self, preview):
        self.preview_canvas.delete("all")
        if preview is None:
            self.preview_canvas_text_id = self.preview_canvas.create_text(
                30, 30, anchor="nw", fill="#f0f0f0",
                text="Khong tao duoc anh preview.",
                font=("Segoe UI", 12, "bold"),
            )
            self.preview_canvas.configure(scrollregion=(0, 0, 1200, 900))
            self.preview_image = None
            self.preview_canvas_image_id = None
            return

        self.preview_image = preview
        canvas_width = max(1, self.preview_canvas.winfo_width())
        canvas_height = max(1, self.preview_canvas.winfo_height())
        offset_x = max(0, int((canvas_width - self.preview_image.width()) / 2.0))
        offset_y = max(0, int((canvas_height - self.preview_image.height()) / 2.0))
        self.preview_canvas_image_id = self.preview_canvas.create_image(offset_x, offset_y, anchor="nw", image=self.preview_image)
        scroll_width = max(canvas_width, offset_x + self.preview_image.width())
        scroll_height = max(canvas_height, offset_y + self.preview_image.height())
        self.preview_canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))
        self.preview_canvas.xview_moveto(0.0)
        self.preview_canvas.yview_moveto(0.0)

    # -------------------------------------------------------------------
    # Parameter & preview change handlers
    # -------------------------------------------------------------------

    def on_scale_change(self, key, value, label_widget):
        numeric_value = float(value)
        if key in ("blur_kernel", "tab_blur_kernel", "tab_canny_low", "tab_canny_high",
                    "tab_close_kernel", "tab_close_iterations"):
            label_widget.configure(text=str(int(round(numeric_value))))
        else:
            label_widget.configure(text="{:.2f}".format(numeric_value))
        self.on_param_change()

    def on_param_change(self):
        if self.auto_update_var.get() and self.current_index >= 0:
            if self.process_update_job is not None:
                self.root.after_cancel(self.process_update_job)
            self.process_update_job = self.root.after(180, self.process_current_image)

    def on_preview_change(self):
        """Called when preview mode or display toggles change - re-render without re-processing."""
        if self.last_image_bgr is not None and self.last_hough_result is not None:
            self.render_preview()

    def get_radial_step_deg(self):
        try:
            value = float(self.radial_step_var.get())
        except (TypeError, ValueError):
            value = 1.0
        value = min(max(value, 0.1), 360.0)
        self.radial_step_var.set("{:g}".format(value))
        return value

    def on_radial_step_changed(self, event=None):
        self.get_radial_step_deg()
        if self.last_image_bgr is not None and self.last_hough_result is not None:
            self.render_preview()

    def get_cached_radial_segments(self, radial_step_deg):
        if self.last_hough_result is None:
            return []
        step_key = round(float(radial_step_deg), 4)
        cache_key = (
            step_key,
            round(float(self.last_hough_result["center_x"]), 3),
            round(float(self.last_hough_result["center_y"]), 3),
            round(float(self.last_hough_result["radius"]), 3),
            len(self.last_tab_results or []),
        )
        if cache_key not in self.radial_segments_cache:
            self.radial_segments_cache[cache_key] = build_radial_ray_segments(
                self.last_tab_results or [],
                self.last_hough_result["center_x"],
                self.last_hough_result["center_y"],
                self.last_hough_result["radius"],
                radial_step_deg,
            )
        return self.radial_segments_cache[cache_key]

    def get_params(self):
        params = {}
        for key, var in self.param_vars.items():
            value = float(var.get())
            if key in ("blur_kernel", "tab_blur_kernel", "tab_canny_low", "tab_canny_high",
                        "tab_close_kernel", "tab_close_iterations"):
                value = int(round(value))
            params[key] = value
        return params

    # -------------------------------------------------------------------
    # Preset management
    # -------------------------------------------------------------------

    def ensure_template_presets_dir(self):
        if not os.path.isdir(TEMPLATE_SETTING_PRESETS_DIR):
            os.makedirs(TEMPLATE_SETTING_PRESETS_DIR)
        return TEMPLATE_SETTING_PRESETS_DIR

    def _build_template_preview_params_payload(self):
        params = self.get_params()
        return {
            "yolo_conf_threshold": float(params.get("yolo_conf", 0.25)),
            "tab_clahe_clip": float(params.get("tab_clahe_clip", 2.0)),
            "tab_blur_kernel": int(round(params.get("tab_blur_kernel", 5))),
            "tab_blur_sigma": float(params.get("tab_blur_sigma", 1.0)),
            "tab_canny_low": int(round(params.get("tab_canny_low", 50))),
            "tab_canny_high": int(round(params.get("tab_canny_high", 150))),
            "tab_close_kernel": int(round(params.get("tab_close_kernel", 3))),
            "tab_close_iterations": int(round(params.get("tab_close_iterations", 1))),
            "radial_step_deg": float(self.get_radial_step_deg()),
            "box_padding_ratio": float(TAB_BOX_PADDING_RATIO),
            "box_padding_min_px": int(TAB_BOX_PADDING_MIN_PX),
            "min_contour_area_ratio": float(TAB_MIN_CONTOUR_AREA_RATIO),
            "contour_keep_distance_ratio": float(TAB_CONTOUR_KEEP_DISTANCE_RATIO),
        }

    def _build_template_preview_views_payload(self):
        return {
            "auto_update": int(self.auto_update_var.get()),
            "show_axes": 1,
            "show_hough": int(bool(self.show_circle_var.get() or self.show_roi_var.get())),
            "show_yolo": 1,
            "show_tab_edges": int(self.show_tab_edges_var.get()),
            "show_radial": int(self.show_radial_lines_var.get()),
            "show_centers": 1,
        }

    def _build_export_hough_result(self):
        if self.last_hough_result is None:
            return None
        scale_factor = float(self.last_scale_factor) if self.last_scale_factor else 1.0
        if scale_factor <= 0.0:
            scale_factor = 1.0
        if abs(scale_factor - 1.0) < 1e-9:
            return self.last_hough_result
        return scale_hough_result(self.last_hough_result, 1.0 / scale_factor)

    def _build_export_radial_rays(self, radial_step_deg):
        scale_factor = float(self.last_scale_factor) if self.last_scale_factor else 1.0
        if scale_factor <= 0.0:
            scale_factor = 1.0

        export_rays = []
        for ray in self.get_cached_radial_segments(radial_step_deg):
            hit_distance = ray.get("hit_distance")
            end_point = ray.get("end_point")
            if end_point is not None and len(end_point) >= 2:
                end_point = [
                    float(end_point[0]) / scale_factor,
                    float(end_point[1]) / scale_factor,
                ]
            export_rays.append({
                "angle_deg": float(ray.get("angle_deg", 0.0)),
                "distance": float(ray.get("distance", 0.0)) / scale_factor,
                "hit_distance": (float(hit_distance) / scale_factor) if hit_distance is not None else None,
                "is_hit": bool(ray.get("is_hit")),
                "tab_index": int(ray["tab_index"]) if ray.get("tab_index") is not None else None,
                "end_point": end_point,
            })
        return export_rays

    def _build_main_gui_config_payload(self, export_hough_result):
        roi = export_hough_result.get("debug", {}).get("roi") if export_hough_result else None
        if roi is not None and len(roi) == 4:
            min_dimension = max(1, min(int(roi[2]), int(roi[3])))
        else:
            min_dimension = max(1, int(round(float(export_hough_result["radius"]) * 2.0)))

        min_radius_ratio = float(self.hough_params.get("min_radius_ratio", DEFAULT_HOUGH_PARAMS["min_radius_ratio"]))
        max_radius_ratio = float(self.hough_params.get("max_radius_ratio", DEFAULT_HOUGH_PARAMS["max_radius_ratio"]))
        min_dist_ratio = float(self.hough_params.get("min_dist_ratio", DEFAULT_HOUGH_PARAMS["min_dist_ratio"]))

        min_radius = max(1, int(round(min_dimension * min_radius_ratio)))
        max_radius = max(min_radius + 1, int(round(min_dimension * max_radius_ratio)))
        min_dist = max(1, int(round(max(20.0, min_dimension * min_dist_ratio))))

        return {
            "clahe": {
                "use": True,
                "clipLimit": float(self.hough_params.get("clahe_clip", DEFAULT_HOUGH_PARAMS["clahe_clip"])),
                "tileGridSize": [8, 8],
            },
            "preprocess": {
                "gaussian_kernel": int(ensure_odd(self.hough_params.get("blur_kernel", DEFAULT_HOUGH_PARAMS["blur_kernel"]))),
                "gaussian_sigma": float(self.hough_params.get("blur_sigma", DEFAULT_HOUGH_PARAMS["blur_sigma"])),
            },
            "hough": {
                "dp": float(self.hough_params.get("hough_dp", DEFAULT_HOUGH_PARAMS["hough_dp"])),
                "param1": int(round(self.hough_params.get("hough_param1", DEFAULT_HOUGH_PARAMS["hough_param1"]))),
                "param2": int(round(self.hough_params.get("hough_param2", DEFAULT_HOUGH_PARAMS["hough_param2"]))),
                "minRadius": min_radius,
                "maxRadius": max_radius,
                "minDist": min_dist,
            },
            "signature": {
                "smooth_kernel": 9,
                "invalid_fill": "nearest",
            },
            "angle": {
                "reliability_mse_threshold": 0.45,
            },
        }

    def build_radial_preset_payload(self):
        if not self.current_image_path or self.last_hough_result is None:
            raise ValueError("Hay chon anh mau va bam Run truoc khi luu preset.")

        radial_step_deg = self.get_radial_step_deg()
        export_hough_result = self._build_export_hough_result()
        export_rays = self._build_export_radial_rays(radial_step_deg)
        if not export_rays:
            raise ValueError("Khong co du lieu radial signature de luu.")

        hit_ray_count = sum(1 for ray in export_rays if ray.get("is_hit"))
        if hit_ray_count <= 0:
            raise ValueError("Chua co tia radial cat trung tai. Hay kiem tra YOLO va tien xu ly truoc khi luu.")

        radial_signature = smooth_circular_profile(
            [float(ray.get("distance", 0.0)) for ray in export_rays],
            9,
        )
        if not radial_signature:
            raise ValueError("R mau rong, khong the luu preset.")

        preview_params = self._build_template_preview_params_payload()
        preview_views = self._build_template_preview_views_payload()
        center_xy = [
            int(round(float(export_hough_result["center_x"]))),
            int(round(float(export_hough_result["center_y"]))),
        ]
        hough_circle = center_xy + [int(round(float(export_hough_result["radius"])))]

        return {
            "schema_version": "template_setting_v2",
            "template_path": self.current_image_path,
            "config": self._build_main_gui_config_payload(export_hough_result),
            "template_setting": {
                "angle_deg": 0.0,
                "center_xy": center_xy,
                "angle_step_input": "{:g}".format(radial_step_deg),
                "preview_preprocess": False,
            },
            "template_preview": {
                "params": preview_params,
                "views": preview_views,
            },
            "template_reference": {
                "schema_version": "template_reference_v1",
                "template_path": self.current_image_path,
                "angle_deg": 0.0,
                "center_xy": center_xy,
                "preview_params": preview_params,
                "hough_circle": hough_circle,
                "tab_count": int(len(self.last_tab_results or [])),
                "hit_ray_count": int(hit_ray_count),
                "ray_count": int(len(export_rays)),
                "radial_signature": [float(value) for value in radial_signature],
                "rays": export_rays,
                "warning": None,
            },
            "params": self.get_params(),
            "preview_mode": self.preview_mode_var.get(),
            "show_roi": int(self.show_roi_var.get()),
            "show_circle": int(self.show_circle_var.get()),
            "show_tab_edges": int(self.show_tab_edges_var.get()),
            "show_radial_lines": int(self.show_radial_lines_var.get()),
            "auto_update": int(self.auto_update_var.get()),
            "radial_step_deg": radial_step_deg,
            "source_hough_params": dict(self.hough_params),
            "source_hough_preset_path": self.loaded_hough_preset_path,
            "export_scale_factor": float(self.last_scale_factor if self.last_scale_factor else 1.0),
            "export_source_app": "pipeline/tao_anh_mau.py",
        }

    def apply_radial_preset_payload(self, payload):
        params = payload.get("params", {})
        if not isinstance(params, dict):
            params = {}
        if not params:
            template_preview = payload.get("template_preview", {})
            template_params = template_preview.get("params", {}) if isinstance(template_preview, dict) else {}
            preview_key_map = {
                "tab_clahe_clip": "tab_clahe_clip",
                "tab_blur_kernel": "tab_blur_kernel",
                "tab_blur_sigma": "tab_blur_sigma",
                "tab_canny_low": "tab_canny_low",
                "tab_canny_high": "tab_canny_high",
                "tab_close_kernel": "tab_close_kernel",
                "tab_close_iterations": "tab_close_iterations",
            }
            for source_key, target_key in preview_key_map.items():
                if source_key in template_params:
                    params[target_key] = template_params[source_key]

        for key, value in params.items():
            if key in self.param_vars:
                self.param_vars[key].set(value)

        preview_mode = payload.get("preview_mode")
        valid_modes = ("result", "enhanced", "blurred", "edges", "masked_edges",
                       "tab_edges", "radial_lines")
        if preview_mode in valid_modes:
            self.preview_mode_var.set(preview_mode)

        self.show_roi_var.set(int(payload.get("show_roi", self.show_roi_var.get())))
        self.show_circle_var.set(int(payload.get("show_circle", self.show_circle_var.get())))
        self.show_tab_edges_var.set(int(payload.get("show_tab_edges", self.show_tab_edges_var.get())))
        self.show_radial_lines_var.set(int(payload.get("show_radial_lines", self.show_radial_lines_var.get())))
        self.auto_update_var.set(int(payload.get("auto_update", self.auto_update_var.get())))

        radial_step_value = payload.get("radial_step_deg")
        if radial_step_value is None:
            template_preview = payload.get("template_preview", {})
            template_params = template_preview.get("params", {}) if isinstance(template_preview, dict) else {}
            radial_step_value = template_params.get("radial_step_deg")
        if radial_step_value is not None:
            self.radial_step_var.set("{:g}".format(float(radial_step_value)))

        source_hough_params = payload.get("source_hough_params")
        if isinstance(source_hough_params, dict):
            merged_hough_params = dict(DEFAULT_HOUGH_PARAMS)
            for key, value in source_hough_params.items():
                if key in merged_hough_params:
                    merged_hough_params[key] = value
            self.hough_params = merged_hough_params
            self.loaded_hough_preset_path = str(payload.get("source_hough_preset_path") or "")
            if self.loaded_hough_preset_path:
                self.hough_preset_var.set("Preset Hough: {}".format(os.path.basename(self.loaded_hough_preset_path)))
            else:
                self.hough_preset_var.set("Preset Hough: da nap tu preset anh mau")

    def save_radial_preset(self):
        self.ensure_template_presets_dir()
        initial_name = "template_setting.json"
        if self.current_image_path:
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            initial_name = base_name + "_setting.json"

        preset_path = filedialog.asksaveasfilename(
            title="Luu preset anh mau cho GUI chinh",
            initialdir=TEMPLATE_SETTING_PRESETS_DIR,
            initialfile=initial_name,
            defaultextension=".json",
            filetypes=[("Preset JSON", "*.json"), ("All files", "*.*")],
        )
        if not preset_path:
            return

        try:
            payload = self.build_radial_preset_payload()
        except ValueError as error:
            self.status_var.set("Chua the luu preset anh mau.")
            messagebox.showwarning("Chua the luu preset anh mau", str(error))
            return

        with open(preset_path, "w", encoding="utf-8") as preset_file:
            json.dump(payload, preset_file, indent=2, ensure_ascii=False)
        self.status_var.set("Da luu preset anh mau: {}".format(os.path.basename(preset_path)))

    def load_radial_preset(self):
        initial_dir = TEMPLATE_SETTING_PRESETS_DIR
        if not os.path.isdir(initial_dir):
            initial_dir = LEGACY_RADIAL_PRESETS_DIR if os.path.isdir(LEGACY_RADIAL_PRESETS_DIR) else BASE_DIR

        preset_path = filedialog.askopenfilename(
            title="Nap preset anh mau / radial signature",
            initialdir=initial_dir,
            filetypes=[("Preset JSON", "*.json"), ("All files", "*.*")],
        )
        if not preset_path:
            return
        with open(preset_path, "r", encoding="utf-8") as preset_file:
            payload = json.load(preset_file)
        self.apply_radial_preset_payload(payload)
        self.status_var.set("Da nap preset anh mau: {}".format(os.path.basename(preset_path)))
        if self.current_index >= 0:
            self.process_current_image()

    def load_hough_preset(self):
        initial_dir = HOUGH_PRESETS_DIR if os.path.isdir(HOUGH_PRESETS_DIR) else BASE_DIR
        preset_path = filedialog.askopenfilename(
            title="Nap preset Hough tu Circle Center",
            initialdir=initial_dir,
            filetypes=[("Preset JSON", "*.json"), ("All files", "*.*")],
        )
        if not preset_path:
            return
        with open(preset_path, "r") as preset_file:
            payload = json.load(preset_file)
        params = payload.get("params", {})
        merged_hough_params = dict(DEFAULT_HOUGH_PARAMS)
        for key, value in params.items():
            if key in merged_hough_params:
                merged_hough_params[key] = value
        self.hough_params = merged_hough_params
        self.loaded_hough_preset_path = preset_path
        self.hough_preset_var.set("Preset Hough: {}".format(os.path.basename(preset_path)))
        self.status_var.set("Da nap preset Hough. Bam Run de xu ly.")
        if self.current_index >= 0:
            self.process_current_image()

    # -------------------------------------------------------------------
    # Image navigation
    # -------------------------------------------------------------------

    def select_input_dir(self):
        initial_dir = self.input_dir_var.get().strip()
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = DEFAULT_IMAGE_DIR if os.path.isdir(DEFAULT_IMAGE_DIR) else os.getcwd()
        selected = filedialog.askdirectory(title="Chon thu muc anh dau vao", initialdir=initial_dir)
        if selected:
            self.input_dir_var.set(selected)
            self.status_var.set("Da chon thu muc anh. Bam Run de xu ly.")

    def load_images_from_dir(self):
        input_dir = self.input_dir_var.get().strip()
        if not input_dir:
            raise ValueError("Hay chon thu muc anh dau vao truoc.")
        if not os.path.isdir(input_dir):
            raise ValueError("Khong ton tai thu muc:\n{}".format(input_dir))
        image_paths = []
        for name in sorted(os.listdir(input_dir), key=natural_sort_key):
            path = os.path.join(input_dir, name)
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)
        if not image_paths:
            raise ValueError("Khong tim thay anh hop le trong thu muc da chon.")
        return image_paths

    def run_detection(self):
        if cv2 is None or np is None:
            messagebox.showerror(
                "Thieu thu vien",
                "Can cai dat opencv-python va numpy truoc khi chay.\n\nLenh goi y:\npip install opencv-python numpy",
            )
            return

        # Load YOLO model if not loaded
        if self.yolo_model is None:
            try:
                self.status_var.set("Dang tai model YOLO tu {}...".format(os.path.basename(YOLO_MODEL_PATH)))
                self.root.update_idletasks()
                self.yolo_model = load_yolo_model(YOLO_MODEL_PATH)
                self.status_var.set("Da tai model YOLO thanh cong.")
            except Exception as error:
                messagebox.showerror("Loi tai model YOLO", str(error))
                self.status_var.set("Khong tai duoc model YOLO.")
                return

        try:
            self.image_paths = self.load_images_from_dir()
        except Exception as error:
            messagebox.showerror("Loi thu muc anh", str(error))
            self.status_var.set("Khong tai duoc danh sach anh.")
            return

        self.current_index = 0
        self.process_current_image()

    def show_next_image(self):
        if not self.image_paths:
            messagebox.showwarning("Chua co danh sach anh", "Hay bam Run de tai thu muc anh truoc.")
            return
        if self.current_index >= len(self.image_paths) - 1:
            self.status_var.set("Da den anh cuoi cung trong thu muc.")
            return
        self.current_index += 1
        self.process_current_image()

    def show_previous_image(self):
        if not self.image_paths:
            messagebox.showwarning("Chua co danh sach anh", "Hay bam Run de tai thu muc anh truoc.")
            return
        if self.current_index <= 0:
            self.status_var.set("Dang o anh dau tien trong thu muc.")
            return
        self.current_index -= 1
        self.process_current_image()

    # -------------------------------------------------------------------
    # Core processing
    # -------------------------------------------------------------------

    def process_current_image(self):
        if self.process_update_job is not None:
            self.root.after_cancel(self.process_update_job)
            self.process_update_job = None
        if cv2 is None or np is None:
            return
        if not self.image_paths or self.current_index < 0 or self.current_index >= len(self.image_paths):
            self.status_var.set("Khong co anh nao de xu ly.")
            return

        image_path = self.image_paths[self.current_index]
        self.current_image_path = image_path
        self.image_var.set(image_path)
        original_image_bgr = cv2.imread(image_path)
        if original_image_bgr is None:
            messagebox.showerror("Loi doc anh", "Khong doc duoc anh:\n{}".format(image_path))
            self.status_var.set("Khong doc duoc anh.")
            return

        image_bgr, scale_factor = upscale_small_image(original_image_bgr)
        self.last_scale_factor = scale_factor

        self.status_var.set("Buoc 1/3: Dang tim tam Hough...")
        self.root.update_idletasks()

        # Step 1: Hough circle detection
        try:
            hough_result = detect_hough_reference_circle(original_image_bgr, self.hough_params)
        except Exception as error:
            self.status_var.set("Loi Hough: {}".format(str(error)))
            messagebox.showerror("Loi Hough", str(error))
            return
        hough_result = scale_hough_result(hough_result, scale_factor)

        center_x = hough_result["center_x"]
        center_y = hough_result["center_y"]

        self.status_var.set("Buoc 2/3: Dang detect tai bang YOLO...")
        self.root.update_idletasks()

        # Step 2: YOLO tab detection
        params = self.get_params()
        yolo_conf = float(params.get("yolo_conf", 0.25))
        detections = []
        if self.yolo_model is not None:
            try:
                detections = detect_tabs_yolo(image_bgr, self.yolo_model, conf_threshold=yolo_conf)
            except Exception as error:
                self.status_var.set("Loi YOLO: {}".format(str(error)))
                detections = []

        self.status_var.set("Buoc 3/3: Dang xu ly bien tai va radial lines...")
        self.root.update_idletasks()

        # Step 3: Tab edge extraction & radial lines
        tab_results = process_all_tabs(image_bgr, detections, center_x, center_y, params)

        # Store results
        self.last_image_bgr = image_bgr
        self.last_hough_result = hough_result
        self.last_detections = detections
        self.last_tab_results = tab_results
        self.radial_segments_cache = {}

        # Update result labels
        self.index_var.set("Anh: {}/{} | {}".format(
            self.current_index + 1, len(self.image_paths), os.path.basename(image_path)))
        self.center_var.set("Tam Hough: ({:.1f}, {:.1f})".format(center_x, center_y))
        self.radius_var.set("Ban kinh Hough: {:.1f} px".format(hough_result["radius"]))
        self.score_var.set("Hough score: {:.4f} | Scale: {:.2f}x".format(
            hough_result["debug"]["score"], scale_factor))

        # Render preview
        self.render_preview()

        self.status_var.set("Hoan tat. Tam: ({:.1f}, {:.1f}) | {} tai | {} bien.".format(
            center_x, center_y, len(detections),
            sum(1 for t in tab_results if t.get("edge_info") is not None)))

    def render_preview(self):
        """Render the preview based on current mode and toggle states."""
        if self.last_image_bgr is None or self.last_hough_result is None:
            return

        preview_mode = self.preview_mode_var.get()
        hough = self.last_hough_result
        tabs = self.last_tab_results or []
        detections = self.last_detections or []
        image_bgr = self.last_image_bgr
        radial_step_deg = self.get_radial_step_deg()
        show_radial_lines = bool(self.show_radial_lines_var.get())
        radial_segments = None
        if show_radial_lines:
            radial_segments = self.get_cached_radial_segments(radial_step_deg)

        if preview_mode == "result":
            preview_source = draw_result(
                image_bgr, hough, tabs,
                show_roi=bool(self.show_roi_var.get()),
                show_circle=bool(self.show_circle_var.get()),
                show_tab_edges=bool(self.show_tab_edges_var.get()),
                show_radial_lines=show_radial_lines,
                radial_step_deg=radial_step_deg,
                radial_segments=radial_segments,
            )
        elif preview_mode == "enhanced":
            preview_source = edge_to_bgr(hough["debug"]["roi_enhanced"])
        elif preview_mode == "blurred":
            preview_source = edge_to_bgr(hough["debug"]["roi_blurred"])
        elif preview_mode == "edges":
            preview_source = edge_to_bgr(hough["debug"]["raw_edges"])
        elif preview_mode == "masked_edges":
            preview_source = edge_to_bgr(hough["debug"]["masked_edges"])
        elif preview_mode == "tab_edges":
            preview_source = draw_tab_edges_only(
                image_bgr, tabs, hough["center_x"], hough["center_y"],
                hough["radius"], radial_step_deg if show_radial_lines else 360.0,
                radial_segments=[],
                use_outer_profile=False)
        elif preview_mode == "radial_lines":
            if show_radial_lines:
                preview_source = draw_radial_lines_only(
                    image_bgr, tabs, hough["center_x"], hough["center_y"],
                    hough["radius"], radial_step_deg, radial_segments=radial_segments)
            else:
                preview_source = image_bgr.copy()
                center = (int(round(hough["center_x"])), int(round(hough["center_y"])))
                cv2.circle(preview_source, center, int(round(hough["radius"])), (160, 160, 160), 1, cv2.LINE_AA)
                cv2.circle(preview_source, center, 6, (0, 0, 255), -1)
                cv2.drawMarker(preview_source, center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        else:
            preview_source = image_bgr

        self.last_preview_source = preview_source
        self.refresh_preview_canvas()
        self.refresh_signature_plot()


def main():
    root = tk.Tk()
    RadialSignatureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
