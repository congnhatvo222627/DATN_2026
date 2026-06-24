"""Step 3: tab-edge filtering."""

import math
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .config import TAB_EDGE_YOLO_MODEL_PATH
from .preprocess import (
    apply_blur,
    apply_clahe,
    preprocess_roi_for_tab_edges as _shared_preprocess_roi_for_tab_edges,
    to_gray,
)


_YOLO_MODEL_CACHE = {"path": None, "model": None}


def _make_odd(value, minimum=1):
    """Return an odd integer >= minimum."""
    value = max(minimum, int(round(float(value))))
    return value if value % 2 == 1 else value + 1


def preprocess_roi_for_tab_edges_local(roi, params):
    """Local wrapper to keep requested module API name stable."""
    return _shared_preprocess_roi_for_tab_edges(roi, params)


def build_canny_edges(preprocessed, params):
    """Build Canny edges from a preprocessed grayscale image."""
    canny_cfg = params.get("canny", {})
    return cv2.Canny(
        preprocessed,
        int(round(float(canny_cfg.get("threshold1", 70)))),
        int(round(float(canny_cfg.get("threshold2", 170)))),
        apertureSize=max(3, int(round(float(canny_cfg.get("aperture_size", 3))))),
        L2gradient=bool(canny_cfg.get("l2_gradient", False)),
    )


def build_radius_mask(shape, center, radius, params):
    """Build a binary annulus mask for the legacy tab-edge pipeline."""
    radius_cfg = params.get("radius_filter", {})
    r_inner = float(radius) * float(radius_cfg.get("r_min_factor", 1.0)) + float(radius_cfg.get("inner_margin_px", 0))
    r_outer = float(radius) * float(radius_cfg.get("r_max_factor", 1.35)) + float(radius_cfg.get("outer_margin_px", 8))
    yy, xx = np.indices(shape[:2])
    rho = np.sqrt((xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2)
    return np.where((rho >= r_inner) & (rho <= r_outer), 255, 0).astype(np.uint8)


def apply_radius_filter(edges, radius_mask):
    """Filter edge pixels by the annulus mask."""
    return cv2.bitwise_and(edges, radius_mask)


def clean_tab_edges_by_components(tab_edges_raw, center, radius, params):
    """Keep only components that fit size and radial rules."""
    component_cfg = params.get("component_filter", {})
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(tab_edges_raw, connectivity=8)
    clean = np.zeros_like(tab_edges_raw)
    kept = 0
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        cx, cy = centroids[label]
        rho_centroid = float(np.hypot(cx - center[0], cy - center[1]))
        if area < int(component_cfg.get("min_area", 30)) or area > int(component_cfg.get("max_area", 20000)):
            continue
        if w < int(component_cfg.get("min_width", 2)) or h < int(component_cfg.get("min_height", 2)):
            continue
        if rho_centroid < radius * float(component_cfg.get("min_radius_mean_factor", 0.95)):
            continue
        if rho_centroid > radius * float(component_cfg.get("max_radius_mean_factor", 1.45)):
            continue
        clean[labels == label] = 255
        kept += 1
    morph_cfg = params.get("morphology", {})
    if morph_cfg.get("use_close", True):
        kernel_size = _make_odd(morph_cfg.get("close_kernel", 3), minimum=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(morph_cfg.get("close_iter", 1))))
    if morph_cfg.get("use_dilate", False):
        kernel_size = _make_odd(morph_cfg.get("dilate_kernel", 3), minimum=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        clean = cv2.dilate(clean, kernel, iterations=max(1, int(morph_cfg.get("dilate_iter", 1))))
    return clean, kept


def make_tab_edge_debug_overlay(roi, center, radius, tab_edges_clean, params):
    """Draw debug circles and edge overlay for the legacy pipeline."""
    if len(roi.shape) == 2:
        overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    else:
        overlay = roi.copy()
    radius_cfg = params.get("radius_filter", {})
    r_inner = int(round(radius * float(radius_cfg.get("r_min_factor", 1.0)) + float(radius_cfg.get("inner_margin_px", 0))))
    r_outer = int(round(radius * float(radius_cfg.get("r_max_factor", 1.35)) + float(radius_cfg.get("outer_margin_px", 8))))
    edges_colored = np.zeros_like(overlay)
    edges_colored[tab_edges_clean > 0] = (255, 255, 255)
    overlay = cv2.addWeighted(overlay, 0.7, edges_colored, 1.0, 0)
    cxy = (int(round(center[0])), int(round(center[1])))
    cv2.circle(overlay, cxy, int(round(radius)), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(overlay, cxy, r_inner, (0, 165, 255), 1, cv2.LINE_AA)
    cv2.circle(overlay, cxy, r_outer, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(overlay, cxy, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    return overlay


def _build_close_edges(edge_image, params):
    """Apply morphology close to an edge image when enabled."""
    morph_cfg = params.get("morphology", {})
    if not morph_cfg.get("use_close", True):
        return edge_image.copy()
    kernel_size = _make_odd(morph_cfg.get("close_kernel", 3), minimum=1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(
        edge_image,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=max(1, int(morph_cfg.get("close_iter", 1))),
    )


def _load_yolo_model(model_path):
    """Load and cache the YOLO model used for tab localization."""
    if YOLO is None:
        raise ImportError("Chua cai ultralytics. Hay cai bang: .\\.venv\\Scripts\\python.exe -m pip install ultralytics")
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError("Khong tim thay model YOLO: {}".format(path))
    resolved = str(path.resolve())
    if _YOLO_MODEL_CACHE["model"] is not None and _YOLO_MODEL_CACHE["path"] == resolved:
        return _YOLO_MODEL_CACHE["model"]
    model = YOLO(resolved)
    _YOLO_MODEL_CACHE["path"] = resolved
    _YOLO_MODEL_CACHE["model"] = model
    return model


def detect_tabs_yolo(roi, params):
    """Run YOLO on one stator ROI and return bounding boxes."""
    yolo_cfg = params.get("yolo", {})
    model_path = str(yolo_cfg.get("model_path", TAB_EDGE_YOLO_MODEL_PATH))
    model = _load_yolo_model(model_path)
    conf_threshold = max(0.01, min(0.99, float(yolo_cfg.get("conf_threshold", 0.25))))
    results = model.predict(source=roi, conf=conf_threshold, verbose=False)
    detections = []
    if results and len(results) > 0:
        result = results[0]
        names = result.names if hasattr(result, "names") else {}
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                detections.append(
                    {
                        "x1": int(round(x1)),
                        "y1": int(round(y1)),
                        "x2": int(round(x2)),
                        "y2": int(round(y2)),
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": str(names.get(cls_id, cls_id)),
                    }
                )
    return sorted(detections, key=lambda item: (item["y1"], item["x1"]))


def _preprocess_tab_crop(tab_crop, params):
    """Preprocess one YOLO crop before contour extraction."""
    preprocess_cfg = params.get("preprocess", {})
    gray = to_gray(tab_crop)
    output = gray.copy()
    logs = []
    if preprocess_cfg.get("use_clahe", True):
        output = apply_clahe(
            output,
            preprocess_cfg.get("clahe_clip_limit", 2.0),
            preprocess_cfg.get("clahe_tile_grid_size", 8),
        )
        logs.append("CLAHE enabled")
    output, blur_logs = apply_blur(output, preprocess_cfg)
    logs.extend(blur_logs)
    edges = build_canny_edges(output, params)
    closed = _build_close_edges(edges, params)
    return {
        "gray": gray,
        "preprocessed": output,
        "edges": edges,
        "closed": closed,
        "logs": logs,
    }


def extract_tab_edge_contours(closed_edges):
    """Find external contours in a tab region."""
    contour_info = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contour_info) == 3:
        _, contours, _ = contour_info
    else:
        contours, _ = contour_info
    return contours or []


def build_outer_profile_contours(edge_points, center_x, center_y, params):
    """Keep only the farthest edge point for each angular bin."""
    if not edge_points:
        return []
    contour_cfg = params.get("contour_filter", {})
    bin_size_deg = max(0.5, float(contour_cfg.get("outer_profile_bin_deg", 1.0)))
    max_point_gap = max(1.0, float(contour_cfg.get("max_point_gap_px", 28.0)))
    max_angle_gap_deg = max(0.5, float(contour_cfg.get("max_angle_gap_deg", 5.0)))
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
        px, py, _dist, angle_deg = point
        if previous_point is not None:
            prev_px, prev_py, _prev_dist, prev_angle_deg = previous_point
            angle_gap = abs(angle_deg - prev_angle_deg)
            point_gap = math.hypot(px - prev_px, py - prev_py)
            if angle_gap > max_angle_gap_deg or point_gap > max_point_gap:
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


def _compute_tab_edge_points_in_roi(detection, preprocessed, center, crop_x1, crop_y1, params):
    """Filter crop contours and build the outer tab-edge profile in ROI coordinates."""
    closed = preprocessed["closed"]
    contours = extract_tab_edge_contours(closed)
    if not contours:
        return None

    contour_cfg = params.get("contour_filter", {})
    center_x, center_y = float(center[0]), float(center[1])
    roi_height, roi_width = closed.shape[:2]
    roi_area = float(max(1, roi_width * roi_height))
    min_area = max(float(contour_cfg.get("min_area", 24)), roi_area * float(contour_cfg.get("min_area_ratio", 0.0015)))

    detection_center_x = (float(detection["x1"]) + float(detection["x2"])) / 2.0
    detection_center_y = (float(detection["y1"]) + float(detection["y2"])) / 2.0
    detection_center_dist = math.hypot(detection_center_x - center_x, detection_center_y - center_y)
    min_keep_distance = detection_center_dist * float(contour_cfg.get("min_keep_distance_ratio", 0.88))

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

        contour_roi = contour.copy()
        contour_roi[:, 0, 0] += crop_x1
        contour_roi[:, 0, 1] += crop_y1

        contour_points = contour_roi[:, 0, :].astype(np.float32)
        dx = contour_points[:, 0] - center_x
        dy = contour_points[:, 1] - center_y
        distances = np.sqrt((dx * dx) + (dy * dy))
        contour_max_dist = float(np.max(distances)) if len(distances) else 0.0
        if contour_max_dist < min_keep_distance:
            continue

        filtered_contours.append(contour_roi)
        moments = cv2.moments(contour_roi)
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

    outer_profile_contours = build_outer_profile_contours(all_edge_pts, center_x, center_y, params)
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
    angle_tolerance = math.radians(float(contour_cfg.get("radial_angle_tolerance_deg", 18.0)))
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


def _tab_color(index):
    """Return a consistent debug color per tab index."""
    colors = [
        (0, 255, 0),
        (255, 100, 0),
        (0, 200, 255),
        (255, 0, 200),
        (100, 255, 100),
        (255, 200, 0),
    ]
    return colors[index % len(colors)]


def _draw_boxes_on_roi(roi, detections):
    """Draw YOLO boxes on an ROI for debugging."""
    if len(roi.shape) == 2:
        output = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    else:
        output = roi.copy()
    for index, det in enumerate(detections):
        color = _tab_color(index)
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = "{} {:.0f}%".format(det.get("class_name", "tab"), det.get("confidence", 0.0) * 100.0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return output


def _merge_crop_into_canvas(canvas, crop, x1, y1):
    """Paste a crop into a grayscale debug canvas using max-combine."""
    y2 = min(canvas.shape[0], y1 + crop.shape[0])
    x2 = min(canvas.shape[1], x1 + crop.shape[1])
    if x2 <= x1 or y2 <= y1:
        return
    crop_view = crop[: y2 - y1, : x2 - x1]
    canvas[y1:y2, x1:x2] = np.maximum(canvas[y1:y2, x1:x2], crop_view)


def _draw_closed_contours(mask, contours, thickness=1):
    """Draw closed contours on a single-channel mask."""
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness, cv2.LINE_AA)


def _draw_open_contours(mask, contours, thickness=1):
    """Draw open polyline contours on a single-channel mask."""
    for contour in contours:
        if contour is None or len(contour) < 2:
            continue
        pts = contour.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(mask, [pts], False, 255, thickness, cv2.LINE_AA)


def _draw_center_marker(image, center):
    """Draw the ROI center marker used by the YOLO-guided debug overlay."""
    cxy = (int(round(center[0])), int(round(center[1])))
    cv2.drawMarker(image, cxy, (255, 0, 0), cv2.MARKER_CROSS, 16, 2)
    cv2.circle(image, cxy, 2, (255, 255, 255), -1, cv2.LINE_AA)


def _make_yolo_debug_overlay(roi, center, radius, contour_list):
    """Render a dark debug view that highlights only tab-edge contours."""
    height, width = roi.shape[:2]
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    overlay[:] = (30, 30, 30)
    contour_mask = np.zeros((height, width), dtype=np.uint8)
    _draw_open_contours(contour_mask, contour_list, thickness=2)
    overlay[contour_mask > 0] = (255, 255, 255)
    cxy = (int(round(center[0])), int(round(center[1])))
    cv2.circle(overlay, cxy, int(round(radius)), (80, 80, 80), 1, cv2.LINE_AA)
    _draw_center_marker(overlay, center)
    return overlay


def _run_yolo_tab_edge_pipeline(roi, center, radius, params):
    """Run YOLO-guided tab-edge extraction on one stator ROI."""
    roi_gray = to_gray(roi)
    roi_preprocessed = np.zeros_like(roi_gray)
    canny_edges = np.zeros_like(roi_gray)
    closed_edges = np.zeros_like(roi_gray)
    logs = ["Tab edge mode: YOLO-guided contour filtering"]

    try:
        detections = detect_tabs_yolo(roi, params)
    except Exception as exc:
        return {"success": False, "data": {}, "images": {"roi_gray": roi_gray}, "logs": logs + [str(exc)]}

    yolo_cfg = params.get("yolo", {})
    logs.append("Model YOLO: {}".format(Path(str(yolo_cfg.get("model_path", TAB_EDGE_YOLO_MODEL_PATH))).name))
    logs.append("So box YOLO: {}".format(len(detections)))
    yolo_boxes = _draw_boxes_on_roi(roi, detections)
    if not detections:
        return {
            "success": False,
            "data": {"point_count": 0, "detection_count": 0, "valid_tab_count": 0},
            "images": {
                "roi_gray": roi_gray,
                "yolo_boxes": yolo_boxes,
            },
            "logs": logs + ["YOLO khong tim thay box tai tren ROI nay."],
        }

    tab_results = []
    height, width = roi.shape[:2]
    padding_ratio = max(0.0, float(yolo_cfg.get("box_padding_ratio", 0.10)))
    padding_min_px = max(0, int(round(float(yolo_cfg.get("box_padding_min_px", 12)))))

    for detection in detections:
        x1 = max(0, int(detection["x1"]))
        y1 = max(0, int(detection["y1"]))
        x2 = min(width, int(detection["x2"]))
        y2 = min(height, int(detection["y2"]))
        if x2 <= x1 or y2 <= y1:
            continue

        pad_x = max(padding_min_px, int(round((x2 - x1) * padding_ratio)))
        pad_y = max(padding_min_px, int(round((y2 - y1) * padding_ratio)))
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(width, x2 + pad_x)
        crop_y2 = min(height, y2 + pad_y)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue

        tab_crop = roi[crop_y1:crop_y2, crop_x1:crop_x2]
        preprocessed = _preprocess_tab_crop(tab_crop, params)
        _merge_crop_into_canvas(roi_preprocessed, preprocessed["preprocessed"], crop_x1, crop_y1)
        _merge_crop_into_canvas(canny_edges, preprocessed["edges"], crop_x1, crop_y1)
        _merge_crop_into_canvas(closed_edges, preprocessed["closed"], crop_x1, crop_y1)

        edge_info = _compute_tab_edge_points_in_roi(detection, preprocessed, center, crop_x1, crop_y1, params)
        tab_results.append(
            {
                "detection": {
                    **detection,
                    "crop_x1": crop_x1,
                    "crop_y1": crop_y1,
                    "crop_x2": crop_x2,
                    "crop_y2": crop_y2,
                },
                "preprocessed": preprocessed,
                "edge_info": edge_info,
            }
        )

    raw_mask = np.zeros_like(roi_gray)
    clean_mask = np.zeros_like(roi_gray)
    raw_contours = []
    clean_contours = []
    valid_tab_count = 0
    for tab in tab_results:
        edge_info = tab.get("edge_info")
        if edge_info is None:
            continue
        valid_tab_count += 1
        raw_contours.extend(edge_info.get("raw_contours", []))
        clean_contours.extend(edge_info.get("contours", []))

    _draw_closed_contours(raw_mask, raw_contours, thickness=1)
    _draw_open_contours(clean_mask, clean_contours, thickness=1)
    debug_overlay = _make_yolo_debug_overlay(roi, center, radius, clean_contours)
    point_count = int(np.count_nonzero(clean_mask))
    logs.append("So tab hop le sau loc contour: {}".format(valid_tab_count))
    logs.append("Edge points sau loc outer profile: {}".format(point_count))

    success = valid_tab_count > 0 and point_count > 0
    if not success:
        logs.append("Khong tao duoc mask bien tai hop le tu YOLO boxes.")
    return {
        "success": success,
        "data": {
            "point_count": point_count,
            "detection_count": len(detections),
            "valid_tab_count": valid_tab_count,
        },
        "images": {
            "debug_overlay": debug_overlay,
            "tab_edges_clean": clean_mask,
            "yolo_boxes": yolo_boxes,
            "tab_edges_raw": raw_mask,
            "closed_edges": closed_edges,
            "canny_edges": canny_edges,
            "roi_preprocessed": roi_preprocessed,
            "roi_gray": roi_gray,
        },
        "logs": logs,
    }


def _run_legacy_tab_edge_pipeline(roi, center, radius, params):
    """Run the previous radius-mask + connected-component pipeline."""
    preprocess_result = preprocess_roi_for_tab_edges_local(roi, params)
    preprocessed = preprocess_result["data"]
    edges = build_canny_edges(preprocessed, params)
    radius_mask = build_radius_mask(preprocessed.shape, center, radius, params)
    tab_edges_raw = apply_radius_filter(edges, radius_mask)
    tab_edges_clean, component_count = clean_tab_edges_by_components(tab_edges_raw, center, radius, params)
    overlay = make_tab_edge_debug_overlay(roi, center, radius, tab_edges_clean, params)
    point_count = int(np.count_nonzero(tab_edges_clean))
    logs = ["Tab edge mode: legacy radius + component"]
    logs.extend(preprocess_result["logs"])
    logs.append("Edge points sau Canny: {}".format(int(np.count_nonzero(edges))))
    logs.append("Edge points sau loc radius + component: {}".format(point_count))
    logs.append("So component giu lai: {}".format(component_count))
    return {
        "success": True,
        "data": {"point_count": point_count, "component_count": component_count},
        "images": {
            "debug_overlay": overlay,
            "tab_edges_clean": tab_edges_clean,
            "tab_edges_raw": tab_edges_raw,
            "radius_mask": radius_mask,
            "canny_edges": edges,
            "roi_preprocessed": preprocess_result["images"]["roi_preprocessed"],
            "roi_gray": preprocess_result["images"]["roi_gray"],
        },
        "logs": logs,
    }


def filter_tab_edges(roi, center, radius, params):
    """Run the configured tab-edge pipeline."""
    if params.get("yolo", {}).get("enabled", True):
        return _run_yolo_tab_edge_pipeline(roi, center, radius, params)
    return _run_legacy_tab_edge_pipeline(roi, center, radius, params)


preprocess_roi_for_tab_edges = preprocess_roi_for_tab_edges_local
