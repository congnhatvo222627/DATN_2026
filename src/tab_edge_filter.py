"""Step 3: report-aligned area-based tab-edge filtering."""

import cv2
import numpy as np

from .preprocess import apply_clahe, to_gray


def _make_odd(value, minimum=1):
    """Return an odd integer >= minimum."""
    value = max(minimum, int(round(float(value))))
    return value if value % 2 == 1 else value + 1


def preprocess_roi_for_tab_edges_local(roi, params):
    """Prepare one ROI with the report-style grayscale -> blur flow."""
    gray = to_gray(roi)
    output = gray.copy()
    logs = ["ROI gray ready"]
    preprocess_cfg = params.get("preprocess", {})
    images = {"roi_gray": gray}

    if preprocess_cfg.get("use_clahe", False):
        output = apply_clahe(
            output,
            preprocess_cfg.get("clahe_clip_limit", 2.0),
            preprocess_cfg.get("clahe_tile_grid_size", 8),
        )
        logs.append("CLAHE enabled")

    kernel = _make_odd(preprocess_cfg.get("gaussian_kernel", 5), minimum=1)
    sigma = max(0.0, float(preprocess_cfg.get("gaussian_sigma", 1.0)))
    output = cv2.GaussianBlur(output, (kernel, kernel), sigma)
    logs.append("Gaussian blur k={} sigma={:.2f}".format(kernel, sigma))
    images["roi_preprocessed"] = output
    return {"success": True, "data": output, "images": images, "logs": logs}


def _get_radius_annulus_bounds(radius, params):
    """Resolve the active annulus bounds around the stator body radius."""
    radius_cfg = params.get("radius_filter", {})
    if not radius_cfg.get("enabled", True):
        return False, 0.0, float("inf")
    r_inner = float(radius) * float(radius_cfg.get("r_min_factor", 1.0)) + float(radius_cfg.get("inner_margin_px", 0))
    r_outer = float(radius) * float(radius_cfg.get("r_max_factor", 1.3)) + float(radius_cfg.get("outer_margin_px", 0))
    if r_outer < r_inner:
        r_inner, r_outer = r_outer, r_inner
    r_inner = max(0.0, r_inner)
    r_outer = max(r_inner + 1.0, r_outer)
    return True, r_inner, r_outer


def build_canny_edges(image, params):
    """Build Canny edges from one preprocessed single-channel image."""
    canny_cfg = params.get("canny", {})
    return cv2.Canny(
        image,
        int(round(float(canny_cfg.get("threshold1", 70)))),
        int(round(float(canny_cfg.get("threshold2", 170)))),
        apertureSize=max(3, int(round(float(canny_cfg.get("aperture_size", 3))))),
        L2gradient=bool(canny_cfg.get("l2_gradient", False)),
    )


def build_radius_mask(shape, center, radius, params):
    """Build the [r_body, r_max] annulus mask used in the report pipeline."""
    enabled, r_inner, r_outer = _get_radius_annulus_bounds(radius, params)
    if not enabled:
        return np.ones(shape[:2], dtype=np.uint8) * 255
    yy, xx = np.indices(shape[:2])
    rho = np.sqrt((xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2)
    return np.where((rho >= r_inner) & (rho <= r_outer), 255, 0).astype(np.uint8)


def apply_radius_filter(binary_image, radius_mask):
    """Filter one binary image by the annulus mask."""
    return cv2.bitwise_and(binary_image, radius_mask)


def _component_angle_span_deg(component_mask, center, bin_size_deg):
    """Estimate how many degrees one connected component occupies."""
    ys, xs = np.where(component_mask)
    if len(xs) == 0:
        return 0.0
    bin_size_deg = max(0.5, float(bin_size_deg))
    angles = (np.degrees(np.arctan2(ys - float(center[1]), xs - float(center[0]))) + 360.0) % 360.0
    bins = np.unique(np.floor(angles / bin_size_deg).astype(np.int32))
    return float(len(bins)) * bin_size_deg


def keep_tab_components(binary_ring, center, params):
    """Keep only area-valid connected components in the masked threshold image."""
    component_cfg = params.get("component_filter", {})
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary_ring, connectivity=8)
    kept_mask = np.zeros_like(binary_ring)
    kept = []
    min_area = max(0, int(round(float(component_cfg.get("min_area", 1500)))))
    max_area = max(0, int(round(float(component_cfg.get("max_area", 30000)))))
    angle_bin_deg = max(0.5, float(component_cfg.get("angle_bin_deg", 5.0)))
    max_angle_span_deg = max(0.0, float(component_cfg.get("max_angle_span_deg", 150.0)))
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        component_mask = labels == label
        angle_span_deg = _component_angle_span_deg(component_mask, center, angle_bin_deg)
        if max_angle_span_deg > 0.0 and angle_span_deg > max_angle_span_deg:
            continue
        kept_mask[component_mask] = 255
        kept.append({"area": area, "angle_span_deg": angle_span_deg})
    return kept_mask, kept


def _build_threshold_mask(preprocessed, params):
    """Create the Otsu threshold mask used before ring filtering."""
    threshold_cfg = params.get("threshold", {})
    use_otsu = bool(threshold_cfg.get("use_otsu", True))
    invert = bool(threshold_cfg.get("invert", True))
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    threshold_value = 0 if use_otsu else int(round(float(threshold_cfg.get("manual_value", 0))))
    threshold_flags = threshold_type | (cv2.THRESH_OTSU if use_otsu else 0)
    actual_threshold, binary = cv2.threshold(preprocessed, threshold_value, 255, threshold_flags)
    mode_name = "Otsu {}".format("binary_inv" if invert else "binary")
    if not use_otsu:
        mode_name = "Manual {} @ {}".format("binary_inv" if invert else "binary", threshold_value)
    return binary, float(actual_threshold), mode_name


def _draw_mask_outline(mask, output, color, thickness=1):
    """Overlay the contour of a single-channel mask onto a BGR image."""
    contour_info = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contour_info) == 3:
        _img, contours, _hierarchy = contour_info
    else:
        contours, _hierarchy = contour_info
    if contours:
        cv2.drawContours(output, contours, -1, color, thickness, cv2.LINE_AA)


def make_tab_edge_debug_overlay(roi, center, radius, tab_mask, tab_edges_clean, params):
    """Draw the report-style debug overlay with annulus and kept tab regions."""
    if len(roi.shape) == 2:
        overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    else:
        overlay = roi.copy()
    enabled, inner_radius, outer_radius = _get_radius_annulus_bounds(radius, params)
    mask_colored = np.zeros_like(overlay)
    mask_colored[tab_mask > 0] = (0, 200, 0)
    overlay = cv2.addWeighted(overlay, 0.78, mask_colored, 0.28, 0)
    _draw_mask_outline(tab_mask, overlay, (0, 255, 0), thickness=1)
    overlay[tab_edges_clean > 0] = (255, 255, 255)
    cxy = (int(round(center[0])), int(round(center[1])))
    cv2.circle(overlay, cxy, int(round(radius)), (0, 255, 0), 1, cv2.LINE_AA)
    if enabled:
        cv2.circle(overlay, cxy, int(round(inner_radius)), (0, 165, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, cxy, int(round(outer_radius)), (255, 0, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(overlay, cxy, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    return overlay


def _run_report_tab_edge_pipeline(roi, center, radius, params):
    """Run the report-aligned threshold -> area filter -> Canny pipeline."""
    preprocess_result = preprocess_roi_for_tab_edges_local(roi, params)
    preprocessed = preprocess_result["data"]
    binary_otsu, threshold_value, threshold_mode = _build_threshold_mask(preprocessed, params)
    radius_mask = build_radius_mask(binary_otsu.shape, center, radius, params)
    binary_ring = apply_radius_filter(binary_otsu, radius_mask)
    tab_mask, kept_components = keep_tab_components(binary_ring, center, params)
    tab_edges_raw = build_canny_edges(binary_ring, params)
    tab_edges_clean = build_canny_edges(tab_mask, params)
    overlay = make_tab_edge_debug_overlay(roi, center, radius, tab_mask, tab_edges_clean, params)
    point_count = int(np.count_nonzero(tab_edges_clean))
    kept_area_px = int(np.count_nonzero(tab_mask))

    logs = ["Tab edge mode: report Otsu + radius + area + Canny"]
    logs.extend(preprocess_result["logs"])
    logs.append("Threshold: {} (value {:.1f})".format(threshold_mode, threshold_value))
    logs.append("Mask vung ban kinh: {} px".format(int(np.count_nonzero(radius_mask))))
    logs.append("Pixel sau threshold + ring mask: {}".format(int(np.count_nonzero(binary_ring))))
    logs.append("So component giu lai: {}".format(len(kept_components)))
    logs.append("Tong dien tich tab_mask: {} px".format(kept_area_px))
    logs.append("Edge points edges_tab: {}".format(point_count))
    for index, component in enumerate(kept_components, start=1):
        logs.append(
            "  Component {:02d}: area={} px, span={:.1f} deg".format(
                index,
                int(component["area"]),
                float(component["angle_span_deg"]),
            )
        )
    if point_count <= 0 or kept_area_px <= 0:
        logs.append("Khong tao duoc edges_tab hop le sau buoc loc dien tich.")

    return {
        "success": point_count > 0 and kept_area_px > 0,
        "data": {
            "point_count": point_count,
            "component_count": len(kept_components),
            "kept_area_px": kept_area_px,
            "threshold_value": threshold_value,
            "threshold_mode": threshold_mode,
        },
        "images": {
            "debug_overlay": overlay,
            "tab_edges_clean": tab_edges_clean,
            "closed_edges": tab_edges_clean,
            "tab_edges_raw": tab_edges_raw,
            "tab_mask": tab_mask,
            "binary_ring": binary_ring,
            "binary_otsu": binary_otsu,
            "radius_mask": radius_mask,
            "canny_edges": tab_edges_clean,
            "roi_preprocessed": preprocess_result["images"]["roi_preprocessed"],
            "roi_gray": preprocess_result["images"]["roi_gray"],
        },
        "logs": logs,
    }


def filter_tab_edges(roi, center, radius, params):
    """Run the report-aligned tab-edge pipeline."""
    return _run_report_tab_edge_pipeline(roi, center, radius, params)


preprocess_roi_for_tab_edges = preprocess_roi_for_tab_edges_local
