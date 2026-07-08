"""Step 3: report-aligned area-based tab-edge filtering."""

import cv2
import numpy as np

from .preprocess import apply_clahe, to_gray


def _make_odd(value, minimum=1):
    """Return an odd integer >= minimum."""
    value = max(minimum, int(round(float(value))))
    return value if value % 2 == 1 else value + 1


def _preprocess_roi(roi, preprocess_cfg, log_prefix, output_key):
    """Prepare one ROI with one configurable grayscale -> blur flow."""
    gray = to_gray(roi)
    output = gray.copy()
    logs = ["{} gray ready".format(log_prefix)]
    images = {"roi_original": roi.copy(), "roi_gray": gray}

    if preprocess_cfg.get("use_clahe", False):
        output = apply_clahe(
            output,
            preprocess_cfg.get("clahe_clip_limit", 2.0),
            preprocess_cfg.get("clahe_tile_grid_size", 8),
        )
        logs.append("{}: CLAHE enabled".format(log_prefix))

    if preprocess_cfg.get("use_gaussian", True):
        kernel = _make_odd(preprocess_cfg.get("gaussian_kernel", 5), minimum=1)
        sigma = max(0.0, float(preprocess_cfg.get("gaussian_sigma", 1.0)))
        output = cv2.GaussianBlur(output, (kernel, kernel), sigma)
        logs.append("{}: Gaussian blur k={} sigma={:.2f}".format(log_prefix, kernel, sigma))
    else:
        logs.append("{}: Gaussian blur TAT".format(log_prefix))
    images[output_key] = output
    return {"success": True, "data": output, "images": images, "logs": logs}


def preprocess_roi_for_ellipse_fit(roi, params):
    """Light ROI preprocessing for ellipse fitting on the stator body."""
    return _preprocess_roi(
        roi,
        params.get("fit_preprocess", params.get("preprocess", {})),
        "Ellipse fit ROI",
        "roi_fit_preprocessed",
    )


def preprocess_roi_for_tab_edges_local(roi, params):
    """Prepare one ROI for tab extraction after the annulus is known."""
    return _preprocess_roi(
        roi,
        params.get("preprocess", {}),
        "Tab ROI",
        "roi_preprocessed",
    )


def _get_radius_filter_spec(radius, params):
    """Resolve the active radius-filter settings for one ROI."""
    radius_cfg = params.get("radius_filter", {})
    if not radius_cfg.get("enabled", True):
        return {"enabled": False, "expected_inner_radius": 0.0, "outer_radius": float("inf")}
    expected_inner_radius = max(0.0, float(radius) * float(radius_cfg.get("r_min_factor", 1.0)))
    outer_radius = max(expected_inner_radius + 1.0, float(radius) * float(radius_cfg.get("r_max_factor", 1.3)))
    search_half_width = max(6.0, float(radius) * 0.035)
    search_min_radius = max(0.0, expected_inner_radius - search_half_width)
    search_max_radius = max(search_min_radius + 1.0, expected_inner_radius + search_half_width)
    ellipse_cut_offset_px = float(radius_cfg.get("ellipse_cut_offset_px", 0.0))
    fallback_to_circle_on_fit_error = bool(radius_cfg.get("fallback_to_circle_on_fit_error", True))
    return {
        "enabled": True,
        "expected_inner_radius": expected_inner_radius,
        "outer_radius": outer_radius,
        "search_min_radius": search_min_radius,
        "search_max_radius": search_max_radius,
        "search_half_width": search_half_width,
        "ellipse_cut_offset_px": ellipse_cut_offset_px,
        "fallback_to_circle_on_fit_error": fallback_to_circle_on_fit_error,
    }


def _fit_failure_result(spec, fit_edges, logs, fallback_reason):
    """Resolve one fit failure according to the fallback setting."""
    fallback_allowed = bool(spec.get("fallback_to_circle_on_fit_error", True))
    if fallback_allowed:
        return {
            "mode": "circle",
            "spec": spec,
            "fit_edges": fit_edges,
            "logs": logs,
            "fallback_reason": fallback_reason,
        }
    logs.append("Ellipse fit: fallback circle dang TAT, dung fit ellipse tai day")
    return {
        "mode": "fit_failed",
        "spec": spec,
        "fit_edges": fit_edges,
        "logs": logs,
        "fallback_reason": fallback_reason,
    }


def _ellipse_residuals(points, ellipse):
    """Measure how far points lie from one ellipse boundary."""
    if points is None or len(points) == 0:
        return np.empty((0,), dtype=np.float32)
    (cx, cy), (axis_w, axis_h), angle_deg = ellipse
    semi_w = max(1e-6, float(axis_w) * 0.5)
    semi_h = max(1e-6, float(axis_h) * 0.5)
    theta = np.deg2rad(float(angle_deg))
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    pts = np.asarray(points, dtype=np.float32)
    dx = pts[:, 0] - float(cx)
    dy = pts[:, 1] - float(cy)
    x_local = (dx * cos_theta) + (dy * sin_theta)
    y_local = (-dx * sin_theta) + (dy * cos_theta)
    normalized = ((x_local / semi_w) ** 2) + ((y_local / semi_h) ** 2)
    return np.abs(normalized - 1.0).astype(np.float32)


def _ellipse_from_boundary_points(preprocessed, center, radius, params):
    """Fit one inner ellipse from edge points near the expected r_min band."""
    spec = _get_radius_filter_spec(radius, params)
    if not spec.get("enabled", False):
        return {"mode": "disabled", "spec": spec, "logs": []}

    fit_edges = build_canny_edges(preprocessed, params)
    ys, xs = np.nonzero(fit_edges > 0)
    logs = [
        "Ellipse fit band tu dong: [{:.1f}, {:.1f}] px quanh r_min={:.1f} (half_width={:.1f})".format(
            spec["search_min_radius"],
            spec["search_max_radius"],
            spec["expected_inner_radius"],
            spec["search_half_width"],
        )
    ]
    if xs.size <= 0:
        logs.append("Ellipse fit: khong co edge de fit, fallback circle")
        return _fit_failure_result(spec, fit_edges, logs, "no_edges")

    dx = xs.astype(np.float32) - float(center[0])
    dy = ys.astype(np.float32) - float(center[1])
    rho = np.sqrt((dx ** 2) + (dy ** 2))
    in_band = (rho >= float(spec["search_min_radius"])) & (rho <= float(spec["search_max_radius"]))
    if not np.any(in_band):
        logs.append("Ellipse fit: khong tim thay diem trong dai tim kiem, fallback circle")
        return _fit_failure_result(spec, fit_edges, logs, "no_band_points")

    band_xs = xs[in_band].astype(np.float32)
    band_ys = ys[in_band].astype(np.float32)
    band_rho = rho[in_band]
    band_angles = np.arctan2(band_ys - float(center[1]), band_xs - float(center[0]))
    expected_radius = float(spec["expected_inner_radius"])
    bin_count = 180
    angle_bins = np.floor(((band_angles + np.pi) / (2.0 * np.pi)) * float(bin_count)).astype(np.int32) % bin_count
    candidate_points = []
    for bin_index in range(bin_count):
        bin_mask = angle_bins == bin_index
        if not np.any(bin_mask):
            continue
        delta = np.abs(band_rho[bin_mask] - expected_radius)
        best_local_index = int(np.argmin(delta))
        point_candidates = np.column_stack((band_xs[bin_mask], band_ys[bin_mask]))
        candidate_points.append(point_candidates[best_local_index])
    if len(candidate_points) < 5:
        logs.append("Ellipse fit: chi co {} diem ung vien, fallback circle".format(len(candidate_points)))
        return _fit_failure_result(spec, fit_edges, logs, "too_few_points")

    candidate_points = np.asarray(candidate_points, dtype=np.float32)
    logs.append("Ellipse fit: {} diem bien hop le sau chia goc".format(len(candidate_points)))

    try:
        ellipse = cv2.fitEllipse(candidate_points.reshape(-1, 1, 2))
    except cv2.error:
        logs.append("Ellipse fit: cv2.fitEllipse that bai, fallback circle")
        return _fit_failure_result(spec, fit_edges, logs, "fit_failed")

    residuals = _ellipse_residuals(candidate_points, ellipse)
    inlier_mask = residuals <= 0.20
    inlier_points = candidate_points[inlier_mask]
    if len(inlier_points) >= 5:
        try:
            ellipse = cv2.fitEllipse(inlier_points.reshape(-1, 1, 2))
            logs.append("Ellipse fit: giu {} / {} diem sau loc outlier".format(len(inlier_points), len(candidate_points)))
        except cv2.error:
            logs.append("Ellipse fit: refit sau loc outlier that bai, giu fit dau")
    else:
        logs.append("Ellipse fit: bo qua loc outlier vi chi con {} diem".format(len(inlier_points)))

    (fit_cx, fit_cy), (axis_w, axis_h), angle_deg = ellipse
    center_shift = float(np.hypot(float(fit_cx) - float(center[0]), float(fit_cy) - float(center[1])))
    semi_major = max(float(axis_w), float(axis_h)) * 0.5
    semi_minor = min(float(axis_w), float(axis_h)) * 0.5
    mean_radius = 0.5 * (semi_major + semi_minor)
    axis_ratio = semi_major / max(1e-6, semi_minor)
    search_span = max(3.0, float(spec["search_max_radius"]) - float(spec["search_min_radius"]))
    max_center_shift = max(12.0, float(radius) * 0.28)
    max_radius_delta = max(search_span + 4.0, float(radius) * 0.16)
    if (
        not np.isfinite(center_shift)
        or not np.isfinite(mean_radius)
        or center_shift > max_center_shift
        or abs(mean_radius - expected_radius) > max_radius_delta
        or axis_ratio > 1.8
        or semi_minor < 6.0
    ):
        logs.append(
            "Ellipse fit: ket qua bat thuong (shift={:.1f}, meanR={:.1f}, ratio={:.2f}), fallback circle".format(
                center_shift,
                mean_radius,
                axis_ratio,
            )
        )
        return _fit_failure_result(spec, fit_edges, logs, "abnormal_fit")

    cut_offset = float(spec.get("ellipse_cut_offset_px", 0.0))
    expanded_axes = (
        max(2.0, float(axis_w) + (2.0 * cut_offset)),
        max(2.0, float(axis_h) + (2.0 * cut_offset)),
    )
    final_ellipse = ((float(fit_cx), float(fit_cy)), expanded_axes, float(angle_deg))
    logs.append(
        "Ellipse fit: center=({:.1f}, {:.1f}), axes=({:.1f}, {:.1f}), angle={:.1f}, cut_offset={:.1f}".format(
            float(fit_cx),
            float(fit_cy),
            expanded_axes[0],
            expanded_axes[1],
            float(angle_deg),
            cut_offset,
        )
    )
    return {
        "mode": "ellipse",
        "spec": spec,
        "fit_edges": fit_edges,
        "ellipse": final_ellipse,
        "raw_ellipse": ((float(fit_cx), float(fit_cy)), (float(axis_w), float(axis_h)), float(angle_deg)),
        "point_count": int(len(candidate_points)),
        "logs": logs,
    }


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


def _build_threshold_mask(preprocessed, params, mask=None):
    """Create the threshold image, optionally using annulus pixels for Otsu statistics."""
    threshold_cfg = params.get("threshold", {})
    use_otsu = bool(threshold_cfg.get("use_otsu", True))
    invert = bool(threshold_cfg.get("invert", True))
    compute_on_full_roi = bool(threshold_cfg.get("compute_on_full_roi", False))
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    manual_threshold = int(round(float(threshold_cfg.get("manual_value", 0))))

    use_mask_for_stats = (mask is not None) and (not compute_on_full_roi)
    if use_mask_for_stats:
        masked_pixels = preprocessed[mask > 0]
    else:
        masked_pixels = preprocessed.reshape(-1)
    if masked_pixels.size <= 0:
        return np.zeros_like(preprocessed), 0.0, "Masked threshold EMPTY"

    if use_otsu:
        masked_view = masked_pixels.reshape(-1, 1)
        actual_threshold, _binary_samples = cv2.threshold(
            masked_view,
            0,
            255,
            threshold_type | cv2.THRESH_OTSU,
        )
        threshold_value = float(actual_threshold)
        if use_mask_for_stats:
            mode_name = "Masked Otsu {}".format("binary_inv" if invert else "binary")
        else:
            mode_name = "Full-ROI Otsu {}".format("binary_inv" if invert else "binary")
    else:
        threshold_value = float(manual_threshold)
        if use_mask_for_stats:
            mode_name = "Masked manual {} @ {}".format("binary_inv" if invert else "binary", manual_threshold)
        else:
            mode_name = "Full-ROI manual {} @ {}".format("binary_inv" if invert else "binary", manual_threshold)

    _actual, binary = cv2.threshold(
        preprocessed,
        threshold_value,
        255,
        threshold_type,
    )
    return binary, float(threshold_value), mode_name


def _clean_binary_in_annulus(binary_ring, annulus_mask, params):
    """Apply binary cleanup only inside the annulus before component filtering."""
    cleanup_cfg = params.get("binary_cleanup", {})
    output = binary_ring.copy()
    logs = []
    if not cleanup_cfg.get("enabled", True):
        logs.append("Binary cleanup: TAT")
        return output, logs

    if cleanup_cfg.get("use_median", True):
        kernel = _make_odd(cleanup_cfg.get("median_kernel", 3), minimum=1)
        output = cv2.medianBlur(output, kernel)
        output = cv2.bitwise_and(output, annulus_mask)
        logs.append("Binary cleanup: median k={}".format(kernel))
    else:
        logs.append("Binary cleanup: median TAT")

    if cleanup_cfg.get("use_morph_open", False):
        kernel_size = _make_odd(cleanup_cfg.get("open_kernel", 3), minimum=1)
        iterations = max(1, int(round(float(cleanup_cfg.get("open_iterations", 1)))))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel, iterations=iterations)
        output = cv2.bitwise_and(output, annulus_mask)
        logs.append("Binary cleanup: morph open k={} iter={}".format(kernel_size, iterations))
    else:
        logs.append("Binary cleanup: morph open TAT")
    return output, logs


def build_radius_mask(shape, center, radius, params, preprocessed=None):
    """Build the [outside inner ellipse/circle] & [inside outer circle] mask."""
    spec = _get_radius_filter_spec(radius, params)
    if not spec.get("enabled", False):
        mask = np.ones(shape[:2], dtype=np.uint8) * 255
        return mask, {"mode": "disabled", "spec": spec, "logs": []}
    yy, xx = np.indices(shape[:2])
    rho = np.sqrt((xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2)
    outer_circle_mask = (rho <= float(spec["outer_radius"])).astype(np.uint8) * 255
    fit_info = _ellipse_from_boundary_points(preprocessed, center, radius, params) if preprocessed is not None else _fit_failure_result(
        spec,
        None,
        ["Ellipse fit: khong co anh preprocess, fallback circle"],
        "missing_preprocessed",
    )

    inner_exclusion_mask = np.zeros(shape[:2], dtype=np.uint8)
    mode = fit_info.get("mode", "circle")
    if mode == "ellipse" and fit_info.get("ellipse") is not None:
        ellipse = fit_info["ellipse"]
        cv2.ellipse(
            inner_exclusion_mask,
            (int(round(float(ellipse[0][0]))), int(round(float(ellipse[0][1])))),
            (max(1, int(round(float(ellipse[1][0]) * 0.5))), max(1, int(round(float(ellipse[1][1]) * 0.5)))),
            float(ellipse[2]),
            0.0,
            360.0,
            255,
            -1,
            cv2.LINE_AA,
        )
    elif mode == "circle":
        inner_exclusion_mask = np.where(rho <= float(spec["expected_inner_radius"]), 255, 0).astype(np.uint8)
    else:
        radius_mask = np.zeros(shape[:2], dtype=np.uint8)
        return radius_mask, {**fit_info, "spec": spec}

    radius_mask = cv2.bitwise_and(outer_circle_mask, cv2.bitwise_not(inner_exclusion_mask))
    return radius_mask, {**fit_info, "spec": spec}


def apply_radius_filter(binary_image, radius_mask):
    """Filter one binary image by the annulus mask."""
    return cv2.bitwise_and(binary_image, radius_mask)


def keep_tab_components(binary_ring, center, params):
    """Keep only connected components whose area passes the configured thresholds."""
    _ = center  # Giu signature hien tai de khong anh huong cac cho goi cu.
    component_cfg = params.get("component_filter", {})
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary_ring, connectivity=8)
    pass_mask = np.zeros_like(binary_ring)
    component_reports = []
    min_area = max(0, int(round(float(component_cfg.get("min_area", 1500)))))
    max_area = max(0, int(round(float(component_cfg.get("max_area", 30000)))))
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        keep = area >= min_area
        if max_area > 0 and area > max_area:
            keep = False
        if keep:
            pass_mask[labels == label] = 255
        component_reports.append({"area": area, "keep": keep})
    component_reports.sort(key=lambda item: item["area"], reverse=True)
    return pass_mask, component_reports


def _draw_mask_outline(mask, output, color, thickness=1):
    """Overlay the contour of a single-channel mask onto a BGR image."""
    contour_info = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contour_info) == 3:
        _img, contours, _hierarchy = contour_info
    else:
        contours, _hierarchy = contour_info
    if contours:
        cv2.drawContours(output, contours, -1, color, thickness, cv2.LINE_AA)


def make_tab_edge_debug_overlay(roi, center, radius, tab_mask, tab_edges_clean, params, radius_mask_info=None):
    """Draw the debug overlay with the mask currently selected for radial matching."""
    if len(roi.shape) == 2:
        overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    else:
        overlay = roi.copy()
    radius_mask_info = radius_mask_info or {"spec": _get_radius_filter_spec(radius, params), "mode": "circle"}
    spec = radius_mask_info.get("spec", _get_radius_filter_spec(radius, params))
    enabled = bool(spec.get("enabled", False))
    mask_colored = np.zeros_like(overlay)
    mask_colored[tab_mask > 0] = (0, 200, 0)
    overlay = cv2.addWeighted(overlay, 0.78, mask_colored, 0.28, 0)
    _draw_mask_outline(tab_mask, overlay, (0, 255, 0), thickness=1)
    overlay[tab_edges_clean > 0] = (255, 255, 255)
    cxy = (int(round(center[0])), int(round(center[1])))
    cv2.circle(overlay, cxy, int(round(radius)), (0, 255, 0), 1, cv2.LINE_AA)
    if enabled:
        if radius_mask_info.get("mode") == "ellipse" and radius_mask_info.get("ellipse") is not None:
            ellipse = radius_mask_info["ellipse"]
            cv2.ellipse(
                overlay,
                (int(round(float(ellipse[0][0]))), int(round(float(ellipse[0][1])))),
                (max(1, int(round(float(ellipse[1][0]) * 0.5))), max(1, int(round(float(ellipse[1][1]) * 0.5)))),
                float(ellipse[2]),
                0.0,
                360.0,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(overlay, cxy, int(round(float(spec.get("expected_inner_radius", radius)))), (0, 165, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, cxy, int(round(float(spec.get("outer_radius", radius)))), (255, 0, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(overlay, cxy, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    return overlay


def _run_report_tab_edge_pipeline(roi, center, radius, params):
    """Run the report-aligned threshold -> area filter -> Canny pipeline."""
    fit_preprocess_result = preprocess_roi_for_ellipse_fit(roi, params)
    fit_preprocessed = fit_preprocess_result["data"]
    radius_mask, radius_mask_info = build_radius_mask(fit_preprocessed.shape, center, radius, params, preprocessed=fit_preprocessed)

    preprocess_result = preprocess_roi_for_tab_edges_local(roi, params)
    preprocessed = preprocess_result["data"]
    binary_otsu, threshold_value, threshold_mode = _build_threshold_mask(preprocessed, params, mask=radius_mask)
    binary_ring_raw = apply_radius_filter(binary_otsu, radius_mask)
    binary_ring, cleanup_logs = _clean_binary_in_annulus(binary_ring_raw, radius_mask, params)
    pass_mask, component_reports = keep_tab_components(binary_ring, center, params)
    tab_mask = pass_mask
    mask_mode = "pass_mask"
    tab_edges_raw = build_canny_edges(binary_ring, params)
    pass_edges = build_canny_edges(pass_mask, params)
    tab_edges_clean = build_canny_edges(tab_mask, params)
    overlay = make_tab_edge_debug_overlay(roi, center, radius, tab_mask, tab_edges_clean, params, radius_mask_info=radius_mask_info)
    point_count = int(np.count_nonzero(tab_edges_clean))
    selected_area_px = int(np.count_nonzero(tab_mask))
    pass_area_px = int(np.count_nonzero(pass_mask))
    kept_component_count = sum(1 for component in component_reports if component["keep"])
    component_cfg = params.get("component_filter", {})
    min_area = max(0, int(round(float(component_cfg.get("min_area", 1500)))))
    max_area = max(0, int(round(float(component_cfg.get("max_area", 30000)))))

    logs = ["Tab edge mode: report Otsu + radius + area + Canny"]
    logs.extend(fit_preprocess_result["logs"])
    logs.extend(preprocess_result["logs"])
    logs.append("Threshold: {} (value {:.1f})".format(threshold_mode, threshold_value))
    logs.append("Tab selection mode: {}".format(mask_mode))
    logs.extend(radius_mask_info.get("logs", []))
    logs.append("Inner boundary mode: {}".format(radius_mask_info.get("mode", "circle")))
    logs.extend(cleanup_logs)
    logs.append("Mask vung ban kinh: {} px".format(int(np.count_nonzero(radius_mask))))
    logs.append("Pixel sau masked threshold: {}".format(int(np.count_nonzero(binary_otsu))))
    logs.append("Pixel trong annulus truoc cleanup: {}".format(int(np.count_nonzero(binary_ring_raw))))
    logs.append("Pixel trong annulus sau cleanup: {}".format(int(np.count_nonzero(binary_ring))))
    logs.append("So component truoc loc: {}".format(len(component_reports)))
    logs.append("Nguong dien tich: min_area={} px | max_area={}".format(min_area, max_area if max_area > 0 else "OFF"))
    logs.append("So component duoc giu: {}".format(kept_component_count))
    logs.append("Tong dien tich pass_mask: {} px".format(pass_area_px))
    logs.append("Tong dien tich tab_mask dang dung: {} px".format(selected_area_px))
    logs.append("Edge points edges_tab: {}".format(point_count))
    for index, component in enumerate(component_reports, start=1):
        logs.append(
            "Component {:02d}: area = {} px -> {}".format(
                index,
                int(component["area"]),
                "KEEP" if component["keep"] else "REMOVE",
            )
        )
    if point_count <= 0 or selected_area_px <= 0:
        logs.append("Khong tao duoc edges_tab hop le sau buoc loc dien tich.")

    return {
        "success": point_count > 0 and selected_area_px > 0,
        "data": {
            "point_count": point_count,
            "component_count": len(component_reports),
            "kept_component_count": kept_component_count,
            "selected_area_px": selected_area_px,
            "kept_area_px": selected_area_px,
            "pass_area_px": pass_area_px,
            "mask_mode": mask_mode,
            "threshold_value": threshold_value,
            "threshold_mode": threshold_mode,
            "inner_boundary_mode": radius_mask_info.get("mode", "circle"),
        },
        "images": {
            "debug_overlay": overlay,
            "tab_edges_clean": tab_edges_clean,
            "closed_edges": tab_edges_clean,
            "tab_edges_raw": tab_edges_raw,
            "tab_mask": tab_mask,
            "area_filtered_mask": pass_mask,
            "selected_mask": tab_mask,
            "pass_mask": pass_mask,
            "pass_edges": pass_edges,
            "binary_ring": binary_ring,
            "binary_ring_raw": binary_ring_raw,
            "binary_otsu": binary_otsu,
            "radius_mask": radius_mask,
            "radius_fit_edges": radius_mask_info.get("fit_edges"),
            "canny_edges": tab_edges_clean,
            "roi_original": preprocess_result["images"]["roi_original"],
            "roi_preprocessed": preprocess_result["images"]["roi_preprocessed"],
            "roi_fit_preprocessed": fit_preprocess_result["images"]["roi_fit_preprocessed"],
            "roi_gray": preprocess_result["images"]["roi_gray"],
        },
        "logs": logs,
    }


def filter_tab_edges(roi, center, radius, params):
    """Run the report-aligned tab-edge pipeline."""
    return _run_report_tab_edge_pipeline(roi, center, radius, params)


preprocess_roi_for_tab_edges = preprocess_roi_for_tab_edges_local
