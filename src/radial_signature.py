"""Step 4: radial signature extraction."""

import io

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _build_radius_band_mask(shape, center, radius, params):
    """Create an annulus mask around the refined stator circle."""
    if radius is None or radius <= 0:
        return np.ones(shape[:2], dtype=np.uint8) * 255
    inner_scale = float(params.get("inner_radius_scale", 0.92))
    outer_scale = float(params.get("outer_radius_scale", 1.42))
    inner_r = max(0.0, float(radius) * min(inner_scale, outer_scale))
    outer_r = max(inner_r + 1.0, float(radius) * max(inner_scale, outer_scale))
    yy, xx = np.indices(shape[:2])
    dist = np.sqrt((xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[(dist >= inner_r) & (dist <= outer_r)] = 255
    return mask


def _dilate_binary(source, kernel_size, iterations):
    """Apply a small dilation to connect thin curved edges before ray casting."""
    kernel_size = max(1, int(round(float(kernel_size))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    iterations = max(0, int(round(float(iterations))))
    if kernel_size <= 1 or iterations <= 0:
        return source
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(source, kernel, iterations=iterations)


def _smooth_circular(values, window):
    window = max(1, int(round(float(window))))
    if window % 2 == 0:
        window += 1
    if window <= 1 or len(values) == 0:
        return values.astype(np.float32)
    pad = window // 2
    extended = np.concatenate([values[-pad:], values, values[:pad]])
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(extended, kernel, mode="same")
    return smoothed[pad : pad + len(values)].astype(np.float32)


def _reject_outlier_bins(signature, valid_mask, params):
    """Drop bins that jump too far away from their local neighborhood."""
    if not params.get("reject_outliers", True):
        return signature.astype(np.float32), valid_mask
    values = signature.astype(np.float32).copy()
    mask = valid_mask.copy()
    if len(values) == 0:
        return values, mask
    window = max(1, int(round(float(params.get("outlier_window", 9)))))
    if window % 2 == 0:
        window += 1
    half = window // 2
    max_delta = max(0.0, float(params.get("outlier_max_delta", 18.0)))
    for idx in range(len(values)):
        if not mask[idx]:
            continue
        neighbors = []
        for offset in range(-half, half + 1):
            if offset == 0:
                continue
            pos = (idx + offset) % len(values)
            if mask[pos]:
                neighbors.append(values[pos])
        if len(neighbors) < max(2, half):
            continue
        local_median = float(np.median(neighbors))
        if abs(float(values[idx]) - local_median) > max_delta:
            mask[idx] = False
            values[idx] = 0.0
    return values, mask


def interpolate_missing_bins(signature, valid_mask, params):
    """Interpolate small missing gaps in a circular signal."""
    if not params.get("interpolate_missing", True):
        return signature.astype(np.float32), valid_mask
    values = signature.astype(np.float32).copy()
    n = len(values)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2:
        return values, valid_mask
    max_gap = int(round(float(params.get("max_gap_to_interpolate", 8))))
    for idx in range(n):
        if valid_mask[idx]:
            continue
        left = None
        right = None
        for offset in range(1, max_gap + 1):
            if valid_mask[(idx - offset) % n]:
                left = (idx - offset) % n
                break
        for offset in range(1, max_gap + 1):
            if valid_mask[(idx + offset) % n]:
                right = (idx + offset) % n
                break
        if left is None or right is None:
            continue
        if (idx - left) % n > max_gap or (right - idx) % n > max_gap:
            continue
        left_val = values[left]
        right_val = values[right]
        span = ((right - left) % n) or n
        pos = ((idx - left) % n) / float(span)
        values[idx] = float(left_val + (right_val - left_val) * pos)
        valid_mask[idx] = True
    return values, valid_mask


def normalize_signature(signature):
    """Normalize a signature for comparison (z-score)."""
    signature = np.asarray(signature, dtype=np.float32)
    if signature.size == 0:
        return signature
    mean = float(signature.mean())
    std = float(signature.std())
    if std < 1e-6:
        return signature - mean
    return (signature - mean) / std


def scale_normalize_signature(signature, radius):
    """Chuan hoa bat bien ti le: rho/R roi tru trung binh.

    Chia cho ban kinh stator (R) neo tuyet doi tan ve "1.0 = nam tren than tron",
    nen stator chup to/nho khac nhau van trung profile. Tru trung binh bo phan nen
    bat bien xoay, giu lai bien do tai (do nho ra cua tai) lam dac trung so khop.
    """
    signature = np.asarray(signature, dtype=np.float32)
    if signature.size == 0 or radius is None or float(radius) <= 0:
        return normalize_signature(signature)
    ratio = signature / float(radius)
    return (ratio - float(ratio.mean())).astype(np.float32)


def is_valid_signature(signature, valid_mask):
    """Check whether a signature has enough valid bins."""
    if signature is None or len(signature) == 0:
        return False
    return int(np.count_nonzero(valid_mask)) >= max(12, len(signature) // 12)


def select_radial_source(edge_image, center, radius, params):
    """Filter the chosen edge image into the radial source used per angle."""
    if edge_image is None:
        raise ValueError("Khong co anh edge de tao radial signature.")
    if len(edge_image.shape) == 3:
        edge_gray = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
    else:
        edge_gray = edge_image.copy()
    source = (edge_gray > 0).astype(np.uint8) * 255
    if params.get("use_radius_band", True):
        radius_band = _build_radius_band_mask(edge_gray.shape, center, radius, params)
        source = cv2.bitwise_and(source, radius_band)
    else:
        radius_band = np.ones(edge_gray.shape[:2], dtype=np.uint8) * 255
    if params.get("use_source_dilate", True):
        source = _dilate_binary(
            source,
            params.get("source_dilate_kernel", 3),
            params.get("source_dilate_iter", 1),
        )
    return source, radius_band


def _sample_ray_hits(source_image, center, angle_rad, start_r, end_r, step_px, thickness):
    """Return farthest hit point along one ray inside the masked edge image."""
    height, width = source_image.shape[:2]
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    offsets = range(-max(0, thickness), max(0, thickness) + 1)
    farthest_rho = 0.0
    farthest_point = None
    radius = float(start_r)
    while radius <= float(end_r):
        base_x = float(center[0]) + radius * cos_a
        base_y = float(center[1]) + radius * sin_a
        for offset in offsets:
            sample_x = int(round(base_x - offset * sin_a))
            sample_y = int(round(base_y + offset * cos_a))
            if sample_x < 0 or sample_x >= width or sample_y < 0 or sample_y >= height:
                continue
            if source_image[sample_y, sample_x] <= 0:
                continue
            rho = float(np.hypot(sample_x - float(center[0]), sample_y - float(center[1])))
            if rho >= farthest_rho:
                farthest_rho = rho
                farthest_point = (sample_x, sample_y)
        radius += float(step_px)
    return farthest_rho, farthest_point


def build_radial_signature(edge_image, center, params, radius=None):
    """Build raw and normalized radial signatures from a filtered edge map."""
    num_angles = max(36, int(round(float(params.get("num_angles", 360)))))
    signature_raw = np.zeros(num_angles, dtype=np.float32)
    valid_mask = np.zeros(num_angles, dtype=bool)
    source_image, radius_band = select_radial_source(edge_image, center, radius, params)
    source_points = int(np.count_nonzero(source_image))
    if radius is not None and radius > 0:
        min_valid_radius = float(radius) * max(0.0, float(params.get("min_valid_radius_scale", 1.0)))
        ray_start = min_valid_radius
        if params.get("use_radius_band", True):
            ray_end = float(radius) * max(
                float(params.get("inner_radius_scale", 1.03)),
                float(params.get("outer_radius_scale", 1.34)),
            )
        else:
            ray_end = max(source_image.shape[:2])
    else:
        min_valid_radius = 0.0
        ray_start = 0.0
        ray_end = max(source_image.shape[:2])
    ray_end = max(ray_start + 1.0, float(ray_end))
    ray_step_px = max(0.5, float(params.get("ray_step_px", 1.0)))
    ray_thickness = max(0, int(round(float(params.get("ray_thickness", 2)))))
    measured_points = []
    for idx in range(num_angles):
        angle_deg = idx * (360.0 / float(num_angles))
        angle_rad = np.deg2rad(angle_deg)
        rho, point = _sample_ray_hits(
            source_image,
            center,
            angle_rad,
            ray_start,
            ray_end,
            ray_step_px,
            ray_thickness,
        )
        if point is not None and rho >= min_valid_radius:
            signature_raw[idx] = rho
            valid_mask[idx] = True
            measured_points.append(point)
    signature_raw, valid_mask = _reject_outlier_bins(signature_raw, valid_mask, params)
    measured_signature = signature_raw.astype(np.float32).copy()
    measured_mask = valid_mask.copy()
    signature_raw, valid_mask = interpolate_missing_bins(signature_raw, valid_mask, params)
    floor_radius = float(radius) if (radius is not None and float(radius) > 0) else 0.0
    floor_to_radius = bool(params.get("floor_to_radius", True))
    if floor_radius > 0 and floor_to_radius:
        # Goc khong cham tai -> lay ban kinh Hough lam nen truoc khi smoothing,
        # de mep tai (vien bo xuong) khong bi keo tut xuong duoi than tron.
        signature_raw = np.where(valid_mask, signature_raw, floor_radius).astype(np.float32)
    if params.get("smooth_signature", True):
        signature_raw = _smooth_circular(signature_raw, params.get("smooth_window", 5))
    if floor_radius > 0 and floor_to_radius:
        # Khong cho bat ky tia nao ngan hon ban kinh Hough.
        signature_raw = np.maximum(signature_raw, floor_radius).astype(np.float32)
    if params.get("scale_normalize", True) and floor_radius > 0:
        signature_norm = scale_normalize_signature(signature_raw, floor_radius)
    else:
        signature_norm = normalize_signature(signature_raw)
    valid_bins = int(np.count_nonzero(valid_mask))
    measured_bins = int(np.count_nonzero(measured_mask))
    return {
        "success": is_valid_signature(signature_raw, valid_mask),
        "data": {
            "signature_raw": signature_raw,
            "signature_norm": signature_norm,
            "valid_mask": valid_mask,
            "valid_bins": valid_bins,
            "measured_signature": measured_signature,
            "measured_mask": measured_mask,
            "measured_bins": measured_bins,
            "measured_points": measured_points,
            "radial_source": source_image,
            "radius_band": radius_band,
        },
        "images": {},
        "logs": [
            "Radial source points: {}".format(source_points),
            "Ray start radius: {:.1f}".format(ray_start),
            "Ray end radius: {:.1f}".format(ray_end),
            "Measured radial bins: {}".format(measured_bins),
            "Valid radial bins: {}".format(valid_bins),
        ],
    }


def draw_radial_rays(roi, center, signature_raw, radius=None, params=None, measured_mask=None):
    """Draw rays from center to radial signature endpoints."""
    if len(roi.shape) == 2:
        output = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    else:
        output = roi.copy()
    params = params or {}
    num_angles = len(signature_raw)
    cxy = (int(round(center[0])), int(round(center[1])))
    cv2.drawMarker(output, cxy, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    if radius is not None:
        cv2.circle(output, cxy, int(round(radius)), (100, 100, 100), 1, cv2.LINE_AA)
    fallback_radius = float(radius) if radius is not None else 0.0
    for idx, rho in enumerate(signature_raw):
        has_measurement = measured_mask is None or bool(measured_mask[idx])
        if rho <= 0 and fallback_radius <= 0:
            continue
        draw_rho = float(rho)
        color = (0, 255, 0)
        end_marker = (0, 255, 255)
        if not has_measurement:
            if fallback_radius <= 0:
                continue
            draw_rho = fallback_radius
            color = (180, 180, 180)
            end_marker = (180, 180, 180)
        angle_deg = idx * (360.0 / float(num_angles))
        angle_rad = np.deg2rad(angle_deg)
        end_pt = (
            int(round(center[0] + draw_rho * np.cos(angle_rad))),
            int(round(center[1] + draw_rho * np.sin(angle_rad))),
        )
        if idx == 0:
            color = (0, 0, 255)
            end_marker = (0, 0, 255)
        cv2.line(output, cxy, end_pt, color, 1, cv2.LINE_AA)
        cv2.circle(output, end_pt, 2, end_marker, -1, cv2.LINE_AA)
    return output


def build_radial_debug_views(roi, center, signature_raw, radius=None, params=None, measured_mask=None, source_images=None):
    """Build radial-ray overlays for the ROI and optional edge-source images.

    Khi xem debug sau buoc radial, user thuong muon thay ngay tia quet tren
    `tab_edges_clean` / `closed_edges`, khong chi tren ROI goc. Ham nay tao:
    - `radial_rays`: overlay tren ROI goc
    - `<name>`: overlay tia tren anh nguon edge
    - `<name>_raw`: ban goc de doi chieu khi can
    """
    overlays = {
        "radial_rays": draw_radial_rays(
            roi,
            center,
            signature_raw,
            radius=radius,
            params=params,
            measured_mask=measured_mask,
        )
    }
    for name, image in (source_images or {}).items():
        if image is None:
            continue
        overlays["{}_raw".format(name)] = image
        overlays[name] = draw_radial_rays(
            image,
            center,
            signature_raw,
            radius=radius,
            params=params,
            measured_mask=measured_mask,
        )
    return overlays


def plot_signature_image(signature_raw, signature_norm=None):
    """Render a signature plot as a BGR image."""
    figure, axes = plt.subplots(2 if signature_norm is not None else 1, 1, figsize=(8, 4.8), dpi=120)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes[0].plot(signature_raw, color="#0f766e", linewidth=1.6)
    axes[0].set_title("Radial signature raw")
    axes[0].grid(True, alpha=0.25)
    if signature_norm is not None:
        axes[1].plot(signature_norm, color="#7c3aed", linewidth=1.6)
        axes[1].set_title("Radial signature normalized")
        axes[1].grid(True, alpha=0.25)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    image_bytes = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
