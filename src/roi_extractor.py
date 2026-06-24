"""Step 2: ROI extraction and local Hough refinement."""

import copy
import math

import cv2
import numpy as np

from .io_utils import ROI_DIR, write_image
from .preprocess import apply_clahe, to_gray
from .visualization import draw_circles, draw_roi_boxes


def _make_odd(value, minimum=1):
    value = max(minimum, int(round(float(value))))
    return value if value % 2 == 1 else value + 1


def _radius_band_mask(shape, center, inner_radius, outer_radius):
    """Build a binary mask that keeps only a ring around the coarse circle."""
    height, width = shape[:2]
    yy, xx = np.ogrid[:height, :width]
    dist = np.sqrt((xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[(dist >= float(inner_radius)) & (dist <= float(outer_radius))] = 255
    return mask


def _edge_support_ratio(edges, cx, cy, radius, ring_width=3):
    """Measure how much edge support exists around one candidate circle."""
    radius = int(round(float(radius)))
    ring_width = max(1, int(round(float(ring_width))))
    if radius < 4:
        return 0.0
    offsets = []
    for ring_radius in range(max(1, radius - ring_width), radius + ring_width + 1):
        sample_count = max(96, int(round(2.0 * math.pi * ring_radius)))
        angles = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
        xs = np.rint(ring_radius * np.cos(angles)).astype(np.int32)
        ys = np.rint(ring_radius * np.sin(angles)).astype(np.int32)
        offsets.append(np.stack((xs, ys), axis=1))
    offsets = np.unique(np.concatenate(offsets, axis=0), axis=0)
    sample_x = int(round(float(cx))) + offsets[:, 0]
    sample_y = int(round(float(cy))) + offsets[:, 1]
    valid = (
        (sample_x >= 0)
        & (sample_x < edges.shape[1])
        & (sample_y >= 0)
        & (sample_y < edges.shape[0])
    )
    valid_count = int(valid.sum())
    if valid_count <= 0:
        return 0.0
    return float((edges[sample_y[valid], sample_x[valid]] > 0).sum()) / float(valid_count)


def _score_circle(circle, edge_image, target_center, target_radius, score_cfg):
    """Score one ROI-local circle by edge support plus soft geometry priors."""
    cx, cy, radius = circle
    support = _edge_support_ratio(
        edge_image,
        cx,
        cy,
        radius,
        ring_width=score_cfg.get("ring_width", 3),
    )
    radius_ref = max(1.0, float(target_radius))
    center_distance = math.hypot(float(cx) - float(target_center[0]), float(cy) - float(target_center[1]))
    center_penalty = center_distance / radius_ref
    radius_penalty = abs(float(radius) - float(target_radius)) / radius_ref
    center_weight = max(0.0, float(score_cfg.get("center_penalty_weight", 0.10)))
    radius_weight = max(0.0, float(score_cfg.get("radius_penalty_weight", 0.20)))
    return float(support - (center_weight * center_penalty) - (radius_weight * radius_penalty))


def _prepare_refine_image(roi, params):
    """Prepare the selected ROI for the local Hough pass."""
    gray = to_gray(roi)
    output = gray.copy()
    logs = ["ROI gray ready"]
    preprocess_cfg = params.get("refine", {}).get("preprocess", {})
    if preprocess_cfg.get("use_clahe", True):
        output = apply_clahe(
            output,
            preprocess_cfg.get("clahe_clip_limit", 2.0),
            preprocess_cfg.get("clahe_tile_grid_size", 8),
        )
        logs.append("ROI refine: CLAHE enabled")
    if preprocess_cfg.get("use_gaussian", True):
        kernel = _make_odd(preprocess_cfg.get("gaussian_kernel", 5), minimum=1)
        output = cv2.GaussianBlur(output, (kernel, kernel), 0)
        logs.append("ROI refine: Gaussian blur k={}".format(kernel))
    return output, logs


def _draw_local_detected(roi, circle):
    """Render a local overlay with the refined circle and center marker."""
    circle_dict = {
        "id": 1,
        "x": float(circle[0]),
        "y": float(circle[1]),
        "r": float(circle[2]),
        "score": float(circle[3]),
    }
    return draw_circles(roi, [circle_dict], color=(0, 255, 0), draw_ids=False, adaptive_style=True)


def _build_effective_circle(item, center_x, center_y, radius_full, score):
    return {
        "id": int(item.get("id", 0)),
        "x": float(center_x),
        "y": float(center_y),
        "r": float(radius_full),
        "score": float(score),
    }


def _selected_images(base_item, effective_item):
    debug = effective_item.get("debug", {}) if effective_item is not None else {}
    return {
        "selected_roi": base_item["roi"] if base_item is not None else None,
        "selected_roi_detected": debug.get("selected_roi_detected"),
        "selected_roi_preprocessed": debug.get("selected_roi_preprocessed"),
        "selected_roi_edges": debug.get("selected_roi_edges"),
        "selected_roi_masked_edges": debug.get("selected_roi_masked_edges"),
    }


def find_roi_item(roi_items, roi_id):
    """Find one ROI item by ID from a list."""
    roi_id_text = str(roi_id)
    for item in roi_items:
        if str(item.get("id")) == roi_id_text:
            return item
    return None


def extract_roi(image, circle, params):
    """Extract one square ROI around a detected coarse circle."""
    x = float(circle.get("x", circle.get("center_x", 0)))
    y = float(circle.get("y", circle.get("center_y", 0)))
    r = float(circle.get("r", circle.get("radius", 0)))
    half = int(max(1, round(float(r) * float(params.get("half_size_scale", 1.30)))))
    h, w = image.shape[:2]
    x1 = max(0, int(round(x)) - half)
    y1 = max(0, int(round(y)) - half)
    x2 = min(w, int(round(x)) + half)
    y2 = min(h, int(round(y)) + half)

    crop_width = max(1, x2 - x1)
    crop_height = max(1, y2 - y1)
    roi = image[y1:y2, x1:x2].copy()
    output_size = int(round(float(params.get("output_size", 0))))

    if output_size > 0 and roi.size > 0:
        roi = cv2.resize(roi, (output_size, output_size), interpolation=cv2.INTER_AREA)
        scale_x = float(crop_width) / float(output_size)
        scale_y = float(crop_height) / float(output_size)
    else:
        scale_x = 1.0
        scale_y = 1.0

    coarse_center_in_roi = ((x - float(x1)) / scale_x, (y - float(y1)) / scale_y)
    coarse_radius_in_roi = float(r) / max(1e-6, (scale_x + scale_y) * 0.5)
    coarse_score = float(circle.get("score", 0.0))
    return {
        "id": int(circle.get("id", 0)),
        "roi": roi,
        "offset_x": x1,
        "offset_y": y1,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "roi_scale_x": scale_x,
        "roi_scale_y": scale_y,
        "coarse_circle": dict(circle),
        "coarse_center_in_roi": coarse_center_in_roi,
        "coarse_radius": coarse_radius_in_roi,
        "center_in_roi": coarse_center_in_roi,
        "radius": coarse_radius_in_roi,
        "center_x": x,
        "center_y": y,
        "radius_full": r,
        "score": coarse_score,
        "circle": _build_effective_circle({"id": int(circle.get("id", 0))}, x, y, r, coarse_score),
        "refined": False,
        "debug": {},
        "logs": [],
    }


def extract_rois(image, circles, params):
    """Extract all coarse ROIs from a detected circle list."""
    return [extract_roi(image, circle, params) for circle in circles]


def refine_circle_in_roi(roi, coarse_center, coarse_radius, params):
    """Run a second Hough pass inside one ROI and pick the best local circle."""
    refine_cfg = params.get("refine", {})
    if roi is None or roi.size == 0:
        raise ValueError("ROI rong, khong the refine circle.")

    preprocessed, logs = _prepare_refine_image(roi, params)
    canny_cfg = refine_cfg.get("canny", {})
    edges = cv2.Canny(
        preprocessed,
        int(round(float(canny_cfg.get("threshold1", 70)))),
        int(round(float(canny_cfg.get("threshold2", 170)))),
    )

    mask_cfg = refine_cfg.get("mask", {})
    if mask_cfg.get("enabled", True):
        inner_radius = max(1.0, float(coarse_radius) * float(mask_cfg.get("inner_radius_scale", 0.82)))
        outer_radius = max(inner_radius + 1.0, float(coarse_radius) * float(mask_cfg.get("outer_radius_scale", 1.20)))
        radius_mask = _radius_band_mask(edges.shape, coarse_center, inner_radius, outer_radius)
        masked_edges = cv2.bitwise_and(edges, edges, mask=radius_mask)
        logs.append(
            "ROI refine: radius mask {:.2f}r -> {:.2f}r".format(
                float(mask_cfg.get("inner_radius_scale", 0.82)),
                float(mask_cfg.get("outer_radius_scale", 1.20)),
            )
        )
    else:
        masked_edges = edges.copy()
        logs.append("ROI refine: radius mask TAT")

    score_cfg = refine_cfg.get("score", {})
    coarse_candidate = (float(coarse_center[0]), float(coarse_center[1]), float(coarse_radius))
    best_circle = coarse_candidate
    best_score = _score_circle(coarse_candidate, masked_edges, coarse_center, coarse_radius, score_cfg)

    if not refine_cfg.get("enabled", True):
        logs.append("ROI refine: dung circle tho tu buoc 1")
    else:
        hough_cfg = refine_cfg.get("hough", {})
        min_radius = max(1, int(round(float(coarse_radius) * float(hough_cfg.get("min_radius_scale", 0.78)))))
        max_radius = max(min_radius + 1, int(round(float(coarse_radius) * float(hough_cfg.get("max_radius_scale", 1.18)))))
        max_center_shift = max(1.0, float(coarse_radius) * float(hough_cfg.get("max_center_shift_scale", 0.45)))
        circles = cv2.HoughCircles(
            preprocessed,
            cv2.HOUGH_GRADIENT,
            dp=max(1.0, float(hough_cfg.get("dp", 1.2))),
            minDist=max(1.0, float(hough_cfg.get("minDist", 20))),
            param1=max(1.0, float(hough_cfg.get("param1", 110))),
            param2=max(1.0, float(hough_cfg.get("param2", 30))),
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        logs.append(
            "ROI refine: Hough r=[{}, {}], shift<={:.1f}px".format(
                min_radius,
                max_radius,
                max_center_shift,
            )
        )

        scored_candidates = []
        if circles is not None:
            for cx, cy, radius in np.round(circles[0]).astype(int):
                shift = math.hypot(float(cx) - float(coarse_center[0]), float(cy) - float(coarse_center[1]))
                if shift > max_center_shift:
                    continue
                score = _score_circle((cx, cy, radius), masked_edges, coarse_center, coarse_radius, score_cfg)
                scored_candidates.append((float(cx), float(cy), float(radius), float(score)))
        logs.append("ROI refine: {} ung vien hop le".format(len(scored_candidates)))
        if scored_candidates:
            best_candidate = max(scored_candidates, key=lambda item: item[3])
            best_circle = best_candidate[:3]
            best_score = best_candidate[3]
        else:
            logs.append("ROI refine: fallback ve circle tho")

    detected_circle = (float(best_circle[0]), float(best_circle[1]), float(best_circle[2]), float(best_score))
    overlay = _draw_local_detected(roi, detected_circle)
    return {
        "center_in_roi": (float(best_circle[0]), float(best_circle[1])),
        "radius": float(best_circle[2]),
        "score": float(best_score),
        "images": {
            "selected_roi_preprocessed": preprocessed,
            "selected_roi_edges": edges,
            "selected_roi_masked_edges": masked_edges,
            "selected_roi_detected": overlay,
        },
        "logs": logs,
    }


def refine_roi_item(roi_item, params):
    """Refine one existing ROI item and return an updated copy."""
    item = copy.deepcopy(roi_item)
    coarse_center = item.get("coarse_center_in_roi", item.get("center_in_roi"))
    coarse_radius = float(item.get("coarse_radius", item.get("radius", 0)))
    refine_result = refine_circle_in_roi(
        item["roi"],
        coarse_center,
        coarse_radius,
        params,
    )
    local_cx, local_cy = refine_result["center_in_roi"]
    local_radius = float(refine_result["radius"])
    scale_x = float(item.get("roi_scale_x", 1.0))
    scale_y = float(item.get("roi_scale_y", 1.0))
    full_center_x = float(item["offset_x"]) + (float(local_cx) * scale_x)
    full_center_y = float(item["offset_y"]) + (float(local_cy) * scale_y)
    full_radius = float(local_radius) * ((scale_x + scale_y) * 0.5)

    item["center_in_roi"] = refine_result["center_in_roi"]
    item["radius"] = local_radius
    item["center_x"] = full_center_x
    item["center_y"] = full_center_y
    item["radius_full"] = full_radius
    item["score"] = float(refine_result["score"])
    item["circle"] = _build_effective_circle(item, full_center_x, full_center_y, full_radius, item["score"])
    item["refined"] = True
    item["debug"] = {**refine_result["images"]}
    item["logs"] = list(refine_result["logs"])
    return item


def build_roi_item_from_image(image, params, roi_id=1):
    """Build a refined ROI item from a standalone ROI image file.

    Dung cho ROI nap tu file ngoai (tab Template / Matching): thay vi doan cung
    tam = (w/2, h/2) va R = 0.35*min, ham nay doan tho R theo dung cach buoc 2 cat
    ROI roi chay Hough-refine de lay tam + ban kinh that. Co fallback an toan.
    """
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("Anh ROI rong, khong the dung lam ROI item.")
    height, width = image.shape[:2]
    half_scale = float(params.get("half_size_scale", 1.30)) or 1.30
    # Buoc 2 cat ROI vuong canh ~ 2*r*half_scale, nen suy nguoc R tu kich thuoc anh.
    seed_radius = 0.5 * float(min(width, height)) / max(1e-6, half_scale)
    if not seed_radius > 0:
        seed_radius = 0.35 * float(min(width, height))
    seed_center = (width / 2.0, height / 2.0)
    coarse_item = {
        "id": int(roi_id),
        "roi": image,
        "offset_x": 0,
        "offset_y": 0,
        "crop_width": width,
        "crop_height": height,
        "roi_scale_x": 1.0,
        "roi_scale_y": 1.0,
        "coarse_circle": {"id": int(roi_id), "x": seed_center[0], "y": seed_center[1], "r": seed_radius},
        "coarse_center_in_roi": seed_center,
        "coarse_radius": seed_radius,
        "center_in_roi": seed_center,
        "radius": seed_radius,
        "center_x": seed_center[0],
        "center_y": seed_center[1],
        "radius_full": seed_radius,
        "score": 0.0,
        "circle": _build_effective_circle({"id": int(roi_id)}, seed_center[0], seed_center[1], seed_radius, 0.0),
        "refined": False,
        "debug": {},
        "logs": [],
    }
    try:
        refined = refine_roi_item(coarse_item, params)
        cx, cy = refined["center_in_roi"]
        refined["logs"] = [
            "ROI ngoai: seed R={:.1f} -> refine R={:.1f}, tam=({:.1f}, {:.1f}), score={:.3f}".format(
                seed_radius, float(refined["radius"]), float(cx), float(cy), float(refined.get("score", 0.0))
            )
        ]
        return refined
    except Exception as exc:
        coarse_item["logs"] = ["ROI ngoai: refine that bai ({}); dung seed tho R={:.1f}.".format(exc, seed_radius)]
        return coarse_item


def save_rois(rois, output_dir=ROI_DIR):
    """Save ROI images to a folder."""
    saved_paths = []
    for item in rois:
        path = output_dir / "roi_stator_{:02d}.png".format(item["id"])
        write_image(path, item["roi"])
        saved_paths.append(str(path))
    return saved_paths


def run_roi_crop_step(image, circles, params, save_all=True):
    """Crop all ROI images from the current Hough result."""
    if not circles:
        return {"success": False, "data": {}, "images": {}, "logs": ["Chua co circle de cat ROI."]}
    rois = extract_rois(image, circles, params)
    saved_paths = save_rois(rois) if save_all else []
    overview = draw_roi_boxes(image, rois)
    first_roi = rois[0] if rois else None
    logs = ["Da cat {} ROI".format(len(rois))]
    if saved_paths:
        logs.append("Da luu {} ROI vao data/roi".format(len(saved_paths)))
    return {
        "success": True,
        "data": {
            "rois": rois,
            "saved_paths": saved_paths,
        },
        "images": {
            "overview": overview,
            "selected_roi": first_roi["roi"] if first_roi else None,
            "selected_roi_detected": None,
            "selected_roi_preprocessed": None,
            "selected_roi_edges": None,
            "selected_roi_masked_edges": None,
        },
        "logs": logs,
    }


def run_roi_refine_step(roi_item, params):
    """Refine one ROI item and return its debug images."""
    refined_item = refine_roi_item(roi_item, params)
    images = _selected_images(roi_item, refined_item)
    return {
        "success": True,
        "data": {
            "roi_item": refined_item,
        },
        "images": images,
        "logs": list(refined_item.get("logs", [])),
    }


def run_roi_step(image, circles, params, selected_id=None, refine_mode="selected", save_all=True):
    """Run ROI crop and optional local Hough refinement."""
    crop_result = run_roi_crop_step(image, circles, params, save_all=save_all)
    if not crop_result["success"]:
        return crop_result

    rois = crop_result["data"]["rois"]
    if not rois:
        return crop_result

    if selected_id is None:
        selected_id = rois[0]["id"]
    base_selected = find_roi_item(rois, selected_id) or rois[0]
    selected_id = base_selected["id"]

    refined_rois = {}
    logs = list(crop_result["logs"])
    if refine_mode == "all":
        for item in rois:
            refined_item = refine_roi_item(item, params)
            refined_rois[item["id"]] = refined_item
            logs.extend(["ROI ID{:02d}: {}".format(item["id"], msg) for msg in refined_item.get("logs", [])])
    elif refine_mode == "selected":
        refined_item = refine_roi_item(base_selected, params)
        refined_rois[selected_id] = refined_item
        logs.extend(["ROI ID{:02d}: {}".format(selected_id, msg) for msg in refined_item.get("logs", [])])

    effective_rois = [refined_rois.get(item["id"], item) for item in rois]
    effective_selected = refined_rois.get(selected_id, base_selected)
    images = {
        "overview": crop_result["images"]["overview"],
        **_selected_images(base_selected, effective_selected),
    }
    return {
        "success": True,
        "data": {
            "rois": rois,
            "effective_rois": effective_rois,
            "refined_rois": refined_rois,
            "selected_id": selected_id,
            "saved_paths": crop_result["data"]["saved_paths"],
        },
        "images": images,
        "logs": logs,
    }
