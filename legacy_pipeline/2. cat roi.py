# -*- coding: utf-8 -*-
"""
Create ROI crops around each detected stator.

The script reuses `1. HoughCircle.py` to detect stator centers, then crops a
square ROI around each circle. When run directly, it opens a Tk GUI so the user
can choose an input image, tune ROI + Hough parameters, preview ROI results,
save/load presets, and export the current ROI result set.
"""

import copy
import functools
import importlib.util
import json
import math
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from matplotlib import gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HOUGH_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "1. HoughCircle.py")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "test_results")
PRESET_DIR = os.path.join(PROJECT_ROOT, "presets", "roi_cut_settings")
PRESET_SCHEMA_VERSION = "roi_cut_setting_v1"

DEFAULT_INPUT_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "test", "test_stator_12.png"),
    os.path.join(PROJECT_ROOT, "data", "test", "test.png"),
    os.path.join(PROJECT_ROOT, "data", "test", "1.png"),
]

ALL_ROI_SELECTOR_LABEL = "Tat ca ROI"


def find_default_input_path():
    for path in DEFAULT_INPUT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return ""


DEFAULT_CONFIG = {
    "input_path": find_default_input_path(),
    "output_dir": DEFAULT_OUTPUT_DIR,
    "expected_count": 12,
    "crop": {
        "x": 0,
        "y": 0,
        "w": 0,
        "h": 0,
    },
    "roi": {
        "half_size_scale": 1.30,
    },
    "detection": {
        "use_fast_mode": True,
        "max_processing_dim": 1750,
    },
    "preview": {
        "overview_max_width": 1600,
        "overview_max_height": 1000,
        "roi_max_size": 420,
        "selected_roi_index": 1,
        "show_all_rois": True,
        "auto_update": True,
        "auto_update_delay_ms": 350,
    },
}

CONFIG = copy.deepcopy(DEFAULT_CONFIG)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_image(path):
    """Read an image with Unicode-path support."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, image):
    """Write an image with Unicode-path support."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError("Khong the ghi anh: {}".format(path))
    encoded.tofile(path)


def merge_nested_dict(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_nested_dict(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def normalize_config(raw_config=None):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if isinstance(raw_config, dict):
        merge_nested_dict(cfg, raw_config)

    cfg["input_path"] = str(cfg.get("input_path") or "")
    cfg["output_dir"] = os.path.abspath(str(cfg.get("output_dir") or DEFAULT_OUTPUT_DIR))
    cfg["expected_count"] = max(1, int(round(float(cfg.get("expected_count", 12)))))

    crop_cfg = cfg["crop"]
    for key in ["x", "y", "w", "h"]:
        crop_cfg[key] = max(0, int(round(float(crop_cfg.get(key, 0)))))

    roi_cfg = cfg["roi"]
    roi_cfg["half_size_scale"] = min(2.8, max(0.8, float(roi_cfg.get("half_size_scale", 1.3))))

    detection_cfg = cfg["detection"]
    detection_cfg["use_fast_mode"] = bool(detection_cfg.get("use_fast_mode", True))
    detection_cfg["max_processing_dim"] = max(
        640,
        int(round(float(detection_cfg.get("max_processing_dim", 1750)))),
    )

    preview_cfg = cfg["preview"]
    preview_cfg["overview_max_width"] = max(
        800,
        int(round(float(preview_cfg.get("overview_max_width", 1600)))),
    )
    preview_cfg["overview_max_height"] = max(
        600,
        int(round(float(preview_cfg.get("overview_max_height", 1000)))),
    )
    preview_cfg["roi_max_size"] = max(180, int(round(float(preview_cfg.get("roi_max_size", 420)))))
    preview_cfg["selected_roi_index"] = max(
        1,
        int(round(float(preview_cfg.get("selected_roi_index", 1)))),
    )
    legacy_right_view_mode = str(preview_cfg.get("right_view_mode", "detail")).lower()
    preview_cfg["show_all_rois"] = bool(preview_cfg.get("show_all_rois", legacy_right_view_mode == "all"))
    preview_cfg["auto_update"] = bool(preview_cfg.get("auto_update", True))
    preview_cfg["auto_update_delay_ms"] = min(
        2000,
        max(120, int(round(float(preview_cfg.get("auto_update_delay_ms", 350))))),
    )

    return cfg


def get_output_paths(output_dir):
    roi_output_dir = os.path.join(output_dir, "roi_images")
    marked_path = os.path.join(output_dir, "Hinh_4_8_toan_canh_danh_dau_ROI.png")
    compare_path = os.path.join(output_dir, "Hinh_4_8_cat_ROI_quanh_tung_stator.png")
    return {
        "roi_output_dir": roi_output_dir,
        "marked_path": marked_path,
        "compare_path": compare_path,
    }


@functools.lru_cache(maxsize=1)
def load_hough_circle_module():
    if not os.path.isfile(HOUGH_SCRIPT_PATH):
        raise FileNotFoundError("Khong tim thay file HoughCircle: {}".format(HOUGH_SCRIPT_PATH))

    spec = importlib.util.spec_from_file_location("hough_circle_module", HOUGH_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Khong the nap module tu file: {}".format(HOUGH_SCRIPT_PATH))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resize_for_display(image, max_width=None, max_height=None):
    h, w = image.shape[:2]
    scale = 1.0

    if max_width is not None and w > 0:
        scale = min(scale, float(max_width) / float(w))
    if max_height is not None and h > 0:
        scale = min(scale, float(max_height) / float(h))

    if scale >= 1.0:
        return image

    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def detect_stator_center_from_hough(input_image_path, image, output_dir, cfg):
    del input_image_path  # kept for backward-compatible signature usage
    del output_dir

    module = load_hough_circle_module()
    required_attrs = ["crop_work_roi", "detect_stator_centers"]
    for attr in required_attrs:
        if not hasattr(module, attr):
            raise AttributeError("File 1. HoughCircle.py thieu ham {}().".format(attr))

    detection_cfg = cfg["detection"]
    max_processing_dim = int(detection_cfg["max_processing_dim"]) if bool(detection_cfg["use_fast_mode"]) else 0
    h, w = image.shape[:2]
    scale = 1.0

    if max_processing_dim > 0 and max(h, w) > max_processing_dim:
        scale = float(max_processing_dim) / float(max(h, w))
        working_image = resize_for_display(
            image,
            max_width=max_processing_dim,
            max_height=max_processing_dim,
        )
    else:
        working_image = image

    crop_cfg = copy.deepcopy(module.CONFIG.get("crop", {}))
    expected_count_backup = module.CONFIG.get("expected_count")
    scaled_crop_cfg = copy.deepcopy(cfg["crop"])
    if scale < 1.0:
        for key in ["x", "y", "w", "h"]:
            scaled_crop_cfg[key] = int(round(float(scaled_crop_cfg.get(key, 0)) * scale))

    try:
        module.CONFIG["crop"] = scaled_crop_cfg
        module.CONFIG["expected_count"] = int(cfg["expected_count"])
        roi_image, offset_xy = module.crop_work_roi(working_image)
        gray, edges, _raw_candidates, circles, _common_radius = module.detect_stator_centers(roi_image)
    finally:
        module.CONFIG["crop"] = crop_cfg
        module.CONFIG["expected_count"] = expected_count_backup

    ox, oy = offset_xy
    scaled_back_circles = []

    for cx, cy, radius, score in circles:
        gx = int(round(float(cx + ox) / scale))
        gy = int(round(float(cy + oy) / scale))
        gr = int(round(float(radius) / scale))
        scaled_back_circles.append((gx, gy, gr, float(score)))

    scaled_back_circles = sorted(scaled_back_circles, key=lambda item: (item[1], item[0]))
    return scaled_back_circles, gray, edges, scale


def crop_stator_roi(image, cx, cy, radius, cfg):
    h, w = image.shape[:2]
    half = int(max(1, round(float(radius) * cfg["roi"]["half_size_scale"])))

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    roi = image[y1:y2, x1:x2].copy()
    return roi, (x1, y1, x2, y2)


def group_circles_by_row(circles, row_gap):
    if not circles:
        return []

    rows = []
    sorted_circles = sorted(circles, key=lambda item: (item[1], item[0]))

    for circle in sorted_circles:
        if not rows:
            rows.append([circle])
            continue

        row_mean_y = float(np.mean([item[1] for item in rows[-1]]))
        if abs(circle[1] - row_mean_y) <= row_gap:
            rows[-1].append(circle)
        else:
            rows.append([circle])

    return [sorted(row, key=lambda item: item[0]) for row in rows]


def infer_single_missing_grid_position(circles, expected_count):
    if expected_count != 12 or len(circles) != 11:
        return None

    median_radius = float(np.median([item[2] for item in circles]))
    row_gap = max(140.0, median_radius * 1.45)
    rows = group_circles_by_row(circles, row_gap=row_gap)
    if len(rows) != 3:
        return None

    row_lengths = sorted(len(row) for row in rows)
    if row_lengths != [3, 4, 4]:
        return None

    full_rows = [row for row in rows if len(row) == 4]
    missing_row_index = next((idx for idx, row in enumerate(rows) if len(row) == 3), None)
    if missing_row_index is None or len(full_rows) != 2:
        return None

    column_centers = []
    for col_idx in range(4):
        col_xs = [row[col_idx][0] for row in full_rows]
        column_centers.append(float(np.mean(col_xs)))

    missing_row = rows[missing_row_index]
    available_columns = list(range(4))
    col_tol = max(120.0, median_radius * 1.6)

    for circle in missing_row:
        best_col = None
        best_dist = float("inf")
        for col_idx in available_columns:
            dist = abs(circle[0] - column_centers[col_idx])
            if dist < best_dist:
                best_dist = dist
                best_col = col_idx

        if best_col is None or best_dist > col_tol:
            return None

        available_columns.remove(best_col)

    if len(available_columns) != 1:
        return None

    missing_col_idx = available_columns[0]
    predicted_x = int(round(column_centers[missing_col_idx]))

    predicted_y = None
    if 0 < missing_row_index < len(rows) - 1:
        upper_row = rows[missing_row_index - 1]
        lower_row = rows[missing_row_index + 1]
        if len(upper_row) == 4 and len(lower_row) == 4:
            predicted_y = int(round((upper_row[missing_col_idx][1] + lower_row[missing_col_idx][1]) / 2.0))

    if predicted_y is None:
        predicted_y = int(round(float(np.mean([item[1] for item in missing_row]))))

    return {
        "x": predicted_x,
        "y": predicted_y,
        "radius": int(round(median_radius)),
    }


def search_missing_circle_local(image, module, existing_circles, predicted_center, target_radius):
    if not hasattr(module, "_collect_candidates") or not hasattr(module, "preprocess_image"):
        return None

    px, py = predicted_center
    patch_half = int(max(target_radius * 2.2, 380))
    h, w = image.shape[:2]
    x1 = max(0, px - patch_half)
    y1 = max(0, py - patch_half)
    x2 = min(w, px + patch_half)
    y2 = min(h, py + patch_half)

    patch = image[y1:y2, x1:x2].copy()
    if patch.size == 0:
        return None

    gray = module.preprocess_image(patch)
    edges = cv2.Canny(
        gray,
        int(module.CONFIG["canny"]["threshold1"]),
        int(module.CONFIG["canny"]["threshold2"]),
    )

    hcfg = module.CONFIG["hough"]
    min_r = max(20, int(round(target_radius * 0.8)))
    max_r = max(min_r + 2, int(round(target_radius * 1.18)))
    min_dist = max(50, int(round(target_radius * 1.05)))
    p2_base = int(hcfg["param2"])
    p2_values = [
        max(12, p2_base - 16),
        max(14, p2_base - 10),
        max(16, p2_base - 6),
        p2_base,
    ]
    relaxed_edge = max(0.10, float(hcfg["edge_score_threshold"]) * 0.55)

    candidates = module._collect_candidates(
        gray,
        edges,
        min_r,
        max_r,
        min_dist,
        p2_values,
        relaxed_edge,
        True,
    )
    if not candidates:
        candidates = module._collect_candidates(
            gray,
            edges,
            min_r,
            max_r,
            min_dist,
            p2_values,
            0.0,
            False,
        )

    best_candidate = None
    best_score = float("-inf")
    for cx, cy, radius, score in candidates:
        gx = int(cx + x1)
        gy = int(cy + y1)
        if any(math.hypot(gx - ex, gy - ey) < target_radius * 1.1 for ex, ey, _er, _es in existing_circles):
            continue

        center_penalty = math.hypot(gx - px, gy - py) / max(1.0, float(target_radius))
        radius_penalty = abs(radius - target_radius) / max(1.0, float(target_radius))
        final_score = float(score) - (0.055 * center_penalty) - (0.03 * radius_penalty)

        if final_score > best_score:
            best_score = final_score
            best_candidate = (gx, gy, int(radius), float(score))

    if best_candidate is None:
        return None

    if best_candidate[3] < 0.12:
        return None

    return best_candidate


def maybe_recover_missing_circle(image, circles, cfg):
    expected_count = int(cfg["expected_count"])
    if not bool(cfg["detection"]["use_fast_mode"]):
        return circles, False
    if len(circles) != max(1, expected_count - 1):
        return circles, False

    missing_info = infer_single_missing_grid_position(circles, expected_count)
    if missing_info is None:
        return circles, False

    module = load_hough_circle_module()
    recovered = search_missing_circle_local(
        image,
        module,
        circles,
        predicted_center=(missing_info["x"], missing_info["y"]),
        target_radius=missing_info["radius"],
    )
    if recovered is None:
        return circles, False

    merged = list(circles) + [recovered]
    merged = module.dedup_circles(
        merged,
        max(70, int(round(missing_info["radius"] * 0.95))),
        preferred_radius=float(missing_info["radius"]),
    )
    merged = sorted(merged, key=lambda item: (item[1], item[0]))

    if len(merged) <= len(circles):
        return circles, False

    return merged[:expected_count], True


def resolve_selected_roi_entry(roi_list, preview_index):
    if not roi_list:
        return None

    resolved_index = max(1, min(int(preview_index), len(roi_list)))
    stator_id, roi_bgr, bbox, circle_info = roi_list[resolved_index - 1]
    return resolved_index, stator_id, roi_bgr, bbox, circle_info


def build_selected_roi_preview(roi_list, preview_index, cfg=None):
    del cfg
    selected_entry = resolve_selected_roi_entry(roi_list, preview_index)
    if selected_entry is None:
        return None

    preview_index, stator_id, roi_bgr, bbox, circle_info = selected_entry
    return {
        "roi_index": preview_index,
        "stator_id": stator_id,
        "bbox": bbox,
        "circle_info": circle_info,
        "roi_bgr": roi_bgr.copy(),
    }


def build_report_figure(display_image, roi_list, cfg):
    preview_cfg = cfg["preview"]
    display_preview = resize_for_display(
        display_image,
        max_width=int(preview_cfg["overview_max_width"]),
        max_height=int(preview_cfg["overview_max_height"]),
    )

    num_roi_show = len(roi_list)
    roi_rows = min(4, max(1, num_roi_show))
    roi_cols = max(1, int(math.ceil(float(num_roi_show) / float(roi_rows))))

    fig_width = 7.0 + (2.9 * roi_cols)
    fig_height = max(6.0, 2.6 * roi_rows)

    fig = Figure(figsize=(fig_width, fig_height))
    grid = gridspec.GridSpec(
        roi_rows,
        roi_cols + 1,
        figure=fig,
        width_ratios=[1.8] + [1.0] * roi_cols,
        wspace=0.04,
        hspace=0.12,
    )

    ax_overview = fig.add_subplot(grid[:, 0])
    ax_overview.imshow(cv2.cvtColor(display_preview, cv2.COLOR_BGR2RGB))
    ax_overview.set_title("(a) Overview with Hough Circle and ROI", pad=6)
    ax_overview.axis("off")

    for i, (idx, roi, _bbox, _circle_info) in enumerate(roi_list):
        roi_preview = resize_for_display(
            roi,
            max_width=int(preview_cfg["roi_max_size"]),
            max_height=int(preview_cfg["roi_max_size"]),
        )
        roi_rgb = cv2.cvtColor(roi_preview, cv2.COLOR_BGR2RGB)
        row_idx = i % roi_rows
        col_idx = (i // roi_rows) + 1

        ax_roi = fig.add_subplot(grid[row_idx, col_idx])
        ax_roi.imshow(roi_rgb)
        ax_roi.set_title("(b{}) ROI stator {}".format(i + 1, idx), fontsize=10, pad=4)
        ax_roi.axis("off")

    fig.subplots_adjust(left=0.02, right=0.99, top=0.96, bottom=0.03)
    return fig


def build_roi_gallery_figure(outputs, cfg, show_all):
    roi_list = outputs["roi_list"]
    preview_cfg = cfg["preview"]
    overview = resize_for_display(
        outputs["display"],
        max_width=int(preview_cfg["overview_max_width"]),
        max_height=int(preview_cfg["overview_max_height"]),
    )

    if show_all:
        roi_entries = [
            (slot_idx, stator_id, roi_bgr)
            for slot_idx, (stator_id, roi_bgr, _bbox, _circle_info) in enumerate(roi_list, start=1)
        ]
        figure_title = "Tong hop ROI tat ca ID"
    else:
        selected_entry = resolve_selected_roi_entry(roi_list, preview_cfg["selected_roi_index"])
        if selected_entry is None:
            roi_entries = []
            figure_title = "Anh ROI dang chon"
        else:
            slot_idx, stator_id, roi_bgr, _bbox, _circle_info = selected_entry
            roi_entries = [(slot_idx, stator_id, roi_bgr)]
            figure_title = "Anh ROI dang chon"

    num_roi_show = len(roi_entries)
    roi_rows = 2 if num_roi_show > 1 else 1
    roi_cols = max(1, int(math.ceil(float(max(1, num_roi_show)) / float(roi_rows))))
    total_cols = max(3, roi_cols)

    fig = Figure(figsize=(15.6, 9.0))
    grid = gridspec.GridSpec(
        3,
        total_cols,
        figure=fig,
        height_ratios=[1.45, 1.0, 1.0],
        wspace=0.10,
        hspace=0.24,
    )

    ax_overview = fig.add_subplot(grid[0, :])
    ax_overview.imshow(cv2.cvtColor(overview, cv2.COLOR_BGR2RGB))
    ax_overview.set_title(
        "Overview | {} stator | Hough scale {:.3f}{}".format(
            len(outputs["circles"]),
            outputs["detection_scale"],
            " | recovered +1" if outputs["recovered_missing"] else "",
        ),
        fontsize=12,
        pad=8,
    )
    ax_overview.axis("off")

    if not roi_entries:
        ax_empty = fig.add_subplot(grid[1:, :])
        ax_empty.text(
            0.5,
            0.5,
            "Chua co ROI de hien thi",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax_empty.axis("off")
        fig.suptitle(figure_title, fontsize=14, y=0.99)
        fig.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.03)
        return fig

    if not show_all and len(roi_entries) == 1:
        slot_idx, stator_id, roi_bgr = roi_entries[0]
        ax_roi = fig.add_subplot(grid[1:, :])
        roi_preview = resize_for_display(
            roi_bgr,
            max_width=int(preview_cfg["roi_max_size"] * 2.4),
            max_height=int(preview_cfg["roi_max_size"] * 1.9),
        )
        ax_roi.imshow(cv2.cvtColor(roi_preview, cv2.COLOR_BGR2RGB))
        ax_roi.set_title("ROI ID{} (slot {})".format(stator_id, slot_idx), fontsize=10, pad=5)
        ax_roi.axis("off")
    else:
        for i, (slot_idx, stator_id, roi_bgr) in enumerate(roi_entries):
            row_idx = 1 + (i // roi_cols)
            col_idx = i % roi_cols
            ax_roi = fig.add_subplot(grid[row_idx, col_idx])
            roi_preview = resize_for_display(
                roi_bgr,
                max_width=int(preview_cfg["roi_max_size"]),
                max_height=int(preview_cfg["roi_max_size"]),
            )
            ax_roi.imshow(cv2.cvtColor(roi_preview, cv2.COLOR_BGR2RGB))
            ax_roi.set_title("ROI ID{} (slot {})".format(stator_id, slot_idx), fontsize=10, pad=5)
            ax_roi.axis("off")

        for i in range(len(roi_entries), roi_rows * roi_cols):
            row_idx = 1 + (i // roi_cols)
            col_idx = i % roi_cols
            ax_empty = fig.add_subplot(grid[row_idx, col_idx])
            ax_empty.axis("off")

    fig.suptitle(figure_title, fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.03)
    return fig


def build_gui_figure(outputs, cfg):
    if bool(cfg["preview"].get("show_all_rois", True)):
        return build_roi_gallery_figure(outputs, cfg, show_all=True)
    return build_roi_gallery_figure(outputs, cfg, show_all=False)


def print_processing_summary(outputs):
    print("So stator phat hien duoc: {}".format(len(outputs["circles"])))
    print("Danh sach stator phat hien duoc:")
    for idx, (cx, cy, radius, score) in enumerate(outputs["circles"], start=1):
        print(
            "ID{idx:02d}: center=({cx}, {cy}), radius={radius}, edge_score={score:.3f}".format(
                idx=idx,
                cx=cx,
                cy=cy,
                radius=radius,
                score=score,
            )
        )

    if outputs["save_outputs"]:
        print("\nDa luu anh toan canh co danh dau ROI tai:")
        print(outputs["marked_path"])
        print("\nDa luu anh ghep dung cho bao cao tai:")
        print(outputs["compare_path"])
        print("\nCac ROI rieng duoc luu tai:")
        print(outputs["roi_output_dir"])


def process_image(
    input_path,
    output_dir=None,
    save_outputs=True,
    status_callback=None,
    attach_figure=True,
    config_override=None,
    preview_roi_index=None,
    include_preview=True,
):
    cfg = normalize_config(config_override if config_override is not None else CONFIG)

    def report_status(message):
        if status_callback is not None:
            status_callback(message)

    if not input_path:
        raise ValueError("Chua co duong dan anh dau vao.")

    if not os.path.isfile(input_path):
        raise FileNotFoundError("Khong tim thay anh dau vao: {}".format(input_path))

    if output_dir is None:
        output_dir = cfg["output_dir"]

    output_dir = os.path.abspath(output_dir)
    paths = get_output_paths(output_dir)
    ensure_dir(output_dir)
    ensure_dir(paths["roi_output_dir"])

    report_status("Dang doc anh dau vao...")
    image = read_image(input_path)
    if image is None:
        raise ValueError("Khong doc duoc anh: {}".format(input_path))

    report_status("Dang tim tam stator bang Hough...")
    circles, gray, edges, detection_scale = detect_stator_center_from_hough(input_path, image, output_dir, cfg)
    if not circles:
        raise RuntimeError("Khong tim thay stator nao de cat ROI.")

    report_status("Dang kiem tra bo sung neu thieu stator...")
    circles, recovered_missing = maybe_recover_missing_circle(image, circles, cfg)

    report_status("Dang cat ROI quanh tung stator...")
    display = image.copy()
    roi_list = []

    for idx, (cx, cy, radius, score) in enumerate(circles, start=1):
        roi, bbox = crop_stator_roi(image, cx, cy, radius, cfg)
        x1, y1, x2, y2 = bbox

        roi_list.append((idx, roi, bbox, (cx, cy, radius, score)))

        if save_outputs:
            roi_path = os.path.join(paths["roi_output_dir"], "roi_stator_{:02d}.png".format(idx))
            write_image(roi_path, roi)

        cv2.circle(display, (cx, cy), radius, (0, 255, 0), 2)
        cv2.drawMarker(display, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            display,
            "ID{}".format(idx),
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            display,
            "s={:.2f}".format(score),
            (x1, min(display.shape[0] - 5, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

    preview_data = None
    if include_preview:
        report_status("Dang tao preview ROI da chon...")
        resolved_index = preview_roi_index if preview_roi_index is not None else cfg["preview"]["selected_roi_index"]
        preview_data = build_selected_roi_preview(roi_list, resolved_index, cfg)

    figure = None
    if save_outputs or attach_figure:
        report_status("Dang tao preview tong hop...")
        figure = build_report_figure(display, roi_list, cfg)

    if save_outputs:
        report_status("Dang luu anh ket qua...")
        write_image(paths["marked_path"], display)
        if figure is None:
            figure = build_report_figure(display, roi_list, cfg)
        figure.savefig(paths["compare_path"], dpi=300, bbox_inches="tight", pad_inches=0.05)

    outputs = {
        "input_path": input_path,
        "output_dir": output_dir,
        "roi_output_dir": paths["roi_output_dir"],
        "marked_path": paths["marked_path"],
        "compare_path": paths["compare_path"],
        "save_outputs": save_outputs,
        "detection_scale": detection_scale,
        "recovered_missing": recovered_missing,
        "image": image,
        "gray": gray,
        "edges": edges,
        "display": display,
        "circles": circles,
        "roi_list": roi_list,
        "preview_roi": preview_data,
        "config": cfg,
        "figure": figure if attach_figure else None,
    }
    if save_outputs:
        print_processing_summary(outputs)
    return outputs


def save_preset_file(path, cfg):
    ensure_dir(os.path.dirname(path))
    payload = {
        "schema_version": PRESET_SCHEMA_VERSION,
        "config": normalize_config(cfg),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_preset_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Preset JSON khong hop le.")

    raw_cfg = payload.get("config", payload)
    return normalize_config(raw_cfg)


def run_gui():
    global CONFIG

    CONFIG = normalize_config(CONFIG)
    ensure_dir(PRESET_DIR)

    root = tk.Tk()
    root.title("Cat ROI quanh tung stator")
    root.geometry("1560x950")
    root.minsize(1280, 820)

    left_shell = tk.Frame(root, width=430, bg="#f3f4f6")
    left_shell.pack(side="left", fill="y")
    left_shell.pack_propagate(False)

    right_panel = tk.Frame(root, padx=8, pady=8)
    right_panel.pack(side="right", fill="both", expand=True)

    canvas_holder = tk.Frame(right_panel, bg="white", bd=1, relief="solid")
    canvas_holder.pack(fill="both", expand=True)

    placeholder = tk.Label(
        canvas_holder,
        text="Overview va ROI cat duoc se hien thi o day.\nKeo slider hoac bam Preview de cap nhat.",
        bg="white",
        fg="#666666",
        font=("Arial", 12),
        justify="center",
    )
    placeholder.pack(expand=True)

    left_canvas = tk.Canvas(left_shell, bg="#f3f4f6", highlightthickness=0)
    left_scrollbar = tk.Scrollbar(left_shell, orient="vertical", command=left_canvas.yview)
    left_canvas.configure(yscrollcommand=left_scrollbar.set)
    left_scrollbar.pack(side="right", fill="y")
    left_canvas.pack(side="left", fill="both", expand=True)

    controls_frame = tk.Frame(left_canvas, bg="#f3f4f6", padx=14, pady=14)
    canvas_window = left_canvas.create_window((0, 0), window=controls_frame, anchor="nw")

    controls_frame.bind(
        "<Configure>",
        lambda _e: left_canvas.configure(scrollregion=left_canvas.bbox("all")),
    )
    left_canvas.bind(
        "<Configure>",
        lambda event: left_canvas.itemconfigure(canvas_window, width=event.width),
    )

    def pointer_inside_left_panel():
        try:
            pointer_x, pointer_y = root.winfo_pointerxy()
            hovered_widget = root.winfo_containing(pointer_x, pointer_y)
        except tk.TclError:
            return False

        while hovered_widget is not None:
            if hovered_widget == left_shell:
                return True
            hovered_widget = hovered_widget.master
        return False

    def on_left_panel_mousewheel(event):
        if not pointer_inside_left_panel():
            return None

        if getattr(event, "delta", 0):
            steps = -int(event.delta / 120) if event.delta % 120 == 0 else (-1 if event.delta > 0 else 1)
        elif getattr(event, "num", None) == 4:
            steps = -1
        elif getattr(event, "num", None) == 5:
            steps = 1
        else:
            steps = 0

        if steps == 0:
            return None

        left_canvas.yview_scroll(steps, "units")
        return "break"

    root.bind_all("<MouseWheel>", on_left_panel_mousewheel, add="+")
    root.bind_all("<Button-4>", on_left_panel_mousewheel, add="+")
    root.bind_all("<Button-5>", on_left_panel_mousewheel, add="+")

    state = {
        "canvas": None,
        "pending_after_id": None,
        "suspend_events": False,
        "current_image_shape": None,
        "last_outputs": None,
        "last_config": normalize_config(CONFIG),
    }
    worker_state = {
        "running": False,
        "pending_request": None,
    }
    event_queue = queue.Queue()
    slider_widgets = {}

    selected_path_var = tk.StringVar(value=CONFIG["input_path"])
    status_text = tk.StringVar(value="Chon anh dau vao, chinh tham so Hough + ROI, sau do xem ket qua cat ROI.")
    image_info_text = tk.StringVar(value="Kich thuoc anh: chua nap")
    output_info_text = tk.StringVar(value=CONFIG["output_dir"])

    vars_gui = {
        "auto_update": tk.BooleanVar(value=bool(CONFIG["preview"]["auto_update"])),
        "use_fast_mode": tk.BooleanVar(value=bool(CONFIG["detection"]["use_fast_mode"])),
        "expected_count": tk.IntVar(value=int(CONFIG["expected_count"])),
        "max_processing_dim": tk.IntVar(value=int(CONFIG["detection"]["max_processing_dim"])),
        "crop_x": tk.IntVar(value=int(CONFIG["crop"]["x"])),
        "crop_y": tk.IntVar(value=int(CONFIG["crop"]["y"])),
        "crop_w": tk.IntVar(value=int(CONFIG["crop"]["w"])),
        "crop_h": tk.IntVar(value=int(CONFIG["crop"]["h"])),
        "half_size_scale": tk.DoubleVar(value=float(CONFIG["roi"]["half_size_scale"])),
        "selected_roi_index": tk.IntVar(value=int(CONFIG["preview"]["selected_roi_index"])),
        "show_all_rois": tk.BooleanVar(value=bool(CONFIG["preview"]["show_all_rois"])),
        "roi_selector_value": tk.StringVar(value=ALL_ROI_SELECTOR_LABEL),
    }
    roi_selector_combo = None

    def clear_canvas():
        if state["canvas"] is not None:
            state["canvas"].figure.clear()
            state["canvas"].get_tk_widget().destroy()
            state["canvas"] = None

    def show_placeholder():
        if state["canvas"] is None:
            placeholder.pack(expand=True)

    def hide_placeholder():
        if placeholder.winfo_exists():
            placeholder.pack_forget()

    def set_running(is_running):
        worker_state["running"] = is_running
        busy_state = "disabled" if is_running else "normal"
        run_button.config(state=busy_state)
        save_button.config(state=busy_state)
        preview_button.config(state=busy_state)
        browse_button.config(state=busy_state)
        preset_save_button.config(state=busy_state)
        preset_load_button.config(state=busy_state)
        reset_button.config(state=busy_state)
        path_entry.config(state=busy_state)
        if roi_selector_combo is not None:
            roi_selector_combo.config(state="disabled" if is_running else "readonly")

    def read_image_shape(path):
        if not path or not os.path.isfile(path):
            return None
        image = read_image(path)
        if image is None:
            return None
        return image.shape[:2]

    def update_slider_limit(key, max_value):
        widget = slider_widgets.get(key)
        if widget is None:
            return
        widget.configure(to=max_value)

    def resolve_roi_count(count_hint=None):
        if isinstance(count_hint, list):
            return max(1, len(count_hint))
        if count_hint is None and state["last_outputs"] is not None:
            return max(1, len(state["last_outputs"]["circles"]))
        if count_hint is None:
            return max(1, int(vars_gui["expected_count"].get()))
        return max(1, int(count_hint))

    def sync_roi_selector(count_hint=None):
        nonlocal roi_selector_combo
        max_count = resolve_roi_count(count_hint)
        values = [ALL_ROI_SELECTOR_LABEL] + ["ID{}".format(idx) for idx in range(1, max_count + 1)]

        if roi_selector_combo is not None:
            roi_selector_combo.configure(values=values)

        resolved_index = min(max(1, vars_gui["selected_roi_index"].get()), max_count)
        if vars_gui["selected_roi_index"].get() != resolved_index:
            state["suspend_events"] = True
            vars_gui["selected_roi_index"].set(resolved_index)
            state["suspend_events"] = False

        selector_label = ALL_ROI_SELECTOR_LABEL if vars_gui["show_all_rois"].get() else "ID{}".format(resolved_index)
        if vars_gui["roi_selector_value"].get() != selector_label:
            state["suspend_events"] = True
            vars_gui["roi_selector_value"].set(selector_label)
            state["suspend_events"] = False

    def update_preview_index_limit(count_hint=None):
        max_count = resolve_roi_count(count_hint)
        if vars_gui["selected_roi_index"].get() > max_count:
            state["suspend_events"] = True
            vars_gui["selected_roi_index"].set(max_count)
            state["suspend_events"] = False
        sync_roi_selector(max_count)

    def update_image_metadata(path):
        shape = read_image_shape(path)
        state["current_image_shape"] = shape
        if shape is None:
            image_info_text.set("Kich thuoc anh: khong doc duoc")
            return

        h, w = shape
        image_info_text.set("Kich thuoc anh: {} x {} px".format(w, h))
        update_slider_limit("crop_x", max(0, w))
        update_slider_limit("crop_y", max(0, h))
        update_slider_limit("crop_w", max(0, w))
        update_slider_limit("crop_h", max(0, h))

        state["suspend_events"] = True
        vars_gui["crop_x"].set(min(vars_gui["crop_x"].get(), w))
        vars_gui["crop_y"].set(min(vars_gui["crop_y"].get(), h))
        vars_gui["crop_w"].set(min(vars_gui["crop_w"].get(), w))
        vars_gui["crop_h"].set(min(vars_gui["crop_h"].get(), h))
        state["suspend_events"] = False

    def collect_config_from_gui():
        raw_cfg = {
            "input_path": selected_path_var.get().strip(),
            "output_dir": output_info_text.get().strip() or DEFAULT_OUTPUT_DIR,
            "expected_count": vars_gui["expected_count"].get(),
            "crop": {
                "x": vars_gui["crop_x"].get(),
                "y": vars_gui["crop_y"].get(),
                "w": vars_gui["crop_w"].get(),
                "h": vars_gui["crop_h"].get(),
            },
            "roi": {
                "half_size_scale": vars_gui["half_size_scale"].get(),
            },
            "detection": {
                "use_fast_mode": vars_gui["use_fast_mode"].get(),
                "max_processing_dim": vars_gui["max_processing_dim"].get(),
            },
            "preview": {
                "overview_max_width": state["last_config"]["preview"]["overview_max_width"],
                "overview_max_height": state["last_config"]["preview"]["overview_max_height"],
                "roi_max_size": state["last_config"]["preview"]["roi_max_size"],
                "selected_roi_index": vars_gui["selected_roi_index"].get(),
                "show_all_rois": vars_gui["show_all_rois"].get(),
                "auto_update": vars_gui["auto_update"].get(),
                "auto_update_delay_ms": state["last_config"]["preview"]["auto_update_delay_ms"],
            },
        }
        cfg = normalize_config(raw_cfg)
        state["last_config"] = cfg
        return cfg

    def load_vars_from_config(cfg):
        state["suspend_events"] = True
        selected_path_var.set(cfg["input_path"])
        output_info_text.set(cfg["output_dir"])
        vars_gui["auto_update"].set(bool(cfg["preview"]["auto_update"]))
        vars_gui["use_fast_mode"].set(bool(cfg["detection"]["use_fast_mode"]))
        vars_gui["expected_count"].set(int(cfg["expected_count"]))
        vars_gui["max_processing_dim"].set(int(cfg["detection"]["max_processing_dim"]))
        vars_gui["crop_x"].set(int(cfg["crop"]["x"]))
        vars_gui["crop_y"].set(int(cfg["crop"]["y"]))
        vars_gui["crop_w"].set(int(cfg["crop"]["w"]))
        vars_gui["crop_h"].set(int(cfg["crop"]["h"]))
        vars_gui["half_size_scale"].set(float(cfg["roi"]["half_size_scale"]))
        vars_gui["selected_roi_index"].set(int(cfg["preview"]["selected_roi_index"]))
        vars_gui["show_all_rois"].set(bool(cfg["preview"]["show_all_rois"]))
        state["last_config"] = normalize_config(cfg)
        state["suspend_events"] = False
        update_image_metadata(cfg["input_path"])
        update_preview_index_limit()

    def build_status_summary(outputs, save_outputs):
        if outputs["config"]["preview"]["show_all_rois"]:
            preview_suffix = " | dang xem tat ca ROI"
        else:
            preview_roi = outputs.get("preview_roi")
            preview_suffix = " | preview ROI ID{}".format(preview_roi["stator_id"]) if preview_roi is not None else ""
        save_suffix = " | da luu ROI/output" if save_outputs else " | preview only"
        recover_suffix = " | da bo sung 1 stator" if outputs["recovered_missing"] else ""
        return (
            "Xong. {} stator | Hough scale {:.3f}{}".format(
                len(outputs["circles"]),
                outputs["detection_scale"],
                recover_suffix,
            )
            + preview_suffix
            + save_suffix
        )

    def render_outputs(outputs):
        clear_canvas()
        figure = build_gui_figure(outputs, outputs["config"])
        hide_placeholder()
        canvas = FigureCanvasTkAgg(figure, master=canvas_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        state["canvas"] = canvas

    def refresh_roi_view_only():
        if state["last_outputs"] is None:
            return

        display_cfg = copy.deepcopy(state["last_outputs"]["config"])
        display_cfg["preview"]["selected_roi_index"] = vars_gui["selected_roi_index"].get()
        display_cfg["preview"]["show_all_rois"] = vars_gui["show_all_rois"].get()
        state["last_config"]["preview"]["selected_roi_index"] = display_cfg["preview"]["selected_roi_index"]
        state["last_config"]["preview"]["show_all_rois"] = display_cfg["preview"]["show_all_rois"]
        render_outputs(
            {
                **state["last_outputs"],
                "config": display_cfg,
            }
        )

    def finish_processing():
        set_running(False)
        pending = worker_state["pending_request"]
        worker_state["pending_request"] = None
        if pending is not None:
            start_processing(**pending)

    def handle_success(payload):
        outputs = payload["outputs"]
        state["last_outputs"] = outputs
        render_outputs(outputs)
        update_preview_index_limit(outputs["circles"])
        status_text.set(build_status_summary(outputs, payload["save_outputs"]))
        finish_processing()

    def handle_error(payload):
        if payload.get("interactive", False):
            messagebox.showerror("Loi xu ly", payload["message"])
        status_text.set(payload["message"])
        show_placeholder()
        finish_processing()

    def poll_worker_events():
        try:
            while True:
                event_type, payload = event_queue.get_nowait()
                if event_type == "status":
                    status_text.set(payload)
                elif event_type == "success":
                    handle_success(payload)
                elif event_type == "error":
                    handle_error(payload)
        except queue.Empty:
            pass
        root.after(80, poll_worker_events)

    def start_processing(save_outputs, reason, interactive):
        global CONFIG
        if worker_state["running"]:
            pending = worker_state["pending_request"]
            worker_state["pending_request"] = {
                "save_outputs": bool(save_outputs) or (bool(pending["save_outputs"]) if pending else False),
                "reason": reason,
                "interactive": bool(interactive) or (bool(pending["interactive"]) if pending else False),
            }
            return

        input_path = selected_path_var.get().strip()
        if not input_path or not os.path.isfile(input_path):
            message = "Khong tim thay anh dau vao. Vui long chon lai file anh."
            if interactive:
                messagebox.showerror("Sai duong dan", message)
            status_text.set(message)
            return

        update_image_metadata(input_path)
        cfg = collect_config_from_gui()
        CONFIG = copy.deepcopy(cfg)
        set_running(True)
        status_text.set("Dang xu ly {}...".format(reason))
        root.update_idletasks()

        def worker():
            try:
                outputs = process_image(
                    input_path,
                    cfg["output_dir"],
                    save_outputs=save_outputs,
                    status_callback=lambda message: event_queue.put(("status", message)),
                    attach_figure=False,
                    config_override=cfg,
                    preview_roi_index=cfg["preview"]["selected_roi_index"],
                    include_preview=True,
                )
                event_queue.put(
                    (
                        "success",
                        {
                            "outputs": outputs,
                            "save_outputs": save_outputs,
                            "reason": reason,
                        },
                    )
                )
            except Exception as exc:
                event_queue.put(
                    (
                        "error",
                        {
                            "message": "Xu ly that bai: {}".format(exc),
                            "interactive": interactive,
                            "reason": reason,
                        },
                    )
                )

        threading.Thread(target=worker, daemon=True).start()

    def request_processing(save_outputs=False, reason="preview", interactive=False):
        if worker_state["running"]:
            pending = worker_state["pending_request"]
            if pending is None:
                worker_state["pending_request"] = {
                    "save_outputs": bool(save_outputs),
                    "reason": reason,
                    "interactive": bool(interactive),
                }
            else:
                pending["save_outputs"] = bool(pending["save_outputs"]) or bool(save_outputs)
                pending["interactive"] = bool(pending["interactive"]) or bool(interactive)
                pending["reason"] = reason
            status_text.set("Dang xu ly... se cap nhat lai khi job hien tai xong.")
            return
        start_processing(save_outputs=save_outputs, reason=reason, interactive=interactive)

    def schedule_auto_preview():
        if state["suspend_events"]:
            return
        if not vars_gui["auto_update"].get():
            return
        if state["pending_after_id"] is not None:
            root.after_cancel(state["pending_after_id"])
            state["pending_after_id"] = None

        delay_ms = int(state["last_config"]["preview"]["auto_update_delay_ms"])

        def _run():
            state["pending_after_id"] = None
            request_processing(save_outputs=False, reason="live preview", interactive=False)

        state["pending_after_id"] = root.after(delay_ms, _run)

    def on_control_change(*_args):
        if state["suspend_events"]:
            return
        update_preview_index_limit()
        schedule_auto_preview()

    def on_path_committed(*_args):
        if state["suspend_events"]:
            return
        update_image_metadata(selected_path_var.get().strip())
        schedule_auto_preview()

    def browse_image():
        file_path = filedialog.askopenfilename(
            title="Chon anh dau vao",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            selected_path_var.set(file_path)
            update_image_metadata(file_path)
            status_text.set("Da chon anh moi. Live preview se cap nhat.")
            schedule_auto_preview()

    def preview_now():
        request_processing(save_outputs=False, reason="preview thu cong", interactive=True)

    def save_outputs_now():
        request_processing(save_outputs=True, reason="luu ROI/output", interactive=True)

    def save_preset():
        try:
            cfg = collect_config_from_gui()
            initial_dir = PRESET_DIR
            ensure_dir(initial_dir)
            initial_name = "roi_cut_setting.json"
            save_path = filedialog.asksaveasfilename(
                title="Luu preset ROI",
                initialdir=initial_dir,
                initialfile=initial_name,
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            )
            if not save_path:
                return
            save_preset_file(save_path, cfg)
            status_text.set("Da luu preset ROI tai: {}".format(save_path))
        except Exception as exc:
            messagebox.showerror("Loi luu preset", str(exc))

    def load_preset():
        try:
            preset_path = filedialog.askopenfilename(
                title="Nap preset ROI",
                initialdir=PRESET_DIR,
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            )
            if not preset_path:
                return
            cfg = load_preset_file(preset_path)
            load_vars_from_config(cfg)
            status_text.set("Da nap preset ROI. Live preview se cap nhat.")
            schedule_auto_preview()
        except Exception as exc:
            messagebox.showerror("Loi nap preset", str(exc))

    def reset_defaults():
        load_vars_from_config(copy.deepcopy(DEFAULT_CONFIG))
        status_text.set("Da reset ve thong so mac dinh.")
        schedule_auto_preview()

    def on_roi_selector_change(_event=None):
        if state["suspend_events"]:
            return

        choice = vars_gui["roi_selector_value"].get().strip()
        state["suspend_events"] = True
        if choice == ALL_ROI_SELECTOR_LABEL or not choice:
            vars_gui["show_all_rois"].set(True)
        else:
            try:
                selected_idx = int(choice.replace("ID", "").strip())
            except ValueError:
                selected_idx = max(1, vars_gui["selected_roi_index"].get())
            vars_gui["show_all_rois"].set(False)
            vars_gui["selected_roi_index"].set(max(1, selected_idx))
        state["suspend_events"] = False
        sync_roi_selector()

        if state["last_outputs"] is not None:
            refresh_roi_view_only()
        else:
            schedule_auto_preview()

    def add_slider(parent, key, label_text, frm, to, resolution=1):
        row = tk.Frame(parent, bg="#f3f4f6")
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label_text, anchor="w", bg="#f3f4f6").pack(fill="x")
        scale = tk.Scale(
            row,
            from_=frm,
            to=to,
            orient="horizontal",
            resolution=resolution,
            variable=vars_gui[key],
            showvalue=True,
            bg="#f3f4f6",
            highlightthickness=0,
            command=lambda _v: on_control_change(),
        )
        scale.pack(fill="x")
        slider_widgets[key] = scale
        return scale

    def add_checkbox(parent, key, label_text):
        check = tk.Checkbutton(
            parent,
            text=label_text,
            variable=vars_gui[key],
            bg="#f3f4f6",
            command=on_control_change,
        )
        check.pack(anchor="w", pady=2)
        return check

    tk.Label(
        controls_frame,
        text="Cat ROI quanh tung stator",
        font=("Arial", 15, "bold"),
        bg="#f3f4f6",
        anchor="w",
    ).pack(fill="x", pady=(0, 10))

    source_frame = tk.LabelFrame(controls_frame, text="1. Anh dau vao", bg="#f3f4f6", padx=8, pady=8)
    source_frame.pack(fill="x", pady=(0, 10))

    path_entry = tk.Entry(source_frame, textvariable=selected_path_var)
    path_entry.pack(fill="x", pady=(0, 6))
    path_entry.bind("<Return>", on_path_committed)
    path_entry.bind("<FocusOut>", on_path_committed)

    browse_button = tk.Button(source_frame, text="Import anh", command=browse_image)
    browse_button.pack(fill="x")

    tk.Label(
        source_frame,
        textvariable=image_info_text,
        justify="left",
        anchor="w",
        bg="#f3f4f6",
        fg="#444444",
    ).pack(fill="x", pady=(6, 0))

    add_checkbox(source_frame, "auto_update", "Auto update khi keo slider")
    add_checkbox(source_frame, "use_fast_mode", "Fast mode cho anh lon")

    action_frame = tk.LabelFrame(controls_frame, text="2. Thao tac", bg="#f3f4f6", padx=8, pady=8)
    action_frame.pack(fill="x", pady=(0, 10))

    preview_button = tk.Button(
        action_frame,
        text="Preview ngay",
        command=preview_now,
        bg="#2d89ef",
        fg="white",
        font=("Arial", 10, "bold"),
    )
    preview_button.pack(fill="x", pady=(0, 6))

    run_button = tk.Button(
        action_frame,
        text="Run + luu ROI",
        command=save_outputs_now,
        bg="#1c9d63",
        fg="white",
        font=("Arial", 10, "bold"),
        height=2,
    )
    run_button.pack(fill="x", pady=(0, 6))
    save_button = run_button

    preset_row = tk.Frame(action_frame, bg="#f3f4f6")
    preset_row.pack(fill="x", pady=(0, 6))
    preset_save_button = tk.Button(preset_row, text="Luu preset", command=save_preset)
    preset_save_button.pack(side="left", fill="x", expand=True, padx=(0, 3))
    preset_load_button = tk.Button(preset_row, text="Nap preset", command=load_preset)
    preset_load_button.pack(side="right", fill="x", expand=True, padx=(3, 0))

    reset_button = tk.Button(action_frame, text="Reset mac dinh", command=reset_defaults)
    reset_button.pack(fill="x")

    roi_frame = tk.LabelFrame(controls_frame, text="3. ROI va Hough", bg="#f3f4f6", padx=8, pady=8)
    roi_frame.pack(fill="x", pady=(0, 10))
    add_slider(roi_frame, "expected_count", "So stator ky vong", 1, 24, 1)
    add_slider(roi_frame, "max_processing_dim", "Max processing dim", 640, 3200, 10)
    add_slider(roi_frame, "half_size_scale", "ROI half size scale", 0.8, 2.8, 0.01)
    tk.Label(
        roi_frame,
        text="Crop lon cho Hough (px). Dat w/h = 0 de dung phan con lai.",
        justify="left",
        wraplength=360,
        bg="#f3f4f6",
        fg="#555555",
    ).pack(fill="x", pady=(4, 2))
    add_slider(roi_frame, "crop_x", "Crop x", 0, 5000, 1)
    add_slider(roi_frame, "crop_y", "Crop y", 0, 5000, 1)
    add_slider(roi_frame, "crop_w", "Crop w", 0, 5000, 1)
    add_slider(roi_frame, "crop_h", "Crop h", 0, 5000, 1)

    preview_frame = tk.LabelFrame(controls_frame, text="4. ROI hien thi", bg="#f3f4f6", padx=8, pady=8)
    preview_frame.pack(fill="x", pady=(0, 10))
    selector_row = tk.Frame(preview_frame, bg="#f3f4f6")
    selector_row.pack(fill="x")
    tk.Label(selector_row, text="ID stator", anchor="w", bg="#f3f4f6").pack(side="left")
    roi_selector_combo = ttk.Combobox(
        selector_row,
        textvariable=vars_gui["roi_selector_value"],
        state="readonly",
    )
    roi_selector_combo.pack(side="right", fill="x", expand=True, padx=(8, 0))
    roi_selector_combo.bind("<<ComboboxSelected>>", on_roi_selector_change)
    tk.Label(
        preview_frame,
        text="Chon 'Tat ca ROI' de xem toan bo ROI, hoac chon ID cu the de xem mot ROI.",
        justify="left",
        wraplength=360,
        bg="#f3f4f6",
        fg="#555555",
    ).pack(fill="x", pady=(6, 0))

    output_frame = tk.LabelFrame(controls_frame, text="5. Thu muc ket qua", bg="#f3f4f6", padx=8, pady=8)
    output_frame.pack(fill="x", pady=(0, 10))
    tk.Label(
        output_frame,
        textvariable=output_info_text,
        justify="left",
        wraplength=360,
        bg="#f3f4f6",
        fg="#444444",
    ).pack(fill="x")

    note_frame = tk.LabelFrame(controls_frame, text="6. Ghi chu", bg="#f3f4f6", padx=8, pady=8)
    note_frame.pack(fill="x", pady=(0, 10))
    tk.Label(
        note_frame,
        text=(
            "- Preview live khong ghi file ra dia.\n"
            "- Nut 'Run + luu ROI' moi thuc hien xuat overview + roi_images.\n"
            "- File nay chi cat ROI sau khi HoughCircle nhan dien va hien thi ket qua ROI.\n"
            "- Cac buoc tien xu ly ROI se duoc chuyen sang file 3.dau vao thuat toan xoay.py.\n"
            "- Preset luu lai crop ROI, fast mode, va cach hien thi ROI."
        ),
        justify="left",
        wraplength=360,
        bg="#f3f4f6",
        fg="#444444",
    ).pack(fill="x")

    tk.Label(
        controls_frame,
        textvariable=status_text,
        justify="left",
        wraplength=380,
        anchor="w",
        bg="#f3f4f6",
        fg="#222222",
    ).pack(fill="x", pady=(4, 0))

    for key in [
        "auto_update",
        "use_fast_mode",
        "expected_count",
        "max_processing_dim",
        "crop_x",
        "crop_y",
        "crop_w",
        "crop_h",
        "half_size_scale",
    ]:
        vars_gui[key].trace_add("write", on_control_change)

    update_image_metadata(selected_path_var.get().strip())
    update_preview_index_limit()
    sync_roi_selector()
    poll_worker_events()

    if selected_path_var.get().strip() and os.path.isfile(selected_path_var.get().strip()):
        root.after(250, lambda: request_processing(save_outputs=False, reason="preview ban dau", interactive=False))

    root.mainloop()


def main():
    run_gui()


if __name__ == "__main__":
    main()
