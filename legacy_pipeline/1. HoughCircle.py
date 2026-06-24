# -*- coding: utf-8 -*-
"""
Buoc 1 - HoughCircle

File nay gop:
- 1.1 tien xu ly truoc houghCircle
- 1.2 Hough Circle

Muc tieu:
- Chon anh dau vao.
- Tune rieng nhom tham so tien xu ly truoc Hough va nhom tham so HoughCircle.
- Preview ngay tren giao dien theo phong cach cua 2. cat roi.
- Luu preset tham so. Sau khi luu, 2. cat roi co the nap dung bo tham so nay
  ma khong can do lai.

Luu y:
- Thuat toan detect tam van tai su dung backend tu file 1.2 Hough Circle.py.
- File nay dong vai tro la buoc setup-tham-so va gui-bo-tham-so cho buoc 2.
"""

import contextlib
import copy
import functools
import importlib.util
import json
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from matplotlib import gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BACKEND_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "1.2 Hough Circle.py")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "pipeline", "1_hough_circle")
PRESET_DIR = os.path.join(PROJECT_ROOT, "presets", "hough_circle_settings")
ACTIVE_PRESET_PATH = os.path.join(PRESET_DIR, "_active_hough_circle.json")
PRESET_SCHEMA_VERSION = "hough_circle_setting_v1"

DEFAULT_INPUT_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "test", "test_stator_12.png"),
    os.path.join(PROJECT_ROOT, "data", "test", "test.png"),
    os.path.join(PROJECT_ROOT, "data", "test", "1.png"),
    os.path.join(PROJECT_ROOT, "data", "samples", "1.png"),
]


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
    "preprocess": {
        "enabled": True,
        "clahe_use": True,
        "clahe_clip_limit": 2.5,
        "clahe_tile_grid_size": 8,
        "gaussian_use": True,
        "gaussian_kernel": 5,
    },
    "canny": {
        "threshold1": 70,
        "threshold2": 170,
    },
    "hough": {
        "dp": 1.2,
        "param1": 110,
        "param2": 38,
        "minRadius": 32,
        "maxRadius": 120,
        "minDist": 120,
        "min_center_dist": 90,
        "edge_score_threshold": 0.18,
        "radius_consensus_tol": 6,
        "radius_final_tol": 4,
        "edge_ring_width": 3,
        "use_common_radius_refine": True,
        "common_radius_deviation_tol": 2,
        "radius_refine_band": 4,
        "center_refine_range": 2,
        "radius_penalty": 0.012,
        "center_penalty": 0.003,
        "common_radius_score_gap": 0.035,
        "force_common_radius": True,
    },
    "preview": {
        "overview_max_width": 1700,
        "overview_max_height": 1000,
        "detail_max_width": 520,
        "detail_max_height": 360,
        "auto_update": True,
        "auto_update_delay_ms": 350,
    },
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, image):
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


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _to_bool(value):
    return bool(value)


def _to_odd_int(value, minimum=1, maximum=31):
    value = int(round(float(value)))
    value = _clamp(value, minimum, maximum)
    if value % 2 == 0:
        value = value + 1 if value < maximum else value - 1
    return max(minimum, value)


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

    pre_cfg = cfg["preprocess"]
    pre_cfg["enabled"] = _to_bool(pre_cfg.get("enabled", True))
    pre_cfg["clahe_use"] = _to_bool(pre_cfg.get("clahe_use", True))
    pre_cfg["clahe_clip_limit"] = round(
        _clamp(float(pre_cfg.get("clahe_clip_limit", 2.5)), 0.5, 8.0),
        2,
    )
    pre_cfg["clahe_tile_grid_size"] = _clamp(
        int(round(float(pre_cfg.get("clahe_tile_grid_size", 8)))),
        2,
        32,
    )
    pre_cfg["gaussian_use"] = _to_bool(pre_cfg.get("gaussian_use", True))
    pre_cfg["gaussian_kernel"] = _to_odd_int(pre_cfg.get("gaussian_kernel", 5), minimum=1, maximum=31)

    canny_cfg = cfg["canny"]
    canny_cfg["threshold1"] = _clamp(int(round(float(canny_cfg.get("threshold1", 70)))), 1, 255)
    canny_cfg["threshold2"] = _clamp(int(round(float(canny_cfg.get("threshold2", 170)))), 1, 255)
    if canny_cfg["threshold2"] < canny_cfg["threshold1"]:
        canny_cfg["threshold2"] = canny_cfg["threshold1"]

    hcfg = cfg["hough"]
    hcfg["dp"] = round(_clamp(float(hcfg.get("dp", 1.2)), 0.8, 3.0), 2)
    hcfg["param1"] = _clamp(int(round(float(hcfg.get("param1", 110)))), 1, 255)
    hcfg["param2"] = _clamp(int(round(float(hcfg.get("param2", 38)))), 1, 255)
    hcfg["minRadius"] = _clamp(int(round(float(hcfg.get("minRadius", 32)))), 1, 2000)
    hcfg["maxRadius"] = _clamp(int(round(float(hcfg.get("maxRadius", 120)))), 2, 3000)
    if hcfg["maxRadius"] < hcfg["minRadius"]:
        hcfg["maxRadius"] = hcfg["minRadius"]
    hcfg["minDist"] = _clamp(int(round(float(hcfg.get("minDist", 120)))), 1, 4000)
    hcfg["min_center_dist"] = _clamp(int(round(float(hcfg.get("min_center_dist", 90)))), 1, 4000)
    hcfg["edge_score_threshold"] = round(
        _clamp(float(hcfg.get("edge_score_threshold", 0.18)), 0.0, 1.0),
        3,
    )
    hcfg["radius_consensus_tol"] = _clamp(int(round(float(hcfg.get("radius_consensus_tol", 6)))), 1, 120)
    hcfg["radius_final_tol"] = _clamp(int(round(float(hcfg.get("radius_final_tol", 4)))), 1, 120)
    hcfg["edge_ring_width"] = _clamp(int(round(float(hcfg.get("edge_ring_width", 3)))), 1, 30)
    hcfg["use_common_radius_refine"] = _to_bool(hcfg.get("use_common_radius_refine", True))
    hcfg["common_radius_deviation_tol"] = _clamp(
        int(round(float(hcfg.get("common_radius_deviation_tol", 2)))),
        0,
        50,
    )
    hcfg["radius_refine_band"] = _clamp(int(round(float(hcfg.get("radius_refine_band", 4)))), 1, 60)
    hcfg["center_refine_range"] = _clamp(int(round(float(hcfg.get("center_refine_range", 2)))), 0, 30)
    hcfg["radius_penalty"] = round(_clamp(float(hcfg.get("radius_penalty", 0.012)), 0.0, 0.2), 4)
    hcfg["center_penalty"] = round(_clamp(float(hcfg.get("center_penalty", 0.003)), 0.0, 0.2), 4)
    hcfg["common_radius_score_gap"] = round(
        _clamp(float(hcfg.get("common_radius_score_gap", 0.035)), 0.0, 0.3),
        4,
    )
    hcfg["force_common_radius"] = _to_bool(hcfg.get("force_common_radius", True))

    preview_cfg = cfg["preview"]
    preview_cfg["overview_max_width"] = _clamp(
        int(round(float(preview_cfg.get("overview_max_width", 1700)))),
        800,
        3200,
    )
    preview_cfg["overview_max_height"] = _clamp(
        int(round(float(preview_cfg.get("overview_max_height", 1000)))),
        600,
        2400,
    )
    preview_cfg["detail_max_width"] = _clamp(
        int(round(float(preview_cfg.get("detail_max_width", 520)))),
        220,
        1600,
    )
    preview_cfg["detail_max_height"] = _clamp(
        int(round(float(preview_cfg.get("detail_max_height", 360)))),
        180,
        1200,
    )
    preview_cfg["auto_update"] = _to_bool(preview_cfg.get("auto_update", True))
    preview_cfg["auto_update_delay_ms"] = _clamp(
        int(round(float(preview_cfg.get("auto_update_delay_ms", 350)))),
        120,
        2000,
    )

    return cfg


def _payload_from_config(cfg):
    return {
        "schema_version": PRESET_SCHEMA_VERSION,
        "config": normalize_config(cfg),
    }


def load_preset_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Preset JSON khong hop le.")

    raw_cfg = payload.get("config", payload)
    return normalize_config(raw_cfg)


def save_active_config(cfg):
    ensure_dir(PRESET_DIR)
    with open(ACTIVE_PRESET_PATH, "w", encoding="utf-8") as handle:
        json.dump(_payload_from_config(cfg), handle, ensure_ascii=False, indent=2)


def save_preset_file(path, cfg, sync_active=True):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_payload_from_config(cfg), handle, ensure_ascii=False, indent=2)
    if sync_active:
        save_active_config(cfg)


def load_active_config():
    if not os.path.isfile(ACTIVE_PRESET_PATH):
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        return load_preset_file(ACTIVE_PRESET_PATH)
    except Exception:
        return copy.deepcopy(DEFAULT_CONFIG)


CONFIG = normalize_config(load_active_config())


@functools.lru_cache(maxsize=1)
def load_backend_module():
    if not os.path.isfile(BACKEND_SCRIPT_PATH):
        raise FileNotFoundError("Khong tim thay backend Hough Circle: {}".format(BACKEND_SCRIPT_PATH))

    spec = importlib.util.spec_from_file_location("pipeline_hough_circle_backend", BACKEND_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Khong the nap backend tu file: {}".format(BACKEND_SCRIPT_PATH))

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

    new_size = (
        max(1, int(round(w * scale))),
        max(1, int(round(h * scale))),
    )
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def crop_work_roi(image, cfg=None):
    active_cfg = normalize_config(cfg if cfg is not None else CONFIG)
    crop_cfg = active_cfg["crop"]
    h, w = image.shape[:2]
    x = max(0, int(crop_cfg["x"]))
    y = max(0, int(crop_cfg["y"]))
    cw = int(crop_cfg["w"]) if int(crop_cfg["w"]) > 0 else (w - x)
    ch = int(crop_cfg["h"]) if int(crop_cfg["h"]) > 0 else (h - y)

    x2 = min(w, x + cw)
    y2 = min(h, y + ch)

    if x >= x2 or y >= y2:
        return image.copy(), (0, 0)

    return image[y:y2, x:x2].copy(), (x, y)


def preprocess_image(image, cfg=None, return_stages=False):
    active_cfg = normalize_config(cfg if cfg is not None else CONFIG)
    pre_cfg = active_cfg["preprocess"]

    gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    gray_hough = gray_raw.copy()

    if pre_cfg["enabled"]:
        if pre_cfg["clahe_use"]:
            tile = int(pre_cfg["clahe_tile_grid_size"])
            clahe = cv2.createCLAHE(
                clipLimit=float(pre_cfg["clahe_clip_limit"]),
                tileGridSize=(tile, tile),
            )
            gray_hough = clahe.apply(gray_hough)

        if pre_cfg["gaussian_use"]:
            kernel = int(pre_cfg["gaussian_kernel"])
            gray_hough = cv2.GaussianBlur(gray_hough, (kernel, kernel), 0)

    stages = {
        "gray_raw": gray_raw,
        "gray_hough": gray_hough,
    }
    if return_stages:
        return gray_hough, stages
    return gray_hough


def _apply_config_to_backend(module, cfg):
    module.CONFIG["input_path"] = cfg["input_path"]
    module.CONFIG["output_dir"] = cfg["output_dir"]
    module.CONFIG["expected_count"] = int(cfg["expected_count"])
    module.CONFIG["crop"] = copy.deepcopy(cfg["crop"])
    module.CONFIG["canny"] = copy.deepcopy(cfg["canny"])
    module.CONFIG["hough"] = copy.deepcopy(cfg["hough"])
    module.CONFIG["clahe"] = {
        "use": bool(cfg["preprocess"]["enabled"] and cfg["preprocess"]["clahe_use"]),
        "clipLimit": float(cfg["preprocess"]["clahe_clip_limit"]),
        "tileGridSize": (
            int(cfg["preprocess"]["clahe_tile_grid_size"]),
            int(cfg["preprocess"]["clahe_tile_grid_size"]),
        ),
    }


@contextlib.contextmanager
def backend_runtime(cfg=None, patch_preprocess=False):
    active_cfg = normalize_config(cfg if cfg is not None else CONFIG)
    module = load_backend_module()
    config_backup = copy.deepcopy(module.CONFIG)
    preprocess_backup = getattr(module, "preprocess_image", None)

    _apply_config_to_backend(module, active_cfg)
    if patch_preprocess and preprocess_backup is not None:
        module.preprocess_image = lambda image: preprocess_image(image, active_cfg)

    try:
        yield module
    finally:
        module.CONFIG.clear()
        module.CONFIG.update(config_backup)
        if preprocess_backup is not None:
            module.preprocess_image = preprocess_backup


def detect_stator_centers(roi_bgr, cfg=None):
    with backend_runtime(cfg, patch_preprocess=True) as module:
        return module.detect_stator_centers(roi_bgr)


def _collect_candidates(gray, edges, min_r, max_r, min_dist, p2_values, edge_threshold, use_edge_gate):
    with backend_runtime(CONFIG, patch_preprocess=False) as module:
        return module._collect_candidates(
            gray,
            edges,
            min_r,
            max_r,
            min_dist,
            p2_values,
            edge_threshold,
            use_edge_gate,
        )


def dedup_circles(candidates, min_dist, preferred_radius=None):
    module = load_backend_module()
    return module.dedup_circles(candidates, min_dist, preferred_radius=preferred_radius)


def draw_results(image, circles, offset_xy=(0, 0), common_radius=None):
    module = load_backend_module()
    return module.draw_results(image, circles, offset_xy, common_radius=common_radius)


def draw_candidate_overlay(image, circles, offset_xy=(0, 0)):
    module = load_backend_module()
    return module.draw_candidate_overlay(image, circles, offset_xy)


def draw_valid_overview(image, circles, offset_xy=(0, 0), common_radius=None):
    module = load_backend_module()
    return module.draw_valid_overview(image, circles, offset_xy, common_radius=common_radius)


def get_output_paths(output_dir):
    return {
        "input_overlay_path": os.path.join(output_dir, "00_input_with_crop.png"),
        "gray_raw_path": os.path.join(output_dir, "01_gray_raw.png"),
        "gray_hough_path": os.path.join(output_dir, "02_gray_used_for_hough.png"),
        "edges_path": os.path.join(output_dir, "03_canny_edges.png"),
        "candidate_path": os.path.join(output_dir, "04_raw_hough_candidates.png"),
        "result_path": os.path.join(output_dir, "05_detected_stator_centers_refined.png"),
        "overview_path": os.path.join(output_dir, "06_valid_stators_overview_refined.png"),
        "figure_path": os.path.join(output_dir, "07_hough_circle_report.png"),
        "preset_copy_path": os.path.join(output_dir, "08_hough_circle_active_preset.json"),
    }


def draw_input_crop_overlay(image, cfg):
    out = image.copy()
    crop_cfg = cfg["crop"]

    x = int(crop_cfg["x"])
    y = int(crop_cfg["y"])
    w = int(crop_cfg["w"])
    h = int(crop_cfg["h"])

    if w > 0 and h > 0:
        x2 = min(out.shape[1], x + w)
        y2 = min(out.shape[0], y + h)
        cv2.rectangle(out, (x, y), (x2, y2), (255, 180, 0), 2)
        cv2.putText(
            out,
            "ROI Hough",
            (x + 8, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 180, 0),
            2,
        )
    else:
        cv2.putText(
            out,
            "Dang dung toan anh cho Hough",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 180, 0),
            2,
        )

    return out


def _to_rgb_for_plot(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_gui_figure(outputs, cfg):
    preview_cfg = cfg["preview"]
    fig = Figure(figsize=(15.8, 9.1))
    grid = gridspec.GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[1.45, 1.0],
        wspace=0.08,
        hspace=0.16,
    )

    overview = resize_for_display(
        outputs["result"],
        max_width=int(preview_cfg["overview_max_width"]),
        max_height=int(preview_cfg["overview_max_height"]),
    )
    candidate = resize_for_display(
        outputs["raw_result"],
        max_width=int(preview_cfg["detail_max_width"] * 1.2),
        max_height=int(preview_cfg["detail_max_height"] * 1.2),
    )
    input_overlay = resize_for_display(
        outputs["input_overlay"],
        max_width=int(preview_cfg["detail_max_width"]),
        max_height=int(preview_cfg["detail_max_height"]),
    )
    gray_raw = resize_for_display(
        outputs["gray_raw"],
        max_width=int(preview_cfg["detail_max_width"]),
        max_height=int(preview_cfg["detail_max_height"]),
    )
    gray_hough = resize_for_display(
        outputs["gray"],
        max_width=int(preview_cfg["detail_max_width"]),
        max_height=int(preview_cfg["detail_max_height"]),
    )
    edges = resize_for_display(
        outputs["edges"],
        max_width=int(preview_cfg["detail_max_width"]),
        max_height=int(preview_cfg["detail_max_height"]),
    )

    ax_main = fig.add_subplot(grid[0, 0:3])
    ax_main.imshow(_to_rgb_for_plot(overview), cmap="gray" if len(overview.shape) == 2 else None)
    title = "Ket qua HoughCircle | {} stator".format(len(outputs["circles"]))
    if outputs["common_radius"] is not None:
        title += " | r_common={} px".format(int(round(outputs["common_radius"])))
    ax_main.set_title(title, fontsize=13, pad=8)
    ax_main.axis("off")

    ax_candidate = fig.add_subplot(grid[0, 3])
    ax_candidate.imshow(_to_rgb_for_plot(candidate), cmap="gray" if len(candidate.shape) == 2 else None)
    ax_candidate.set_title("Ung vien Hough truoc loc", fontsize=10, pad=6)
    ax_candidate.axis("off")

    lower_items = [
        ("Anh dau vao / ROI Hough", input_overlay),
        ("Gray goc", gray_raw),
        (
            "Gray dua vao Hough"
            if cfg["preprocess"]["enabled"]
            else "Gray dua thang vao Hough",
            gray_hough,
        ),
        ("Canny edges", edges),
    ]

    for col_idx, (title_text, image) in enumerate(lower_items):
        ax = fig.add_subplot(grid[1, col_idx])
        ax.imshow(_to_rgb_for_plot(image), cmap="gray" if len(image.shape) == 2 else None)
        ax.set_title(title_text, fontsize=10, pad=6)
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.03)
    return fig


def print_processing_summary(outputs):
    print("So ung vien truoc loc: {}".format(len(outputs["raw_candidates"])))
    print("So stator phat hien: {}".format(len(outputs["circles"])))
    print("Preprocess truoc Hough: {}".format("BAT" if outputs["config"]["preprocess"]["enabled"] else "TAT"))
    if outputs["common_radius"] is not None:
        print("Ban kinh dong thuan r_common: {} px".format(int(round(outputs["common_radius"]))))

    for idx, (cx, cy, radius, score) in enumerate(outputs["circles"], start=1):
        gx = int(cx + outputs["offset_xy"][0])
        gy = int(cy + outputs["offset_xy"][1])
        print(
            "ID{idx:02d}: center=({gx}, {gy}), radius={radius}, score={score:.3f}".format(
                idx=idx,
                gx=gx,
                gy=gy,
                radius=radius,
                score=score,
            )
        )

    if outputs["save_outputs"]:
        print("Da luu ket qua tai: {}".format(outputs["output_dir"]))
        print("Da cap nhat preset active cho 2. cat roi tai: {}".format(ACTIVE_PRESET_PATH))


def process_image(
    input_path,
    output_dir=None,
    save_outputs=True,
    status_callback=None,
    attach_figure=True,
    config_override=None,
):
    cfg = normalize_config(config_override if config_override is not None else CONFIG)

    def report_status(message):
        if status_callback is not None:
            status_callback(message)

    if not input_path:
        raise ValueError("Chua co duong dan anh dau vao.")
    if not os.path.isfile(input_path):
        raise FileNotFoundError("Khong tim thay anh dau vao: {}".format(input_path))

    output_dir = os.path.abspath(output_dir or cfg["output_dir"])
    ensure_dir(output_dir)
    output_paths = get_output_paths(output_dir)

    report_status("Dang doc anh dau vao...")
    image = read_image(input_path)
    if image is None:
        raise ValueError("Khong doc duoc anh: {}".format(input_path))

    report_status("Dang cat ROI lon cho Hough...")
    roi, offset_xy = crop_work_roi(image, cfg)

    report_status("Dang tien xu ly ROI truoc Hough...")
    _gray_hough, stages = preprocess_image(roi, cfg, return_stages=True)

    report_status("Dang chay Hough Circle...")
    gray, edges, raw_candidates, circles, common_radius = detect_stator_centers(roi, cfg)
    if not circles:
        raise RuntimeError("Khong tim thay stator nao bang Hough Circle.")

    report_status("Dang tao hinh preview...")
    input_overlay = draw_input_crop_overlay(image, cfg)
    raw_result = draw_candidate_overlay(image, raw_candidates, offset_xy)
    result = draw_results(image, circles, offset_xy, common_radius=common_radius)
    valid_overview = draw_valid_overview(image, circles, offset_xy, common_radius=common_radius)
    figure = build_gui_figure(
        {
            "result": result,
            "raw_result": raw_result,
            "input_overlay": input_overlay,
            "gray_raw": stages["gray_raw"],
            "gray": gray,
            "edges": edges,
            "circles": circles,
            "common_radius": common_radius,
        },
        cfg,
    ) if (save_outputs or attach_figure) else None

    outputs = {
        "input_path": input_path,
        "output_dir": output_dir,
        "save_outputs": bool(save_outputs),
        "config": cfg,
        "offset_xy": offset_xy,
        "image": image,
        "roi": roi,
        "input_overlay": input_overlay,
        "gray_raw": stages["gray_raw"],
        "gray": gray,
        "edges": edges,
        "raw_candidates": raw_candidates,
        "circles": circles,
        "common_radius": common_radius,
        "raw_result": raw_result,
        "result": result,
        "valid_overview": valid_overview,
        "figure": figure if attach_figure else None,
        "output_paths": output_paths,
    }

    if save_outputs:
        report_status("Dang luu output va preset active...")
        write_image(output_paths["input_overlay_path"], input_overlay)
        write_image(output_paths["gray_raw_path"], stages["gray_raw"])
        write_image(output_paths["gray_hough_path"], gray)
        write_image(output_paths["edges_path"], edges)
        write_image(output_paths["candidate_path"], raw_result)
        write_image(output_paths["result_path"], result)
        write_image(output_paths["overview_path"], valid_overview)
        if figure is None:
            figure = build_gui_figure(outputs, cfg)
            outputs["figure"] = figure if attach_figure else None
        figure.savefig(output_paths["figure_path"], dpi=300, bbox_inches="tight", pad_inches=0.05)
        save_active_config(cfg)
        save_preset_file(output_paths["preset_copy_path"], cfg, sync_active=False)

    print_processing_summary(outputs)
    return outputs


def save_current_preset_from_gui(path, cfg):
    save_preset_file(path, cfg, sync_active=True)


def run_gui():
    global CONFIG

    CONFIG = normalize_config(load_active_config())
    ensure_dir(PRESET_DIR)

    root = tk.Tk()
    root.title("Buoc 1 - HoughCircle")
    root.geometry("1600x980")
    root.minsize(1320, 840)

    left_shell = tk.Frame(root, width=450, bg="#f3f4f6")
    left_shell.pack(side="left", fill="y")
    left_shell.pack_propagate(False)

    right_panel = tk.Frame(root, padx=8, pady=8)
    right_panel.pack(side="right", fill="both", expand=True)

    canvas_holder = tk.Frame(right_panel, bg="white", bd=1, relief="solid")
    canvas_holder.pack(fill="both", expand=True)

    placeholder = tk.Label(
        canvas_holder,
        text=(
            "Preview HoughCircle se hien thi o day.\n"
            "Dieu chinh thong so tien xu ly va Hough roi bam Preview de xem nhanh."
        ),
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
    status_text = tk.StringVar(
        value="Chon anh dau vao, tune tien xu ly + Hough, sau do luu bo tham so de buoc 2 dung lai."
    )
    image_info_text = tk.StringVar(value="Kich thuoc anh: chua nap")
    output_info_text = tk.StringVar(value=CONFIG["output_dir"])

    vars_gui = {
        "auto_update": tk.BooleanVar(value=bool(CONFIG["preview"]["auto_update"])),
        "expected_count": tk.IntVar(value=int(CONFIG["expected_count"])),
        "crop_x": tk.IntVar(value=int(CONFIG["crop"]["x"])),
        "crop_y": tk.IntVar(value=int(CONFIG["crop"]["y"])),
        "crop_w": tk.IntVar(value=int(CONFIG["crop"]["w"])),
        "crop_h": tk.IntVar(value=int(CONFIG["crop"]["h"])),
        "pre_enabled": tk.BooleanVar(value=bool(CONFIG["preprocess"]["enabled"])),
        "clahe_use": tk.BooleanVar(value=bool(CONFIG["preprocess"]["clahe_use"])),
        "clahe_clip_limit": tk.DoubleVar(value=float(CONFIG["preprocess"]["clahe_clip_limit"])),
        "clahe_tile_grid_size": tk.IntVar(value=int(CONFIG["preprocess"]["clahe_tile_grid_size"])),
        "gaussian_use": tk.BooleanVar(value=bool(CONFIG["preprocess"]["gaussian_use"])),
        "gaussian_kernel": tk.IntVar(value=int(CONFIG["preprocess"]["gaussian_kernel"])),
        "canny_t1": tk.IntVar(value=int(CONFIG["canny"]["threshold1"])),
        "canny_t2": tk.IntVar(value=int(CONFIG["canny"]["threshold2"])),
        "dp": tk.DoubleVar(value=float(CONFIG["hough"]["dp"])),
        "param1": tk.IntVar(value=int(CONFIG["hough"]["param1"])),
        "param2": tk.IntVar(value=int(CONFIG["hough"]["param2"])),
        "minRadius": tk.IntVar(value=int(CONFIG["hough"]["minRadius"])),
        "maxRadius": tk.IntVar(value=int(CONFIG["hough"]["maxRadius"])),
        "minDist": tk.IntVar(value=int(CONFIG["hough"]["minDist"])),
        "min_center_dist": tk.IntVar(value=int(CONFIG["hough"]["min_center_dist"])),
        "edge_score_threshold": tk.DoubleVar(value=float(CONFIG["hough"]["edge_score_threshold"])),
        "radius_consensus_tol": tk.IntVar(value=int(CONFIG["hough"]["radius_consensus_tol"])),
        "radius_final_tol": tk.IntVar(value=int(CONFIG["hough"]["radius_final_tol"])),
        "edge_ring_width": tk.IntVar(value=int(CONFIG["hough"]["edge_ring_width"])),
        "use_common_radius_refine": tk.BooleanVar(value=bool(CONFIG["hough"]["use_common_radius_refine"])),
        "common_radius_deviation_tol": tk.IntVar(value=int(CONFIG["hough"]["common_radius_deviation_tol"])),
        "radius_refine_band": tk.IntVar(value=int(CONFIG["hough"]["radius_refine_band"])),
        "center_refine_range": tk.IntVar(value=int(CONFIG["hough"]["center_refine_range"])),
        "radius_penalty": tk.DoubleVar(value=float(CONFIG["hough"]["radius_penalty"])),
        "center_penalty": tk.DoubleVar(value=float(CONFIG["hough"]["center_penalty"])),
        "common_radius_score_gap": tk.DoubleVar(value=float(CONFIG["hough"]["common_radius_score_gap"])),
        "force_common_radius": tk.BooleanVar(value=bool(CONFIG["hough"]["force_common_radius"])),
    }

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
        preview_button.config(state=busy_state)
        run_button.config(state=busy_state)
        browse_button.config(state=busy_state)
        preset_save_button.config(state=busy_state)
        preset_load_button.config(state=busy_state)
        reset_button.config(state=busy_state)
        path_entry.config(state=busy_state)

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
            "preprocess": {
                "enabled": vars_gui["pre_enabled"].get(),
                "clahe_use": vars_gui["clahe_use"].get(),
                "clahe_clip_limit": vars_gui["clahe_clip_limit"].get(),
                "clahe_tile_grid_size": vars_gui["clahe_tile_grid_size"].get(),
                "gaussian_use": vars_gui["gaussian_use"].get(),
                "gaussian_kernel": vars_gui["gaussian_kernel"].get(),
            },
            "canny": {
                "threshold1": vars_gui["canny_t1"].get(),
                "threshold2": vars_gui["canny_t2"].get(),
            },
            "hough": {
                "dp": vars_gui["dp"].get(),
                "param1": vars_gui["param1"].get(),
                "param2": vars_gui["param2"].get(),
                "minRadius": vars_gui["minRadius"].get(),
                "maxRadius": vars_gui["maxRadius"].get(),
                "minDist": vars_gui["minDist"].get(),
                "min_center_dist": vars_gui["min_center_dist"].get(),
                "edge_score_threshold": vars_gui["edge_score_threshold"].get(),
                "radius_consensus_tol": vars_gui["radius_consensus_tol"].get(),
                "radius_final_tol": vars_gui["radius_final_tol"].get(),
                "edge_ring_width": vars_gui["edge_ring_width"].get(),
                "use_common_radius_refine": vars_gui["use_common_radius_refine"].get(),
                "common_radius_deviation_tol": vars_gui["common_radius_deviation_tol"].get(),
                "radius_refine_band": vars_gui["radius_refine_band"].get(),
                "center_refine_range": vars_gui["center_refine_range"].get(),
                "radius_penalty": vars_gui["radius_penalty"].get(),
                "center_penalty": vars_gui["center_penalty"].get(),
                "common_radius_score_gap": vars_gui["common_radius_score_gap"].get(),
                "force_common_radius": vars_gui["force_common_radius"].get(),
            },
            "preview": {
                "overview_max_width": state["last_config"]["preview"]["overview_max_width"],
                "overview_max_height": state["last_config"]["preview"]["overview_max_height"],
                "detail_max_width": state["last_config"]["preview"]["detail_max_width"],
                "detail_max_height": state["last_config"]["preview"]["detail_max_height"],
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
        vars_gui["expected_count"].set(int(cfg["expected_count"]))
        vars_gui["crop_x"].set(int(cfg["crop"]["x"]))
        vars_gui["crop_y"].set(int(cfg["crop"]["y"]))
        vars_gui["crop_w"].set(int(cfg["crop"]["w"]))
        vars_gui["crop_h"].set(int(cfg["crop"]["h"]))
        vars_gui["pre_enabled"].set(bool(cfg["preprocess"]["enabled"]))
        vars_gui["clahe_use"].set(bool(cfg["preprocess"]["clahe_use"]))
        vars_gui["clahe_clip_limit"].set(float(cfg["preprocess"]["clahe_clip_limit"]))
        vars_gui["clahe_tile_grid_size"].set(int(cfg["preprocess"]["clahe_tile_grid_size"]))
        vars_gui["gaussian_use"].set(bool(cfg["preprocess"]["gaussian_use"]))
        vars_gui["gaussian_kernel"].set(int(cfg["preprocess"]["gaussian_kernel"]))
        vars_gui["canny_t1"].set(int(cfg["canny"]["threshold1"]))
        vars_gui["canny_t2"].set(int(cfg["canny"]["threshold2"]))
        vars_gui["dp"].set(float(cfg["hough"]["dp"]))
        vars_gui["param1"].set(int(cfg["hough"]["param1"]))
        vars_gui["param2"].set(int(cfg["hough"]["param2"]))
        vars_gui["minRadius"].set(int(cfg["hough"]["minRadius"]))
        vars_gui["maxRadius"].set(int(cfg["hough"]["maxRadius"]))
        vars_gui["minDist"].set(int(cfg["hough"]["minDist"]))
        vars_gui["min_center_dist"].set(int(cfg["hough"]["min_center_dist"]))
        vars_gui["edge_score_threshold"].set(float(cfg["hough"]["edge_score_threshold"]))
        vars_gui["radius_consensus_tol"].set(int(cfg["hough"]["radius_consensus_tol"]))
        vars_gui["radius_final_tol"].set(int(cfg["hough"]["radius_final_tol"]))
        vars_gui["edge_ring_width"].set(int(cfg["hough"]["edge_ring_width"]))
        vars_gui["use_common_radius_refine"].set(bool(cfg["hough"]["use_common_radius_refine"]))
        vars_gui["common_radius_deviation_tol"].set(int(cfg["hough"]["common_radius_deviation_tol"]))
        vars_gui["radius_refine_band"].set(int(cfg["hough"]["radius_refine_band"]))
        vars_gui["center_refine_range"].set(int(cfg["hough"]["center_refine_range"]))
        vars_gui["radius_penalty"].set(float(cfg["hough"]["radius_penalty"]))
        vars_gui["center_penalty"].set(float(cfg["hough"]["center_penalty"]))
        vars_gui["common_radius_score_gap"].set(float(cfg["hough"]["common_radius_score_gap"]))
        vars_gui["force_common_radius"].set(bool(cfg["hough"]["force_common_radius"]))
        state["last_config"] = normalize_config(cfg)
        state["suspend_events"] = False
        update_image_metadata(cfg["input_path"])

    def build_status_summary(outputs, save_outputs):
        pre_text = "co preprocess" if outputs["config"]["preprocess"]["enabled"] else "khong preprocess"
        save_suffix = (
            " | da luu output + cap nhat preset active cho 2. cat roi"
            if save_outputs
            else " | preview only"
        )
        return "Xong. {} stator | {}".format(len(outputs["circles"]), pre_text) + save_suffix

    def render_outputs(outputs):
        clear_canvas()
        figure = build_gui_figure(outputs, outputs["config"])
        hide_placeholder()
        canvas = FigureCanvasTkAgg(figure, master=canvas_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        state["canvas"] = canvas

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

    def run_and_save():
        request_processing(save_outputs=True, reason="luu output + preset active", interactive=True)

    def save_preset():
        try:
            cfg = collect_config_from_gui()
            initial_dir = PRESET_DIR
            ensure_dir(initial_dir)
            initial_name = "hough_circle_setting.json"
            save_path = filedialog.asksaveasfilename(
                title="Luu preset HoughCircle",
                initialdir=initial_dir,
                initialfile=initial_name,
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            )
            if not save_path:
                return
            save_current_preset_from_gui(save_path, cfg)
            status_text.set(
                "Da luu preset HoughCircle. 2. cat roi co the dung ngay bo tham so active nay."
            )
        except Exception as exc:
            messagebox.showerror("Loi luu preset", str(exc))

    def load_preset():
        try:
            preset_path = filedialog.askopenfilename(
                title="Nap preset HoughCircle",
                initialdir=PRESET_DIR,
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            )
            if not preset_path:
                return
            cfg = load_preset_file(preset_path)
            load_vars_from_config(cfg)
            status_text.set("Da nap preset HoughCircle. Live preview se cap nhat.")
            schedule_auto_preview()
        except Exception as exc:
            messagebox.showerror("Loi nap preset", str(exc))

    def reset_defaults():
        load_vars_from_config(copy.deepcopy(DEFAULT_CONFIG))
        status_text.set("Da reset ve thong so mac dinh.")
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
        text="Buoc 1 - HoughCircle",
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
        text="Run + luu active cho ROI",
        command=run_and_save,
        bg="#1c9d63",
        fg="white",
        font=("Arial", 10, "bold"),
        height=2,
    )
    run_button.pack(fill="x", pady=(0, 6))

    preset_row = tk.Frame(action_frame, bg="#f3f4f6")
    preset_row.pack(fill="x", pady=(0, 6))
    preset_save_button = tk.Button(preset_row, text="Luu preset", command=save_preset)
    preset_save_button.pack(side="left", fill="x", expand=True, padx=(0, 3))
    preset_load_button = tk.Button(preset_row, text="Nap preset", command=load_preset)
    preset_load_button.pack(side="right", fill="x", expand=True, padx=(3, 0))

    reset_button = tk.Button(action_frame, text="Reset mac dinh", command=reset_defaults)
    reset_button.pack(fill="x")

    preprocess_frame = tk.LabelFrame(
        controls_frame,
        text="3. Tien xu ly truoc Hough",
        bg="#f3f4f6",
        padx=8,
        pady=8,
    )
    preprocess_frame.pack(fill="x", pady=(0, 10))
    add_checkbox(preprocess_frame, "pre_enabled", "Bat tien xu ly truoc Hough")
    add_checkbox(preprocess_frame, "clahe_use", "Dung CLAHE")
    add_slider(preprocess_frame, "clahe_clip_limit", "CLAHE clipLimit", 0.5, 8.0, 0.1)
    add_slider(preprocess_frame, "clahe_tile_grid_size", "CLAHE tileGridSize", 2, 32, 1)
    add_checkbox(preprocess_frame, "gaussian_use", "Dung Gaussian Blur")
    add_slider(preprocess_frame, "gaussian_kernel", "Gaussian kernel (odd)", 1, 31, 2)

    hough_frame = tk.LabelFrame(controls_frame, text="4. HoughCircle", bg="#f3f4f6", padx=8, pady=8)
    hough_frame.pack(fill="x", pady=(0, 10))
    add_slider(hough_frame, "expected_count", "So stator ky vong", 1, 24, 1)
    tk.Label(
        hough_frame,
        text="Crop lon cho Hough (px). Dat w/h = 0 de dung phan con lai cua anh.",
        justify="left",
        wraplength=380,
        bg="#f3f4f6",
        fg="#555555",
    ).pack(fill="x", pady=(4, 2))
    add_slider(hough_frame, "crop_x", "Crop x", 0, 5000, 1)
    add_slider(hough_frame, "crop_y", "Crop y", 0, 5000, 1)
    add_slider(hough_frame, "crop_w", "Crop w", 0, 5000, 1)
    add_slider(hough_frame, "crop_h", "Crop h", 0, 5000, 1)
    add_slider(hough_frame, "canny_t1", "Canny threshold1", 1, 255, 1)
    add_slider(hough_frame, "canny_t2", "Canny threshold2", 1, 255, 1)
    add_slider(hough_frame, "dp", "Hough dp", 0.8, 3.0, 0.05)
    add_slider(hough_frame, "param1", "Hough param1", 1, 255, 1)
    add_slider(hough_frame, "param2", "Hough param2", 1, 255, 1)
    add_slider(hough_frame, "minRadius", "minRadius", 1, 600, 1)
    add_slider(hough_frame, "maxRadius", "maxRadius", 1, 900, 1)
    add_slider(hough_frame, "minDist", "minDist", 1, 1200, 1)
    add_slider(hough_frame, "min_center_dist", "min_center_dist", 1, 1200, 1)
    add_slider(hough_frame, "edge_score_threshold", "edge_score_threshold", 0.0, 1.0, 0.01)
    add_slider(hough_frame, "radius_consensus_tol", "radius_consensus_tol", 1, 60, 1)
    add_slider(hough_frame, "radius_final_tol", "radius_final_tol", 1, 60, 1)
    add_slider(hough_frame, "edge_ring_width", "edge_ring_width", 1, 20, 1)
    add_checkbox(hough_frame, "use_common_radius_refine", "Bat refine theo r_common")
    add_slider(hough_frame, "common_radius_deviation_tol", "common_radius_deviation_tol", 0, 20, 1)
    add_slider(hough_frame, "radius_refine_band", "radius_refine_band", 1, 20, 1)
    add_slider(hough_frame, "center_refine_range", "center_refine_range", 0, 12, 1)
    add_slider(hough_frame, "radius_penalty", "radius_penalty", 0.0, 0.1, 0.001)
    add_slider(hough_frame, "center_penalty", "center_penalty", 0.0, 0.1, 0.001)
    add_slider(hough_frame, "common_radius_score_gap", "common_radius_score_gap", 0.0, 0.2, 0.001)
    add_checkbox(hough_frame, "force_common_radius", "Ve vong tron theo r_common")

    output_frame = tk.LabelFrame(controls_frame, text="5. Thu muc ket qua", bg="#f3f4f6", padx=8, pady=8)
    output_frame.pack(fill="x", pady=(0, 10))
    tk.Label(
        output_frame,
        textvariable=output_info_text,
        justify="left",
        wraplength=380,
        bg="#f3f4f6",
        fg="#444444",
    ).pack(fill="x")

    note_frame = tk.LabelFrame(controls_frame, text="6. Ghi chu", bg="#f3f4f6", padx=8, pady=8)
    note_frame.pack(fill="x", pady=(0, 10))
    tk.Label(
        note_frame,
        text=(
            "- Preview live khong ghi file ra dia.\n"
            "- Nut 'Run + luu active cho ROI' se luu output va cap nhat preset active.\n"
            "- Nut 'Luu preset' cung cap nhat preset active de 2. cat roi dung lai.\n"
            "- Giao dien nay chi dung de setup bo thong so cho phat hien tam Hough.\n"
            "- 2. cat roi giu nguyen workflow, chi lay lai bo thong so da luu."
        ),
        justify="left",
        wraplength=380,
        bg="#f3f4f6",
        fg="#444444",
    ).pack(fill="x")

    tk.Label(
        controls_frame,
        textvariable=status_text,
        justify="left",
        wraplength=400,
        anchor="w",
        bg="#f3f4f6",
        fg="#222222",
    ).pack(fill="x", pady=(4, 0))

    for key in [
        "auto_update",
        "expected_count",
        "crop_x",
        "crop_y",
        "crop_w",
        "crop_h",
        "pre_enabled",
        "clahe_use",
        "clahe_clip_limit",
        "clahe_tile_grid_size",
        "gaussian_use",
        "gaussian_kernel",
        "canny_t1",
        "canny_t2",
        "dp",
        "param1",
        "param2",
        "minRadius",
        "maxRadius",
        "minDist",
        "min_center_dist",
        "edge_score_threshold",
        "radius_consensus_tol",
        "radius_final_tol",
        "edge_ring_width",
        "use_common_radius_refine",
        "common_radius_deviation_tol",
        "radius_refine_band",
        "center_refine_range",
        "radius_penalty",
        "center_penalty",
        "common_radius_score_gap",
        "force_common_radius",
    ]:
        vars_gui[key].trace_add("write", on_control_change)

    update_image_metadata(selected_path_var.get().strip())
    poll_worker_events()

    if selected_path_var.get().strip() and os.path.isfile(selected_path_var.get().strip()):
        root.after(250, lambda: request_processing(save_outputs=False, reason="preview ban dau", interactive=False))

    root.mainloop()


def main():
    run_gui()


if __name__ == "__main__":
    main()
