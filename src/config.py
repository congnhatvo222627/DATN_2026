"""Project configuration and default parameters."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
TEMPLATE_DIR = DATA_DIR / "template"
ROI_DIR = DATA_DIR / "roi"
OUTPUT_DIR = DATA_DIR / "output"
PRESET_DIR = BASE_DIR / "presets"
LOGO_PATH = ASSETS_DIR / "logo_bkhn.png"

APP_TITLE = "He thong nhan dien tam va goc xoay stator"

HOUGH_PRESET_PATH = PRESET_DIR / "hough_preset.json"
ROI_PRESET_PATH = PRESET_DIR / "roi_preset.json"
TAB_EDGE_PRESET_PATH = PRESET_DIR / "tab_edge_preset.json"
RADIAL_PRESET_PATH = PRESET_DIR / "radial_preset.json"
RADIAL_SIGNATURE_PRESET_PATH = PRESET_DIR / "radial_signature_preset.json"
CALIBRATION_PRESET_PATH = PRESET_DIR / "calibration_preset.json"
TEMPLATE_DATA_PATH = TEMPLATE_DIR / "template_data.json"
TEMPLATE_ROI_PATH = TEMPLATE_DIR / "template_roi.png"

DEFAULT_HOUGH_PARAMS = {
    "expected_count": 12,
    "fast_mode": {
        "enabled": False,
        "max_processing_dim": 1400,
    },
    "preprocess": {
        "enabled": True,
        "use_clahe": True,
        "clahe_clip_limit": 2.5,
        "clahe_tile_grid_size": 8,
        "use_gaussian": True,
        "gaussian_kernel": 5,
    },
    "hough": {
        "dp": 1.2,
        "param1": 110,
        "param2": 38,
        "minDist": 120,
        "minRadius": 32,
        "maxRadius": 140,
        "min_center_dist": 90,
    },
    "filter": {
        "use_radius_consensus": True,
        "radius_consensus_tol": 6,
        "force_common_radius": True,
    },
}

DEFAULT_ROI_PARAMS = {
    "half_size_scale": 1.30,
    "output_size": 0,
    "refine": {
        "enabled": True,
        "preprocess": {
            "use_clahe": True,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": 8,
            "use_gaussian": True,
            "gaussian_kernel": 5,
        },
        "canny": {
            "threshold1": 70,
            "threshold2": 170,
        },
        "mask": {
            "enabled": True,
            "inner_radius_scale": 0.82,
            "outer_radius_scale": 1.20,
        },
        "hough": {
            "dp": 1.2,
            "param1": 110,
            "param2": 30,
            "minDist": 20,
            "min_radius_scale": 0.78,
            "max_radius_scale": 1.18,
            "max_center_shift_scale": 0.45,
        },
        "least_squares": {
            "enabled": False,
            "band_width_px": 3.0,
            "min_points": 24,
            "max_center_shift_scale": 0.12,
            "max_radius_delta_scale": 0.12,
            "score_tolerance": 0.02,
        },
        "score": {
            "ring_width": 3,
            "center_penalty_weight": 0.10,
            "radius_penalty_weight": 0.20,
        },
    },
}

DEFAULT_TAB_EDGE_PARAMS = {
    "preprocess": {
        "use_clahe": False,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": 8,
        "gaussian_kernel": 5,
        "gaussian_sigma": 1.0,
    },
    "threshold": {
        "use_otsu": True,
        "manual_value": 0,
        "invert": True,
    },
    "radius_filter": {
        "enabled": True,
        "r_min_factor": 1.0,
        "r_max_factor": 1.3,
        "inner_margin_px": 0.0,
        "outer_margin_px": 0.0,
    },
    "component_filter": {
        "min_area": 1500,
        "max_area": 0,
        "angle_bin_deg": 5.0,
        "max_angle_span_deg": 0.0,
    },
    "canny": {
        "threshold1": 70,
        "threshold2": 170,
        "aperture_size": 3,
        "l2_gradient": False,
    },
}

DEFAULT_RADIAL_PARAMS = {
    "source_mode": "tab_edges_clean",
    "use_radius_band": True,
    "inner_radius_scale": 1.0,
    "outer_radius_scale": 1.3,
    "use_source_dilate": False,
    "source_dilate_kernel": 3,
    "source_dilate_iter": 1,
    "num_angles": 360,
    "ray_step_px": 1.0,
    "ray_thickness": 2,
    "min_valid_radius_scale": 1.0,
    "floor_to_radius": True,
    "reject_outliers": False,
    "outlier_window": 11,
    "outlier_max_delta": 16.0,
    "interpolate_missing": True,
    "max_gap_to_interpolate": 18,
    "smooth_signature": True,
    "smooth_window": 7,
    "scale_normalize": True,
    "invert_angle": False,
}
