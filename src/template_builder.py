"""Step 5: template generation."""

import json
from pathlib import Path

from .tab_edge_filter import filter_tab_edges
from .radial_signature import build_radial_signature, draw_radial_rays, plot_signature_image


def _pick_edge_image(images, source_mode):
    """Pick the preferred debug edge image without ndarray truthiness issues."""
    for key in (source_mode, "closed_edges", "tab_edges_clean"):
        if key in images and images[key] is not None:
            return images[key]
    return None


def build_template_from_roi(roi, center, radius, tab_edge_params, radial_params):
    """Build template data from one ROI."""
    tab_result = filter_tab_edges(roi, center, radius, tab_edge_params)
    if not tab_result["success"]:
        return {"success": False, "data": {}, "images": tab_result.get("images", {}), "logs": tab_result.get("logs", [])}
    source_mode = str(radial_params.get("source_mode", "closed_edges"))
    source_image = _pick_edge_image(tab_result["images"], source_mode)
    radial_result = build_radial_signature(source_image, center, radial_params, radius=radius)
    if not radial_result["success"]:
        return {
            "success": False,
            "data": {},
            "images": {
                **tab_result["images"],
                "radial_source": radial_result["data"]["radial_source"],
                "radius_band": radial_result["data"]["radius_band"],
                "radial_rays": draw_radial_rays(
                    roi,
                    center,
                    radial_result["data"]["signature_raw"],
                    radius=radius,
                    measured_mask=radial_result["data"]["measured_mask"],
                ),
                "signature_plot": plot_signature_image(radial_result["data"]["signature_raw"], radial_result["data"]["signature_norm"]),
            },
            "logs": tab_result["logs"] + radial_result["logs"],
        }
    template_data = {
        "signature_raw": radial_result["data"]["signature_raw"].tolist(),
        "signature_norm": radial_result["data"]["signature_norm"].tolist(),
        "center": [float(center[0]), float(center[1])],
        "radius": float(radius),
        "tab_edge_params": tab_edge_params,
        "radial_params": radial_params,
        "note": "0 degree template",
    }
    images = {
        "radial_rays": draw_radial_rays(
            roi,
            center,
            radial_result["data"]["signature_raw"],
            radius=radius,
            measured_mask=radial_result["data"]["measured_mask"],
        ),
        "radial_source": radial_result["data"]["radial_source"],
        "radius_band": radial_result["data"]["radius_band"],
        **tab_result["images"],
        "signature_plot": plot_signature_image(radial_result["data"]["signature_raw"], radial_result["data"]["signature_norm"]),
        "template_roi": roi.copy(),
    }
    return {"success": True, "data": template_data, "images": images, "logs": tab_result["logs"] + radial_result["logs"]}


def save_template_data(path, template_data):
    """Save template JSON."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(template_data, handle, ensure_ascii=False, indent=2)
    return path_obj


def load_template_data(path):
    """Load template JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_template_step(roi, center, radius, tab_edge_params, radial_params):
    """Run template creation step."""
    return build_template_from_roi(roi, center, radius, tab_edge_params, radial_params)
