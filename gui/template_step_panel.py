"""GUI panel for step 5: template creation."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.config import (
    DEFAULT_RADIAL_PARAMS,
    DEFAULT_ROI_PARAMS,
    DEFAULT_TAB_EDGE_PARAMS,
    INPUT_DIR,
    RADIAL_PRESET_PATH,
    ROI_PRESET_PATH,
    TAB_EDGE_PRESET_PATH,
    TEMPLATE_DATA_PATH,
    TEMPLATE_ROI_PATH,
)
from src.io_utils import read_image, write_image
from src.preset_store import load_preset
from src.roi_extractor import build_roi_item_from_image
from src.template_builder import save_template_data
from src.pipeline_runner import run_step_template

from .common_widgets import StepPanelBase


class TemplateStepPanel(StepPanelBase):
    """Panel for creating template_data.json from a detected stator ID or an external ROI."""

    def __init__(self, master, app):
        super().__init__(master, app, [], {})
        self.roi_path_var = tk.StringVar()
        self.roi_id_var = tk.StringVar()
        ttk.Label(self.toolbar, text="Stator ID").pack(side="left")
        self.roi_combo = ttk.Combobox(self.toolbar, textvariable=self.roi_id_var, state="readonly", width=8)
        self.roi_combo.pack(side="left", padx=(2, 8))
        ttk.Label(self.toolbar, text="hoac file").pack(side="left")
        ttk.Entry(self.toolbar, textvariable=self.roi_path_var, width=28).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(self.toolbar, text="Chon ROI", command=self.choose_roi).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Run", command=self.run_step).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save Template", command=self.save_template).pack(side="left", padx=3)
        self.latest_result = None
        self.latest_roi_item = None
        self.refresh_ids()

    def refresh_ids(self):
        """Populate the stator ID combobox from the ROI step result."""
        roi_result = self.app.shared.get("roi_result")
        ids = [str(item["id"]) for item in roi_result["data"]["rois"]] if roi_result and roi_result["success"] else []
        self.roi_combo["values"] = ids
        if ids and not self.roi_id_var.get():
            self.roi_id_var.set(ids[0])

    def choose_roi(self):
        path = filedialog.askopenfilename(title="Chon ROI mau 0 do", initialdir=str(INPUT_DIR))
        if path:
            self.roi_path_var.set(path)

    def _roi_item_by_id(self, roi_id):
        """Return the best available ROI item for one detected stator ID."""
        effective_items = self.app.shared.get("roi_effective_items", {})
        if roi_id in effective_items:
            return effective_items[roi_id]
        roi_result = self.app.shared.get("roi_result")
        if roi_result and roi_result["success"]:
            for item in roi_result["data"]["rois"]:
                if item["id"] == roi_id:
                    return item
        return None

    def _resolve_roi_item(self):
        """Pick the ROI item: external file first, then selected ID, then first ROI."""
        roi_path = self.roi_path_var.get().strip()
        if roi_path:
            image = read_image(roi_path)
            if image is None:
                raise ValueError("Khong doc duoc ROI mau.")
            roi_params = load_preset(ROI_PRESET_PATH, DEFAULT_ROI_PARAMS)
            return build_roi_item_from_image(image, roi_params, roi_id=1)
        roi_id_text = self.roi_id_var.get().strip()
        if roi_id_text:
            item = self._roi_item_by_id(int(roi_id_text))
            if item is not None:
                return item
        effective_items = self.app.shared.get("roi_effective_items", {})
        if effective_items:
            return effective_items[sorted(effective_items.keys())[0]]
        roi_result = self.app.shared.get("roi_result")
        if roi_result and roi_result["success"] and roi_result["data"]["rois"]:
            return roi_result["data"]["rois"][0]
        raise ValueError("Hay chon stator ID (chay buoc 2 truoc) hoac chon file ROI mau.")

    @staticmethod
    def _full_center(roi_item):
        center_in_roi = roi_item.get("center_in_roi", (0.0, 0.0))
        return [
            float(roi_item.get("center_x", center_in_roi[0])),
            float(roi_item.get("center_y", center_in_roi[1])),
        ]

    def run_step(self):
        self.refresh_ids()
        try:
            roi_item = self._resolve_roi_item()
        except Exception as exc:
            messagebox.showwarning("Template", str(exc))
            return
        tab_params = load_preset(TAB_EDGE_PRESET_PATH, DEFAULT_TAB_EDGE_PARAMS)
        radial_params = load_preset(RADIAL_PRESET_PATH, DEFAULT_RADIAL_PARAMS)
        result = run_step_template(roi_item, tab_params, radial_params)
        roi_logs = roi_item.get("logs") or []
        if roi_logs:
            result["logs"] = list(roi_logs) + list(result.get("logs", []))
        self.latest_result = result
        self.latest_roi_item = roi_item
        if result["success"]:
            # Gan kem metadata stator that (ID, tam toan anh, ban kinh) de buoc Match hien thi.
            result["data"]["source_id"] = int(roi_item.get("id", 1))
            result["data"]["center_full"] = self._full_center(roi_item)
            result["data"]["radius_full"] = float(roi_item.get("radius_full", roi_item.get("radius", 0.0)))
            self.app.shared["template_data"] = result["data"]
            self.app.shared["template_roi_image"] = result["images"].get("template_roi")
        self.set_result(result)

    def save_template(self):
        if not self.latest_result or not self.latest_result["success"]:
            messagebox.showwarning("Template", "Hay run template truoc.")
            return
        template_roi = self.latest_result["images"].get("template_roi")
        if template_roi is not None:
            write_image(TEMPLATE_ROI_PATH, template_roi)
            self.latest_result["data"]["template_roi_path"] = str(TEMPLATE_ROI_PATH)
        save_template_data(TEMPLATE_DATA_PATH, self.latest_result["data"])
        messagebox.showinfo("Template", "Da luu template_data.json va template_roi.png")
