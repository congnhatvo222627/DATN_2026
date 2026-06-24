"""Standalone lightweight GUI to tune ONLY the HoughCircle step (buoc 1).

Cach chay:
    python scripts/gui_hough_tuner.py

Nguyen tac:
- KHONG viet lai thuat toan Hough trong file nay.
- Chi import va goi `run_hough_step(image, params)` tu `src/hough_detector.py`.
- KHONG goi main_gui.py, KHONG chay full pipeline.
- KHONG xu ly ROI / tab edge / radial signature o day.
- GUI nhe, dung Tkinter + Pillow (khong dung matplotlib de hien thi anh).
"""

import copy
import os
import queue
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk


# --- Cho phep import package `src` khi chay truc tiep `python scripts/...` ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DEFAULT_HOUGH_PARAMS,
    HOUGH_PRESET_PATH,
    INPUT_DIR,
    OUTPUT_DIR,
)
from src.hough_detector import run_hough_step  # noqa: E402
from src.io_utils import (  # noqa: E402
    ensure_dir,
    ensure_project_dirs,
    get_first_image,
    read_image,
    write_image,
)
from src.preset_store import load_preset, save_preset  # noqa: E402
from src.visualization import cv_bgr_to_rgb, resize_for_display  # noqa: E402


# Thu muc luu anh debug khi bam Run thu cong.
TUNER_OUTPUT_DIR = OUTPUT_DIR / "hough_tuner"

# Khoang debounce cho Auto Update (ms), nam trong 400-700 ms theo yeu cau.
DEBOUNCE_MS = 500

# Chu ky poll ket qua tu thread nen (ms).
POLL_MS = 60

# Cac o tham so co the chinh, gom theo nhom.
# O so co them "min"/"max" de dung lam khoang cho thanh truot (slider).
FIELD_SPECS = [
    {"path": "expected_count", "label": "expected_count", "type": "int", "group": "Hough", "min": 1, "max": 30},
    {"path": "preprocess.use_clahe", "label": "use_clahe", "type": "bool", "group": "Preprocess"},
    {"path": "preprocess.clahe_clip_limit", "label": "clahe_clip_limit", "type": "float", "group": "Preprocess", "min": 0.1, "max": 10.0},
    {"path": "preprocess.clahe_tile_grid_size", "label": "clahe_tile_grid_size", "type": "int", "group": "Preprocess", "min": 1, "max": 32},
    {"path": "preprocess.use_gaussian", "label": "use_gaussian", "type": "bool", "group": "Preprocess"},
    {"path": "preprocess.gaussian_kernel", "label": "gaussian_kernel", "type": "int", "group": "Preprocess", "min": 1, "max": 31},
    {"path": "hough.dp", "label": "hough dp", "type": "float", "group": "Hough", "min": 1.0, "max": 3.0},
    {"path": "hough.param1", "label": "hough param1", "type": "float", "group": "Hough", "min": 1, "max": 300},
    {"path": "hough.param2", "label": "hough param2", "type": "float", "group": "Hough", "min": 1, "max": 200},
    {"path": "hough.minDist", "label": "hough minDist", "type": "float", "group": "Hough", "min": 10, "max": 500},
    {"path": "hough.minRadius", "label": "hough minRadius", "type": "int", "group": "Hough", "min": 1, "max": 300},
    {"path": "hough.maxRadius", "label": "hough maxRadius", "type": "int", "group": "Hough", "min": 2, "max": 500},
    {"path": "hough.min_center_dist", "label": "min_center_dist", "type": "float", "group": "Hough", "min": 1, "max": 500},
    {"path": "filter.radius_consensus_tol", "label": "radius_consensus_tol", "type": "int", "group": "Filter", "min": 1, "max": 50},
    {"path": "filter.force_common_radius", "label": "force_common_radius", "type": "bool", "group": "Filter"},
]

# Ten combobox -> ten key trong result["images"].
DISPLAY_TO_IMAGE_KEY = {
    "Original": "original",
    "Preprocessed": "preprocessed",
    "Hough All": "hough_all",
    "Hough Filtered": "hough_filtered",
}
DISPLAY_ORDER = ["Original", "Preprocessed", "Hough All", "Hough Filtered"]


def get_nested(data, path, default=None):
    """Doc gia tri theo duong dan kieu 'a.b.c' tu dict long nhau."""
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_nested(data, path, value):
    """Ghi gia tri vao dict long nhau theo duong dan 'a.b.c'."""
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


class HoughTunerApp:
    """GUI nhe chi de tinh chinh buoc HoughCircle."""

    def __init__(self, root):
        self.root = root
        self.root.title("Hough Tuner - chi tinh chinh buoc 1 (HoughCircle)")
        self.root.geometry("1180x780")
        self.root.minsize(980, 620)

        # Trang thai
        self.source_image = None          # anh goc BGR dung de XU LY (khong resize)
        self.display_images = {}          # ten hien thi -> anh OpenCV (BGR/gray)
        self.last_result = None
        self._photo = None                # giu tham chieu PhotoImage tranh bi GC
        self._debounce_id = None          # id cua after() de debounce auto update
        self.field_vars = {}              # path -> (tk.Variable, type)
        self.field_scales = {}            # path -> ttk.Scale (chi cho o so)
        self._sync_guard = False          # tranh vong lap dong bo slider <-> entry
        self._busy = False                # dang co worker chay Hough
        self._pending = None              # request bi gop lai khi worker dang chay
        self._event_queue = queue.Queue()  # ket qua tu thread nen -> main thread
        self._poll_id = None              # id cua after() dang poll queue

        self.params = load_preset(HOUGH_PRESET_PATH, DEFAULT_HOUGH_PARAMS)

        self._build_ui()
        self._set_params(self.params)
        self._try_load_default_image()

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self):
        # Toolbar tren cung: chon anh + mo thu muc output.
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill="x", padx=8, pady=6)
        ttk.Button(toolbar, text="Chon anh khay", command=self.choose_image).pack(side="left")
        self.image_path_var = tk.StringVar(value="")
        ttk.Entry(toolbar, textvariable=self.image_path_var, state="readonly").pack(
            side="left", padx=6, fill="x", expand=True
        )
        ttk.Button(toolbar, text="Open Output Folder", command=self.open_output_folder).pack(side="left", padx=3)

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(body)
        body.add(left, weight=2)
        self._build_controls(left)

        right = ttk.Frame(body)
        body.add(right, weight=3)
        self._build_display(right)

    def _build_controls(self, parent):
        # Hang nut dieu khien.
        self.auto_update_var = tk.BooleanVar(value=False)
        ctrl_top = ttk.Frame(parent)
        ctrl_top.pack(fill="x", pady=(0, 4))
        ttk.Checkbutton(ctrl_top, text="Auto Update", variable=self.auto_update_var).pack(side="left")
        ttk.Button(ctrl_top, text="Run Hough", command=self.run_hough_manual).pack(side="left", padx=3)

        ctrl_bot = ttk.Frame(parent)
        ctrl_bot.pack(fill="x", pady=(0, 6))
        ttk.Button(ctrl_bot, text="Save Preset", command=self.save_preset_file).pack(side="left", padx=2)
        ttk.Button(ctrl_bot, text="Load Preset", command=self.load_preset_file).pack(side="left", padx=2)
        ttk.Button(ctrl_bot, text="Reset Default", command=self.reset_params).pack(side="left", padx=2)

        # Fast preview: thu nho anh khi xu ly cho muot (giong fast mode legacy).
        # Tham so do dai duoc scale theo ti le nen gia tri hien thi/preset van la pixel anh goc.
        fast_frame = ttk.LabelFrame(parent, text="Fast preview (chong lag voi anh lon)")
        fast_frame.pack(fill="x", padx=2, pady=(0, 6))
        self.fast_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            fast_frame,
            text="Bat fast preview (thu nho anh de xu ly nhanh)",
            variable=self.fast_mode_var,
        ).pack(anchor="w", padx=6, pady=(2, 0))
        dim_row = ttk.Frame(fast_frame)
        dim_row.pack(fill="x", padx=6, pady=2)
        ttk.Label(dim_row, text="Max canh xu ly (px)", width=20).pack(side="left")
        self.max_proc_dim_var = tk.StringVar(value="1400")
        ttk.Entry(dim_row, textvariable=self.max_proc_dim_var, width=8).pack(side="right")
        self.fast_mode_var.trace_add("write", lambda *_a: self._on_param_changed())
        self.max_proc_dim_var.trace_add("write", lambda *_a: self._on_param_changed())

        # Checkbox bat/tat tien xu ly truoc Hough.
        self.preprocess_enabled_var = tk.BooleanVar(value=bool(get_nested(self.params, "preprocess.enabled", True)))
        pre_row = ttk.Frame(parent)
        pre_row.pack(fill="x", pady=(0, 4))
        ttk.Checkbutton(
            pre_row,
            text="Bat tien xu ly truoc Hough (tat = dung anh xam goc)",
            variable=self.preprocess_enabled_var,
            command=self._on_preprocess_toggle,
        ).pack(side="left")
        self.preprocess_enabled_var.trace_add("write", lambda *_a: self._on_param_changed())

        # Cac nhom tham so.
        groups = {}
        for spec in FIELD_SPECS:
            group_name = spec.get("group", "General")
            if group_name not in groups:
                frame = ttk.LabelFrame(parent, text=group_name)
                frame.pack(fill="x", padx=2, pady=4)
                groups[group_name] = frame
            self._build_field_row(groups[group_name], spec)

    def _build_field_row(self, container, spec):
        row = ttk.Frame(container)
        row.pack(fill="x", padx=6, pady=2)
        path = spec["path"]
        field_type = spec.get("type", "str")
        ttk.Label(row, text=spec["label"], width=20).pack(side="left")

        if field_type == "bool":
            var = tk.BooleanVar(value=False)
            self.field_vars[path] = (var, field_type)
            ttk.Checkbutton(row, variable=var).pack(side="right")
            var.trace_add("write", lambda *_a, p=path: self._on_field_var_changed(p))
            return

        # O so: thanh truot de KEO + o nhap so de GO nhanh, dong bo 2 chieu.
        var = tk.StringVar(value="")
        self.field_vars[path] = (var, field_type)
        ttk.Entry(row, textvariable=var, width=7).pack(side="right")
        scale = ttk.Scale(
            row,
            orient="horizontal",
            from_=float(spec.get("min", 0)),
            to=float(spec.get("max", 100)),
            command=lambda raw, p=path: self._on_scale(p, raw),
        )
        scale.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.field_scales[path] = scale
        var.trace_add("write", lambda *_a, p=path: self._on_field_var_changed(p))

    def _build_display(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=(0, 6))
        ttk.Label(top, text="Anh debug").pack(side="left")
        self.image_selector_var = tk.StringVar(value="Original")
        selector = ttk.Combobox(
            top,
            textvariable=self.image_selector_var,
            values=DISPLAY_ORDER,
            state="readonly",
            width=18,
        )
        selector.pack(side="left", padx=6)
        selector.bind("<<ComboboxSelected>>", lambda _e: self._show_selected_image())

        self.detected_var = tk.StringVar(value="Detected: - / -")
        self.detected_label = tk.Label(
            top, textvariable=self.detected_var, font=("Segoe UI", 16, "bold"), fg="#555555"
        )
        self.detected_label.pack(side="right")

        self.image_label = ttk.Label(parent, anchor="center", text="Chua co anh hien thi")
        self.image_label.pack(fill="both", expand=True)

        ttk.Label(parent, text="Log").pack(anchor="w", pady=(6, 2))
        log_frame = ttk.Frame(parent)
        log_frame.pack(fill="x")
        self.log_text = tk.Text(log_frame, height=12, wrap="word")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set, state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    # ----------------------------------------------------------- Tham so ---
    def _collect_params(self):
        """Gom toan bo tham so tren GUI thanh dict long nhau cho run_hough_step."""
        result = {}
        set_nested(result, "preprocess.enabled", bool(self.preprocess_enabled_var.get()))
        for path, (var, field_type) in self.field_vars.items():
            value = var.get()
            if field_type == "int":
                value = int(round(float(str(value).strip() or 0)))
            elif field_type == "float":
                value = float(str(value).strip() or 0)
            elif field_type == "bool":
                value = bool(value)
            set_nested(result, path, value)
        # Giu nguyen cac key con lai (vd filter.use_radius_consensus) tu preset.
        set_nested(result, "filter.use_radius_consensus",
                   bool(get_nested(self.params, "filter.use_radius_consensus", True)))
        return result

    def _set_params(self, params):
        """Cap nhat cac o tren GUI tu dict tham so."""
        self.params = params
        self.preprocess_enabled_var.set(bool(get_nested(params, "preprocess.enabled", True)))
        for spec in FIELD_SPECS:
            path = spec["path"]
            var, field_type = self.field_vars[path]
            value = get_nested(params, path, spec.get("default"))
            if field_type == "bool":
                var.set(bool(value))
            else:
                var.set(str(value))

    # ----------------------------------------------------- Su kien tham so ---
    def _on_preprocess_toggle(self):
        if self.source_image is not None:
            self._append_log(["Tien xu ly truoc Hough: {}".format("BAT" if self.preprocess_enabled_var.get() else "TAT")])

    @staticmethod
    def _format_value(field_type, value):
        """Dinh dang gia tri tu thanh truot thanh chuoi gon cho o nhap."""
        if field_type == "int":
            return str(int(round(float(value))))
        text = "{:.2f}".format(float(value))
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text

    def _on_scale(self, path, raw_value):
        """Keo thanh truot -> cap nhat o nhap so (co debounce qua trace)."""
        if self._sync_guard:
            return
        var, field_type = self.field_vars[path]
        text = self._format_value(field_type, raw_value)
        if var.get() == text:
            return
        # Guard de var.set khong day nguoc lai thanh truot khi dang keo.
        self._sync_guard = True
        try:
            var.set(text)
        finally:
            self._sync_guard = False

    def _on_field_var_changed(self, path):
        """O so/checkbox doi -> dong bo thanh truot + debounce auto update."""
        if not self._sync_guard:
            scale = self.field_scales.get(path)
            if scale is not None:
                try:
                    value = float(self.field_vars[path][0].get())
                except (ValueError, TypeError):
                    value = None
                if value is not None:
                    self._sync_guard = True
                    try:
                        scale.set(value)
                    finally:
                        self._sync_guard = False
        self._on_param_changed()

    def _on_param_changed(self):
        """Goi khi mot o tham so thay doi -> debounce neu Auto Update bat."""
        if not self.auto_update_var.get():
            return
        if self._debounce_id is not None:
            self.root.after_cancel(self._debounce_id)
        self._debounce_id = self.root.after(DEBOUNCE_MS, self._auto_run)

    def _auto_run(self):
        self._debounce_id = None
        if self.source_image is None:
            return
        # Auto update: chi cap nhat anh + log, KHONG luu file de tranh lag.
        self._request_run(save_to_disk=False)

    # ------------------------------------------------------------- Anh ---
    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Chon anh khay",
            initialdir=str(INPUT_DIR if INPUT_DIR.is_dir() else PROJECT_ROOT),
            filetypes=[("Anh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Tat ca", "*.*")],
        )
        if path:
            self._load_image(path)

    def _try_load_default_image(self):
        """Mac dinh doc anh dau tien trong data/input/ neu co."""
        first = get_first_image(INPUT_DIR)
        if first is not None:
            self._load_image(str(first))
        else:
            self._set_log([
                "Chua co anh trong data/input/.",
                "Bam 'Chon anh khay' de chon anh, hoac dat anh vao data/input/.",
            ])

    def _load_image(self, path):
        image = read_image(path)
        if image is None:
            self._set_log(["Khong doc duoc anh: {}".format(path)])
            messagebox.showwarning("Loi anh", "Khong doc duoc anh:\n{}".format(path))
            return
        self.source_image = image
        self.image_path_var.set(path)
        # Reset cac anh debug, chi giu anh goc de xem ngay.
        self.display_images = {"Original": image, "Preprocessed": None, "Hough All": None, "Hough Filtered": None}
        self.image_selector_var.set("Original")
        self.detected_var.set("Detected: - / -")
        self.detected_label.configure(fg="#555555")
        self._render_image(image)
        self._set_log([
            "Da chon anh: {}".format(path),
            "Kich thuoc: {}x{}".format(image.shape[1], image.shape[0]),
        ])
        if self.auto_update_var.get():
            self._on_param_changed()

    def _render_image(self, image):
        """Hien thi anh OpenCV len Label (BGR->RGB, resize ban COPY de hien thi)."""
        if image is None:
            self.image_label.configure(image="", text="Chua co anh hien thi")
            self._photo = None
            return
        w = self.image_label.winfo_width()
        h = self.image_label.winfo_height()
        max_w = (w - 8) if w > 50 else 840
        max_h = (h - 8) if h > 50 else 640
        display = resize_for_display(image, max_width=max_w, max_height=max_h)
        rgb = cv_bgr_to_rgb(display)
        pil_image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=self._photo, text="")

    def _show_selected_image(self):
        name = self.image_selector_var.get()
        self._render_image(self.display_images.get(name))

    # ---------------------------------------------------------- Run Hough ---
    def run_hough_manual(self):
        """Nut Run Hough: chay va LUU anh debug ra data/output/hough_tuner/."""
        self._request_run(save_to_disk=True)

    def _max_proc_dim(self):
        """Doc max canh xu ly (px) tu o nhap, co gioi han duoi an toan."""
        try:
            return max(320, int(round(float(self.max_proc_dim_var.get()))))
        except (ValueError, TypeError):
            return 1400

    def _prepare_processing(self, image, params):
        """Tao anh + tham so de chay Hough.

        Neu bat fast preview va anh lon hon max canh xu ly, thu nho anh va scale
        cac tham so do dai theo ti le. Gia tri tren GUI/preset GIU NGUYEN pixel goc.
        """
        if not self.fast_mode_var.get():
            return image, params, 1.0
        height, width = image.shape[:2]
        max_dim = self._max_proc_dim()
        if max(height, width) <= max_dim:
            return image, params, 1.0
        scale = float(max_dim) / float(max(height, width))
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        proc_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return proc_image, self._scale_params(params, scale), scale

    @staticmethod
    def _scale_params(params, scale):
        """Scale cac tham so phu thuoc kich thuoc theo ti le anh."""
        scaled = copy.deepcopy(params)
        hough = scaled.get("hough", {})
        for key in ("minDist", "minRadius", "maxRadius", "min_center_dist"):
            if key in hough:
                hough[key] = max(1.0, float(hough[key]) * scale)
        flt = scaled.get("filter", {})
        if "radius_consensus_tol" in flt:
            flt["radius_consensus_tol"] = max(1, int(round(float(flt["radius_consensus_tol"]) * scale)))
        return scaled

    def _request_run(self, save_to_disk=False):
        """Yeu cau chay Hough; gop request khi worker dang ban (chong don job)."""
        if self.source_image is None:
            self._set_log(["Chua co anh. Hay bam 'Chon anh khay' truoc."])
            return
        try:
            params = self._collect_params()
        except ValueError as exc:
            self._set_log(["Tham so khong hop le: {}".format(exc)])
            return
        # Giu lai preset goc de khong mat cac key khong hien tren GUI.
        self.params = params
        if self._busy:
            prev_save = bool(self._pending) and bool(self._pending.get("save"))
            self._pending = {"save": prev_save or bool(save_to_disk)}
            return
        self._start_worker(params, save_to_disk)

    def _start_worker(self, params, save_to_disk):
        """Chay run_hough_step trong thread nen de GUI khong bi dong.

        Worker TUYET DOI khong dung tk (khong thread-safe); no chi day ket qua
        vao queue, con main thread poll qua _poll_worker.
        """
        image = self.source_image  # tham chieu, run_hough_step khong sua anh goc
        proc_image, proc_params, scale = self._prepare_processing(image, params)
        self._busy = True
        self.detected_var.set("Dang chay Hough...")
        self.detected_label.configure(fg="#888888")

        def worker():
            try:
                result = run_hough_step(proc_image, proc_params)
                error = None
            except Exception as exc:  # marshal loi ve main thread qua queue
                result, error = None, exc
            self._event_queue.put((result, error, save_to_disk, scale, params))

        threading.Thread(target=worker, daemon=True).start()
        if self._poll_id is None:
            self._poll_id = self.root.after(POLL_MS, self._poll_worker)

    def _poll_worker(self):
        """Main thread: doc ket qua worker tu queue va cap nhat GUI."""
        self._poll_id = None
        try:
            while True:
                result, error, save_to_disk, scale, params = self._event_queue.get_nowait()
                self._on_worker_done(result, error, save_to_disk, scale, params)
        except queue.Empty:
            pass
        if self._busy and self._poll_id is None:
            self._poll_id = self.root.after(POLL_MS, self._poll_worker)

    def _on_worker_done(self, result, error, save_to_disk, scale, params):
        """Cap nhat GUI sau khi worker xong + chay request bi gop neu co."""
        self._busy = False
        if error is not None:
            self.detected_var.set("Detected: - / -")
            self.detected_label.configure(fg="#c0392b")
            self._set_log(["Loi chay Hough: {}".format(error)])
        elif result is not None:
            self._apply_result(result, save_to_disk, scale, params)
        pending = self._pending
        self._pending = None
        if pending is not None:
            self._request_run(save_to_disk=bool(pending.get("save")))

    def _apply_result(self, result, save_to_disk, scale, params):
        self.last_result = result
        images = result.get("images", {})
        self.display_images = {
            name: images.get(key) for name, key in DISPLAY_TO_IMAGE_KEY.items()
        }
        filtered = result.get("data", {}).get("circles_filtered", [])
        expected = int(round(float(params.get("expected_count", 12))))
        self._update_detected_label(len(filtered), expected)
        self._update_log(result, params, scale)
        self._show_selected_image()
        if save_to_disk:
            self._save_debug_images()

    def _update_detected_label(self, count, expected):
        self.detected_var.set("Detected: {} / {}".format(count, expected))
        self.detected_label.configure(fg="#1a8a1a" if count == expected else "#c0392b")

    def _save_debug_images(self):
        out_dir = ensure_dir(TUNER_OUTPUT_DIR)
        saved = []
        for display_name, key in DISPLAY_TO_IMAGE_KEY.items():
            image = self.display_images.get(display_name)
            if image is None:
                continue
            write_image(out_dir / "{}.png".format(key), image)
            saved.append("{}.png".format(key))
        self._append_log(["Da luu anh debug vao: {}".format(out_dir)] + ["- " + name for name in saved])

    # ------------------------------------------------------------- Preset ---
    def save_preset_file(self):
        try:
            params = self._collect_params()
        except ValueError as exc:
            self._append_log(["Khong luu duoc preset, tham so loi: {}".format(exc)])
            return
        self.params = params
        save_preset(HOUGH_PRESET_PATH, params)
        self._append_log(["Da luu hough_preset.json", "Tai: {}".format(HOUGH_PRESET_PATH)])

    def load_preset_file(self):
        params = load_preset(HOUGH_PRESET_PATH, DEFAULT_HOUGH_PARAMS)
        self._set_params(params)
        self._append_log(["Da load hough_preset.json"])
        if self.auto_update_var.get():
            self._on_param_changed()

    def reset_params(self):
        self._set_params(DEFAULT_HOUGH_PARAMS)
        self._append_log(["Da reset ve DEFAULT_HOUGH_PARAMS"])

    def open_output_folder(self):
        out_dir = ensure_dir(TUNER_OUTPUT_DIR)
        try:
            os.startfile(str(out_dir))  # Windows
        except AttributeError:
            self._append_log(["He dieu hanh khong ho tro mo thu muc tu dong: {}".format(out_dir)])
        except OSError as exc:
            self._append_log(["Khong mo duoc thu muc: {}".format(exc)])

    # --------------------------------------------------------------- Log ---
    def _update_log(self, result, params, scale=1.0):
        lines = []
        if self.image_path_var.get():
            lines.append("Anh: {}".format(self.image_path_var.get()))
        if scale < 1.0 and self.source_image is not None:
            height, width = self.source_image.shape[:2]
            lines.append(
                "Fast preview: xu ly o {:.0f}% ({}x{} px). Tham so hien thi/preset van la pixel anh goc.".format(
                    scale * 100.0,
                    int(round(width * scale)),
                    int(round(height * scale)),
                )
            )
        lines.extend(result.get("logs", []))

        filtered = result.get("data", {}).get("circles_filtered", [])
        count = len(filtered)
        expected = int(round(float(params.get("expected_count", 12))))
        lines.append("")
        if count > expected:
            lines.append("Phat hien QUA NHIEU circle ({} > {}). Goi y:".format(count, expected))
            lines.append("- Tang hough param2")
            lines.append("- Tang hough minDist")
            lines.append("- Tang min_center_dist")
            lines.append("- Siet minRadius / maxRadius")
        elif count < expected:
            lines.append("Phat hien QUA IT circle ({} < {}). Goi y:".format(count, expected))
            lines.append("- Giam hough param2")
            lines.append("- Mo rong minRadius / maxRadius")
            lines.append("- Kiem tra anh Preprocessed trong combobox")
            lines.append("- Bat/tat CLAHE de so sanh")
        else:
            lines.append("OK: dung {} / {} stator.".format(count, expected))
        self._set_log(lines)

    def _set_log(self, lines):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", "\n".join(lines))
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _append_log(self, lines):
        self.log_text.configure(state="normal")
        existing = self.log_text.get("1.0", "end").strip()
        prefix = (existing + "\n") if existing else ""
        self.log_text.insert("end", prefix + "\n".join(lines))
        self.log_text.configure(state="disabled")
        self.log_text.see("end")


def main():
    ensure_project_dirs()
    root = tk.Tk()
    HoughTunerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
