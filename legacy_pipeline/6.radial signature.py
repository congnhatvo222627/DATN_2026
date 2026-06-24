# -*- coding: utf-8 -*-
"""
6.radial signature.py
GUI xac dinh TAM va GOC XOAY stator bang BIEN DOI CUC (Polar) + tuong quan chuan hoa.

Phien ban nay THAY HAN phuong phap Radial Signature cu bang huong Polar:
- Tam = trong tam mask stator (Otsu + morphology + contour ngoai lon nhat) -> on dinh,
  khong "nhay" nhu Hough Circle.
- Unwrap anh quanh tam sang toa do cuc (warpPolar), chi giu VANH NGOAI (rang + vau)
  -> mot phep xoay vat ly tro thanh phep DICH theo truc goc.
- So khop bang Normalized Cross-Correlation (NCC) theo goc dich tren toan bo van anh cuc
  -> tim goc lam NCC lon nhat, tinh chinh sub-degree bang noi suy parabol.
- Co co che ve sinh warpPolar (WARP_FILL_OUTLIERS + nan_to_num + clip) de tranh
  gia tri rac/khong khoi tao o vung lay mau ngoai bien anh.
- Co tin hieu tin cay: NCC dinh + do net (sigma). NCC thap -> KHONG TIN CAY.

Ket qua tra ve: TAM (cx, cy) tinh bang pixel va GOC XOAY (do) so voi anh mau.
"""

from __future__ import division, print_function

import os
import csv
import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    from tkinter.scrolledtext import ScrolledText
except ImportError:
    import Tkinter as tk
    import ttk
    import tkFileDialog as filedialog
    import tkMessageBox as messagebox
    from ScrolledText import ScrolledText

from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =========================================================
# 1. THONG SO XU LY
# =========================================================

# Tien xu ly
CLAHE_CLIP_LIMIT = 1.0
CLAHE_TILE_GRID_SIZE = (7, 7)
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 2.0

# Tach mask stator de tim trong tam
MASK_MORPH_KERNEL = 7      # kernel dong (close) khep lo nho tren mask

# Bien doi cuc (polar)
POLAR_N_ANG = 360          # so buoc goc (1 do / buoc)
POLAR_N_RAD = 160          # so mau theo ban kinh
POLAR_R_IN_RATIO = 0.55    # ban kinh trong cua vanh (theo R_eq) - bo long stator phang
POLAR_R_OUT_RATIO = 1.40   # ban kinh ngoai cua vanh (theo R_eq) - om het rang + vau

# Do tin cay
NCC_RELIABLE = 0.35        # NCC dinh < nguong nay -> khong tin cay
SHARP_RELIABLE = 2.0       # do net dinh (sigma) < nguong nay -> khong tin cay

# Hien thi
DISPLAY_MAX_WIDTH = 720
DISPLAY_MAX_HEIGHT = 520

DEFAULT_OUTPUT_DIR = os.path.join("data", "test_results", "polar_match")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. DOC / GHI ANH HO TRO DUONG DAN TIENG VIET
# =========================================================

def read_image(path, grayscale=False):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imdecode(data, flag)


def write_image(path, image):
    ext = os.path.splitext(path)[1]
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError("Khong ghi duoc anh: {0}".format(path))
    encoded.tofile(path)


def prepare_gray(image):
    if image is None:
        raise ValueError("Anh dau vao khong hop le")
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    return clahe.apply(gray)


def preprocess(image):
    """gray -> CLAHE -> Gaussian nhe (de tach mask on dinh)."""
    gray = prepare_gray(image)
    enhanced = apply_clahe(gray)
    blur = cv2.GaussianBlur(enhanced, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    return gray, enhanced, blur


# =========================================================
# 3. TIM TAM BANG TRONG TAM MASK (on dinh hon Hough)
# =========================================================

def find_center_by_mask(blur):
    """
    Nhi phan stator (kim loai sang tren nen toi) bang Otsu, dong lo,
    lay contour ngoai lon nhat -> trong tam (cx, cy) va ban kinh tuong duong R_eq.
    Tra ve them mask de hien thi/debug.
    """
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((MASK_MORPH_KERNEL, MASK_MORPH_KERNEL), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = blur.shape[:2]
        return (w / 2.0, h / 2.0), min(w, h) * 0.35, th, False

    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] <= 1e-6:
        h, w = blur.shape[:2]
        return (w / 2.0, h / 2.0), min(w, h) * 0.35, th, False

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    r_eq = float(np.sqrt(cv2.contourArea(c) / np.pi))
    return (cx, cy), r_eq, th, True


# =========================================================
# 4. BIEN DOI CUC (POLAR) + VANH NGOAI
# =========================================================

def polar_annulus(gray, center, r_eq,
                  n_ang=POLAR_N_ANG, n_rad=POLAR_N_RAD,
                  r_in=POLAR_R_IN_RATIO, r_out=POLAR_R_OUT_RATIO):
    """
    Unwrap anh quanh 'center' sang toa do cuc, chi giu vanh [r_in, r_out] * R_eq.
    Tra ve mang (n_ang, n_rad_vanh) float64 da ve sinh gia tri rac.
    Hang = goc (0..360), cot = ban kinh.
    """
    maxr = r_eq * r_out
    src = np.ascontiguousarray(gray, dtype=np.float32)
    p = cv2.warpPolar(src, (n_rad, n_ang), (float(center[0]), float(center[1])), maxr,
                      cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS)
    # Ve sinh: warpPolar co the de lai gia tri rac/NaN o vung lay mau ngoai bien anh
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, 255.0)
    c_in = int(n_rad * (float(r_in) / float(r_out)))
    return p[:, c_in:].astype(np.float64)


def angular_profile(polar):
    """Bien dang 1D theo goc = trung binh cuong do theo ban kinh (de hien thi/bang)."""
    return polar.mean(axis=1)


# =========================================================
# 5. SO KHOP BANG TUONG QUAN CHUAN HOA (NCC) THEO GOC DICH
# =========================================================

def ncc_curve(ref_polar, cur_polar):
    """
    NCC theo tung goc dich s = 0..n_ang-1 tren TOAN BO van anh cuc.
    Mot phep xoay vat ly = mot phep dich hang -> dinh NCC cho biet goc xoay.
    """
    a = ref_polar - ref_polar.mean()
    b = cur_polar - cur_polar.mean()
    na = np.sqrt((a * a).sum())
    nb = np.sqrt((b * b).sum())
    denom = na * nb + 1e-9
    n = ref_polar.shape[0]
    curve = np.empty(n, dtype=np.float64)
    for s in range(n):
        curve[s] = (a * np.roll(b, s, axis=0)).sum() / denom
    return curve


def refine_peak_parabolic(curve, idx):
    """Noi suy parabol quanh dinh cuc dai -> offset sub-degree trong (-1, 1)."""
    n = len(curve)
    yl = float(curve[(idx - 1) % n])
    ym = float(curve[idx])
    yr = float(curve[(idx + 1) % n])
    denom = yl - 2.0 * ym + yr
    if abs(denom) < 1e-12:
        return 0.0
    off = 0.5 * (yl - yr) / denom
    if not np.isfinite(off) or abs(off) > 1.0:
        return 0.0
    return float(off)


def match_polar(ref_result, cur_result):
    """So khop hai ket qua polar -> goc xoay, NCC dinh, do tin cay."""
    ref_polar = ref_result["polar"]
    cur_polar = cur_result["polar"]

    curve = ncc_curve(ref_polar, cur_polar)
    best_shift = int(np.argmax(curve))
    best_ncc = float(curve[best_shift])

    offset = refine_peak_parabolic(curve, best_shift)
    n = len(curve)
    refined_angle = ((best_shift + offset) % n) * (360.0 / n)
    coarse_angle = best_shift * (360.0 / n)

    mean_c = float(curve.mean())
    std_c = float(curve.std())
    sharpness = (best_ncc - mean_c) / (std_c + 1e-9)
    is_reliable = (best_ncc >= NCC_RELIABLE) and (sharpness >= SHARP_RELIABLE)

    # Bien dang goc 1D da dich (de hien thi/bang)
    cur_profile = angular_profile(cur_polar)
    shifted_cur_profile = np.roll(cur_profile, best_shift)

    return {
        "ncc_curve": curve,
        "best_shift": best_shift,
        "coarse_angle": coarse_angle,
        "refined_angle": refined_angle,
        "parabolic_offset": offset,
        "best_ncc": best_ncc,
        "sharpness": sharpness,
        "is_reliable": is_reliable,
        "shifted_cur_profile": shifted_cur_profile,
    }


# =========================================================
# 6. VE ANH HIEN THI (tam + vanh + duong goc)
# =========================================================

def _style(image_shape):
    h, w = image_shape[:2]
    scale = float(np.clip(min(h, w) / 640.0, 0.30, 1.0))
    return {
        "ring": max(1, int(round(2 * scale))),
        "line": max(1, int(round(2 * scale))),
        "dot_in": max(2, int(round(4 * scale))),
        "dot_out": max(4, int(round(9 * scale))),
        "font": max(0.35, 0.6 * scale),
        "txt": max(1, int(round(2 * scale))),
        "lt": cv2.LINE_AA,
    }


def make_overlay(gray, center, r_eq, lines, scale=1.0):
    """
    Ve overlay: tam, vanh trong/ngoai dung de matching, va cac duong goc.
    'lines' = list (angle_deg, color_bgr, label). 'scale' de ve tren anh da resize.
    """
    if len(gray.shape) == 2:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        bgr = gray.copy()
    st = _style(bgr.shape)
    cx = int(round(center[0] * scale))
    cy = int(round(center[1] * scale))
    r_in = int(round(r_eq * POLAR_R_IN_RATIO * scale))
    r_out = int(round(r_eq * POLAR_R_OUT_RATIO * scale))

    # Vanh dung de so khop
    cv2.circle(bgr, (cx, cy), r_in, (0, 165, 255), st["ring"], lineType=st["lt"])
    cv2.circle(bgr, (cx, cy), r_out, (0, 255, 0), st["ring"], lineType=st["lt"])

    for deg, color, label in lines:
        th = np.deg2rad(deg)
        x = int(round(cx + r_out * np.cos(th)))
        y = int(round(cy + r_out * np.sin(th)))
        cv2.line(bgr, (cx, cy), (x, y), color, st["line"], lineType=st["lt"])
        if label:
            cv2.putText(bgr, label, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX,
                        st["font"], color, st["txt"], lineType=st["lt"])

    cv2.circle(bgr, (cx, cy), st["dot_out"], (0, 0, 255), st["ring"], lineType=st["lt"])
    cv2.circle(bgr, (cx, cy), st["dot_in"], (0, 0, 255), -1, lineType=st["lt"])
    cv2.putText(bgr, "Center", (cx + st["dot_out"] + 2, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, st["font"], (0, 0, 255), st["txt"], lineType=st["lt"])
    return bgr


def resize_for_display_with_meta(image_bgr, max_width, max_height, allow_upscale=False):
    h, w = image_bgr.shape[:2]
    if max_width <= 0 or max_height <= 0:
        return image_bgr.copy(), 1.0
    scale = min(max_width / float(w), max_height / float(h))
    if not allow_upscale:
        scale = min(scale, 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=interp), scale


def cv_to_tk_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(image_rgb))


# =========================================================
# 7. PIPELINE XU LY 1 ANH
# =========================================================

def process_single_image(image_path):
    image = read_image(image_path, grayscale=False)
    if image is None:
        raise ValueError("Khong doc duoc anh: {0}".format(image_path))

    gray, enhanced, blur = preprocess(image)
    center, r_eq, mask, ok = find_center_by_mask(blur)
    polar = polar_annulus(enhanced, center, r_eq)
    prof = angular_profile(polar)

    return {
        "path": image_path,
        "gray": gray,
        "enhanced": enhanced,
        "mask": mask,
        "center": (float(center[0]), float(center[1])),
        "center_int": (int(round(center[0])), int(round(center[1]))),
        "r_eq": float(r_eq),
        "mask_ok": bool(ok),
        "polar": polar,
        "ang_profile": prof,
    }


# =========================================================
# 8. GUI
# =========================================================

class PolarMatchGUI(object):
    def __init__(self, root):
        self.root = root
        self.root.title("So khop Polar - xac dinh tam & goc xoay stator")
        self.root.geometry("1680x980")
        self.root.minsize(1300, 760)

        self.reference_path = None
        self.current_path = None
        self.reference_result = None
        self.current_result = None
        self.match_result = None

        self.popup_windows = []
        self.tree_columns = ("angle", "ncc", "ref_prof", "roi_prof")
        self.build_gui()

    # ------------------------- Layout -------------------------
    def build_gui(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        tk.Button(top, text="Chon anh mau", font=("Arial", 11), command=self.choose_reference_image).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Chon anh ROI", font=("Arial", 11), command=self.choose_current_image).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Chay xu ly", font=("Arial", 11), command=self.process_images).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Luu ket qua", font=("Arial", 11), command=self.save_result).pack(side=tk.LEFT, padx=4)

        self.reference_path_label = tk.Label(top, text="Anh mau: chua chon", anchor="w", font=("Arial", 10))
        self.reference_path_label.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)
        self.current_path_label = tk.Label(top, text="Anh ROI: chua chon", anchor="w", font=("Arial", 10))
        self.current_path_label.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

        main = tk.Frame(self.root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.main_paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        compare_frame = ttk.Frame(self.main_paned)
        result_side = ttk.Frame(self.main_paned)
        self.main_paned.add(compare_frame, weight=7)
        self.main_paned.add(result_side, weight=4)

        self.compare_paned = ttk.PanedWindow(compare_frame, orient=tk.HORIZONTAL)
        self.compare_paned.pack(fill=tk.BOTH, expand=True)
        ref_col = ttk.Frame(self.compare_paned)
        cur_col = ttk.Frame(self.compare_paned)
        self.compare_paned.add(ref_col, weight=1)
        self.compare_paned.add(cur_col, weight=1)

        self.reference_image_panel = self.build_image_section(ref_col, "Anh mau + tam/vanh")
        self.reference_plot_panel = self.build_plot_section(ref_col, "Anh cuc (unwrap) mau", "Phong to mau", self.open_reference_plot_popup)
        self.current_image_panel = self.build_image_section(cur_col, "Anh ROI + tam/vanh + goc")
        self.current_plot_panel = self.build_plot_section(cur_col, "Anh cuc (unwrap) ROI", "Phong to ROI", self.open_current_plot_popup)

        self.right_paned = ttk.PanedWindow(result_side, orient=tk.VERTICAL)
        self.right_paned.pack(fill=tk.BOTH, expand=True)
        result_frame = ttk.Frame(self.right_paned)
        ncc_frame = ttk.Frame(self.right_paned)
        table_frame = ttk.Frame(self.right_paned)
        self.right_paned.add(result_frame, weight=2)
        self.right_paned.add(ncc_frame, weight=3)
        self.right_paned.add(table_frame, weight=5)

        self.build_header(result_frame, "Ket qua so khop", "Phong to", self.open_result_popup)
        body = tk.Frame(result_frame)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.info_text = ScrolledText(body, width=48, height=16, font=("Consolas", 10), wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        self.build_header(ncc_frame, "NCC theo goc dich", "Phong to NCC", self.open_ncc_popup)
        ncc_body = tk.Frame(ncc_frame)
        ncc_body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.ncc_figure = Figure(figsize=(5.8, 2.8), dpi=100)
        self.ncc_axes = self.ncc_figure.add_subplot(111)
        self.ncc_canvas = FigureCanvasTkAgg(self.ncc_figure, master=ncc_body)
        self.ncc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.enable_ctrl_mousewheel_zoom(self.ncc_axes, self.ncc_canvas)
        self.draw_empty_ncc_plot()

        self.build_header(table_frame, "Bang gia tri theo goc", "Phong to bang", self.open_table_popup)
        tbody = tk.Frame(table_frame)
        tbody.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.tree = ttk.Treeview(tbody, columns=self.tree_columns, show="headings", height=26)
        self.configure_tree_columns(self.tree)
        sy = ttk.Scrollbar(tbody, orient=tk.VERTICAL, command=self.tree.yview)
        sx = ttk.Scrollbar(tbody, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        tbody.grid_rowconfigure(0, weight=1)
        tbody.grid_columnconfigure(0, weight=1)

    def build_image_section(self, parent, title):
        frame = tk.LabelFrame(parent, text=title, font=("Arial", 11))
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 8))
        frame.configure(height=DISPLAY_MAX_HEIGHT + 40)
        frame.pack_propagate(False)
        label = tk.Label(frame, bg="black", relief=tk.SUNKEN, anchor="center")
        label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        panel = {"frame": frame, "label": label, "result": None, "kind": "ref", "preview_bgr": None, "resize_job": None}
        label.bind("<Configure>", lambda e, p=panel: self.schedule_panel_render(p))
        return panel

    def build_plot_section(self, parent, title, button_text, button_command):
        frame = tk.LabelFrame(parent, text=title, font=("Arial", 11))
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 8))
        header = tk.Frame(frame)
        header.pack(fill=tk.X, padx=6, pady=(6, 2))
        tk.Button(header, text=button_text, font=("Arial", 9), command=button_command).pack(side=tk.RIGHT)
        body = tk.Frame(frame)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        figure = Figure(figsize=(5.6, 2.2), dpi=100)
        axes = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=body)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.enable_ctrl_mousewheel_zoom(axes, canvas)
        panel = {"frame": frame, "figure": figure, "axes": axes, "canvas": canvas}
        self.draw_empty_polar_plot(panel, title)
        return panel

    def build_header(self, parent, title, button_text, button_command):
        header = tk.Frame(parent)
        header.pack(fill=tk.X, padx=6, pady=(6, 4))
        tk.Label(header, text=title, font=("Arial", 11, "bold"), anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(header, text=button_text, font=("Arial", 9), command=button_command).pack(side=tk.RIGHT)

    # ------------------------- Hien thi anh -------------------------
    def get_panel_target_size(self, panel):
        label = panel["label"]
        label.update_idletasks()
        w = label.winfo_width()
        h = label.winfo_height()
        if w <= 1:
            w = DISPLAY_MAX_WIDTH
        if h <= 1:
            h = DISPLAY_MAX_HEIGHT
        return max(1, w - 8), max(1, h - 8)

    def schedule_panel_render(self, panel):
        prev = panel.get("resize_job")
        if prev is not None:
            try:
                self.root.after_cancel(prev)
            except Exception:
                pass
        panel["resize_job"] = self.root.after(60, lambda p=panel: self._finish_panel_render(p))

    def _finish_panel_render(self, panel):
        panel["resize_job"] = None
        self.render_image_panel(panel)

    def build_overlay_for_panel(self, panel, target_w, target_h):
        result = panel.get("result")
        gray = result["gray"]
        base_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        _, scale = resize_for_display_with_meta(base_bgr, target_w, target_h, allow_upscale=True)
        resized_gray = cv2.resize(gray, (max(1, int(round(gray.shape[1] * scale))),
                                          max(1, int(round(gray.shape[0] * scale)))),
                                  interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)

        # Duong goc: anh mau co duong 0 do; anh ROI them duong goc xoay tim duoc
        lines = [(0.0, (0, 0, 255), "0")]
        if panel.get("kind") == "cur" and self.match_result is not None:
            ang = self.match_result["refined_angle"]
            lines.append((ang, (255, 255, 0), "%.1f" % ang))
        return make_overlay(resized_gray, result["center"], result["r_eq"], lines, scale=scale)

    def render_image_panel(self, panel):
        target_w, target_h = self.get_panel_target_size(panel)
        if panel.get("result") is not None:
            display_bgr = self.build_overlay_for_panel(panel, target_w, target_h)
        elif panel.get("preview_bgr") is not None:
            display_bgr, _ = resize_for_display_with_meta(panel["preview_bgr"], target_w, target_h, allow_upscale=True)
        else:
            return
        image_tk = cv_to_tk_image(display_bgr)
        panel["label"].config(image=image_tk)
        panel["label"].image = image_tk

    def preview_original_image(self, file_path, panel, plot_panel, title_plot):
        image = read_image(file_path, grayscale=False)
        if image is None:
            messagebox.showerror("Loi", "Khong doc duoc anh: {0}".format(file_path))
            return
        gray = prepare_gray(image)
        panel["preview_bgr"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        panel["result"] = None
        self.render_image_panel(panel)
        self.draw_empty_polar_plot(plot_panel, title_plot)

    # ------------------------- Plot -------------------------
    def draw_empty_polar_plot(self, plot_panel, title):
        ax = plot_panel["axes"]
        ax.clear()
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Ban kinh (mau)", fontsize=8)
        ax.set_ylabel("Goc (deg)", fontsize=8)
        ax.text(0.5, 0.5, "Chua co anh cuc", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        plot_panel["figure"].subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.20)
        plot_panel["canvas"].draw_idle()

    def draw_polar_on_axes(self, axes, polar, title, mark_angle=None):
        axes.clear()
        axes.imshow(polar, aspect="auto", origin="lower", cmap="gray",
                    extent=[0, polar.shape[1], 0, 360])
        if mark_angle is not None:
            axes.axhline(mark_angle % 360, color="#00d0ff", linestyle="--", linewidth=1.2)
        axes.axhline(0, color="red", linestyle="--", linewidth=1.0)
        axes.set_title(title, fontsize=9)
        axes.set_xlabel("Ban kinh (mau)", fontsize=8)
        axes.set_ylabel("Goc (deg)", fontsize=8)
        axes.tick_params(axis="both", labelsize=8)

    def refresh_polar_plot(self, plot_panel, polar, title, mark_angle=None):
        self.draw_polar_on_axes(plot_panel["axes"], polar, title, mark_angle)
        plot_panel["figure"].subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.20)
        plot_panel["canvas"].draw_idle()

    def draw_empty_ncc_plot(self):
        ax = self.ncc_axes
        ax.clear()
        ax.set_title("NCC theo goc dich")
        ax.set_xlabel("Goc dich (deg)")
        ax.set_ylabel("NCC")
        ax.text(0.5, 0.5, "Chua co du lieu NCC", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, alpha=0.25)
        self.ncc_figure.tight_layout()
        self.ncc_canvas.draw_idle()

    def draw_ncc_on_axes(self, axes, match):
        axes.clear()
        curve = match["ncc_curve"]
        x = np.arange(len(curve))
        axes.plot(x, curve, color="#1f77b4", linewidth=1.6, label="NCC")
        axes.axvline(match["best_shift"], color="red", linestyle="--", linewidth=1.2,
                     label="Goc = {0:.2f} deg".format(match["refined_angle"]))
        axes.scatter([match["best_shift"]], [match["best_ncc"]], color="red", s=34, zorder=3)
        flag = "TIN CAY" if match["is_reliable"] else "KHONG TIN CAY"
        axes.set_title("NCC dinh = {0:.3f} | do net = {1:.1f} sigma | {2}".format(
            match["best_ncc"], match["sharpness"], flag))
        axes.set_xlabel("Goc dich (deg)")
        axes.set_ylabel("NCC")
        axes.set_xlim(0, max(359, len(curve) - 1))
        axes.grid(True, alpha=0.3)
        axes.legend(loc="best")

    def refresh_ncc_plot(self):
        if self.match_result is None:
            self.draw_empty_ncc_plot()
            return
        self.draw_ncc_on_axes(self.ncc_axes, self.match_result)
        self.ncc_figure.tight_layout()
        self.ncc_canvas.draw_idle()

    def enable_ctrl_mousewheel_zoom(self, axes, canvas):
        def on_scroll(event):
            if event.inaxes != axes:
                return
            key = str(event.key).lower() if event.key is not None else ""
            if "control" not in key and "ctrl" not in key:
                return
            factor = 0.9 if event.button == "up" else 1.1
            x_min, x_max = axes.get_xlim()
            y_min, y_max = axes.get_ylim()
            xc = event.xdata if event.xdata is not None else (x_min + x_max) / 2.0
            yc = event.ydata if event.ydata is not None else (y_min + y_max) / 2.0
            axes.set_xlim(xc - (xc - x_min) * factor, xc + (x_max - xc) * factor)
            axes.set_ylim(yc - (yc - y_min) * factor, yc + (y_max - yc) * factor)
            canvas.draw_idle()
        canvas.mpl_connect("scroll_event", on_scroll)

    # ------------------------- Table -------------------------
    def configure_tree_columns(self, tree):
        tree.heading("angle", text="Goc")
        tree.heading("ncc", text="NCC")
        tree.heading("ref_prof", text="Profile mau")
        tree.heading("roi_prof", text="Profile ROI dich")
        tree.column("angle", width=60, anchor="center", stretch=False)
        tree.column("ncc", width=100, anchor="center", stretch=False)
        tree.column("ref_prof", width=110, anchor="center", stretch=False)
        tree.column("roi_prof", width=130, anchor="center", stretch=False)

    def populate_tree(self, tree):
        for item in tree.get_children():
            tree.delete(item)
        if self.reference_result is None or self.match_result is None:
            return
        curve = self.match_result["ncc_curve"]
        ref_prof = self.reference_result["ang_profile"]
        roi_prof = self.match_result["shifted_cur_profile"]
        n = len(curve)
        for i in range(n):
            tree.insert("", tk.END, values=(
                "{0}".format(i),
                "{0:.4f}".format(curve[i]),
                "{0:.2f}".format(ref_prof[i]),
                "{0:.2f}".format(roi_prof[i]),
            ))

    def update_table(self):
        self.populate_tree(self.tree)

    # ------------------------- Commands -------------------------
    def choose_reference_image(self):
        path = filedialog.askopenfilename(
            title="Chon anh mau",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return
        self.reference_path = path
        self.reference_path_label.config(text="Anh mau: {0}".format(path))
        self.preview_original_image(path, self.reference_image_panel, self.reference_plot_panel, "Anh cuc (unwrap) mau")

    def choose_current_image(self):
        path = filedialog.askopenfilename(
            title="Chon anh ROI hien tai",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return
        self.current_path = path
        self.current_path_label.config(text="Anh ROI: {0}".format(path))
        self.preview_original_image(path, self.current_image_panel, self.current_plot_panel, "Anh cuc (unwrap) ROI")

    def process_images(self):
        if not self.reference_path:
            messagebox.showwarning("Canh bao", "Vui long chon anh mau truoc.")
            return
        if not self.current_path:
            messagebox.showwarning("Canh bao", "Vui long chon anh ROI truoc.")
            return
        try:
            self.reference_result = process_single_image(self.reference_path)
            self.current_result = process_single_image(self.current_path)
            self.match_result = match_polar(self.reference_result, self.current_result)
            self.update_visuals()
            self.update_info_text()
            self.update_table()
        except Exception as exc:
            messagebox.showerror("Loi xu ly", str(exc))

    def update_visuals(self):
        self.reference_image_panel["result"] = self.reference_result
        self.reference_image_panel["preview_bgr"] = None
        self.reference_image_panel["kind"] = "ref"
        self.current_image_panel["result"] = self.current_result
        self.current_image_panel["preview_bgr"] = None
        self.current_image_panel["kind"] = "cur"
        self.render_image_panel(self.reference_image_panel)
        self.render_image_panel(self.current_image_panel)

        self.refresh_polar_plot(self.reference_plot_panel, self.reference_result["polar"], "Anh cuc (unwrap) mau", mark_angle=0)
        self.refresh_polar_plot(self.current_plot_panel, self.current_result["polar"], "Anh cuc (unwrap) ROI",
                                mark_angle=self.match_result["refined_angle"])
        self.refresh_ncc_plot()

    def update_info_text(self):
        ref = self.reference_result
        cur = self.current_result
        match = self.match_result
        flag = "TIN CAY" if match["is_reliable"] else "KHONG TIN CAY (NCC thap)"
        inverse_angle = (360.0 - match["refined_angle"]) % 360.0
        lines = [
            "KET QUA SO KHOP POLAR",
            "",
            "TAM ROI (pixel)   : ({0:.1f}, {1:.1f})".format(cur["center"][0], cur["center"][1]),
            "TAM mau (pixel)   : ({0:.1f}, {1:.1f})".format(ref["center"][0], ref["center"][1]),
            "GOC XOAY (do)     : {0:.2f}".format(match["refined_angle"]),
            "Goc tho (do)      : {0:.0f}".format(match["coarse_angle"]),
            "Goc dao chieu     : {0:.2f}".format(inverse_angle),
            "NCC dinh          : {0:.4f}".format(match["best_ncc"]),
            "Do net dinh       : {0:.2f} sigma".format(match["sharpness"]),
            "Offset parabol    : {0:.4f} buoc".format(match["parabolic_offset"]),
            "Danh gia          : {0}".format(flag),
            "",
            "Anh mau:",
            "  Duong dan : {0}".format(ref["path"]),
            "  R_eq      : {0:.1f}".format(ref["r_eq"]),
            "  Mask OK   : {0}".format(ref["mask_ok"]),
            "",
            "Anh ROI:",
            "  Duong dan : {0}".format(cur["path"]),
            "  R_eq      : {0:.1f}".format(cur["r_eq"]),
            "  Mask OK   : {0}".format(cur["mask_ok"]),
            "",
            "Ghi chu:",
            "  Tam = trong tam mask stator (pixel).",
            "  Goc xoay = goc dich lam NCC anh cuc lon nhat (tuong doi so voi anh mau).",
            "  Vanh so khop: [{0:.2f}, {1:.2f}] x R_eq.".format(POLAR_R_IN_RATIO, POLAR_R_OUT_RATIO),
        ]
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, "\n".join(lines))

    # ------------------------- Popup -------------------------
    def get_result_text(self):
        return self.info_text.get("1.0", tk.END).strip()

    def open_result_popup(self):
        if not self.get_result_text():
            messagebox.showwarning("Canh bao", "Chua co ket qua de phong to.")
            return
        popup = tk.Toplevel(self.root)
        popup.title("Ket qua so khop - phong to")
        popup.geometry("900x700")
        frame = tk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        text = tk.Text(frame, font=("Consolas", 11), wrap=tk.WORD)
        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=sb.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        text.insert("1.0", self.get_result_text())
        self.popup_windows.append(popup)

    def open_ncc_popup(self):
        if self.match_result is None:
            messagebox.showwarning("Canh bao", "Chua co du lieu NCC de phong to.")
            return
        popup = tk.Toplevel(self.root)
        popup.title("Do thi NCC theo goc dich")
        popup.geometry("1200x780")
        frame = tk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        self.draw_ncc_on_axes(ax, self.match_result)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.enable_ctrl_mousewheel_zoom(ax, canvas)
        canvas.draw_idle()
        self.popup_windows.append(popup)

    def open_reference_plot_popup(self):
        if self.reference_result is None:
            messagebox.showwarning("Canh bao", "Chua co anh cuc mau de phong to.")
            return
        self.open_polar_popup("Anh cuc (unwrap) mau", self.reference_result["polar"], 0)

    def open_current_plot_popup(self):
        if self.current_result is None:
            messagebox.showwarning("Canh bao", "Chua co anh cuc ROI de phong to.")
            return
        mark = self.match_result["refined_angle"] if self.match_result is not None else None
        self.open_polar_popup("Anh cuc (unwrap) ROI", self.current_result["polar"], mark)

    def open_polar_popup(self, title, polar, mark_angle):
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("1100x760")
        frame = tk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        fig = Figure(figsize=(9, 6), dpi=100)
        ax = fig.add_subplot(111)
        self.draw_polar_on_axes(ax, polar, title, mark_angle)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.enable_ctrl_mousewheel_zoom(ax, canvas)
        canvas.draw_idle()
        self.popup_windows.append(popup)

    def open_table_popup(self):
        if self.reference_result is None or self.match_result is None:
            messagebox.showwarning("Canh bao", "Chua co bang du lieu de phong to.")
            return
        popup = tk.Toplevel(self.root)
        popup.title("Bang gia tri theo goc - phong to")
        popup.geometry("1100x760")
        frame = tk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        tree = ttk.Treeview(frame, columns=self.tree_columns, show="headings")
        self.configure_tree_columns(tree)
        sy = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        sx = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        self.populate_tree(tree)
        self.popup_windows.append(popup)

    # ------------------------- Save -------------------------
    def save_result(self):
        if self.reference_result is None or self.current_result is None or self.match_result is None:
            messagebox.showwarning("Canh bao", "Chua co ket qua de luu.")
            return
        path = filedialog.asksaveasfilename(
            title="Luu ket qua", defaultextension=".csv",
            filetypes=[("CSV file", "*.csv"), ("Text file", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            if os.path.splitext(path)[1].lower() == ".txt":
                with open(path, "w", encoding="utf-8-sig") as f:
                    f.write(self.get_result_text())
            else:
                self.save_result_csv(path)
            messagebox.showinfo("Thanh cong", "Da luu ket qua:\n{0}".format(path))
        except Exception as exc:
            messagebox.showerror("Loi", str(exc))

    def save_result_csv(self, path):
        ref = self.reference_result
        cur = self.current_result
        match = self.match_result
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["reference_image_path", ref["path"]])
            w.writerow(["current_roi_image_path", cur["path"]])
            w.writerow(["center_ref_x", "{0:.3f}".format(ref["center"][0])])
            w.writerow(["center_ref_y", "{0:.3f}".format(ref["center"][1])])
            w.writerow(["r_eq_ref", "{0:.3f}".format(ref["r_eq"])])
            w.writerow(["center_roi_x", "{0:.3f}".format(cur["center"][0])])
            w.writerow(["center_roi_y", "{0:.3f}".format(cur["center"][1])])
            w.writerow(["r_eq_roi", "{0:.3f}".format(cur["r_eq"])])
            w.writerow(["angle_refined_deg", "{0:.4f}".format(match["refined_angle"])])
            w.writerow(["angle_coarse_deg", "{0:.4f}".format(match["coarse_angle"])])
            w.writerow(["best_ncc", "{0:.6f}".format(match["best_ncc"])])
            w.writerow(["sharpness_sigma", "{0:.4f}".format(match["sharpness"])])
            w.writerow(["is_reliable", int(match["is_reliable"])])
            w.writerow([])
            w.writerow(["angle_deg", "ncc", "ref_profile", "roi_profile_shifted"])
            curve = match["ncc_curve"]
            ref_prof = ref["ang_profile"]
            roi_prof = match["shifted_cur_profile"]
            for i in range(len(curve)):
                w.writerow([i, "{0:.6f}".format(curve[i]),
                            "{0:.6f}".format(ref_prof[i]), "{0:.6f}".format(roi_prof[i])])


if __name__ == "__main__":
    root = tk.Tk()
    app = PolarMatchGUI(root)
    root.mainloop()
