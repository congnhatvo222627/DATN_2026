# -*- coding: utf-8 -*-
"""
GUI giam sat tien xu ly ROI truoc Canny phuc vu Radial Signature.

Luong xu ly:
ROI goc -> Anh xam -> Tang tuong phan nhe neu can -> Loc nhieu -> Canny

Chuc nang:
- Chon anh ROI stator don.
- Hien thi ket qua sau tung buoc tren man hinh.
- Tuy chinh cac thong so quan trong ben trai.
- Nut Reset dua ve bo thong so chuan ban dau.
- Luu tung anh trung gian va hinh tong hop vao thu muc output.

Yeu cau thu vien:
pip install opencv-python numpy matplotlib
"""

import copy
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =========================================================
# 1. DUONG DAN MAC DINH
# =========================================================
DEFAULT_ROI_PATH = "data/test_results/roi_images/roi_stator_01.png"
OUTPUT_DIR = "data/test_results/roi_canny_monitor"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. BO THONG SO CHUAN BAN DAU
# =========================================================
DEFAULT_CONFIG = {
    "contrast": {
        "use_clahe": False,
        "clipLimit": 2.0,
        "tileGridSize": 8,
    },

    "denoise": {
        # Cac gia tri: "Gaussian", "Median", "Bilateral", "None"
        "method": "Gaussian",
        "gaussian_kernel": 5,
        "gaussian_sigma": 0.0,
        "median_kernel": 5,
        "bilateral_d": 7,
        "bilateral_sigmaColor": 50,
        "bilateral_sigmaSpace": 50,
    },

    "canny": {
        "low": 60,
        "high": 160,
        # OpenCV Canny chi chap nhan aperture_size = 3, 5, 7
        "aperture_size": 3,
        "L2gradient": False,
    },

    "display": {
        "dpi": 300,
        "save_outputs": True,
    }
}

CONFIG = copy.deepcopy(DEFAULT_CONFIG)


# =========================================================
# 3. HAM DOC/GHI ANH HO TRO DUONG DAN TIENG VIET
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
        raise ValueError("Khong the ghi anh: {}".format(path))
    encoded.tofile(path)


def make_odd(value, minimum=1):
    k = max(minimum, int(value))
    if k % 2 == 0:
        k += 1
    return k


def valid_canny_aperture(value):
    value = int(value)
    if value <= 3:
        return 3
    if value <= 5:
        return 5
    return 7


# =========================================================
# 4. CAC BUOC XU LY ANH ROI
# =========================================================
def convert_to_gray(roi):
    if roi is None:
        raise ValueError("Anh ROI rong.")
    if len(roi.shape) == 3:
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return roi.copy()


def enhance_contrast_if_needed(gray):
    if not CONFIG["contrast"]["use_clahe"]:
        return gray.copy(), "Khong dung CLAHE"

    tile = max(1, int(CONFIG["contrast"]["tileGridSize"]))
    clahe = cv2.createCLAHE(
        clipLimit=float(CONFIG["contrast"]["clipLimit"]),
        tileGridSize=(tile, tile)
    )
    enhanced = clahe.apply(gray)
    return enhanced, "CLAHE"


def denoise_image(image):
    method = str(CONFIG["denoise"]["method"])

    if method == "None":
        return image.copy(), "Khong loc nhieu"

    if method == "Gaussian":
        k = make_odd(CONFIG["denoise"]["gaussian_kernel"], 1)
        sigma = float(CONFIG["denoise"]["gaussian_sigma"])
        out = cv2.GaussianBlur(image, (k, k), sigma)
        return out, "Gaussian Blur"

    if method == "Median":
        k = make_odd(CONFIG["denoise"]["median_kernel"], 1)
        out = cv2.medianBlur(image, k)
        return out, "Median Blur"

    if method == "Bilateral":
        d = max(1, int(CONFIG["denoise"]["bilateral_d"]))
        sigma_color = max(1, int(CONFIG["denoise"]["bilateral_sigmaColor"]))
        sigma_space = max(1, int(CONFIG["denoise"]["bilateral_sigmaSpace"]))
        out = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return out, "Bilateral Filter"

    raise ValueError("Phuong phap loc nhieu khong hop le: {}".format(method))


def apply_canny(image):
    low = max(0, int(CONFIG["canny"]["low"]))
    high = max(low + 1, int(CONFIG["canny"]["high"]))
    aperture = valid_canny_aperture(CONFIG["canny"]["aperture_size"])
    l2 = bool(CONFIG["canny"]["L2gradient"])

    edges = cv2.Canny(
        image,
        threshold1=low,
        threshold2=high,
        apertureSize=aperture,
        L2gradient=l2
    )
    return edges


def process_roi(roi, save_outputs=True):
    """Tra ve tat ca anh trung gian trong pipeline."""
    roi_display = roi.copy()
    if len(roi_display.shape) == 2:
        roi_display = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR)

    gray = convert_to_gray(roi)
    enhanced, contrast_name = enhance_contrast_if_needed(gray)
    denoised, denoise_name = denoise_image(enhanced)
    edges = apply_canny(denoised)

    if save_outputs:
        write_image(os.path.join(OUTPUT_DIR, "01_roi_goc.png"), roi_display)
        write_image(os.path.join(OUTPUT_DIR, "02_anh_xam.png"), gray)
        write_image(os.path.join(OUTPUT_DIR, "03_tang_tuong_phan.png"), enhanced)
        write_image(os.path.join(OUTPUT_DIR, "04_loc_nhieu.png"), denoised)
        write_image(os.path.join(OUTPUT_DIR, "05_canny.png"), edges)

    fig = create_result_figure(
        roi_display=roi_display,
        gray=gray,
        enhanced=enhanced,
        denoised=denoised,
        edges=edges,
        contrast_name=contrast_name,
        denoise_name=denoise_name,
    )

    figure_path = os.path.join(OUTPUT_DIR, "Hinh_giam_sat_tien_xu_ly_ROI_truoc_Canny.png")
    if save_outputs:
        fig.savefig(
            figure_path,
            dpi=int(CONFIG["display"]["dpi"]),
            bbox_inches="tight"
        )

    return {
        "figure": fig,
        "figure_path": figure_path,
        "roi_display": roi_display,
        "gray": gray,
        "enhanced": enhanced,
        "denoised": denoised,
        "edges": edges,
        "contrast_name": contrast_name,
        "denoise_name": denoise_name,
    }


# =========================================================
# 5. TAO HINH HIEN THI CAC BUOC
# =========================================================
def create_result_figure(roi_display, gray, enhanced, denoised, edges, contrast_name, denoise_name):
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.ravel()

    axes[0].imshow(cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB))
    axes[0].set_title("(a) ROI goc", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("(b) Anh xam", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(enhanced, cmap="gray")
    axes[2].set_title("(c) Tang tuong phan: {}".format(contrast_name), fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(denoised, cmap="gray")
    axes[3].set_title("(d) Loc nhieu: {}".format(denoise_name), fontsize=11)
    axes[3].axis("off")

    axes[4].imshow(edges, cmap="gray")
    axes[4].set_title("(e) Canny", fontsize=11)
    axes[4].axis("off")

    # O cuoi hien thi lai Canny phong to/ket qua dau ra
    axes[5].imshow(edges, cmap="gray")
    axes[5].set_title("(f) Anh bien dau ra", fontsize=11)
    axes[5].axis("off")

    fig.tight_layout()
    return fig


# =========================================================
# 6. GUI
# =========================================================
def run_gui():
    root = tk.Tk()
    root.title("Giam sat tien xu ly ROI truoc Canny")
    root.geometry("1480x900")
    root.minsize(1250, 780)

    left_panel = tk.Frame(root, width=390, padx=12, pady=12, bg="#f3f4f6")
    left_panel.pack(side="left", fill="y")
    left_panel.pack_propagate(False)

    right_panel = tk.Frame(root, padx=8, pady=8)
    right_panel.pack(side="right", fill="both", expand=True)

    canvas_holder = tk.Frame(right_panel, bg="white", bd=1, relief="solid")
    canvas_holder.pack(fill="both", expand=True)

    roi_path_var = tk.StringVar(value=DEFAULT_ROI_PATH)
    status_var = tk.StringVar(value="Chon anh ROI, chinh tham so neu can, sau do bam Run.")
    canvas_state = {"canvas": None}

    vars_gui = {}

    def load_vars_from_config():
        vars_gui["use_clahe"].set(CONFIG["contrast"]["use_clahe"])
        vars_gui["clipLimit"].set(str(CONFIG["contrast"]["clipLimit"]))
        vars_gui["tileGridSize"].set(str(CONFIG["contrast"]["tileGridSize"]))

        vars_gui["denoise_method"].set(CONFIG["denoise"]["method"])
        vars_gui["gaussian_kernel"].set(str(CONFIG["denoise"]["gaussian_kernel"]))
        vars_gui["gaussian_sigma"].set(str(CONFIG["denoise"]["gaussian_sigma"]))
        vars_gui["median_kernel"].set(str(CONFIG["denoise"]["median_kernel"]))
        vars_gui["bilateral_d"].set(str(CONFIG["denoise"]["bilateral_d"]))
        vars_gui["bilateral_sigmaColor"].set(str(CONFIG["denoise"]["bilateral_sigmaColor"]))
        vars_gui["bilateral_sigmaSpace"].set(str(CONFIG["denoise"]["bilateral_sigmaSpace"]))

        vars_gui["canny_low"].set(str(CONFIG["canny"]["low"]))
        vars_gui["canny_high"].set(str(CONFIG["canny"]["high"]))
        vars_gui["aperture_size"].set(str(CONFIG["canny"]["aperture_size"]))
        vars_gui["L2gradient"].set(CONFIG["canny"]["L2gradient"])

        vars_gui["dpi"].set(str(CONFIG["display"]["dpi"]))
        vars_gui["save_outputs"].set(CONFIG["display"]["save_outputs"])

    def apply_gui_settings():
        try:
            CONFIG["contrast"]["use_clahe"] = bool(vars_gui["use_clahe"].get())
            CONFIG["contrast"]["clipLimit"] = max(0.1, float(vars_gui["clipLimit"].get()))
            CONFIG["contrast"]["tileGridSize"] = max(1, int(vars_gui["tileGridSize"].get()))

            method = vars_gui["denoise_method"].get()
            if method not in ["Gaussian", "Median", "Bilateral", "None"]:
                raise ValueError("Denoise method khong hop le.")
            CONFIG["denoise"]["method"] = method
            CONFIG["denoise"]["gaussian_kernel"] = make_odd(vars_gui["gaussian_kernel"].get(), 1)
            CONFIG["denoise"]["gaussian_sigma"] = max(0.0, float(vars_gui["gaussian_sigma"].get()))
            CONFIG["denoise"]["median_kernel"] = make_odd(vars_gui["median_kernel"].get(), 1)
            CONFIG["denoise"]["bilateral_d"] = max(1, int(vars_gui["bilateral_d"].get()))
            CONFIG["denoise"]["bilateral_sigmaColor"] = max(1, int(vars_gui["bilateral_sigmaColor"].get()))
            CONFIG["denoise"]["bilateral_sigmaSpace"] = max(1, int(vars_gui["bilateral_sigmaSpace"].get()))

            low = max(0, int(vars_gui["canny_low"].get()))
            high = max(low + 1, int(vars_gui["canny_high"].get()))
            aperture = valid_canny_aperture(vars_gui["aperture_size"].get())
            CONFIG["canny"]["low"] = low
            CONFIG["canny"]["high"] = high
            CONFIG["canny"]["aperture_size"] = aperture
            CONFIG["canny"]["L2gradient"] = bool(vars_gui["L2gradient"].get())

            CONFIG["display"]["dpi"] = max(72, int(vars_gui["dpi"].get()))
            CONFIG["display"]["save_outputs"] = bool(vars_gui["save_outputs"].get())

            # Cap nhat lai cac o neu code da tu sua so chan thanh so le
            load_vars_from_config()

        except ValueError as exc:
            raise ValueError("Thong so nhap tren giao dien khong hop le.\n{}".format(exc))

    def clear_canvas():
        if canvas_state["canvas"] is not None:
            plt.close(canvas_state["canvas"].figure)
            canvas_state["canvas"].get_tk_widget().destroy()
            canvas_state["canvas"] = None

    def choose_roi():
        file_path = filedialog.askopenfilename(
            title="Chon anh ROI",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            roi_path_var.set(file_path)
            status_var.set("Da chon ROI. Bam Run de xu ly.")

    def run_processing():
        try:
            apply_gui_settings()
            path = roi_path_var.get().strip()
            if not path:
                raise ValueError("Chua chon anh ROI.")
            if not os.path.isfile(path):
                raise ValueError("Khong tim thay anh ROI: {}".format(path))

            roi = read_image(path, grayscale=False)
            if roi is None:
                raise ValueError("Khong doc duoc anh ROI: {}".format(path))

            status_var.set("Dang xu ly...")
            root.update_idletasks()
            clear_canvas()

            outputs = process_roi(
                roi,
                save_outputs=bool(CONFIG["display"]["save_outputs"])
            )

            canvas = FigureCanvasTkAgg(outputs["figure"], master=canvas_holder)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas_state["canvas"] = canvas

            h, w = roi.shape[:2]
            edge_pixels = int((outputs["edges"] > 0).sum())
            status_var.set(
                "Hoan tat. ROI: {}x{}. Pixel bien Canny: {}. Luu tai: {}".format(
                    w, h, edge_pixels, outputs["figure_path"]
                )
            )

        except Exception as exc:
            messagebox.showerror("Loi xu ly", str(exc))
            status_var.set("Xu ly that bai.")

    def reset_defaults():
        global CONFIG
        CONFIG = copy.deepcopy(DEFAULT_CONFIG)
        load_vars_from_config()
        status_var.set("Da reset ve bo thong so chuan ban dau. Bam Run de xu ly lai.")

    def add_entry(parent, label_text, variable, width=10):
        row = tk.Frame(parent, bg="#f3f4f6")
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label_text, width=22, anchor="w", bg="#f3f4f6").pack(side="left")
        tk.Entry(row, textvariable=variable, width=width).pack(side="right")

    # Khoi tao bien Tkinter
    vars_gui.update({
        "use_clahe": tk.BooleanVar(),
        "clipLimit": tk.StringVar(),
        "tileGridSize": tk.StringVar(),
        "denoise_method": tk.StringVar(),
        "gaussian_kernel": tk.StringVar(),
        "gaussian_sigma": tk.StringVar(),
        "median_kernel": tk.StringVar(),
        "bilateral_d": tk.StringVar(),
        "bilateral_sigmaColor": tk.StringVar(),
        "bilateral_sigmaSpace": tk.StringVar(),
        "canny_low": tk.StringVar(),
        "canny_high": tk.StringVar(),
        "aperture_size": tk.StringVar(),
        "L2gradient": tk.BooleanVar(),
        "dpi": tk.StringVar(),
        "save_outputs": tk.BooleanVar(),
    })
    load_vars_from_config()

    # =====================================================
    # GIAO DIEN BEN TRAI
    # =====================================================
    tk.Label(
        left_panel,
        text="Tien xu ly ROI truoc Canny",
        font=("Arial", 14, "bold"),
        bg="#f3f4f6",
        anchor="w"
    ).pack(fill="x", pady=(0, 10))

    source_frame = tk.LabelFrame(left_panel, text="1. Anh dau vao ROI", bg="#f3f4f6", padx=8, pady=8)
    source_frame.pack(fill="x", pady=(0, 8))
    tk.Entry(source_frame, textvariable=roi_path_var).pack(fill="x", pady=(0, 6))
    tk.Button(source_frame, text="Import ROI", command=choose_roi).pack(fill="x")

    button_frame = tk.Frame(left_panel, bg="#f3f4f6")
    button_frame.pack(fill="x", pady=(0, 10))
    tk.Button(
        button_frame,
        text="Run",
        command=run_processing,
        height=2,
        bg="#2d89ef",
        fg="white",
        font=("Arial", 11, "bold")
    ).pack(side="left", fill="x", expand=True, padx=(0, 5))
    tk.Button(
        button_frame,
        text="Reset",
        command=reset_defaults,
        height=2,
        bg="#666666",
        fg="white",
        font=("Arial", 11, "bold")
    ).pack(side="right", fill="x", expand=True, padx=(5, 0))

    scroll_container = tk.Frame(left_panel, bg="#f3f4f6")
    scroll_container.pack(fill="both", expand=True)

    tool_canvas = tk.Canvas(scroll_container, bg="#f3f4f6", highlightthickness=0)
    scrollbar = tk.Scrollbar(scroll_container, orient="vertical", command=tool_canvas.yview)
    settings_panel = tk.Frame(tool_canvas, bg="#f3f4f6")

    settings_panel.bind(
        "<Configure>",
        lambda e: tool_canvas.configure(scrollregion=tool_canvas.bbox("all"))
    )
    tool_canvas.create_window((0, 0), window=settings_panel, anchor="nw", width=350)
    tool_canvas.configure(yscrollcommand=scrollbar.set)
    tool_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    contrast_frame = tk.LabelFrame(settings_panel, text="2. Tang tuong phan nhe", bg="#f3f4f6", padx=8, pady=8)
    contrast_frame.pack(fill="x", pady=(0, 8))
    tk.Checkbutton(contrast_frame, text="Dung CLAHE neu can", variable=vars_gui["use_clahe"], bg="#f3f4f6").pack(anchor="w")
    add_entry(contrast_frame, "CLAHE clipLimit", vars_gui["clipLimit"])
    add_entry(contrast_frame, "CLAHE tileGridSize", vars_gui["tileGridSize"])

    denoise_frame = tk.LabelFrame(settings_panel, text="3. Loc nhieu", bg="#f3f4f6", padx=8, pady=8)
    denoise_frame.pack(fill="x", pady=(0, 8))

    method_row = tk.Frame(denoise_frame, bg="#f3f4f6")
    method_row.pack(fill="x", pady=3)
    tk.Label(method_row, text="Denoise method", width=16, anchor="w", bg="#f3f4f6").pack(side="left")
    tk.OptionMenu(method_row, vars_gui["denoise_method"], "Gaussian", "Median", "Bilateral", "None").pack(side="right", fill="x", expand=True)

    add_entry(denoise_frame, "Gaussian kernel", vars_gui["gaussian_kernel"])
    add_entry(denoise_frame, "Gaussian sigma", vars_gui["gaussian_sigma"])
    add_entry(denoise_frame, "Median kernel", vars_gui["median_kernel"])
    add_entry(denoise_frame, "Bilateral d", vars_gui["bilateral_d"])
    add_entry(denoise_frame, "Bilateral sigmaColor", vars_gui["bilateral_sigmaColor"])
    add_entry(denoise_frame, "Bilateral sigmaSpace", vars_gui["bilateral_sigmaSpace"])

    canny_frame = tk.LabelFrame(settings_panel, text="4. Canny", bg="#f3f4f6", padx=8, pady=8)
    canny_frame.pack(fill="x", pady=(0, 8))
    add_entry(canny_frame, "Canny Low", vars_gui["canny_low"])
    add_entry(canny_frame, "Canny High", vars_gui["canny_high"])
    add_entry(canny_frame, "Aperture size", vars_gui["aperture_size"])
    tk.Checkbutton(canny_frame, text="Dung L2gradient", variable=vars_gui["L2gradient"], bg="#f3f4f6").pack(anchor="w")

    display_frame = tk.LabelFrame(settings_panel, text="5. Hien thi va luu ket qua", bg="#f3f4f6", padx=8, pady=8)
    display_frame.pack(fill="x", pady=(0, 8))
    add_entry(display_frame, "DPI luu hinh", vars_gui["dpi"])
    tk.Checkbutton(display_frame, text="Luu anh trung gian", variable=vars_gui["save_outputs"], bg="#f3f4f6").pack(anchor="w")

    note_frame = tk.LabelFrame(settings_panel, text="Ghi chu nhanh", bg="#f3f4f6", padx=8, pady=8)
    note_frame.pack(fill="x", pady=(0, 8))
    tk.Label(
        note_frame,
        text=(
            "Luong xu ly: ROI goc -> Anh xam -> Tang tuong phan -> Loc nhieu -> Canny.\n"
            "Neu anh da ro bien, co the tat CLAHE.\n"
            "Neu Canny qua nhieu bien vu, tang Gaussian kernel hoac tang Canny Low/High.\n"
            "Neu mat bien, giam Canny Low/High hoac giam loc nhieu."
        ),
        justify="left",
        wraplength=320,
        bg="#f3f4f6",
        fg="#555555"
    ).pack(fill="x")

    tk.Label(
        settings_panel,
        textvariable=status_var,
        justify="left",
        wraplength=330,
        anchor="w",
        bg="#f3f4f6",
        fg="#333333"
    ).pack(fill="x", pady=(8, 8))

    root.mainloop()


# =========================================================
# 7. CHAY CHUONG TRINH
# =========================================================
def main():
    roi = read_image(DEFAULT_ROI_PATH, grayscale=False)
    if roi is None:
        raise ValueError("Khong doc duoc ROI mac dinh: {}".format(DEFAULT_ROI_PATH))
    outputs = process_roi(roi, save_outputs=True)
    plt.show()
    print("Da luu hinh tai:", outputs["figure_path"])


if __name__ == "__main__":
    run_gui()
