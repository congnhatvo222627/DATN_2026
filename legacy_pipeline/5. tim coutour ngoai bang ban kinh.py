# -*- coding: utf-8 -*-
"""
GUI Radial Signature 360 tia từ tâm stator - bản chống nhiễu tốt hơn.

Chức năng:
- Chọn ảnh đầu vào.
- Gaussian Blur -> Canny.
- Tìm tâm bằng Hough Circle, nếu không tìm được thì dùng tâm ảnh.
- Từ tâm quét 360 tia, bước góc 1 độ.
- Mỗi góc lấy điểm Canny xa nhất, nhưng:
    + Giới hạn bán kính quét tối đa.
    + Chỉ nhận điểm biên nếu quanh nó có cụm pixel Canny đủ lớn.
    + Lọc các tia dài bất thường bằng local median.
- Vẽ 360 đường bán kính.
- Đường góc 0 độ màu đỏ.
- Các tia bị phát hiện là nhiễu được đánh dấu màu tím.
- Bảng bên phải hiển thị 360 giá trị raw và filtered.
"""

import os
import csv
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk


# =========================================================
# 1. THÔNG SỐ XỬ LÝ
# =========================================================

GAUSSIAN_KERNEL = (5, 5)

CANNY_LOW = 70
CANNY_HIGH = 170

HOUGH_DP = 1.2
HOUGH_PARAM1 = 120
HOUGH_PARAM2 = 30

ANGLE_STEP_DEG = 1

# Giới hạn bán kính quét.
# Nếu còn ăn ra nhiễu ngoài, giảm xuống 1.25 hoặc 1.20.
# Nếu bị mất tai/răng thật, tăng lên 1.40.
MAX_RADIUS_RATIO = 1.32

# Kiểm tra cụm pixel Canny xung quanh điểm đang xét.
# Tránh nhận 1 pixel nhiễu đơn lẻ.
EDGE_NEIGHBOR_WINDOW = 3
EDGE_NEIGHBOR_MIN_COUNT = 2

# Lọc gai bất thường trên radial signature.
LOCAL_MEDIAN_WINDOW = 9
SPIKE_THRESHOLD_PX = 35

# Hiển thị
DISPLAY_MAX_WIDTH = 850
DISPLAY_MAX_HEIGHT = 850

SHOW_REMOVED_OUTLIERS = True

SCRIPT3_OUTPUT_DIR = os.path.join("data", "test_results", "roi_canny_monitor")
DEFAULT_INPUT_IMAGE = os.path.join(SCRIPT3_OUTPUT_DIR, "05_canny.png")

INPUT_MODE_CANNY_READY = "canny_ready"
INPUT_MODE_RAW_ROI = "raw_roi"

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


# =========================================================
# 2. ĐỌC / GHI ẢNH HỖ TRỢ ĐƯỜNG DẪN TIẾNG VIỆT
# =========================================================

def read_image(path, grayscale=False):
    data = np.fromfile(path, dtype=np.uint8)

    if data.size == 0:
        return None

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imdecode(data, flag)


def write_image(path, image):
    ext = os.path.splitext(path)[1]
    success, encoded = cv2.imencode(ext, image)

    if not success:
        raise ValueError(f"Không ghi được ảnh: {path}")

    encoded.tofile(path)


def prepare_gray_roi(roi):
    if roi is None:
        raise ValueError("Ảnh đầu vào không hợp lệ")

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = gray.astype(np.uint8)

    return gray


def apply_clahe(gray):
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    return clahe.apply(gray)


def normalize_edge_image(edge_like):
    gray = prepare_gray_roi(edge_like)
    _, edge = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return edge


def get_default_input_path():
    return os.path.abspath(DEFAULT_INPUT_IMAGE)


# =========================================================
# 3. TÌM TÂM BẰNG HOUGH CIRCLE
# =========================================================

def find_center_by_hough_or_image_center(gray_blur):
    h, w = gray_blur.shape[:2]

    min_radius = int(min(w, h) * 0.20)
    max_radius = int(min(w, h) * 0.48)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=min(w, h) // 2,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        image_center = np.array([w / 2.0, h / 2.0])

        best_circle = min(
            circles,
            key=lambda c: np.linalg.norm(np.array([c[0], c[1]]) - image_center)
        )

        cx, cy, r = best_circle
        return (int(cx), int(cy)), int(r), True

    cx = w // 2
    cy = h // 2
    r = int(min(w, h) * 0.35)

    return (cx, cy), r, False


# =========================================================
# 4. KIỂM TRA ĐIỂM CANNY CÓ PHẢI NHIỄU ĐƠN LẺ KHÔNG
# =========================================================

def has_edge_neighbor(edge_image, x, y, window=3, min_count=2):
    h, w = edge_image.shape[:2]
    half = window // 2

    x1 = max(0, x - half)
    x2 = min(w, x + half + 1)
    y1 = max(0, y - half)
    y2 = min(h, y + half + 1)

    patch = edge_image[y1:y2, x1:x2]
    count = np.count_nonzero(patch)

    return count >= min_count


# =========================================================
# 5. RADIAL SIGNATURE TỪ CANNY CÓ GIỚI HẠN BÁN KÍNH
# =========================================================

def radial_signature_from_canny_robust(edge_image, center, estimated_radius,
                                       hough_found=True,
                                       angle_step_deg=1):
    """
    Với mỗi góc theta:
    - Quét từ tâm ra ngoài.
    - Chỉ quét tới r_max hợp lý, không quét ra tận mép ảnh.
    - Chỉ nhận điểm Canny nếu quanh nó có cụm pixel lân cận.
    - Điểm cuối cùng hợp lệ là điểm biên xa nhất.

    Không dùng mask vành khăn.
    Chỉ dùng giới hạn r_max để tránh bắt nhiễu quá xa bên ngoài.
    """
    h, w = edge_image.shape[:2]
    cx, cy = center

    angles = np.arange(0, 360, angle_step_deg, dtype=np.float32)

    signature_raw = np.full(len(angles), np.nan, dtype=np.float32)
    points_raw = [None] * len(angles)

    # Bán kính tối đa không vượt khỏi ảnh.
    r_to_border = int(max(
        np.hypot(cx, cy),
        np.hypot(w - cx, cy),
        np.hypot(cx, h - cy),
        np.hypot(w - cx, h - cy)
    ))

    if hough_found and estimated_radius > 0:
        r_max = int(min(r_to_border, estimated_radius * MAX_RADIUS_RATIO))
    else:
        r_max = r_to_border

    for i, deg in enumerate(angles):
        theta = np.deg2rad(deg)

        best_r = np.nan
        best_point = None

        for r in range(0, r_max + 1):
            x = int(round(cx + r * np.cos(theta)))
            y = int(round(cy + r * np.sin(theta)))

            if x < 0 or x >= w or y < 0 or y >= h:
                break

            if edge_image[y, x] > 0:
                if has_edge_neighbor(
                    edge_image,
                    x,
                    y,
                    window=EDGE_NEIGHBOR_WINDOW,
                    min_count=EDGE_NEIGHBOR_MIN_COUNT
                ):
                    best_r = r
                    best_point = (x, y)

        if best_point is not None:
            signature_raw[i] = best_r
            points_raw[i] = best_point

    valid = ~np.isnan(signature_raw)
    valid_count = int(np.sum(valid))

    # Nội suy các góc không có điểm Canny.
    signature_interp = circular_interpolate_missing(signature_raw)

    # Rebuild điểm theo signature đã nội suy.
    points_interp = rebuild_points_from_signature(
        center=center,
        angles=angles,
        signature=signature_interp,
        image_shape=edge_image.shape
    )

    return angles, signature_raw, signature_interp, points_raw, points_interp, valid_count, r_max


# =========================================================
# 6. NỘI SUY DẠNG VÒNG TRÒN
# =========================================================

def circular_interpolate_missing(signal):
    """
    Nội suy các giá trị NaN trong tín hiệu 360 độ.
    Có xử lý vòng tròn 359 -> 0.
    """
    r = signal.copy().astype(np.float32)
    n = len(r)

    valid = ~np.isnan(r)

    if np.sum(valid) == 0:
        return np.zeros(n, dtype=np.float32)

    if np.sum(valid) == 1:
        return np.full(n, r[valid][0], dtype=np.float32)

    idx = np.arange(n)
    valid_idx = idx[valid]
    valid_val = r[valid]

    # Nhân bản dữ liệu sang trái/phải để nội suy vòng tròn.
    extended_idx = np.r_[valid_idx - n, valid_idx, valid_idx + n]
    extended_val = np.r_[valid_val, valid_val, valid_val]

    result = np.interp(idx, extended_idx, extended_val).astype(np.float32)
    return result


# =========================================================
# 7. LỌC OUTLIER BẰNG LOCAL MEDIAN
# =========================================================

def remove_local_spikes_circular(signature, window_size=9, spike_threshold=35):
    """
    Loại các gai dài bất thường trên Radial Signature.

    Ý tưởng:
    - R(theta) thường thay đổi tương đối liên tục theo góc.
    - Nếu một điểm dài vọt lên quá nhiều so với median lân cận,
      coi là outlier do nhiễu ngoài.
    """
    r = signature.copy().astype(np.float32)
    n = len(r)

    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    padded = np.r_[r[-pad:], r, r[:pad]]

    local_median = np.zeros_like(r)

    for i in range(n):
        local_median[i] = np.median(padded[i:i + window_size])

    # Chỉ lọc các điểm dài bất thường.
    outlier_mask = r > (local_median + spike_threshold)

    # Thêm kiểm tra global để tránh bỏ nhầm vùng tai thật lớn nhưng thay đổi từ từ.
    q1 = np.percentile(r, 25)
    q3 = np.percentile(r, 75)
    iqr = q3 - q1
    global_upper = q3 + 1.5 * iqr

    outlier_mask = outlier_mask & (r > global_upper * 0.95)

    filtered = r.copy()

    invalid = outlier_mask
    valid = ~invalid

    if np.sum(valid) >= 2:
        idx = np.arange(n)
        valid_idx = idx[valid]
        valid_val = filtered[valid]

        extended_idx = np.r_[valid_idx - n, valid_idx, valid_idx + n]
        extended_val = np.r_[valid_val, valid_val, valid_val]

        filtered[invalid] = np.interp(idx[invalid], extended_idx, extended_val)

    return filtered.astype(np.float32), outlier_mask, local_median, global_upper


# =========================================================
# 8. DỰNG LẠI ĐIỂM TỪ SIGNATURE
# =========================================================

def rebuild_points_from_signature(center, angles, signature, image_shape):
    h, w = image_shape[:2]
    cx, cy = center

    points = []

    for deg, r in zip(angles, signature):
        theta = np.deg2rad(deg)

        x = int(round(cx + r * np.cos(theta)))
        y = int(round(cy + r * np.sin(theta)))

        if 0 <= x < w and 0 <= y < h:
            points.append((x, y))
        else:
            points.append(None)

    return points


def get_overlay_style(image_shape):
    h, w = image_shape[:2]
    min_dim = float(min(h, w))

    # ROI nho se bi phong to de vua khung, nen can giam size cua chu/marker
    # truoc khi hien thi de giu cam quan gan giong ROI lon.
    scale = float(np.clip(min_dim / 640.0, 0.24, 1.0))

    style = {
        "raw_line_thickness": 1,
        "raw_point_radius": max(1, int(round(2 * scale))),
        "main_line_thickness": 1,
        "zero_line_thickness": max(1, int(round(2 * scale))),
        "point_radius": max(1, int(round(1 * scale))),
        "center_inner_radius": max(2, int(round(5 * scale))),
        "center_outer_radius": max(4, int(round(10 * scale))),
        "center_ring_thickness": max(1, int(round(2 * scale))),
        "center_font_scale": max(0.22, 0.55 * scale),
        "zero_font_scale": max(0.24, 0.65 * scale),
        "text_thickness": max(1, int(round(2 * scale))),
        "center_offset_x": max(6, int(round(12 * scale))),
        "center_offset_y": max(6, int(round(12 * scale))),
        "zero_offset_x": max(4, int(round(8 * scale))),
        "zero_offset_y": max(4, int(round(8 * scale))),
        "line_type": cv2.LINE_AA,
    }
    return style


# =========================================================
# 9. VẼ 360 ĐƯỜNG BÁN KÍNH
# =========================================================

def draw_radial_lines_on_roi(roi_gray, center,
                             angles,
                             points_filtered,
                             points_raw=None,
                             outlier_mask=None):
    """
    Vẽ kết quả tốt nhất:
    - Các tia sau lọc: xanh lá.
    - Góc 0 độ: đỏ.
    - Các tia raw bị loại: tím.
    """
    if len(roi_gray.shape) == 2:
        overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    else:
        overlay = roi_gray.copy()

    cx, cy = center
    style = get_overlay_style(overlay.shape)

    # Vẽ các tia raw bị loại bằng màu tím để thấy nhiễu ban đầu.
    if SHOW_REMOVED_OUTLIERS and points_raw is not None and outlier_mask is not None:
        for i, is_outlier in enumerate(outlier_mask):
            if not is_outlier:
                continue

            p_raw = points_raw[i]
            if p_raw is None:
                continue

            x_raw, y_raw = p_raw
            cv2.line(
                overlay,
                (cx, cy),
                (x_raw, y_raw),
                (255, 0, 255),
                style["raw_line_thickness"],
                lineType=style["line_type"]
            )
            cv2.circle(
                overlay,
                (x_raw, y_raw),
                style["raw_point_radius"],
                (255, 0, 255),
                -1,
                lineType=style["line_type"]
            )

    # Vẽ 360 tia sau lọc.
    for i, point in enumerate(points_filtered):
        if point is None:
            continue

        x, y = point
        deg = int(angles[i])

        if deg == 0:
            line_color = (0, 0, 255)      # đỏ cho 0 độ
            point_color = (0, 0, 255)
            thickness = style["zero_line_thickness"]
        else:
            line_color = (0, 255, 0)      # xanh lá
            point_color = (0, 255, 255)   # vàng
            thickness = style["main_line_thickness"]

        cv2.line(
            overlay,
            (cx, cy),
            (x, y),
            line_color,
            thickness,
            lineType=style["line_type"]
        )

        cv2.circle(
            overlay,
            (x, y),
            style["point_radius"],
            point_color,
            -1,
            lineType=style["line_type"]
        )

    # Vẽ tâm
    cv2.circle(
        overlay,
        (cx, cy),
        style["center_inner_radius"],
        (0, 0, 255),
        -1,
        lineType=style["line_type"]
    )
    cv2.circle(
        overlay,
        (cx, cy),
        style["center_outer_radius"],
        (0, 0, 255),
        style["center_ring_thickness"],
        lineType=style["line_type"]
    )

    cv2.putText(
        overlay,
        "Center",
        (cx + style["center_offset_x"], cy - style["center_offset_y"]),
        cv2.FONT_HERSHEY_SIMPLEX,
        style["center_font_scale"],
        (0, 0, 255),
        style["text_thickness"],
        lineType=style["line_type"]
    )

    # Ghi chú góc 0 độ
    if len(points_filtered) > 0 and points_filtered[0] is not None:
        x0, y0 = points_filtered[0]
        cv2.putText(
            overlay,
            "0 deg",
            (x0 + style["zero_offset_x"], y0 - style["zero_offset_y"]),
            cv2.FONT_HERSHEY_SIMPLEX,
            style["zero_font_scale"],
            (0, 0, 255),
            style["text_thickness"],
            lineType=style["line_type"]
        )

    return overlay


def draw_hough_circle_overlay(roi_gray, center, radius, hough_found):
    if len(roi_gray.shape) == 2:
        overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    else:
        overlay = roi_gray.copy()

    cx, cy = center
    style = get_overlay_style(overlay.shape)

    cv2.circle(
        overlay,
        (cx, cy),
        max(2, style["center_inner_radius"] - 1),
        (0, 0, 255),
        -1,
        lineType=style["line_type"]
    )

    if radius > 0:
        color = (0, 255, 0) if hough_found else (0, 165, 255)
        cv2.circle(
            overlay,
            (cx, cy),
            int(radius),
            color,
            style["center_ring_thickness"],
            lineType=style["line_type"]
        )

    label = "HoughCircle" if hough_found else "Image center fallback"
    cv2.putText(
        overlay,
        label,
        (max(10, cx - 80), max(20, cy - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        style["center_font_scale"],
        (0, 255, 255),
        style["text_thickness"],
        lineType=style["line_type"]
    )

    return overlay


def compose_output_panel(hough_overlay, canny_image, radial_overlay):
    def prepare_panel(image_bgr, title):
        panel = resize_for_display(image_bgr, max_width=420, max_height=260)
        panel = panel.copy()
        cv2.putText(
            panel,
            title,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )
        return panel

    canny_bgr = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR)

    panels = [
        prepare_panel(hough_overlay, "HoughCircle"),
        prepare_panel(canny_bgr, "Canny"),
        prepare_panel(radial_overlay, "Radial Signature"),
    ]

    target_height = max(panel.shape[0] for panel in panels)
    normalized = []

    for panel in panels:
        h, w = panel.shape[:2]
        if h != target_height:
            new_w = max(1, int(round(w * target_height / float(h))))
            panel = cv2.resize(panel, (new_w, target_height), interpolation=cv2.INTER_AREA)
        normalized.append(panel)

    separator = np.full((target_height, 12, 3), 30, dtype=np.uint8)
    combined = normalized[0]

    for panel in normalized[1:]:
        combined = np.hstack([combined, separator.copy(), panel])

    return combined


# =========================================================
# 10. CHUYỂN ẢNH OPENCV SANG TKINTER
# =========================================================

def resize_for_display(image_bgr, max_width=850, max_height=850, allow_upscale=False):
    h, w = image_bgr.shape[:2]

    scale = min(max_width / w, max_height / h)

    if not allow_upscale:
        scale = min(scale, 1.0)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA

    resized = cv2.resize(
        image_bgr,
        (new_w, new_h),
        interpolation=interpolation
    )

    return resized


def resize_for_display_with_meta(image_bgr, max_width=850, max_height=850, allow_upscale=False):
    h, w = image_bgr.shape[:2]

    scale = min(max_width / w, max_height / h)

    if not allow_upscale:
        scale = min(scale, 1.0)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interpolation)

    return resized, scale


def scale_point(point, scale):
    if point is None:
        return None

    x, y = point
    return (int(round(x * scale)), int(round(y * scale)))


def scale_points(points, scale):
    if points is None:
        return None

    return [scale_point(point, scale) for point in points]


def render_result_overlay_for_display(
    roi_gray,
    center,
    angles,
    points_filtered,
    points_raw,
    outlier_mask,
    target_width,
    target_height
):
    if len(roi_gray.shape) == 2:
        base_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    else:
        base_bgr = roi_gray.copy()

    resized_bgr, scale = resize_for_display_with_meta(
        base_bgr,
        target_width,
        target_height,
        allow_upscale=True
    )

    scaled_center = scale_point(center, scale)
    scaled_points_filtered = scale_points(points_filtered, scale)
    scaled_points_raw = scale_points(points_raw, scale)

    resized_gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    return draw_radial_lines_on_roi(
        roi_gray=resized_gray,
        center=scaled_center,
        angles=angles,
        points_filtered=scaled_points_filtered,
        points_raw=scaled_points_raw,
        outlier_mask=outlier_mask
    )


def cv_to_tk_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)
    return image_tk


# =========================================================
# 11. GUI APP
# =========================================================

class RadialSignatureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Radial Signature 360 điểm - chống nhiễu")
        self.root.geometry("1450x880")

        self.image_path = get_default_input_path()
        self.input_mode = INPUT_MODE_CANNY_READY

        self.roi_gray = None
        self.canny = None
        self.hough_overlay = None
        self.result_overlay = None
        self.output_panel = None

        self.angles = None
        self.signature_raw = None
        self.signature_interp = None
        self.signature_filtered = None
        self.points_raw = None
        self.points_filtered = None
        self.outlier_mask = None

        self.display_image_tk = None
        self.display_source_image = None
        self.display_render_payload = None
        self.hough_image_tk = None
        self.canny_image_tk = None
        self.result_image_tk = None
        self.popup_windows = []

        self.build_gui()
        self.load_default_input_image()

    def build_gui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        btn_choose = tk.Button(
            top_frame,
            text="Chọn ảnh đầu vào",
            font=("Arial", 11),
            command=self.choose_image
        )
        btn_choose.pack(side=tk.LEFT, padx=5)

        btn_process = tk.Button(
            top_frame,
            text="Chạy Radial Signature",
            font=("Arial", 11),
            command=self.process_image
        )
        btn_process.pack(side=tk.LEFT, padx=5)

        btn_save_image = tk.Button(
            top_frame,
            text="Lưu ảnh kết quả",
            font=("Arial", 11),
            command=self.save_result_image
        )
        btn_save_image.pack(side=tk.LEFT, padx=5)

        btn_save_csv = tk.Button(
            top_frame,
            text="Lưu CSV",
            font=("Arial", 11),
            command=self.save_csv
        )
        btn_save_csv.pack(side=tk.LEFT, padx=5)

        self.path_label = tk.Label(
            top_frame,
            text="Chưa chọn ảnh",
            font=("Arial", 10),
            anchor="w"
        )
        self.path_label.pack(side=tk.LEFT, padx=15, fill=tk.X, expand=True)

        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        image_frame = tk.LabelFrame(
            main_frame,
            text="Ảnh kết quả: tâm + 360 tia sau lọc nhiễu",
            font=("Arial", 11)
        )
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.image_label = tk.Label(image_frame, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_label.bind("<Configure>", self.on_main_image_resize)

        result_frame = tk.LabelFrame(
            main_frame,
            text="Bảng 360 giá trị Radial Signature",
            font=("Arial", 11)
        )
        result_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        result_button_frame = tk.Frame(result_frame)
        result_button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 0))

        btn_show_hough = tk.Button(
            result_button_frame,
            text="Xem HoughCircle",
            font=("Arial", 10),
            command=self.show_hough_window
        )
        btn_show_hough.pack(side=tk.LEFT, padx=3)

        btn_show_canny = tk.Button(
            result_button_frame,
            text="Xem Canny",
            font=("Arial", 10),
            command=self.show_canny_window
        )
        btn_show_canny.pack(side=tk.LEFT, padx=3)

        columns = ("angle", "r_raw", "r_filtered", "x", "y", "status")

        self.tree = ttk.Treeview(
            result_frame,
            columns=columns,
            show="headings",
            height=36
        )

        self.tree.heading("angle", text="Góc")
        self.tree.heading("r_raw", text="R raw")
        self.tree.heading("r_filtered", text="R lọc")
        self.tree.heading("x", text="X")
        self.tree.heading("y", text="Y")
        self.tree.heading("status", text="Trạng thái")

        self.tree.column("angle", width=60, anchor="center")
        self.tree.column("r_raw", width=80, anchor="center")
        self.tree.column("r_filtered", width=80, anchor="center")
        self.tree.column("x", width=70, anchor="center")
        self.tree.column("y", width=70, anchor="center")
        self.tree.column("status", width=90, anchor="center")

        self.tree.tag_configure("outlier", foreground="purple")
        self.tree.tag_configure("angle0", foreground="red")

        scrollbar = ttk.Scrollbar(
            result_frame,
            orient=tk.VERTICAL,
            command=self.tree.yview
        )

        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0), pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)

        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.info_label = tk.Label(
            bottom_frame,
            text="Thông tin xử lý sẽ hiển thị tại đây.",
            font=("Arial", 10),
            anchor="w",
            justify=tk.LEFT
        )
        self.info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def get_main_image_target_size(self):
        # Lay kich thuoc thuc te cua khung den de anh duoc fit vua khung.
        width = self.image_label.winfo_width()
        height = self.image_label.winfo_height()

        if width <= 1:
            width = DISPLAY_MAX_WIDTH
        if height <= 1:
            height = DISPLAY_MAX_HEIGHT

        return width, height

    def set_main_display_image(self, image_bgr):
        self.display_render_payload = None
        self.display_source_image = image_bgr
        self.refresh_main_display_image()

    def set_main_display_result(self, roi_gray, center, angles, points_filtered, points_raw, outlier_mask):
        self.display_source_image = None
        self.display_render_payload = {
            "roi_gray": roi_gray,
            "center": center,
            "angles": angles,
            "points_filtered": points_filtered,
            "points_raw": points_raw,
            "outlier_mask": outlier_mask,
        }
        self.refresh_main_display_image()

    def refresh_main_display_image(self):
        if self.display_source_image is None and self.display_render_payload is None:
            return

        target_width, target_height = self.get_main_image_target_size()

        if self.display_render_payload is not None:
            payload = self.display_render_payload
            display_bgr = render_result_overlay_for_display(
                roi_gray=payload["roi_gray"],
                center=payload["center"],
                angles=payload["angles"],
                points_filtered=payload["points_filtered"],
                points_raw=payload["points_raw"],
                outlier_mask=payload["outlier_mask"],
                target_width=target_width,
                target_height=target_height
            )
        else:
            display_bgr = resize_for_display(
                self.display_source_image,
                target_width,
                target_height,
                allow_upscale=True
            )

        self.result_image_tk = cv_to_tk_image(display_bgr)
        self.image_label.config(image=self.result_image_tk)

    def on_main_image_resize(self, event):
        if self.display_source_image is not None:
            self.refresh_main_display_image()

    def load_default_input_image(self):
        default_path = self.image_path

        if not default_path or not os.path.isfile(default_path):
            self.path_label.config(
                text=f"Chua thay anh dau ra tu buoc 3: {default_path}"
            )
            self.info_label.config(
                text="Hay chay file '3.dau vao thuat toan xoay.py' de tao 05_canny.png, hoac chon anh khac."
            )
            return

        self.load_image_from_path(
            default_path,
            input_mode=INPUT_MODE_CANNY_READY,
            success_message=(
                "Da nap anh Canny tu file 3. Anh nay se duoc dung truc tiep cho "
                "Radial Signature, khong xu ly lai CLAHE/Gaussian/Canny."
            )
        )

    def load_image_from_path(self, file_path, input_mode, success_message):
        self.image_path = file_path
        self.input_mode = input_mode
        self.path_label.config(text=file_path)

        img = read_image(file_path, grayscale=False)

        if img is None:
            raise ValueError(f"Khong doc duoc anh: {file_path}")

        gray = prepare_gray_roi(img)

        preview_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.display_image_tk = cv_to_tk_image(preview_bgr)
        self.set_main_display_image(preview_bgr)

        self.clear_table()
        self.info_label.config(text=success_message)

class PatchedRadialSignatureGUI(RadialSignatureGUI):
    def choose_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh ROI stator",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.image_path = file_path
        self.path_label.config(text=file_path)

        img = read_image(file_path, grayscale=False)

        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh.")
            return

        gray = prepare_gray_roi(img)

        preview_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.display_image_tk = cv_to_tk_image(preview_bgr)
        self.set_main_display_image(preview_bgr)

        self.clear_table()
        self.info_label.config(
            text="Đã chọn ảnh. Nhấn 'Chạy Radial Signature' để xử lý."
        )

    def process_image(self):
        if self.image_path is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh đầu vào trước.")
            return

        roi = read_image(self.image_path, grayscale=False)

        if roi is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh đầu vào.")
            return

        roi_gray = prepare_gray_roi(roi)

        roi_blur = cv2.GaussianBlur(
            roi_gray,
            GAUSSIAN_KERNEL,
            0
        )

        canny = cv2.Canny(
            roi_blur,
            CANNY_LOW,
            CANNY_HIGH
        )

        center, estimated_radius, hough_found = find_center_by_hough_or_image_center(
            roi_blur
        )

        angles, signature_raw, signature_interp, points_raw, points_interp, valid_count, r_max = radial_signature_from_canny_robust(
            edge_image=canny,
            center=center,
            estimated_radius=estimated_radius,
            hough_found=hough_found,
            angle_step_deg=ANGLE_STEP_DEG
        )

        signature_filtered, outlier_mask, local_median, global_upper = remove_local_spikes_circular(
            signature_interp,
            window_size=LOCAL_MEDIAN_WINDOW,
            spike_threshold=SPIKE_THRESHOLD_PX
        )

        points_filtered = rebuild_points_from_signature(
            center=center,
            angles=angles,
            signature=signature_filtered,
            image_shape=canny.shape
        )

        hough_overlay = draw_hough_circle_overlay(
            roi_gray=roi_gray,
            center=center,
            radius=estimated_radius,
            hough_found=hough_found
        )

        result_overlay = draw_radial_lines_on_roi(
            roi_gray=roi_gray,
            center=center,
            angles=angles,
            points_filtered=points_filtered,
            points_raw=points_raw,
            outlier_mask=outlier_mask
        )

        self.roi_gray = roi_gray
        self.canny = canny
        self.hough_overlay = hough_overlay
        self.result_overlay = result_overlay
        self.output_panel = result_overlay

        self.angles = angles
        self.signature_raw = signature_raw
        self.signature_interp = signature_interp
        self.signature_filtered = signature_filtered
        self.points_raw = points_raw
        self.points_filtered = points_filtered
        self.outlier_mask = outlier_mask

        self.set_main_display_result(
            roi_gray=roi_gray,
            center=center,
            angles=angles,
            points_filtered=points_filtered,
            points_raw=points_raw,
            outlier_mask=outlier_mask
        )

        self.update_table()

        info = (
            f"Tâm: {center} | "
            f"Hough found: {hough_found} | "
            f"R Hough: {estimated_radius} | "
            f"r_max quét: {r_max} | "
            f"Số góc có điểm Canny trực tiếp: {valid_count}/{len(angles)} | "
            f"Số outlier đã lọc: {int(np.sum(outlier_mask))} | "
            f"Ngưỡng global upper ≈ {global_upper:.1f}"
        )

        self.info_label.config(text=info)

    def update_table(self):
        self.clear_table()

        if self.angles is None:
            return

        for i in range(len(self.angles)):
            deg = int(self.angles[i])

            r_raw = self.signature_raw[i]
            r_filtered = self.signature_filtered[i]

            point = self.points_filtered[i]

            if np.isnan(r_raw):
                r_raw_str = "-"
            else:
                r_raw_str = f"{r_raw:.2f}"

            if point is None:
                x_str = "-"
                y_str = "-"
            else:
                x_str = str(point[0])
                y_str = str(point[1])

            if self.outlier_mask[i]:
                status = "Outlier"
                tag = "outlier"
            elif deg == 0:
                status = "0 deg"
                tag = "angle0"
            else:
                status = "OK"
                tag = ""

            self.tree.insert(
                "",
                tk.END,
                values=(
                    f"{deg}°",
                    r_raw_str,
                    f"{r_filtered:.2f}",
                    x_str,
                    y_str,
                    status
                ),
                tags=(tag,)
            )

    def clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def show_image_popup(self, title, image_bgr):
        if image_bgr is None:
            messagebox.showwarning("Canh bao", f"Chua co anh {title.lower()} de hien thi.")
            return

        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("920x760")

        frame = tk.Frame(popup, bg="black")
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        display_bgr = resize_for_display(image_bgr, 880, 700)
        popup_tk = cv_to_tk_image(display_bgr)

        label = tk.Label(frame, image=popup_tk, bg="black")
        label.image = popup_tk
        label.pack(fill=tk.BOTH, expand=True)

        self.popup_windows.append(popup)
        popup.protocol("WM_DELETE_WINDOW", lambda win=popup: self.close_popup(win))

    def close_popup(self, popup):
        if popup in self.popup_windows:
            self.popup_windows.remove(popup)
        popup.destroy()

    def show_hough_window(self):
        self.show_image_popup("HoughCircle", self.hough_overlay)

    def show_canny_window(self):
        if self.canny is None:
            self.show_image_popup("Canny", None)
            return

        canny_bgr = cv2.cvtColor(self.canny, cv2.COLOR_GRAY2BGR)
        self.show_image_popup("Canny", canny_bgr)

    def save_result_image(self):
        if self.output_panel is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh kết quả để lưu.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả",
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("JPG image", "*.jpg"),
                ("BMP image", "*.bmp"),
                ("All files", "*.*")
            ]
        )

        if not save_path:
            return

        try:
            write_image(save_path, self.output_panel)
            messagebox.showinfo("Thành công", f"Đã lưu ảnh:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def save_csv(self):
        if self.angles is None or self.signature_filtered is None:
            messagebox.showwarning("Cảnh báo", "Chưa có dữ liệu để lưu.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Lưu CSV kết quả",
            defaultextension=".csv",
            filetypes=[
                ("CSV file", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not save_path:
            return

        try:
            with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "angle_deg",
                    "radius_raw",
                    "radius_filtered",
                    "x_filtered",
                    "y_filtered",
                    "is_outlier"
                ])

                for i in range(len(self.angles)):
                    deg = int(self.angles[i])

                    r_raw = self.signature_raw[i]
                    r_filtered = self.signature_filtered[i]
                    point = self.points_filtered[i]

                    if point is None:
                        x, y = "", ""
                    else:
                        x, y = point

                    writer.writerow([
                        deg,
                        "" if np.isnan(r_raw) else f"{r_raw:.3f}",
                        f"{r_filtered:.3f}",
                        x,
                        y,
                        int(self.outlier_mask[i])
                    ])

            messagebox.showinfo("Thành công", f"Đã lưu CSV:\n{save_path}")

        except Exception as e:
            messagebox.showerror("Lỗi", str(e))


# =========================================================
# 12. CHẠY CHƯƠNG TRÌNH
# =========================================================

    def choose_image(self):
        file_path = filedialog.askopenfilename(
            title="Chon anh ROI stator",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            self.load_image_from_path(
                file_path,
                input_mode=INPUT_MODE_RAW_ROI,
                success_message=(
                    "Da chon anh ROI goc. Khi chay, anh se duoc xu ly theo chuoi: "
                    "Grayscale -> CLAHE -> Gaussian Blur -> Canny -> Radial Signature."
                )
            )
        except Exception as exc:
            messagebox.showerror("Loi", str(exc))

    def process_image(self):
        if self.image_path is None:
            messagebox.showwarning("Canh bao", "Vui long chon anh dau vao truoc.")
            return

        if not os.path.isfile(self.image_path):
            messagebox.showwarning(
                "Canh bao",
                "Khong tim thay anh dau vao. Hay chay file '3.dau vao thuat toan xoay.py' de tao 05_canny.png, hoac chon anh khac."
            )
            return

        roi = read_image(self.image_path, grayscale=False)

        if roi is None:
            messagebox.showerror("Loi", f"Khong doc duoc anh dau vao: {self.image_path}")
            return

        roi_gray = prepare_gray_roi(roi)
        roi_display_gray = roi_gray

        if self.input_mode == INPUT_MODE_CANNY_READY:
            edge_image = normalize_edge_image(roi_gray)
            image_for_hough = edge_image
        elif self.input_mode == INPUT_MODE_RAW_ROI:
            roi_clahe = apply_clahe(roi_gray)
            roi_blur = cv2.GaussianBlur(
                roi_clahe,
                GAUSSIAN_KERNEL,
                0
            )
            edge_image = cv2.Canny(
                roi_blur,
                CANNY_LOW,
                CANNY_HIGH
            )
            image_for_hough = roi_blur
        else:
            messagebox.showerror("Loi", f"Input mode khong hop le: {self.input_mode}")
            return

        center, estimated_radius, hough_found = find_center_by_hough_or_image_center(
            image_for_hough
        )

        angles, signature_raw, signature_interp, points_raw, points_interp, valid_count, r_max = radial_signature_from_canny_robust(
            edge_image=edge_image,
            center=center,
            estimated_radius=estimated_radius,
            hough_found=hough_found,
            angle_step_deg=ANGLE_STEP_DEG
        )

        signature_filtered, outlier_mask, local_median, global_upper = remove_local_spikes_circular(
            signature_interp,
            window_size=LOCAL_MEDIAN_WINDOW,
            spike_threshold=SPIKE_THRESHOLD_PX
        )

        points_filtered = rebuild_points_from_signature(
            center=center,
            angles=angles,
            signature=signature_filtered,
            image_shape=edge_image.shape
        )

        hough_overlay = draw_hough_circle_overlay(
            roi_gray=roi_display_gray,
            center=center,
            radius=estimated_radius,
            hough_found=hough_found
        )

        result_overlay = draw_radial_lines_on_roi(
            roi_gray=roi_display_gray,
            center=center,
            angles=angles,
            points_filtered=points_filtered,
            points_raw=points_raw,
            outlier_mask=outlier_mask
        )

        self.roi_gray = roi_display_gray
        self.canny = edge_image
        self.hough_overlay = hough_overlay
        self.result_overlay = result_overlay
        self.output_panel = result_overlay

        self.angles = angles
        self.signature_raw = signature_raw
        self.signature_interp = signature_interp
        self.signature_filtered = signature_filtered
        self.points_raw = points_raw
        self.points_filtered = points_filtered
        self.outlier_mask = outlier_mask

        self.set_main_display_result(
            roi_gray=roi_display_gray,
            center=center,
            angles=angles,
            points_filtered=points_filtered,
            points_raw=points_raw,
            outlier_mask=outlier_mask
        )

        self.update_table()

        info = (
            f"Mode: {self.input_mode} | "
            f"Anh vao: {self.image_path} | "
            f"Tam: {center} | "
            f"Hough found: {hough_found} | "
            f"R Hough: {estimated_radius} | "
            f"r_max quet: {r_max} | "
            f"So goc co diem Canny truc tiep: {valid_count}/{len(angles)} | "
            f"So outlier da loc: {int(np.sum(outlier_mask))} | "
            f"Nguong global upper ~= {global_upper:.1f}"
        )

        self.info_label.config(text=info)

if __name__ == "__main__":
    root = tk.Tk()
    app = PatchedRadialSignatureGUI(root)
    root.mainloop()
