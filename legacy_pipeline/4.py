# -*- coding: utf-8 -*-
"""
So sánh các phương án tiền xử lý ROI phục vụ Radial Signature.

Đầu vào:
- Ảnh đã được cắt ROI quanh một stator.

Các phương án so sánh:
1. ROI -> Gaussian Blur -> Canny
2. ROI -> Gaussian Blur -> Canny -> Morphology nhẹ
3. ROI -> Gaussian Blur -> Canny -> Contour
4. ROI -> Gaussian Blur -> Morphology trực tiếp trên ảnh xám

Mục đích:
- Đánh giá ảnh Canny trước khi đưa vào Radial Signature.
- Kiểm tra xem morphology nhẹ có giúp nối biên mà không làm mất tai/răng stator hay không.
- Bổ sung thêm một đầu ra contour để quan sát đường bao sau bước phát hiện biên.
- Cho thấy morphology trực tiếp trên ảnh xám không tạo ra ảnh biên phù hợp.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. ĐƯỜNG DẪN
# =========================================================
roi_path = r"C:\Users\congn\Desktop\vision_stator_project\data\input_images\3.jpg"
output_dir = r"C:\Users\congn\Desktop\vision_stator_project\data\test_results"
os.makedirs(output_dir, exist_ok=True)


# =========================================================
# 2. THÔNG SỐ XỬ LÝ
# =========================================================
GAUSSIAN_KERNEL = (5, 5)

CANNY_LOW = 70
CANNY_HIGH = 170

# Morphology nhẹ sau Canny
LIGHT_MORPH_KERNEL_SIZE = 3
LIGHT_MORPH_ITER = 1

# Morphology trực tiếp trên ảnh xám, chỉ để so sánh
GRAY_MORPH_KERNEL_SIZE = 5
GRAY_MORPH_ITER = 1

# Contour từ ảnh biên
CONTOUR_MIN_AREA = 80
CONTOUR_THICKNESS = 2


# =========================================================
# 3. HÀM ĐỌC ẢNH HỖ TRỢ ĐƯỜNG DẪN TIẾNG VIỆT
# =========================================================
def read_image(path, grayscale=False):
    data = np.fromfile(path, dtype=np.uint8)

    if data.size == 0:
        return None

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imdecode(data, flag)


# =========================================================
# 4. TIỀN XỬ LÝ ROI
# =========================================================
def prepare_gray_roi(roi):
    if roi is None:
        raise ValueError("ROI đầu vào không hợp lệ")

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = gray.astype(np.uint8)

    return gray


def light_morphology_after_canny(edges):
    """
    Morphology rất nhẹ sau Canny.
    Dùng MORPH_CLOSE với kernel CROSS 3x3 để nối các đoạn biên đứt nhỏ.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (LIGHT_MORPH_KERNEL_SIZE, LIGHT_MORPH_KERNEL_SIZE)
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=LIGHT_MORPH_ITER
    )

    return closed


def contour_from_edges(roi_gray, edge_image):
    """
    Tạo ảnh contour từ ảnh biên.
    Giữ contour ngoài cùng và lọc bớt nhiễu nhỏ theo diện tích.
    """
    contours, _ = cv2.findContours(
        edge_image.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) >= CONTOUR_MIN_AREA
    ]

    contour_image = np.zeros_like(roi_gray)

    if filtered_contours:
        cv2.drawContours(
            contour_image,
            filtered_contours,
            -1,
            255,
            CONTOUR_THICKNESS
        )

    return contour_image, filtered_contours


def morphology_direct_on_gray(gray_blur):
    """
    Morphology trực tiếp trên ảnh xám.
    Chỉ dùng để minh họa rằng đầu ra vẫn là ảnh mức xám,
    không phải ảnh biên hoặc mask nhị phân.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (GRAY_MORPH_KERNEL_SIZE, GRAY_MORPH_KERNEL_SIZE)
    )

    result = cv2.morphologyEx(
        gray_blur,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=GRAY_MORPH_ITER
    )

    return result


# =========================================================
# 5. CHƯƠNG TRÌNH CHÍNH
# =========================================================
roi = read_image(roi_path, grayscale=False)

if roi is None:
    raise ValueError(f"Không đọc được ROI: {roi_path}")

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

canny_light_morph = light_morphology_after_canny(canny)
contour_image, filtered_contours = contour_from_edges(roi_gray, canny_light_morph)
gray_direct_morph = morphology_direct_on_gray(roi_blur)


# =========================================================
# 6. LƯU ẢNH RIÊNG
# =========================================================
cv2.imwrite(os.path.join(output_dir, "roi_01_gray.png"), roi_gray)
cv2.imwrite(os.path.join(output_dir, "roi_02_gaussian_blur.png"), roi_blur)
cv2.imwrite(os.path.join(output_dir, "roi_03_canny.png"), canny)
cv2.imwrite(os.path.join(output_dir, "roi_04_canny_light_morph.png"), canny_light_morph)
cv2.imwrite(os.path.join(output_dir, "roi_05_contour.png"), contour_image)
cv2.imwrite(os.path.join(output_dir, "roi_06_gray_direct_morph.png"), gray_direct_morph)


# =========================================================
# 7. TẠO ẢNH GHÉP CHO BÁO CÁO
# =========================================================
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.ravel()

axes[0].imshow(roi_gray, cmap="gray")
axes[0].set_title("(a) ROI goc", fontsize=11)
axes[0].axis("off")

axes[1].imshow(roi_blur, cmap="gray")
axes[1].set_title("(b) Gaussian Blur", fontsize=11)
axes[1].axis("off")

axes[2].imshow(canny, cmap="gray")
axes[2].set_title("(c) Canny", fontsize=11)
axes[2].axis("off")

axes[3].imshow(canny_light_morph, cmap="gray")
axes[3].set_title("(d) Canny + Close nhe", fontsize=11)
axes[3].axis("off")

axes[4].imshow(contour_image, cmap="gray")
axes[4].set_title("(e) Contour tu Canny + Close", fontsize=11)
axes[4].axis("off")

axes[5].imshow(gray_direct_morph, cmap="gray")
axes[5].set_title("(f) Morphology truc tiep anh xam", fontsize=11)
axes[5].axis("off")

plt.tight_layout()

figure_path = os.path.join(
    output_dir,
    "Hinh_4_9_so_sanh_tien_xu_ly_ROI_RadialSignature.png"
)

plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("Da luu anh ghep tai:")
print(figure_path)
print("Cac anh trung gian da luu tai:")
print(output_dir)
print(f"So contour giu lai: {len(filtered_contours)}")
