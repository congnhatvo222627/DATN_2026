# -*- coding: utf-8 -*-
"""
PIPELINE STATOR - TOAN BO CODE 7 BUOC GOP CHUNG (file tham khao).

Day la ban gop noi dung 7 file trong thu muc pipeline/ theo thu tu xu ly.
Moi buoc nam trong mot block "===== BUOC x =====" rieng biet.

LUU Y: cac file goc co the dung duong dan tuyet doi toi
  C:\\Users\\congn\\Desktop\\vision_stator_project\\data\\...
Khi chay can sua lai duong dan cho phu hop voi datn-copy/data/.
De chay tung buoc rieng le, dung truc tiep file .py tuong ung trong pipeline/.
"""

# ==============================================================================
# ===== BUOC 1.1 =====
# File goc: 1.1 tien xử lý trước houghCircle.py
# Vai tro : Tiền xử lý ảnh trước Hough: xám → CLAHE (tăng tương phản cục bộ) → Gaussian Blur.
# Dau ra  : Hình so sánh 3 bước (Hinh_4_7_*).
# ==============================================================================

import cv2
import matplotlib.pyplot as plt
import os

# =========================================================
# 1. ĐƯỜNG DẪN ẢNH ĐẦU VÀO
# =========================================================
image_path = r"C:\Users\congn\Desktop\vision_stator_project\data\input_images\2.png"

# Thư mục lưu kết quả
output_dir = r"C:\Users\congn\Desktop\vision_stator_project\data\test_results\roi_images"
os.makedirs(output_dir, exist_ok=True)

# Tên file ảnh đầu ra
output_path = os.path.join(output_dir, "hinh_clahe_gaussian.png")

# =========================================================
# 2. ĐỌC ẢNH
# =========================================================
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError("Không đọc được ảnh. Kiểm tra lại đường dẫn image_path.")

# Chuyển sang ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================================================
# 3. CLAHE - TĂNG TƯƠNG PHẢN CỤC BỘ
# =========================================================
# clipLimit không nên quá cao để tránh cháy sáng và khuếch đại nhiễu
clahe = cv2.createCLAHE(
    clipLimit=3.0,
    tileGridSize=(8, 8)
)

clahe_img = clahe.apply(gray)

# =========================================================
# 4. CLAHE + GAUSSIAN BLUR
# =========================================================
# Gaussian nhẹ để giảm nhiễu sau khi đã tăng tương phản
clahe_gaussian = cv2.GaussianBlur(clahe_img, (5, 5), 0)

# =========================================================
# 5. HIỂN THỊ 3 ẢNH
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

images = [gray, clahe_img, clahe_gaussian]
titles = [
    "(a) Ảnh gốc mức xám",
    "(b) Sau CLAHE",
    "(c) CLAHE + Gaussian Blur"
]

for ax, image, title in zip(axes, images, titles):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=14)
    ax.axis("off")

plt.tight_layout()

# =========================================================
# 6. LƯU ẢNH ĐỂ ĐƯA VÀO BÁO CÁO
# =========================================================
plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.08
)

plt.show()

print("Đã lưu hình tại:", output_path)


# ==============================================================================
# ===== BUOC 1.2 =====
# File goc: 1.2 Hough Circle.py
# Vai tro : Tìm tâm 12 stator bằng Hough Circle, chấm điểm bám biên trên ảnh Canny, lọc theo bán kính đồng thuận, chuẩn hoá bán kính chung.
# Dau ra  : Ảnh đánh dấu 12 tâm + ID + score.
# ==============================================================================

# -*- coding: utf-8 -*-
"""
Nhan dien tam stator bang Hough Circle co hieu chinh ban kinh dong thuan.

Chuc nang chinh:
- Doc anh dau vao.
- Cat ROI vung lam viec neu co cau hinh crop.
- Tien xu ly: anh xam -> CLAHE -> Gaussian Blur.
- Tim nhieu ung vien bang Hough Circle voi nhieu muc param2.
- Tinh edge-score tren anh Canny de danh gia do bam bien.
- Loai cac vong tron trung tam gan nhau.
- Loc theo ban kinh dong thuan de giu dung so stator mong muon.
- Tinh ban kinh chung r_common bang median.
- Hieu chinh lai ban kinh theo r_common de han che loi bam nham giua vong trong/vong ngoai.
- Ve va luu ket qua: tam, ban kinh, ID, score.

Luu y:
- Hough Circle duoc dung chu yeu de tim tam stator.
- Ban kinh cuoi cung duoc chuan hoa theo ban kinh dong thuan vi cac stator co cung kich thuoc thuc te.
"""

import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np


CONFIG = {
    "input_path": r"C:\Users\congn\Desktop\vision_stator_project\data\input_images\full_khay.png",
    "output_dir": r"C:\Users\congn\Desktop\vision_stator_project\data\test_results\hough_circle_12_stators",

    # So stator mong muon trong anh toan canh
    "expected_count": 12,

    # ROI vung lam viec lon. Neu w=0, h=0 thi xu ly toan anh.
    # Nen cat bao quanh 2 khay va chua bien an toan, khong cat qua sat.
    "crop": {"x": 0, "y": 0, "w": 0, "h": 0},

    "clahe": {
        "use": True,
        "clipLimit": 2.5,
        "tileGridSize": (8, 8),
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

        # Nguong diem bam bien cua ung vien Hough
        "edge_score_threshold": 0.18,

        # Loc ban kinh dong thuan so bo
        "radius_consensus_tol": 6,
        "radius_final_tol": 4,

        # Do day vung vanh tron khi tinh edge-score
        "edge_ring_width": 3,

        # Hieu chinh ban kinh theo r_common
        "use_common_radius_refine": True,
        "common_radius_deviation_tol": 2,
        "radius_refine_band": 4,
        "center_refine_range": 2,

        # Uu tien ban kinh gan r_common, tranh nhay sang vong ngoai
        "radius_penalty": 0.012,
        "center_penalty": 0.003,

        # Neu score tai r_common gan voi score tot nhat, uu tien r_common
        "common_radius_score_gap": 0.035,

        # True: ve tat ca stator bang cung ban kinh dong thuan sau khi tinh xong tam.
        # Phu hop khi cac stator co cung kich thuoc thuc te.
        "force_common_radius": True,
    },
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_image(path):
    """Doc anh bang np.fromfile de ho tro duong dan co ky tu tieng Viet."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, image):
    """Ghi anh bang cv2.imencode de ho tro duong dan co ky tu tieng Viet."""
    ext = os.path.splitext(path)[1]
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError("Khong the ghi anh: {}".format(path))
    encoded.tofile(path)


def preprocess_image(image):
    """Anh goc -> anh xam -> CLAHE -> Gaussian Blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    if CONFIG["clahe"]["use"]:
        clahe = cv2.createCLAHE(
            clipLimit=float(CONFIG["clahe"]["clipLimit"]),
            tileGridSize=CONFIG["clahe"]["tileGridSize"],
        )
        gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def crop_work_roi(image):
    """Cat ROI vung lam viec lon. Neu crop w/h = 0 thi xu ly toan anh."""
    h, w = image.shape[:2]
    x = max(0, int(CONFIG["crop"]["x"]))
    y = max(0, int(CONFIG["crop"]["y"]))
    cw = int(CONFIG["crop"]["w"]) if int(CONFIG["crop"]["w"]) > 0 else (w - x)
    ch = int(CONFIG["crop"]["h"]) if int(CONFIG["crop"]["h"]) > 0 else (h - y)

    x2 = min(w, x + cw)
    y2 = min(h, y + ch)

    if x >= x2 or y >= y2:
        return image.copy(), (0, 0)

    return image[y:y2, x:x2].copy(), (x, y)


def circle_edge_score(edges, cx, cy, radius, ring_width=None):
    """
    Tinh ti le diem bien Canny nam tren vanh tron ban kinh radius.
    Score cao nghia la vong tron bam bien tot.
    """
    if ring_width is None:
        ring_width = int(CONFIG["hough"]["edge_ring_width"])

    radius = int(round(radius))
    if radius < 8:
        return 0.0

    h, w = edges.shape[:2]
    cx = int(round(cx))
    cy = int(round(cy))

    if cx - radius < 0 or cy - radius < 0 or cx + radius >= w or cy + radius >= h:
        return 0.0

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = (dist >= radius - ring_width) & (dist <= radius + ring_width)

    total = int(ring.sum())
    if total == 0:
        return 0.0

    return float((edges[ring] > 0).sum()) / float(total)


def estimate_dominant_radius(candidates, tol):
    """Uoc luong ban kinh pho bien nhat tu tap ung vien."""
    if not candidates:
        return None

    step = max(1, int(tol))
    bins = {}

    for _, _, radius, score in candidates:
        key = int(round(float(radius) / float(step)))
        entry = bins.setdefault(key, {"count": 0, "score_sum": 0.0, "radii": []})
        entry["count"] += 1
        entry["score_sum"] += float(score)
        entry["radii"].append(int(radius))

    best_key = max(
        bins.items(),
        key=lambda kv: (kv[1]["count"], kv[1]["score_sum"]),
    )[0]

    return float(np.median(bins[best_key]["radii"]))


def radius_consensus_filter(candidates, tol):
    """Loc ung vien co ban kinh gan ban kinh dong thuan so bo."""
    if not candidates:
        return []

    dominant_radius = estimate_dominant_radius(candidates, tol)
    if dominant_radius is None:
        return candidates

    return [item for item in candidates if abs(item[2] - dominant_radius) <= tol]


def radius_consistency_refine(candidates, expected_count, base_tol):
    """
    Chon nhom ung vien co ban kinh dong deu nhat.
    Uu tien nhom co so luong nhieu, tong score cao, do trai ban kinh nho.
    """
    if not candidates:
        return [], None

    radii = [item[2] for item in candidates]
    unique_radii = sorted(set(radii))

    best_subset = list(candidates)
    best_center = float(np.median(radii))
    best_key = (-1, -1.0, float("inf"))

    for center_radius in unique_radii:
        subset = [item for item in candidates if abs(item[2] - center_radius) <= base_tol]
        if not subset:
            continue

        count = len(subset)
        score_sum = sum(item[3] for item in subset)
        spread = max(item[2] for item in subset) - min(item[2] for item in subset)
        key = (count, score_sum, -spread)

        if key > best_key:
            best_key = key
            best_subset = subset
            best_center = float(np.median([item[2] for item in subset]))

    refined = [item for item in candidates if abs(item[2] - best_center) <= base_tol]

    # Neu sau loc bi thieu stator, lay them cac ung vien gan ban kinh dong thuan nhat
    target_count = min(int(expected_count), len(candidates)) if expected_count > 0 else len(candidates)
    if len(refined) < target_count:
        remaining = [item for item in candidates if item not in refined]
        remaining = sorted(
            remaining,
            key=lambda item: (
                abs(item[2] - best_center),
                -item[3],
            ),
        )
        refined.extend(remaining[: target_count - len(refined)])

    refined = sorted(
        refined,
        key=lambda item: (
            abs(item[2] - best_center),
            -item[3],
        ),
    )

    return refined, best_center


def dedup_circles(candidates, min_dist, preferred_radius=None):
    """Loai cac vong tron co tam qua gan nhau, giu ung vien tot hon."""
    if preferred_radius is None:
        ordered = sorted(candidates, key=lambda x: x[3], reverse=True)
    else:
        ordered = sorted(
            candidates,
            key=lambda x: (
                abs(x[2] - preferred_radius),
                -x[3],
            ),
        )

    kept = []
    for cx, cy, radius, score in ordered:
        ok = True
        for kx, ky, _, _ in kept:
            if math.hypot(cx - kx, cy - ky) < min_dist:
                ok = False
                break
        if ok:
            kept.append((int(cx), int(cy), int(radius), float(score)))

    return kept


def _collect_candidates(gray, edges, min_r, max_r, min_dist, p2_values, edge_threshold, use_edge_gate):
    """Chay HoughCircles voi nhieu muc param2 va thu thap ung vien."""
    h, w = gray.shape[:2]
    hcfg = CONFIG["hough"]
    candidates = []

    for param2 in p2_values:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=float(hcfg["dp"]),
            minDist=float(min_dist),
            param1=float(hcfg["param1"]),
            param2=float(param2),
            minRadius=int(min_r),
            maxRadius=int(max_r),
        )

        if circles is None:
            continue

        for cx, cy, radius in np.round(circles[0]).astype(int):
            if cx - radius < 0 or cy - radius < 0 or cx + radius >= w or cy + radius >= h:
                continue
            if not (min_r <= radius <= max_r):
                continue

            score = circle_edge_score(edges, cx, cy, radius)
            if use_edge_gate and score < float(edge_threshold):
                continue

            candidates.append((int(cx), int(cy), int(radius), float(score)))

    return candidates


def refine_one_circle_by_common_radius(edges, circle, r_common):
    """
    Hieu chinh 1 vong tron quanh ban kinh dong thuan.

    Y tuong:
    - Giu tam Hough lam tam ban dau.
    - Thu dich tam nho quanh tam cu.
    - Thu ban kinh quanh r_common.
    - Uu tien ban kinh gan r_common de tranh bam nham sang vong ngoai.
    - Neu r_common co score gan voi score tot nhat thi snap ve r_common.
    """
    h, w = edges.shape[:2]
    hcfg = CONFIG["hough"]

    cx0, cy0, r_old, old_score = circle
    r0 = int(round(r_common))

    center_range = int(hcfg["center_refine_range"])
    radius_band = int(hcfg["radius_refine_band"])
    radius_penalty = float(hcfg["radius_penalty"])
    center_penalty = float(hcfg["center_penalty"])
    score_gap = float(hcfg["common_radius_score_gap"])

    best_cx = int(cx0)
    best_cy = int(cy0)
    best_r = int(r_old)
    best_score = float(old_score)
    best_final_score = -1e9

    best_common = None
    best_common_score = -1.0
    best_common_final_score = -1e9

    for dx in range(-center_range, center_range + 1):
        for dy in range(-center_range, center_range + 1):
            trial_cx = int(cx0 + dx)
            trial_cy = int(cy0 + dy)

            for dr in range(-radius_band, radius_band + 1):
                trial_r = int(r0 + dr)
                if trial_r < 8:
                    continue
                if trial_cx - trial_r < 0 or trial_cy - trial_r < 0:
                    continue
                if trial_cx + trial_r >= w or trial_cy + trial_r >= h:
                    continue

                score = circle_edge_score(edges, trial_cx, trial_cy, trial_r)

                # Phat diem neu lech xa r_common hoac dich tam qua nhieu
                final_score = score
                final_score -= radius_penalty * abs(trial_r - r0)
                final_score -= center_penalty * math.hypot(dx, dy)

                if final_score > best_final_score:
                    best_final_score = final_score
                    best_cx = trial_cx
                    best_cy = trial_cy
                    best_r = trial_r
                    best_score = score

                if trial_r == r0 and final_score > best_common_final_score:
                    best_common_final_score = final_score
                    best_common = (trial_cx, trial_cy, trial_r, score)
                    best_common_score = score

    # Neu ban kinh chung co score gan voi phuong an tot nhat thi uu tien r_common
    if best_common is not None and best_common_score >= best_score - score_gap:
        return (
            int(best_common[0]),
            int(best_common[1]),
            int(best_common[2]),
            float(best_common[3]),
        )

    return (int(best_cx), int(best_cy), int(best_r), float(best_score))


def refine_circles_by_common_radius(edges, circles):
    """
    Tinh r_common va hieu chinh ban kinh cho toan bo stator.
    Neu force_common_radius=True, ban kinh ve cuoi cung se dung cung r_common.
    """
    if not circles:
        return [], None

    hcfg = CONFIG["hough"]
    if not hcfg["use_common_radius_refine"]:
        r_common = float(np.median([item[2] for item in circles]))
        return circles, r_common

    r_common = float(np.median([item[2] for item in circles]))
    r_common_int = int(round(r_common))
    deviation_tol = int(hcfg["common_radius_deviation_tol"])
    force_common = bool(hcfg["force_common_radius"])

    refined = []
    for circle in circles:
        cx, cy, radius, score = circle

        # Neu ban kinh lech nhieu thi quet lai ca tam va ban kinh quanh r_common.
        # Neu khong lech nhieu, van tinh lai score tai r_common de ket qua dong deu.
        if abs(radius - r_common_int) > deviation_tol:
            new_circle = refine_one_circle_by_common_radius(edges, circle, r_common_int)
        else:
            new_score = circle_edge_score(edges, cx, cy, r_common_int)
            new_circle = (int(cx), int(cy), int(r_common_int), float(new_score))

        # Chuan hoa ban kinh ve r_common neu cac stator co cung kich thuoc thuc te.
        if force_common:
            ncx, ncy, _, _ = new_circle
            common_score = circle_edge_score(edges, ncx, ncy, r_common_int)
            new_circle = (int(ncx), int(ncy), int(r_common_int), float(common_score))

        refined.append(new_circle)

    return refined, float(r_common_int)


def detect_stator_centers(roi_bgr):
    """Phat hien dong thoi nhieu stator trong ROI/toan anh."""
    gray = preprocess_image(roi_bgr)
    edges = cv2.Canny(
        gray,
        int(CONFIG["canny"]["threshold1"]),
        int(CONFIG["canny"]["threshold2"]),
    )

    h, w = gray.shape[:2]
    hcfg = CONFIG["hough"]
    raw_candidates = []

    min_r = int(hcfg["minRadius"])
    max_r = int(hcfg["maxRadius"])
    min_dist = int(hcfg["minDist"])

    # Lan 1: tham so chinh va 2 muc gan ke
    p2_list = [
        int(hcfg["param2"]),
        int(hcfg["param2"] + 6),
        max(18, int(hcfg["param2"] - 6)),
    ]
    candidates = _collect_candidates(
        gray,
        edges,
        min_r,
        max_r,
        min_dist,
        p2_list,
        hcfg["edge_score_threshold"],
        True,
    )
    raw_candidates.extend(candidates)

    # Fallback 1: neu khong co ung vien, giam nguong va noi tham so
    if not candidates:
        relaxed_edge = max(0.03, float(hcfg["edge_score_threshold"]) * 0.35)
        max_r2 = max(max_r, int(min(h, w) * 0.48))
        min_dist2 = max(40, int(min_dist * 0.6))
        p2_list2 = [
            max(14, int(hcfg["param2"] - 12)),
            max(18, int(hcfg["param2"] - 6)),
            int(hcfg["param2"]),
        ]
        candidates = _collect_candidates(
            gray,
            edges,
            min_r,
            max_r2,
            min_dist2,
            p2_list2,
            relaxed_edge,
            True,
        )
        raw_candidates.extend(candidates)

    # Fallback 2: neu van khong co ung vien, bo edge gate
    if not candidates:
        max_r3 = max(max_r, int(min(h, w) * 0.48))
        min_dist3 = max(35, int(min_dist * 0.5))
        p2_list3 = [
            max(12, int(hcfg["param2"] - 14)),
            max(16, int(hcfg["param2"] - 8)),
        ]
        candidates = _collect_candidates(
            gray,
            edges,
            min_r,
            max_r3,
            min_dist3,
            p2_list3,
            0.0,
            False,
        )
        raw_candidates.extend(candidates)

    if not raw_candidates:
        return gray, edges, [], [], None

    # Uoc luong ban kinh pho bien tu tat ca ung vien
    preferred_radius = estimate_dominant_radius(
        raw_candidates,
        int(hcfg["radius_consensus_tol"]),
    )

    # Loai cac vong tron trung tam gan nhau
    candidates = dedup_circles(
        raw_candidates,
        max(int(hcfg["min_center_dist"]), int(min_dist * 0.7)),
        preferred_radius=preferred_radius,
    )

    # Loc ban kinh dong thuan
    candidates = radius_consensus_filter(candidates, int(hcfg["radius_consensus_tol"]))

    expected = int(CONFIG["expected_count"])
    candidates, dominant_radius = radius_consistency_refine(
        candidates,
        expected,
        int(hcfg["radius_final_tol"]),
    )

    # Giu dung so luong mong muon
    candidates = sorted(
        candidates,
        key=lambda x: (
            abs(x[2] - (dominant_radius if dominant_radius is not None else x[2])),
            -x[3],
        ),
    )
    if expected > 0:
        candidates = candidates[:expected]

    # Hieu chinh ban kinh theo r_common
    candidates, common_radius = refine_circles_by_common_radius(edges, candidates)

    # Sap xep ID theo hang: tren xuong duoi, trai sang phai
    candidates = sorted(candidates, key=lambda x: (x[1], x[0]))
    raw_candidates = sorted(raw_candidates, key=lambda x: x[3], reverse=True)

    return gray, edges, raw_candidates, candidates, common_radius


def draw_results(image, circles, offset_xy=(0, 0), common_radius=None):
    out = image.copy()
    ox, oy = offset_xy

    for idx, (cx, cy, radius, score) in enumerate(circles, start=1):
        gx = int(cx + ox)
        gy = int(cy + oy)

        cv2.circle(out, (gx, gy), int(radius), (0, 255, 0), 2)
        cv2.drawMarker(out, (gx, gy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(
            out,
            "ID{:02d}".format(idx),
            (gx + 8, gy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            out,
            "r={} s={:.3f}".format(radius, score),
            (gx + 8, gy + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 220, 255),
            1,
        )

    if common_radius is not None:
        cv2.putText(
            out,
            "common radius = {} px".format(int(round(common_radius))),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    return out


def draw_candidate_overlay(image, circles, offset_xy=(0, 0)):
    out = image.copy()
    ox, oy = offset_xy

    for cx, cy, radius, score in circles:
        gx = int(cx + ox)
        gy = int(cy + oy)
        color = (0, min(255, int(80 + score * 700)), 255)

        cv2.circle(out, (gx, gy), int(radius), color, 1)
        cv2.drawMarker(out, (gx, gy), (255, 180, 0), cv2.MARKER_CROSS, 10, 1)

    return out


def draw_valid_overview(image, circles, offset_xy=(0, 0), common_radius=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ox, oy = offset_xy

    for idx, (cx, cy, radius, score) in enumerate(circles, start=1):
        gx = int(cx + ox)
        gy = int(cy + oy)

        cv2.circle(out, (gx, gy), int(radius), (0, 255, 0), 3)
        cv2.drawMarker(out, (gx, gy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(
            out,
            "ID{:02d}".format(idx),
            (gx + 10, gy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            out,
            "C=({}, {})".format(gx, gy),
            (gx + 10, gy + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            out,
            "R={} S={:.3f}".format(radius, score),
            (gx + 10, gy + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
        )

    if common_radius is not None:
        cv2.putText(
            out,
            "R_common = {} px".format(int(round(common_radius))),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    return out


def process_image(input_path, output_dir=None, save_outputs=True):
    if output_dir is None:
        output_dir = CONFIG["output_dir"]

    ensure_dir(output_dir)

    image = read_image(input_path)
    if image is None:
        raise IOError("Khong doc duoc anh: {}".format(input_path))

    roi, offset_xy = crop_work_roi(image)
    gray, edges, raw_candidates, circles, common_radius = detect_stator_centers(roi)

    if not circles:
        raise RuntimeError("Khong tim thay stator nao bang Hough Circle.")

    raw_result = draw_candidate_overlay(image, raw_candidates, offset_xy)
    result = draw_results(image, circles, offset_xy, common_radius=common_radius)
    valid_overview = draw_valid_overview(image, circles, offset_xy, common_radius=common_radius)

    if save_outputs:
        write_image(os.path.join(output_dir, "00_raw_hough_candidates.png"), raw_result)
        write_image(os.path.join(output_dir, "01_detected_stator_centers_refined.png"), result)
        write_image(os.path.join(output_dir, "02_preprocess_gray.png"), gray)
        write_image(os.path.join(output_dir, "03_canny_edges.png"), edges)
        write_image(os.path.join(output_dir, "04_valid_stators_overview_refined.png"), valid_overview)

    print("So ung vien truoc loc: {}".format(len(raw_candidates)))
    print("So stator phat hien: {}".format(len(circles)))
    if common_radius is not None:
        print("Ban kinh dong thuan r_common: {} px".format(int(round(common_radius))))

    for idx, (cx, cy, radius, score) in enumerate(circles, start=1):
        gx = int(cx + offset_xy[0])
        gy = int(cy + offset_xy[1])
        print(
            "ID{idx:02d}: center=({gx}, {gy}), radius={radius}, score={score:.3f}".format(
                idx=idx,
                gx=gx,
                gy=gy,
                radius=radius,
                score=score,
            )
        )

    if save_outputs:
        print("Da luu ket qua tai: {}".format(output_dir))

    return {
        "image": image,
        "gray": gray,
        "edges": edges,
        "raw_candidates": raw_candidates,
        "circles": circles,
        "common_radius": common_radius,
        "raw_result": raw_result,
        "result": result,
        "valid_overview": valid_overview,
        "output_dir": output_dir,
    }


def fit_image_for_screen(image, max_width=1200, max_height=800):
    h, w = image.shape[:2]
    scale = min(float(max_width) / float(w), float(max_height) / float(h), 1.0)
    if scale >= 1.0:
        return image
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def show_result_windows(outputs):
    windows = [
        ("Detected Stator Centers - Refined", outputs["result"]),
        ("Raw Hough Candidates", outputs["raw_result"]),
        ("Valid Stators Overview - Refined", outputs["valid_overview"]),
        ("Preprocess Gray", outputs["gray"]),
        ("Canny Edges", outputs["edges"]),
    ]

    for title, image in windows:
        cv2.imshow(title, fit_image_for_screen(image))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_gui():
    root = tk.Tk()
    root.title("Hough Circle - Stator Detection Refined")
    root.geometry("760x230")
    root.resizable(False, False)

    selected_path = tk.StringVar(value=CONFIG["input_path"])
    status_text = tk.StringVar(value="Chon anh, sau do bam Run.")

    def browse_image():
        file_path = filedialog.askopenfilename(
            title="Chon anh dau vao",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            selected_path.set(file_path)
            status_text.set("Da chon anh. San sang xu ly.")

    def run_detection():
        input_path = selected_path.get().strip()
        if not input_path:
            messagebox.showwarning("Thieu anh", "Vui long chon anh dau vao.")
            return
        if not os.path.isfile(input_path):
            messagebox.showerror("Sai duong dan", "Khong tim thay file anh da chon.")
            return

        try:
            status_text.set("Dang xu ly anh...")
            root.update_idletasks()
            outputs = process_image(input_path, CONFIG["output_dir"], save_outputs=True)
            msg = "Hoan tat. Tim thay {} stator.".format(len(outputs["circles"]))
            if outputs["common_radius"] is not None:
                msg += " R_common = {} px.".format(int(round(outputs["common_radius"])))
            status_text.set(msg + " Dang hien thi ket qua.")
            show_result_windows(outputs)
        except Exception as exc:
            status_text.set("Xu ly that bai.")
            messagebox.showerror("Loi xu ly", str(exc))

    frame = tk.Frame(root, padx=16, pady=16)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="Anh dau vao:", anchor="w", font=("Arial", 11, "bold")).pack(fill="x")

    path_frame = tk.Frame(frame)
    path_frame.pack(fill="x", pady=(8, 12))

    tk.Entry(path_frame, textvariable=selected_path, font=("Arial", 10)).pack(
        side="left", fill="x", expand=True
    )
    tk.Button(path_frame, text="Import anh", width=14, command=browse_image).pack(side="left", padx=(8, 0))

    button_frame = tk.Frame(frame)
    button_frame.pack(fill="x", pady=(0, 12))
    tk.Button(
        button_frame,
        text="Run",
        width=16,
        height=2,
        bg="#2d89ef",
        fg="white",
        command=run_detection,
    ).pack(side="left")

    tk.Label(
        frame,
        textvariable=status_text,
        anchor="w",
        justify="left",
        fg="#1f1f1f",
        wraplength=720,
    ).pack(fill="x")

    tk.Label(
        frame,
        text="Ket qua se duoc luu trong thu muc output. Ban kinh cuoi cung duoc hieu chinh theo R_common.",
        anchor="w",
        justify="left",
        fg="#555555",
        wraplength=720,
    ).pack(fill="x", pady=(10, 0))

    root.mainloop()


def main():
    process_image(CONFIG["input_path"], CONFIG["output_dir"], save_outputs=True)


if __name__ == "__main__":
    run_gui()


# ==============================================================================
# ===== BUOC 2 =====
# File goc: 2. cat roi.py
# Vai tro : Cắt ROI vuông quanh từng stator (nạp động logic từ file 1.2).
# Dau ra  : Hinh_4_8_*, ảnh ROI từng stator. ⚠️ Phụ thuộc file 1.2 (phải nằm chung thư mục).
# ==============================================================================

# -*- coding: utf-8 -*-
"""
Tạo Hình 4.8: Cắt ROI quanh từng stator sau khi phát hiện Hough Circle.
Thông số Hough Circle lấy theo file code nhận diện stator hiện tại.
"""

import os
import math
import importlib.util
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


# =========================
# 1. CONFIG lấy theo code hiện tại
# =========================
CONFIG = {
    "expected_count": 12,

    "roi": {
        "half_size_scale": 1.3
    },

    "clahe": {
        "use": True,
        "clipLimit": 2.5,
        "tileGridSize": (8, 8)
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
    }
}


# =========================
# 2. Đường dẫn ảnh
# =========================
input_path = r"C:\Users\congn\Desktop\vision_stator_project\data\input_images\2.png"

output_dir = "data/test_results"
roi_output_dir = os.path.join(output_dir, "roi_images")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(roi_output_dir, exist_ok=True)

HOUGH_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "1.2 Hough Circle.py"
)


# =========================
# 3. Hàm đọc ảnh hỗ trợ đường dẫn tiếng Việt
# =========================
def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# =========================
# 4. Tiền xử lý ảnh trước Hough Circle
# =========================
def preprocess_image(image):
    if image is None:
        raise ValueError("Ảnh đầu vào không hợp lệ")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if CONFIG["clahe"]["use"]:
        clahe = cv2.createCLAHE(
            clipLimit=CONFIG["clahe"]["clipLimit"],
            tileGridSize=CONFIG["clahe"]["tileGridSize"]
        )
        gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray


# =========================
# 5. Tính điểm bám biên của đường tròn
# =========================
def circle_edge_score(edges, cx, cy, r):
    if r < 8:
        return 0.0

    h, w = edges.shape[:2]

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    ring = (dist >= r - 3) & (dist <= r + 3)
    total = int(ring.sum())

    if total == 0:
        return 0.0

    return float((edges[ring] > 0).sum()) / float(total)


# =========================
# 6. Lọc các đường tròn có bán kính cùng nhóm
# =========================
def radius_consensus_filter(candidates, tol):
    if not candidates:
        return []

    bins = {}

    for c in candidates:
        r = int(c[2])
        k = int(round(float(r) / float(max(1, tol))))
        bins[k] = bins.get(k, 0) + 1

    dominant_radius = max(bins.items(), key=lambda kv: kv[1])[0] * max(1, tol)

    filtered = [
        c for c in candidates
        if abs(c[2] - dominant_radius) <= tol
    ]

    return filtered


# =========================
# 7. Loại các đường tròn trùng nhau
# =========================
def dedup_circles(candidates, min_dist):
    kept = []

    for cx, cy, r, score in sorted(candidates, key=lambda x: x[3], reverse=True):
        is_valid = True

        for kx, ky, _, _ in kept:
            if math.hypot(cx - kx, cy - ky) < min_dist:
                is_valid = False
                break

        if is_valid:
            kept.append((cx, cy, r, score))

    return kept


# =========================
# 8. Phát hiện tâm stator bằng Hough Circle
# =========================
def detect_stator_center(image):
    gray = preprocess_image(image)
    h, w = gray.shape[:2]

    edges = cv2.Canny(gray, 70, 170)

    hcfg = CONFIG["hough"]

    min_r = int(hcfg["minRadius"])
    max_r = int(hcfg["maxRadius"])
    min_dist = int(hcfg["minDist"])

    p2_values = [
        int(hcfg["param2"]),
        int(hcfg["param2"] + 6),
        max(18, int(hcfg["param2"] - 6))
    ]

    candidates = []

    for p2 in p2_values:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=float(hcfg["dp"]),
            minDist=float(min_dist),
            param1=float(hcfg["param1"]),
            param2=float(p2),
            minRadius=int(min_r),
            maxRadius=int(max_r),
        )

        if circles is None:
            continue

        for cx, cy, r in np.round(circles[0]).astype(int):
            # Loại đường tròn nằm sát biên ảnh
            if cx - r < 0 or cy - r < 0 or cx + r >= w or cy + r >= h:
                continue

            # Lọc theo bán kính
            if not (min_r <= r <= max_r):
                continue

            # Chấm điểm mức độ bám biên
            score = circle_edge_score(edges, cx, cy, r)

            if score < float(hcfg["edge_score_threshold"]):
                continue

            candidates.append((int(cx), int(cy), int(r), float(score)))

    # Loại các đường tròn trùng tâm hoặc quá gần nhau
    candidates = dedup_circles(
        candidates,
        max(int(hcfg["min_center_dist"]), int(min_dist * 0.7))
    )

    # Với ảnh nhiều stator, ưu tiên các đường tròn có bán kính cùng nhóm
    candidates = radius_consensus_filter(
        candidates,
        int(hcfg["radius_consensus_tol"])
    )

    # Sắp xếp theo điểm bám biên và bán kính
    candidates = sorted(
        candidates,
        key=lambda x: (x[3], x[2]),
        reverse=True
    )

    # Chỉ lấy đúng số stator kỳ vọng
    expected = int(CONFIG["expected_count"])

    if expected > 0:
        candidates = candidates[:expected]

    # Sắp xếp lại theo thứ tự từ trên xuống, trái sang phải
    candidates = sorted(candidates, key=lambda x: (x[1], x[0]))

    return [(c[0], c[1], c[2], c[3]) for c in candidates], gray, edges


def load_hough_circle_module():
    if not os.path.isfile(HOUGH_SCRIPT_PATH):
        raise FileNotFoundError(f"Khong tim thay file Hough Circle: {HOUGH_SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("hough_circle_module", HOUGH_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Khong the nap module tu file: {HOUGH_SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_stator_center_from_hough(input_image_path, image):
    module = load_hough_circle_module()

    if not hasattr(module, "process_image"):
        raise AttributeError("File 1.2 Hough Circle.py khong co ham process_image().")

    outputs = module.process_image(
        input_image_path,
        output_dir=output_dir,
        save_outputs=False
    )

    detected_image = outputs.get("image")
    if detected_image is None:
        raise ValueError("Khong nhan duoc anh ket qua tu file 1.2 Hough Circle.py")

    if detected_image.shape[:2] != image.shape[:2]:
        raise ValueError("Kich thuoc anh tu file Hough khong khop voi anh dau vao ROI.")

    circles = outputs.get("circles", [])
    gray = outputs.get("gray")
    edges = outputs.get("edges")

    return circles, gray, edges


# =========================
# 9. Cắt ROI quanh stator
# =========================
def crop_stator_roi(image, cx, cy, r):
    h, w = image.shape[:2]

    # Thu ROI sát theo stator hơn thay vì lấy quá rộng.
    half = int(max(1, round(float(r) * CONFIG["roi"]["half_size_scale"])))

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    roi = image[y1:y2, x1:x2].copy()

    return roi, (x1, y1, x2, y2)


# =========================
# 10. Chương trình chính
# =========================
image = read_image(input_path)

if image is None:
    raise ValueError(f"Không đọc được ảnh: {input_path}")

circles, hough_input, edges = detect_stator_center_from_hough(input_path, image)

print(f"Số stator phát hiện được: {len(circles)}")

# Ảnh để vẽ kết quả
display = image.copy()

roi_list = []

for idx, (cx, cy, r, score) in enumerate(circles, start=1):
    roi, bbox = crop_stator_roi(image, cx, cy, r)
    x1, y1, x2, y2 = bbox

    roi_list.append((idx, roi, bbox, (cx, cy, r, score)))

    # Lưu ROI riêng
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    roi_path = os.path.join(roi_output_dir, f"roi_stator_{idx:02d}.png")
    cv2.imwrite(roi_path, roi_gray)

    # Vẽ đường tròn Hough
    cv2.circle(display, (cx, cy), r, (0, 255, 0), 2)

    # Vẽ tâm
    cv2.drawMarker(
        display,
        (cx, cy),
        (0, 255, 255),
        cv2.MARKER_CROSS,
        14,
        2
    )

    # Vẽ khung ROI
    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Ghi ID
    cv2.putText(
        display,
        f"ID{idx}",
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    # Ghi score
    cv2.putText(
        display,
        f"s={score:.2f}",
        (x1, min(display.shape[0] - 5, y2 + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1
    )


# =========================
# 11. Lưu ảnh toàn cảnh có đánh dấu ROI
# =========================
marked_path = os.path.join(output_dir, "Hinh_4_8_toan_canh_danh_dau_ROI.png")
cv2.imwrite(marked_path, display)


# =========================
# 12. Tạo ảnh ghép dùng cho báo cáo
# =========================
num_roi_show = min(len(roi_list), int(CONFIG["expected_count"]))
roi_cols = 3
roi_rows = 4

fig = plt.figure(figsize=(14, 10))
grid = gridspec.GridSpec(
    roi_rows,
    roi_cols + 1,
    figure=fig,
    width_ratios=[1.8, 1.0, 1.0, 1.0],
    wspace=0.04,
    hspace=0.12
)

ax_overview = fig.add_subplot(grid[:, 0])
ax_overview.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
ax_overview.set_title("(a) Overview with Hough Circle and ROI", pad=6)
ax_overview.axis("off")

for i in range(num_roi_show):
    idx, roi, bbox, circle_info = roi_list[i]

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    row_idx = i % roi_rows
    col_idx = (i // roi_rows) + 1

    ax_roi = fig.add_subplot(grid[row_idx, col_idx])
    ax_roi.imshow(roi_rgb)
    ax_roi.set_title(f"(b{i + 1}) ROI stator {idx}", fontsize=10, pad=4)
    ax_roi.axis("off")

fig.subplots_adjust(left=0.02, right=0.99, top=0.96, bottom=0.03)

compare_path = os.path.join(output_dir, "Hinh_4_8_cat_ROI_quanh_tung_stator.png")
plt.savefig(compare_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.show()


# =========================
# 13. In thông tin kết quả
# =========================
print("Danh sách stator phát hiện được:")
for idx, (cx, cy, r, score) in enumerate(circles, start=1):
    print(f"ID{idx}: center=({cx}, {cy}), radius={r}, edge_score={score:.3f}")

print("\nĐã lưu ảnh toàn cảnh có đánh dấu ROI tại:")
print(marked_path)

print("\nĐã lưu ảnh ghép dùng cho báo cáo tại:")
print(compare_path)

print("\nCác ROI riêng được lưu tại:")
print(roi_output_dir)


# ==============================================================================
# ===== BUOC 3 =====
# File goc: 3.dau vao thuat toan xoay.py
# Vai tro : GUI giám sát tiền xử lý ROI trước Canny (xám → tăng tương phản → lọc nhiễu → Canny).
# Dau ra  : Hinh_4_9_*.
# ==============================================================================

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


# ==============================================================================
# ===== BUOC 4 =====
# File goc: 4.py
# Vai tro : So sánh 4 phương án tiền xử lý ROI (Canny / +Morphology / +Contour / Morphology trực tiếp).
# Dau ra  : Ảnh so sánh các phương án.
# ==============================================================================

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


# ==============================================================================
# ===== BUOC 5 =====
# File goc: 5. tim coutour ngoai bang ban kinh.py
# Vai tro : GUI quét 360 tia từ tâm, lấy điểm Canny xa nhất hợp lệ, nội suy & lọc spike → đường bao ngoài.
# Dau ra  : Hinh_4_10_*, radial_signature_360_values.npy/.txt.
# ==============================================================================

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


# ==============================================================================
# ===== BUOC 6 =====
# File goc: 6.radial signature.py
# Vai tro : GUI so khớp radial signature mẫu ↔ ROI bằng MSE sau chuẩn hoá → suy ra góc xoay.
# Dau ra  : Đồ thị so khớp + góc xoay.
# ==============================================================================

# -*- coding: utf-8 -*-
"""
6.radial signature.py
GUI so khop Radial Signature de xac dinh goc xoay stator.

Phien ban viet lai:
- Lay thong so va logic Radial Signature tu file 5 / code bam sat bien.
- Xu ly anh mau va anh ROI cung mot pipeline.
- Tim tam bang Hough Circle.
- Quet 360 tia tu tam tren anh Canny, lay diem bien xa nhat hop le.
- Noi suy diem thieu, loc spike bang local median.
- So khop Radial Signature mau va ROI bang MSE sau chuan hoa.
- Mac dinh dung signature lam muot de so khop, giam nham do rang lap lai.
- Hien thi anh radial dep hon: resize anh truoc, scale toa do roi moi ve tia.
- GUI co PanedWindow de keo ngang giua vung anh va vung ket qua, keo ngang giua mau/ROI.
- Do thi co the phong to va Ctrl + lan chuot de zoom.
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
# 1. THONG SO XU LY - lay theo file radial bam sat bien
# =========================================================

# Tien xu ly
CLAHE_CLIP_LIMIT = 1.0
CLAHE_TILE_GRID_SIZE = (7, 7)
USE_CLAHE_FOR_RAW_ROI = True

GAUSSIAN_KERNEL = (7, 7)
GAUSSIAN_SIGMA = 8.0

CANNY_LOW = 70
CANNY_HIGH = 170

# Hough Circle
HOUGH_DP = 1.2
HOUGH_PARAM1 = 120
HOUGH_PARAM2 = 30

# Radial Signature
ANGLE_STEP_DEG = 1
MAX_RADIUS_RATIO = 1.32
EDGE_NEIGHBOR_WINDOW = 3
EDGE_NEIGHBOR_MIN_COUNT = 2
LOCAL_MEDIAN_WINDOW = 9
SPIKE_THRESHOLD_PX = 35

# Matching
MATCH_USE_SMOOTHED_SIGNATURE = True
MATCH_SMOOTH_WINDOW = 31          # 21/31/41: tang neu bi nham do rang lap lai
MATCH_TRY_BOTH_DIRECTIONS = True  # thu ca np.roll(+shift) va np.roll(-shift)

# Hien thi
DISPLAY_MAX_WIDTH = 720
DISPLAY_MAX_HEIGHT = 520
SHOW_REMOVED_OUTLIERS = True

# Neu muon giam do day tia tren man hinh: de 2 hoac 3.
# Neu muon ve du 360 tia giong file 5: de 1.
DISPLAY_RAY_STEP = 1

DEFAULT_OUTPUT_DIR = os.path.join("data", "test_results", "radial_signature_match")
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


def prepare_gray_roi(image):
    if image is None:
        raise ValueError("Anh dau vao khong hop le")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

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


def preprocess_roi(image):
    """Pipeline xu ly ROI goc: gray -> CLAHE -> Gaussian -> Canny."""
    gray = prepare_gray_roi(image)
    if USE_CLAHE_FOR_RAW_ROI:
        enhanced = apply_clahe(gray)
    else:
        enhanced = gray.copy()
    blur = cv2.GaussianBlur(enhanced, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    canny = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return gray, enhanced, blur, canny


# =========================================================
# 3. TIM TAM BANG HOUGH CIRCLE
# =========================================================

def find_center_by_hough_or_image_center(gray_blur):
    h, w = gray_blur.shape[:2]

    min_radius = int(min(w, h) * 0.20)
    max_radius = int(min(w, h) * 0.48)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=max(1, min(w, h) // 2),
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
# 4. RADIAL SIGNATURE TU CANNY
# =========================================================

def has_edge_neighbor(edge_image, x, y, window=3, min_count=2):
    h, w = edge_image.shape[:2]
    half = window // 2

    x1 = max(0, x - half)
    x2 = min(w, x + half + 1)
    y1 = max(0, y - half)
    y2 = min(h, y + half + 1)

    patch = edge_image[y1:y2, x1:x2]
    return int(np.count_nonzero(patch)) >= min_count


def circular_interpolate_missing(signal):
    values = signal.copy().astype(np.float32)
    n = len(values)
    valid = ~np.isnan(values)

    if np.sum(valid) == 0:
        return np.zeros(n, dtype=np.float32)

    if np.sum(valid) == 1:
        return np.full(n, values[valid][0], dtype=np.float32)

    idx = np.arange(n)
    valid_idx = idx[valid]
    valid_val = values[valid]

    extended_idx = np.r_[valid_idx - n, valid_idx, valid_idx + n]
    extended_val = np.r_[valid_val, valid_val, valid_val]

    return np.interp(idx, extended_idx, extended_val).astype(np.float32)


def rebuild_points_from_signature(center, angles, signature, image_shape):
    h, w = image_shape[:2]
    cx, cy = center
    points = []

    for deg, radius in zip(angles, signature):
        theta = np.deg2rad(deg)
        x = int(round(cx + radius * np.cos(theta)))
        y = int(round(cy + radius * np.sin(theta)))

        if 0 <= x < w and 0 <= y < h:
            points.append((x, y))
        else:
            points.append(None)

    return points


def radial_signature_from_canny_robust(edge_image, center, estimated_radius,
                                       hough_found=True, angle_step_deg=1):
    """
    Moi goc quet tu tam ra ngoai.
    Lay diem Canny xa nhat hop le, co kiem tra cum pixel lan can.
    Gioi han r_max theo R Hough de tranh bat nhieu ben ngoai.
    """
    h, w = edge_image.shape[:2]
    cx, cy = center

    angles = np.arange(0, 360, angle_step_deg, dtype=np.float32)
    signature_raw = np.full(len(angles), np.nan, dtype=np.float32)
    points_raw = [None] * len(angles)

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

        for radius in range(0, r_max + 1):
            x = int(round(cx + radius * np.cos(theta)))
            y = int(round(cy + radius * np.sin(theta)))

            if x < 0 or x >= w or y < 0 or y >= h:
                break

            if edge_image[y, x] > 0:
                if has_edge_neighbor(
                    edge_image, x, y,
                    window=EDGE_NEIGHBOR_WINDOW,
                    min_count=EDGE_NEIGHBOR_MIN_COUNT
                ):
                    best_r = radius
                    best_point = (x, y)

        if best_point is not None:
            signature_raw[i] = best_r
            points_raw[i] = best_point

    valid_count = int(np.sum(~np.isnan(signature_raw)))
    signature_interp = circular_interpolate_missing(signature_raw)
    points_interp = rebuild_points_from_signature(
        center=center,
        angles=angles,
        signature=signature_interp,
        image_shape=edge_image.shape
    )

    return angles, signature_raw, signature_interp, points_raw, points_interp, valid_count, r_max


def remove_local_spikes_circular(signature, window_size=9, spike_threshold=35):
    """Loc cac gai ban kinh dai bat thuong bang local median vong tron."""
    values = signature.copy().astype(np.float32)
    n = len(values)

    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    padded = np.r_[values[-pad:], values, values[:pad]]
    local_median = np.zeros_like(values)

    for i in range(n):
        local_median[i] = np.median(padded[i:i + window_size])

    outlier_mask = values > (local_median + spike_threshold)

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    global_upper = q3 + 1.5 * iqr

    outlier_mask = outlier_mask & (values > global_upper * 0.95)

    filtered = values.copy()
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
# 5. MATCHING SIGNATURE
# =========================================================

def smooth_signature_circular(signature, window_size=31):
    """Lam muot Radial Signature dang vong tron de giam nhieu/rang lap lai."""
    sig = signature.astype(np.float32).copy()
    n = len(sig)

    if window_size <= 1:
        return sig

    if window_size % 2 == 0:
        window_size += 1

    window_size = min(window_size, n if n % 2 == 1 else n - 1)
    pad = window_size // 2
    padded = np.r_[sig[-pad:], sig, sig[:pad]]
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def normalize_signature(signature):
    sig = signature.astype(np.float32).copy()
    sig = sig - np.mean(sig)
    std = np.std(sig)
    if std > 1e-6:
        sig = sig / std
    return sig


def _mse_curve_for_roll(ref_norm, cur_norm, direction=1):
    n = len(ref_norm)
    curve = np.zeros(n, dtype=np.float32)
    for shift in range(n):
        shifted = np.roll(cur_norm, direction * shift)
        curve[shift] = np.mean((ref_norm - shifted) ** 2)
    return curve


def match_radial_signature(reference_signature, current_signature, angle_step_deg=1):
    """
    So khop bang MSE sau chuan hoa.
    Mac dinh lam muot signature de giam nham do cac rang lap lai.
    Thu ca 2 chieu roll neu MATCH_TRY_BOTH_DIRECTIONS=True.
    """
    ref_base = reference_signature.astype(np.float32)
    cur_base = current_signature.astype(np.float32)

    if MATCH_USE_SMOOTHED_SIGNATURE:
        ref_match = smooth_signature_circular(ref_base, MATCH_SMOOTH_WINDOW)
        cur_match = smooth_signature_circular(cur_base, MATCH_SMOOTH_WINDOW)
    else:
        ref_match = ref_base.copy()
        cur_match = cur_base.copy()

    ref_norm = normalize_signature(ref_match)
    cur_norm = normalize_signature(cur_match)

    curve_plus = _mse_curve_for_roll(ref_norm, cur_norm, direction=1)
    best_plus = int(np.argmin(curve_plus))
    mse_plus = float(curve_plus[best_plus])

    if MATCH_TRY_BOTH_DIRECTIONS:
        curve_minus = _mse_curve_for_roll(ref_norm, cur_norm, direction=-1)
        best_minus = int(np.argmin(curve_minus))
        mse_minus = float(curve_minus[best_minus])

        if mse_minus < mse_plus:
            best_shift = best_minus
            best_direction = -1
            mse_curve = curve_minus
            best_mse = mse_minus
        else:
            best_shift = best_plus
            best_direction = 1
            mse_curve = curve_plus
            best_mse = mse_plus
    else:
        best_shift = best_plus
        best_direction = 1
        mse_curve = curve_plus
        best_mse = mse_plus

    best_angle = (best_shift * angle_step_deg) % 360.0

    # Signature ROI sau khi can chinh theo chieu da chon.
    shifted_cur_raw = np.roll(cur_base, best_direction * best_shift)
    shifted_cur_match = np.roll(cur_match, best_direction * best_shift)
    shifted_cur_norm = np.roll(cur_norm, best_direction * best_shift)

    pointwise_sq_error = (ref_norm - shifted_cur_norm) ** 2

    return {
        "best_shift": best_shift,
        "best_angle": best_angle,
        "best_mse": best_mse,
        "best_direction": best_direction,
        "mse_curve": mse_curve,
        "ref_match": ref_match,
        "cur_match": cur_match,
        "ref_norm": ref_norm,
        "cur_norm": cur_norm,
        "shifted_cur_raw": shifted_cur_raw,
        "shifted_cur_match": shifted_cur_match,
        "shifted_cur_norm": shifted_cur_norm,
        "pointwise_sq_error": pointwise_sq_error,
    }


MIN_PARABOLIC_OFFSET = 0.1  # buoc (= do khi angle_step_deg=1); duoi nguong nay giu nguyen goc nguyen

def refine_angle_parabolic(mse_curve, best_index, angle_step_deg=1):
    n = len(mse_curve)
    prev_index = (best_index - 1) % n
    next_index = (best_index + 1) % n

    y_prev = float(mse_curve[prev_index])
    y_curr = float(mse_curve[best_index])
    y_next = float(mse_curve[next_index])
    denom = y_prev - 2.0 * y_curr + y_next

    # Parabol phai mo len (denom > 0) va best_index phai la cuc tieu that su
    if denom <= 1e-12 or y_prev <= y_curr or y_next <= y_curr:
        return (best_index * angle_step_deg) % 360.0, 0.0

    offset = 0.5 * (y_prev - y_next) / denom
    offset = float(np.clip(offset, -1.0, 1.0))

    # Chi ap dung sub-degree khi offset du lon, tranh nhieu lam sai goc nguyen
    if abs(offset) < MIN_PARABOLIC_OFFSET:
        return (best_index * angle_step_deg) % 360.0, 0.0

    refined_angle = ((best_index + offset) * angle_step_deg) % 360.0
    return refined_angle, offset


# =========================================================
# 6. VE ANH HIEN THI
# =========================================================

def get_overlay_style(image_shape):
    h, w = image_shape[:2]
    min_dim = float(min(h, w))

    # ROI nho khi phong to can marker/text nho de khong che anh.
    scale = float(np.clip(min_dim / 640.0, 0.24, 1.0))

    return {
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


def draw_radial_lines_on_roi(roi_gray, center, angles, points_filtered,
                             points_raw=None, outlier_mask=None,
                             display_ray_step=1):
    if len(roi_gray.shape) == 2:
        overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    else:
        overlay = roi_gray.copy()

    cx, cy = center
    style = get_overlay_style(overlay.shape)

    if SHOW_REMOVED_OUTLIERS and points_raw is not None and outlier_mask is not None:
        for i, is_outlier in enumerate(outlier_mask):
            if not is_outlier or points_raw[i] is None:
                continue
            if display_ray_step > 1 and (i % display_ray_step != 0):
                continue
            x_raw, y_raw = points_raw[i]
            cv2.line(overlay, (cx, cy), (x_raw, y_raw), (255, 0, 255),
                     style["raw_line_thickness"], lineType=style["line_type"])
            cv2.circle(overlay, (x_raw, y_raw), style["raw_point_radius"],
                       (255, 0, 255), -1, lineType=style["line_type"])

    for i, point in enumerate(points_filtered):
        if point is None:
            continue
        deg = int(angles[i])
        if deg != 0 and display_ray_step > 1 and (i % display_ray_step != 0):
            continue

        x, y = point
        if deg == 0:
            line_color = (0, 0, 255)
            point_color = (0, 0, 255)
            thickness = style["zero_line_thickness"]
        else:
            line_color = (0, 255, 0)
            point_color = (0, 255, 255)
            thickness = style["main_line_thickness"]

        cv2.line(overlay, (cx, cy), (x, y), line_color, thickness,
                 lineType=style["line_type"])
        cv2.circle(overlay, (x, y), style["point_radius"], point_color,
                   -1, lineType=style["line_type"])

    cv2.circle(overlay, (cx, cy), style["center_inner_radius"], (0, 0, 255),
               -1, lineType=style["line_type"])
    cv2.circle(overlay, (cx, cy), style["center_outer_radius"], (0, 0, 255),
               style["center_ring_thickness"], lineType=style["line_type"])

    cv2.putText(overlay, "Center",
                (cx + style["center_offset_x"], cy - style["center_offset_y"]),
                cv2.FONT_HERSHEY_SIMPLEX, style["center_font_scale"],
                (0, 0, 255), style["text_thickness"], lineType=style["line_type"])

    if len(points_filtered) > 0 and points_filtered[0] is not None:
        x0, y0 = points_filtered[0]
        cv2.putText(overlay, "0 deg", (x0 + style["zero_offset_x"], y0 - style["zero_offset_y"]),
                    cv2.FONT_HERSHEY_SIMPLEX, style["zero_font_scale"],
                    (0, 0, 255), style["text_thickness"], lineType=style["line_type"])

    return overlay


def draw_hough_circle_overlay(roi_gray, center, radius, hough_found):
    if len(roi_gray.shape) == 2:
        overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    else:
        overlay = roi_gray.copy()

    cx, cy = center
    style = get_overlay_style(overlay.shape)

    cv2.circle(overlay, (cx, cy), max(2, style["center_inner_radius"] - 1),
               (0, 0, 255), -1, lineType=style["line_type"])

    if radius > 0:
        color = (0, 255, 0) if hough_found else (0, 165, 255)
        cv2.circle(overlay, (cx, cy), int(radius), color,
                   style["center_ring_thickness"], lineType=style["line_type"])

    label = "HoughCircle" if hough_found else "Image center fallback"
    cv2.putText(overlay, label, (max(10, cx - 80), max(20, cy - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, style["center_font_scale"],
                (0, 255, 255), style["text_thickness"], lineType=style["line_type"])
    return overlay


def resize_for_display_with_meta(image_bgr, max_width=850, max_height=850, allow_upscale=False):
    h, w = image_bgr.shape[:2]
    if max_width <= 0 or max_height <= 0:
        return image_bgr.copy(), 1.0

    scale = min(max_width / float(w), max_height / float(h))
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


def render_result_overlay_for_display(result, target_width, target_height):
    """Resize anh truoc, scale toa do roi moi ve radial de tia hien thi dep."""
    roi_gray = result["gray"]
    base_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    resized_bgr, scale = resize_for_display_with_meta(
        base_bgr,
        target_width,
        target_height,
        allow_upscale=True
    )

    scaled_center = scale_point(result["center"], scale)
    scaled_points_filtered = scale_points(result["points_filtered"], scale)
    scaled_points_raw = scale_points(result["points_raw"], scale)
    resized_gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    return draw_radial_lines_on_roi(
        roi_gray=resized_gray,
        center=scaled_center,
        angles=result["angles"],
        points_filtered=scaled_points_filtered,
        points_raw=scaled_points_raw,
        outlier_mask=result["outlier_mask"],
        display_ray_step=DISPLAY_RAY_STEP
    )


def cv_to_tk_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    return ImageTk.PhotoImage(image_pil)


# =========================================================
# 7. PIPELINE XU LY 1 ANH
# =========================================================

def process_single_image(image_path):
    image = read_image(image_path, grayscale=False)
    if image is None:
        raise ValueError("Khong doc duoc anh: {0}".format(image_path))

    gray, enhanced, blur, canny = preprocess_roi(image)
    center, estimated_radius, hough_found = find_center_by_hough_or_image_center(blur)

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

    radial_overlay_full = draw_radial_lines_on_roi(
        roi_gray=gray,
        center=center,
        angles=angles,
        points_filtered=points_filtered,
        points_raw=points_raw,
        outlier_mask=outlier_mask,
        display_ray_step=1
    )

    hough_overlay = draw_hough_circle_overlay(gray, center, estimated_radius, hough_found)

    return {
        "path": image_path,
        "gray": gray,
        "enhanced": enhanced,
        "blur": blur,
        "canny": canny,
        "center": center,
        "estimated_radius": estimated_radius,
        "hough_found": hough_found,
        "angles": angles,
        "signature_raw": signature_raw,
        "signature_interp": signature_interp,
        "signature_filtered": signature_filtered,
        "signature_match": smooth_signature_circular(signature_filtered, MATCH_SMOOTH_WINDOW) if MATCH_USE_SMOOTHED_SIGNATURE else signature_filtered.copy(),
        "points_raw": points_raw,
        "points_interp": points_interp,
        "points_filtered": points_filtered,
        "valid_count": valid_count,
        "r_max": r_max,
        "outlier_mask": outlier_mask,
        "local_median": local_median,
        "global_upper": global_upper,
        "radial_overlay": radial_overlay_full,
        "hough_overlay": hough_overlay,
    }


# =========================================================
# 8. GUI
# =========================================================

class RadialSignatureMatchGUI(object):
    def __init__(self, root):
        self.root = root
        self.root.title("So khop Radial Signature - xac dinh goc xoay stator")
        self.root.geometry("1680x980")
        self.root.minsize(1300, 760)

        self.reference_path = None
        self.current_path = None
        self.reference_result = None
        self.current_result = None
        self.match_result = None

        self.reference_image_panel = None
        self.current_image_panel = None
        self.reference_plot_panel = None
        self.current_plot_panel = None
        self.match_plot_panel = None

        self.popup_windows = []
        self.tree_columns = ("angle", "r_ref", "r_roi", "ref_norm", "roi_norm", "sq_err")

        self.build_gui()

    # ------------------------- GUI layout -------------------------
    def build_gui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        tk.Button(top_frame, text="Chon anh mau", font=("Arial", 11), command=self.choose_reference_image).pack(side=tk.LEFT, padx=4)
        tk.Button(top_frame, text="Chon anh ROI", font=("Arial", 11), command=self.choose_current_image).pack(side=tk.LEFT, padx=4)
        tk.Button(top_frame, text="Chay xu ly", font=("Arial", 11), command=self.process_images).pack(side=tk.LEFT, padx=4)
        tk.Button(top_frame, text="Luu ket qua", font=("Arial", 11), command=self.save_result).pack(side=tk.LEFT, padx=4)

        self.reference_path_label = tk.Label(top_frame, text="Anh mau: chua chon", anchor="w", font=("Arial", 10))
        self.reference_path_label.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

        self.current_path_label = tk.Label(top_frame, text="Anh ROI: chua chon", anchor="w", font=("Arial", 10))
        self.current_path_label.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.main_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        compare_frame = ttk.Frame(self.main_paned)
        result_side_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(compare_frame, weight=7)
        self.main_paned.add(result_side_frame, weight=4)

        self.compare_paned = ttk.PanedWindow(compare_frame, orient=tk.HORIZONTAL)
        self.compare_paned.pack(fill=tk.BOTH, expand=True)

        ref_col = ttk.Frame(self.compare_paned)
        cur_col = ttk.Frame(self.compare_paned)
        self.compare_paned.add(ref_col, weight=1)
        self.compare_paned.add(cur_col, weight=1)

        self.reference_image_panel = self.build_image_section(ref_col, "Anh mau + Radial Signature")
        self.reference_plot_panel = self.build_plot_section(ref_col, "Do thi Radial Signature mau", "Phong to do thi mau", self.open_reference_plot_popup)
        self.current_image_panel = self.build_image_section(cur_col, "Anh ROI + Radial Signature")
        self.current_plot_panel = self.build_plot_section(cur_col, "Do thi Radial Signature ROI", "Phong to do thi ROI", self.open_current_plot_popup)

        self.right_paned = ttk.PanedWindow(result_side_frame, orient=tk.VERTICAL)
        self.right_paned.pack(fill=tk.BOTH, expand=True)

        result_frame = ttk.Frame(self.right_paned)
        mse_frame = ttk.Frame(self.right_paned)
        table_frame = ttk.Frame(self.right_paned)
        self.right_paned.add(result_frame, weight=2)
        self.right_paned.add(mse_frame, weight=3)
        self.right_paned.add(table_frame, weight=5)

        self.build_right_section_header(result_frame, "Ket qua so khop", "Phong to", self.open_result_popup)
        result_body = tk.Frame(result_frame)
        result_body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.info_text = ScrolledText(result_body, width=48, height=16, font=("Consolas", 10), wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        self.build_right_section_header(mse_frame, "MSE theo goc dich", "Phong to MSE", self.open_mse_popup)
        mse_body = tk.Frame(mse_frame)
        mse_body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.mse_figure = Figure(figsize=(5.8, 2.8), dpi=100)
        self.mse_axes = self.mse_figure.add_subplot(111)
        self.mse_canvas = FigureCanvasTkAgg(self.mse_figure, master=mse_body)
        self.mse_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.enable_ctrl_mousewheel_zoom(self.mse_axes, self.mse_canvas)
        self.draw_empty_mse_plot()

        self.build_right_section_header(table_frame, "Bang gia tri theo goc", "Phong to bang", self.open_table_popup)
        table_body = tk.Frame(table_frame)
        table_body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.tree = ttk.Treeview(table_body, columns=self.tree_columns, show="headings", height=26)
        self.configure_tree_columns(self.tree)
        scrollbar_y = ttk.Scrollbar(table_body, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(table_body, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        table_body.grid_rowconfigure(0, weight=1)
        table_body.grid_columnconfigure(0, weight=1)

    def build_image_section(self, parent, title):
        frame = tk.LabelFrame(parent, text=title, font=("Arial", 11))
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 8))
        frame.configure(height=DISPLAY_MAX_HEIGHT + 40)
        frame.pack_propagate(False)

        label = tk.Label(frame, bg="black", relief=tk.SUNKEN, anchor="center")
        label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        panel = {"frame": frame, "label": label, "result": None, "preview_bgr": None, "resize_job": None}
        label.bind("<Configure>", lambda event, p=panel: self.schedule_panel_render(p))
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
        self.draw_empty_radial_plot(panel, title)
        return panel

    def build_right_section_header(self, parent, title, button_text, button_command):
        header = tk.Frame(parent)
        header.pack(fill=tk.X, padx=6, pady=(6, 4))
        tk.Label(header, text=title, font=("Arial", 11, "bold"), anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(header, text=button_text, font=("Arial", 9), command=button_command).pack(side=tk.RIGHT)

    # ------------------------- Hien thi anh -------------------------
    def get_panel_target_size(self, image_panel):
        label = image_panel["label"]
        label.update_idletasks()
        width = label.winfo_width()
        height = label.winfo_height()
        if width <= 1:
            width = DISPLAY_MAX_WIDTH
        if height <= 1:
            height = DISPLAY_MAX_HEIGHT
        return max(1, width - 8), max(1, height - 8)

    def schedule_panel_render(self, image_panel):
        previous = image_panel.get("resize_job")
        if previous is not None:
            try:
                self.root.after_cancel(previous)
            except Exception:
                pass
        image_panel["resize_job"] = self.root.after(60, lambda p=image_panel: self._finish_panel_render(p))

    def _finish_panel_render(self, image_panel):
        image_panel["resize_job"] = None
        self.render_image_panel(image_panel)

    def render_image_panel(self, image_panel):
        target_w, target_h = self.get_panel_target_size(image_panel)

        if image_panel.get("result") is not None:
            display_bgr = render_result_overlay_for_display(image_panel["result"], target_w, target_h)
        elif image_panel.get("preview_bgr") is not None:
            display_bgr, _ = resize_for_display_with_meta(image_panel["preview_bgr"], target_w, target_h, allow_upscale=True)
        else:
            return

        image_tk = cv_to_tk_image(display_bgr)
        image_panel["label"].config(image=image_tk)
        image_panel["label"].image = image_tk
        image_panel["image_tk"] = image_tk

    def preview_original_image(self, file_path, image_panel, plot_panel, title_plot):
        image = read_image(file_path, grayscale=False)
        if image is None:
            messagebox.showerror("Loi", "Khong doc duoc anh: {0}".format(file_path))
            return
        gray = prepare_gray_roi(image)
        preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        image_panel["preview_bgr"] = preview
        image_panel["result"] = None
        self.render_image_panel(image_panel)
        self.draw_empty_radial_plot(plot_panel, title_plot)

    # ------------------------- Plot -------------------------
    def draw_empty_radial_plot(self, plot_panel, title):
        ax = plot_panel["axes"]
        fig = plot_panel["figure"]
        ax.clear()
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Goc (deg)", fontsize=8)
        ax.set_ylabel("R", fontsize=8)
        ax.set_xlim(0, 359)
        ax.text(0.5, 0.5, "Chua co Radial Signature", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=8)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)
        plot_panel["canvas"].draw_idle()

    def draw_radial_plot_on_axes(self, axes, angles, signature, title, signature_smooth=None):
        axes.clear()
        axes.plot(angles, signature, color="#0b5cff", linewidth=1.2, label="R filtered")
        if signature_smooth is not None:
            axes.plot(angles, signature_smooth, color="#ff7f0e", linewidth=1.2, label="R match smooth")
            axes.legend(loc="best", fontsize=8)
        axes.set_title(title, fontsize=9)
        axes.set_xlabel("Goc (deg)", fontsize=8)
        axes.set_ylabel("R", fontsize=8)
        axes.set_xlim(0, 359)
        axes.margins(x=0.0, y=0.08)
        axes.grid(True, alpha=0.25)
        axes.tick_params(axis="both", labelsize=8)

    def refresh_radial_plot(self, plot_panel, angles, signature, title, signature_smooth=None):
        self.draw_radial_plot_on_axes(plot_panel["axes"], angles, signature, title, signature_smooth)
        plot_panel["figure"].subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)
        plot_panel["canvas"].draw_idle()

    def draw_empty_mse_plot(self):
        ax = self.mse_axes
        ax.clear()
        ax.set_title("MSE theo goc dich")
        ax.set_xlabel("Goc dich (deg)")
        ax.set_ylabel("MSE")
        ax.text(0.5, 0.5, "Chua co du lieu MSE", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, alpha=0.25)
        self.mse_figure.tight_layout()
        self.mse_canvas.draw_idle()

    def draw_mse_plot_on_axes(self, axes, mse_curve, best_shift, refined_angle, best_mse, direction):
        axes.clear()
        x_values = np.arange(len(mse_curve), dtype=np.float32)
        axes.plot(x_values, mse_curve, color="#1f77b4", linewidth=1.6, label="MSE")
        axes.axvline(best_shift, color="red", linestyle="--", linewidth=1.2, label="Best shift = {0}".format(best_shift))
        axes.scatter([best_shift], [best_mse], color="red", s=34, zorder=3)
        dir_text = "+roll" if direction == 1 else "-roll"
        axes.set_title("MSE theo goc dich | refined = {0:.2f} deg | {1}".format(refined_angle, dir_text))
        axes.set_xlabel("Goc dich (deg)")
        axes.set_ylabel("MSE")
        axes.set_xlim(0, max(359, len(mse_curve) - 1))
        axes.margins(x=0.0, y=0.08)
        axes.grid(True, alpha=0.3)
        axes.legend(loc="best")

    def refresh_mse_plot(self):
        if self.match_result is None:
            self.draw_empty_mse_plot()
            return
        self.draw_mse_plot_on_axes(
            self.mse_axes,
            self.match_result["mse_curve"],
            self.match_result["best_shift"],
            self.match_result["refined_angle"],
            self.match_result["best_mse"],
            self.match_result["best_direction"]
        )
        self.mse_figure.tight_layout()
        self.mse_canvas.draw_idle()

    def enable_ctrl_mousewheel_zoom(self, axes, canvas):
        def on_scroll(event):
            if event.inaxes != axes:
                return
            key = str(event.key).lower() if event.key is not None else ""
            if "control" not in key and "ctrl" not in key:
                return
            scale_factor = 0.9 if event.button == "up" else 1.1
            x_min, x_max = axes.get_xlim()
            y_min, y_max = axes.get_ylim()
            x_center = event.xdata if event.xdata is not None else (x_min + x_max) / 2.0
            y_center = event.ydata if event.ydata is not None else (y_min + y_max) / 2.0
            new_x_min = x_center - (x_center - x_min) * scale_factor
            new_x_max = x_center + (x_max - x_center) * scale_factor
            new_y_min = y_center - (y_center - y_min) * scale_factor
            new_y_max = y_center + (y_max - y_center) * scale_factor
            if abs(new_x_max - new_x_min) < 1e-6 or abs(new_y_max - new_y_min) < 1e-12:
                return
            axes.set_xlim(new_x_min, new_x_max)
            axes.set_ylim(new_y_min, new_y_max)
            canvas.draw_idle()
        canvas.mpl_connect("scroll_event", on_scroll)

    # ------------------------- Table -------------------------
    def configure_tree_columns(self, tree_widget):
        tree_widget.heading("angle", text="Goc")
        tree_widget.heading("r_ref", text="R ref")
        tree_widget.heading("r_roi", text="R roi shifted")
        tree_widget.heading("ref_norm", text="Ref norm")
        tree_widget.heading("roi_norm", text="ROI norm shifted")
        tree_widget.heading("sq_err", text="Sq err")
        tree_widget.column("angle", width=55, anchor="center", stretch=False)
        tree_widget.column("r_ref", width=88, anchor="center", stretch=False)
        tree_widget.column("r_roi", width=100, anchor="center", stretch=False)
        tree_widget.column("ref_norm", width=88, anchor="center", stretch=False)
        tree_widget.column("roi_norm", width=115, anchor="center", stretch=False)
        tree_widget.column("sq_err", width=88, anchor="center", stretch=False)

    def populate_tree_widget(self, tree_widget):
        for item in tree_widget.get_children():
            tree_widget.delete(item)

        if self.reference_result is None or self.match_result is None:
            return

        ref_sig = self.reference_result["signature_filtered"]
        roi_sig = self.match_result["shifted_cur_raw"]
        ref_norm = self.match_result["ref_norm"]
        roi_norm = self.match_result["shifted_cur_norm"]
        sq_err = self.match_result["pointwise_sq_error"]

        for i, deg in enumerate(self.reference_result["angles"]):
            tree_widget.insert(
                "", tk.END,
                values=(
                    "{0}".format(int(deg)),
                    "{0:.2f}".format(ref_sig[i]),
                    "{0:.2f}".format(roi_sig[i]),
                    "{0:.3f}".format(ref_norm[i]),
                    "{0:.3f}".format(roi_norm[i]),
                    "{0:.4f}".format(sq_err[i])
                )
            )

    def update_table(self):
        self.populate_tree_widget(self.tree)

    # ------------------------- Commands -------------------------
    def choose_reference_image(self):
        file_path = filedialog.askopenfilename(
            title="Chon anh mau",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if not file_path:
            return
        self.reference_path = file_path
        self.reference_path_label.config(text="Anh mau: {0}".format(file_path))
        self.preview_original_image(file_path, self.reference_image_panel, self.reference_plot_panel, "Radial Signature mau")

    def choose_current_image(self):
        file_path = filedialog.askopenfilename(
            title="Chon anh ROI hien tai",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if not file_path:
            return
        self.current_path = file_path
        self.current_path_label.config(text="Anh ROI: {0}".format(file_path))
        self.preview_original_image(file_path, self.current_image_panel, self.current_plot_panel, "Radial Signature ROI")

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

            match = match_radial_signature(
                self.reference_result["signature_filtered"],
                self.current_result["signature_filtered"],
                angle_step_deg=ANGLE_STEP_DEG
            )
            refined_angle, parabolic_offset = refine_angle_parabolic(
                match["mse_curve"],
                match["best_shift"],
                angle_step_deg=ANGLE_STEP_DEG
            )
            match["refined_angle"] = refined_angle
            match["parabolic_offset"] = parabolic_offset
            self.match_result = match

            self.update_visuals()
            self.update_info_text()
            self.update_table()

        except Exception as exc:
            messagebox.showerror("Loi xu ly", str(exc))

    def update_visuals(self):
        self.reference_image_panel["result"] = self.reference_result
        self.reference_image_panel["preview_bgr"] = None
        self.current_image_panel["result"] = self.current_result
        self.current_image_panel["preview_bgr"] = None
        self.render_image_panel(self.reference_image_panel)
        self.render_image_panel(self.current_image_panel)

        self.refresh_radial_plot(
            self.reference_plot_panel,
            self.reference_result["angles"],
            self.reference_result["signature_filtered"],
            "Radial Signature mau",
            self.reference_result["signature_match"] if MATCH_USE_SMOOTHED_SIGNATURE else None
        )
        self.refresh_radial_plot(
            self.current_plot_panel,
            self.current_result["angles"],
            self.current_result["signature_filtered"],
            "Radial Signature ROI",
            self.current_result["signature_match"] if MATCH_USE_SMOOTHED_SIGNATURE else None
        )
        self.refresh_mse_plot()

    def update_info_text(self):
        ref = self.reference_result
        cur = self.current_result
        match = self.match_result
        dir_text = "+roll" if match["best_direction"] == 1 else "-roll"
        inverse_angle = (360.0 - match["refined_angle"]) % 360.0

        lines = [
            "KET QUA SO KHOP RADIAL SIGNATURE",
            "",
            "Goc lech tho: {0:.1f} do".format(match["best_angle"]),
            "Goc lech sau tinh chinh: {0:.2f} do".format(match["refined_angle"]),
            "Goc dao chieu quy uoc: {0:.2f} do".format(inverse_angle),
            "MSE nho nhat: {0:.6f}".format(match["best_mse"]),
            "Offset parabol: {0:.4f} buoc".format(match["parabolic_offset"]),
            "Chieu roll duoc chon: {0}".format(dir_text),
            "Signature matching: {0}, window = {1}".format(
                "smoothed" if MATCH_USE_SMOOTHED_SIGNATURE else "filtered raw",
                MATCH_SMOOTH_WINDOW if MATCH_USE_SMOOTHED_SIGNATURE else 0
            ),
            "",
            "Anh mau:",
            "  Duong dan: {0}".format(ref["path"]),
            "  Tam = {0}".format(ref["center"]),
            "  R Hough = {0}".format(ref["estimated_radius"]),
            "  Hough found = {0}".format(ref["hough_found"]),
            "  So tia hop le = {0}/{1}".format(ref["valid_count"], len(ref["angles"])),
            "  r_max quet = {0}".format(ref["r_max"]),
            "  Canny = ({0}, {1})".format(CANNY_LOW, CANNY_HIGH),
            "",
            "Anh ROI:",
            "  Duong dan: {0}".format(cur["path"]),
            "  Tam = {0}".format(cur["center"]),
            "  R Hough = {0}".format(cur["estimated_radius"]),
            "  Hough found = {0}".format(cur["hough_found"]),
            "  So tia hop le = {0}/{1}".format(cur["valid_count"], len(cur["angles"])),
            "  r_max quet = {0}".format(cur["r_max"]),
            "  Canny = ({0}, {1})".format(CANNY_LOW, CANNY_HIGH),
            "",
            "Ghi chu:",
            "  Bang hien thi ROI sau khi da dich theo best_shift.",
            "  Neu quy uoc goc bi nguoc, dung gia tri 'Goc dao chieu quy uoc'.",
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
        text_widget = tk.Text(frame, font=("Consolas", 11), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.insert("1.0", self.get_result_text())
        self.popup_windows.append(popup)

    def open_mse_popup(self):
        if self.match_result is None:
            messagebox.showwarning("Canh bao", "Chua co du lieu MSE de phong to.")
            return
        popup = tk.Toplevel(self.root)
        popup.title("Do thi MSE theo goc dich")
        popup.geometry("1200x780")
        frame = tk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        self.draw_mse_plot_on_axes(ax, self.match_result["mse_curve"], self.match_result["best_shift"],
                                   self.match_result["refined_angle"], self.match_result["best_mse"],
                                   self.match_result["best_direction"])
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.enable_ctrl_mousewheel_zoom(ax, canvas)
        canvas.draw_idle()
        self.popup_windows.append(popup)

    def open_reference_plot_popup(self):
        if self.reference_result is None:
            messagebox.showwarning("Canh bao", "Chua co do thi mau de phong to.")
            return
        self.open_radial_plot_popup("Do thi Radial Signature mau", "Radial Signature mau",
                                    self.reference_result["angles"], self.reference_result["signature_filtered"],
                                    self.reference_result["signature_match"] if MATCH_USE_SMOOTHED_SIGNATURE else None)

    def open_current_plot_popup(self):
        if self.current_result is None:
            messagebox.showwarning("Canh bao", "Chua co do thi ROI de phong to.")
            return
        self.open_radial_plot_popup("Do thi Radial Signature ROI", "Radial Signature ROI",
                                    self.current_result["angles"], self.current_result["signature_filtered"],
                                    self.current_result["signature_match"] if MATCH_USE_SMOOTHED_SIGNATURE else None)

    def open_radial_plot_popup(self, title_window, title_plot, angles, signature, signature_smooth=None):
        popup = tk.Toplevel(self.root)
        popup.title(title_window)
        popup.geometry("1200x760")
        frame = tk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        fig = Figure(figsize=(10, 5.8), dpi=100)
        ax = fig.add_subplot(111)
        self.draw_radial_plot_on_axes(ax, angles, signature, title_plot, signature_smooth)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12)
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
        popup.geometry("1200x760")
        frame = tk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        tree = ttk.Treeview(frame, columns=self.tree_columns, show="headings")
        self.configure_tree_columns(tree)
        scrollbar_y = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        self.populate_tree_widget(tree)
        self.popup_windows.append(popup)

    # ------------------------- Save -------------------------
    def save_result(self):
        if self.reference_result is None or self.current_result is None or self.match_result is None:
            messagebox.showwarning("Canh bao", "Chua co ket qua de luu.")
            return
        save_path = filedialog.asksaveasfilename(
            title="Luu ket qua",
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv"), ("Text file", "*.txt"), ("All files", "*.*")]
        )
        if not save_path:
            return
        try:
            ext = os.path.splitext(save_path)[1].lower()
            if ext == ".txt":
                self.save_result_txt(save_path)
            else:
                self.save_result_csv(save_path)
            messagebox.showinfo("Thanh cong", "Da luu ket qua:\n{0}".format(save_path))
        except Exception as exc:
            messagebox.showerror("Loi", str(exc))

    def save_result_txt(self, save_path):
        with open(save_path, "w", encoding="utf-8-sig") as f:
            f.write(self.get_result_text())

    def save_result_csv(self, save_path):
        ref = self.reference_result
        cur = self.current_result
        match = self.match_result
        with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["reference_image_path", ref["path"]])
            writer.writerow(["current_roi_image_path", cur["path"]])
            writer.writerow(["center_ref_x", ref["center"][0]])
            writer.writerow(["center_ref_y", ref["center"][1]])
            writer.writerow(["radius_ref", ref["estimated_radius"]])
            writer.writerow(["center_roi_x", cur["center"][0]])
            writer.writerow(["center_roi_y", cur["center"][1]])
            writer.writerow(["radius_roi", cur["estimated_radius"]])
            writer.writerow(["angle_raw_deg", "{0:.6f}".format(match["best_angle"])])
            writer.writerow(["angle_refined_deg", "{0:.6f}".format(match["refined_angle"])])
            writer.writerow(["best_mse", "{0:.9f}".format(match["best_mse"])])
            writer.writerow(["best_direction", match["best_direction"]])
            writer.writerow(["match_smooth_window", MATCH_SMOOTH_WINDOW if MATCH_USE_SMOOTHED_SIGNATURE else 0])
            writer.writerow(["valid_ray_ref", ref["valid_count"]])
            writer.writerow(["valid_ray_roi", cur["valid_count"]])
            writer.writerow([])
            writer.writerow(["angle_deg", "R_ref", "R_roi_shifted", "R_ref_norm", "R_roi_norm_shifted", "Sq_err"])
            for i, deg in enumerate(ref["angles"]):
                writer.writerow([
                    int(deg),
                    "{0:.6f}".format(ref["signature_filtered"][i]),
                    "{0:.6f}".format(match["shifted_cur_raw"][i]),
                    "{0:.6f}".format(match["ref_norm"][i]),
                    "{0:.6f}".format(match["shifted_cur_norm"][i]),
                    "{0:.9f}".format(match["pointwise_sq_error"][i]),
                ])


if __name__ == "__main__":
    root = tk.Tk()
    app = RadialSignatureMatchGUI(root)
    root.mainloop()

