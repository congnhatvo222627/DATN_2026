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

_RING_OFFSET_CACHE = {}


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

    cache_key = (radius, int(ring_width))
    offsets = _RING_OFFSET_CACHE.get(cache_key)
    if offsets is None:
        ring_offsets = []
        inner_r = max(1, radius - int(ring_width))
        outer_r = radius + int(ring_width)

        for ring_r in range(inner_r, outer_r + 1):
            num_samples = max(96, int(round(2.0 * math.pi * ring_r)))
            angles = np.linspace(0.0, 2.0 * math.pi, num_samples, endpoint=False)
            xs = np.rint(ring_r * np.cos(angles)).astype(np.int32)
            ys = np.rint(ring_r * np.sin(angles)).astype(np.int32)
            ring_offsets.append(np.stack((xs, ys), axis=1))

        offsets = np.unique(np.concatenate(ring_offsets, axis=0), axis=0)
        _RING_OFFSET_CACHE[cache_key] = offsets

    sample_x = cx + offsets[:, 0]
    sample_y = cy + offsets[:, 1]
    valid = (
        (sample_x >= 0)
        & (sample_x < w)
        & (sample_y >= 0)
        & (sample_y < h)
    )

    valid_count = int(valid.sum())
    if valid_count == 0:
        return 0.0

    return float((edges[sample_y[valid], sample_x[valid]] > 0).sum()) / float(valid_count)


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


def process_image(input_path, output_dir=None, save_outputs=True, build_visuals=True):
    if output_dir is None:
        output_dir = CONFIG["output_dir"]

    ensure_dir(output_dir)
    build_visuals = bool(build_visuals or save_outputs)

    image = read_image(input_path)
    if image is None:
        raise IOError("Khong doc duoc anh: {}".format(input_path))

    roi, offset_xy = crop_work_roi(image)
    gray, edges, raw_candidates, circles, common_radius = detect_stator_centers(roi)

    if not circles:
        raise RuntimeError("Khong tim thay stator nao bang Hough Circle.")

    raw_result = None
    result = None
    valid_overview = None

    if build_visuals:
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
