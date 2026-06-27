"""Hieu chuan toa do anh -> toa do thuc bang homography.

Nhap toi thieu 4 cap diem (image_x, image_y) <-> (real_x, real_y), uoc luong ma
tran homography 3x3 bang cv2.findHomography, sau do dung de bien doi toa do tam
stator tu pixel anh sang toa do thuc khi truyen thong cho thiet bi khac.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .io_utils import ensure_dir

# So cap diem toi thieu de uoc luong homography (4 cap = 8 phuong trinh).
MIN_CALIBRATION_POINTS = 4

POINT_KEYS = ("image_x", "image_y", "real_x", "real_y")


def validate_point_pairs(point_pairs):
    """Kiem tra danh sach cap diem, raise ValueError neu khong hop le.

    Moi phan tu la dict co du 4 khoa image_x/image_y/real_x/real_y dang so.
    """
    if len(point_pairs) < MIN_CALIBRATION_POINTS:
        raise ValueError(
            "Can it nhat {} cap diem hieu chuan (hien co {}).".format(
                MIN_CALIBRATION_POINTS, len(point_pairs)
            )
        )
    for index, pair in enumerate(point_pairs, start=1):
        for key in POINT_KEYS:
            value = pair.get(key)
            if value is None or value == "":
                raise ValueError("Cap diem {} thieu gia tri '{}'.".format(index, key))
            try:
                float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    "Cap diem {} co gia tri '{}' khong phai so: {!r}.".format(index, key, value)
                )


def compute_homography(point_pairs):
    """Tinh ma tran homography 3x3 (numpy) tu cac cap diem hieu chuan.

    Raise ValueError neu thieu diem hoac cau hinh suy bien.
    """
    validate_point_pairs(point_pairs)
    src = np.array(
        [[float(pair["image_x"]), float(pair["image_y"])] for pair in point_pairs],
        dtype=np.float64,
    )
    dst = np.array(
        [[float(pair["real_x"]), float(pair["real_y"])] for pair in point_pairs],
        dtype=np.float64,
    )
    matrix, _mask = cv2.findHomography(src, dst, method=0)
    if matrix is None:
        raise ValueError("Khong tinh duoc homography (cac diem co the suy bien/thang hang).")
    return matrix


def transform_point(matrix, x, y):
    """Bien doi mot diem anh (x, y) sang toa do thuc bang ma tran homography."""
    point = np.array([float(x), float(y), 1.0], dtype=np.float64)
    mapped = np.asarray(matrix, dtype=np.float64) @ point
    if abs(mapped[2]) < 1e-12:
        raise ValueError("Homography suy bien khi bien doi diem.")
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def save_calibration(path, point_pairs):
    """Luu danh sach cap diem hieu chuan ra JSON."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    payload = {
        "point_pairs": [
            {key: float(pair[key]) for key in POINT_KEYS} for pair in point_pairs
        ]
    }
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path_obj


def load_calibration(path):
    """Doc danh sach cap diem hieu chuan tu JSON. Tra ve [] neu chua co file."""
    path_obj = Path(path)
    if not path_obj.is_file():
        return []
    with path_obj.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        pairs = payload.get("point_pairs", [])
    elif isinstance(payload, list):
        pairs = payload
    else:
        pairs = []
    return [{key: pair.get(key) for key in POINT_KEYS} for pair in pairs]
