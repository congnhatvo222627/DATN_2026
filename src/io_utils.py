"""File and image IO helpers."""

from pathlib import Path

import cv2
import numpy as np

from .config import INPUT_DIR, OUTPUT_DIR, PRESET_DIR, ROI_DIR, TEMPLATE_DIR


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path):
    """Create a directory if missing and return it as Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def ensure_project_dirs():
    """Ensure the standard project folders exist."""
    for path in [INPUT_DIR, TEMPLATE_DIR, ROI_DIR, OUTPUT_DIR, PRESET_DIR]:
        ensure_dir(path)


def read_image(path, grayscale=False):
    """Read an image with Unicode-path support."""
    path_obj = Path(path)
    if not path_obj.is_file():
        return None
    data = np.fromfile(str(path_obj), dtype=np.uint8)
    if data.size == 0:
        return None
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imdecode(data, flag)


def write_image(path, image):
    """Write an image with Unicode-path support."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    ext = path_obj.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError("Khong the ghi anh: {}".format(path_obj))
    encoded.tofile(str(path_obj))
    return path_obj


def list_images(folder):
    """Return sorted image paths in a folder."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return []
    return sorted(
        [path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )


def get_first_image(folder):
    """Return the first image path in a folder or None."""
    images = list_images(folder)
    return images[0] if images else None

