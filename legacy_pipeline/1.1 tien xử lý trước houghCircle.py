# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import subprocess
import sys


SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

RUNTIME_CANDIDATES = [
    os.path.join(ROOT_DIR, ".venv311", "Scripts", "python.exe"),
    r"C:\Program Files\OPT\DeepVision3\python.exe",
]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "pipeline", "1_1_pre_hough_circle")


def choose_runtime():
    for candidate in RUNTIME_CANDIDATES:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def maybe_relaunch_with_supported_python():
    target_python = choose_runtime()
    if not target_python:
        return

    current_python = os.path.normcase(os.path.abspath(sys.executable))
    target_python = os.path.normcase(target_python)

    if current_python == target_python:
        return

    subprocess.check_call([target_python, SCRIPT_PATH] + sys.argv[1:])
    raise SystemExit(0)


maybe_relaunch_with_supported_python()

import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_default_image():
    candidate_dirs = [
        os.path.join(ROOT_DIR, "data", "test"),
        os.path.join(ROOT_DIR, "data", "samples"),
    ]

    for folder in candidate_dirs:
        if not os.path.isdir(folder):
            continue

        for file_name in sorted(os.listdir(folder)):
            path = os.path.join(folder, file_name)
            if os.path.isfile(path) and path.lower().endswith(IMAGE_EXTENSIONS):
                return os.path.abspath(path)

    raise FileNotFoundError(
        "Khong tim thay anh mac dinh trong data/test hoac data/samples."
    )


def resolve_image_path():
    if len(sys.argv) > 1:
        return os.path.abspath(os.path.expanduser(sys.argv[1]))
    return find_default_image()


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    image_path = resolve_image_path()
    if not os.path.exists(image_path):
        raise FileNotFoundError("Khong tim thay anh dau vao: {}".format(image_path))

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_name = "{}_clahe_gaussian.png".format(
        os.path.splitext(os.path.basename(image_path))[0]
    )
    output_path = os.path.join(OUTPUT_DIR, output_name)

    img = read_image(image_path)
    if img is None:
        raise FileNotFoundError("Khong doc duoc anh: {}".format(image_path))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8),
    )
    clahe_img = clahe.apply(gray)
    clahe_gaussian = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    images = [gray, clahe_img, clahe_gaussian]
    titles = [
        "(a) Anh goc muc xam",
        "(b) Sau CLAHE",
        "(c) CLAHE + Gaussian Blur",
    ]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    print("Da dung anh dau vao:", image_path)
    print("Da luu hinh tai:", output_path)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
