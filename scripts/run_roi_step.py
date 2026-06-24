"""Standalone test runner for step 2: ROI extraction."""

from _common import (
    bootstrap,
    get_first_tray_image,
    load_standard_presets,
    print_logs,
    print_missing,
    run_step_hough,
    run_step_roi,
    save_result_images,
    save_rois_if_success,
)


def main():
    bootstrap()
    image_path = get_first_tray_image()
    if image_path is None:
        print_missing(
            [
                "Chua co anh trong data/input/.",
                "Hay dat anh khay vao data/input/ roi chay lai:",
                "python scripts/run_roi_step.py",
            ]
        )
        return

    presets = load_standard_presets()
    hough_result = run_step_hough(str(image_path), presets["hough"])
    if not hough_result["success"]:
        print_logs(hough_result)
        print("Khong the chay ROI vi Hough that bai.")
        return

    roi_result = run_step_roi(
        hough_result["images"]["original"],
        hough_result["data"]["circles_filtered"],
        presets["roi"],
    )
    print_logs(roi_result)
    if not roi_result["success"]:
        print("Buoc ROI that bai.")
        return

    save_result_images("run_roi_step", roi_result)
    save_rois_if_success(roi_result)
    print("So ROI da cat:", len(roi_result["data"]["rois"]))


if __name__ == "__main__":
    main()
