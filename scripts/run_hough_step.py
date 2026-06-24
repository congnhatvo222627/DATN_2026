"""Standalone test runner for step 1: Hough."""

from _common import (
    bootstrap,
    get_first_tray_image,
    load_standard_presets,
    print_logs,
    print_missing,
    run_step_hough,
    save_result_images,
)


def main():
    bootstrap()
    image_path = get_first_tray_image()
    if image_path is None:
        print_missing(
            [
                "Chua co anh trong data/input/.",
                "Hay dat anh khay vao data/input/ roi chay lai:",
                "python scripts/run_hough_step.py",
            ]
        )
        return

    presets = load_standard_presets()
    result = run_step_hough(str(image_path), presets["hough"])
    print_logs(result)
    if not result["success"]:
        print("Buoc Hough that bai.")
        return

    save_result_images("run_hough_step", result)
    print("So circle sau loc:", len(result["data"]["circles_filtered"]))


if __name__ == "__main__":
    main()
