"""Standalone test runner for step 7: full pipeline."""

from _common import (
    bootstrap,
    get_first_tray_image,
    get_template_data_or_help,
    load_standard_presets,
    print_logs,
    print_missing,
    run_full_pipeline,
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
                "python scripts/run_full_pipeline.py",
            ]
        )
        return

    template_data, template_messages = get_template_data_or_help()
    if template_data is None:
        print_missing(template_messages)
        return

    presets = load_standard_presets()
    result = run_full_pipeline(
        str(image_path),
        presets["hough"],
        presets["roi"],
        presets["tab_edge"],
        presets["radial"],
        template_data,
    )
    print_logs(result)
    if not result["success"]:
        print("Full pipeline that bai.")
        return

    save_result_images("run_full_pipeline", result)
    print("Final result:", result["data"]["final_result_path"])
    print("Results CSV :", result["data"]["results_csv_path"])


if __name__ == "__main__":
    main()
