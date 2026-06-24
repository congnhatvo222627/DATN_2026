"""Standalone test runner for step 5: template creation."""

from _common import (
    bootstrap,
    get_first_roi_item,
    load_standard_presets,
    print_logs,
    print_missing,
    run_step_template,
    save_result_images,
    save_template_if_success,
)


def main():
    bootstrap()
    roi_item, messages = get_first_roi_item(prefer_template=True)
    if roi_item is None:
        print_missing(
            messages
            + [
                "Goi y: dat ROI mau 0 do vao data/template/ hoac tao ROI truoc.",
            ]
        )
        return

    presets = load_standard_presets()
    result = run_step_template(roi_item, presets["tab_edge"], presets["radial"])
    print_logs(result)
    if not result["success"]:
        print("Buoc Template that bai.")
        return

    save_result_images("run_template_step", result)
    save_template_if_success(result)
    print("Da tao template thanh cong.")


if __name__ == "__main__":
    main()
