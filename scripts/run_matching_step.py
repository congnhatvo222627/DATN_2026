"""Standalone test runner for step 6: ROI-vs-template matching."""

from _common import (
    bootstrap,
    get_first_roi_item,
    get_template_data_or_help,
    load_standard_presets,
    print_logs,
    print_missing,
    run_step_matching,
    save_result_images,
)


def main():
    bootstrap()
    template_data, template_messages = get_template_data_or_help()
    if template_data is None:
        print_missing(template_messages)
        return

    roi_item, messages = get_first_roi_item()
    if roi_item is None:
        print_missing(messages)
        return

    presets = load_standard_presets()
    result = run_step_matching(roi_item, template_data, presets["tab_edge"], presets["radial"])
    print_logs(result)
    if not result["success"]:
        print("Buoc Matching that bai.")
        return

    save_result_images("run_matching_step", result)
    print("Angle deg:", result["data"]["angle_deg"])
    print("Min error:", result["data"]["min_error"])


if __name__ == "__main__":
    main()
