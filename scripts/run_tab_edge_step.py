"""Standalone test runner for step 3: tab-edge filtering."""

from _common import (
    bootstrap,
    get_first_roi_item,
    load_standard_presets,
    print_logs,
    print_missing,
    run_step_tab_edges,
    save_result_images,
)


def main():
    bootstrap()
    roi_item, messages = get_first_roi_item()
    if roi_item is None:
        print_missing(messages)
        return

    presets = load_standard_presets()
    result = run_step_tab_edges(roi_item, presets["tab_edge"])
    print_logs(result)
    if not result["success"]:
        print("Buoc Tab Edge that bai.")
        return

    save_result_images("run_tab_edge_step", result)
    print("Point count:", result["data"]["point_count"])
    print("Component count:", result["data"]["component_count"])


if __name__ == "__main__":
    main()
