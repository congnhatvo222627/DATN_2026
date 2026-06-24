"""Standalone test runner for step 4: radial signature."""

from _common import (
    bootstrap,
    get_first_roi_item,
    load_standard_presets,
    print_logs,
    print_missing,
    run_step_radial,
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
    tab_result = run_step_tab_edges(roi_item, presets["tab_edge"])
    if not tab_result["success"]:
        print_logs(tab_result)
        print("Khong the chay Radial vi Tab Edge that bai.")
        return

    radial_result = run_step_radial(roi_item, tab_result["images"]["tab_edges_clean"], presets["radial"])
    print_logs(radial_result)
    if not radial_result["success"]:
        print("Buoc Radial that bai.")
        return

    radial_result["images"] = {**tab_result["images"], **radial_result["images"]}
    save_result_images("run_radial_step", radial_result)
    print("Valid bins:", radial_result["data"]["valid_bins"])


if __name__ == "__main__":
    main()
