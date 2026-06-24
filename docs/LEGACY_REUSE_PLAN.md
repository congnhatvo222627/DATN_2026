# LEGACY_REUSE_PLAN.md

Tai lieu nay tong hop ket qua doc va phan tich legacy pipeline.

Luu y:

- Trong workspace hien tai, cac file legacy dang nam o thu muc goc, chua co thu muc `pipeline/`.
- Van coi toan bo nhung file nay la legacy reference.
- Khong copy nguyen file cu sang code moi.
- Chi boc tach ham tot, logic tot, y tuong tot, va viet lai theo kien truc `src/` + `gui/`.

## Bang danh gia tan dung legacy

| File cu | Chuc nang hien tai | Diem tot | Van de | Phan nen tan dung | Module moi | Muc uu tien |
| ------- | ------------------ | -------- | ------ | ----------------- | ---------- | ----------- |
| `1. HoughCircle.py` | GUI buoc 1 gom tien xu ly truoc Hough, tune tham so, auto update, luu/nap preset, va goi backend tu `1.2 Hough Circle.py`. | Co preview nhanh, co preset JSON, co `normalize_config()`, co `read_image()` / `write_image()` ho tro duong dan tieng Viet, co flow UI ro rang cho buoc 1. | Tron GUI, preset, IO, visualization, va Hough backend trong mot file; dynamic import file legacy; phu thuoc preset active de file ROI nap lai. | Y tuong chia nhom tham so, `normalize_config()`, `merge_nested_dict()`, flow auto update, layout preview `Original / Preprocessed / Hough All / Hough Filtered`, output log va preset. | `src/io_utils.py`, `src/preset_store.py`, `src/preprocess.py`, `src/hough_detector.py`, `src/visualization.py`, `gui/hough_step_panel.py` | Cao |
| `1.1 tien xu ly truoc houghCircle.py` | Script CLI tien xu ly anh truoc Hough: gray -> CLAHE -> Gaussian Blur, luu hinh so sanh. | Don gian, de hieu, co `read_image()` ho tro Unicode path, the hien ro chuoi tien xu ly co ban truoc Hough. | Co co che relaunch python voi runtime hard-code, khong co GUI, khong co preset, chi la script mot lan. | Y tuong `find_default_image()`, `read_image()`, va chuoi `gray -> CLAHE -> Gaussian` cho `preprocess_for_hough()`. | `src/io_utils.py`, `src/preprocess.py` | Trung binh |
| `1.2 Hough Circle.py` | Backend Hough tim tam stator, cham diem bam bien, loc trung tam, loc ban kinh dong thuan, refine theo `r_common`, co GUI don gian. | Gia tri thuat toan cao nhat cho buoc 1; co `crop_work_roi()`, `circle_edge_score()`, `_collect_candidates()`, `dedup_circles()`, `radius_consistency_refine()`, `refine_circles_by_common_radius()`, `detect_stator_centers()`, `read_image()` / `write_image()` Unicode-safe. | `CONFIG` hard-code duong dan tuyet doi, GUI va thuat toan cung file, luu output truc tiep, khong theo contract `success/data/images/logs`. | Hau het core Hough logic va radius-consensus logic; y tuong draw overlay va crop ROI lon truoc Hough. | `src/preprocess.py`, `src/hough_detector.py`, `src/visualization.py` | Cao |
| `2. cat roi.py` | GUI cat ROI quanh tung stator dua tren Hough, co preview tung ROI / tat ca ROI, luu/nap preset, co co che thu hoi stator bi thieu trong khay 12 vi tri. | File co gia tri tan dung cao nhat cho buoc 2; `crop_stator_roi()` gon, an toan bien anh; co `detect_stator_center_from_hough()`, `build_roi_gallery_figure()`, `build_selected_roi_preview()`, `save_preset_file()` / `load_preset_file()`, Unicode IO, auto update, flow ROI ro rang. | Van phu thuoc dynamic import sang file Hough legacy; GUI, orchestration, recovery heuristic, va ROI logic van o chung mot file; ten output phuc vu bao cao cu. | Logic cat ROI, metadata ROI, preview chon ROI, gallery tong hop, luu ROI, va co the giu `maybe_recover_missing_circle()` lam y tuong nang cao sau nay. | `src/io_utils.py`, `src/preset_store.py`, `src/roi_extractor.py`, `src/visualization.py`, `src/pipeline_runner.py`, `gui/roi_step_panel.py` | Cao |
| `3.dau vao thuat toan xoay.py` | GUI giam sat tien xu ly ROI truoc Canny: gray, CLAHE tuy chon, loc nhieu, Canny, luu anh trung gian. | Co cac helper de tach `to_gray`, CLAHE, blur, Canny; co `make_odd()` va kiem tra `aperture_size`; preview ro tung buoc. | Chua lam radius filter, component filter, morphology theo spec moi; khong co preset save/load; phu thuoc global `CONFIG`; chi moi den muc Canny. | Y tuong preprocess ROI, `make_odd()`, blur method `Gaussian / Median / Bilateral / None`, kiem tra tham so Canny, flow hien thi chuoi anh trung gian. | `src/preprocess.py`, `gui/tab_edge_step_panel.py`, `src/visualization.py` | Trung binh |
| `4.py` | Script thu nghiem so sanh cac phuong an tien xu ly ROI: Canny, morphology nhe, contour, morphology truc tiep tren anh xam. | Cho thay morphology close nhe sau Canny co the huu ich; contour tu Canny co the dung cho debug; giup loai bo huong morphology truc tiep tren anh xam. | Hard-code duong dan tuyet doi; khong co GUI; khong co preset; `cv2.imwrite()` khong ho tro duong dan Unicode tot bang `imencode/tofile`; code nam het trong script body. | Chi nen giu y tuong thuc nghiem: morphology close nhe sau Canny va contour debug. | `src/tab_edge_filter.py`, `src/visualization.py` | Thap |
| `5. tim coutour ngoai bang ban kinh.py` | GUI Radial Signature 360 tia tu anh Canny, tim tam bang Hough hoac tam anh, noi suy gia tri thieu, loc spike, ve radial rays va bang gia tri. | Co `radial_signature_from_canny_robust()`, `circular_interpolate_missing()`, `remove_local_spikes_circular()`, `draw_radial_lines_on_roi()`, `draw_hough_circle_overlay()`, resize/display helpers; co GUI debug truc quan. | Di truc tiep tu Canny sang signature, khong qua `tab_edges_clean` nhu pipeline da chot; khong co radius mask + component filter; file dai, GUI va thuat toan tron nhau; khong co preset JSON. | Y tuong 360 bin, noi suy vong tron, ve tia radial, loc spike cuc bo, va cach hien thi `valid_count`; can viet lai de dau vao la `tab_edges_clean`. | `src/radial_signature.py`, `src/visualization.py`, `gui/radial_step_panel.py` | Trung binh |
| `6.radial signature.py` | GUI tinh tam va goc xoay bang Polar transform + NCC, khong con dung Radial Signature + MSE nhu pipeline moi. | Co `refine_peak_parabolic()` hay cho buoc lam min / max sub-degree; co overlay hien thi tam va vanh matching; Unicode IO tot. | Khac thuat toan da chot; dung Polar + NCC thay cho `tab_edges_clean -> Radial Signature -> MSE`; khong co preset; global constant nhieu; GUI tron voi matching. | Chi nen tham khao rieng `refine_peak_parabolic()` va mot vai y tuong overlay / do tin cay; khong dung truc tiep pipeline matching. | `src/angle_matcher.py`, `src/visualization.py` | Thap |
| `pipeline_full_code.py` | Ban gop tat ca cac buoc vao mot file duy nhat, de tham khao luong tong the cua pipeline cu. | Huu ich de doi chieu ten ham, thu tu xu ly, va truy vet luong tong. | Qua dai, trung lap ham, nhieu doan hard-code duong dan tuyet doi, tron tat ca buoc vao mot file, co nhieu phuong phap cu va moi song song, rat kho bao tri. | Chi dung de tham khao luong tong va doi ten ham; khong boc tach truc tiep tu day neu da co file goc rieng. | `src/pipeline_runner.py` chi de tham khao thu tu buoc | Thap |
| `tao_anh_mau.py` | GUI tao anh mau / template, ket hop Hough preset, preview radial, luu/nap preset anh mau, nhung dua vao YOLO de tim tab. | Co UI y tuong tot cho template step; co `smooth_circular_profile()`, `ensure_odd()`, `clamp_value()`, `mask_edges_by_radius()`, `draw_tab_edges_only()`, `draw_radial_lines_only()`, va co schema luu preset template. | Dung YOLO / `best.pt`, trai voi quy tac pipeline chinh; phu thuoc `ultralytics`; co nhieu duong dan ngoai project; co it nhat mot cho dung `cv2.imread()` thay vi Unicode-safe IO; tron qua nhieu vai tro trong mot file. | Chi nen giu y tuong GUI template, schema `template_data`, mot so helper smoothing / visualization; bo hoan toan phan YOLO khi sang kien truc moi. | `src/template_builder.py`, `src/radial_signature.py`, `src/visualization.py`, `src/preset_store.py`, `gui/template_step_panel.py` | Trung binh |

## Tong ket muc uu tien tan dung

### Uu tien cao

- `2. cat roi.py`: file gia tri tan dung cao nhat cho buoc ROI.
- `1.2 Hough Circle.py`: file gia tri tan dung cao nhat cho core Hough detector.
- `1. HoughCircle.py`: gia tri cao ve preset, live preview, auto update, va flow GUI buoc 1.
- Cac helper `read_image()` / `write_image()` dung `np.fromfile` + `cv2.imdecode` / `cv2.imencode` + `tofile`.

### Uu tien trung binh

- `1.1 tien xu ly truoc houghCircle.py`: chuoi preprocess co ban truoc Hough.
- `3.dau vao thuat toan xoay.py`: ROI preprocess + Canny + blur selector.
- `5. tim coutour ngoai bang ban kinh.py`: y tuong radial 360 tia, noi suy, spike filter, ray overlay.
- `tao_anh_mau.py`: y tuong template GUI, smoothing, visualization, schema luu template, nhung phai loai YOLO.

### Khong nen dung truc tiep

- `pipeline_full_code.py`: chi dung tham khao luong tong.
- `6.radial signature.py`: khac thuat toan da chot, chi nen tham khao mot vai helper nho.
- `4.py`: script thu nghiem, khong phu hop de phat trien truc tiep.

## Nguyen tac refactor tu legacy sang kien truc moi

- Khong xoa file cu.
- Khong sua file cu.
- Khong import dong qua lai giua cac file legacy trong code moi.
- Viet lai theo hop dong tra ve thong nhat: `success`, `data`, `images`, `logs`.
- Tach ro:
  - `src/` cho IO, preprocess, detector, ROI, tab-edge, radial, matcher, runner, visualization.
  - `gui/` cho panel debug va widget hien thi.
- Neu mot ham legacy co gia tri, boc tach y tuong va viet lai sach theo config / preset moi thay vi copy nguyen van.
