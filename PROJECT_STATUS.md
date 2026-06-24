# PROJECT_STATUS.md

## Muc tieu hien tai

Xay dung lai project `detect_stator` tu pipeline cu theo huong module hoa, co GUI debug tung buoc va co chuong trinh tong hop.

## Checklist

- [x] Tao tai lieu dinh huong project
- [x] Phan tich code cu
- [x] Lap ke hoach tan dung code cu
- [x] Tao cau truc project moi
- [x] Tao `src/io_utils.py`
- [x] Tao `src/preset_store.py`
- [x] Refactor buoc 1 HoughCircle
- [x] Refactor buoc 2 Cat ROI
- [x] Refactor buoc 3 Loc bien tai
- [x] Refactor buoc 4 Radial Signature
- [x] Refactor buoc 5 Tao mau 0 do
- [x] Refactor buoc 6 So khop MSE
- [x] Tao buoc 7 Tong hop
- [x] Tao GUI debug tung buoc
- [x] Tao chuc nang auto update
- [x] Tao chuc nang save/load preset
- [ ] Test voi anh khay 12 stator
- [ ] Test voi ROI mau
- [ ] Xuat `final_result.png`
- [ ] Xuat `results.csv`

## Loi hien tai

- Chua co bo anh test trong `data/input/` de chay full pipeline thuc te.
- Chua co `template_data.json` thuc te de test matching va full pipeline den cuoi.
- Can tinh chinh tham so Hough, Tab Edge, va Radial tren du lieu that.
- Python mac dinh `python` tren may dang tro den IRayple va thieu `cv2`; runtime verify dang dung la `C:\Program Files\OPT\DeepVision3\python.exe`.
- Step 3 nay da chuyen sang huong YOLO-guided edge extraction, nen moi may chay GUI/pipeline can co `ultralytics` va load duoc `best.pt`.

## Da trien khai

- Da tao day du cau truc `src/`, `gui/`, `data/`, `presets/`, `main.py`, `main_gui.py`, `requirements.txt`.
- Da tao them `scripts/` de chay test rieng tung buoc ma khong sua thuat toan trong `src/`.
- Da viet cac module core: `config`, `io_utils`, `preset_store`, `preprocess`, `hough_detector`, `roi_extractor`, `tab_edge_filter`, `radial_signature`, `template_builder`, `angle_matcher`, `pipeline_runner`, `visualization`.
- Da tao GUI 7 panel theo huong `GUI chi goi ham trong src`.
- Da tao preset mac dinh va co che save/load preset.
- Da verify import toan project, khoi tao GUI, va CLI khong crash khi thieu anh.
- Da gom 10 file pipeline legacy vao thu muc `legacy_pipeline/` de giam roi thu muc goc.
- Da xoa cac thu muc `__pycache__/` khong can thiet.
- Da them `fast mode` cho Hough trong core `src/` theo huong detect tren anh resize, scale circle ve anh goc, roi cat ROI tu anh goc.
- Da chuyen Hough panel va Full Pipeline panel sang worker thread + queue + `after()` de GUI khong bi dung khi detect.
- Da chuyen buoc 3 sang huong `YOLO box -> preprocess crop tai -> contour filter -> outer profile`, de `tab_edges_clean` chi con bien tai sach hon cho Radial Signature.

## File nen tan dung uu tien

- `2. cat roi.py`: logic cat ROI, preview ROI, save/load preset, va flow buoc 2.
- `1.2 Hough Circle.py`: core Hough detection, dedup, radius consensus, common-radius refine.
- `1. HoughCircle.py`: config normalize, auto update, preset UX, va bo cuc preview buoc 1.
- Cac helper `read_image()` / `write_image()` dung `np.fromfile` + `cv2.imdecode` va `cv2.imencode` + `tofile`.

## File khong nen dung truc tiep

- `pipeline_full_code.py`: monolith tham khao, khong phat trien tiep truc tiep.
- `6.radial signature.py`: khac thuat toan da chot.
- `4.py`: script thu nghiem, khong co kien truc mo rong.
- Phan YOLO trong `tao_anh_mau.py`: chi dung tham khao y tuong xu ly box/crop contour; khong tiep tuc phat trien truc tiep trong file legacy.

## Viec tiep theo

- Dat anh khay that vao `data/input/`.
- Tao ROI mau 0 do va luu `template_data.json` bang GUI.
- Tune Hough preset, ROI preset, Tab Edge preset, va Radial preset tren du lieu that.
- Chay full pipeline tren anh khay 12 stator va danh dau cac diem can refine.

## Nhat ky thay doi

| Ngay | Thay doi | File lien quan | Ghi chu |
| ---- | -------- | -------------- | ------- |
| 2026-06-23 | Tao bo tai lieu dinh huong ban dau cho project | `AGENTS.md`, `PROJECT_STATUS.md`, `docs/*.md` | Chi tao tai lieu, chua sua code legacy |
| 2026-06-23 | Hoan tat doc va phan tich 10 file legacy pipeline, cap nhat bang tan dung | `docs/LEGACY_REUSE_PLAN.md`, `PROJECT_STATUS.md` | Chua viet code moi, chua sua file legacy |
| 2026-06-23 | Trien khai kien truc moi `src/` + `gui/`, tao CLI/GUI, preset, va runner tung buoc | `src/*.py`, `gui/*.py`, `main.py`, `main_gui.py`, `requirements.txt`, `PROJECT_STATUS.md` | Da verify import, GUI init, va CLI xu ly truong hop thieu anh |
| 2026-06-23 | Don dep cau truc project, chuyen script cu vao `legacy_pipeline/` va xoa `__pycache__/` | `legacy_pipeline/*`, `PROJECT_STATUS.md` | Giu legacy de tham khao nhung lam gon thu muc goc |
| 2026-06-23 | Tao bo `scripts/` de test rieng tung buoc va xu ly truong hop thieu input an toan | `scripts/*.py`, `PROJECT_STATUS.md` | Moi script chay duoc bang `python scripts/<ten>.py` |
| 2026-06-23 | Tao GUI nhe doc lap chi de tinh chinh buoc HoughCircle, chi goi `run_hough_step` | `scripts/gui_hough_tuner.py`, `PROJECT_STATUS.md` | Chay `python scripts/gui_hough_tuner.py`; co debounce auto update, save/load `hough_preset.json`, khong sua thuat toan src |
| 2026-06-23 | Chong lag cho Hough tuner: tham so dang slider + nhap nhanh; them fast preview (thu nho anh + scale tham so do dai) va chay Hough trong thread nen + queue | `scripts/gui_hough_tuner.py`, `PROJECT_STATUS.md` | Hoc tu fast mode/threading cua legacy `2. cat roi.py`; GUI khong dong, preset van la pixel anh goc |
| 2026-06-23 | Them thanh truot (slider) + o nhap nhanh cho tham so so trong GUI chinh; spec co min/max thi hien slider | `gui/common_widgets.py`, `gui/hough_step_panel.py`, `gui/roi_step_panel.py`, `gui/tab_edge_step_panel.py`, `gui/radial_step_panel.py`, `PROJECT_STATUS.md` | Sua chung `ParameterPanel` nen ap dung cho moi tab; get_data/set_data va auto update giu nguyen |
| 2026-06-23 | Sua loi anh debug nhay ve muc dau (roi_gray) khi auto update; giu nguyen anh debug dang chon qua cac lan chay lai | `gui/common_widgets.py`, `PROJECT_STATUS.md` | `set_options` giu lua chon cu neu con; chi fallback ve anh dau khi key cu bien mat |
| 2026-06-23 | Them fast mode cho Hough trong GUI chinh va core, scale circle ve anh goc truoc khi cat ROI, va chay detect/full pipeline trong thread nen | `src/hough_detector.py`, `src/config.py`, `gui/hough_step_panel.py`, `gui/full_pipeline_panel.py`, `docs/PIPELINE_DESIGN.md`, `PROJECT_STATUS.md` | Luong moi: detect tren anh resize -> scale ve anh goc -> cat ROI anh goc -> xu ly goc tren ROI; GUI poll ket qua bang `after()` |
| 2026-06-23 | Chuyen buoc 3 `Tab Edges` sang huong YOLO-guided contour filtering, dung `best.pt` de khoanh vung tai tren ROI roi moi loc outer profile | `src/tab_edge_filter.py`, `src/config.py`, `gui/tab_edge_step_panel.py`, `presets/tab_edge_preset.json`, `requirements.txt`, `docs/PIPELINE_DESIGN.md`, `PROJECT_STATUS.md` | Dau ra `tab_edges_clean` chi giu bien tai; model duoc cache de tranh load lai moi ROI |
