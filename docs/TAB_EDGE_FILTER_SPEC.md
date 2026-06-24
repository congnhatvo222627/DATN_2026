# TAB_EDGE_FILTER_SPEC.md

Tai lieu nay mo ta chi tiet buoc loc bien tai stator, la buoc tao `tab_edges_clean` cho Radial Signature.

## Muc tieu anh dau ra

- Nen den.
- Chi giu bien trang thuoc vung tai stator.
- Bien trong long stator, ranh tan nhiet, va nhieu nho phai duoc loai bo.
- Co the ve tam va duong tron than de debug.

## Luong xu ly

`ROI`
-> `grayscale`
-> `CLAHE`
-> `blur`
-> `Canny`
-> `radius_mask`
-> `tab_edges_raw`
-> `connected component`
-> `morphology close` nhe neu can
-> `tab_edges_clean`

## Muc tieu xu ly

- Giu lai nhung diem bien nam o vung tai stator.
- Loai bo cac bien nam sau trong long stator.
- Loai bo nhieu do texture, ranh, va diem le tan man.
- Tao anh dau vao on dinh cho buoc Radial Signature.

## Tham so

- `use_clahe`
- `clahe_clip_limit`
- `blur_method`
- `gaussian_kernel`
- `canny_threshold1`
- `canny_threshold2`
- `r_min_factor`
- `r_max_factor`
- `inner_margin_px`
- `outer_margin_px`
- `min_area`
- `max_area`
- `min_width`
- `min_height`
- `min_radius_mean_factor`
- `max_radius_mean_factor`
- `use_close`
- `close_kernel`

## Cong thuc loc radius

```text
rho = sqrt((x - cx)^2 + (y - cy)^2)
```

Giu diem bien neu:

```text
r_inner <= rho <= r_outer
```

Trong do:

```text
r_inner = radius * r_min_factor + inner_margin_px
r_outer = radius * r_max_factor + outer_margin_px
```

## Giai thich quy tac radius mask

- `r_inner` giup loai bo bien nam qua sat tam va bien trong long stator.
- `r_outer` giup chan nhieu nam qua xa than tron hoac vuot ra ngoai vung mong muon.
- Cap `r_min_factor` va `r_max_factor` can duoc tune theo ROI thuc te va kich thuoc tai stator.

## Connected component filter

Giu component neu:

- `min_area <= area <= max_area`
- `w >= min_width`
- `h >= min_height`
- `radius * min_radius_mean_factor <= rho_centroid <= radius * max_radius_mean_factor`

## Giai thich connected component

- `area` dung de bo cac diem nhiu qua nho va cac cum qua lon bat thuong.
- `w` va `h` dung de tranh giu lai cac component day mot chieu rat ngan.
- `rho_centroid` dung de kiem tra component co nam quanh vung tai stator hay khong.

## Morphology close

- Chi dung morphology close nhe neu can noi cac doan bien dut ngan.
- Khong duoc close qua manh den muc bien trong long va bien tai dính vao nhau.
- `close_kernel` nen nho va de tune tu GUI.

## Yeu cau debug

Ham xu ly buoc nay can tra ve cac anh debug sau:

- `roi_gray`
- `roi_preprocessed`
- `canny_edges`
- `radius_mask`
- `tab_edges_raw`
- `tab_edges_clean`
- `debug_overlay`

## Noi dung `debug_overlay`

- Ve tam ROI.
- Ve duong tron `radius`.
- Ve hai vong `r_inner` va `r_outer`.
- Co the to mau component duoc giu va component bi loai.

## Tieu chi danh gia dat yeu cau

- `tab_edges_clean` phai tap trung quanh cac tai stator.
- Khong con nhieu bien trong long stator.
- Signature tao ra o buoc sau phai co hinh dang on dinh va lap lai duoc.
- Khi doi ROI giua cac stator khac nhau, buoc nay van con kha nang tune thong qua preset thay vi sua code.
