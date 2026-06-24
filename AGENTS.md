# AGENTS.md

## Muc tieu file nay

Day la file luat tong cho Codex va cac lan lam viec tiep theo trong project `detect_stator`.
Muc dich cua file nay la giu dung huong refactor, khong de pipeline moi di lech khoi y tuong da thong nhat.

## Muc tieu project

Ten project: `detect_stator`

De tai:
Xay dung he thong thi giac may 2D xac dinh vi tri va goc xoay stator phuc vu robot cap phoi CNC.

### Dau vao

- Anh khay chua nhieu stator, thuong la 12 stator.
- Anh hoac ROI stator mau 0 do.

### Dau ra

- ID tung stator.
- Toa do tam pixel: `center_x`, `center_y`.
- Ban kinh stator.
- Goc xoay `theta` so voi mau 0 do.
- Anh debug tung buoc.
- Anh `final_result` co ve tam, ID, duong tron, mui ten goc.
- File `results.csv`.
- Sau nay co the them Homography va TCP/IP.

## Giai doan hien tai

- Buoc hien tai chi tao tai lieu dinh huong project.
- Khong viet code xu ly anh o giai doan nay.
- Code cu chi duoc xem la legacy reference.
- Khong xoa code cu.
- Khong sua truc tiep code cu neu chua co yeu cau ro.

## Pipeline 7 buoc

### Buoc 1: HoughCircle

- Nhap anh khay.
- Co nhom tham so tien xu ly.
- Co the bat/tat tien xu ly.
- Neu tat tien xu ly thi dua anh xam goc vao HoughCircle.
- Co nhom tham so HoughCircle.
- Co auto update khi chinh tham so.
- Co nut luu preset.
- Dau ra: danh sach `circle` gom `ID`, `center_x`, `center_y`, `radius`.

### Buoc 2: Cat ROI

- Dung ket qua HoughCircle.
- Cat ROI quanh tung stator.
- Co the xem tung ROI theo ID.
- Co the luu ROI ra `data/roi/`.
- Uu tien tan dung logic tu file `2. cat roi.py`.

### Buoc 3: Loc bien tai stator

- Nhap ROI.
- Dung YOLO de khoanh vung tung tai tren ROI.
- Tien xu ly tung crop tai.
- Canny.
- Morph close.
- Loc contour theo dien tich va khoang cach so voi tam.
- Lay outer profile contour.
- Dau ra la `tab_edges_clean`, chi giu bien tai stator.
- Day la buoc rat quan trong vi anh huong truc tiep den Radial Signature.

### Buoc 4: Radial Signature

- Dung `tab_edges_clean`.
- Tu tam quet cac tia theo goc.
- Moi goc lay khoang cach xa nhat tu tam den bien tai.
- Hien thi anh ve tia va do thi signature.

### Buoc 5: Tao mau 0 do

- Nhap ROI mau hoac anh mau.
- Xu ly giong buoc 3 va buoc 4.
- Luu `template_signature`, ban kinh mau, tam mau, `params_used`.
- Du lieu mau duoc dung lai cho buoc so khop va tong hop.

### Buoc 6: So khop binh phuong toi thieu

- Nhap ROI test.
- Tao signature ROI test.
- Load `template_signature`.
- Dich vong signature va tinh MSE.
- Goc co MSE nho nhat la `angle_deg`.
- Hien thi do thi MSE theo goc.

### Buoc 7: Tong hop

- Nhap anh khay 12 stator.
- Load Hough preset.
- HoughCircle.
- Cat ROI.
- Load tab-edge preset.
- Loc bien tai.
- Tao Radial Signature.
- Load `template_signature`.
- So khop MSE.
- Xuat `final_result` va `results.csv`.

## Quy tac bat buoc

- Khong xoa code cu.
- Khong sua truc tiep code cu neu chua co yeu cau ro.
- Khong phat trien tiep bang cach gop tat ca vao mot file lon.
- Khong viet thuat toan xu ly anh truc tiep trong GUI.
- Thuat toan phai nam trong `src/`.
- GUI chi goi ham trong `src/`.
- Preset tham so phai luu bang JSON.
- Moi buoc phai co the chay doc lap.
- Khi bam Run o buoc nao tren GUI, ket qua buoc do phai hien ngay tren giao dien.
- Khong chi in terminal.
- Khong chi luu anh ra thu muc ma khong hien thi.
- Code phai de doc, chia module ro rang.
- Ham phai co docstring ngan.
- Co log loi ro rang.
- Ho tro duong dan tieng Viet bang `np.fromfile` / `cv2.imdecode` va `cv2.imencode` / `tofile`.
- Hien tai duoc phep dung YOLO rieng o buoc 3 de khoanh vung tai stator, nhung dau ra van phai la `tab_edges_clean` de phuc vu Radial Signature.
- Khong dung template matching toan anh de thay the Radial Signature.

## Nguyen tac lam viec voi legacy pipeline

- Legacy files la nguon tham khao, khong phai noi tiep tuc code truc tiep.
- `2. cat roi.py` la ung vien uu tien de boc tach y tuong va ham tot.
- `pipeline_full_code.py` chi dung de doi chieu luong tong the, khong dung lam file phat trien chinh.
- Moi quyet dinh refactor phai cap nhat tai lieu truoc hoac cung luc voi code moi.
