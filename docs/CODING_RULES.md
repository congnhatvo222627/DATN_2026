# CODING_RULES.md

Tai lieu nay chot cac quy tac code cho project `detect_stator`.

## Cong nghe chinh

- Dung Python, OpenCV, NumPy, Tkinter / ttk.
- Co the dung Pillow de hien thi anh trong Tkinter.
- Co the dung matplotlib de ve signature va MSE.

## Nguyen tac kien truc

- Moi duong dan nam trong config hoac truyen qua GUI.
- Ho tro duong dan tieng Viet.
- Khong de code xu ly anh phu thuoc GUI.
- Khong viet thuat toan xu ly anh truc tiep trong callback giao dien.
- Thuat toan dat trong `src/`.
- GUI chi goi ham da tach rieng.
- Moi buoc nen co the chay doc lap.
- Preset tham so phai luu bang JSON.

## Quy uoc ket qua ham xu ly

Moi ham xu ly nen tra ve `dict` theo mau:

```python
{
    "success": bool,
    "data": ...,
    "images": ...,
    "logs": [...],
}
```

## Quy tac an toan va debug

- Khong crash khi thieu anh.
- Log loi ro rang.
- Luon co anh debug cho tung buoc quan trong.
- Neu buoc xu ly that bai, phai tra log de GUI hien duoc nguyen nhan.

## Quy tac ve thuat toan

- Khong dung YOLO hoac deep learning trong pipeline chinh.
- Khong dung template matching toan anh de thay the Radial Signature.
- Uu tien giu pipeline classical vision theo tai lieu thiet ke.
- Buoc loc bien tai va Radial Signature la phan cot loi, khong duoc doi bang mot cach tiep can hoan toan khac neu chua co phe duyet.

## Quy tac ve duong dan va IO

- Ho tro duong dan tieng Viet bang `np.fromfile` / `cv2.imdecode`.
- Khi ghi anh, uu tien `cv2.imencode` / `tofile`.
- Khong hard-code duong dan tuyet doi trong code moi.

## Quy tac ve style code

- Ten module va ham phai ro nghia.
- Ham nen ngan gon va co docstring ngan.
- Tach IO, xu ly anh, luu preset, va GUI thanh module rieng.
- Khong bien mot file thanh noi chua toan bo pipeline.
