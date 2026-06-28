"""Bang mau + style tap trung cho toan bo GUI (chi thay doi giao dien, khong dung thuat toan).

Goi `apply_theme(root)` mot lan sau khi tao cua so goc. Cac panel co the dung
PALETTE / FONTS de dong bo mau cho nhung widget tk thuan (tk.Button, tk.Label...).
"""

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk


# ---------------------------------------------------------------------------
# Bang mau - tone sang, sach, chuyen nghiep (lay diem nhan xanh navy + xanh duong).
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#eef1f6",          # nen tong cua app
    "surface": "#ffffff",      # the/card noi dung
    "surface_alt": "#f4f7fb",  # nen phu nhe
    "border": "#d4dbe5",       # vien
    "border_soft": "#e2e8f0",  # vien nhat
    "text": "#1f2a37",         # chu chinh
    "muted": "#64748b",        # chu phu
    "primary": "#2563eb",      # mau diem nhan (nut chinh)
    "primary_hover": "#1d4ed8",
    "primary_press": "#1e40af",
    "primary_text": "#ffffff",
    "header": "#16233a",       # thanh header toi
    "header_text": "#eef2f8",
    "header_muted": "#9fb0c8",
    "success": "#15803d",
    "success_hover": "#166534",
    "danger": "#dc2626",
    "danger_hover": "#b91c1c",
    "row_alt": "#f6f8fb",      # dong xen ke trong bang
    "sel": "#dbeafe",          # nen dong dang chon
}


FONTS = {
    "base": ("Segoe UI", 10),
    "small": ("Segoe UI", 9),
    "bold": ("Segoe UI", 10, "bold"),
    "section": ("Segoe UI Semibold", 10),
    "title": ("Segoe UI Semibold", 17),
    "subtitle": ("Segoe UI", 10),
    "metric": ("Segoe UI", 15, "bold"),
    "tab": ("Segoe UI Semibold", 10),
}


_APPLIED = False
_IMAGE_REFS = []          # giu tham chieu anh khoi bi GC
_CHECK_READY = False      # element check chi tao 1 lan


def _setup_checkmark(style, root):
    """Ve o tich hinh vuong bo goc + dau V mau trang khi duoc chon.

    Dung anh tu PIL nen hien dung dau tich (✓) thay vi dau X cua theme clam.
    Neu thieu PIL thi bo qua, giu indicator mac dinh.
    """
    global _CHECK_READY
    if _CHECK_READY:
        return
    try:
        from PIL import Image, ImageDraw, ImageTk
    except Exception:
        return

    p = PALETTE
    size = 20

    def _box(fill, outline):
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([1, 1, size - 2, size - 2], radius=5, fill=fill, outline=outline, width=2)
        return img, d

    unchecked, _ = _box(p["surface"], p["border"])
    hover, _ = _box(p["surface"], p["primary"])
    checked, d = _box(p["primary"], p["primary"])
    # Dau V
    d.line([(5, 10), (9, 14), (15, 6)], fill="#ffffff", width=2, joint="curve")

    img_un = ImageTk.PhotoImage(unchecked)
    img_hover = ImageTk.PhotoImage(hover)
    img_ch = ImageTk.PhotoImage(checked)
    _IMAGE_REFS.extend([img_un, img_hover, img_ch])

    try:
        style.element_create(
            "Custom.Checkbutton.indicator",
            "image",
            img_un,
            ("selected", img_ch),
            ("active", "!selected", img_hover),
            sticky="",
            padding=2,
        )
    except tk.TclError:
        return

    style.layout(
        "TCheckbutton",
        [
            (
                "Checkbutton.padding",
                {
                    "sticky": "nswe",
                    "children": [
                        ("Custom.Checkbutton.indicator", {"side": "left", "sticky": ""}),
                        ("Checkbutton.label", {"side": "left", "sticky": "nswe"}),
                    ],
                },
            )
        ],
    )
    _CHECK_READY = True


def apply_theme(root):
    """Ap dung theme cho toan bo ung dung. An toan khi goi nhieu lan."""
    global _APPLIED
    p = PALETTE

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    # Font mac dinh cho ca widget tk thuan (menu, entry...).
    try:
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            tkfont.nametofont(name).configure(family="Segoe UI", size=10)
    except tk.TclError:
        pass

    root.configure(bg=p["bg"])
    root.option_add("*Font", "{Segoe UI} 10")
    # Combobox dropdown (Listbox) - tk thuan, phai chinh qua option_add.
    root.option_add("*TCombobox*Listbox.background", p["surface"])
    root.option_add("*TCombobox*Listbox.foreground", p["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", p["primary"])
    root.option_add("*TCombobox*Listbox.selectForeground", p["primary_text"])
    root.option_add("*TCombobox*Listbox.font", "{Segoe UI} 10")
    # Tooltip/menu
    root.option_add("*Menu.background", p["surface"])
    root.option_add("*Menu.foreground", p["text"])
    root.option_add("*Menu.activeBackground", p["primary"])
    root.option_add("*Menu.activeForeground", p["primary_text"])
    root.option_add("*Menu.relief", "flat")

    # ----- Nen chung -----
    style.configure(".", background=p["bg"], foreground=p["text"], font=FONTS["base"])
    style.configure("TFrame", background=p["bg"])
    style.configure("TLabel", background=p["bg"], foreground=p["text"])
    style.configure("TLabelframe", background=p["bg"])

    # The noi dung mau trang
    style.configure("Card.TFrame", background=p["surface"], relief="flat")
    style.configure("Surface.TLabel", background=p["surface"], foreground=p["text"])
    style.configure("Muted.TLabel", background=p["bg"], foreground=p["muted"])
    style.configure("Section.TLabel", background=p["bg"], foreground=p["text"], font=FONTS["section"])
    style.configure("Metric.TLabel", background=p["bg"], foreground=p["muted"], font=FONTS["metric"])

    # Thanh cong cu
    style.configure("Toolbar.TFrame", background=p["surface"])

    # ----- Header toi -----
    style.configure("Header.TFrame", background=p["header"])
    style.configure("Header.TLabel", background=p["header"], foreground=p["header_text"])
    style.configure("HeaderTitle.TLabel", background=p["header"], foreground=p["header_text"], font=FONTS["title"])
    style.configure("HeaderSubtitle.TLabel", background=p["header"], foreground=p["header_muted"], font=FONTS["subtitle"])

    # ----- Nut -----
    style.configure(
        "TButton",
        background=p["surface"],
        foreground=p["text"],
        bordercolor=p["border"],
        focuscolor=p["surface_alt"],
        relief="flat",
        padding=(12, 6),
        font=FONTS["base"],
    )
    style.map(
        "TButton",
        background=[("disabled", p["surface_alt"]), ("pressed", p["border_soft"]), ("active", p["surface_alt"])],
        foreground=[("disabled", p["muted"])],
        bordercolor=[("active", p["primary"])],
    )

    style.configure(
        "Accent.TButton",
        background=p["primary"],
        foreground=p["primary_text"],
        bordercolor=p["primary"],
        relief="flat",
        padding=(14, 6),
        font=FONTS["bold"],
    )
    style.map(
        "Accent.TButton",
        background=[("disabled", "#9db8ef"), ("pressed", p["primary_press"]), ("active", p["primary_hover"])],
        foreground=[("disabled", "#eef2ff")],
    )

    style.configure("Success.TButton", background=p["success"], foreground="#ffffff", bordercolor=p["success"], padding=(14, 6), font=FONTS["bold"])
    style.map("Success.TButton", background=[("disabled", "#86b89a"), ("active", p["success_hover"]), ("pressed", p["success_hover"])])

    style.configure("Danger.TButton", background=p["danger"], foreground="#ffffff", bordercolor=p["danger"], padding=(14, 6), font=FONTS["bold"])
    style.map("Danger.TButton", background=[("disabled", "#e8a3a3"), ("active", p["danger_hover"]), ("pressed", p["danger_hover"])])

    # Menubutton (nut Image input ▾)
    style.configure("TMenubutton", background=p["surface"], foreground=p["text"], relief="flat", padding=(12, 6), arrowcolor=p["muted"])
    style.map("TMenubutton", background=[("active", p["surface_alt"])])

    # ----- LabelFrame (nhom tham so) -----
    style.configure(
        "TLabelframe",
        background=p["surface"],
        bordercolor=p["border"],
        relief="solid",
        borderwidth=1,
        padding=8,
    )
    style.configure(
        "TLabelframe.Label",
        background=p["surface"],
        foreground=p["primary"],
        font=FONTS["section"],
    )

    # ----- Entry / Combobox / Spinbox -----
    for widget in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(
            widget,
            fieldbackground=p["surface"],
            background=p["surface"],
            foreground=p["text"],
            bordercolor=p["border"],
            lightcolor=p["border"],
            darkcolor=p["border"],
            insertcolor=p["text"],
            arrowcolor=p["muted"],
            padding=4,
        )
        style.map(
            widget,
            bordercolor=[("focus", p["primary"]), ("active", p["primary"])],
            fieldbackground=[("readonly", p["surface"]), ("disabled", p["surface_alt"])],
            foreground=[("disabled", p["muted"])],
        )

    # ----- Checkbutton (dau tich V thay vi dau X) -----
    style.configure("TCheckbutton", background=p["surface"], foreground=p["text"], focuscolor=p["surface"])
    style.map("TCheckbutton", background=[("active", p["surface"])])
    _setup_checkmark(style, root)

    # ----- Scale (thanh truot) - to, ro, mau diem nhan -----
    style.configure(
        "Horizontal.TScale",
        background=p["primary"],          # mau con truot
        troughcolor=p["border_soft"],
        bordercolor=p["primary"],
        lightcolor=p["primary"],
        darkcolor=p["primary"],
        sliderthickness=22,
        sliderlength=18,
        troughrelief="flat",
    )
    style.map(
        "Horizontal.TScale",
        background=[("active", p["primary_hover"]), ("pressed", p["primary_press"])],
    )

    # ----- Scrollbar -----
    for orient in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(
            orient,
            background=p["border_soft"],
            troughcolor=p["bg"],
            bordercolor=p["bg"],
            arrowcolor=p["muted"],
            relief="flat",
        )
        style.map(orient, background=[("active", p["border"])])

    # ----- Treeview (bang ket qua) -----
    style.configure(
        "Treeview",
        background=p["surface"],
        fieldbackground=p["surface"],
        foreground=p["text"],
        bordercolor=p["border"],
        borderwidth=1,
        relief="solid",
        rowheight=26,
        font=FONTS["base"],
    )
    style.map(
        "Treeview",
        background=[("selected", p["sel"])],
        foreground=[("selected", p["text"])],
    )
    style.configure(
        "Treeview.Heading",
        background=p["surface_alt"],
        foreground=p["muted"],
        relief="flat",
        font=FONTS["section"],
        padding=(6, 6),
    )
    style.map("Treeview.Heading", background=[("active", p["border_soft"])])

    # ----- Notebook (cac tab buoc) -----
    style.configure("TNotebook", background=p["bg"], borderwidth=0, tabmargins=(8, 6, 8, 0))
    style.configure(
        "TNotebook.Tab",
        background=p["surface_alt"],
        foreground=p["muted"],
        bordercolor=p["border_soft"],
        padding=(16, 9),
        font=FONTS["tab"],
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", p["surface"]), ("active", p["surface"])],
        foreground=[("selected", p["primary"]), ("active", p["text"])],
        expand=[("selected", (0, 0, 0, 0))],
    )

    style.configure("TPanedwindow", background=p["bg"])
    style.configure("TSeparator", background=p["border"])

    _APPLIED = True
    return style


def style_treeview_rows(tree):
    """Bat dong xen ke mau cho mot Treeview de bang de doc hon."""
    tree.tag_configure("oddrow", background=PALETTE["surface"])
    tree.tag_configure("evenrow", background=PALETTE["row_alt"])
