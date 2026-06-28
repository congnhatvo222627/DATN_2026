"""Reusable Tkinter widgets for the step-debug GUI."""

import copy
import tkinter as tk
from tkinter import scrolledtext, ttk

from PIL import Image, ImageTk

from src.visualization import cv_bgr_to_rgb, resize_for_display

from .theme import PALETTE


def get_nested(data, path, default=None):
    """Read a dotted path from a nested dict."""
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_nested(data, path, value):
    """Write a dotted path into a nested dict."""
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


class ImageViewer(ttk.Frame):
    """Simple image viewer for OpenCV images."""

    def __init__(self, master):
        super().__init__(master)
        self.label = tk.Label(
            self,
            anchor="center",
            background=PALETTE["surface"],
            foreground=PALETTE["muted"],
            text="Chua co anh hien thi",
            font=("Segoe UI", 11),
            bd=1,
            relief="solid",
            highlightthickness=0,
        )
        self.label.configure(highlightbackground=PALETTE["border"])
        self.label.pack(fill="both", expand=True)
        self._photo = None

    def set_image(self, image):
        """Render an OpenCV image."""
        if image is None:
            self.label.configure(image="", text="Chua co anh hien thi")
            self._photo = None
            return
        width = max(480, self.winfo_width() - 12)
        height = max(320, self.winfo_height() - 12)
        display = resize_for_display(image, max_width=width, max_height=height)
        rgb = cv_bgr_to_rgb(display)
        pil_image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(pil_image)
        self.label.configure(image=self._photo, text="")


class LogPanel(ttk.Frame):
    """Read-only log output."""

    def __init__(self, master, height=10):
        super().__init__(master)
        self.text = scrolledtext.ScrolledText(
            self,
            height=height,
            wrap="word",
            background=PALETTE["surface"],
            foreground=PALETTE["text"],
            insertbackground=PALETTE["text"],
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            font=("Consolas", 9),
            padx=8,
            pady=6,
        )
        self.text.pack(fill="both", expand=True)
        self.text.configure(state="disabled")

    def set_lines(self, lines):
        """Replace log contents."""
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", "\n".join(lines))
        self.text.configure(state="disabled")


class DebugImageSelector(ttk.Frame):
    """Combobox for debug image names."""

    def __init__(self, master, on_change):
        super().__init__(master)
        ttk.Label(self, text="Anh debug").pack(side="left", padx=(0, 6))
        self.var = tk.StringVar()
        self.combo = ttk.Combobox(self, textvariable=self.var, state="readonly", width=28)
        self.combo.pack(side="left", fill="x", expand=True)
        self.combo.bind("<<ComboboxSelected>>", lambda _event: on_change(self.var.get()))

    def set_options(self, names):
        """Set combobox values, giu lai lua chon hien tai neu van con.

        Tra ve ten dang duoc chon de phia goi biet anh nao can hien thi.
        """
        self.combo["values"] = names
        current = self.var.get()
        if current not in names:
            self.var.set(names[0] if names else "")
        return self.var.get()


class ResultTable(ttk.Frame):
    """Treeview table for final pipeline results."""

    def __init__(self, master, columns, height=8):
        super().__init__(master)
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=height)
        for column in columns:
            self.tree.heading(column, text=column)
            self.tree.column(column, width=110, anchor="center")
        scroll_y = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll_y.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")

    def set_rows(self, rows):
        """Replace table rows."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in rows:
            self.tree.insert("", "end", values=row)


class ParameterPanel(ttk.Frame):
    """Editable nested-parameter form built from field specs.

    O so co ca "min" va "max" trong spec se hien them thanh truot (slider) de keo,
    dong bo 2 chieu voi o nhap so. Cac spec khong co min/max giu nguyen o nhap.
    """

    def __init__(self, master, field_specs, initial_data, on_change=None):
        super().__init__(master)
        self.field_specs = field_specs
        self.on_change = on_change
        self.vars = {}
        self._scales = {}
        self._sync_guard = False
        self._groups = {}
        self._build(initial_data)

    def _build(self, initial_data):
        for spec in self.field_specs:
            group_name = spec.get("group", "General")
            if group_name not in self._groups:
                frame = ttk.LabelFrame(self, text=group_name)
                frame.pack(fill="x", padx=4, pady=4)
                self._groups[group_name] = frame
            container = self._groups[group_name]
            row = ttk.Frame(container, style="Card.TFrame")
            row.pack(fill="x", padx=6, pady=3)
            path = spec["path"]
            field_type = spec.get("type", "str")
            value = get_nested(initial_data, path, spec.get("default"))
            ttk.Label(row, text=spec["label"], width=22, style="Surface.TLabel").pack(side="left")

            if field_type == "bool":
                var = tk.BooleanVar(value=bool(value))
                self.vars[path] = (var, field_type)
                ttk.Checkbutton(row, variable=var).pack(side="right")
            elif field_type in ("int", "float") and "min" in spec and "max" in spec:
                # Thanh truot de KEO + o nhap so de GO nhanh, dong bo 2 chieu.
                var = tk.StringVar(value=str(value))
                self.vars[path] = (var, field_type)
                ttk.Entry(row, textvariable=var, width=7).pack(side="right")
                scale = ttk.Scale(
                    row,
                    orient="horizontal",
                    from_=float(spec["min"]),
                    to=float(spec["max"]),
                    command=lambda raw, p=path: self._on_scale(p, raw),
                )
                scale.pack(side="left", fill="x", expand=True, padx=(0, 6))
                self._scales[path] = scale
                # Dat vi tri thanh truot khop gia tri thuc ngay tu dau, tranh
                # tinh trang slider nam o min con o nhap lai hien gia tri khac.
                try:
                    scale.set(float(value))
                except (ValueError, TypeError):
                    pass
            else:
                var = tk.StringVar(value=str(value))
                self.vars[path] = (var, field_type)
                ttk.Entry(row, textvariable=var, width=12).pack(side="right", fill="x", expand=True)

            var.trace_add("write", lambda *_a, p=path: self._on_var_changed(p))

    @staticmethod
    def _format_value(field_type, value):
        """Dinh dang gia tri tu thanh truot thanh chuoi gon cho o nhap."""
        if field_type == "int":
            return str(int(round(float(value))))
        text = "{:.2f}".format(float(value))
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text

    def _on_scale(self, path, raw_value):
        """Keo thanh truot -> cap nhat o nhap so."""
        if self._sync_guard:
            return
        var, field_type = self.vars[path]
        text = self._format_value(field_type, raw_value)
        if var.get() == text:
            return
        self._sync_guard = True
        try:
            var.set(text)
        finally:
            self._sync_guard = False

    def _on_var_changed(self, path):
        """O so/checkbox doi -> dong bo thanh truot + goi on_change."""
        if not self._sync_guard:
            scale = self._scales.get(path)
            if scale is not None:
                try:
                    value = float(self.vars[path][0].get())
                except (ValueError, TypeError):
                    value = None
                if value is not None:
                    self._sync_guard = True
                    try:
                        scale.set(value)
                    finally:
                        self._sync_guard = False
        if self.on_change is not None:
            self.on_change()

    def get_data(self):
        """Collect the nested parameter dict."""
        result = {}
        for path, (var, field_type) in self.vars.items():
            value = var.get()
            if field_type == "int":
                value = int(round(float(value or 0)))
            elif field_type == "float":
                value = float(value or 0)
            elif field_type == "bool":
                value = bool(value)
            set_nested(result, path, value)
        return result

    def set_data(self, data):
        """Update fields from a nested dict."""
        for spec in self.field_specs:
            path = spec["path"]
            value = get_nested(data, path, spec.get("default"))
            var, field_type = self.vars[path]
            if field_type == "bool":
                var.set(bool(value))
            else:
                var.set(str(value))


class StepPanelBase(ttk.Frame):
    """Shared layout and helpers for each step panel."""

    def __init__(self, master, app, field_specs, initial_params):
        super().__init__(master)
        self.app = app
        self.current_images = {}
        self._auto_job = None

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=8, pady=6)
        self.toolbar = toolbar

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.body = body
        self._sash_done = False
        self.bind("<Map>", self._init_sash, add="+")

        left = ttk.Frame(body)
        body.add(left, weight=3)
        self.left_panel = left

        self.debug_selector = DebugImageSelector(left, self._on_select_image)
        self.debug_selector.pack(fill="x", pady=(0, 6))

        self.image_viewer = ImageViewer(left)
        self.image_viewer.pack(fill="both", expand=True)

        right_host = ttk.Frame(body)
        # weight=0: khi phong to cua so, phan du don het cho khung anh ben trai,
        # con cot tham so giu nguyen be rong -> khong bao gio bi bop hep.
        body.add(right_host, weight=0)
        self.right_host = right_host

        self.right_canvas = tk.Canvas(
            right_host, highlightthickness=0, borderwidth=0, background=PALETTE["bg"]
        )
        self.right_canvas.pack(side="left", fill="both", expand=True)
        self.right_scrollbar = ttk.Scrollbar(right_host, orient="vertical", command=self.right_canvas.yview)
        self.right_scrollbar.pack(side="right", fill="y")
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)

        right = ttk.Frame(self.right_canvas)
        self.right_panel = right
        self._right_canvas_window = self.right_canvas.create_window((0, 0), window=right, anchor="nw")
        self.right_canvas.bind("<Configure>", self._on_right_canvas_configure)
        self.right_panel.bind("<Configure>", self._on_right_panel_configure)
        self.bind_all("<MouseWheel>", self._on_global_mousewheel, add="+")

        self.auto_update_var = tk.BooleanVar(value=False)
        auto_row = ttk.Frame(right, style="Card.TFrame", padding=(10, 7))
        auto_row.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(
            auto_row, text="Tu dong cap nhat khi chinh tham so", variable=self.auto_update_var
        ).pack(side="left")

        self.parameter_panel = ParameterPanel(right, field_specs, initial_params, on_change=self._on_params_changed)
        self.parameter_panel.pack(fill="x", expand=False)

        self.log_label = ttk.Label(right, text="Log", style="Section.TLabel")
        self.log_label.pack(anchor="w", pady=(8, 4))
        self.log_panel = LogPanel(right, height=14)
        self.log_panel.pack(fill="both", expand=True)

    RIGHT_PANEL_WIDTH = 600

    def _init_sash(self, _event=None):
        """Dat vach chia khung lan dau de cot tham so rong co dinh ben phai.

        Tranh truong hop phong to cua so lam cot tham so bi bop hep, che mat o nhap.
        """
        if self._sash_done:
            return
        self.update_idletasks()
        width = self.body.winfo_width()
        if width <= 1:
            self.after(120, self._init_sash)
            return
        try:
            self.body.sashpos(0, max(400, width - self.RIGHT_PANEL_WIDTH))
            self._sash_done = True
        except Exception:
            pass

    def _on_right_canvas_configure(self, event):
        """Keep the scrollable right panel matched to the canvas width."""
        self.right_canvas.itemconfigure(self._right_canvas_window, width=event.width)

    def _on_right_panel_configure(self, _event):
        """Refresh the scrollbar region when the right panel changes height."""
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

    @staticmethod
    def _is_descendant(widget, ancestor):
        """Return True when a widget belongs to the given parent subtree."""
        current = widget
        while current is not None:
            if current == ancestor:
                return True
            current = getattr(current, "master", None)
        return False

    def _on_global_mousewheel(self, event):
        """Let the mouse wheel scroll the parameter column on the right side."""
        if not self.winfo_exists():
            return
        widget = event.widget
        if not (
            self._is_descendant(widget, self.right_panel)
            or widget == self.right_canvas
            or self._is_descendant(widget, self.right_scrollbar)
        ):
            return
        first, last = self.right_canvas.yview()
        if first <= 0.0 and last >= 1.0:
            return
        delta = int(-1 * (event.delta / 120.0))
        if delta != 0:
            self.right_canvas.yview_scroll(delta, "units")

    def _on_params_changed(self):
        if self.auto_update_var.get():
            self.schedule_auto_run(self.run_step)

    def schedule_auto_run(self, callback):
        """Schedule an auto-run after a short delay."""
        if self._auto_job is not None:
            self.after_cancel(self._auto_job)
        self._auto_job = self.after(400, callback)

    def set_result(self, result):
        """Update debug images and logs from a step result.

        Giu nguyen anh debug dang xem khi chay lai (vd auto update khi keo slider),
        chi nhay ve anh dau tien khi lua chon cu khong con trong ket qua moi.
        """
        self.current_images = copy.deepcopy(result.get("images", {}))
        names = list(self.current_images.keys())
        selected = self.debug_selector.set_options(names)
        if selected in self.current_images:
            self.image_viewer.set_image(self.current_images[selected])
        elif names:
            self.image_viewer.set_image(self.current_images[names[0]])
        self.log_panel.set_lines(result.get("logs", []))

    def _on_select_image(self, name):
        if name in self.current_images:
            self.image_viewer.set_image(self.current_images[name])

    def run_step(self):
        """Implemented by subclasses."""
        raise NotImplementedError
