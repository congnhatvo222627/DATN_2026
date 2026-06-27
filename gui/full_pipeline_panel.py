"""GUI panel for step 7: full tray pipeline.

Nhap anh toan khay + template mau, chuong trinh tu chay luong tab 1 -> tab 6 (coarse
Hough -> ROI -> Hough refine -> tab edge -> radial -> MSE) o thread nen va tra ve bang
ket qua giong tab 6. Tham so lay tu cac preset da luu o tab truoc.
"""

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.calibration import (
    MIN_CALIBRATION_POINTS,
    POINT_KEYS,
    compute_homography,
    load_calibration,
    save_calibration,
    transform_point,
)
from src.config import (
    CALIBRATION_PRESET_PATH,
    DEFAULT_HOUGH_PARAMS,
    DEFAULT_ROI_PARAMS,
    HOUGH_PRESET_PATH,
    INPUT_DIR,
    ROI_PRESET_PATH,
    TEMPLATE_DATA_PATH,
)
from src.io_utils import write_image
from src.pipeline_runner import run_full_pipeline
from src.preset_store import load_preset, load_radial_signature_preset
from src.tcp_client import TcpClient
from src.template_builder import load_template_data
from src.visualization import normalize_display_angle_deg

from .common_widgets import ResultTable, StepPanelBase


POLL_MS = 60


class FullPipelinePanel(StepPanelBase):
    """Panel for running the complete tray pipeline in one shot."""

    def __init__(self, master, app):
        self._busy = False
        self._pending = None
        self._event_queue = queue.Queue()
        self._poll_id = None
        self.tcp_client = TcpClient()
        self._mode_guard = False
        self._recv_queue = queue.Queue()
        self._recv_poll_id = None
        self._send_index = 0
        super().__init__(master, app, [], {})
        self.template_data = None
        self.image_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self._stopped = False

        ttk.Label(self.toolbar, text="Image input").pack(side="left")
        ttk.Entry(self.toolbar, textvariable=self.image_path_var, width=30).pack(side="left", padx=6, fill="x", expand=True)

        # Nut "Image input" co 2 che do: chon anh tu file hoac ket noi camera.
        self.source_menubutton = ttk.Menubutton(self.toolbar, text="Image input ▾")
        source_menu = tk.Menu(self.source_menubutton, tearoff=False)
        source_menu.add_command(label="Chon anh tu file", command=self.choose_image)
        source_menu.add_command(label="Ket noi camera", command=self.connect_camera)
        self.source_menubutton["menu"] = source_menu
        self.source_menubutton.pack(side="left", padx=3)

        ttk.Button(self.toolbar, text="Load Template", command=self.load_template).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load as Template", command=self.load_template_from_file).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save image", command=self.save_image).pack(side="left", padx=3)

        # Run / Stop noi bat: nut mau xanh va do, chu dam, co ky hieu.
        self.run_button = tk.Button(
            self.toolbar,
            text="▶ Run",
            command=self.run_step,
            bg="#2e7d32",
            fg="white",
            activebackground="#1b5e20",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            relief="raised",
            padx=12,
            cursor="hand2",
        )
        self.run_button.pack(side="left", padx=(8, 2))
        self.stop_button = tk.Button(
            self.toolbar,
            text="■ Stop",
            command=self.stop_step,
            bg="#c62828",
            fg="white",
            activebackground="#8e0000",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            relief="raised",
            padx=12,
            cursor="hand2",
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=2)
        ttk.Label(self.toolbar, textvariable=self.status_var).pack(side="left", padx=(8, 0))

        self._build_right_table()

    def _build_right_table(self):
        """Cot phai: bang ket qua + cau hinh TCP/IP + hieu chuan + che do gui.

        Bo han o Log cua tab nay, thay bang phan truyen thong de gui toa do
        (anh hoac thuc sau hieu chuan homography) toi server (vd Hercules).
        """
        self.parameter_panel.pack_forget()
        self.log_label.pack_forget()
        self.log_panel.pack_forget()
        ttk.Label(self.right_panel, text="Bang ket qua toan khay").pack(anchor="w", pady=(2, 4))
        self.table = ResultTable(
            self.right_panel,
            ["ID", "center_x", "center_y", "radius", "angle_deg", "min_error", "status"],
            height=14,
        )
        self.table.pack(fill="x")
        self._build_tcp_section()
        self._build_calibration_section()
        self._build_communication_section()

    # ---- TCP/IP -----------------------------------------------------------
    def _build_tcp_section(self):
        """Cau hinh ket noi TCP/IP: phan mem la client, server la thiet bi khac."""
        frame = ttk.LabelFrame(self.right_panel, text="Cau hinh TCP/IP (client)")
        frame.pack(fill="x", pady=(10, 4))

        self.tcp_host_var = tk.StringVar(value="127.0.0.1")
        self.tcp_port_var = tk.StringVar(value="9000")
        self.tcp_ack_var = tk.StringVar(value="done")
        self.tcp_status_var = tk.StringVar(value="Chua ket noi")

        row = ttk.Frame(frame)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Label(row, text="Host/IP", width=8).pack(side="left")
        ttk.Entry(row, textvariable=self.tcp_host_var, width=16).pack(side="left", padx=(0, 8))
        ttk.Label(row, text="Port", width=5).pack(side="left")
        ttk.Entry(row, textvariable=self.tcp_port_var, width=8).pack(side="left")

        row2 = ttk.Frame(frame)
        row2.pack(fill="x", padx=6, pady=2)
        ttk.Label(row2, text="ACK", width=8).pack(side="left")
        ttk.Entry(row2, textvariable=self.tcp_ack_var, width=8).pack(side="left")
        ttk.Label(
            row2, text="(moi lan server gui chuoi nay -> gui ID ke tiep)"
        ).pack(side="left", padx=(6, 0))

        row3 = ttk.Frame(frame)
        row3.pack(fill="x", padx=6, pady=(2, 6))
        self.tcp_connect_button = ttk.Button(row3, text="Ket noi", command=self.toggle_connection)
        self.tcp_connect_button.pack(side="left")
        ttk.Label(row3, textvariable=self.tcp_status_var).pack(side="left", padx=(10, 0))

    # ---- Calibration ------------------------------------------------------
    def _build_calibration_section(self):
        """Nhap cac cap diem hieu chuan toa do anh -> toa do thuc (homography)."""
        frame = ttk.LabelFrame(
            self.right_panel,
            text="Hieu chuan toa do - Homography (toi thieu {} cap diem)".format(
                MIN_CALIBRATION_POINTS
            ),
        )
        frame.pack(fill="x", pady=(10, 4))

        columns = ("image_x", "image_y", "real_x", "real_y")
        self.calib_tree = ttk.Treeview(frame, columns=columns, show="headings", height=5)
        headings = {
            "image_x": "anh X (px)",
            "image_y": "anh Y (px)",
            "real_x": "thuc X",
            "real_y": "thuc Y",
        }
        for column in columns:
            self.calib_tree.heading(column, text=headings[column])
            self.calib_tree.column(column, width=90, anchor="center")
        self.calib_tree.pack(fill="x", padx=6, pady=(4, 2))

        entry_row = ttk.Frame(frame)
        entry_row.pack(fill="x", padx=6, pady=2)
        self.calib_entry_vars = {key: tk.StringVar() for key in POINT_KEYS}
        for key in POINT_KEYS:
            ttk.Entry(entry_row, textvariable=self.calib_entry_vars[key], width=8).pack(
                side="left", padx=(0, 4)
            )
        ttk.Button(entry_row, text="Them cap", command=self.add_calibration_point).pack(side="left", padx=(4, 0))

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x", padx=6, pady=(2, 4))
        ttk.Button(btn_row, text="Xoa cap chon", command=self.remove_calibration_point).pack(side="left")
        ttk.Button(btn_row, text="Luu hieu chuan", command=self.save_calibration_points).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Kiem tra homography", command=self.check_homography).pack(side="left")

        self.calib_status_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.calib_status_var, wraplength=360).pack(
            anchor="w", padx=6, pady=(0, 4)
        )

        for pair in load_calibration(CALIBRATION_PRESET_PATH):
            if all(pair.get(key) is not None for key in POINT_KEYS):
                self.calib_tree.insert(
                    "", "end", values=tuple(pair[key] for key in POINT_KEYS)
                )
        self._update_calibration_status()

    # ---- Communication mode ----------------------------------------------
    def _build_communication_section(self):
        """Chon che do gui (toa do anh hoac toa do thuc) + nut gui ket qua."""
        frame = ttk.LabelFrame(self.right_panel, text="Truyen thong - che do gui")
        frame.pack(fill="x", pady=(10, 8))

        self.send_image_var = tk.BooleanVar(value=True)
        self.send_real_var = tk.BooleanVar(value=False)
        self.send_image_var.trace_add("write", lambda *_a: self._on_mode_toggle("image"))
        self.send_real_var.trace_add("write", lambda *_a: self._on_mode_toggle("real"))

        ttk.Checkbutton(
            frame, text="Gui toa do anh (pixel)", variable=self.send_image_var
        ).pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Checkbutton(
            frame,
            text="Gui toa do thuc (sau hieu chuan homography)",
            variable=self.send_real_var,
        ).pack(anchor="w", padx=6, pady=(0, 4))

        send_row = ttk.Frame(frame)
        send_row.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Button(send_row, text="Gui ID tiep theo", command=self.send_results).pack(side="left")
        self.send_status_var = tk.StringVar(value="")
        ttk.Label(send_row, textvariable=self.send_status_var, wraplength=300).pack(
            side="left", padx=(10, 0)
        )

    def _on_mode_toggle(self, which):
        """Hai o tick loai tru nhau: tick o nay thi bo o kia."""
        if self._mode_guard:
            return
        self._mode_guard = True
        try:
            if which == "image" and self.send_image_var.get():
                self.send_real_var.set(False)
            elif which == "real" and self.send_real_var.get():
                self.send_image_var.set(False)
        finally:
            self._mode_guard = False

    # ---- TCP actions ------------------------------------------------------
    def toggle_connection(self):
        """Ket noi hoac ngat ket noi TCP/IP toi server."""
        if self.tcp_client.connected:
            self._disconnect()
            return
        host = self.tcp_host_var.get().strip()
        port = self.tcp_port_var.get().strip()
        if not host or not port:
            messagebox.showwarning("TCP/IP", "Hay nhap Host va Port.")
            return
        try:
            self.tcp_client.connect(host, int(port))
        except ValueError:
            messagebox.showerror("TCP/IP", "Port phai la so nguyen.")
            return
        except OSError as exc:
            messagebox.showerror("TCP/IP", "Khong ket noi duoc {}:{}\n{}".format(host, port, exc))
            return
        self.tcp_status_var.set("Da ket noi {}:{} - dang cho ACK".format(host, port))
        self.tcp_connect_button.configure(text="Ngat ket noi")
        self._send_index = 0  # moi lan ket noi bat dau gui lai tu ID dau
        # Lang nghe du lieu server gui ve (chay o thread nen, day vao queue).
        self.tcp_client.start_listening(lambda text: self._recv_queue.put(text))
        if self._recv_poll_id is None:
            self._recv_poll_id = self.after(POLL_MS, self._poll_recv)

    def _disconnect(self):
        """Ngat ket noi va dung poll nhan du lieu."""
        self.tcp_client.disconnect()
        if self._recv_poll_id is not None:
            self.after_cancel(self._recv_poll_id)
            self._recv_poll_id = None
        self.tcp_status_var.set("Chua ket noi")
        self.tcp_connect_button.configure(text="Ket noi")

    def _poll_recv(self):
        """Doc cac chuoi server gui ve tu queue (chay o main thread)."""
        self._recv_poll_id = None
        try:
            while True:
                text = self._recv_queue.get_nowait()
                if text is None:
                    # Server da dong ket noi.
                    self._disconnect()
                    return
                self._on_ack_received(text)
        except queue.Empty:
            pass
        if self.tcp_client.connected and self._recv_poll_id is None:
            self._recv_poll_id = self.after(POLL_MS, self._poll_recv)

    def _on_ack_received(self, text):
        """Xu ly chuoi server gui ve. Neu khop ACK -> tu dong gui tung ID."""
        token = self.tcp_ack_var.get().strip()
        received = text.strip()
        is_ack = bool(received) and (received == token if token else True)
        if not is_ack:
            self.send_status_var.set("Nhan: {!r} (khong khop ACK {!r})".format(received, token))
            return
        self._send_next_id(triggered_by_ack=True)

    # ---- Calibration actions ---------------------------------------------
    def _collect_point_pairs(self):
        """Doc cac cap diem hieu chuan tu bang."""
        pairs = []
        for item in self.calib_tree.get_children():
            values = self.calib_tree.item(item, "values")
            pairs.append(dict(zip(POINT_KEYS, values)))
        return pairs

    def add_calibration_point(self):
        """Them mot cap diem tu cac o nhap (kiem tra phai la so)."""
        values = {}
        for key in POINT_KEYS:
            raw = self.calib_entry_vars[key].get().strip()
            try:
                values[key] = float(raw)
            except ValueError:
                messagebox.showwarning("Hieu chuan", "Gia tri '{}' phai la so.".format(key))
                return
        self.calib_tree.insert("", "end", values=tuple(values[key] for key in POINT_KEYS))
        for key in POINT_KEYS:
            self.calib_entry_vars[key].set("")
        self._update_calibration_status()

    def remove_calibration_point(self):
        """Xoa cap diem dang chon trong bang."""
        for item in self.calib_tree.selection():
            self.calib_tree.delete(item)
        self._update_calibration_status()

    def save_calibration_points(self):
        """Luu cac cap diem hieu chuan ra preset JSON."""
        pairs = self._collect_point_pairs()
        try:
            save_calibration(CALIBRATION_PRESET_PATH, pairs)
        except Exception as exc:
            messagebox.showerror("Hieu chuan", str(exc))
            return
        messagebox.showinfo("Hieu chuan", "Da luu {} cap diem.".format(len(pairs)))

    def check_homography(self):
        """Thu tinh homography tu cac cap diem hien tai va bao ket qua."""
        try:
            compute_homography(self._collect_point_pairs())
        except ValueError as exc:
            messagebox.showerror("Homography", str(exc))
            return
        messagebox.showinfo("Homography", "Tinh homography thanh cong, san sang gui toa do thuc.")

    def _update_calibration_status(self):
        """Cap nhat dong trang thai theo so cap diem da nhap."""
        count = len(self.calib_tree.get_children())
        if count >= MIN_CALIBRATION_POINTS:
            self.calib_status_var.set("Da co {} cap diem - du de gui toa do thuc.".format(count))
        else:
            self.calib_status_var.set(
                "Co {}/{} cap diem - chua du de gui toa do thuc.".format(
                    count, MIN_CALIBRATION_POINTS
                )
            )

    # ---- Send results -----------------------------------------------------
    def _format_id_line(self, row, homography):
        """Tao mot dong gui cho mot stator theo khung co nhan field.

        Vi du: <START>,ID=1,X=90.38,Y=119.85,THETA=53.70,STATUS=OK,<END>
        Neu homography khac None thi bien doi tam sang toa do thuc truoc khi gui.
        """
        x = row["center_x"]
        y = row["center_y"]
        if homography is not None:
            x, y = transform_point(homography, x, y)
        angle = normalize_display_angle_deg(row["angle_deg"])
        return "<START>,ID={},X={:.2f},Y={:.2f},THETA={:.2f},STATUS={},<END>\n".format(
            int(row["id"]), x, y, angle, str(row["status"]).upper()
        )

    def _send_next_id(self, triggered_by_ack=False):
        """Gui MOT ID tiep theo qua TCP/IP theo che do da chon.

        Moi lan goi (do server phan hoi ACK hoac do bam nut) se gui mot stator:
        lan 1 -> ID dau, lan 2 -> ID ke tiep, ... het danh sach thi quay lai dau.
        Khi triggered_by_ack=True thi loi hien o dong trang thai thay vi popup.
        """
        def report_error(title, message):
            if triggered_by_ack:
                self.send_status_var.set(message)
            else:
                messagebox.showerror(title, message)

        send_real = self.send_real_var.get()
        send_image = self.send_image_var.get()
        if not send_real and not send_image:
            report_error("Truyen thong", "Hay tick chon mot che do gui.")
            return

        result = self.app.shared.get("full_pipeline_result")
        if not result or not result.get("success"):
            report_error("Truyen thong", "Chua co ket qua. Hay chay Run truoc.")
            return
        rows = result["data"]["results"]
        if not rows:
            report_error("Truyen thong", "Khong co ID nao de gui.")
            return

        homography = None
        if send_real:
            try:
                homography = compute_homography(self._collect_point_pairs())
            except ValueError as exc:
                report_error(
                    "Truyen thong",
                    "Chua the gui toa do thuc. Yeu cau nhap du cap diem hieu chuan. {}".format(exc),
                )
                return

        if not self.tcp_client.connected:
            report_error("Truyen thong", "Chua ket noi TCP/IP. Hay ket noi truoc.")
            return

        # Quay vong qua danh sach ID, moi lan goi gui dung mot ID.
        if self._send_index >= len(rows):
            self._send_index = 0
        row = rows[self._send_index]
        try:
            self.tcp_client.send_text(self._format_id_line(row, homography))
        except RuntimeError as exc:
            report_error("Truyen thong", str(exc))
            self._disconnect()
            return
        mode = "toa do thuc" if send_real else "toa do anh"
        self.send_status_var.set(
            "Da gui ID{} ({}) - {}/{}.".format(
                int(row["id"]), mode, self._send_index + 1, len(rows)
            )
        )
        self._send_index += 1

    def send_results(self):
        """Nut bam: gui ID tiep theo qua TCP/IP (giong nhu nhan mot ACK)."""
        self._send_next_id(triggered_by_ack=False)

    def choose_image(self):
        path = filedialog.askopenfilename(title="Chon anh khay", initialdir=str(INPUT_DIR))
        if path:
            self.image_path_var.set(path)
            self.status_var.set("")

    def connect_camera(self):
        """Che do ket noi camera (placeholder - chua co thiet bi that).

        Hien thi trang thai camera len khung anh ben tren. Khi co camera that
        se cap nhat phan doc khung hinh sau.
        """
        self.image_path_var.set("[CAMERA] (chua ket noi thiet bi that)")
        self.status_var.set("Che do camera")
        self.image_viewer.label.configure(
            image="",
            text="Che do CAMERA\n\nChua ket noi thiet bi that.\nSe cap nhat phan doc khung hinh sau.",
        )
        self.image_viewer._photo = None

    def save_image(self):
        """Luu anh debug dang hien thi ra file (ho tro duong dan Unicode)."""
        name = self.debug_selector.var.get()
        image = self.current_images.get(name)
        if image is None:
            messagebox.showwarning("Save image", "Chua co anh de luu. Hay chay Run truoc.")
            return
        path = filedialog.asksaveasfilename(
            title="Luu anh",
            defaultextension=".png",
            initialfile="{}.png".format(name or "image"),
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Tat ca", "*.*")],
        )
        if not path:
            return
        try:
            write_image(path, image)
        except Exception as exc:
            messagebox.showerror("Save image", str(exc))
            return
        messagebox.showinfo("Save image", "Da luu anh:\n{}".format(path))

    def load_template(self):
        try:
            self.template_data = load_template_data(TEMPLATE_DATA_PATH)
            messagebox.showinfo("Template", "Da nap template_data.json")
        except Exception as exc:
            messagebox.showerror("Template", str(exc))

    def load_template_from_file(self):
        """Nap template tu file JSON nguoi dung tu chon trong thu muc bat ky."""
        path = filedialog.askopenfilename(
            title="Chon file template (.json)",
            filetypes=[("JSON", "*.json"), ("Tat ca", "*.*")],
        )
        if not path:
            return
        try:
            self.template_data = load_template_data(path)
            messagebox.showinfo("Template", "Da nap template tu:\n{}".format(path))
        except Exception as exc:
            messagebox.showerror("Template", str(exc))

    def run_step(self):
        image_path = self.image_path_var.get().strip() or self.app.shared.get("image_path", "")
        if not image_path or image_path.startswith("[CAMERA]"):
            messagebox.showwarning("Full Pipeline", "Hay chon anh khay truoc (che do camera chua san sang).")
            return
        if self._busy:
            self._pending = True
            self.status_var.set("Dang chay... se cap nhat lai")
            return
        template_data = self.template_data or self.app.shared.get("template_data")
        if template_data is None:
            try:
                template_data = load_template_data(TEMPLATE_DATA_PATH)
            except Exception as exc:
                messagebox.showwarning("Template", "Chua co template. Hay tao o tab 5. ({})".format(exc))
                return
        self.template_data = template_data
        preset_bundle = load_radial_signature_preset()
        request = {
            "image_path": image_path,
            "hough_params": self.app.shared.get("hough_params") or load_preset(HOUGH_PRESET_PATH, DEFAULT_HOUGH_PARAMS),
            "roi_params": self.app.shared.get("roi_params") or load_preset(ROI_PRESET_PATH, DEFAULT_ROI_PARAMS),
            "tab_edge_params": self.app.shared.get("tab_edge_params") or preset_bundle["tab_edge_params"],
            "radial_params": self.app.shared.get("radial_params") or preset_bundle["radial_params"],
            "template_data": template_data,
        }
        self._start_worker(request)

    def stop_step(self):
        """Dung pipeline: huy lan chay dang cho va bo ket qua sap tra ve.

        Luong xu ly chay o thread nen khong the bi huy ngay giua chung, nen
        Stop se bo qua ket qua khi no hoan tat va dua giao dien ve trang thai
        san sang.
        """
        if not self._busy:
            return
        self._stopped = True
        self._pending = None
        self.status_var.set("Da yeu cau dung...")
        self.stop_button.configure(state="disabled")

    def _start_worker(self, request):
        self._busy = True
        self._stopped = False
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Dang chay full pipeline...")
        self.log_panel.set_lines(
            [
                "Dang chay full pipeline o thread nen (GUI van responsive)...",
                "Coarse Hough (fast mode neu bat) -> ROI -> Hough refine -> tab edge -> radial -> MSE.",
            ]
        )

        def worker():
            try:
                result = run_full_pipeline(
                    request["image_path"],
                    request["hough_params"],
                    request["roi_params"],
                    request["tab_edge_params"],
                    request["radial_params"],
                    request["template_data"],
                    save_debug=False,
                )
                error = None
            except Exception as exc:
                result, error = None, exc
            self._event_queue.put((result, error))

        threading.Thread(target=worker, daemon=True).start()
        if self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._poll_worker)

    def _poll_worker(self):
        self._poll_id = None
        try:
            while True:
                result, error = self._event_queue.get_nowait()
                self._on_worker_done(result, error)
        except queue.Empty:
            pass
        if self._busy and self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._poll_worker)

    def _on_worker_done(self, result, error):
        self._busy = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_var.set("")
        if self._stopped:
            # Nguoi dung da bam Stop: bo qua ket qua lan chay nay.
            self._stopped = False
            self.status_var.set("Da dung.")
            return
        if error is not None:
            result = {"success": False, "data": {}, "images": {}, "logs": [str(error)]}
        if result["success"]:
            rows = [
                (
                    row["id"],
                    int(round(row["center_x"])),
                    int(round(row["center_y"])),
                    int(round(row["radius"])),
                    round(normalize_display_angle_deg(row["angle_deg"]), 2),
                    round(row["min_error"], 4),
                    row["status"],
                )
                for row in result["data"]["results"]
            ]
            self.table.set_rows(rows)
            self._send_index = 0  # ket qua moi -> gui lai tu ID dau khi co ACK
            self.app.shared["full_pipeline_result"] = result
            self.set_result(result)
        else:
            self.table.set_rows([])
            self.set_result(result)

        if self._pending:
            self._pending = None
            self.run_step()
