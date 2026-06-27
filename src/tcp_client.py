"""TCP/IP client don gian de truyen thong voi thiet bi/server khac.

Phan mem dong vai tro CLIENT, ket noi toi mot server (vi du Hercules dang o che
do TCP Server) theo host:port. Dung de mo phong truyen thong: mo ket noi, gui
chuoi du lieu (toa do anh hoac toa do thuc sau hieu chuan), nhan ACK tu server,
roi dong ket noi.
"""

import socket
import threading


class TcpClient:
    """Wrapper mong quanh socket TCP client, an toan goi tu GUI thread."""

    def __init__(self):
        self._sock = None
        self._lock = threading.Lock()
        self._listener = None
        self._stop_event = threading.Event()
        self.host = None
        self.port = None

    @property
    def connected(self):
        return self._sock is not None

    def connect(self, host, port, timeout=5.0):
        """Mo ket noi toi server host:port. Raise neu that bai."""
        with self._lock:
            if self._sock is not None:
                raise RuntimeError("Da ket noi roi. Hay ngat ket noi truoc.")
            sock = socket.create_connection((host, int(port)), timeout=timeout)
            sock.settimeout(timeout)
            self._sock = sock
            self.host = host
            self.port = int(port)

    def start_listening(self, on_message):
        """Chay thread nen doc du lieu server gui ve.

        on_message(text) duoc goi voi moi chuoi nhan duoc (tu thread nen), va
        goi on_message(None) mot lan khi ket noi bi dong. Phia GUI nen day vao
        queue roi xu ly o main thread, khong dung tkinter truc tiep trong callback.
        """
        sock = self._sock
        if sock is None:
            raise RuntimeError("Chua ket noi TCP/IP.")
        self._stop_event.clear()
        self._listener = threading.Thread(
            target=self._listen_loop, args=(sock, on_message), daemon=True
        )
        self._listener.start()

    def _listen_loop(self, sock, on_message):
        while not self._stop_event.is_set():
            try:
                data = sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break  # server dong ket noi
            on_message(data.decode("utf-8", errors="replace"))
        if not self._stop_event.is_set():
            on_message(None)

    def disconnect(self):
        """Dong ket noi neu dang mo va dung thread lang nghe."""
        with self._lock:
            self._stop_event.set()
            if self._sock is not None:
                try:
                    self._sock.close()
                finally:
                    self._sock = None
                    self.host = None
                    self.port = None
        self._listener = None

    def send_text(self, text):
        """Gui mot chuoi (UTF-8) toi server. Raise neu chua ket noi/loi gui."""
        with self._lock:
            if self._sock is None:
                raise RuntimeError("Chua ket noi TCP/IP.")
            try:
                self._sock.sendall(text.encode("utf-8"))
            except OSError as exc:
                # Ket noi hong -> don dep de lan sau biet la da ngat.
                self._stop_event.set()
                try:
                    self._sock.close()
                finally:
                    self._sock = None
                    self.host = None
                    self.port = None
                raise RuntimeError("Loi gui du lieu, ket noi da dong: {}".format(exc))
