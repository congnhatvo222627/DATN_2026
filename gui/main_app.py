"""Main Tkinter application for detect_stator."""

from tkinter import ttk

from PIL import Image, ImageTk

from src.config import APP_TITLE, LOGO_PATH
from src.io_utils import ensure_project_dirs
from src.preset_store import ensure_default_presets

from .app_state import AppState
from .full_pipeline_panel import FullPipelinePanel
from .hough_step_panel import HoughStepPanel
from .matching_step_panel import MatchingStepPanel
from .radial_signature_panel import RadialSignaturePanel
from .roi_step_panel import RoiStepPanel
from .template_step_panel import TemplateStepPanel
from .theme import apply_theme


PANEL_SPECS = [
    ("hough_panel", HoughStepPanel, "  1 · Hough  "),
    ("roi_panel", RoiStepPanel, "  2 · ROI  "),
    ("radial_signature_panel", RadialSignaturePanel, "  3 · Radial Signature  "),
    ("template_panel", TemplateStepPanel, "  4 · Template 0°  "),
    ("matching_panel", MatchingStepPanel, "  5 · Match MSE  "),
    ("full_pipeline_panel", FullPipelinePanel, "  6 · Full Pipeline  "),
]


class StatorVisionApp:
    """Tkinter app that hosts all 6 pipeline step panels."""

    def __init__(self, root):
        self.root = root
        self.shared = AppState()
        ensure_project_dirs()
        ensure_default_presets()

        root.title(APP_TITLE)
        root.geometry("1680x980")
        root.minsize(1320, 840)
        apply_theme(root)
        self._set_app_icon(root)
        # Mo to toan man hinh cho vua layout (man hinh nho van thay du nut).
        try:
            root.state("zoomed")
        except Exception:
            pass

        body = ttk.Frame(root, padding=(10, 8, 10, 10))
        body.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(fill="both", expand=True)

        self._build_panels()
        self.tab_edge_panel = self.radial_signature_panel
        self.radial_panel = self.radial_signature_panel

    def _build_panels(self):
        """Create notebook tabs in one place to keep wiring easy to maintain."""
        for attr_name, panel_cls, tab_title in PANEL_SPECS:
            panel = panel_cls(self.notebook, self)
            setattr(self, attr_name, panel)
            self.notebook.add(panel, text=tab_title)

    def _set_app_icon(self, root):
        """Dat logo Bach Khoa lam icon cua so (thay icon chiec la mac dinh)."""
        try:
            image = Image.open(LOGO_PATH)
            self._app_icon = ImageTk.PhotoImage(image)
            root.iconphoto(True, self._app_icon)
        except Exception:
            # Khong co logo cung khong sao, giu icon mac dinh.
            pass
