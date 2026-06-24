"""Main Tkinter application for detect_stator."""

import tkinter as tk
from tkinter import ttk

from src.io_utils import ensure_project_dirs
from src.preset_store import ensure_default_presets

from .full_pipeline_panel import FullPipelinePanel
from .hough_step_panel import HoughStepPanel
from .matching_step_panel import MatchingStepPanel
from .radial_step_panel import RadialStepPanel
from .roi_step_panel import RoiStepPanel
from .tab_edge_step_panel import TabEdgeStepPanel
from .template_step_panel import TemplateStepPanel


class StatorVisionApp:
    """Tkinter app that hosts all 7 pipeline step panels."""

    def __init__(self, root):
        self.root = root
        self.shared = {}
        ensure_project_dirs()
        ensure_default_presets()

        root.title("detect_stator - Step Debug GUI")
        root.geometry("1680x980")
        root.minsize(1320, 840)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        self.hough_panel = HoughStepPanel(notebook, self)
        self.roi_panel = RoiStepPanel(notebook, self)
        self.tab_edge_panel = TabEdgeStepPanel(notebook, self)
        self.radial_panel = RadialStepPanel(notebook, self)
        self.template_panel = TemplateStepPanel(notebook, self)
        self.matching_panel = MatchingStepPanel(notebook, self)
        self.full_pipeline_panel = FullPipelinePanel(notebook, self)

        notebook.add(self.hough_panel, text="1. Hough")
        notebook.add(self.roi_panel, text="2. ROI")
        notebook.add(self.tab_edge_panel, text="3. Tab Edges")
        notebook.add(self.radial_panel, text="4. Radial Signature")
        notebook.add(self.template_panel, text="5. Template 0°")
        notebook.add(self.matching_panel, text="6. Match MSE")
        notebook.add(self.full_pipeline_panel, text="7. Full Pipeline")
