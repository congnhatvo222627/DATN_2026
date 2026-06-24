"""Launch the Tkinter GUI."""

import tkinter as tk

from gui.main_app import StatorVisionApp


def main():
    """Create and run the main GUI."""
    root = tk.Tk()
    StatorVisionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
