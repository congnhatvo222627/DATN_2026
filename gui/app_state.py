"""Shared GUI state container.

Giu API dang dict de cac panel hien tai tiep tuc doc/ghi nhu cu, nhung gom
toan bo state chung vao mot lop rieng de de theo doi va mo rong sau nay.
"""

class AppState:
    """Dictionary-like store shared across GUI panels."""

    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def setdefault(self, key, default=None):
        return self._data.setdefault(key, default)

    def update(self, other, **kwargs):
        self._data.update(other, **kwargs)

    def clear(self):
        self._data.clear()

    def items(self):
        return self._data.items()

    def snapshot(self):
        """Return a shallow copy for debug/logging when needed."""
        return dict(self._data)
