"""Cross-section diameter profile chart widget — Feature 2.

Embeds a Matplotlib figure inside a QWidget showing:
  • Equivalent diameter (mm) vs arc-length position (mm) — solid blue line
  • Mean diameter — dashed grey reference line
  • Min/max shaded band
  • Stenosis marker if stenosis_ratio < 0.7
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy


# ──────────────────────────────────────────────────────────────────────────── #
# Chart widget                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

class CrossSectionChart(QWidget):
    """
    A compact Matplotlib chart showing the vessel diameter profile.

    Usage::

        chart = CrossSectionChart(parent)
        chart.plot(cross_section_result)
        chart.clear()
    """

    # Dark-theme palette matching the application style
    _BG     = "#111111"
    _FG     = "#EBEBEB"
    _GRID   = "#363636"
    _BLUE   = "#A8B8C6"
    _BAND   = "#8B9BAA"
    _MEAN   = "#9B9B9B"
    _RED    = "#f85149"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._fig = Figure(figsize=(4, 2.2), dpi=96, facecolor=self._BG)
        self._ax  = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._init_axes()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def plot(self, result) -> None:
        """
        Render the diameter profile from a :class:`CrossSectionResult`.

        Parameters
        ----------
        result : CrossSectionResult
        """
        ax = self._ax
        ax.clear()
        self._style_axes(ax)

        arc  = result.arc_positions_mm
        diam = result.diameters_mm
        mean = result.mean_diameter_mm

        # Main diameter line
        ax.plot(arc, diam, color=self._BLUE, linewidth=1.5, label="Ø equiv.")

        # Mean reference line
        ax.axhline(mean, color=self._MEAN, linewidth=1.0,
                   linestyle="--", label=f"Media {mean:.2f} mm")

        # Min / max shaded band
        ax.fill_between(arc, result.min_diameter_mm, result.max_diameter_mm,
                        color=self._BAND, alpha=0.15, linewidth=0)

        # Stenosis marker at minimum
        if result.stenosis_ratio < 0.7:
            min_idx = int(np.argmin(diam))
            ax.plot(arc[min_idx], diam[min_idx], "v", color=self._RED,
                    markersize=6, zorder=5,
                    label=f"Estenosis {(1 - result.stenosis_ratio)*100:.0f}%")

        # Legend + labels
        ax.set_xlabel("Posición arco (mm)", fontsize=7, color=self._FG)
        ax.set_ylabel("Ø (mm)", fontsize=7, color=self._FG)
        ax.legend(fontsize=6, loc="upper right",
                  facecolor=self._BG, edgecolor=self._GRID,
                  labelcolor=self._FG)

        self._fig.tight_layout(pad=0.8)
        self._canvas.draw_idle()

    def clear(self) -> None:
        """Remove the profile and show the empty placeholder."""
        self._ax.clear()
        self._style_axes(self._ax)
        self._init_axes()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _init_axes(self) -> None:
        ax = self._ax
        self._style_axes(ax)
        ax.text(
            0.5, 0.5,
            "Sin datos — extrae la línea central\ny luego analiza secciones",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=7, color=self._MEAN,
        )
        self._canvas.draw_idle()

    def _style_axes(self, ax) -> None:
        ax.set_facecolor(self._BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(self._GRID)
        ax.tick_params(colors=self._FG, labelsize=6)
        ax.grid(True, color=self._GRID, linewidth=0.5, linestyle="--")
