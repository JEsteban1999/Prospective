"""Longitudinal timeline chart widget — Feature 6.

Embeds a Matplotlib figure showing the evolution of up to two morphometric
metrics across sessions.  A second Y-axis is used for the optional overlay
metric so scales are independent.

The chart uses the same dark-theme palette as the cross-section chart.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy

logger = logging.getLogger(__name__)

# Dark-theme palette (violet brand system)
_BG    = "#111111"
_FG    = "#EBEBEB"
_GRID  = "#363636"
_BLUE  = "#A8B8C6"
_GREEN = "#3fb950"
_RED   = "#f85149"
_AMBE  = "#d29922"


class LongitudinalChart(QWidget):
    """
    Matplotlib chart showing metric(s) vs session date.

    Usage::

        chart = LongitudinalChart(parent)
        chart.plot(series, primary_metric="volume_mm3",
                   secondary_metric="aspect_ratio")
        chart.clear()
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fig = Figure(figsize=(5, 2.8), dpi=96, facecolor=_BG)
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

    def plot(
        self,
        series,
        primary_metric: str   = "volume_mm3",
        secondary_metric: str | None = None,
    ) -> None:
        """
        Render the longitudinal chart.

        Parameters
        ----------
        series           : LongitudinalSeries
        primary_metric   : key from TRACKED_METRICS
        secondary_metric : optional second metric plotted on a right Y-axis
        """
        from prospective.processing.longitudinal import METRIC_LABELS

        ax = self._ax
        ax.clear()
        if hasattr(ax, "_ax2") and ax._ax2 is not None:
            try:
                ax._ax2.remove()
            except Exception:
                pass
            ax._ax2 = None

        self._style_axes(ax)

        if len(series) < 1:
            self._draw_placeholder(ax, "Sin datos longitudinales")
            self._canvas.draw_idle()
            return

        days   = series.days_from_first()
        dates  = series.dates()
        x_tick_labels = [d.strftime("%Y-%m-%d") for d in dates]

        # Primary metric
        primary_vals = series.values(primary_metric)
        primary_lbl  = METRIC_LABELS.get(primary_metric, primary_metric)

        ax.plot(days, primary_vals, "o-", color=_BLUE,
                linewidth=1.8, markersize=5, label=primary_lbl)

        # Trend line
        slope, r2 = series.trend(primary_metric)
        if len(days) >= 2 and abs(slope) > 1e-9:
            x_arr = np.array(days)
            y_fit = slope * x_arr + (np.array(primary_vals).mean()
                                     - slope * x_arr.mean())
            ax.plot(x_arr, y_fit, "--", color=_BLUE, linewidth=1.0,
                    alpha=0.55, label=f"Tendencia (R²={r2:.2f})")

        ax.set_ylabel(primary_lbl, fontsize=7, color=_BLUE)
        ax.tick_params(axis="y", colors=_BLUE, labelsize=6)

        # Secondary metric on twin axis
        if secondary_metric and secondary_metric != primary_metric:
            sec_vals = series.values(secondary_metric)
            sec_lbl  = METRIC_LABELS.get(secondary_metric, secondary_metric)
            ax2 = ax.twinx()
            ax2.set_facecolor(_BG)
            for sp in ax2.spines.values():
                sp.set_edgecolor(_GRID)
            ax2.plot(days, sec_vals, "s--", color=_GREEN,
                     linewidth=1.4, markersize=4, label=sec_lbl)
            ax2.set_ylabel(sec_lbl, fontsize=7, color=_GREEN)
            ax2.tick_params(axis="y", colors=_GREEN, labelsize=6)
            ax._ax2 = ax2
        else:
            ax._ax2 = None

        # X axis: show dates
        ax.set_xticks(days)
        ax.set_xticklabels(x_tick_labels, rotation=30, ha="right", fontsize=6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        if hasattr(ax, "_ax2") and ax._ax2 is not None:
            h2, l2 = ax._ax2.get_legend_handles_labels()
            handles += h2
            labels  += l2
        if handles:
            ax.legend(handles, labels, fontsize=6, loc="upper left",
                      facecolor=_BG, edgecolor=_GRID, labelcolor=_FG)

        self._fig.tight_layout(pad=0.8)
        self._canvas.draw_idle()

    def clear(self) -> None:
        ax = self._ax
        ax.clear()
        if hasattr(ax, "_ax2") and ax._ax2 is not None:
            try:
                ax._ax2.remove()
            except Exception:
                pass
            ax._ax2 = None
        self._style_axes(ax)
        self._init_axes()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _init_axes(self) -> None:
        self._ax._ax2 = None
        self._draw_placeholder(self._ax, "Sin datos — agrega sesiones para comparar")
        self._canvas.draw_idle()

    def _draw_placeholder(self, ax, text: str) -> None:
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color=_GRID)

    def _style_axes(self, ax) -> None:
        ax.set_facecolor(_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(_GRID)
        ax.tick_params(colors=_FG, labelsize=6)
        ax.grid(True, color=_GRID, linewidth=0.5, linestyle="--")
