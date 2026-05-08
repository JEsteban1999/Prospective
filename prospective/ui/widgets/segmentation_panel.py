"""Segmentation dock panel — Alternative B design (3-slider interface).

Three simple sliders control the most important parameters:
  • Umbral HU   — iso-surface threshold
  • Limpieza    — fragment/topology cleanup (0-10 scale)
  • Suavizado   — smoothing + decimation (0-10 scale)

An optional "▶ Avanzado" collapsible section exposes the individual
pipeline parameters for expert users.

Signals
-------
seg_ready(SegmentationResult | GrowResult)   — full-quality segmentation done
preview_ready(SegmentationResult)            — fast low-res preview ready (MPR only)
grow_ready(GrowResult)                       — grow-from-seeds done
mesh_visibility_changed(bool)
volume_visibility_changed(bool)
export_stl_requested / export_obj_requested
"""
from __future__ import annotations

import logging

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from prospective.processing.segmentation import (
    GrowResult,
    GrowSegmentationPipeline,
    SegmentationPipeline,
    SegmentationResult,
)
from prospective.ui.icons import I as _I

logger = logging.getLogger(__name__)


def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True

# Debounce delay before firing the fast preview (ms)
_PREVIEW_DEBOUNCE_MS = 400

# ──────────────────────────────────────────────────────────────────────────── #
# Mapping tables                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

# (top_n, min_component_verts, morpho_closing_mm)
_LIMPIEZA_MAP = [
    (0,    0,  0.0),  # 0 — Ninguna
    (0,   30,  0.0),  # 1
    (0,  100,  0.0),  # 2
    (0,  200,  0.0),  # 3
    (0,  500,  0.5),  # 4
    (20,   0,  0.5),  # 5 — Media
    (15,   0,  0.5),  # 6
    (10,   0,  0.5),  # 7
    (7,    0,  1.0),  # 8
    (5,    0,  1.0),  # 9
    (3,    0,  1.0),  # 10 — Máxima
]

_LIMPIEZA_LABELS = [
    "Ninguna", "Mínima", "Baja", "Baja+",
    "Media−", "Media", "Media+",
    "Alta−", "Alta", "Muy alta", "Máxima",
]

# (gaussian_sigma, smooth_iterations, target_reduction)
_SUAVIZADO_MAP = [
    (0.0,  0, 0.40),  # 0 — Ninguno
    (0.0,  5, 0.50),  # 1
    (0.0, 10, 0.55),  # 2
    (0.3, 10, 0.60),  # 3
    (0.3, 15, 0.65),  # 4
    (0.3, 20, 0.70),  # 5 — Estándar
    (0.5, 20, 0.70),  # 6
    (0.5, 25, 0.75),  # 7
    (0.5, 30, 0.75),  # 8
    (0.8, 35, 0.80),  # 9
    (1.0, 40, 0.85),  # 10 — Máximo
]

_SUAVIZADO_LABELS = [
    "Ninguno", "Mínimo", "Bajo", "Bajo+",
    "Medio−", "Estándar", "Medio+",
    "Alto−", "Alto", "Muy alto", "Máximo",
]


def _make_bar(value: int, total: int = 10) -> str:
    filled = round(value * 8 / total)   # scale to 8 chars
    return "■" * filled + "□" * (8 - filled)


# ──────────────────────────────────────────────────────────────────────────── #
# Background workers                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

class _SegWorker(QThread):
    finished = pyqtSignal(object)   # SegmentationResult
    error    = pyqtSignal(str)

    def __init__(self, pipeline, volume, spacing, parent=None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._volume   = volume
        self._spacing  = spacing

    def run(self) -> None:
        try:
            result = self._pipeline.run(self._volume, self._spacing)
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Segmentation failed")
            self.error.emit(str(exc))


class _PreviewWorker(QThread):
    """Runs run_fast_preview() on a background thread."""
    finished = pyqtSignal(object)   # SegmentationResult (is_preview=True)
    error    = pyqtSignal(str)

    def __init__(self, pipeline, volume, spacing, parent=None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._volume   = volume
        self._spacing  = spacing

    def run(self) -> None:
        try:
            result = self._pipeline.run_fast_preview(self._volume, self._spacing)
            self.finished.emit(result)
        except Exception as exc:
            logger.debug("Preview failed: %s", exc)
            self.error.emit(str(exc))


class _GrowWorker(QThread):
    """Background thread for grow-from-seeds segmentation (S-3)."""
    finished = pyqtSignal(object)   # GrowResult
    error    = pyqtSignal(str)

    def __init__(self, pipeline, volume, spacing, seeds, parent=None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._volume   = volume
        self._spacing  = spacing
        self._seeds    = seeds

    def run(self) -> None:
        try:
            result = self._pipeline.run(self._volume, self._spacing, self._seeds)
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Grow-from-seeds failed")
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────── #
# Panel widget                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

class SegmentationPanel(QWidget):
    """
    Self-contained segmentation controls — Alternative B (3-slider) design.

    Signals
    -------
    seg_ready(result)         — full-quality segmentation completed
    preview_ready(result)     — fast low-res preview ready (MPR viewer only)
    grow_ready(result)        — grow-from-seeds completed
    mesh_visibility_changed(bool)
    volume_visibility_changed(bool)
    export_stl_requested / export_obj_requested
    """

    seg_ready                 = pyqtSignal(object)
    preview_ready             = pyqtSignal(object)
    grow_ready                = pyqtSignal(object)
    mesh_visibility_changed   = pyqtSignal(bool)
    volume_visibility_changed = pyqtSignal(bool)
    export_stl_requested      = pyqtSignal()
    export_obj_requested      = pyqtSignal()
    request_mpr_seed          = pyqtSignal()   # ask main window for current MPR position
    request_auto_threshold    = pyqtSignal()   # ask main window for last cursor HU

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._volume        = None
        self._spacing       = None
        self._worker:         _SegWorker     | None = None
        self._preview_worker: _PreviewWorker | None = None
        self._grow_worker:    _GrowWorker    | None = None
        self._last_result:  SegmentationResult | GrowResult | None = None
        self._grow_seeds:   list[tuple[int, int, int]] = []
        self._custom_mode   = False
        self._modality      = "CT"

        # Debounce timer for fast preview
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(_PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._run_preview)

        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_session_state(self) -> dict:
        return {
            "threshold_hu":     self._thr_spin.value(),
            "threshold_max_hu": self._thr_max_spin.value(),
            "thr_max_enabled":  self._chk_thr_max.isChecked(),
            "limpieza":         self._lim_slider.value(),
            "suavizado":        self._suav_slider.value(),
            "adv_sigma":        self._adv_sigma.value(),
            "adv_smooth":       self._adv_smooth.value(),
            "adv_frags":        self._adv_frags.value(),
            "adv_closing":      self._adv_closing.value(),
            "adv_top_n":        self._adv_top_n.value(),
            "adv_dec":          self._adv_dec_slider.value(),
            "adv_vessel_hu":    self._adv_vessel_hu.value(),
            "adv_vessel_dil":   self._adv_vessel_dil.value(),
            "auto_preview":     self._chk_preview.isChecked(),
        }

    def restore_session_state(self, d: dict) -> None:
        """Restore all panel controls from a saved state dict.

        All widget signals are blocked for the duration of the restore so that
        slider handlers (_on_limpieza_changed, _on_suavizado_changed) cannot
        overwrite the saved advanced-spinbox values with their preset mappings.
        Display labels and dependent state are refreshed manually afterwards.
        """
        _widgets = [
            self._thr_slider, self._thr_spin,
            self._thr_max_spin, self._chk_thr_max,
            self._lim_slider, self._suav_slider,
            self._adv_sigma, self._adv_smooth, self._adv_frags,
            self._adv_closing, self._adv_top_n, self._adv_dec_slider,
            self._adv_vessel_hu, self._adv_vessel_dil,
            self._chk_preview,
        ]
        for w in _widgets:
            w.blockSignals(True)
        try:
            # ── Threshold ──────────────────────────────────────────────────── #
            thr_hu = int(d.get("threshold_hu", 200))
            self._thr_spin.setValue(thr_hu)
            self._thr_slider.setValue(thr_hu)          # keep slider in sync

            self._thr_max_spin.setValue(int(d.get("threshold_max_hu", 600)))
            thr_max_on = bool(d.get("thr_max_enabled", False))
            self._chk_thr_max.setChecked(thr_max_on)
            self._thr_max_spin.setEnabled(thr_max_on)  # direct property, no signal needed

            # ── Simple sliders ─────────────────────────────────────────────── #
            self._lim_slider.setValue(int(d.get("limpieza", 5)))
            self._suav_slider.setValue(int(d.get("suavizado", 5)))

            # ── Advanced spinboxes ─────────────────────────────────────────── #
            # Set these AFTER the sliders so their saved values are the final
            # ones — slider handlers would otherwise overwrite them.
            self._adv_sigma.setValue(float(d.get("adv_sigma", 0.3)))
            self._adv_smooth.setValue(int(d.get("adv_smooth", 20)))
            self._adv_frags.setValue(int(d.get("adv_frags", 0)))
            self._adv_closing.setValue(float(d.get("adv_closing", 0.5)))
            self._adv_top_n.setValue(int(d.get("adv_top_n", 20)))
            self._adv_dec_slider.setValue(int(d.get("adv_dec", 70)))
            self._adv_vessel_hu.setValue(int(d.get("adv_vessel_hu", 0)))
            self._adv_vessel_dil.setValue(float(d.get("adv_vessel_dil", 3.0)))

            # ── Misc ───────────────────────────────────────────────────────── #
            self._chk_preview.setChecked(bool(d.get("auto_preview", True)))
        finally:
            for w in _widgets:
                w.blockSignals(False)

        # ── Manually refresh display labels (normally driven by signals) ── #
        lim_v = self._lim_slider.value()
        self._lim_bar_lbl.setText(
            f"{_make_bar(lim_v)}  {_LIMPIEZA_LABELS[lim_v]}"
        )

        suav_v = self._suav_slider.value()
        self._suav_bar_lbl.setText(
            f"{_make_bar(suav_v)}  {_SUAVIZADO_LABELS[suav_v]}"
        )

        dec_v = self._adv_dec_slider.value()
        self._adv_dec_lbl.setText(f"{dec_v}%")

        # Refresh vessel-enhancement status label
        self._on_vessel_hu_changed(self._adv_vessel_hu.value())

    def set_volume(self, volume, spacing) -> None:
        """Called when a new DICOM series is loaded."""
        self._volume  = volume
        self._spacing = spacing
        self._btn_seg.setEnabled(True)
        self._lbl_stats.setText("—")
        self._set_export_enabled(False)
        # Reset grow seeds
        self._grow_seeds = []
        self._lbl_seeds.setText("Sin semillas")
        self._btn_grow.setEnabled(False)
        nz, ny, nx = volume.shape
        self._spin_seed_z.setRange(0, nz - 1)
        self._spin_seed_y.setRange(0, ny - 1)
        self._spin_seed_x.setRange(0, nx - 1)
        self._spin_seed_z.setValue(nz // 2)
        self._spin_seed_y.setValue(ny // 2)
        self._spin_seed_x.setValue(nx // 2)

    def apply_auto_threshold(self, hu: float) -> None:
        """
        Set the lower threshold to 80 % of *hu* and disable the upper bound.

        This is the "one-click" workflow: the user hovers over a bright
        vessel (HU is shown in the MPR footer) and presses ⚡.  The system
        sets the threshold just below that value so the whole contrast bolus
        is captured without requiring any manual HU knowledge.

        For XA / 3DRA data where vessels are the BRIGHTEST structures
        (e.g. HU ≈ 4 000–5 000) this gives a clean vessel mesh with a
        single lower threshold and no upper bound needed.
        """
        if hu <= 0:
            return

        lower = int(hu * 0.80)

        for w in (self._thr_slider, self._thr_spin):
            w.blockSignals(True)
        self._thr_slider.setValue(lower)
        self._thr_spin.setValue(lower)
        for w in (self._thr_slider, self._thr_spin):
            w.blockSignals(False)

        # Disable the upper threshold — in contrast 3DRA vessels are the
        # brightest structure so no upper bound is needed.
        self._chk_thr_max.blockSignals(True)
        self._chk_thr_max.setChecked(False)
        self._thr_max_spin.setEnabled(False)
        self._chk_thr_max.blockSignals(False)

        # Also update the grow-from-seeds HU range
        lo_seed = int(hu * 0.75)
        hi_seed = int(hu * 1.10)
        self._spin_grow_lo.blockSignals(True)
        self._spin_grow_hi.blockSignals(True)
        self._spin_grow_lo.setValue(lo_seed)
        self._spin_grow_hi.setValue(hi_seed)
        self._spin_grow_lo.blockSignals(False)
        self._spin_grow_hi.blockSignals(False)

        logger.info(
            "Auto-threshold from cursor HU=%.0f → lower=%d  seeds=[%d, %d]",
            hu, lower, lo_seed, hi_seed,
        )

        # Fire preview immediately so the user sees the result right away
        self._schedule_preview()

    def set_modality(
        self,
        modality: str,
        window_center: float = 40.0,
        window_width: float = 400.0,
    ) -> None:
        """
        Configure segmentation defaults based on DICOM modality.

        For XA / RF / DX (X-ray based modalities) the pipeline activates
        the band-pass filter automatically: voxels above the upper threshold
        (typically the skull/calvaria) are excluded from the mesh.

        Parameters
        ----------
        modality:      DICOM Modality tag value ("CT", "XA", "MR", …).
        window_center: DICOM WindowCenter from the loaded series header.
        window_width:  DICOM WindowWidth from the loaded series header.
        """
        self._modality = modality.upper()
        _XA_MODALITIES = {"XA", "RF", "DX", "CR", "DR"}

        if self._modality in _XA_MODALITIES:
            # --- Derive band-pass thresholds -----------------------------------
            # XA DICOM headers often carry a display window (WC/WW) that spans
            # the ENTIRE data range (e.g. WW=7578 → useless for segmentation).
            # When WW > 2000 we fall back to actual volume statistics: contrast-
            # filled vessels are in the upper-bright minority, dense bone/skull is
            # the very top sliver.  Percentiles on the real data are far more
            # robust than the header metadata.
            if self._volume is not None and window_width > 2000:
                # --- Percentile-based band-pass for wide-window 3DRA data -------
                # In contrast-enhanced 3DRA (rotational angiography), contrast
                # vessels are the BRIGHTEST structures.  We use a BAND-PASS:
                #   lower = p90  — bottom of the brightest 10 % (vessels + bone)
                #   upper = p99  — top cut-off to exclude saturated/artifact pixels
                #                  (typically scanner bed, collimator edges, etc.)
                # The 9-percentile-wide band captures vessel contrast while
                # discarding the very-bright saturated artefacts that generate
                # large noisy fragments in the mesh.
                flat = self._volume.ravel().astype("float32")
                lower = float(np.percentile(flat, 90))
                upper = float(np.percentile(flat, 99))
                logger.info(
                    "XA band-pass from volume percentiles: "
                    "lower=%.0f (p90)  upper=%.0f (p99)  (WW=%.0f too wide)",
                    lower, upper, window_width,
                )
            else:
                # Narrower-window data (e.g. cone-beam CT with calibrated WC/WW)
                lower = max(-200.0, window_center - window_width * 0.35)
                upper = window_center + window_width * 0.45
                logger.info(
                    "XA band-pass from WC/WW: lower=%.0f  upper=%.0f",
                    lower, upper,
                )

            # Block signals to avoid triggering two preview runs
            for w in (self._thr_slider, self._thr_spin, self._thr_max_spin):
                w.blockSignals(True)
            self._thr_slider.setValue(int(lower))
            self._thr_spin.setValue(int(lower))
            self._thr_max_spin.setValue(int(upper))
            for w in (self._thr_slider, self._thr_spin, self._thr_max_spin):
                w.blockSignals(False)

            # Always enable the band-pass filter for XA (upper > lower guaranteed)
            self._chk_thr_max.blockSignals(True)
            self._chk_thr_max.setChecked(True)
            self._thr_max_spin.setEnabled(True)
            self._chk_thr_max.blockSignals(False)

            # Pre-fill grow-from-seeds HU range (tight around the vessel band)
            seed_hi = int(upper)
            self._spin_grow_lo.blockSignals(True)
            self._spin_grow_hi.blockSignals(True)
            self._spin_grow_lo.setValue(int(lower))
            self._spin_grow_hi.setValue(seed_hi)
            self._spin_grow_lo.blockSignals(False)
            self._spin_grow_hi.blockSignals(False)

            # Auto-set Limpieza = 7 and Suavizado = 6 for XA data:
            # XA meshes are inherently noisier than CT — more aggressive
            # fragment removal and smoothing produce a usable result.
            # We block signals to avoid double-firing _schedule_preview.
            for w in (self._lim_slider, self._suav_slider):
                w.blockSignals(True)
            self._lim_slider.setValue(7)
            self._suav_slider.setValue(6)
            for w in (self._lim_slider, self._suav_slider):
                w.blockSignals(False)
            self._on_limpieza_changed(7)   # sync advanced spinboxes + bar label
            self._on_suavizado_changed(6)  # (these methods block their own signals)

            self._lbl_xa_hint.setVisible(True)
            # Auto-expand the Advanced section so users can see the XA indicator
            # and the Umbral máx. control without having to open it manually
            self._adv_widget.setVisible(True)
            self._btn_adv.setText("▼ Avanzado")
            logger.info(
                "XA modality — band-pass: lower=%.0f  upper=%.0f  "
                "Limpieza=7  Suavizado=6",
                lower, upper,
            )
        else:
            self._lbl_xa_hint.setVisible(False)

    def add_grow_seed(self, z: int, y: int, x: int) -> None:
        self._grow_seeds.append((int(z), int(y), int(x)))
        self._update_seeds_label()
        if self._volume is not None:
            self._btn_grow.setEnabled(True)

    @property
    def last_result(self) -> SegmentationResult | GrowResult | None:
        return self._last_result

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        # Outer layout holds only the scroll area
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        scroll.setWidget(container)

        # ── 1. Parámetros de segmentación ─────────────────────────────── #
        params_grp = QGroupBox("Parámetros de segmentación")
        pf = QFormLayout()
        pf.setLabelAlignment(Qt.AlignRight)
        pf.setSpacing(6)

        # ---- Umbral HU (slider + spinbox, synced) ----
        thr_row = QWidget()
        thr_lay = QHBoxLayout(thr_row)
        thr_lay.setContentsMargins(0, 0, 0, 0)
        thr_lay.setSpacing(4)

        self._thr_slider = QSlider(Qt.Horizontal)
        self._thr_slider.setRange(-200, 10000)
        self._thr_slider.setValue(200)
        self._thr_slider.setToolTip(
            "Valor umbral de la iso-superficie (mínimo):\n"
            "  • CTA cerebral (vasos con contraste): 150–250 HU\n"
            "  • Hueso / cráneo CTA: 400–700 HU\n\n"
            "Para XA / 3DRA (angiografía rotacional):\n"
            "  • El umbral mínimo debe capturar los vasos con contraste.\n"
            "  • Activa también el Umbral máx. para excluir el cráneo.\n"
            "  • Rango típico: WC − WW×0.35  …  WC + WW×0.45\n"
            "    (WC y WW del header DICOM de la serie).\n\n"
            "Umbral bajo → más tejido capturado.\n"
            "Umbral alto → solo estructuras muy brillantes."
        )
        thr_lay.addWidget(self._thr_slider, stretch=1)

        self._thr_spin = QSpinBox()
        self._thr_spin.setRange(-200, 10000)
        self._thr_spin.setValue(200)
        self._thr_spin.setSuffix(" HU")
        self._thr_spin.setFixedWidth(80)
        thr_lay.addWidget(self._thr_spin)

        # Bidirectional sync + preview
        self._thr_slider.valueChanged.connect(self._thr_spin.setValue)
        self._thr_spin.valueChanged.connect(self._thr_slider.setValue)
        self._thr_slider.valueChanged.connect(self._schedule_preview)
        pf.addRow("Umbral HU:", thr_row)

        # ---- Auto-umbral desde cursor ----
        # Button kept in the model but hidden: the per-pixel HU cursor tracking
        # (sigMouseMoved) was removed from SliceWidget because it caused severe
        # visual glitches across all four views.
        self._btn_auto_thr = QPushButton("⚡  Auto-umbral desde cursor")
        self._btn_auto_thr.setVisible(False)   # hidden — HU tracking removed
        self._btn_auto_thr.clicked.connect(self.request_auto_threshold)

        # ---- Limpieza slider ----
        lim_row = QWidget()
        lim_lay = QHBoxLayout(lim_row)
        lim_lay.setContentsMargins(0, 0, 0, 0)
        lim_lay.setSpacing(6)

        self._lim_slider = QSlider(Qt.Horizontal)
        self._lim_slider.setRange(0, 10)
        self._lim_slider.setValue(5)
        self._lim_slider.setToolTip(
            "Controla la agresividad del filtrado de ruido y fragmentos.\n\n"
            "0 — Ninguna: conserva todo (máximo detalle, más ruido).\n"
            "5 — Media: equilibrio recomendado para CTA.\n"
            "10 — Máxima: solo los 3 componentes más grandes."
        )
        lim_lay.addWidget(self._lim_slider, stretch=1)

        self._lim_bar_lbl = QLabel()
        self._lim_bar_lbl.setStyleSheet(
            "color:#A8B8C6; font-family:monospace; font-size:11px;"
        )
        self._lim_bar_lbl.setFixedWidth(130)
        lim_lay.addWidget(self._lim_bar_lbl)

        pf.addRow("Limpieza:", lim_row)
        self._lim_slider.valueChanged.connect(self._on_limpieza_changed)

        # ---- Suavizado slider ----
        suav_row = QWidget()
        suav_lay = QHBoxLayout(suav_row)
        suav_lay.setContentsMargins(0, 0, 0, 0)
        suav_lay.setSpacing(6)

        self._suav_slider = QSlider(Qt.Horizontal)
        self._suav_slider.setRange(0, 10)
        self._suav_slider.setValue(5)
        self._suav_slider.setToolTip(
            "Controla el suavizado de la malla 3D resultante.\n\n"
            "0 — Ninguno: malla cruda sin procesar (más vértices, más detalle).\n"
            "5 — Estándar: suavizado y decimación equilibrados.\n"
            "10 — Máximo: malla muy lisa y compacta (pierde detalle fino)."
        )
        suav_lay.addWidget(self._suav_slider, stretch=1)

        self._suav_bar_lbl = QLabel()
        self._suav_bar_lbl.setStyleSheet(
            "color:#A8B8C6; font-family:monospace; font-size:11px;"
        )
        self._suav_bar_lbl.setFixedWidth(130)
        suav_lay.addWidget(self._suav_bar_lbl)

        pf.addRow("Suavizado:", suav_row)
        self._suav_slider.valueChanged.connect(self._on_suavizado_changed)

        params_grp.setLayout(pf)
        layout.addWidget(params_grp)

        # ── 2. Collapsible "Avanzado" section ─────────────────────────── #
        adv_header = QWidget()
        adv_header_lay = QHBoxLayout(adv_header)
        adv_header_lay.setContentsMargins(0, 0, 0, 0)
        adv_header_lay.setSpacing(4)

        self._btn_adv = QToolButton()
        self._btn_adv.setText("▶ Avanzado")
        self._btn_adv.setProperty("role", "adv_toggle")
        self._btn_adv.clicked.connect(self._toggle_advanced)
        adv_header_lay.addWidget(self._btn_adv)

        self._lbl_custom = QLabel("(personalizado)")
        self._lbl_custom.setStyleSheet("color:#e3b341; font-size:10px;")
        self._lbl_custom.setVisible(False)
        adv_header_lay.addWidget(self._lbl_custom)
        adv_header_lay.addStretch()

        layout.addWidget(adv_header)

        self._adv_widget = QWidget()
        self._adv_widget.setProperty("role", "adv_section")
        adv_form = QFormLayout(self._adv_widget)
        adv_form.setLabelAlignment(Qt.AlignRight)
        adv_form.setSpacing(4)

        # ── Umbral máximo HU (band-pass) ───────────────────────────────── #
        thr_max_row = QWidget()
        thr_max_lay = QHBoxLayout(thr_max_row)
        thr_max_lay.setContentsMargins(0, 0, 0, 0)
        thr_max_lay.setSpacing(4)

        self._chk_thr_max = QCheckBox()
        self._chk_thr_max.setChecked(False)
        self._chk_thr_max.setToolTip(
            "Filtro banda-paso: segmenta solo vóxeles en [umbral mínimo, umbral máximo].\n"
            "Útil cuando hay estructuras muy brillantes sobre los vasos (ej. cráneo en XA)\n"
            "que se quieren excluir poniendo un límite superior."
        )
        thr_max_lay.addWidget(self._chk_thr_max)

        self._thr_max_spin = QSpinBox()
        self._thr_max_spin.setRange(0, 30000)
        self._thr_max_spin.setValue(600)
        self._thr_max_spin.setSuffix(" HU")
        self._thr_max_spin.setFixedWidth(90)
        self._thr_max_spin.setEnabled(False)
        thr_max_lay.addWidget(self._thr_max_spin, stretch=1)

        self._chk_thr_max.toggled.connect(self._thr_max_spin.setEnabled)
        self._chk_thr_max.toggled.connect(self._schedule_preview)
        self._thr_max_spin.valueChanged.connect(self._schedule_preview)
        adv_form.addRow("Umbral máx.:", thr_max_row)

        # XA indicator — shown when modality auto-configured
        self._lbl_xa_hint = QLabel(
            "⚠ Modalidad XA — umbral banda-paso p90–p99 auto-configurado.\n"
            "Limpieza y Suavizado aumentados para reducir ruido de fondo."
        )
        self._lbl_xa_hint.setStyleSheet(
            "color:#e3b341; font-size:10px; padding:4px 6px;"
            "background:rgba(60,45,0,60); border-radius:6px;"
        )
        self._lbl_xa_hint.setWordWrap(True)
        self._lbl_xa_hint.setVisible(False)
        adv_form.addRow("", self._lbl_xa_hint)

        # Gauss previo
        self._adv_sigma = QDoubleSpinBox()
        self._adv_sigma.setRange(0.0, 3.0)
        self._adv_sigma.setSingleStep(0.1)
        self._adv_sigma.setValue(0.3)
        self._adv_sigma.setSuffix(" σ")
        self._adv_sigma.valueChanged.connect(self._on_advanced_changed)
        adv_form.addRow("Gauss previo:", self._adv_sigma)

        # Iter. suavizado
        self._adv_smooth = QSpinBox()
        self._adv_smooth.setRange(0, 60)
        self._adv_smooth.setValue(20)
        self._adv_smooth.valueChanged.connect(self._on_advanced_changed)
        adv_form.addRow("Iter. suavizado:", self._adv_smooth)

        # Limpiar frags. >=
        self._adv_frags = QSpinBox()
        self._adv_frags.setRange(0, 50000)
        self._adv_frags.setSingleStep(50)
        self._adv_frags.setValue(0)
        self._adv_frags.valueChanged.connect(self._on_advanced_changed)
        adv_form.addRow("Limpiar frags. ≥:", self._adv_frags)

        # Cierre morfo.
        self._adv_closing = QDoubleSpinBox()
        self._adv_closing.setRange(0.0, 3.0)
        self._adv_closing.setSingleStep(0.5)
        self._adv_closing.setValue(0.5)
        self._adv_closing.setSuffix(" mm")
        self._adv_closing.valueChanged.connect(self._on_advanced_changed)
        adv_form.addRow("Cierre morfo.:", self._adv_closing)

        # Top-N comp.
        self._adv_top_n = QSpinBox()
        self._adv_top_n.setRange(0, 100)
        self._adv_top_n.setValue(20)
        self._adv_top_n.valueChanged.connect(self._on_advanced_changed)
        adv_form.addRow("Top-N comp.:", self._adv_top_n)

        # Decimación (slider + label %)
        dec_row = QWidget()
        dec_lay = QHBoxLayout(dec_row)
        dec_lay.setContentsMargins(0, 0, 0, 0)
        dec_lay.setSpacing(4)

        self._adv_dec_slider = QSlider(Qt.Horizontal)
        self._adv_dec_slider.setRange(0, 95)
        self._adv_dec_slider.setValue(70)
        self._adv_dec_slider.valueChanged.connect(
            lambda v: self._adv_dec_lbl.setText(f"{v}%")
        )
        self._adv_dec_slider.valueChanged.connect(self._on_advanced_changed)
        dec_lay.addWidget(self._adv_dec_slider, stretch=1)

        self._adv_dec_lbl = QLabel("70%")
        self._adv_dec_lbl.setFixedWidth(34)
        dec_lay.addWidget(self._adv_dec_lbl)
        adv_form.addRow("Decimación:", dec_row)

        # Umbral vasos (HU)
        self._adv_vessel_hu = QSpinBox()
        self._adv_vessel_hu.setRange(0, 2000)
        self._adv_vessel_hu.setSingleStep(10)
        self._adv_vessel_hu.setValue(0)
        self._adv_vessel_hu.setSuffix(" HU")
        self._adv_vessel_hu.valueChanged.connect(self._on_vessel_hu_changed)
        self._adv_vessel_hu.valueChanged.connect(self._on_advanced_changed)
        adv_form.addRow("Umbral vasos (HU):", self._adv_vessel_hu)

        # Radio dilatación
        self._adv_vessel_dil = QDoubleSpinBox()
        self._adv_vessel_dil.setRange(0.5, 10.0)
        self._adv_vessel_dil.setSingleStep(0.5)
        self._adv_vessel_dil.setValue(3.0)
        self._adv_vessel_dil.setSuffix(" mm")
        self._adv_vessel_dil.setEnabled(False)
        self._adv_vessel_dil.valueChanged.connect(self._on_advanced_changed)
        adv_form.addRow("Radio dilatación:", self._adv_vessel_dil)

        # Estado vasos
        self._lbl_vessel_status = QLabel("Desactivado")
        self._lbl_vessel_status.setProperty("role", "muted")
        adv_form.addRow("Estado vasos:", self._lbl_vessel_status)

        self._adv_widget.setVisible(False)
        layout.addWidget(self._adv_widget)

        # ── 3. Vista previa en tiempo real ───────────────────────────────── #
        prev_grp = QGroupBox("Vista previa en tiempo real")
        pvf = QVBoxLayout()
        pvf.setSpacing(4)

        self._chk_preview = QCheckBox(
            "Actualizar vista previa al cambiar parámetros"
        )
        self._chk_preview.setChecked(True)
        self._chk_preview.setToolTip(
            f"Cuando está activo, espera {_PREVIEW_DEBOUNCE_MS} ms tras el último\n"
            "cambio y genera una malla de previsualización rápida (resolución\n"
            "reducida 2×, sin suavizado ni decimación)."
        )
        pvf.addWidget(self._chk_preview)

        self._lbl_preview_status = QLabel("—")
        self._lbl_preview_status.setProperty("role", "muted")
        pvf.addWidget(self._lbl_preview_status)

        prev_grp.setLayout(pvf)
        layout.addWidget(prev_grp)

        # ── 4. Botón segmentación completa ────────────────────────────────── #
        self._btn_seg = QPushButton("Segmentar  (calidad completa)")
        self._btn_seg.setEnabled(False)
        self._btn_seg.setMinimumHeight(32)
        self._btn_seg.setObjectName("btn_primary")
        self._btn_seg.clicked.connect(self._run_segmentation)
        layout.addWidget(self._btn_seg)

        # ── 5. Status label ───────────────────────────────────────────────── #
        self._status_lbl = QLabel("")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setProperty("role", "muted")
        layout.addWidget(self._status_lbl)

        # ── 6. Visibilidad ───────────────────────────────────────────────── #
        vis_grp = QGroupBox("Visibilidad")
        vf2 = QHBoxLayout()
        self._chk_mesh = QCheckBox("Malla 3D")
        self._chk_mesh.setChecked(True)
        self._chk_mesh.setEnabled(False)
        self._chk_mesh.toggled.connect(self.mesh_visibility_changed)
        vf2.addWidget(self._chk_mesh)
        self._chk_vol = QCheckBox("Volumen")
        self._chk_vol.setChecked(True)
        self._chk_vol.toggled.connect(self.volume_visibility_changed)
        vf2.addWidget(self._chk_vol)
        vis_grp.setLayout(vf2)
        layout.addWidget(vis_grp)

        # ── 7. Estadísticas de malla ─────────────────────────────────────── #
        stats_grp = QGroupBox("Malla segmentada")
        sf = QFormLayout()
        self._lbl_stats = QLabel("—")
        self._lbl_stats.setWordWrap(True)
        sf.addRow("Info:", self._lbl_stats)
        stats_grp.setLayout(sf)
        layout.addWidget(stats_grp)

        # ── 8. Exportar ──────────────────────────────────────────────────── #
        export_grp = QGroupBox("Exportar malla")
        ef = QHBoxLayout()
        self._btn_stl = QPushButton("Exportar STL")
        self._btn_stl.setEnabled(False)
        self._btn_stl.clicked.connect(self.export_stl_requested)
        ef.addWidget(self._btn_stl)
        self._btn_obj = QPushButton("Exportar OBJ")
        self._btn_obj.setEnabled(False)
        self._btn_obj.clicked.connect(self.export_obj_requested)
        ef.addWidget(self._btn_obj)
        export_grp.setLayout(ef)
        layout.addWidget(export_grp)

        # ── 9. S-3 Grow-from-seeds (oculto por defecto — herramienta avanzada) ── #
        seeds_header = QWidget()
        seeds_header_lay = QHBoxLayout(seeds_header)
        seeds_header_lay.setContentsMargins(0, 4, 0, 0)
        seeds_header_lay.setSpacing(4)
        self._btn_seeds_toggle = QToolButton()
        self._btn_seeds_toggle.setText("▶ Herramientas avanzadas de segmentación")
        self._btn_seeds_toggle.clicked.connect(self._toggle_seeds_group)
        seeds_header_lay.addWidget(self._btn_seeds_toggle)
        seeds_header_lay.addStretch()
        layout.addWidget(seeds_header)

        self._seeds_container = QWidget()
        seeds_inner = QVBoxLayout(self._seeds_container)
        seeds_inner.setContentsMargins(0, 0, 0, 0)
        self._build_seeds_group(seeds_inner)
        self._seeds_container.setVisible(False)   # hidden by default
        layout.addWidget(self._seeds_container)

        layout.addStretch()

        # Initialise bar labels with default slider values
        self._on_limpieza_changed(self._lim_slider.value())
        self._on_suavizado_changed(self._suav_slider.value())

    def _build_seeds_group(self, layout: QVBoxLayout) -> None:
        """Build the S-3 grow-from-seeds group box."""
        grow_grp = QGroupBox("Crecimiento por semillas  (S-3)")
        grow_grp.setToolTip(
            "Segmentación ConnectedThreshold (SimpleITK):\n"
            "Parte de un punto semilla dentro de un vaso y crece hacia todos\n"
            "los vóxeles CONECTADOS cuyo HU esté en el rango definido.\n\n"
            "★ Método recomendado para aislar el árbol vascular:\n"
            "  Al crecer solo por conectividad, el hueso y tejido desconectado\n"
            "  quedan automáticamente excluidos sin necesidad de ajustar HU."
        )
        gf = QVBoxLayout()
        gf.setSpacing(6)

        # ── Hint label ─────────────────────────────────────────────────── #
        hint = QLabel(
            "★  Coloca la semilla en un vaso brillante en el MPR y pulsa\n"
            "    «Segmentar por crecimiento» para aislar el árbol vascular\n"
            "    sin capturar hueso ni tejido blando."
        )
        hint.setStyleSheet(
            "color:#58a6ff; font-size:10px; padding:4px 6px;"
            "background:rgba(30,60,100,60); border-radius:8px;"
        )
        hint.setWordWrap(True)
        gf.addWidget(hint)

        hu_row = QWidget()
        hu_lay = QHBoxLayout(hu_row)
        hu_lay.setContentsMargins(0, 0, 0, 0)
        hu_lay.setSpacing(4)
        hu_lay.addWidget(QLabel("HU mín:"))
        self._spin_grow_lo = QSpinBox()
        self._spin_grow_lo.setRange(-1024, 30000)
        self._spin_grow_lo.setValue(80)
        self._spin_grow_lo.setSuffix(" HU")
        self._spin_grow_lo.setFixedWidth(90)
        hu_lay.addWidget(self._spin_grow_lo)
        hu_lay.addWidget(QLabel("máx:"))
        self._spin_grow_hi = QSpinBox()
        self._spin_grow_hi.setRange(-1024, 30000)
        self._spin_grow_hi.setValue(600)
        self._spin_grow_hi.setSuffix(" HU")
        self._spin_grow_hi.setFixedWidth(90)
        hu_lay.addWidget(self._spin_grow_hi)
        gf.addWidget(hu_row)

        seed_frm = QFormLayout()
        seed_frm.setLabelAlignment(Qt.AlignRight)
        seed_xyz = QWidget()
        seed_lay = QHBoxLayout(seed_xyz)
        seed_lay.setContentsMargins(0, 0, 0, 0)
        seed_lay.setSpacing(4)
        self._spin_seed_z = QSpinBox()
        self._spin_seed_z.setRange(0, 9999)
        self._spin_seed_z.setPrefix("Z:")
        self._spin_seed_y = QSpinBox()
        self._spin_seed_y.setRange(0, 9999)
        self._spin_seed_y.setPrefix("Y:")
        self._spin_seed_x = QSpinBox()
        self._spin_seed_x.setRange(0, 9999)
        self._spin_seed_x.setPrefix("X:")
        for sp in (self._spin_seed_z, self._spin_seed_y, self._spin_seed_x):
            sp.setFixedWidth(72)
            seed_lay.addWidget(sp)
        seed_frm.addRow("Semilla (voxel):", seed_xyz)

        btn_add_seed = QPushButton("+ Añadir semilla")
        btn_add_seed.clicked.connect(self._add_manual_seed)
        seed_frm.addRow("", btn_add_seed)

        self._btn_mpr_seed = QPushButton(f"{_I.SEED} Semilla en posición MPR")
        self._btn_mpr_seed.setToolTip(
            "Coloca una semilla en la posición actual del MPR.\n\n"
            "Cómo usar:\n"
            "  1. Navega los planos Axial/Sagital/Coronal hasta que\n"
            "     el cruce de los tres planos quede DENTRO de un vaso\n"
            "     brillante (blanco brillante en el MPR).\n"
            "  2. Haz clic en este botón — la semilla se coloca en esa\n"
            "     posición (z=axial, y=coronal, x=sagital).\n"
            "  3. Repite para añadir más semillas en ramas distintas.\n\n"
            "Consejo: usa el valor HU que aparece en el pie de las vistas\n"
            "para calibrar el rango HU mín/máx antes de crecer."
        )
        if _is_dark():
            self._btn_mpr_seed.setStyleSheet(
                "QPushButton{background:#112a1a;border:1px solid #2a5a30;"
                "border-radius:8px;color:#3fb950;padding:3px 8px;}"
                "QPushButton:hover{background:#1a3f26;border-color:#3fb950;}"
            )
        else:
            self._btn_mpr_seed.setStyleSheet(
                "QPushButton{background:#e6f4ea;border:1px solid #2da44e;"
                "border-radius:8px;color:#1a7a35;padding:3px 8px;}"
                "QPushButton:hover{background:#2da44e;color:#ffffff;}"
            )
        self._btn_mpr_seed.clicked.connect(self.request_mpr_seed)
        seed_frm.addRow("", self._btn_mpr_seed)

        btn_center_seed = QPushButton("Semilla central")
        btn_center_seed.clicked.connect(self._add_center_seed)
        seed_frm.addRow("", btn_center_seed)

        gf.addLayout(seed_frm)

        self._lbl_seeds = QLabel("Sin semillas")
        self._lbl_seeds.setProperty("role", "muted")
        self._lbl_seeds.setWordWrap(True)
        gf.addWidget(self._lbl_seeds)

        btn_clear_seeds = QPushButton("Limpiar semillas")
        btn_clear_seeds.setStyleSheet("font-size:10px;")
        btn_clear_seeds.clicked.connect(self._clear_grow_seeds)
        gf.addWidget(btn_clear_seeds)

        # ── Post-grow cleanup controls ─────────────────────────────────── #
        cleanup_frm = QFormLayout()
        cleanup_frm.setLabelAlignment(Qt.AlignRight)
        cleanup_frm.setSpacing(4)

        self._spin_grow_top_n = QSpinBox()
        self._spin_grow_top_n.setRange(0, 20)
        self._spin_grow_top_n.setValue(1)
        self._spin_grow_top_n.setToolTip(
            "Tras el crecimiento, mantener solo los N componentes conexos más\n"
            "grandes del resultado. 0 = conservar todos.\n\n"
            "Valor recomendado: 1 (solo el árbol vascular principal).\n"
            "Si el árbol tiene ramas desconectadas (ej. vertebral + carótida\n"
            "sin colateral visible) usa 2–3."
        )
        cleanup_frm.addRow("Top-N componentes:", self._spin_grow_top_n)

        self._spin_grow_closing = QDoubleSpinBox()
        self._spin_grow_closing.setRange(0.0, 2.0)
        self._spin_grow_closing.setSingleStep(0.5)
        self._spin_grow_closing.setValue(0.0)
        self._spin_grow_closing.setSuffix(" mm")
        self._spin_grow_closing.setToolTip(
            "Cierre morfológico aplicado a la máscara de crecimiento antes\n"
            "de generar la malla. Rellena huecos en vasos finos.\n"
            "0 = desactivado.  0.5–1.0 mm recomendado si hay discontinuidades."
        )
        cleanup_frm.addRow("Cierre post-grow:", self._spin_grow_closing)
        gf.addLayout(cleanup_frm)

        self._btn_grow = QPushButton(f"{_I.GROWTH} Segmentar por crecimiento")
        self._btn_grow.setEnabled(False)
        self._btn_grow.setMinimumHeight(32)
        self._btn_grow.setObjectName("btn_success")
        self._btn_grow.clicked.connect(self._run_grow_segmentation)
        gf.addWidget(self._btn_grow)

        self._grow_status_lbl = QLabel("")
        self._grow_status_lbl.setAlignment(Qt.AlignCenter)
        self._grow_status_lbl.setProperty("role", "muted")
        gf.addWidget(self._grow_status_lbl)

        grow_grp.setLayout(gf)
        layout.addWidget(grow_grp)

    # ------------------------------------------------------------------ #
    # Simple-slider handlers                                               #
    # ------------------------------------------------------------------ #

    def _on_limpieza_changed(self, v: int) -> None:
        top_n, min_frags, closing = _LIMPIEZA_MAP[v]
        label = _LIMPIEZA_LABELS[v]
        bar = _make_bar(v)

        self._lim_bar_lbl.setText(f"{bar}  {label}")

        # Push to advanced spinboxes without triggering _on_advanced_changed
        self._adv_top_n.blockSignals(True)
        self._adv_frags.blockSignals(True)
        self._adv_closing.blockSignals(True)
        self._adv_top_n.setValue(top_n)
        self._adv_frags.setValue(min_frags)
        self._adv_closing.setValue(closing)
        self._adv_top_n.blockSignals(False)
        self._adv_frags.blockSignals(False)
        self._adv_closing.blockSignals(False)

        self._schedule_preview()

    def _on_suavizado_changed(self, v: int) -> None:
        sigma, smooth, reduction_frac = _SUAVIZADO_MAP[v]
        label = _SUAVIZADO_LABELS[v]
        bar = _make_bar(v)

        self._suav_bar_lbl.setText(f"{bar}  {label}")

        dec_pct = int(round(reduction_frac * 100))
        self._adv_sigma.blockSignals(True)
        self._adv_smooth.blockSignals(True)
        self._adv_dec_slider.blockSignals(True)
        self._adv_sigma.setValue(sigma)
        self._adv_smooth.setValue(smooth)
        self._adv_dec_slider.setValue(dec_pct)
        self._adv_dec_lbl.setText(f"{dec_pct}%")
        self._adv_sigma.blockSignals(False)
        self._adv_smooth.blockSignals(False)
        self._adv_dec_slider.blockSignals(False)

        self._schedule_preview()

    def _on_advanced_changed(self) -> None:
        """Called when user manually edits any advanced spinbox."""
        self._custom_mode = True
        self._lbl_custom.setVisible(True)

    # ------------------------------------------------------------------ #
    # Advanced-section toggle                                              #
    # ------------------------------------------------------------------ #

    def _toggle_advanced(self) -> None:
        visible = self._adv_widget.isVisible()
        self._adv_widget.setVisible(not visible)
        self._btn_adv.setText("▼ Avanzado" if not visible else "▶ Avanzado")

    def _toggle_seeds_group(self) -> None:
        visible = self._seeds_container.isVisible()
        self._seeds_container.setVisible(not visible)
        self._btn_seeds_toggle.setText(
            "▼ Herramientas avanzadas de segmentación"
            if not visible
            else "▶ Herramientas avanzadas de segmentación"
        )

    # ------------------------------------------------------------------ #
    # Vessel-enhancement helper                                            #
    # ------------------------------------------------------------------ #

    def _on_vessel_hu_changed(self, value: int) -> None:
        active = value > 0
        self._adv_vessel_dil.setEnabled(active)
        if active:
            main_hu = self._thr_spin.value()
            if value >= main_hu:
                self._lbl_vessel_status.setText(
                    "⚠ Umbral secundario debe ser menor que el principal."
                )
                self._lbl_vessel_status.setStyleSheet(
                    "color:#f85149; font-size:10px;"
                )
            else:
                diff = main_hu - value
                self._lbl_vessel_status.setText(
                    f"Activo — recupera vasos {diff} HU por debajo del umbral principal."
                )
                self._lbl_vessel_status.setStyleSheet(
                    "color:#3fb950; font-size:10px;"
                )
        else:
            self._lbl_vessel_status.setText("Desactivado")
            self._lbl_vessel_status.setStyleSheet("")

    # ------------------------------------------------------------------ #
    # Pipeline parameter helpers                                           #
    # ------------------------------------------------------------------ #

    def _get_pipeline_params(self) -> dict:
        vessel_hu = self._adv_vessel_hu.value()
        max_hu = (
            float(self._thr_max_spin.value())
            if self._chk_thr_max.isChecked()
            else 0.0
        )
        return dict(
            threshold_hu        = float(self._thr_spin.value()),
            threshold_max_hu    = max_hu,
            gaussian_sigma      = self._adv_sigma.value(),
            smooth_iterations   = self._adv_smooth.value(),
            target_reduction    = self._adv_dec_slider.value() / 100.0,
            min_component_verts = self._adv_frags.value(),
            morpho_closing_mm   = self._adv_closing.value(),
            keep_top_n          = self._adv_top_n.value(),
            vessel_lower_hu     = float(vessel_hu) if vessel_hu > 0 else 0.0,
            vessel_dilation_mm  = self._adv_vessel_dil.value(),
        )

    # ------------------------------------------------------------------ #
    # Preview logic                                                        #
    # ------------------------------------------------------------------ #

    def _schedule_preview(self) -> None:
        """Reset the debounce timer; preview fires _PREVIEW_DEBOUNCE_MS later."""
        if (
            self._chk_preview.isChecked()
            and self._volume is not None
            and (self._preview_worker is None or not self._preview_worker.isRunning())
        ):
            self._preview_timer.start()   # restart (debounce)

    def _run_preview(self) -> None:
        """Start fast preview worker if conditions are met."""
        if self._volume is None:
            return
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return   # previous preview still running — skip

        params = self._get_pipeline_params()
        # Override for fast preview
        params["gaussian_sigma"]      = 0.0
        params["smooth_iterations"]   = 0
        params["target_reduction"]    = 0.0
        params["min_component_verts"] = 0
        if params["keep_top_n"] <= 0:
            params["keep_top_n"] = 20

        pipeline = SegmentationPipeline(**params)

        self._lbl_preview_status.setText(f"{_I.WAIT} Generando previsualización…")
        self._lbl_preview_status.setStyleSheet("color:#e3b341; font-size:10px;")

        self._preview_worker = _PreviewWorker(
            pipeline, self._volume, self._spacing, self
        )
        self._preview_worker.finished.connect(self._on_preview_done)
        self._preview_worker.error.connect(self._on_preview_error)
        self._preview_worker.start()

    def _on_preview_done(self, result: SegmentationResult) -> None:
        self._lbl_preview_status.setText(
            f"✔ Vista previa — {result.n_vertices:,} verts "
            f"(umbral {result.threshold_hu:.0f} HU)"
        )
        self._lbl_preview_status.setStyleSheet("color:#3fb950; font-size:10px;")
        self._chk_mesh.setEnabled(True)
        self.preview_ready.emit(result)

    def _on_preview_error(self, msg: str) -> None:
        self._lbl_preview_status.setText("Sin superficie a este umbral.")
        self._lbl_preview_status.setStyleSheet("color:#f85149; font-size:10px;")

    # ------------------------------------------------------------------ #
    # Full segmentation                                                    #
    # ------------------------------------------------------------------ #

    def _run_segmentation(self) -> None:
        if self._volume is None:
            return

        # Cancel any pending preview
        self._preview_timer.stop()

        pipeline = SegmentationPipeline(**self._get_pipeline_params())

        self._btn_seg.setEnabled(False)
        self._btn_seg.setText("Calculando…")
        self._status_lbl.setText("Ejecutando Marching Cubes…")
        self._lbl_preview_status.setText("—")
        self._lbl_preview_status.setStyleSheet("")

        self._worker = _SegWorker(pipeline, self._volume, self._spacing, self)
        self._worker.finished.connect(self._on_seg_done)
        self._worker.error.connect(self._on_seg_error)
        self._worker.start()

    def _on_seg_done(self, result: SegmentationResult) -> None:
        self._last_result = result
        self._btn_seg.setEnabled(True)
        self._btn_seg.setText("Segmentar  (calidad completa)")

        dual_note = ""
        if self._adv_vessel_hu.value() > 0:
            dual_note = "\n⊕ Modo vasos finos activo"

        n_removed = getattr(result, "n_fragments_removed", 0)
        frag_line = f"\nFragmentos eliminados: {n_removed}" if n_removed > 0 else ""
        self._lbl_stats.setText(
            f"{result.n_vertices:,} vértices\n"
            f"{result.n_triangles:,} triángulos\n"
            f"Umbral: {result.threshold_hu:.0f} HU\n"
            f"Decimación: {result.reduction_pct:.1f}%"
            f"{frag_line}{dual_note}"
        )

        self._status_lbl.setText("✔ Segmentación completa.")
        self._status_lbl.setStyleSheet("color:#3fb950; font-size:10px;")
        self._chk_mesh.setEnabled(True)
        self._set_export_enabled(True)
        self.seg_ready.emit(result)

    def _on_seg_error(self, msg: str) -> None:
        self._btn_seg.setEnabled(True)
        self._btn_seg.setText("Segmentar  (calidad completa)")
        self._status_lbl.setText("")
        self._status_lbl.setStyleSheet("")
        QMessageBox.critical(self, "Error de segmentación", msg)

    def _set_export_enabled(self, enabled: bool) -> None:
        self._btn_stl.setEnabled(enabled)
        self._btn_obj.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    # S-3 Grow-from-seeds                                                  #
    # ------------------------------------------------------------------ #

    def _add_manual_seed(self) -> None:
        z, y, x = (self._spin_seed_z.value(),
                   self._spin_seed_y.value(),
                   self._spin_seed_x.value())
        self._grow_seeds.append((z, y, x))
        self._update_seeds_label()
        if self._volume is not None:
            self._btn_grow.setEnabled(True)

    def _add_center_seed(self) -> None:
        if self._volume is None:
            return
        nz, ny, nx = self._volume.shape
        self._grow_seeds.append((nz // 2, ny // 2, nx // 2))
        self._update_seeds_label()
        self._btn_grow.setEnabled(True)

    def _clear_grow_seeds(self) -> None:
        self._grow_seeds = []
        self._update_seeds_label()
        self._btn_grow.setEnabled(False)

    def _update_seeds_label(self) -> None:
        if not self._grow_seeds:
            self._lbl_seeds.setText("Sin semillas")
        else:
            lines = [f"  (Z={z}, Y={y}, X={x})" for z, y, x in self._grow_seeds]
            self._lbl_seeds.setText(
                f"{len(self._grow_seeds)} semilla(s):\n" + "\n".join(lines)
            )

    def _run_grow_segmentation(self) -> None:
        if self._volume is None or not self._grow_seeds:
            return
        lo, hi = float(self._spin_grow_lo.value()), float(self._spin_grow_hi.value())
        if lo >= hi:
            QMessageBox.warning(self, "Rango HU inválido",
                                "HU mínimo debe ser menor que HU máximo.")
            return

        pipeline = GrowSegmentationPipeline(
            lower_hu          = lo,
            upper_hu          = hi,
            smooth_iterations = self._adv_smooth.value(),
            target_reduction  = self._adv_dec_slider.value() / 100.0,
            keep_top_n        = self._spin_grow_top_n.value(),
            morpho_closing_mm = self._spin_grow_closing.value(),
        )
        self._btn_grow.setEnabled(False)
        self._btn_grow.setText("Calculando…")
        self._grow_status_lbl.setText("Ejecutando ConnectedThreshold…")

        self._grow_worker = _GrowWorker(
            pipeline, self._volume, self._spacing, list(self._grow_seeds), self
        )
        self._grow_worker.finished.connect(self._on_grow_done)
        self._grow_worker.error.connect(self._on_grow_error)
        self._grow_worker.start()

    def _on_grow_done(self, result: GrowResult) -> None:
        self._last_result = result
        self._btn_grow.setEnabled(True)
        self._btn_grow.setText(f"{_I.GROWTH} Segmentar por crecimiento")

        n_removed = getattr(result, "n_fragments_removed", 0)
        frag_line = (
            f"\nFragmentos eliminados: {n_removed}"
            if n_removed > 0
            else ""
        )
        self._grow_status_lbl.setText(
            f"✔ Crecimiento completado.{frag_line}"
        )
        self._grow_status_lbl.setStyleSheet(
            "color:#3fb950; font-size:10px;"
        )

        self._lbl_stats.setText(
            f"{result.n_vertices:,} vértices\n"
            f"{result.n_triangles:,} triángulos\n"
            f"Rango HU: [{result.lower_hu:.0f}, {result.upper_hu:.0f}]\n"
            f"Vóxeles segmentados: {result.n_voxels:,}\n"
            f"Semillas: {len(result.seeds)}"
            + (f"\nFragmentos eliminados: {n_removed}" if n_removed > 0 else "")
        )
        self._chk_mesh.setEnabled(True)
        self._set_export_enabled(True)
        self.grow_ready.emit(result)
        self.seg_ready.emit(result)

    def _on_grow_error(self, msg: str) -> None:
        self._btn_grow.setEnabled(True)
        self._btn_grow.setText(f"{_I.GROWTH} Segmentar por crecimiento")
        self._grow_status_lbl.setText("")
        self._grow_status_lbl.setStyleSheet("")
        QMessageBox.critical(self, "Error — crecimiento por semillas", msg)
