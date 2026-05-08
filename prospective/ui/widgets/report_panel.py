"""PDF report panel — A-04-05.

Collects patient data, triggers a VTK screenshot, and calls ReportGenerator.
Wired to MainWindow via signals.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from prospective.io.dicom_sr import DicomSRGenerator
from prospective.io.report_generator import ClipEntry, CoilEntry, PatientInfo, ReportData, ReportGenerator

logger = logging.getLogger(__name__)


class ReportPanel(QWidget):
    """
    Self-contained panel for building and exporting the surgical PDF report.

    Signals
    -------
    screenshot_requested()   — ask MainWindow to capture the current 3D view
                               and call deliver_screenshot(bytes) afterwards.
    """

    screenshot_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._morphometrics: dict[str, Any] = {}
        self._clips: list[ClipEntry] = []
        self._coils: list[CoilEntry] = []
        self._stents: list[dict[str, Any]] = []
        self._trajectory: dict[str, Any] = {}
        self._risk_label: str = ""
        self._treatment: dict[str, Any] = {}
        self._screenshot_png: bytes | None = None
        self._pending_generate: bool = False
        self._series_meta: dict[str, Any] = {}   # set by MainWindow after DICOM load

        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API — called by MainWindow                                   #
    # ------------------------------------------------------------------ #

    def set_morphometrics(self, result) -> None:
        """Accept a MorphometricResult and cache it for the report."""
        if result is None:
            self._morphometrics = {}
            self._risk_label = ""
            return
        self._morphometrics = {
            "volume_mm3":         result.volume_mm3,
            "surface_area_mm2":   result.surface_area_mm2,
            "eq_sphere_diam_mm":  result.eq_sphere_diam_mm,
            "bbox_l_mm":          result.bbox_l_mm,
            "bbox_w_mm":          result.bbox_w_mm,
            "bbox_h_mm":          result.bbox_h_mm,
            "max_diameter_mm":    result.max_diameter_mm,
            "neck_diameter_mm":   result.neck_diameter_mm,
            "dome_height_mm":     result.dome_height_mm,
            "dome_to_neck_ratio": result.dome_to_neck_ratio,
            "aspect_ratio":       result.aspect_ratio,
            "compactness":        result.compactness,
        }
        self._risk_label = result.rupture_risk_label
        self._lbl_morpho_status.setText(
            f"Morfometría: Ø {result.max_diameter_mm:.1f} mm, "
            f"DNR {result.dome_to_neck_ratio:.2f}, AR {result.aspect_ratio:.2f}"
        )
        self._update_generate_button()

    def set_treatment_decision(self, decision) -> None:
        """Accept a TreatmentDecision and cache it for the report."""
        if decision is None:
            self._treatment = {}
            self._lbl_decision_status.setText("Decisión terapéutica: no disponible.")
            self._lbl_decision_status.setStyleSheet("")
            return
        self._treatment = {
            "recommendation":     decision.recommendation,
            "recommendation_key": decision.recommendation_key,
            "confidence":         decision.confidence,
            "clip_pct":           decision.clip_pct,
            "endo_pct":           decision.endo_pct,
            "factors": [
                {
                    "name":      f.name,
                    "direction": f.direction,
                    "points":    f.points,
                    "detail":    f.detail,
                }
                for f in decision.factors
            ],
            "notes": list(decision.notes),
        }
        color = {"clip": "#8B9BAA", "endo": "#3fb950",
                 "mdt": "#d29922", "surveillance": "#9B9B9B"}.get(
            decision.recommendation_key, "#9B9B9B"
        )
        self._lbl_decision_status.setText(
            f"Decisión: {decision.recommendation}  —  Confianza: {decision.confidence}"
        )
        self._lbl_decision_status.setStyleSheet(
            f"color:{color}; font-size:10px; font-weight:bold;"
        )

    def set_clips(self, clips: list[ClipEntry]) -> None:
        self._clips = clips
        n = len(clips)
        self._lbl_clips_status.setText(f"{n} clip(s) planificado(s).")
        self._update_generate_button()

    def set_coils(self, coils: list[CoilEntry]) -> None:
        self._coils = coils

    def set_stents(self, stents: list[dict]) -> None:
        """Receive placed stent/flow-diverter data (plain dicts) from MainWindow."""
        self._stents = stents

    def set_series_meta(self, meta: dict[str, Any]) -> None:
        """Receive DICOM series metadata for the SR header."""
        self._series_meta = meta

    def set_trajectory(self, entry: list[float], target: list[float]) -> None:
        if entry and target:
            dx = target[0] - entry[0]
            dy = target[1] - entry[1]
            dz = target[2] - entry[2]
            depth = math.sqrt(dx*dx + dy*dy + dz*dz)
            # Angle from vertical (z-axis)
            xy = math.sqrt(dx*dx + dy*dy)
            angle = math.degrees(math.atan2(xy, abs(dz))) if dz != 0 else 90.0
            self._trajectory = {
                "entry":     entry,
                "target":    target,
                "depth_mm":  depth,
                "angle_deg": angle,
            }
        else:
            self._trajectory = {}

    def deliver_screenshot(self, png_bytes: bytes) -> None:
        """Called by MainWindow with the captured PNG bytes."""
        self._screenshot_png = png_bytes
        self._lbl_screenshot.setText("Captura: lista.")
        if self._pending_generate:
            self._pending_generate = False
            self._do_generate()

    # ------------------------------------------------------------------ #
    # UI                                                                   #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Datos del paciente ────────────────────────────────────────── #
        pat_grp = QGroupBox("Datos del paciente")
        pf = QFormLayout()
        pf.setLabelAlignment(Qt.AlignRight)

        self._edit_name      = QLineEdit()
        self._edit_name.setPlaceholderText("Apellidos, Nombre")
        self._edit_id        = QLineEdit()
        self._edit_id.setPlaceholderText("NHC / ID")
        self._edit_dob       = QLineEdit()
        self._edit_dob.setPlaceholderText("DD/MM/AAAA")
        self._edit_surgeon   = QLineEdit()
        self._edit_surgeon   .setPlaceholderText("Dr. / Dra. …")
        self._edit_institution = QLineEdit("Fundación Universitaria Navarra UNINAVARRA")

        pf.addRow("Paciente:",     self._edit_name)
        pf.addRow("N.º historia:", self._edit_id)
        pf.addRow("F. nacimiento:", self._edit_dob)
        pf.addRow("Cirujano:",     self._edit_surgeon)
        pf.addRow("Institución:",  self._edit_institution)
        pat_grp.setLayout(pf)
        layout.addWidget(pat_grp)

        # ── Recomendación terapéutica ─────────────────────────────────── #
        treat_grp = QGroupBox("Recomendación terapéutica")
        tf = QVBoxLayout()
        self._edit_treatment = QLineEdit()
        self._edit_treatment.setPlaceholderText(
            "p. ej. Clipado quirúrgico, embolización con coils, vigilancia…"
        )
        tf.addWidget(self._edit_treatment)
        treat_grp.setLayout(tf)
        layout.addWidget(treat_grp)

        # ── Notas ─────────────────────────────────────────────────────── #
        notes_grp = QGroupBox("Notas / observaciones")
        nf = QVBoxLayout()
        self._edit_notes = QPlainTextEdit()
        self._edit_notes.setPlaceholderText(
            "Hallazgos relevantes, comorbilidades, indicaciones clínicas…"
        )
        self._edit_notes.setMaximumHeight(80)
        nf.addWidget(self._edit_notes)
        notes_grp.setLayout(nf)
        layout.addWidget(notes_grp)

        # ── Estado de los datos ───────────────────────────────────────── #
        status_grp = QGroupBox("Datos disponibles")
        sf = QVBoxLayout()

        self._lbl_morpho_status = QLabel("Morfometría: no disponible.")
        self._lbl_morpho_status.setProperty("role", "muted")
        self._lbl_morpho_status.setWordWrap(True)
        sf.addWidget(self._lbl_morpho_status)

        self._lbl_decision_status = QLabel("Decisión terapéutica: no disponible.")
        self._lbl_decision_status.setProperty("role", "muted")
        self._lbl_decision_status.setWordWrap(True)
        sf.addWidget(self._lbl_decision_status)

        self._lbl_clips_status = QLabel("Clips: ninguno.")
        self._lbl_clips_status.setProperty("role", "muted")
        sf.addWidget(self._lbl_clips_status)

        self._lbl_screenshot = QLabel("Captura 3D: pendiente.")
        self._lbl_screenshot.setProperty("role", "muted")
        sf.addWidget(self._lbl_screenshot)

        status_grp.setLayout(sf)
        layout.addWidget(status_grp)

        # ── Botones ───────────────────────────────────────────────────── #
        btn_row = QWidget()
        br_lay = QHBoxLayout(btn_row)
        br_lay.setContentsMargins(0, 0, 0, 0)
        br_lay.setSpacing(6)

        self._btn_screenshot = QPushButton("Capturar 3D")
        self._btn_screenshot.setToolTip(
            "Toma una captura del visor 3D actual y la incluye en el informe."
        )
        self._btn_screenshot.clicked.connect(self._request_screenshot)
        br_lay.addWidget(self._btn_screenshot)

        try:
            from prospective.ui.themes import is_dark as _rp_is_dark
            _rp_dark = _rp_is_dark()
        except Exception:
            _rp_dark = True

        self._btn_generate = QPushButton("Generar PDF")
        self._btn_generate.setMinimumHeight(32)
        if _rp_dark:
            self._btn_generate.setStyleSheet(
                "QPushButton{background:#102050;border:1px solid rgba(80,140,255,100);"
                "border-radius:9px;color:#c8d6e8;font-weight:bold;}"
                "QPushButton:hover{background:#2979ff;border-color:#5aa0ff;}"
                "QPushButton:disabled{background:#0c1828;color:#3a5070;"
                "border-color:rgba(80,140,255,25);}"
            )
        else:
            self._btn_generate.setStyleSheet(
                "QPushButton{background:#EBF5FF;border:1px solid rgba(80,140,255,120);"
                "border-radius:9px;color:#1a4080;font-weight:bold;}"
                "QPushButton:hover{background:#2979ff;border-color:#5aa0ff;color:#ffffff;}"
                "QPushButton:disabled{background:#F0F4F8;color:#AAAAAA;"
                "border-color:rgba(80,140,255,40);}"
            )
        self._btn_generate.clicked.connect(self._generate)
        br_lay.addWidget(self._btn_generate)

        layout.addWidget(btn_row)

        # ── DICOM SR ──────────────────────────────────────────────────── #
        self._btn_sr = QPushButton("Exportar DICOM SR")
        self._btn_sr.setToolTip(
            "Genera un DICOM Structured Report (.dcm) con las medidas y el plan\n"
            "quirúrgico, para archivar en el PACS junto a las imágenes originales."
        )
        if _rp_dark:
            self._btn_sr.setStyleSheet(
                "QPushButton{background:#0a1428;border:1px solid rgba(80,140,255,40);"
                "border-radius:9px;color:#7a98c0;}"
                "QPushButton:hover{background:#0f1e3a;color:#d0d9ea;}"
            )
        else:
            self._btn_sr.setStyleSheet(
                "QPushButton{background:#EBF0F5;border:1px solid rgba(80,140,255,80);"
                "border-radius:9px;color:#4E6678;}"
                "QPushButton:hover{background:#DDE5EC;color:#2E4A5F;}"
            )
        self._btn_sr.clicked.connect(self._export_sr)
        layout.addWidget(self._btn_sr)

        layout.addStretch()

        self._update_generate_button()

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    def _request_screenshot(self) -> None:
        self._lbl_screenshot.setText("Captura 3D: capturando…")
        self._screenshot_png = None
        self.screenshot_requested.emit()

    def _generate(self) -> None:
        if self._screenshot_png is None:
            # Capture first, then generate when delivered
            self._pending_generate = True
            self._lbl_screenshot.setText("Captura 3D: capturando…")
            self.screenshot_requested.emit()
        else:
            self._do_generate()

    def _do_generate(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar informe PDF",
            "",
            "PDF (*.pdf)",
        )
        if not path:
            self._pending_generate = False
            return

        patient = PatientInfo(
            name        = self._edit_name.text().strip() or "Anónimo",
            id          = self._edit_id.text().strip(),
            dob         = self._edit_dob.text().strip(),
            surgeon     = self._edit_surgeon.text().strip(),
            institution = self._edit_institution.text().strip()
                          or "Fundación Universitaria Navarra UNINAVARRA",
            notes       = self._edit_notes.toPlainText().strip(),
        )

        data = ReportData(
            patient        = patient,
            morphometrics  = self._morphometrics,
            clips          = self._clips,
            coils          = self._coils,
            trajectory     = self._trajectory,
            screenshot_png = self._screenshot_png,
            risk_label     = self._risk_label,
            treatment      = self._treatment,
        )

        try:
            gen = ReportGenerator(data)
            out = gen.generate(path)
            QMessageBox.information(
                self,
                "Informe generado",
                f"PDF guardado en:\n{out}",
            )
            try:
                import hashlib
                import pathlib
                from prospective.audit.skull_chain import SkullChain, ACT_REPORT_GENERATED
                from prospective.auth.auth_manager import AuthManager
                file_hash = hashlib.sha256(pathlib.Path(out).read_bytes()).hexdigest()
                user = AuthManager.instance().current_user
                SkullChain.instance().append(
                    ACT_REPORT_GENERATED,
                    {
                        "filename": pathlib.Path(out).name,
                        "file_hash": file_hash,
                        "patient": patient.name,
                        "surgeon": patient.surgeon,
                    },
                    username=user.username if user else "",
                )
            except Exception:
                pass  # audit must never block clinical workflow
        except Exception as exc:
            logger.exception("PDF generation failed")
            QMessageBox.critical(self, "Error al generar PDF", str(exc))

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _update_generate_button(self) -> None:
        # Always allow generation — data availability shown as status labels
        self._btn_generate.setEnabled(True)

    def _export_sr(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar DICOM SR",
            "plan_quirurgico_sr.dcm",
            "DICOM (*.dcm);;Todos los archivos (*)",
        )
        if not path:
            return

        # Merge risk_label into morphometrics dict for the generator
        morpho = dict(self._morphometrics)
        if self._risk_label:
            morpho["risk_label"] = self._risk_label

        # Convert ClipEntry list to plain dicts
        clips_dicts = [
            {
                "index":           c.index,
                "name":            c.name,
                "position_mm":     c.position_mm,
                "orientation_deg": c.orientation_deg,
                "is_custom":       c.is_custom,
            }
            for c in self._clips
        ]

        # Convert CoilEntry list to plain dicts
        coils_dicts = [
            {
                "index":        c.index,
                "name":         c.name,
                "coil_type":    c.coil_type,
                "diameter_mm":  c.diameter_mm,
                "length_cm":    c.length_cm,
                "manufacturer": c.manufacturer,
                "position_mm":  c.position_mm,
                "is_custom":    c.is_custom,
            }
            for c in self._coils
        ]

        try:
            gen = DicomSRGenerator(
                series_meta  = self._series_meta,
                morphometrics= morpho,
                clips        = clips_dicts,
                trajectory   = self._trajectory,
                stents       = self._stents,
                coils        = coils_dicts,
            )
            out = gen.generate(path)
            QMessageBox.information(
                self, "DICOM SR exportado",
                f"Archivo guardado en:\n{out}"
            )
        except Exception as exc:
            logger.exception("DICOM SR export failed")
            QMessageBox.critical(self, "Error al exportar DICOM SR", str(exc))

    def get_session_state(self) -> dict:
        return {
            "patient_name":    self._edit_name.text(),
            "patient_id":      self._edit_id.text(),
            "patient_dob":     self._edit_dob.text(),
            "surgeon":         self._edit_surgeon.text(),
            "institution":     self._edit_institution.text(),
            "notes":           self._edit_notes.toPlainText(),
            "treatment":       self._edit_treatment.text(),
        }

    def restore_session_state(self, d: dict) -> None:
        self._edit_name.setText(d.get("patient_name", ""))
        self._edit_id.setText(d.get("patient_id", ""))
        self._edit_dob.setText(d.get("patient_dob", ""))
        self._edit_surgeon.setText(d.get("surgeon", ""))
        self._edit_institution.setText(d.get("institution", "Fundación Universitaria Navarra UNINAVARRA"))
        self._edit_notes.setPlainText(d.get("notes", ""))
        self._edit_treatment.setText(d.get("treatment", ""))
