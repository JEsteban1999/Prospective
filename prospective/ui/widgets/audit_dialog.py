"""AuditDialog — SkullChain cryptographic audit trail viewer.

Shows all chain blocks in a read-only table with integrity verification
and TXT export.  Uses the same dark theme as the rest of the application.
"""
from __future__ import annotations

import logging

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from prospective.audit.skull_chain import (
    ACT_INTEGRITY_CHECK,
    SkullChain,
)

logger = logging.getLogger(__name__)

# ── Colour mappings ────────────────────────────────────────────────────────── #

_ACTION_LABELS: dict[str, str] = {
    "LOGIN":                   "Inicio de sesión",
    "PATIENT_LOADED":          "Paciente cargado",
    "SERIES_LOADED":           "Serie DICOM cargada",
    "SEGMENTATION_COMPLETE":   "Segmentación completada",
    "MESH_EXPORTED":           "Malla exportada",
    "DEVICE_PLACED":           "Dispositivo colocado",
    "DEVICE_REMOVED":          "Dispositivo retirado",
    "TREATMENT_DECISION":      "Decisión terapéutica",
    "REPORT_GENERATED":        "Informe PDF generado",
    "INTEGRITY_CHECK":         "Verificación de integridad",
    "GENESIS":                 "Bloque génesis",
}

# QColor values for action column foreground
_ACTION_COLORS: dict[str, str] = {
    "LOGIN":                   "#9B9B9B",   # grey
    "PATIENT_LOADED":          "#8B9BAA",   # violet
    "SERIES_LOADED":           "#A8B8C6",   # violet-light
    "SEGMENTATION_COMPLETE":   "#8B9BAA",   # violet
    "MESH_EXPORTED":           "#e3b341",   # yellow
    "DEVICE_PLACED":           "#39d353",   # cyan-green
    "DEVICE_REMOVED":          "#f85149",   # red
    "TREATMENT_DECISION":      "#A8B8C6",   # smoke grey
    "REPORT_GENERATED":        "#56d364",   # green
    "INTEGRITY_CHECK":         "#ffa657",   # orange
    "GENESIS":                 "#6e7681",   # dim grey
}

_COLUMNS = ["#", "Fecha/Hora", "Usuario", "Acción", "Paciente (hash)", "Hash bloque"]
_COL_WIDTHS = [40, 160, 110, 200, 140, 140]

def _dialog_style() -> str:
    """Return the audit dialog stylesheet for the current theme."""
    try:
        from prospective.ui.themes import current_theme
        t = current_theme()
    except Exception:
        t = "dark"
    if t == "light":
        bg, panel, border = "#FFFFFF", "#F7F7F7", "#E5E5E5"
        txt, muted, sel = "#0D0D0D", "#6B6B6B", "#8B9BAA"
    else:  # dark
        bg, panel, border = "#1F1F1F", "#2A2A2A", "#363636"
        txt, muted, sel = "#EBEBEB", "#9B9B9B", "#4E6678"
    return f"""
QDialog {{ background-color: {bg}; color: {txt}; }}
QLabel {{ color: {txt}; border: none; background: transparent; }}
QLabel#subtitle {{ color: {muted}; font-size: 11px; }}
QLabel#status   {{ color: {muted}; font-size: 11px; }}
QTableWidget {{
    background-color: {panel};
    color: {txt};
    gridline-color: {border};
    border: 1px solid {border};
    selection-background-color: {sel};
    selection-color: #ffffff;
    alternate-background-color: {bg};
}}
QTableWidget::item {{ padding: 2px 6px; }}
QHeaderView::section {{
    background-color: {panel};
    color: {muted};
    border: none;
    border-bottom: 1px solid {border};
    padding: 4px 6px;
    font-weight: bold;
}}
QPushButton {{
    background-color: {panel};
    color: {txt};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 5px 12px;
    min-width: 100px;
}}
QPushButton:hover {{ background-color: {sel}; color: #ffffff; border-color: {sel}; }}
QPushButton:pressed {{ background-color: {bg}; }}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: {bg}; width: 10px; height: 10px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {border}; border-radius: 9px; min-height: 20px;
}}
"""


class AuditDialog(QDialog):
    """Modal dialog showing the full SkullChain audit trail."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SkullChain™ — Auditoría clínica")
        self.setMinimumSize(920, 560)
        self.resize(1000, 640)
        self.setStyleSheet(_dialog_style())
        self._last_verify_ok: bool | None = None
        self._build_ui()
        self._load_table()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        try:
            from prospective.ui.glass_utils import enable_acrylic
            enable_acrylic(self)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 10)
        root.setSpacing(8)

        # Title
        title = QLabel("SkullChain™ — Cadena de auditoría clínica")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        # Subtitle
        subtitle = QLabel(
            "Registro criptográfico de todas las acciones clínicas. "
            "Cualquier modificación retroactiva invalida la cadena."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # Table
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        for col, width in enumerate(_COL_WIDTHS):
            self._table.setColumnWidth(col, width)
        self._table.setSortingEnabled(False)
        root.addWidget(self._table)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        from prospective.ui.icons import I as _I
        self._btn_refresh = QPushButton(f"{_I.REFRESH} Actualizar")
        self._btn_refresh.setToolTip("Recargar la tabla desde la base de datos")
        self._btn_refresh.clicked.connect(self._load_table)

        self._btn_verify = QPushButton(f"{_I.SEARCH} Verificar integridad")
        self._btn_verify.setToolTip("Recalcular todos los hashes y detectar manipulaciones")
        self._btn_verify.clicked.connect(self._verify)

        self._btn_export = QPushButton(f"{_I.DOC} Exportar TXT")
        self._btn_export.setToolTip("Exportar registro completo a fichero de texto")
        self._btn_export.clicked.connect(self._export_txt)

        btn_close = QPushButton("Cerrar")
        btn_close.setToolTip("Cerrar este diálogo")
        btn_close.clicked.connect(self.close)

        btn_row.addWidget(self._btn_refresh)
        btn_row.addWidget(self._btn_verify)
        btn_row.addWidget(self._btn_export)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

        # Status bar
        self._lbl_status = QLabel("")
        self._lbl_status.setObjectName("status")
        root.addWidget(self._lbl_status)

    # ------------------------------------------------------------------ #
    # Table population                                                     #
    # ------------------------------------------------------------------ #

    def _load_table(self) -> None:
        """(Re)populate the table from SkullChain."""
        blocks = SkullChain.instance().get_all_blocks()
        self._table.setRowCount(0)
        self._table.setRowCount(len(blocks))

        for row, blk in enumerate(blocks):
            action_raw = blk.get("action", "")
            action_label = _ACTION_LABELS.get(action_raw, action_raw)
            pat_hash = blk.get("patient_hash", "")
            bh = blk.get("block_hash", "")

            pat_disp = (pat_hash[:8] + "…") if pat_hash else "—"
            bh_disp  = (bh[:12] + "…") if bh else "—"

            values = [
                str(blk.get("id", "")),
                blk.get("iso_ts", ""),
                blk.get("username", "") or "—",
                action_label,
                pat_disp,
                bh_disp,
            ]

            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                if col == 3:  # Action column: colour-code
                    color_hex = _ACTION_COLORS.get(action_raw, "#9B9B9B")
                    item.setForeground(QColor(color_hex))
                self._table.setItem(row, col, item)

        from prospective.ui.icons import I as _I
        count = len(blocks)
        verify_str = ""
        if self._last_verify_ok is True:
            verify_str = f"  |  {_I.STATUS_OK} Integridad verificada"
        elif self._last_verify_ok is False:
            verify_str = f"  |  {_I.STATUS_WARN} Integridad comprometida"
        self._lbl_status.setText(f"Total: {count} bloques{verify_str}")

    # ------------------------------------------------------------------ #
    # Button handlers                                                      #
    # ------------------------------------------------------------------ #

    def _verify(self) -> None:
        """Run integrity check, show result, and append an INTEGRITY_CHECK block."""
        chain = SkullChain.instance()
        all_ok, broken = chain.verify_integrity()
        block_count = len(chain.get_all_blocks())

        # Append audit event for this verification
        try:
            chain.append(
                ACT_INTEGRITY_CHECK,
                {"blocks_checked": block_count, "ok": all_ok, "broken_count": len(broken)},
            )
        except Exception:
            pass

        self._last_verify_ok = all_ok

        if all_ok:
            QMessageBox.information(
                self,
                "Verificación de integridad",
                f"{_I.STATUS_OK} Cadena íntegra — {block_count} bloques verificados.",
            )
        else:
            detail_lines = "\n".join(
                f"  Bloque #{b['id']} ({b['action']}): {b['reason']}"
                for b in broken
            )
            QMessageBox.warning(
                self,
                f"{_I.STATUS_WARN} Integridad comprometida",
                f"Se detectaron {len(broken)} bloque(s) alterado(s):\n\n{detail_lines}",
            )

        self._load_table()

    def _export_txt(self) -> None:
        """Export the full chain to a plain-text file chosen by the user."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar auditoría SkullChain",
            "skullchain_audit.txt",
            "Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return
        try:
            SkullChain.instance().export_audit_txt(path)
            QMessageBox.information(
                self, "Exportación completada", f"Archivo guardado en:\n{path}"
            )
        except Exception as exc:
            logger.exception("SkullChain export failed")
            QMessageBox.critical(self, "Error al exportar", str(exc))
