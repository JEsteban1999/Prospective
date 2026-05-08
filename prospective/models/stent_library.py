"""Intracranial stent / flow-diverter library.

Covers the devices most commonly used for cerebral aneurysm treatment:
  - Flow diverters  (Pipeline PED, Surpass Streamline, FRED)
  - Stent-assisted coiling stents (Neuroform Atlas, Enterprise 2, Leo+)

All lengths in mm.  Porosity is open-area percentage (%).
Wire diameter in micrometres (µm).

This module is pure data — no VTK or Qt dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StentType(Enum):
    FLOW_DIVERTER = "Desviador de flujo"
    INTRACRANIAL  = "Stent intracraneal"
    BRAIDED       = "Stent trenzado"     # low-porosity braided (LVIS, Leo+)


@dataclass(frozen=True)
class StentSpec:
    """Geometric / clinical specification of one stent model."""

    name: str
    stent_type: StentType
    diameter_mm: float        # nominal deployed diameter
    length_mm: float          # deployed length
    porosity_pct: float       # open-area percentage (flow diverter ≈ 30 %, stent ≈ 75 %)
    wire_diameter_um: float   # strut/wire thickness in micrometres
    n_wires: int              # total number of braided wires (for visualisation)
    manufacturer: str
    compatible_wire: str      # compatible guidewire size (e.g. '0.014"')
    catheter_id_fr: float     # minimum compatible microcatheter in French

    @property
    def display_label(self) -> str:
        return (
            f"{self.name}  "
            f"(Ø{self.diameter_mm:.1f} mm × {self.length_mm:.0f} mm)"
        )

    @property
    def info_text(self) -> str:
        return (
            f"Tipo: {self.stent_type.value}\n"
            f"Diámetro: {self.diameter_mm:.1f} mm  |  Longitud: {self.length_mm:.0f} mm\n"
            f"Porosidad: {self.porosity_pct:.0f} %  |  "
            f"Guía: {self.compatible_wire}\n"
            f"Catéter mín.: {self.catheter_id_fr:.1f} Fr  |  {self.manufacturer}"
        )


# ── Catalogue ─────────────────────────────────────────────────────────────── #

STENT_CATALOGUE: list[StentSpec] = [

    # ── Pipeline Embolization Device (Medtronic) ─────────────────────── #
    StentSpec("Pipeline PED 2.5×10",  StentType.FLOW_DIVERTER, 2.5, 10, 30, 32, 48, "Medtronic",    '0.027"', 2.8),
    StentSpec("Pipeline PED 3.0×14",  StentType.FLOW_DIVERTER, 3.0, 14, 30, 32, 48, "Medtronic",    '0.027"', 2.8),
    StentSpec("Pipeline PED 3.5×18",  StentType.FLOW_DIVERTER, 3.5, 18, 30, 32, 48, "Medtronic",    '0.027"', 2.8),
    StentSpec("Pipeline PED 4.0×20",  StentType.FLOW_DIVERTER, 4.0, 20, 30, 32, 48, "Medtronic",    '0.027"', 2.8),
    StentSpec("Pipeline PED 4.5×25",  StentType.FLOW_DIVERTER, 4.5, 25, 30, 32, 48, "Medtronic",    '0.027"', 2.8),
    StentSpec("Pipeline PED 5.0×30",  StentType.FLOW_DIVERTER, 5.0, 30, 30, 32, 48, "Medtronic",    '0.027"', 2.8),

    # ── Surpass Streamline (Stryker) ─────────────────────────────────── #
    StentSpec("Surpass 2.5×15",       StentType.FLOW_DIVERTER, 2.5, 15, 32, 35, 72, "Stryker",      '0.027"', 2.8),
    StentSpec("Surpass 3.0×20",       StentType.FLOW_DIVERTER, 3.0, 20, 32, 35, 72, "Stryker",      '0.027"', 2.8),
    StentSpec("Surpass 3.5×25",       StentType.FLOW_DIVERTER, 3.5, 25, 32, 35, 72, "Stryker",      '0.027"', 2.8),
    StentSpec("Surpass 4.0×30",       StentType.FLOW_DIVERTER, 4.0, 30, 32, 35, 72, "Stryker",      '0.027"', 2.8),

    # ── FRED (MicroVention) ───────────────────────────────────────────── #
    StentSpec("FRED 3.0×13",          StentType.FLOW_DIVERTER, 3.0, 13, 35, 48, 48, "MicroVention", '0.027"', 2.8),
    StentSpec("FRED 3.5×15",          StentType.FLOW_DIVERTER, 3.5, 15, 35, 48, 48, "MicroVention", '0.027"', 2.8),
    StentSpec("FRED 4.0×18",          StentType.FLOW_DIVERTER, 4.0, 18, 35, 48, 48, "MicroVention", '0.027"', 2.8),
    StentSpec("FRED 4.5×22",          StentType.FLOW_DIVERTER, 4.5, 22, 35, 48, 48, "MicroVention", '0.027"', 2.8),

    # ── Neuroform Atlas (Stryker) ─────────────────────────────────────── #
    StentSpec("Neuroform Atlas 3.0×15", StentType.INTRACRANIAL, 3.0, 15, 78, 55, 12, "Stryker",  '0.014"', 2.3),
    StentSpec("Neuroform Atlas 3.5×20", StentType.INTRACRANIAL, 3.5, 20, 78, 55, 12, "Stryker",  '0.014"', 2.3),
    StentSpec("Neuroform Atlas 4.0×21", StentType.INTRACRANIAL, 4.0, 21, 78, 55, 12, "Stryker",  '0.014"', 2.3),
    StentSpec("Neuroform Atlas 4.5×24", StentType.INTRACRANIAL, 4.5, 24, 78, 55, 12, "Stryker",  '0.014"', 2.3),

    # ── Enterprise 2 (Codman / DePuy Synthes) ────────────────────────── #
    StentSpec("Enterprise 2  3.0×14",  StentType.INTRACRANIAL, 3.0, 14, 75, 46, 16, "Codman",    '0.014"', 2.3),
    StentSpec("Enterprise 2  3.5×22",  StentType.INTRACRANIAL, 3.5, 22, 75, 46, 16, "Codman",    '0.014"', 2.3),
    StentSpec("Enterprise 2  4.0×22",  StentType.INTRACRANIAL, 4.0, 22, 75, 46, 16, "Codman",    '0.014"', 2.3),

    # ── Leo+ (Balt) ───────────────────────────────────────────────────── #
    StentSpec("Leo+  3.5×18",          StentType.INTRACRANIAL, 3.5, 18, 72, 65, 16, "Balt",      '0.014"', 2.3),
    StentSpec("Leo+  4.0×25",          StentType.INTRACRANIAL, 4.0, 25, 72, 65, 16, "Balt",      '0.014"', 2.3),
    StentSpec("Leo+  4.5×35",          StentType.INTRACRANIAL, 4.5, 35, 72, 65, 16, "Balt",      '0.014"', 2.3),

    # ── LVIS Jr (MicroVention) ────────────────────────────────────────── #
    # Low-profile Visualized Intraluminal Support, 0.017" microcatheter
    # Braided design with ~23% porosity (high metal coverage) and radio-
    # opaque helical markers for precise deployment visualization.
    # Compatible with XT-17 / SL-10 microcatheters (2.1 Fr).
    StentSpec("LVIS Jr 2.5×11",        StentType.BRAIDED, 2.5, 11, 23, 75, 48, "MicroVention", '0.017"', 2.1),
    StentSpec("LVIS Jr 2.5×16",        StentType.BRAIDED, 2.5, 16, 23, 75, 48, "MicroVention", '0.017"', 2.1),
    StentSpec("LVIS Jr 2.5×22",        StentType.BRAIDED, 2.5, 22, 23, 75, 48, "MicroVention", '0.017"', 2.1),
    StentSpec("LVIS Jr 3.0×18",        StentType.BRAIDED, 3.0, 18, 23, 75, 48, "MicroVention", '0.017"', 2.1),
    StentSpec("LVIS Jr 3.0×25",        StentType.BRAIDED, 3.0, 25, 23, 75, 48, "MicroVention", '0.017"', 2.1),
    StentSpec("LVIS Jr 3.0×33",        StentType.BRAIDED, 3.0, 33, 23, 75, 48, "MicroVention", '0.017"', 2.1),

    # ── LVIS (MicroVention) ───────────────────────────────────────────── #
    # Standard LVIS for larger vessels (≥3.5 mm), 0.021" microcatheter.
    StentSpec("LVIS 3.5×15",           StentType.BRAIDED, 3.5, 15, 23, 90, 64, "MicroVention", '0.021"', 2.3),
    StentSpec("LVIS 3.5×25",           StentType.BRAIDED, 3.5, 25, 23, 90, 64, "MicroVention", '0.021"', 2.3),
    StentSpec("LVIS 4.0×20",           StentType.BRAIDED, 4.0, 20, 23, 90, 64, "MicroVention", '0.021"', 2.3),
    StentSpec("LVIS 4.0×32",           StentType.BRAIDED, 4.0, 32, 23, 90, 64, "MicroVention", '0.021"', 2.3),
    StentSpec("LVIS 4.5×25",           StentType.BRAIDED, 4.5, 25, 23, 90, 64, "MicroVention", '0.021"', 2.3),
    StentSpec("LVIS 4.5×35",           StentType.BRAIDED, 4.5, 35, 23, 90, 64, "MicroVention", '0.021"', 2.3),
]


def stents_for_vessel(vessel_diameter_mm: float) -> list[StentSpec]:
    """
    Return stents whose nominal diameter is appropriate for *vessel_diameter_mm*.

    Rule of thumb: stent_diameter = vessel × 1.0–1.20 (slight oversizing).
    """
    lo = vessel_diameter_mm * 0.85
    hi = vessel_diameter_mm * 1.25
    return [s for s in STENT_CATALOGUE if lo <= s.diameter_mm <= hi]
