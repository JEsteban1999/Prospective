"""Embolization coil library — MOD-04 / A-03-07.

Covers the most common endovascular coil systems used for cerebral
aneurysm embolization (sac packing).  Coils are deployed via
microcatheter into the aneurysm sac to promote thrombosis.

Treatment strategy
------------------
1. Framing coil  — large, stiff, shapes to dome wall (select Ø ≥ dome Ø)
2. Filling coils — standard helical coils fill the interior volume
3. Finishing coils — ultra-soft, pack residual voids, final packing

Packing density
---------------
Target packing density  ≥ 25 % of aneurysm volume is associated with
lower recurrence rates (Sluzewski et al., *AJNR* 2004).

References: manufacturer IFUs and published sizing charts.
All dimensions in mm / cm.  Wire diameter in micrometres (µm).
This module is pure data — no VTK or Qt dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto


# ──────────────────────────────────────────────────────────────────────────── #
# Enums                                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

class CoilType(Enum):
    FRAMING    = "Enmarcado"      # first coil — large, stiff frame
    FILLING    = "Relleno"        # standard helical filling
    FINISHING  = "Acabado"        # ultra-soft, final packing
    COMPLEX_3D = "Complejo 3D"    # random 3-D shape for irregular sacs
    HYDROCOIL  = "Hidrocoil"      # hydrogel-expanding (MicroVention)


# ──────────────────────────────────────────────────────────────────────────── #
# Data model                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class CoilSpec:
    """Geometric and clinical specification of one coil model."""

    name: str
    coil_type: CoilType
    diameter_mm: float       # nominal coil diameter (match dome diameter)
    length_cm: float         # stretched wire length
    shape_3d: str            # "Esférico", "Helicoidal", "Complejo 3D", etc.
    wire_diameter_um: float  # bare platinum wire thickness in µm
    manufacturer: str
    compatible_wire: str     # required microwire ('0.014"' or '0.010"')
    catheter_id_fr: float    # minimum microcatheter inner diameter (Fr)

    @property
    def display_label(self) -> str:
        return f"{self.name}  (Ø{self.diameter_mm:.0f} mm × {self.length_cm:.0f} cm)"

    @property
    def info_text(self) -> str:
        return (
            f"Tipo: {self.coil_type.value}\n"
            f"Diámetro: {self.diameter_mm:.0f} mm  |  Longitud: {self.length_cm:.0f} cm\n"
            f"Forma: {self.shape_3d}  |  Guía: {self.compatible_wire}\n"
            f"Catéter mín.: {self.catheter_id_fr:.1f} Fr  |  {self.manufacturer}"
        )

    @property
    def wire_volume_mm3(self) -> float:
        """Bare platinum wire volume (cylinder approximation)."""
        r_mm = (self.wire_diameter_um * 1e-3) / 2.0
        return math.pi * r_mm * r_mm * self.length_cm * 10.0


# ──────────────────────────────────────────────────────────────────────────── #
# Catalogue                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

COIL_CATALOGUE: list[CoilSpec] = [

    # ══════════════════════════════════════════════════════════════════════
    # Stryker — Target 360° / Target Ultra / Target Nano
    # Soft detachable coils with uniform softness distribution
    # ══════════════════════════════════════════════════════════════════════

    # Target 360° (complex 3-D framing)
    CoilSpec("Target 360° 4mm×8cm",   CoilType.FRAMING,   4,  8, "Complejo 3D", 254, "Stryker", '0.014"', 1.7),
    CoilSpec("Target 360° 5mm×10cm",  CoilType.FRAMING,   5, 10, "Complejo 3D", 254, "Stryker", '0.014"', 1.7),
    CoilSpec("Target 360° 6mm×15cm",  CoilType.FRAMING,   6, 15, "Complejo 3D", 254, "Stryker", '0.014"', 1.7),
    CoilSpec("Target 360° 8mm×20cm",  CoilType.FRAMING,   8, 20, "Complejo 3D", 254, "Stryker", '0.014"', 1.7),
    CoilSpec("Target 360° 10mm×25cm", CoilType.FRAMING,  10, 25, "Complejo 3D", 254, "Stryker", '0.014"', 1.7),
    CoilSpec("Target 360° 12mm×30cm", CoilType.FRAMING,  12, 30, "Complejo 3D", 254, "Stryker", '0.014"', 1.7),

    # Target Ultra (soft filling)
    CoilSpec("Target Ultra 3mm×6cm",  CoilType.FILLING,   3,  6, "Helicoidal",  254, "Stryker", '0.014"', 1.7),
    CoilSpec("Target Ultra 4mm×8cm",  CoilType.FILLING,   4,  8, "Helicoidal",  254, "Stryker", '0.014"', 1.7),
    CoilSpec("Target Ultra 5mm×12cm", CoilType.FILLING,   5, 12, "Helicoidal",  254, "Stryker", '0.014"', 1.7),
    CoilSpec("Target Ultra 6mm×15cm", CoilType.FILLING,   6, 15, "Helicoidal",  254, "Stryker", '0.014"', 1.7),
    CoilSpec("Target Ultra 7mm×18cm", CoilType.FILLING,   7, 18, "Helicoidal",  254, "Stryker", '0.014"', 1.7),

    # Target Nano (finishing, 0.010" compatible)
    CoilSpec("Target Nano 2mm×4cm",   CoilType.FINISHING,  2,  4, "Helicoidal", 127, "Stryker", '0.010"', 1.5),
    CoilSpec("Target Nano 3mm×6cm",   CoilType.FINISHING,  3,  6, "Helicoidal", 127, "Stryker", '0.010"', 1.5),
    CoilSpec("Target Nano 4mm×8cm",   CoilType.FINISHING,  4,  8, "Helicoidal", 127, "Stryker", '0.010"', 1.5),

    # ══════════════════════════════════════════════════════════════════════
    # Penumbra — Ruby / Coil 400
    # ══════════════════════════════════════════════════════════════════════

    # Ruby (large framing, 0.027" compatible — fills large sacs quickly)
    CoilSpec("Ruby 6mm×20cm",         CoilType.FRAMING,   6, 20, "Complejo 3D", 254, "Penumbra", '0.027"', 2.8),
    CoilSpec("Ruby 8mm×25cm",         CoilType.FRAMING,   8, 25, "Complejo 3D", 254, "Penumbra", '0.027"', 2.8),
    CoilSpec("Ruby 10mm×30cm",        CoilType.FRAMING,  10, 30, "Complejo 3D", 254, "Penumbra", '0.027"', 2.8),
    CoilSpec("Ruby 12mm×40cm",        CoilType.FRAMING,  12, 40, "Complejo 3D", 254, "Penumbra", '0.027"', 2.8),
    CoilSpec("Ruby 14mm×50cm",        CoilType.FRAMING,  14, 50, "Complejo 3D", 254, "Penumbra", '0.027"', 2.8),

    # Coil 400 (standard filling, 0.014")
    CoilSpec("Coil 400 3mm×6cm",      CoilType.FILLING,   3,  6, "Helicoidal",  254, "Penumbra", '0.014"', 1.7),
    CoilSpec("Coil 400 4mm×8cm",      CoilType.FILLING,   4,  8, "Helicoidal",  254, "Penumbra", '0.014"', 1.7),
    CoilSpec("Coil 400 5mm×10cm",     CoilType.FILLING,   5, 10, "Helicoidal",  254, "Penumbra", '0.014"', 1.7),
    CoilSpec("Coil 400 6mm×15cm",     CoilType.FILLING,   6, 15, "Helicoidal",  254, "Penumbra", '0.014"', 1.7),

    # ══════════════════════════════════════════════════════════════════════
    # MicroVention — HydroCoil / MicroPlex 18 & 10
    # HydroCoil expands up to 9× after contact with blood/saline
    # ══════════════════════════════════════════════════════════════════════

    # HydroCoil (expanding hydrogel coating — superior packing density)
    CoilSpec("HydroCoil 4mm×8cm",     CoilType.HYDROCOIL,  4,  8, "Hidrocoil",  254, "MicroVention", '0.014"', 1.7),
    CoilSpec("HydroCoil 5mm×12cm",    CoilType.HYDROCOIL,  5, 12, "Hidrocoil",  254, "MicroVention", '0.014"', 1.7),
    CoilSpec("HydroCoil 6mm×15cm",    CoilType.HYDROCOIL,  6, 15, "Hidrocoil",  254, "MicroVention", '0.014"', 1.7),
    CoilSpec("HydroCoil 8mm×20cm",    CoilType.HYDROCOIL,  8, 20, "Hidrocoil",  254, "MicroVention", '0.014"', 1.7),
    CoilSpec("HydroCoil 10mm×25cm",   CoilType.HYDROCOIL, 10, 25, "Hidrocoil",  254, "MicroVention", '0.014"', 1.7),

    # MicroPlex 18 (framing/filling)
    CoilSpec("MicroPlex 18 4mm×8cm",  CoilType.FRAMING,   4,  8, "Complejo 3D", 254, "MicroVention", '0.014"', 1.7),
    CoilSpec("MicroPlex 18 6mm×15cm", CoilType.FRAMING,   6, 15, "Complejo 3D", 254, "MicroVention", '0.014"', 1.7),
    CoilSpec("MicroPlex 18 8mm×20cm", CoilType.FRAMING,   8, 20, "Complejo 3D", 254, "MicroVention", '0.014"', 1.7),

    # MicroPlex 10 (finishing, soft, 0.010")
    CoilSpec("MicroPlex 10 2mm×3cm",  CoilType.FINISHING,  2,  3, "Helicoidal", 127, "MicroVention", '0.010"', 1.5),
    CoilSpec("MicroPlex 10 3mm×6cm",  CoilType.FINISHING,  3,  6, "Helicoidal", 127, "MicroVention", '0.010"', 1.5),
    CoilSpec("MicroPlex 10 4mm×8cm",  CoilType.FINISHING,  4,  8, "Helicoidal", 127, "MicroVention", '0.010"', 1.5),

    # ══════════════════════════════════════════════════════════════════════
    # Medtronic — Axium 3D / Axium Prime
    # ══════════════════════════════════════════════════════════════════════

    CoilSpec("Axium 3D 4mm×10cm",     CoilType.COMPLEX_3D, 4, 10, "Complejo 3D", 254, "Medtronic", '0.014"', 1.7),
    CoilSpec("Axium 3D 6mm×15cm",     CoilType.COMPLEX_3D, 6, 15, "Complejo 3D", 254, "Medtronic", '0.014"', 1.7),
    CoilSpec("Axium 3D 8mm×20cm",     CoilType.COMPLEX_3D, 8, 20, "Complejo 3D", 254, "Medtronic", '0.014"', 1.7),
    CoilSpec("Axium Prime 4mm×8cm",   CoilType.FILLING,    4,  8, "Helicoidal",  254, "Medtronic", '0.014"', 1.7),
    CoilSpec("Axium Prime 6mm×15cm",  CoilType.FILLING,    6, 15, "Helicoidal",  254, "Medtronic", '0.014"', 1.7),
]


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

def coils_for_aneurysm(
    dome_diameter_mm: float,
    coil_type: CoilType | None = None,
) -> list[CoilSpec]:
    """
    Return coils appropriate for an aneurysm of *dome_diameter_mm*.

    Sizing rule (framing coil): coil Ø ≥ dome Ø × 1.0–1.2.
    Sizing rule (filling/finishing): coil Ø ≤ dome Ø × 0.9.

    If *coil_type* is given, only coils of that type are returned.
    """
    result = []
    for c in COIL_CATALOGUE:
        if coil_type is not None and c.coil_type != coil_type:
            continue
        if c.coil_type == CoilType.FRAMING:
            # Framing: coil diameter ≈ dome size or slightly larger
            if dome_diameter_mm * 0.9 <= c.diameter_mm <= dome_diameter_mm * 1.3:
                result.append(c)
        elif c.coil_type == CoilType.HYDROCOIL:
            # HydroCoil expands up to 9×; treat like filling but slightly smaller
            if c.diameter_mm <= dome_diameter_mm * 1.0:
                result.append(c)
        else:
            # Filling / finishing / complex: coil diameter ≤ dome
            if c.diameter_mm <= dome_diameter_mm * 1.0:
                result.append(c)
    return result


def estimate_coil_count(
    aneurysm_volume_mm3: float,
    coil_spec: CoilSpec,
    target_packing_pct: float = 25.0,
) -> int:
    """
    Estimate how many coils of *coil_spec* are needed to achieve
    *target_packing_pct* % packing density in an aneurysm of
    *aneurysm_volume_mm3*.

    Returns the estimated coil count (minimum 1).
    """
    if coil_spec.wire_volume_mm3 <= 0:
        return 1
    target_volume = aneurysm_volume_mm3 * target_packing_pct / 100.0
    n = math.ceil(target_volume / coil_spec.wire_volume_mm3)
    return max(1, n)
