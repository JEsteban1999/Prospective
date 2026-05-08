"""Aneurysm clip library — standard sizes and types.

Reference dimensions based on commonly used titanium clip systems
(Sugita, Aesculap Yasargil, Codman).  All lengths in mm.

This module is pure data — no VTK or Qt dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar


class ClipShape(Enum):
    STRAIGHT    = "Recto"
    CURVED      = "Curvo"
    ANGLED      = "Angulado 90°"
    ANGLED_45   = "Angulado 45°"
    BAYONET     = "Bayoneta"
    FENESTRATED = "Fenestrado"


@dataclass(frozen=True)
class ClipSpec:
    """Geometric specification of one clip model."""
    name: str               # display name
    shape: ClipShape
    blade_length_mm: float  # jaw opening span
    blade_width_mm: float   # blade thickness (transverse)
    blade_height_mm: float  # blade depth (along jaw axis)
    spring_length_mm: float # hinge body length
    closing_force_g: float  # approximate closing force in grams
    manufacturer: str

    @property
    def display_label(self) -> str:
        return f"{self.name}  ({self.blade_length_mm:.0f} mm)"


# ── Catalogue ─────────────────────────────────────────────────────────────── #
# Simplified subset covering the most common sizes used in intracranial
# aneurysm surgery.

CLIP_CATALOGUE: list[ClipSpec] = [

    # ══════════════════════════════════════════════════════════════════════
    # Yasargil (Karl Storz) — gold standard in cerebrovascular surgery
    # Titanium alloy, closing force ~75–200 g
    # ══════════════════════════════════════════════════════════════════════

    # ── Yasargil Straight ─────────────────────────────────────────────── #
    ClipSpec("Yasargil Mini recto",       ClipShape.STRAIGHT,   5.0, 1.0, 0.8,  5.5,  75,  "Yasargil/KS"),
    ClipSpec("Yasargil Recto 7mm",        ClipShape.STRAIGHT,   7.0, 1.1, 0.9,  6.5, 100,  "Yasargil/KS"),
    ClipSpec("Yasargil Recto 9mm",        ClipShape.STRAIGHT,   9.0, 1.3, 1.0,  7.0, 120,  "Yasargil/KS"),
    ClipSpec("Yasargil Recto 11mm",       ClipShape.STRAIGHT,  11.0, 1.4, 1.0,  7.5, 140,  "Yasargil/KS"),
    ClipSpec("Yasargil Recto 14mm",       ClipShape.STRAIGHT,  14.0, 1.5, 1.1,  8.0, 160,  "Yasargil/KS"),
    ClipSpec("Yasargil Recto 19mm",       ClipShape.STRAIGHT,  19.0, 1.5, 1.1,  9.0, 190,  "Yasargil/KS"),

    # ── Yasargil Curved ───────────────────────────────────────────────── #
    ClipSpec("Yasargil Curvo 7mm",        ClipShape.CURVED,     7.0, 1.1, 0.9,  6.5, 100,  "Yasargil/KS"),
    ClipSpec("Yasargil Curvo 9mm",        ClipShape.CURVED,     9.0, 1.3, 1.0,  7.0, 120,  "Yasargil/KS"),
    ClipSpec("Yasargil Curvo 11mm",       ClipShape.CURVED,    11.0, 1.4, 1.0,  7.5, 140,  "Yasargil/KS"),
    ClipSpec("Yasargil Curvo 14mm",       ClipShape.CURVED,    14.0, 1.5, 1.1,  8.0, 160,  "Yasargil/KS"),

    # ── Yasargil Angled 45° ───────────────────────────────────────────── #
    ClipSpec("Yasargil Angulado 45° 7mm", ClipShape.ANGLED_45,  7.0, 1.1, 0.9,  6.5, 100,  "Yasargil/KS"),
    ClipSpec("Yasargil Angulado 45° 9mm", ClipShape.ANGLED_45,  9.0, 1.3, 1.0,  7.0, 120,  "Yasargil/KS"),
    ClipSpec("Yasargil Angulado 45° 11mm",ClipShape.ANGLED_45, 11.0, 1.4, 1.0,  7.5, 140,  "Yasargil/KS"),

    # ── Yasargil Angled 90° ───────────────────────────────────────────── #
    ClipSpec("Yasargil Angulado 90° 7mm", ClipShape.ANGLED,     7.0, 1.1, 0.9,  6.5, 100,  "Yasargil/KS"),
    ClipSpec("Yasargil Angulado 90° 9mm", ClipShape.ANGLED,     9.0, 1.3, 1.0,  7.0, 120,  "Yasargil/KS"),

    # ── Yasargil Bayonet ──────────────────────────────────────────────── #
    ClipSpec("Yasargil Bayoneta 7mm",     ClipShape.BAYONET,    7.0, 1.1, 0.9,  8.0, 105,  "Yasargil/KS"),
    ClipSpec("Yasargil Bayoneta 11mm",    ClipShape.BAYONET,   11.0, 1.4, 1.0, 10.0, 145,  "Yasargil/KS"),
    ClipSpec("Yasargil Bayoneta 14mm",    ClipShape.BAYONET,   14.0, 1.5, 1.1, 11.0, 165,  "Yasargil/KS"),

    # ── Yasargil Fenestrated ──────────────────────────────────────────── #
    ClipSpec("Yasargil Fenestrado 7mm",   ClipShape.FENESTRATED, 7.0, 1.1, 0.9,  7.0, 105,  "Yasargil/KS"),
    ClipSpec("Yasargil Fenestrado 9mm",   ClipShape.FENESTRATED, 9.0, 1.3, 1.0,  7.5, 125,  "Yasargil/KS"),
    ClipSpec("Yasargil Fenestrado 11mm",  ClipShape.FENESTRATED,11.0, 1.4, 1.0,  8.0, 145,  "Yasargil/KS"),

    # ══════════════════════════════════════════════════════════════════════
    # Sugita (Mizuho) — widely used in Asia and Latin America
    # ══════════════════════════════════════════════════════════════════════

    # ── Straight ──────────────────────────────────────────────────────── #
    ClipSpec("Sugita Mini recto",         ClipShape.STRAIGHT,   5.0, 1.0, 0.8,  5.0,  70,  "Sugita"),
    ClipSpec("Sugita Recto S",            ClipShape.STRAIGHT,   7.0, 1.2, 0.9,  6.0,  80,  "Sugita"),
    ClipSpec("Sugita Recto M",            ClipShape.STRAIGHT,  10.0, 1.4, 1.0,  7.0,  90,  "Sugita"),
    ClipSpec("Sugita Recto L",            ClipShape.STRAIGHT,  12.0, 1.4, 1.0,  7.0,  95,  "Sugita"),
    ClipSpec("Sugita Recto XL",           ClipShape.STRAIGHT,  15.0, 1.5, 1.1,  8.0, 100,  "Sugita"),
    ClipSpec("Sugita Recto XXL",          ClipShape.STRAIGHT,  20.0, 1.5, 1.1,  9.0, 110,  "Sugita"),

    # ── Curved ────────────────────────────────────────────────────────── #
    ClipSpec("Sugita Curvo Mini",         ClipShape.CURVED,     5.0, 1.0, 0.8,  5.0,  70,  "Sugita"),
    ClipSpec("Sugita Curvo S",            ClipShape.CURVED,     7.0, 1.2, 0.9,  6.0,  80,  "Sugita"),
    ClipSpec("Sugita Curvo M",            ClipShape.CURVED,    10.0, 1.4, 1.0,  7.0,  90,  "Sugita"),
    ClipSpec("Sugita Curvo L",            ClipShape.CURVED,    12.0, 1.4, 1.0,  7.0,  95,  "Sugita"),

    # ── Fenestrated ───────────────────────────────────────────────────── #
    ClipSpec("Sugita Fenestrado S",       ClipShape.FENESTRATED, 7.0, 1.2, 0.9,  7.0,  85,  "Sugita"),
    ClipSpec("Sugita Fenestrado M",       ClipShape.FENESTRATED,10.0, 1.4, 1.0,  8.0,  95,  "Sugita"),

    # ══════════════════════════════════════════════════════════════════════
    # Aesculap (B. Braun) — standard European system
    # ══════════════════════════════════════════════════════════════════════

    ClipSpec("Aesculap Angulado 90° S",   ClipShape.ANGLED,     7.0, 1.2, 0.9,  6.0,  80,  "Aesculap"),
    ClipSpec("Aesculap Angulado 90° M",   ClipShape.ANGLED,    10.0, 1.4, 1.0,  7.0,  90,  "Aesculap"),
    ClipSpec("Aesculap Recto S",          ClipShape.STRAIGHT,   7.0, 1.2, 0.9,  6.0,  80,  "Aesculap"),
    ClipSpec("Aesculap Recto M",          ClipShape.STRAIGHT,  10.0, 1.4, 1.0,  7.0,  90,  "Aesculap"),
    ClipSpec("Aesculap Fenestrado M",     ClipShape.FENESTRATED,10.0, 1.4, 1.0,  8.0,  92,  "Aesculap"),

    # ══════════════════════════════════════════════════════════════════════
    # Codman (DePuy Synthes / J&J) — common in US / Latin America
    # ══════════════════════════════════════════════════════════════════════

    ClipSpec("Codman Bayoneta S",         ClipShape.BAYONET,    7.0, 1.2, 0.9,  8.0,  85,  "Codman"),
    ClipSpec("Codman Bayoneta M",         ClipShape.BAYONET,   10.0, 1.4, 1.0, 10.0,  95,  "Codman"),
    ClipSpec("Codman Recto S",            ClipShape.STRAIGHT,   7.0, 1.2, 0.9,  6.0,  80,  "Codman"),
    ClipSpec("Codman Recto M",            ClipShape.STRAIGHT,  10.0, 1.4, 1.0,  7.0,  90,  "Codman"),
]


def clips_for_neck(neck_diameter_mm: float) -> list[ClipSpec]:
    """
    Return clips whose blade length is appropriate for a given neck diameter.

    Rule of thumb: blade_length ≥ neck_diameter + 1 mm
    and blade_length ≤ neck_diameter * 2.5
    """
    lo = neck_diameter_mm + 1.0
    hi = neck_diameter_mm * 2.5
    return [c for c in CLIP_CATALOGUE if lo <= c.blade_length_mm <= hi]
