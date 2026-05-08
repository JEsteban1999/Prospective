"""Clip selection assistant — Feature 8.

Scores every clip in the catalogue against the patient's morphometrics and
returns a ranked list of recommendations.

Scoring model (weighted sum, final score 0–100)
------------------------------------------------
1. **Coverage score** (weight 0.45)
   ``coverage_ratio = blade_length / neck_diameter``
   Ideal range 1.2–1.5. Score peaks at 1.35 (Gaussian σ=0.25).

2. **Shape fit score** (weight 0.40)
   Rule-based on clinical guidelines:
   * Wide neck (neck ≥ 5 mm)   → prefer FENESTRATED > CURVED > ANGLED
   * Deep dome  (AR ≥ 1.5)     → prefer ANGLED > BAYONET > CURVED
   * Standard                  → prefer STRAIGHT > CURVED > ANGLED_45

3. **Closing force score** (weight 0.15)
   Optimal window 80–160 g.  Penalty outside range.

Filters applied before scoring
-------------------------------
* blade_length ≥ neck_mm + 1 mm  (minimum safety coverage)
* blade_length ≤ neck_mm × 3.0   (avoid oversized blades)

References
----------
* Lawton 2011, "Seven Aneurysms", clip selection algorithm
* Molyneux et al. neck width ≥ 4 mm as wide-neck threshold
* Pierot & Wakhloo 2013 — shape-based selection rationale
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from prospective.models.clip_library import ClipSpec, ClipShape, CLIP_CATALOGUE

# ──────────────────────────────────────────────────────────────────────────── #
# Constants                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

WIDE_NECK_THRESHOLD_MM: float = 5.0   # neck ≥ 5 mm → wide neck
DEEP_DOME_AR_THRESHOLD: float = 1.5   # AR ≥ 1.5   → deep dome / high rupture risk

# Weights for the three score components (must sum to 1)
_W_COVERAGE: float = 0.45
_W_SHAPE:    float = 0.40
_W_FORCE:    float = 0.15

# Coverage ratio ideal centre and Gaussian σ
_COV_IDEAL: float = 1.35
_COV_SIGMA: float = 0.25

# Closing force optimal window (g)
_FORCE_OPT_LO: float = 80.0
_FORCE_OPT_HI: float = 160.0

# Blade length bounds relative to neck
_BLADE_MIN_OVER:  float = 1.0    # blade_length >= neck + BLADE_MIN_OVER
_BLADE_MAX_RATIO: float = 3.0    # blade_length <= neck * BLADE_MAX_RATIO


# ──────────────────────────────────────────────────────────────────────────── #
# Shape fit tables                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

# shape → fitness score (0–1) for each clinical context
_SHAPE_FIT_WIDE_NECK: dict[ClipShape, float] = {
    ClipShape.FENESTRATED: 1.00,
    ClipShape.CURVED:      0.75,
    ClipShape.ANGLED:      0.65,
    ClipShape.ANGLED_45:   0.60,
    ClipShape.STRAIGHT:    0.50,
    ClipShape.BAYONET:     0.45,
}

_SHAPE_FIT_DEEP_DOME: dict[ClipShape, float] = {
    ClipShape.ANGLED:      1.00,
    ClipShape.BAYONET:     0.90,
    ClipShape.ANGLED_45:   0.85,
    ClipShape.CURVED:      0.70,
    ClipShape.STRAIGHT:    0.55,
    ClipShape.FENESTRATED: 0.40,
}

_SHAPE_FIT_STANDARD: dict[ClipShape, float] = {
    ClipShape.STRAIGHT:    1.00,
    ClipShape.CURVED:      0.90,
    ClipShape.ANGLED_45:   0.70,
    ClipShape.ANGLED:      0.60,
    ClipShape.BAYONET:     0.50,
    ClipShape.FENESTRATED: 0.35,
}


# ──────────────────────────────────────────────────────────────────────────── #
# Result dataclass                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class ClipRecommendation:
    """Scored clip recommendation for a specific patient morphometry."""

    clip:             ClipSpec
    score:            float          # 0–100 composite score
    coverage_ratio:   float          # blade_length / neck_mm
    safety_margin_mm: float          # blade_length − neck_mm
    reasons:          list[str]      = field(default_factory=list)

    @property
    def score_label(self) -> str:
        if self.score >= 75:
            return "Excelente"
        if self.score >= 55:
            return "Bueno"
        if self.score >= 35:
            return "Aceptable"
        return "Marginal"

    @property
    def coverage_label(self) -> str:
        if self.coverage_ratio >= 1.2:
            return f"×{self.coverage_ratio:.2f} ✔"
        return f"×{self.coverage_ratio:.2f} ⚠"

    def to_dict(self) -> dict:
        return {
            "clip":           self.clip.name,
            "shape":          self.clip.shape.value,
            "blade_mm":       self.clip.blade_length_mm,
            "manufacturer":   self.clip.manufacturer,
            "score":          round(self.score, 1),
            "coverage_ratio": round(self.coverage_ratio, 3),
            "safety_mm":      round(self.safety_margin_mm, 2),
            "label":          self.score_label,
        }


# ──────────────────────────────────────────────────────────────────────────── #
# Scoring helpers                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _coverage_score(coverage_ratio: float) -> float:
    """Gaussian-shaped coverage score (0–1) centred at _COV_IDEAL."""
    return math.exp(-0.5 * ((coverage_ratio - _COV_IDEAL) / _COV_SIGMA) ** 2)


def _shape_score(clip: ClipSpec, neck_mm: float, ar: float) -> float:
    """Return shape fitness (0–1) based on clinical context."""
    if neck_mm >= WIDE_NECK_THRESHOLD_MM:
        return _SHAPE_FIT_WIDE_NECK.get(clip.shape, 0.4)
    if ar >= DEEP_DOME_AR_THRESHOLD:
        return _SHAPE_FIT_DEEP_DOME.get(clip.shape, 0.4)
    return _SHAPE_FIT_STANDARD.get(clip.shape, 0.4)


def _force_score(force_g: float) -> float:
    """Closing force fitness (0–1): 1.0 inside window, linear ramp outside."""
    if _FORCE_OPT_LO <= force_g <= _FORCE_OPT_HI:
        return 1.0
    if force_g < _FORCE_OPT_LO:
        return max(0.0, force_g / _FORCE_OPT_LO)
    return max(0.0, 1.0 - (force_g - _FORCE_OPT_HI) / _FORCE_OPT_HI)


def _build_reasons(
    clip: ClipSpec,
    neck_mm: float,
    ar: float,
    coverage_ratio: float,
    safety_mm: float,
) -> list[str]:
    reasons: list[str] = []

    if coverage_ratio >= 1.2:
        reasons.append(f"Cobertura adecuada (×{coverage_ratio:.2f})")
    else:
        reasons.append(f"Cobertura justa (×{coverage_ratio:.2f}) — verificar en IQ")

    if neck_mm >= WIDE_NECK_THRESHOLD_MM and clip.shape == ClipShape.FENESTRATED:
        reasons.append("Fenestrado indicado para cuello ancho")
    elif ar >= DEEP_DOME_AR_THRESHOLD and clip.shape in (ClipShape.ANGLED, ClipShape.BAYONET):
        reasons.append("Angulado/Bayoneta indicado para domo profundo (AR alto)")

    if safety_mm < 1.5:
        reasons.append(f"Margen de seguridad pequeño ({safety_mm:.1f} mm)")
    else:
        reasons.append(f"Margen de seguridad: {safety_mm:.1f} mm")

    return reasons


# ──────────────────────────────────────────────────────────────────────────── #
# Public API                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

def recommend_clips(
    neck_mm: float,
    aspect_ratio: float,
    catalogue: Sequence[ClipSpec] | None = None,
    n: int = 8,
) -> list[ClipRecommendation]:
    """
    Score all clips in *catalogue* and return the top *n* recommendations
    sorted by descending composite score.

    Parameters
    ----------
    neck_mm       : aneurysm neck diameter in mm
    aspect_ratio  : dome-to-neck aspect ratio (unitless)
    catalogue     : clip catalogue to search (defaults to CLIP_CATALOGUE)
    n             : maximum number of results to return

    Returns
    -------
    list of :class:`ClipRecommendation`, best first
    """
    if catalogue is None:
        catalogue = CLIP_CATALOGUE

    if neck_mm <= 0:
        return []

    lo = neck_mm + _BLADE_MIN_OVER
    hi = neck_mm * _BLADE_MAX_RATIO

    recommendations: list[ClipRecommendation] = []
    for clip in catalogue:
        bl = clip.blade_length_mm
        if bl < lo or bl > hi:
            continue   # outside acceptable range

        cov  = bl / neck_mm
        cov_s  = _coverage_score(cov)
        shp_s  = _shape_score(clip, neck_mm, aspect_ratio)
        frc_s  = _force_score(clip.closing_force_g)

        composite = (_W_COVERAGE * cov_s + _W_SHAPE * shp_s + _W_FORCE * frc_s) * 100.0
        safety_mm = bl - neck_mm

        reasons = _build_reasons(clip, neck_mm, aspect_ratio, cov, safety_mm)

        recommendations.append(
            ClipRecommendation(
                clip=clip,
                score=round(composite, 1),
                coverage_ratio=round(cov, 3),
                safety_margin_mm=round(safety_mm, 2),
                reasons=reasons,
            )
        )

    recommendations.sort(key=lambda r: r.score, reverse=True)
    return recommendations[:n]


def recommend_from_morpho(result, catalogue=None, n: int = 8) -> list[ClipRecommendation]:
    """
    Convenience wrapper that extracts *neck_mm* and *aspect_ratio* directly
    from a :class:`~prospective.processing.morphometrics.MorphometricResult`.
    """
    return recommend_clips(
        neck_mm=result.neck_diameter_mm,
        aspect_ratio=result.aspect_ratio,
        catalogue=catalogue,
        n=n,
    )
