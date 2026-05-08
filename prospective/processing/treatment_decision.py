"""Treatment-strategy decision engine — CLIP vs ENDOVASCULAR.

Implements a multi-factor evidence-based scoring algorithm that weighs
morphometric measurements (from MorphometricResult) and optional clinical
inputs (location, rupture status) to produce a quantitative recommendation.

References
----------
- Molyneux et al., ISAT 2002 (NEJM) — ruptured aneurysms
- Spetzler et al., BRAT 2013 — unruptured aneurysms
- Dhar et al. 2008 — bottleneck factor / shape indices
- Raghavan et al. 2005 — undulation index
- AHA/ASA Guidelines 2015 — aneurysm management
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────── #
# Aneurysm location constants                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

LOCATION_UNKNOWN   = "Desconocida / No especificada"
LOCATION_MCA       = "ACM — Arteria Cerebral Media"
LOCATION_ACA_ACOA  = "ACA / ACoA — Arteria Comunicante Anterior"
LOCATION_ICA_PROX  = "ACI proximal (segm. cavernoso / clinoideo)"
LOCATION_ICA_DIST  = "ACI distal (PCOM / oftálmica)"
LOCATION_PCOM      = "ACoP — Arteria Comunicante Posterior"
LOCATION_BASILAR   = "Basilar (punta, tronco o AICA)"
LOCATION_PICA      = "PICA / Vertebral"
LOCATION_OTHER     = "Otra localización"

LOCATIONS: list[str] = [
    LOCATION_UNKNOWN,
    LOCATION_MCA,
    LOCATION_ACA_ACOA,
    LOCATION_ICA_PROX,
    LOCATION_ICA_DIST,
    LOCATION_PCOM,
    LOCATION_BASILAR,
    LOCATION_PICA,
    LOCATION_OTHER,
]

# ──────────────────────────────────────────────────────────────────────────── #
# Data classes                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class DecisionFactor:
    """One contributing factor in the CLIP vs ENDO decision."""
    name:        str    # Short display name
    detail:      str    # One-line clinical rationale
    direction:   str    # "clip" | "endo" | "neutral"
    points:      int    # Magnitude of contribution (always ≥ 0)


@dataclass
class TreatmentDecision:
    """Full output of the decision engine."""

    # ── Scores ────────────────────────────────────────────────────────── #
    clip_raw:  int   # sum of clip-direction factor points
    endo_raw:  int   # sum of endo-direction factor points
    balance:   int   # clip_raw − endo_raw  (positive → clip preferred)

    # ── Normalised percentages for the gauge ─────────────────────────── #
    clip_pct:  int   # 0-100
    endo_pct:  int   # 0-100

    # ── Recommendation ───────────────────────────────────────────────── #
    recommendation: str   # display string
    recommendation_key: str  # "clip" | "endo" | "mdt" | "surveillance"
    confidence:     str   # "Alta" | "Moderada" | "Baja"
    icon:           str   # single emoji for the recommendation badge

    # ── Factors list ─────────────────────────────────────────────────── #
    factors: List[DecisionFactor] = field(default_factory=list)

    # ── Free-text notes ───────────────────────────────────────────────── #
    notes: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────── #
# Scoring engine                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

def compute_decision(
    morpho,
    location: str = LOCATION_UNKNOWN,
    ruptured: bool = False,
) -> TreatmentDecision:
    """Compute the CLIP vs ENDOVASCULAR recommendation.

    Parameters
    ----------
    morpho:    MorphometricResult — computed morphometry for the aneurysm.
    location:  One of the LOCATIONS constants (or empty string).
    ruptured:  True if the aneurysm has already ruptured (SAH).

    Returns
    -------
    TreatmentDecision  with scores, recommendation and factor breakdown.
    """
    clip_pts = 0
    endo_pts = 0
    factors: list[DecisionFactor] = []
    notes:   list[str]            = []

    def _add(name: str, detail: str, direction: str, pts: int) -> None:
        nonlocal clip_pts, endo_pts
        factors.append(DecisionFactor(name, detail, direction, pts))
        if direction == "clip":
            clip_pts += pts
        elif direction == "endo":
            endo_pts += pts

    # ── Special case: very small aneurysm (< 3 mm) ────────────────────── #
    if morpho.max_diameter_mm < 3.0:
        notes.append(
            "Aneurisma muy pequeño (<3 mm): el riesgo procedimental generalmente "
            "supera el riesgo de ruptura. Se recomienda seguimiento con imagen."
        )
        return TreatmentDecision(
            clip_raw=0, endo_raw=0, balance=0,
            clip_pct=50, endo_pct=50,
            recommendation="VIGILANCIA ACTIVA",
            recommendation_key="surveillance",
            confidence="Alta",
            icon="👁",
            factors=[], notes=notes,
        )

    # ── Factor 1: Neck diameter ─────────────────────────────────────────── #
    neck = morpho.neck_diameter_mm
    if neck < 4.0:
        _add(
            f"Cuello estrecho  ({neck:.1f} mm < 4 mm)",
            "Cuello < 4 mm: retención óptima del coil sin stent de soporte.",
            "endo", 25,
        )
    elif neck <= 5.0:
        _add(
            f"Cuello intermedio  ({neck:.1f} mm, 4–5 mm)",
            "Cuello borderline: posible stent-assisted coiling o clipping.",
            "endo", 5,
        )
    else:
        _add(
            f"Cuello ancho  ({neck:.1f} mm > 5 mm)",
            "Cuello ≥ 5 mm: retención de coil difícil; clipping o flow diverter.",
            "clip", 25,
        )

    # ── Factor 2: Aspect Ratio (AR = dome_height / neck) ─────────────── #
    ar = morpho.aspect_ratio
    if ar > 2.0:
        _add(
            f"Aspect Ratio alto  (AR = {ar:.2f} > 2.0)",
            "AR > 2: domo profundo relativo al cuello — geometría favorable para coiling.",
            "endo", 20,
        )
    elif ar > 1.3:
        _add(
            f"Aspect Ratio moderado  (AR = {ar:.2f}, 1.3–2.0)",
            "AR 1.3–2.0: geometría ligeramente favorable para coiling.",
            "endo", 10,
        )
    else:
        _add(
            f"Aspect Ratio bajo  (AR = {ar:.2f} < 1.3)",
            "AR < 1.3: saco corto y ancho — acceso quirúrgico favorable.",
            "clip", 10,
        )

    # ── Factor 3: Dome-to-Neck Ratio (DNR) ────────────────────────────── #
    dnr = morpho.dome_to_neck_ratio
    if dnr > 2.0:
        _add(
            f"DNR favorable para coiling  (DNR = {dnr:.2f} > 2.0)",
            "DNR > 2: domo amplio relativo al cuello — buena retención de coils.",
            "endo", 15,
        )
    elif dnr > 1.5:
        _add(
            f"DNR moderado  (DNR = {dnr:.2f}, 1.5–2.0)",
            "DNR 1.5–2.0: leve preferencia por coiling.",
            "endo", 8,
        )
    else:
        _add(
            f"DNR bajo  (DNR = {dnr:.2f} < 1.5)",
            "DNR < 1.5: cuello ancho relativo al domo — clipping más efectivo.",
            "clip", 15,
        )

    # ── Factor 4: Maximum diameter ─────────────────────────────────────── #
    diam = morpho.max_diameter_mm
    if diam > 25.0:
        _add(
            f"Aneurisma gigante  (Ø = {diam:.1f} mm > 25 mm)",
            "Gigante (>25 mm): flow diverter (PED) es tratamiento de elección.",
            "endo", 20,
        )
        notes.append(
            "Aneurisma gigante: considerar flow diverter (Pipeline, Surpass) "
            "o bypass quirúrgico con exclusión."
        )
    elif diam >= 12.0:
        _add(
            f"Aneurisma grande  (Ø = {diam:.1f} mm, 12–25 mm)",
            "Grande (12–25 mm): ligera preferencia endovascular; valorar complejidad.",
            "endo", 5,
        )
    elif diam < 5.0:
        _add(
            f"Aneurisma pequeño  (Ø = {diam:.1f} mm < 5 mm)",
            "Pequeño (<5 mm): clipping más fiable para exclusión completa.",
            "clip", 8,
        )
    # 5–12 mm: neutral (no factor added)

    # ── Factor 5: Bottleneck Factor (BF = max_dome_diam / neck) ────────── #
    bf = morpho.bottleneck_factor
    if bf > 2.0:
        _add(
            f"Bottleneck Factor alto  (BF = {bf:.2f} > 2.0)",
            "BF > 2: cuello muy estrecho relativo al domo — ideal para coiling.",
            "endo", 12,
        )
    elif bf > 1.5:
        _add(
            f"Bottleneck Factor moderado  (BF = {bf:.2f}, 1.5–2.0)",
            "BF 1.5–2.0: cuello moderadamente estrecho.",
            "endo", 6,
        )
    elif bf > 0 and bf <= 1.2:
        _add(
            f"Bottleneck Factor bajo  (BF = {bf:.2f} ≤ 1.2)",
            "BF ≤ 1.2: domo ancho (no hay efecto de cuello) — clipping favorable.",
            "clip", 8,
        )

    # ── Factor 6: Undulation Index (UI — dome irregularity) ────────────── #
    ui = morpho.undulation_index
    if ui > 0.20:
        _add(
            f"Domo muy irregular  (UI = {ui:.3f} > 0.20)",
            "UI > 0.20: morfología lobulada; riesgo de llenado incompleto con coils.",
            "clip", 10,
        )
    elif ui > 0.10:
        _add(
            f"Domo moderadamente irregular  (UI = {ui:.3f}, 0.10–0.20)",
            "UI 0.10–0.20: cierta irregularidad; leve preferencia por clipping.",
            "clip", 5,
        )
    elif ui < 0.05:
        _add(
            f"Domo regular  (UI = {ui:.3f} < 0.05)",
            "Domo esférico regular — favorable para empaquetado con coils.",
            "endo", 5,
        )

    # ── Factor 7: Location (clinical input) ────────────────────────────── #
    loc = location.strip()
    if loc == LOCATION_MCA:
        _add(
            "Localización ACM (Arteria Cerebral Media)",
            "ACM: acceso quirúrgico directo — clipping de elección en la mayoría de centros.",
            "clip", 20,
        )
    elif loc == LOCATION_ACA_ACOA:
        _add(
            "Localización ACA / ACoA",
            "ACoA: abordaje quirúrgico bien establecido; ligera preferencia por clipping.",
            "clip", 10,
        )
    elif loc == LOCATION_ICA_PROX:
        _add(
            "Localización ACI proximal (cavernoso / clinoideo)",
            "ACI proximal: acceso endovascular más seguro en la mayoría de casos.",
            "endo", 10,
        )
    elif loc == LOCATION_ICA_DIST:
        _add(
            "Localización ACI distal (PCOM / oftálmica)",
            "ACI distal: factible por ambas vías; leve preferencia endovascular.",
            "endo", 5,
        )
    elif loc == LOCATION_PCOM:
        _add(
            "Localización ACoP (Comunicante Posterior)",
            "PCOM: tratable por ambas vías; el tamaño y morfología determinan la estrategia.",
            "neutral", 0,
        )
    elif loc == LOCATION_BASILAR:
        _add(
            "Localización Basilar (punta, tronco o AICA)",
            "Basilar: acceso quirúrgico de alta complejidad — endovascular de elección.",
            "endo", 25,
        )
    elif loc == LOCATION_PICA:
        _add(
            "Localización PICA / Vertebral",
            "Circulación posterior: endovascular preferido por acceso quirúrgico difícil.",
            "endo", 20,
        )

    # ── Factor 8: Rupture status ────────────────────────────────────────── #
    if ruptured:
        _add(
            "Aneurisma roto (HSA activa)",
            "Roto: ISAT 2002 demostró superioridad de coiling en aneurismas accesibles.",
            "endo", 15,
        )
        notes.append(
            "Aneurisma roto: el coiling es de primera elección si la morfología lo permite "
            "(ISAT 2002). Si no es factible endovascularmente, clipping de urgencia."
        )

    # ── Compute final scores ────────────────────────────────────────────── #
    total = clip_pts + endo_pts
    if total == 0:
        clip_pct = endo_pct = 50
    else:
        clip_pct = round(clip_pts / total * 100)
        endo_pct = 100 - clip_pct

    balance = clip_pts - endo_pts

    # ── Recommendation ─────────────────────────────────────────────────── #
    abs_bal = abs(balance)
    if abs_bal >= 50:
        confidence = "Alta"
    elif abs_bal >= 25:
        confidence = "Moderada"
    else:
        confidence = "Baja"

    if balance >= 20:
        rec      = "CLIPPING QUIRÚRGICO"
        rec_key  = "clip"
        icon     = "✂"
    elif balance <= -20:
        rec      = "TRATAMIENTO ENDOVASCULAR"
        rec_key  = "endo"
        icon     = "💊"
    else:
        rec      = "DISCUSIÓN MULTIDISCIPLINARIA"
        rec_key  = "mdt"
        icon     = "👥"
        if confidence == "Baja":
            notes.append(
                "Resultado ambiguo: se recomienda presentar el caso en sesión "
                "neuroendovascular multidisciplinaria antes de decidir la estrategia."
            )

    # Log summary
    logger.debug(
        "TreatmentDecision: clip=%d  endo=%d  balance=%d  → %s (%s)",
        clip_pts, endo_pts, balance, rec_key, confidence,
    )

    return TreatmentDecision(
        clip_raw=clip_pts,
        endo_raw=endo_pts,
        balance=balance,
        clip_pct=clip_pct,
        endo_pct=endo_pct,
        recommendation=rec,
        recommendation_key=rec_key,
        confidence=confidence,
        icon=icon,
        factors=factors,
        notes=notes,
    )
