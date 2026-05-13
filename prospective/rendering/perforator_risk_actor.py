"""VTK actors for perforator risk visualisation.

Public API
----------
build_risk_actors(result)  → PerforatorRiskActors (iterable of vtkActor)

The returned actors can be added / removed atomically::

    actors = build_risk_actors(result)
    for a in actors:
        renderer.AddActor(a)
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import vtk

from prospective.processing.perforator_risk import (
    PerforatorCandidate,
    PerforatorRiskResult,
)

# ──────────────────────────────────────────────────────────────────────────── #
# Style constants                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

# LUT: 0 = outside (dim), 1 = high (red), 2 = medium (amber), 3 = low (green)
_LUT_COLORS = [
    (0.25, 0.35, 0.50, 0.45),   # 0 — outside zone (slate, semi-transparent)
    (0.95, 0.20, 0.15, 1.00),   # 1 — high risk (red)
    (1.00, 0.65, 0.00, 1.00),   # 2 — medium risk (amber)
    (0.20, 0.85, 0.30, 1.00),   # 3 — low risk (green)
]
_SPHERE_RADIUS_MM = 0.9
_LABEL_FONT_SIZE  = 10
_RING_OPACITY     = 0.07


# ──────────────────────────────────────────────────────────────────────────── #
# Return type                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

class PerforatorRiskActors(NamedTuple):
    """All VTK actors for one perforator risk overlay."""

    overlay:     vtk.vtkActor        # colour-mapped vessel surface
    spheres:     list[vtk.vtkActor]  # candidate marker spheres (one per candidate)
    labels:      list                 # vtkBillboardTextActor3D labels (one per candidate)
    neck_ring:   vtk.vtkActor        # translucent boundary sphere at r_low
    risk_levels: tuple[int, ...] = ()
    # Parallel to spheres/labels: risk_level of the i-th candidate (1=high,2=med,3=low)

    def __iter__(self):
        yield self.overlay
        yield from self.spheres
        yield from self.labels
        yield self.neck_ring

    def set_visible(self, visible: bool) -> None:
        """Show or hide ALL actors (overlay + markers + labels + neck ring)."""
        v = int(visible)
        for a in self:
            a.SetVisibility(v)

    def set_overlay_visible(self, visible: bool) -> None:
        """Show or hide the colour-mapped vessel surface only."""
        self.overlay.SetVisibility(int(visible))

    def set_neck_ring_visible(self, visible: bool) -> None:
        """Show or hide the translucent risk-boundary sphere."""
        self.neck_ring.SetVisibility(int(visible))

    def set_risk_level_visible(self, risk_level: int, visible: bool) -> None:
        """Show or hide sphere markers + labels for one risk level (1/2/3)."""
        v = int(visible)
        for i, lvl in enumerate(self.risk_levels):
            if lvl == risk_level:
                if i < len(self.spheres):
                    self.spheres[i].SetVisibility(v)
                if i < len(self.labels):
                    self.labels[i].SetVisibility(v)


# ──────────────────────────────────────────────────────────────────────────── #
# Public factory                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

def build_risk_actors(result: PerforatorRiskResult) -> PerforatorRiskActors:
    """
    Build all VTK actors for a :class:`~prospective.processing.perforator_risk.PerforatorRiskResult`.

    Parameters
    ----------
    result : output of ``compute_perforator_risk``

    Returns
    -------
    PerforatorRiskActors — NamedTuple, iterable over all actors.
    """
    return PerforatorRiskActors(
        overlay     = _build_overlay(result),
        spheres     = _build_spheres(result.candidates),
        labels      = _build_labels(result.candidates),
        neck_ring   = _build_neck_ring(result.neck_origin, result.zone_radii_mm[2]),
        risk_levels = tuple(c.risk_level for c in result.candidates),
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Private helpers                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

def _build_lut() -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(4)
    for i, (r, g, b, a) in enumerate(_LUT_COLORS):
        lut.SetTableValue(i, r, g, b, a)
    lut.SetTableRange(0, 3)
    lut.Build()
    return lut


def _build_overlay(result: PerforatorRiskResult) -> vtk.vtkActor:
    """Colour-mapped vessel surface based on RiskZone scalars."""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(result.risk_poly)
    mapper.ScalarVisibilityOn()
    mapper.SetScalarRange(0, 3)
    mapper.SetLookupTable(_build_lut())
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarModeToUsePointData()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    p = actor.GetProperty()
    p.SetOpacity(0.80)
    p.SetAmbient(0.30)
    p.SetDiffuse(0.70)
    p.SetSpecular(0.10)
    return actor


def _build_spheres(candidates: list[PerforatorCandidate]) -> list[vtk.vtkActor]:
    """Small sphere markers at each candidate position, colour-coded by risk."""
    actors: list[vtk.vtkActor] = []
    for c in candidates:
        src = vtk.vtkSphereSource()
        src.SetCenter(*c.position)
        src.SetRadius(_SPHERE_RADIUS_MM)
        src.SetThetaResolution(16)
        src.SetPhiResolution(16)
        src.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(src.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        r, g, b = c.risk_color
        p = actor.GetProperty()
        p.SetColor(r, g, b)
        p.SetOpacity(0.92)
        p.SetAmbient(0.50)
        p.SetDiffuse(0.60)
        p.SetSpecular(0.40)
        p.SetSpecularPower(30.0)
        actors.append(actor)
    return actors


def _build_labels(candidates: list[PerforatorCandidate]) -> list:
    """Billboard text labels showing index, risk and distance."""
    label_actors = []
    for c in candidates:
        text  = f"P{c.index + 1} · {c.risk_label}\n{c.distance_to_neck_mm:.1f} mm"
        label = vtk.vtkBillboardTextActor3D()
        label.SetInput(text)
        # Offset slightly so label doesn't overlap the sphere
        label.SetPosition(
            c.position[0] + 1.2,
            c.position[1] + 1.2,
            c.position[2] + 1.2,
        )
        tp = label.GetTextProperty()
        tp.SetFontSize(_LABEL_FONT_SIZE)
        r, g, b = c.risk_color
        tp.SetColor(r, g, b)
        tp.SetBackgroundColor(0.06, 0.06, 0.06)
        tp.SetBackgroundOpacity(0.60)
        tp.SetBold(True)
        tp.ShadowOn()
        label_actors.append(label)
    return label_actors


def _build_neck_ring(
    neck_origin: tuple[float, float, float],
    radius_mm: float,
    resolution: int = 64,
) -> vtk.vtkActor:
    """Translucent wireframe sphere shell marking the outer risk boundary."""
    src = vtk.vtkSphereSource()
    src.SetCenter(*neck_origin)
    src.SetRadius(radius_mm)
    src.SetThetaResolution(resolution)
    src.SetPhiResolution(resolution)
    src.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(src.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    p = actor.GetProperty()
    p.SetColor(0.85, 0.85, 0.20)       # pale yellow
    p.SetOpacity(_RING_OPACITY)
    p.SetRepresentationToWireframe()
    p.SetLineWidth(0.6)
    p.SetLighting(False)
    return actor
