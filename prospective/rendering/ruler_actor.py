"""VTK actor factory for 3-D measurement rulers — Feature 3.

A ruler consists of:
  • A solid line connecting both endpoints.
  • A small sphere marker at each endpoint.
  • A ``vtkBillboardTextActor3D`` label placed at the midpoint showing
    the straight-line distance in millimetres.

All actors are returned as a :class:`RulerActors` named-tuple so the caller
can add/remove them from the renderer individually and toggle visibility.

Usage::

    actors = build_ruler(pt_a, pt_b, color=(1, 0.9, 0))
    for a in actors:
        renderer.AddActor(a)

    # Remove later
    for a in actors:
        renderer.RemoveActor(a)
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import vtk


# ──────────────────────────────────────────────────────────────────────────── #
# Return type                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

class RulerActors(NamedTuple):
    """All VTK actors that make up a single ruler."""
    line:   vtk.vtkActor                    # connecting line
    sphere_a: vtk.vtkActor                  # endpoint A marker
    sphere_b: vtk.vtkActor                  # endpoint B marker
    label:  vtk.vtkBillboardTextActor3D     # distance text

    def __iter__(self):
        return iter((self.line, self.sphere_a, self.sphere_b, self.label))

    def set_visible(self, visible: bool) -> None:
        v = int(visible)
        for actor in self:
            actor.SetVisibility(v)

    def set_color(self, rgb: tuple[float, float, float]) -> None:
        self.line.GetProperty().SetColor(*rgb)
        self.sphere_a.GetProperty().SetColor(*rgb)
        self.sphere_b.GetProperty().SetColor(*rgb)


# ──────────────────────────────────────────────────────────────────────────── #
# Default style constants                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

_DEFAULT_COLOR     = (1.0, 0.85, 0.0)   # amber / gold
_SELECTED_COLOR    = (0.2, 0.85, 1.0)   # cyan
_LINE_WIDTH        = 2.5                 # px
_SPHERE_RADIUS     = 0.9                 # mm
_LABEL_FONT_SIZE   = 16                  # pt


# ──────────────────────────────────────────────────────────────────────────── #
# Public factory                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

def build_ruler(
    pt_a: tuple[float, float, float] | np.ndarray,
    pt_b: tuple[float, float, float] | np.ndarray,
    color: tuple[float, float, float] = _DEFAULT_COLOR,
    label_suffix: str = "",
) -> RulerActors:
    """
    Build a complete ruler assembly between *pt_a* and *pt_b*.

    Parameters
    ----------
    pt_a, pt_b : (3,) array-like in world-space mm
    color : RGB triple in [0, 1]
    label_suffix : extra text appended to the distance label (e.g. measurement name)

    Returns
    -------
    RulerActors
    """
    a = np.asarray(pt_a, dtype=float)
    b = np.asarray(pt_b, dtype=float)
    dist_mm = float(np.linalg.norm(b - a))
    mid = (a + b) / 2.0

    line_actor   = _build_line(a, b, color)
    sphere_a     = _build_sphere(a, color)
    sphere_b     = _build_sphere(b, color)
    label_actor  = _build_label(mid, dist_mm, label_suffix)

    return RulerActors(line_actor, sphere_a, sphere_b, label_actor)


def ruler_distance(actors: RulerActors) -> float:
    """Return the Euclidean distance (mm) stored in the ruler label."""
    text = actors.label.GetInput()
    # Label format: "12.34 mm [suffix]"
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return 0.0


# ──────────────────────────────────────────────────────────────────────────── #
# Internal helpers                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _build_line(
    a: np.ndarray,
    b: np.ndarray,
    color: tuple,
) -> vtk.vtkActor:
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(*a)
    pts.InsertNextPoint(*b)

    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, 0)
    line.GetPointIds().SetId(1, 1)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(cells)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(_LINE_WIDTH)
    actor.GetProperty().SetOpacity(0.95)
    actor.GetProperty().SetLighting(False)   # always visible regardless of scene light
    return actor


def _build_sphere(
    pt: np.ndarray,
    color: tuple,
) -> vtk.vtkActor:
    src = vtk.vtkSphereSource()
    src.SetCenter(*pt)
    src.SetRadius(_SPHERE_RADIUS)
    src.SetPhiResolution(12)
    src.SetThetaResolution(12)
    src.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(src.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(0.95)
    return actor


def _build_label(
    mid: np.ndarray,
    dist_mm: float,
    suffix: str,
) -> vtk.vtkBillboardTextActor3D:
    text = f"{dist_mm:.2f} mm"
    if suffix:
        text += f"  {suffix}"

    actor = vtk.vtkBillboardTextActor3D()
    actor.SetInput(text)
    actor.SetPosition(*mid)

    tp = actor.GetTextProperty()
    tp.SetFontSize(_LABEL_FONT_SIZE)
    tp.SetColor(1.0, 1.0, 1.0)
    tp.SetBackgroundColor(0.08, 0.08, 0.08)
    tp.SetBackgroundOpacity(0.65)
    tp.SetBold(True)
    tp.SetFontFamilyToArial()
    tp.ShadowOn()
    return actor


# ──────────────────────────────────────────────────────────────────────────── #
# Preview actor (while user is picking second point)                             #
# ──────────────────────────────────────────────────────────────────────────── #

def build_preview_ruler(
    pt_a: tuple[float, float, float] | np.ndarray,
    pt_b: tuple[float, float, float] | np.ndarray,
) -> RulerActors:
    """
    Temporary dashed ruler shown while the user is picking *pt_b*.
    Uses a distinct cyan colour and lower opacity.
    """
    actors = build_ruler(pt_a, pt_b, color=_SELECTED_COLOR)
    actors.line.GetProperty().SetLineStipplePattern(0xAAAA)   # dashed
    actors.line.GetProperty().SetOpacity(0.7)
    actors.sphere_b.GetProperty().SetOpacity(0.5)
    return actors
