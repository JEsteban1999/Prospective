"""3-D morphometric overlay actors — Feature 5.

Builds a set of VTK actors that annotate a MorphometricResult directly in
the planning scene:

  • A translucent disc at the detected neck plane
  • A dashed line from neck centroid to apex (dome height)
  • A solid line spanning the maximum diameter
  • Billboard text labels for key values (Ø_neck, H_dome, Ø_max, AR, DNR)

All actors are grouped in a :class:`MorphoOverlayActors` named-tuple so
the caller can add/remove them atomically.

Usage::

    actors = build_morpho_overlay(result, poly_data)
    for a in actors:
        renderer.AddActor(a)
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from prospective.processing.morphometrics import MorphometricResult


# ──────────────────────────────────────────────────────────────────────────── #
# Style constants                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

_NECK_COLOR    = (0.20, 0.75, 1.00)   # sky blue
_DOME_COLOR    = (1.00, 0.55, 0.10)   # orange
_MAXD_COLOR    = (0.85, 0.20, 0.20)   # red
_DISC_OPACITY  = 0.35
_LINE_WIDTH    = 2.0
_LABEL_SIZE    = 13


# ──────────────────────────────────────────────────────────────────────────── #
# Return type                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

class MorphoOverlayActors(NamedTuple):
    """All VTK actors for one morphometric overlay."""
    neck_disc:    vtk.vtkActor                    # translucent neck-plane disc
    neck_outline: vtk.vtkActor                    # neck circle outline
    dome_line:    vtk.vtkActor                    # neck-to-apex dashed line
    maxd_line:    vtk.vtkActor                    # max-diameter span
    labels:       list                            # vtkBillboardTextActor3D

    def __iter__(self):
        yield self.neck_disc
        yield self.neck_outline
        yield self.dome_line
        yield self.maxd_line
        yield from self.labels

    def set_visible(self, visible: bool) -> None:
        v = int(visible)
        for a in self:
            a.SetVisibility(v)


# ──────────────────────────────────────────────────────────────────────────── #
# Public factory                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

def build_morpho_overlay(
    result: MorphometricResult,
    poly_data: vtk.vtkPolyData,
) -> MorphoOverlayActors:
    """
    Build all morphometric overlay actors from a :class:`MorphometricResult`.

    Parameters
    ----------
    result    : MorphometricResult from MorphometricAnalyzer.analyze()
    poly_data : the isolated aneurysm mesh (used to locate apex point)

    Returns
    -------
    MorphoOverlayActors
    """
    centroid = np.asarray(result.centroid, dtype=float)
    axis     = np.asarray(result.principal_axis, dtype=float)
    axis    /= np.linalg.norm(axis) + 1e-12

    # ── Reconstruct neck position in world space ────────────────────────── #
    pts_arr = _mesh_points(poly_data)
    proj    = (pts_arr - centroid) @ axis
    p_min   = proj.min()
    p_max   = proj.max()
    extent  = p_max - p_min

    neck_pos_world  = centroid + (p_min + result.neck_plane_pos * extent) * axis
    neck_radius     = result.neck_diameter_mm / 2.0

    # Apex = point on mesh farthest from neck plane in axis direction
    apex_world = pts_arr[np.argmax(proj)]

    # Max-diameter endpoints: the two mesh points with greatest distance
    # projected perpendicular to the axis (approximate with bounding box)
    max_d_pt1, max_d_pt2 = _max_diameter_endpoints(pts_arr, centroid, axis,
                                                    result.max_diameter_mm)

    # ── Build actors ────────────────────────────────────────────────────── #
    neck_disc    = _build_disc(neck_pos_world, axis, neck_radius)
    neck_outline = _build_circle(neck_pos_world, axis, neck_radius, _NECK_COLOR)
    dome_line    = _build_line(neck_pos_world, apex_world, _DOME_COLOR, dashed=True)
    maxd_line    = _build_line(max_d_pt1, max_d_pt2, _MAXD_COLOR, dashed=False)

    # ── Labels ──────────────────────────────────────────────────────────── #
    label_actors = []

    # Neck diameter label — offset sideways from neck circle
    perp = _perp_to(axis)
    neck_lbl_pos = neck_pos_world + perp * (neck_radius * 1.3)
    label_actors.append(_label(
        neck_lbl_pos,
        f"Ø cuello: {result.neck_diameter_mm:.2f} mm",
        _NECK_COLOR,
    ))

    # Dome height label — midpoint of dome line
    dome_mid = (neck_pos_world + apex_world) / 2.0 + perp * 1.5
    label_actors.append(_label(
        dome_mid,
        f"H domo: {result.dome_height_mm:.2f} mm",
        _DOME_COLOR,
    ))

    # Max diameter label — midpoint of max-diam line
    maxd_mid = (max_d_pt1 + max_d_pt2) / 2.0 + axis * 2.0
    label_actors.append(_label(
        maxd_mid,
        f"Ø max: {result.max_diameter_mm:.2f} mm",
        _MAXD_COLOR,
    ))

    # AR / DNR summary — near apex
    risk_color = {"Bajo": (0.25, 0.85, 0.25),
                  "Moderado": (1.0, 0.75, 0.1),
                  "Alto": (0.95, 0.20, 0.15)}.get(result.rupture_risk_label,
                                                   (0.9, 0.9, 0.9))
    label_actors.append(_label(
        apex_world + axis * 2.0,
        (f"AR: {result.aspect_ratio:.2f}  DNR: {result.dome_to_neck_ratio:.2f}\n"
         f"Riesgo: {result.rupture_risk_label}"),
        risk_color,
    ))

    return MorphoOverlayActors(
        neck_disc    = neck_disc,
        neck_outline = neck_outline,
        dome_line    = dome_line,
        maxd_line    = maxd_line,
        labels       = label_actors,
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Internal helpers                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _mesh_points(poly_data: vtk.vtkPolyData) -> np.ndarray:
    from vtk.util.numpy_support import vtk_to_numpy
    return vtk_to_numpy(poly_data.GetPoints().GetData()).astype(float)


def _perp_to(v: np.ndarray) -> np.ndarray:
    """Return a unit vector perpendicular to *v*."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p = np.cross(v, helper)
    n = np.linalg.norm(p)
    return p / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])


def _max_diameter_endpoints(
    pts: np.ndarray,
    centroid: np.ndarray,
    axis: np.ndarray,
    max_diam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate max-diameter line using PCA of residual coordinates."""
    proj_along  = ((pts - centroid) @ axis)[:, None] * axis
    residual    = pts - centroid - proj_along    # perpendicular component
    # PCA of perpendicular coordinates
    cov = np.cov(residual.T)
    try:
        _, vecs = np.linalg.eigh(cov)
        lat_axis = vecs[:, -1]                   # largest lateral variance
    except np.linalg.LinAlgError:
        lat_axis = _perp_to(axis)
    lat_axis /= np.linalg.norm(lat_axis) + 1e-12
    # Find extremes along lat_axis
    lat_proj = (pts - centroid) @ lat_axis
    pt1 = pts[np.argmin(lat_proj)]
    pt2 = pts[np.argmax(lat_proj)]
    return pt1, pt2


def _build_disc(
    centre: np.ndarray,
    normal: np.ndarray,
    radius: float,
    resolution: int = 64,
) -> vtk.vtkActor:
    """Translucent filled circle (disc) at the neck plane."""
    src = vtk.vtkDiskSource()
    src.SetInnerRadius(0.0)
    src.SetOuterRadius(radius)
    src.SetRadialResolution(1)
    src.SetCircumferentialResolution(resolution)
    src.Update()

    # Orient the disc to align with normal
    tf = vtk.vtkTransformPolyDataFilter()
    t  = vtk.vtkTransform()
    t.PostMultiply()
    # DiskSource lies in XY plane; rotate to *normal*
    default_n = np.array([0.0, 0.0, 1.0])
    rot_axis  = np.cross(default_n, normal)
    rot_len   = np.linalg.norm(rot_axis)
    if rot_len > 1e-6:
        rot_axis /= rot_len
        angle_deg = math.degrees(math.acos(
            float(np.clip(np.dot(default_n, normal), -1, 1))
        ))
        t.RotateWXYZ(angle_deg, *rot_axis)
    t.Translate(*centre)
    tf.SetTransform(t)
    tf.SetInputConnection(src.GetOutputPort())
    tf.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tf.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*_NECK_COLOR)
    actor.GetProperty().SetOpacity(_DISC_OPACITY)
    actor.GetProperty().SetLighting(False)
    return actor


def _build_circle(
    centre: np.ndarray,
    normal: np.ndarray,
    radius: float,
    color: tuple,
    resolution: int = 64,
) -> vtk.vtkActor:
    """Solid circle outline (polyline) at the neck plane."""
    perp = _perp_to(normal)
    bino = np.cross(normal, perp)

    angles = np.linspace(0, 2 * math.pi, resolution, endpoint=False)
    ring   = np.array([
        centre + radius * math.cos(a) * perp + radius * math.sin(a) * bino
        for a in angles
    ], dtype=float)

    vtk_pts = vtk.vtkPoints()
    for p in ring:
        vtk_pts.InsertNextPoint(*p)
    vtk_pts.InsertNextPoint(*ring[0])   # close loop

    lines = vtk.vtkCellArray()
    lines.InsertNextCell(resolution + 1)
    for i in range(resolution + 1):
        lines.InsertCellPoint(i % resolution)

    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_pts)
    pd.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(_LINE_WIDTH + 1)
    actor.GetProperty().SetOpacity(0.95)
    actor.GetProperty().SetLighting(False)
    return actor


def _build_line(
    pt_a: np.ndarray,
    pt_b: np.ndarray,
    color: tuple,
    dashed: bool = False,
) -> vtk.vtkActor:
    """A simple line actor between two points."""
    vtk_pts = vtk.vtkPoints()
    vtk_pts.InsertNextPoint(*pt_a)
    vtk_pts.InsertNextPoint(*pt_b)

    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, 0)
    line.GetPointIds().SetId(1, 1)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)

    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_pts)
    pd.SetLines(cells)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(_LINE_WIDTH)
    actor.GetProperty().SetOpacity(0.90)
    actor.GetProperty().SetLighting(False)
    if dashed:
        actor.GetProperty().SetLineStipplePattern(0xF0F0)
    return actor


def _label(
    pos: np.ndarray,
    text: str,
    color: tuple,
) -> vtk.vtkBillboardTextActor3D:
    actor = vtk.vtkBillboardTextActor3D()
    actor.SetInput(text)
    actor.SetPosition(*pos)
    tp = actor.GetTextProperty()
    tp.SetFontSize(_LABEL_SIZE)
    tp.SetColor(*color)
    tp.SetBackgroundColor(0.06, 0.06, 0.06)
    tp.SetBackgroundOpacity(0.60)
    tp.SetBold(True)
    tp.ShadowOn()
    return actor
