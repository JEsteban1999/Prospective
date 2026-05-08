"""VTK geometry factory for embolization coil actors.

Each coil is visualised as a platinum-coloured helical tube that fills
the aneurysm sac.  Two representations are available:

* ``build_coil_actor(spec)``  — isolated coil in its natural shape
  (helix or 3-D random shape), useful for the catalogue browser.
* ``build_coil_in_sac(spec, centroid, radius_mm)`` — coil packed inside
  a sphere approximating the aneurysm dome; multiple packed coils
  together give a realistic post-embolization visualisation.

Coordinate convention
---------------------
Coils are built at the origin in local frame; the caller applies the
world-space vtkTransform.
"""
from __future__ import annotations

import math
import random

import vtk

from prospective.models.coil_library import CoilSpec, CoilType


# ──────────────────────────────────────────────────────────────────────────── #
# Public API                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

def build_coil_actor(
    spec: CoilSpec,
    coil_index: int = 0,
) -> vtk.vtkActor:
    """
    Return a vtkActor representing *spec* in its natural helical shape.

    *coil_index* seeds the random variation so stacked coils differ
    slightly in rotation and radius for a realistic look.
    """
    rng = random.Random(coil_index * 1337 + 42)

    coil_r   = spec.diameter_mm / 2.0
    wire_r   = (spec.wire_diameter_um * 1e-3) / 2.0
    length_mm = spec.length_cm * 10.0

    if spec.coil_type in (CoilType.COMPLEX_3D, CoilType.FRAMING):
        pts = _complex_3d_path(coil_r, length_mm, rng)
    elif spec.coil_type == CoilType.HYDROCOIL:
        # HydroCoil: denser helix to suggest expanded coating
        pts = _helix_path(coil_r * 0.9, length_mm, turns=int(length_mm / 3.5))
    else:
        pts = _helix_path(coil_r, length_mm, turns=int(length_mm / (math.pi * 2)))

    return _tube_actor_from_pts(pts, wire_r, _platinum_color())


def build_coil_in_sac(
    spec: CoilSpec,
    centroid: tuple[float, float, float],
    sac_radius_mm: float,
    coil_index: int = 0,
) -> vtk.vtkActor:
    """
    Return a vtkActor showing *spec* packed inside a spherical sac.

    The coil is generated as a constrained random walk confined within
    a sphere of *sac_radius_mm* centred on *centroid*.  Successive
    coils (different *coil_index* values) will fill the sac
    progressively.
    """
    rng = random.Random(coil_index * 2053 + 17)

    wire_r    = (spec.wire_diameter_um * 1e-3) / 2.0
    length_mm = spec.length_cm * 10.0
    pts       = _constrained_walk(centroid, sac_radius_mm, length_mm, rng)

    return _tube_actor_from_pts(pts, wire_r, _platinum_color())


# ──────────────────────────────────────────────────────────────────────────── #
# Path generators                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _helix_path(
    radius: float,
    length_mm: float,
    turns: int = 8,
) -> vtk.vtkPoints:
    """Simple helix: radius *radius*, *turns* complete rotations."""
    n_pts = max(turns * 32, 64)
    pts   = vtk.vtkPoints()
    pts.SetNumberOfPoints(n_pts)
    pitch = length_mm / max(turns, 1)
    for i in range(n_pts):
        t     = i / (n_pts - 1)
        angle = t * turns * 2.0 * math.pi
        x     = radius * math.cos(angle)
        y     = radius * math.sin(angle)
        z     = t * length_mm - length_mm / 2.0
        pts.SetPoint(i, x, y, z)
    return pts


def _complex_3d_path(
    radius: float,
    length_mm: float,
    rng: random.Random,
) -> vtk.vtkPoints:
    """Pseudo-random 3-D complex coil path (Lissajous-like with noise)."""
    n_pts = 256
    pts   = vtk.vtkPoints()
    pts.SetNumberOfPoints(n_pts)

    # Phase offsets give a non-repeating 3-D Lissajous figure
    px = rng.uniform(0, 2 * math.pi)
    py = rng.uniform(0, 2 * math.pi)
    pz = rng.uniform(0, 2 * math.pi)
    fx, fy, fz = 3, 4, 5   # frequency ratios

    noise_amp = radius * 0.12

    for i in range(n_pts):
        t = i / (n_pts - 1) * 2 * math.pi
        x = radius * math.cos(fx * t + px) + rng.gauss(0, noise_amp)
        y = radius * math.sin(fy * t + py) + rng.gauss(0, noise_amp)
        z = radius * math.cos(fz * t + pz) + rng.gauss(0, noise_amp)
        pts.SetPoint(i, x, y, z)

    return pts


def _constrained_walk(
    centroid: tuple[float, float, float],
    sac_r: float,
    length_mm: float,
    rng: random.Random,
) -> vtk.vtkPoints:
    """
    Random walk of total arc-length ≈ *length_mm* confined within a sphere
    of radius *sac_r* centred on *centroid*.
    """
    cx, cy, cz = centroid
    step   = 0.8   # mm per step
    n_pts  = max(int(length_mm / step), 32)
    pts    = vtk.vtkPoints()
    pts.SetNumberOfPoints(n_pts)

    # Start at a random position inside the sac (≤ 0.6 R from centre)
    start_r = sac_r * 0.6 * rng.random()
    theta   = rng.uniform(0, math.pi)
    phi     = rng.uniform(0, 2 * math.pi)
    x = cx + start_r * math.sin(theta) * math.cos(phi)
    y = cy + start_r * math.sin(theta) * math.sin(phi)
    z = cz + start_r * math.cos(theta)

    dx, dy, dz = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
    norm = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
    dx, dy, dz = dx/norm, dy/norm, dz/norm

    for i in range(n_pts):
        pts.SetPoint(i, x, y, z)

        # Random walk with drift toward centre when near boundary
        dist_sq = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
        if dist_sq > (sac_r * 0.85)**2:
            # Push back toward centre
            d  = math.sqrt(dist_sq)
            dx = (cx - x) / d * 0.7 + rng.gauss(0, 0.3)
            dy = (cy - y) / d * 0.7 + rng.gauss(0, 0.3)
            dz = (cz - z) / d * 0.7 + rng.gauss(0, 0.3)
        else:
            dx += rng.gauss(0, 0.4)
            dy += rng.gauss(0, 0.4)
            dz += rng.gauss(0, 0.4)

        norm = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        dx, dy, dz = dx/norm, dy/norm, dz/norm
        x += dx * step
        y += dy * step
        z += dz * step

    return pts


# ──────────────────────────────────────────────────────────────────────────── #
# VTK assembly                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

def _tube_actor_from_pts(
    pts: vtk.vtkPoints,
    tube_radius: float,
    color: tuple[float, float, float],
) -> vtk.vtkActor:
    """Build a vtkActor from a point path using vtkTubeFilter."""
    n = pts.GetNumberOfPoints()

    # Build polyline
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(n)
    for i in range(n):
        line.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(cells)

    # Smooth with spline to reduce jaggedness
    spline = vtk.vtkSplineFilter()
    spline.SetInputData(poly)
    spline.SetSubdivideToLength()
    spline.SetLength(max(tube_radius * 2.0, 0.05))
    spline.Update()

    tube = vtk.vtkTubeFilter()
    tube.SetInputConnection(spline.GetOutputPort())
    tube.SetRadius(max(tube_radius, 0.05))
    tube.SetNumberOfSides(8)
    tube.CappingOn()
    tube.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetMetallic(0.95)
    prop.SetRoughness(0.20)
    prop.SetAmbient(0.25)
    prop.SetDiffuse(0.60)
    prop.SetSpecular(0.90)
    prop.SetSpecularPower(80.0)
    return actor


def _platinum_color() -> tuple[float, float, float]:
    """Platinum white — standard coil material appearance."""
    return (0.90, 0.89, 0.88)
