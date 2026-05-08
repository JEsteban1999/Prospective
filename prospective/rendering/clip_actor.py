"""VTK geometry factory for aneurysm clip actors.

Builds a simplified but recognisable 3-D representation of each
ClipShape.  The clip sits at the origin, jaws along the Y-axis,
spring body along the X-axis.  The caller is responsible for
applying the world-space transform.

Coordinate convention (clip local frame)
-----------------------------------------
  +X  →  spring direction (toward surgeon)
  +Y  →  jaw span axis  (blade opening plane)
  +Z  →  blade depth axis (perpendicular to jaw plane)

The clip is assembled from vtkPolyData primitives merged with
vtkAppendPolyData so a single vtkActor is sufficient.
"""
from __future__ import annotations

import math

import vtk

from prospective.models.clip_library import ClipShape, ClipSpec


def build_clip_actor(spec: ClipSpec) -> vtk.vtkActor:
    """
    Return a vtkActor representing *spec* in its local frame.

    The actor origin is at the hinge centre.
    Blade jaws open along ±Y; spring extends along -X.
    """
    append = vtk.vtkAppendPolyData()

    L  = spec.blade_length_mm
    W  = spec.blade_width_mm
    H  = spec.blade_height_mm
    SL = spec.spring_length_mm

    # ── Spring / hinge body ────────────────────────────────────────────── #
    spring = _box(-SL, 0.0, -W / 2, W / 2, -H / 2, H / 2)
    append.AddInputData(spring)

    # ── Upper blade ────────────────────────────────────────────────────── #
    if spec.shape == ClipShape.STRAIGHT:
        upper = _box(0.0, L, W * 0.1, W * 0.1 + H, -H / 2, H / 2)
        lower = _box(0.0, L, -W * 0.1 - H, -W * 0.1, -H / 2, H / 2)

    elif spec.shape == ClipShape.CURVED:
        upper = _curved_blade( 1.0, L, W * 0.1, H, n=20)
        lower = _curved_blade(-1.0, L, W * 0.1, H, n=20)

    elif spec.shape in (ClipShape.ANGLED, ClipShape.ANGLED_45):
        # Straight shaft + distal turn (90° or 45°)
        shaft_l   = L * 0.55
        tip_l     = L * 0.45
        angle_deg = 45.0 if spec.shape == ClipShape.ANGLED_45 else 90.0
        upper = _angled_blade( 1.0, shaft_l, tip_l, W * 0.1, H, angle_deg)
        lower = _angled_blade(-1.0, shaft_l, tip_l, W * 0.1, H, angle_deg)

    elif spec.shape == ClipShape.BAYONET:
        # Offset shaft to avoid visual field obstruction
        upper = _bayonet_blade( 1.0, L, W * 0.1, H, offset_z=2.5)
        lower = _bayonet_blade(-1.0, L, W * 0.1, H, offset_z=2.5)

    else:  # FENESTRATED — straight blades + central window
        upper = _box(0.0, L, W * 0.1, W * 0.1 + H, -H / 2, H / 2)
        lower = _box(0.0, L, -W * 0.1 - H, -W * 0.1, -H / 2, H / 2)
        # Fenestration ring on upper blade
        ring = _fenestration_ring(L * 0.35, L * 0.65, W * 0.1 + H / 2, H)
        append.AddInputData(ring)

    append.AddInputData(upper)
    append.AddInputData(lower)
    append.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(append.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(0.75, 0.80, 0.90)      # titanium-like silver-blue
    prop.SetMetallic(0.85)
    prop.SetRoughness(0.25)
    prop.SetAmbient(0.20)
    prop.SetDiffuse(0.70)
    prop.SetSpecular(0.80)
    prop.SetSpecularPower(60.0)
    return actor


# ──────────────────────────────────────────────────────────────────────────── #
# Geometry helpers                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def _box(x0, x1, y0, y1, z0, z1) -> vtk.vtkPolyData:
    src = vtk.vtkCubeSource()
    src.SetBounds(x0, x1, y0, y1, z0, z1)
    src.Update()
    return src.GetOutput()


def _curved_blade(
    sign: float, length: float, gap: float, height: float, n: int = 20
) -> vtk.vtkPolyData:
    """Arc-shaped blade sweeping from hinge toward tip."""
    radius = length / (math.pi / 3)   # ~60° arc
    angle_span = length / radius       # radians

    pts  = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    for i in range(n):
        t0 = i       / n * angle_span
        t1 = (i + 1) / n * angle_span
        for t in (t0, t1):
            x  =  radius * math.sin(t)
            y0 = sign * (gap + radius * (1 - math.cos(t)))
            y1 = y0 + sign * height
            pts.InsertNextPoint(x, y0, -height / 2)
            pts.InsertNextPoint(x, y0,  height / 2)
            pts.InsertNextPoint(x, y1, -height / 2)
            pts.InsertNextPoint(x, y1,  height / 2)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    # Convex hull approximation — just use a cube for the curved version
    # (full triangulation is complex; a box gives sufficient visual cue)
    del poly

    # Fall back: curved approximated as straight for geometry simplicity
    return _box(0.0, length, sign * gap, sign * (gap + height), -height / 2, height / 2)


def _angled_blade(
    sign: float, shaft_l: float, tip_l: float, gap: float, height: float,
    angle_deg: float = 90.0,
) -> vtk.vtkPolyData:
    """Straight shaft with a distal turn at *angle_deg* degrees (45° or 90°)."""
    import math
    append = vtk.vtkAppendPolyData()
    shaft = _box(0.0, shaft_l, sign * gap, sign * (gap + height), -height / 2, height / 2)
    append.AddInputData(shaft)
    # Tip goes along Z (90°) or diagonally (45°)
    rad   = math.radians(angle_deg)
    tip_x = tip_l * math.cos(rad)
    tip_z = tip_l * math.sin(rad) * sign
    tip   = _box(shaft_l - height, shaft_l + tip_x,
                 sign * gap, sign * (gap + height),
                 -height / 2, -height / 2 + abs(tip_z) + height)
    append.AddInputData(tip)
    append.Update()
    return append.GetOutput()


def _bayonet_blade(
    sign: float, length: float, gap: float, height: float, offset_z: float
) -> vtk.vtkPolyData:
    append = vtk.vtkAppendPolyData()
    seg1 = _box(0.0, length * 0.3,
                sign * gap, sign * (gap + height), -height / 2, height / 2)
    # Z-offset transition
    seg2 = _box(length * 0.3, length * 0.5,
                sign * gap, sign * (gap + height), -height / 2, offset_z)
    seg3 = _box(length * 0.5, length,
                sign * gap, sign * (gap + height), offset_z - height / 2, offset_z + height / 2)
    for s in (seg1, seg2, seg3):
        append.AddInputData(s)
    append.Update()
    return append.GetOutput()


def _fenestration_ring(x0, x1, y_center, height) -> vtk.vtkPolyData:
    """Simple rectangular ring (outer box minus inner void approximated by a thin frame)."""
    append = vtk.vtkAppendPolyData()
    thick = height * 0.25
    top   = _box(x0, x1, y_center + height / 2 - thick, y_center + height / 2, -height / 2, height / 2)
    bot   = _box(x0, x1, y_center - height / 2, y_center - height / 2 + thick, -height / 2, height / 2)
    left  = _box(x0, x0 + thick, y_center - height / 2, y_center + height / 2, -height / 2, height / 2)
    right = _box(x1 - thick, x1, y_center - height / 2, y_center + height / 2, -height / 2, height / 2)
    for p in (top, bot, left, right):
        append.AddInputData(p)
    append.Update()
    return append.GetOutput()
