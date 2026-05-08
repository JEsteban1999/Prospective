"""VTK geometry factory for intracranial stent / flow-diverter actors.

Produces a braided-wire representation of a stent aligned along the Z axis,
centred at the origin.  The caller is responsible for applying the world-space
transform via ``actor.SetUserTransform()``.

Coordinate convention (stent local frame)
------------------------------------------
  +Z  →  proximal → distal axis (blood-flow direction)
  X/Y →  cross-section plane

Appearance
----------
  Flow diverter  — dense braid (many thin wires, low porosity),  blue-silver
  Intracranial   — sparse open-cell design (fewer struts),        silver-grey
"""
from __future__ import annotations

import math

import vtk

from prospective.models.stent_library import StentSpec, StentType


def build_stent_actor(spec: StentSpec) -> vtk.vtkActor:
    """Return a *vtkActor* representing *spec* in its local frame."""

    R        = spec.diameter_mm / 2.0
    L        = spec.length_mm
    n_family = spec.n_wires // 2          # wires per helix family
    wire_r   = max(0.04, spec.wire_diameter_um / 2_000.0)  # µm → mm radius

    # Number of helix turns along the stent depends on stent type
    if spec.stent_type == StentType.FLOW_DIVERTER:
        n_turns   = max(5, int(L / (spec.diameter_mm * 0.7)))
    else:
        n_turns   = max(2, int(L / (spec.diameter_mm * 1.8)))

    n_pts = 64   # points per helix polyline

    append = vtk.vtkAppendPolyData()

    # Two families: clockwise (sign=+1) and counter-clockwise (sign=-1)
    for sign in (+1, -1):
        for i in range(n_family):
            phase = 2.0 * math.pi * i / n_family

            pts = vtk.vtkPoints()
            for j in range(n_pts + 1):
                t     = j / n_pts
                z     = -L / 2.0 + t * L
                theta = sign * (phase + n_turns * 2.0 * math.pi * t)
                pts.InsertNextPoint(
                    R * math.cos(theta),
                    R * math.sin(theta),
                    z,
                )

            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(n_pts + 1)
            for j in range(n_pts + 1):
                polyline.GetPointIds().SetId(j, j)

            cells = vtk.vtkCellArray()
            cells.InsertNextCell(polyline)

            helix_pd = vtk.vtkPolyData()
            helix_pd.SetPoints(pts)
            helix_pd.SetLines(cells)

            tube = vtk.vtkTubeFilter()
            tube.SetInputData(helix_pd)
            tube.SetRadius(wire_r)
            tube.SetNumberOfSides(6)
            tube.CappingOn()
            tube.Update()

            append.AddInputData(tube.GetOutput())

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
    prop  = actor.GetProperty()

    if spec.stent_type == StentType.FLOW_DIVERTER:
        prop.SetColor(0.55, 0.72, 0.95)   # blue-silver (cobalt-chromium / platinum)
        prop.SetOpacity(0.90)
    else:
        prop.SetColor(0.82, 0.84, 0.90)   # silver-grey (nitinol / platinum)
        prop.SetOpacity(0.92)

    prop.SetAmbient(0.25)
    prop.SetDiffuse(0.60)
    prop.SetSpecular(0.95)
    prop.SetSpecularPower(90.0)

    return actor
