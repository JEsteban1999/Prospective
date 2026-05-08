"""VTK actor factory for vessel centreline visualisation.

Creates a tube actor coloured by local vessel radius (radius colormap).
Endpoint sphere markers (source = green, target = red) are returned
separately for overlay in the planning window.
"""
from __future__ import annotations

import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk

from prospective.processing.centerline import CenterlineResult


# ──────────────────────────────────────────────────────────────────────────── #
# Colormap: cool-warm radius scale                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def _radius_lut(min_r: float, max_r: float) -> vtk.vtkLookupTable:
    """Blue (thin) → cyan → green → yellow → red (wide)."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(min_r, max_r)
    # Manual gradient: blue → cyan → green → yellow → red
    colors = [
        (0.0,  0.18, 0.25, 0.85),   # deep blue  (thin)
        (0.25, 0.10, 0.75, 0.90),   # cyan
        (0.50, 0.15, 0.85, 0.20),   # green
        (0.75, 0.95, 0.85, 0.10),   # yellow
        (1.0,  0.90, 0.15, 0.10),   # red        (wide)
    ]
    for i in range(256):
        t = i / 255.0
        # Find surrounding colour stops
        for idx in range(len(colors) - 1):
            t0, r0, g0, b0 = colors[idx]
            t1, r1, g1, b1 = colors[idx + 1]
            if t0 <= t <= t1:
                s = (t - t0) / (t1 - t0)
                r = r0 + s * (r1 - r0)
                g = g0 + s * (g1 - g0)
                b = b0 + s * (b1 - b0)
                lut.SetTableValue(i, r, g, b, 1.0)
                break
    lut.Build()
    return lut


# ──────────────────────────────────────────────────────────────────────────── #
# Public factories                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def build_centerline_actor(result: CenterlineResult) -> vtk.vtkActor:
    """
    Return a tube actor for the centreline coloured by vessel radius.

    The tube radius is fixed at 0.35 mm (purely visual) — not the vessel radius.
    Colour encodes the actual vessel radius via the lookup table.
    """
    if result.poly_data is None:
        raise ValueError("CenterlineResult.poly_data is None — call extract() first.")

    # Reparametrize with a spline so VTK computes smooth internal tangents;
    # this prevents the parallel-transport frame from accumulating twist at the
    # endpoints where the original voxel path has short, randomly-oriented segments.
    spline = vtk.vtkSplineFilter()
    spline.SetInputData(result.poly_data)
    spline.SetSubdivideToLength()
    spline.SetLength(0.4)          # resample at 0.4 mm — finer than the 0.5 mm path

    # Tube filter around the reparametrized polyline.
    # SetCapping(False): flat end-caps use the (potentially twisted) endpoint frame;
    # removing them eliminates the propeller artefact at source/target.
    tube = vtk.vtkTubeFilter()
    tube.SetInputConnection(spline.GetOutputPort())
    tube.SetRadius(0.35)           # visual tube radius (mm) — thinner than vessel
    tube.SetNumberOfSides(16)
    tube.SetCapping(False)

    lut = _radius_lut(result.min_radius_mm, result.max_radius_mm)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())   # lazy pipeline — no manual Update()
    mapper.SetScalarModeToUsePointData()
    mapper.SetColorModeToMapScalars()
    mapper.SelectColorArray("Radius")
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(result.min_radius_mm, result.max_radius_mm)
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.92)
    actor.GetProperty().SetAmbient(0.20)
    actor.GetProperty().SetDiffuse(0.70)
    actor.GetProperty().SetSpecular(0.30)
    actor.GetProperty().SetSpecularPower(20)
    return actor


def build_endpoint_actors(
    result: CenterlineResult,
) -> tuple[vtk.vtkActor, vtk.vtkActor]:
    """Return (source_actor, target_actor) — green/red spheres at endpoints."""
    source_pt = result.points[0]
    target_pt = result.points[-1]

    def _sphere(pt, color):
        s = vtk.vtkSphereSource()
        s.SetCenter(*pt)
        s.SetRadius(1.2)
        s.SetPhiResolution(16)
        s.SetThetaResolution(16)
        s.Update()
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(s.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(*color)
        a.GetProperty().SetOpacity(0.95)
        return a

    return (
        _sphere(source_pt, (0.10, 0.85, 0.20)),   # green — source
        _sphere(target_pt, (0.90, 0.15, 0.10)),   # red   — target
    )


def build_scalar_bar(result: CenterlineResult) -> vtk.vtkScalarBarActor:
    """Return a small colour legend for the radius colormap."""
    lut = _radius_lut(result.min_radius_mm, result.max_radius_mm)
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetTitle("Radio (mm)")
    bar.SetNumberOfLabels(4)
    bar.SetWidth(0.08)
    bar.SetHeight(0.30)
    bar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    bar.GetPositionCoordinate().SetValue(0.91, 0.05)
    bar.GetTitleTextProperty().SetFontSize(10)
    bar.GetLabelTextProperty().SetFontSize(8)
    bar.GetTitleTextProperty().SetColor(0.9, 0.9, 0.9)
    bar.GetLabelTextProperty().SetColor(0.9, 0.9, 0.9)
    return bar
