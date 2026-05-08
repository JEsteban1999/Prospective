"""Mesh region-of-interest (ROI) cropping utilities — A-02-05b.

Two non-destructive operations are provided; neither modifies the input mesh:

  clip_box(poly, xmin, xmax, ymin, ymax, zmin, zmax)
      Keep geometry inside an axis-aligned bounding box.
      Uses six chained vtkClipPolyData filters (one plane per face).

  clip_sphere(poly, center, radius)
      Keep geometry inside a sphere defined by (center, radius).
      Uses a single vtkClipPolyData with a vtkSphere implicit function.

Plane math (vtkClipPolyData with InsideOutOff — the default):
  The filter keeps cells where f(p) <= 0.
  For vtkPlane: f(p) = normal · (p − origin).

  To keep x >= xmin → normal=(-1,0,0), origin=(xmin,0,0)
    f = -(x-xmin) <= 0  when x >= xmin  ✓
  To keep x <= xmax → normal=(+1,0,0), origin=(xmax,0,0)
    f =  (x-xmax) <= 0  when x <= xmax  ✓
  (same pattern for Y and Z)

Sphere math:
  vtkSphere: f(p) = |p - center|² − radius²
  InsideOutOff keeps f <= 0, i.e., inside the sphere  ✓
"""
from __future__ import annotations

import logging
import math

import vtk

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────── #

def clip_box(
    poly: vtk.vtkPolyData,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    zmin: float, zmax: float,
) -> vtk.vtkPolyData:
    """Return the portion of *poly* that lies inside the axis-aligned box.

    Parameters
    ----------
    poly:
        Input surface mesh (not modified).
    xmin, xmax, ymin, ymax, zmin, zmax:
        Box bounds in the same coordinate space as the mesh (usually mm).

    Returns
    -------
    vtkPolyData with orphan points removed by vtkCleanPolyData.
    Returns an empty vtkPolyData if the box excludes all geometry.
    """
    if poly.GetNumberOfPoints() == 0:
        return poly

    # Six planes: (normal, origin).  InsideOutOff keeps f(p) = N·(p-O) <= 0.
    _planes = [
        ((-1,  0,  0), (xmin,  0,    0   )),  # keep x >= xmin
        (( 1,  0,  0), (xmax,  0,    0   )),  # keep x <= xmax
        (( 0, -1,  0), (0,     ymin,  0  )),  # keep y >= ymin
        (( 0,  1,  0), (0,     ymax,  0  )),  # keep y <= ymax
        (( 0,  0, -1), (0,     0,    zmin)),  # keep z >= zmin
        (( 0,  0,  1), (0,     0,    zmax)),  # keep z <= zmax
    ]

    data: vtk.vtkPolyData = poly
    for normal, origin in _planes:
        if data.GetNumberOfPoints() == 0:
            logger.debug("clip_box: mesh became empty after a plane clip — stopping early")
            break
        plane = vtk.vtkPlane()
        plane.SetNormal(*normal)
        plane.SetOrigin(*origin)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(data)
        clipper.SetClipFunction(plane)
        clipper.InsideOutOff()   # keep where f(p) <= 0
        clipper.Update()
        data = clipper.GetOutput()

    # Remove orphan points produced by intermediate clips
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(data)
    clean.Update()
    result = clean.GetOutput()

    logger.info(
        "clip_box: %d → %d vertices  (bounds [%.1f,%.1f] [%.1f,%.1f] [%.1f,%.1f])",
        poly.GetNumberOfPoints(), result.GetNumberOfPoints(),
        xmin, xmax, ymin, ymax, zmin, zmax,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────── #

def clip_sphere(
    poly: vtk.vtkPolyData,
    center: tuple[float, float, float],
    radius: float,
) -> vtk.vtkPolyData:
    """Return the portion of *poly* that lies inside the sphere.

    Parameters
    ----------
    poly:
        Input surface mesh (not modified).
    center:
        Sphere centre (cx, cy, cz) in the same coordinate space as the mesh.
    radius:
        Sphere radius in mm.

    Returns
    -------
    vtkPolyData with orphan points removed by vtkCleanPolyData.
    Returns an empty vtkPolyData if the sphere excludes all geometry.
    """
    if poly.GetNumberOfPoints() == 0:
        return poly
    if radius <= 0:
        logger.warning("clip_sphere: radius <= 0, returning empty mesh")
        return vtk.vtkPolyData()

    sphere = vtk.vtkSphere()
    sphere.SetCenter(*center)
    sphere.SetRadius(radius)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(poly)
    clipper.SetClipFunction(sphere)
    clipper.InsideOutOff()   # vtkSphere f(p) = |p-c|²-r²; f<=0 inside sphere ✓
    clipper.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(clipper.GetOutputPort())
    clean.Update()
    result = clean.GetOutput()

    logger.info(
        "clip_sphere: %d → %d vertices  (center=(%.1f,%.1f,%.1f)  r=%.1f mm)",
        poly.GetNumberOfPoints(), result.GetNumberOfPoints(),
        center[0], center[1], center[2], radius,
    )
    return result
