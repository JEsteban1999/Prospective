"""Clip–vessel collision detection — F-04 / MOD-05.

Uses vtkCollisionDetectionFilter (OBB-tree based) to check whether a
placed clip intersects the segmented vascular mesh.
"""
from __future__ import annotations

import logging

import vtk

logger = logging.getLogger(__name__)


def check_collision(
    vessel_mesh: vtk.vtkPolyData,
    clip_poly: vtk.vtkPolyData,
    clip_transform: vtk.vtkTransform,
) -> tuple[bool, int]:
    """
    Return (collision_detected, n_contact_cells).

    Parameters
    ----------
    vessel_mesh:    vtkPolyData of the full segmented vasculature.
    clip_poly:      vtkPolyData of the clip in its local frame.
    clip_transform: world-space transform already applied to the clip actor.
    """
    if vessel_mesh is None or clip_poly is None:
        return False, 0

    # vtkCollisionDetectionFilter needs triangulated input
    tri_vessel = vtk.vtkTriangleFilter()
    tri_vessel.SetInputData(vessel_mesh)
    tri_vessel.Update()

    tri_clip = vtk.vtkTriangleFilter()
    tri_clip.SetInputData(clip_poly)
    tri_clip.Update()

    identity = vtk.vtkTransform()
    identity.Identity()

    col = vtk.vtkCollisionDetectionFilter()
    col.SetInputData(0, tri_vessel.GetOutput())
    col.SetInputData(1, tri_clip.GetOutput())
    col.SetTransform(0, identity)
    col.SetTransform(1, clip_transform)
    col.SetCollisionModeToFirstContact()   # fast — stops at first hit
    col.GenerateScalarsOn()
    col.Update()

    n = col.GetNumberOfContacts()
    detected = n > 0
    logger.debug("Collision check: %d contacts", n)
    return detected, n
