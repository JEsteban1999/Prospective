"""Unit tests for prospective.processing.collision — check_collision().

Strategy: build simple VTK primitives (spheres, cubes) analytically
so we know the ground truth for collision / no-collision cases.
"""
from __future__ import annotations

import vtk
import pytest

from prospective.processing.collision import check_collision


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

def _sphere(center=(0.0, 0.0, 0.0), radius: float = 5.0) -> vtk.vtkPolyData:
    src = vtk.vtkSphereSource()
    src.SetCenter(*center)
    src.SetRadius(radius)
    src.SetThetaResolution(24)
    src.SetPhiResolution(24)
    src.Update()
    return src.GetOutput()


def _cube(center=(0.0, 0.0, 0.0), length: float = 4.0) -> vtk.vtkPolyData:
    src = vtk.vtkCubeSource()
    src.SetCenter(*center)
    src.SetXLength(length)
    src.SetYLength(length)
    src.SetZLength(length)
    src.Update()
    return src.GetOutput()


def _identity_transform() -> vtk.vtkTransform:
    t = vtk.vtkTransform()
    t.Identity()
    return t


def _translate_transform(dx: float, dy: float, dz: float) -> vtk.vtkTransform:
    t = vtk.vtkTransform()
    t.Translate(dx, dy, dz)
    return t


# ──────────────────────────────────────────────────────────────────────────── #
# Guard tests — None inputs                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

class TestCheckCollisionNullGuard:

    def test_none_vessel_returns_no_collision(self):
        detected, n = check_collision(None, _sphere(), _identity_transform())
        assert detected is False
        assert n == 0

    def test_none_clip_returns_no_collision(self):
        detected, n = check_collision(_sphere(), None, _identity_transform())
        assert detected is False
        assert n == 0

    def test_both_none_returns_no_collision(self):
        detected, n = check_collision(None, None, _identity_transform())
        assert detected is False
        assert n == 0


# ──────────────────────────────────────────────────────────────────────────── #
# Geometry tests                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

class TestCheckCollisionGeometry:

    def test_overlapping_spheres_collide(self):
        """Two spheres whose surfaces intersect must trigger a collision.

        Vessel at origin r=5, clip at origin r=5 translated 4 mm right.
        Center distance (4) < r1+r2 (10) and > |r1-r2| (0)  → surfaces cross.
        """
        vessel = _sphere(center=(0, 0, 0), radius=5.0)
        clip   = _sphere(center=(0, 0, 0), radius=5.0)
        tx = _translate_transform(4.0, 0.0, 0.0)
        detected, n = check_collision(vessel, clip, tx)
        assert detected is True
        assert n > 0

    def test_nested_spheres_no_surface_contact(self):
        """A small sphere inside a larger one has no surface intersection."""
        vessel = _sphere(center=(0, 0, 0), radius=8.0)
        clip   = _sphere(center=(0, 0, 0), radius=3.0)
        detected, n = check_collision(vessel, clip, _identity_transform())
        # Surfaces never cross — OBB-tree collision returns False
        assert detected is False
        assert n == 0

    def test_far_apart_meshes_no_collision(self):
        """Vessel at origin, clip moved far away — no collision."""
        vessel = _sphere(center=(0, 0, 0), radius=5.0)
        clip   = _sphere(center=(0, 0, 0), radius=3.0)
        # Move clip 50 mm away — well beyond both radii
        tx = _translate_transform(50.0, 0.0, 0.0)
        detected, n = check_collision(vessel, clip, tx)
        assert detected is False
        assert n == 0

    def test_touching_but_not_penetrating(self):
        """Clip exactly touching the vessel surface may or may not register
        as a collision depending on OBB precision, but must not raise."""
        vessel = _sphere(center=(0, 0, 0), radius=5.0)
        clip   = _sphere(center=(0, 0, 0), radius=2.0)
        # Move clip so surfaces just touch (5 + 2 = 7 mm apart)
        tx = _translate_transform(7.0, 0.0, 0.0)
        detected, n = check_collision(vessel, clip, tx)
        # Only check types; exact result is filter-precision dependent
        assert isinstance(detected, bool)
        assert isinstance(n, int)

    def test_return_types_are_correct(self):
        vessel = _sphere()
        clip   = _cube()
        detected, n = check_collision(vessel, clip, _identity_transform())
        assert isinstance(detected, bool)
        assert isinstance(n, int)
        assert n >= 0
