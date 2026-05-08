"""Mesh export utilities — STL and OBJ via VTK writers.

A-02-06: Exportación de malla 3D a STL / OBJ desde segmentación
"""
from __future__ import annotations

import logging
from pathlib import Path

import vtk

logger = logging.getLogger(__name__)


class MeshExporter:
    """Export a vtkPolyData mesh to STL or OBJ."""

    @staticmethod
    def export_stl(poly_data: vtk.vtkPolyData, path: str, binary: bool = True) -> None:
        """
        Write *poly_data* to *path* as an STL file.

        Parameters
        ----------
        poly_data: vtkPolyData to export.
        path:      Destination file path (will be created/overwritten).
        binary:    True → binary STL (smaller); False → ASCII STL.
        """
        path = str(Path(path).with_suffix(".stl"))
        writer = vtk.vtkSTLWriter()
        writer.SetInputData(poly_data)
        writer.SetFileName(path)
        if binary:
            writer.SetFileTypeToBinary()
        else:
            writer.SetFileTypeToASCII()
        writer.Write()
        n = poly_data.GetNumberOfPolys()
        size_kb = Path(path).stat().st_size / 1024
        logger.info("Exported STL: %s  (%d triangles, %.1f KB)", path, n, size_kb)

    @staticmethod
    def export_obj(poly_data: vtk.vtkPolyData, path: str) -> None:
        """
        Write *poly_data* to *path* as a Wavefront OBJ file.

        A companion .mtl file is created automatically by the VTK writer.
        """
        path = str(Path(path).with_suffix(".obj"))
        writer = vtk.vtkOBJWriter()
        writer.SetInputData(poly_data)
        writer.SetFileName(path)
        writer.Write()
        n = poly_data.GetNumberOfPolys()
        logger.info("Exported OBJ: %s  (%d triangles)", path, n)

    @staticmethod
    def save_vtp(poly_data: vtk.vtkPolyData, path: str) -> None:
        """Write *poly_data* to *path* as a binary VTK XML PolyData (.vtp) file.

        This is the preferred format for session persistence: it is lossless,
        compact (binary), and round-trips perfectly through load_vtp().
        """
        path = str(Path(path).with_suffix(".vtp"))
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(poly_data)
        writer.SetFileName(path)
        writer.SetDataModeToBinary()
        writer.Write()
        n = poly_data.GetNumberOfPolys()
        size_kb = Path(path).stat().st_size / 1024
        logger.info("Saved VTP mesh: %s  (%d triangles, %.1f KB)", path, n, size_kb)

    @staticmethod
    def load_vtp(path: str) -> "vtk.vtkPolyData | None":
        """Read a VTK XML PolyData (.vtp) file and return the vtkPolyData.

        Returns None (and logs a warning) if the file cannot be read.
        """
        p = Path(path)
        if not p.exists():
            logger.warning("VTP file not found: %s", path)
            return None
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(p))
        reader.Update()
        pd = reader.GetOutput()
        if pd is None or pd.GetNumberOfPoints() == 0:
            logger.warning("VTP file empty or unreadable: %s", path)
            return None
        logger.info(
            "Loaded VTP mesh: %s  (%d verts, %d tris)",
            path, pd.GetNumberOfPoints(), pd.GetNumberOfPolys(),
        )
        return pd

    @staticmethod
    def mesh_stats(poly_data: vtk.vtkPolyData) -> dict:
        """Return a dict with basic mesh statistics."""
        return {
            "vertices":  poly_data.GetNumberOfPoints(),
            "triangles": poly_data.GetNumberOfPolys(),
            "edges":     poly_data.GetNumberOfLines(),
        }
