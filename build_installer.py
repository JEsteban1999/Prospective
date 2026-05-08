"""
build_installer.py — PROSPECTIVE Windows installer builder
===========================================================

Run from the project root:
    python build_installer.py

What it does
------------
1. Checks that all required dependencies are installed.
2. Runs PyInstaller with prospective.spec.
3. Prints the output folder path on success.

The result is a stand-alone folder:
    dist/PROSPECTIVE/
        PROSPECTIVE.exe     ← launcher
        _internal/          ← all Python + VTK + Qt libs
        resources/          ← intro video
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# ── Dependency pre-check ──────────────────────────────────────────────────── #

REQUIRED = [
    ("PyInstaller",   "pyinstaller"),
    ("PyQt5",         "PyQt5"),
    ("vtk",           "vtk"),
    ("pydicom",       "pydicom"),
    ("reportlab",     "reportlab"),
    ("SimpleITK",     "SimpleITK"),
    ("imageio",       "imageio"),
    ("imageio_ffmpeg","imageio-ffmpeg"),
    ("sqlalchemy",    "SQLAlchemy"),
    ("numpy",         "numpy"),
    ("scipy",         "scipy"),
    ("trimesh",       "trimesh"),
]

missing = []
for module, pip_name in REQUIRED:
    try:
        __import__(module)
    except ImportError:
        missing.append(pip_name)

if missing:
    print("❌  Missing dependencies — run:")
    print(f"    pip install {' '.join(missing)}")
    sys.exit(1)

print("✓  All dependencies present.\n")

# ── Clean previous build ──────────────────────────────────────────────────── #

for folder in ("build", "dist/PROSPECTIVE"):
    p = ROOT / folder
    if p.exists():
        shutil.rmtree(p)
        print(f"  Cleaned: {p}")

# ── Run PyInstaller ───────────────────────────────────────────────────────── #

spec = ROOT / "prospective.spec"
print(f"\n▶  Running PyInstaller with {spec.name} …\n")

result = subprocess.run(
    [sys.executable, "-m", "PyInstaller", str(spec), "--clean", "--noconfirm"],
    cwd=ROOT,
)

if result.returncode != 0:
    print("\n❌  PyInstaller failed — see output above.")
    sys.exit(result.returncode)

# ── Report ────────────────────────────────────────────────────────────────── #

dist_dir = ROOT / "dist" / "PROSPECTIVE"
exe      = dist_dir / "PROSPECTIVE.exe"

print("\n" + "=" * 60)
if exe.exists():
    size_mb = sum(f.stat().st_size for f in dist_dir.rglob("*") if f.is_file()) / 1e6
    print(f"✅  Build successful!")
    print(f"    Folder : {dist_dir}")
    print(f"    Exe    : {exe}")
    print(f"    Size   : {size_mb:.0f} MB (total folder)")
else:
    print("⚠   Build completed but PROSPECTIVE.exe not found — check output above.")
print("=" * 60 + "\n")
