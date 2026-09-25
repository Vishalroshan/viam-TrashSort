"""Where generated files go.

Scans, marked-up frames, point clouds and meshes are all regenerable output,
so they land in one gitignored directory instead of scattering across the
repo root. Paths are anchored to the repo, not the current directory, so a
run from anywhere writes to the same place.

Set TRASHSORT_OUTPUT_DIR to redirect; every CLI --out flag still wins over
this default.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = Path(os.environ.get("TRASHSORT_OUTPUT_DIR",
                                REPO_ROOT / "output"))

ASSETS_DIR = REPO_ROOT / "assets"

DOTENV = REPO_ROOT / ".env"


def out_path(name) -> str:
    """Absolute path for a generated file named `name`, creating the dir.

    Returns str, not Path: callers pass it straight to open() and to Viam and
    Open3D APIs, some of which do not accept path-like objects.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(OUTPUT_DIR / name)
