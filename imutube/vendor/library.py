# imutube/vendor/library.py
from __future__ import annotations

import sys
from pathlib import Path


def ensure_vp3d_on_path(project_root: Path) -> Path:
    """
    Makes VideoPose3D/common importable as `import common.*` without modifying VP3D repo.
    """
    vp3d_root = project_root / "VideoPose3D"
    if not vp3d_root.exists():
        raise FileNotFoundError(f"VideoPose3D not found at: {vp3d_root}")

    vp3d_root_str = str(vp3d_root.resolve())
    if vp3d_root_str not in sys.path:
        sys.path.insert(0, vp3d_root_str)

    return vp3d_root


def ensure_openpose_on_path(project_root: Path) -> Path:
    """
    Makes OpenPose python bindings importable as `import pyopenpose`.
    Tries common build locations.
    """
    op_root = project_root / "openpose"
    if not op_root.exists():
        raise FileNotFoundError(f"openpose not found at: {op_root}")

    candidates = [
        op_root / "build" / "python",
        op_root / "build" / "python" / "openpose",
        op_root / "python",  # some builds
    ]

    added = False
    for p in candidates:
        if p.exists():
            p_str = str(p.resolve())
            if p_str not in sys.path:
                sys.path.insert(0, p_str)
            added = True

    if not added:
        raise FileNotFoundError(
            "OpenPose python bindings not found. Expected one of:\n"
            + "\n".join(str(p) for p in candidates)
        )

    return op_root