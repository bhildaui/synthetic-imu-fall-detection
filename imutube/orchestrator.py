from pathlib import Path
from imutube.config import *
from imutube.stages import *
from imutube.vendor.library import ensure_openpose_on_path, ensure_vp3d_on_path


def run(frames_dir: Path, work_dir: Path, cfg: PipelineConfig):
    work_dir.mkdir(exist_ok=True)

    ensure_openpose_on_path(work_dir)
    ensure_vp3d_on_path(work_dir)

    json2d = work_dir / "json" / "2d"
    npz2d = work_dir / "2d.npz"
    npy3d = work_dir / "json" / "3d"/ "3d.json"
    npy3d_local = work_dir / "json" / "3d"/ "3d_local.json"
    video = work_dir / "final.mp4"

    res = extract_2d(frames_dir, json2d, cfg.openpose)
    convert_openpose_json_to_vp3d_npz(
        json2d, npz2d, cfg.subject, cfg.action, res.width, res.height
    )
    infer_3d(npz2d, npy3d, cfg.vp3d)
    render(frames_dir, json2d, npy3d, npy3d_local, video, cfg.render)