from pathlib import Path
from imutube.orchestrator import run
from imutube.config import *
from imutube.utils.paths import ROOT


def main():
    cfg = PipelineConfig(
        openpose=OpenPoseConfig(Path("openpose/models")),
        vp3d=VP3DConfig(
            checkpoint=Path("VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin"),
            checkpoint_dir=Path("VideoPose3D/checkpoint")
        ),
        render=RenderConfig()
    )
    run(Path("URFD/fall-07-cam0-rgb/fall-07-cam0-rgb/"), ROOT, cfg)

if __name__ == "__main__":
    main()
