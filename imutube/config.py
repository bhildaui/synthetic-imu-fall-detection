from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class OpenPoseConfig:
    model_folder: Path
    model_pose: str = "COCO"
    net_resolution: str = "-320x176"
    render: bool = False

@dataclass(frozen=True)
class VP3DConfig:
    checkpoint: Path
    checkpoint_dir: Path
    dataset_key: str = "custom"
    subject: str = "S1"
    action: str = "custom"

@dataclass(frozen=True)
class RenderConfig:
    fps: int = 30
    show_2d: bool = True
    show_3d: bool = True

@dataclass(frozen=True)
class PipelineConfig:
    openpose: OpenPoseConfig
    vp3d: VP3DConfig
    render: RenderConfig
    subject: str = "S1"
    action: str = "custom"