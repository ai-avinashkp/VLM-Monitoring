from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CameraConfig:
    index: int = 0
    backend: str = "dshow"
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class ModelConfig:
    id: str = "Qwen/Qwen3-VL-2B-Instruct"
    dtype: str = "bfloat16"
    quantize_4bit: bool = False
    image_side: int = 640
    max_new_tokens: int = 260
    static_cache: bool = True
    max_cache_len: int = 2048


@dataclass
class ReasonerConfig:
    heartbeat_s: float = 10.0
    min_gap_s: float = 2.0
    change_threshold: float = 22.0
    motion_threshold: float = 6.0
    settle_s: float = 0.6


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    stream_fps: int = 24
    jpeg_quality: int = 80
    boost_cpu: bool = True


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def _fill(cls, data: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path = ROOT / "config.yaml") -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    raw = raw or {}
    return Config(
        camera=_fill(CameraConfig, raw.get("camera", {})),
        model=_fill(ModelConfig, raw.get("model", {})),
        reasoner=_fill(ReasonerConfig, raw.get("reasoner", {})),
        server=_fill(ServerConfig, raw.get("server", {})),
    )
