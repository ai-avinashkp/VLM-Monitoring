from ..config import ModelConfig
from .base import VLM, Generation, fit_image


def build_vlm(cfg: ModelConfig) -> VLM:
    if cfg.id == "mock":
        from .mock import MockVLM
        return MockVLM(image_side=cfg.image_side)
    from .hf_vlm import HFVLM
    return HFVLM(cfg.id, dtype=cfg.dtype, quantize_4bit=cfg.quantize_4bit,
                 image_side=cfg.image_side, max_new_tokens=cfg.max_new_tokens,
                 static_cache=cfg.static_cache, max_cache_len=cfg.max_cache_len)


__all__ = ["VLM", "Generation", "build_vlm", "fit_image"]
