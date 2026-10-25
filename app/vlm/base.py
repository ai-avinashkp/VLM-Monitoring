from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass
class Generation:
    text: str
    latency_s: float
    new_tokens: int


class VLM(Protocol):
    name: str
    image_side: int

    def generate(self, image: Image.Image | list[Image.Image], system: str, prompt: str,
                 max_new_tokens: int | None = None, prefill: str = "",
                 on_text: Callable[[str], bool | None] | None = None) -> Generation: ...

    def vram(self) -> dict | None: ...

    def close(self) -> None: ...


def fit_image(image: Image.Image, side: int) -> Image.Image:
    """Downscale so the longest side is at most `side` px; this bounds the visual token count."""
    w, h = image.size
    scale = side / max(w, h)
    if scale >= 1:
        return image
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
