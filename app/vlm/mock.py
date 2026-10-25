"""Fake VLM so the UI and reasoner can run without a GPU or model download."""
import json
import random
import time
from collections.abc import Callable

from PIL import Image

from .base import Generation

SCENES = [
    ("You're sitting at your desk looking at the screen, calm and focused.",
     [("Wearing black rectangular glasses", [380, 250, 620, 330]),
      ("Grey hoodie with the hood down", [250, 560, 760, 1000]),
      ("Room lit mostly by the monitor glow", None)]),
    ("You're at your desk with a mask on, glancing down at your phone.",
     [("Light-blue surgical mask covering nose and mouth", [400, 330, 600, 520]),
      ("Holding a phone low in your right hand", [600, 700, 760, 900]),
      ("Looking down rather than at the camera", [360, 180, 640, 520])]),
]


class MockVLM:
    name = "mock-vlm"

    def __init__(self, image_side: int = 640, latency_s: float = 1.2, **_):
        self.image_side = image_side
        self.latency_s = latency_s
        self._i = 0

    def generate(self, image: Image.Image | list[Image.Image], system: str, prompt: str,
                 max_new_tokens: int | None = None, prefill: str = "",
                 on_text: Callable[[str], bool | None] | None = None) -> Generation:
        if isinstance(image, list):  # before/after comparison
            time.sleep(self.latency_s * 0.3)
            return Generation("You switched between the two mock scenes.", self.latency_s * 0.3, 10)
        if "Question:" in prompt:
            time.sleep(self.latency_s * random.uniform(0.3, 0.6))
            return Generation("This is the mock model, so I can't really see the frame, "
                              "but the live video is coming through fine.", self.latency_s, 20)
        summary, obs = SCENES[self._i % len(SCENES)]
        self._i += 1
        payload = {
            "summary": summary,
            "observations": [[t, *b] if b else [t] for t, b in obs],
        }
        text = json.dumps(payload)
        chunks = max(1, len(text) // 6)  # pretend ~6 characters per token
        for k in range(1, chunks + 1):
            time.sleep(self.latency_s / chunks)
            if on_text and on_text(text[: len(text) * k // chunks]):
                text = text[: len(text) * k // chunks]
                break
        return Generation(text, self.latency_s, chunks)

    def close(self) -> None:
        pass

    def vram(self) -> None:
        return None
