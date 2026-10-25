"""Decides when the scene has changed enough (and settled) to be worth a new VLM pass.

Frames are shrunk to a small grayscale grid and compared block by block, so a local
change such as putting on glasses registers even though most of the frame is unchanged.
Global brightness shifts (auto-exposure) are cancelled by subtracting each frame's mean.
"""
import cv2
import numpy as np

GRID_W, GRID_H = 64, 36  # downscaled frame
BLOCK = 6  # block side in grid pixels -> 10 x 6 blocks


def small_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    g = cv2.resize(gray, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    return g - g.mean()


def block_change(a: np.ndarray, b: np.ndarray) -> float:
    """Largest mean absolute difference over any block (0-255 scale)."""
    d = np.abs(a - b)
    h, w = d.shape[0] // BLOCK * BLOCK, d.shape[1] // BLOCK * BLOCK
    blocks = d[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean(axis=(1, 3))
    return float(blocks.max())


class ChangeDetector:
    def __init__(self, change_threshold: float, motion_threshold: float, settle_s: float):
        self.change_threshold = change_threshold
        self.motion_threshold = motion_threshold
        self.settle_s = settle_s
        self._ref: np.ndarray | None = None  # frame at last analysis
        self._prev: np.ndarray | None = None  # previous checked frame
        self._still_since: float | None = None
        self.last_change = 0.0
        self.last_motion = 0.0

    def update(self, frame: np.ndarray, now: float) -> bool:
        """True when the scene differs from the last analysed frame and has stopped moving."""
        cur = small_gray(frame)
        self.last_motion = block_change(cur, self._prev) if self._prev is not None else 0.0
        self._prev = cur
        if self.last_motion > self.motion_threshold:
            self._still_since = None
        elif self._still_since is None:
            self._still_since = now

        if self._ref is None:
            return True
        self.last_change = block_change(cur, self._ref)
        settled = self._still_since is not None and now - self._still_since >= self.settle_s
        return self.last_change > self.change_threshold and settled

    def mark_analyzed(self, frame: np.ndarray):
        self._ref = small_gray(frame)
