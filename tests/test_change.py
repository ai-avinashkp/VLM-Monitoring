import numpy as np

from app.change import ChangeDetector


def frame(value=100, patch=None):
    f = np.full((360, 640, 3), value, np.uint8)
    rng = np.random.default_rng(0)
    f = np.clip(f.astype(int) + rng.integers(-3, 4, f.shape), 0, 255).astype(np.uint8)  # sensor noise
    if patch is not None:
        y, x, v = patch
        f[y:y + 60, x:x + 90] = v
    return f


def test_first_frame_triggers_then_still_scene_does_not():
    d = ChangeDetector(change_threshold=14, motion_threshold=6, settle_s=0.5)
    assert d.update(frame(), 0.0)
    d.mark_analyzed(frame())
    for t in np.arange(0.1, 3, 0.1):
        assert not d.update(frame(), t)


def test_local_change_triggers_only_after_settling():
    d = ChangeDetector(change_threshold=14, motion_threshold=6, settle_s=0.5)
    d.update(frame(), 0.0)
    d.mark_analyzed(frame())
    changed = frame(patch=(120, 270, 30))  # small dark patch, e.g. glasses/mask
    assert not d.update(changed, 1.0)  # just moved: not settled yet
    assert not d.update(changed, 1.2)
    assert d.update(changed, 1.8)  # still since 1.2 for >= 0.5 s -> trigger
    d.mark_analyzed(changed)
    assert not d.update(changed, 2.0)


def test_global_brightness_shift_is_ignored():
    d = ChangeDetector(change_threshold=14, motion_threshold=6, settle_s=0.0)
    d.update(frame(100), 0.0)
    d.mark_analyzed(frame(100))
    d.update(frame(130), 1.0)
    assert not d.update(frame(130), 2.0)  # auto-exposure brightening, no real change
