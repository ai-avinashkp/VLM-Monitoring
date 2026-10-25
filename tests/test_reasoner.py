import time

import numpy as np

from app.config import ReasonerConfig
from app.reasoner import Reasoner
from app.state import AppState
from app.vlm.mock import MockVLM


class FakeCamera:
    connected, fps = True, 30.0

    def __init__(self):
        self.frame = np.full((360, 640, 3), 90, np.uint8)

    def latest(self):
        return self.frame, 1, time.time()


def wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def make(heartbeat=60.0):
    state = AppState(heartbeat_s=heartbeat, image_side=640)
    cfg = ReasonerConfig(heartbeat_s=heartbeat, min_gap_s=0.0, settle_s=0.0)
    r = Reasoner(FakeCamera(), MockVLM(latency_s=0.05), state, cfg)
    return state, r


def test_first_look_creates_scene_and_start_event():
    state, r = make()
    r.start()
    try:
        assert wait_for(lambda: state.scene is not None)
        assert state.scene["reason"] == "first look"
        assert state.scene["observations"][0]["box"] is not None
        assert state.events[0]["kind"] == "start"
        assert state.events[0]["thumb"].startswith("data:image/jpeg;base64,")
    finally:
        r.stop()


def test_manual_trigger_and_change_event():
    state, r = make()
    r.start()
    try:
        assert wait_for(lambda: state.scene is not None)
        state.analyze_now.set()
        assert wait_for(lambda: state.scene["reason"] == "manual")
        assert wait_for(lambda: any(e["kind"] == "change" for e in state.events))
    finally:
        r.stop()


def test_questions_are_answered_even_when_paused():
    state, r = make()
    state.status["paused"] = True
    r.start()
    try:
        q = state.ask("What am I holding?")
        assert wait_for(lambda: len(state.answers) == 1)
        assert state.answers[0]["id"] == q["id"]
        assert state.answers[0]["answer"]
        assert state.scene is None or state.scene["reason"] != "heartbeat"
    finally:
        r.stop()
