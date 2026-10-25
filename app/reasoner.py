"""Worker thread: decides when to look, asks the VLM, and turns answers into scene updates."""
import base64
import logging
import queue
import re
import threading
import time

import cv2
import numpy as np
from PIL import Image

from . import prompts
from .camera import Camera
from .change import ChangeDetector
from .config import ReasonerConfig
from .parsing import parse_partial, parse_scene
from .state import AppState
from .vlm import VLM

log = logging.getLogger(__name__)

TICK_S = 0.15
REMOVAL_RE = re.compile(r"(took off|take off|taken off|removed|put down|put away|took out|set down)", re.I)


def thumbnail(frame: np.ndarray, width: int = 240) -> str:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


class Reasoner:
    def __init__(self, camera: Camera, vlm: VLM, state: AppState, cfg: ReasonerConfig):
        self.camera, self.vlm, self.state, self.cfg = camera, vlm, state, cfg
        self.detector = ChangeDetector(cfg.change_threshold, cfg.motion_threshold, cfg.settle_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="reasoner", daemon=True)
        self._last_run = 0.0
        self._last_frame: np.ndarray | None = None  # frame of the previous description

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=30)  # lets a running generate() finish before the model is freed

    # ------------------------------------------------------------------ loop
    def _run(self):
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # keep the monitor alive whatever the model does
                log.exception("Reasoner tick failed")
                self.state.update_status(busy=None)
                self._stop.wait(1)

    def _tick(self):
        st = self.state.status
        frame, _, _ = self.camera.latest()
        if frame is None:
            self._stop.wait(0.2)
            return

        try:  # questions jump the queue and work while paused
            q = self.state.questions.get_nowait()
        except queue.Empty:
            q = None
        if q:
            self._answer(q, frame)
            return

        now = time.time()
        changed = self.detector.update(frame, now)
        if st["paused"] and not self.state.analyze_now.is_set():
            self._stop.wait(TICK_S)
            return

        reason = None
        if self.state.analyze_now.is_set():
            reason = "manual"
        elif self.state.scene is None:
            reason = "first look"
        elif changed:
            reason = "scene changed"
        elif now - self._last_run >= st["heartbeat_s"]:
            reason = "heartbeat"

        if reason and now - self._last_run >= self.cfg.min_gap_s:
            self.state.analyze_now.clear()
            self._describe(frame, reason)
        else:
            self._stop.wait(TICK_S)

    # --------------------------------------------------------------- actions
    def _describe(self, frame: np.ndarray, reason: str):
        self.vlm.image_side = self.state.status["image_side"]
        # Compare with the previous frame only when it really differs (or on request); a plain
        # heartbeat over a still scene just refreshes the description.
        compare = self._last_frame is not None and (
            reason == "manual" or self.detector.last_change > self.cfg.change_threshold)
        before = self._last_frame
        self.detector.mark_analyzed(frame)
        self._last_frame = frame
        self._last_run = time.time()
        self.state.update_status(busy={"kind": "describe", "reason": reason, "since": self._last_run})

        first = self.state.scene is None
        h, w = frame.shape[:2]
        shown = {"summary": "", "n": 0}

        def on_text(text: str) -> bool:
            # Stream: publish the summary as soon as its string closes, then each observation
            # as it completes, so the UI fills in well before the whole reply is done.
            part = parse_partial(text, w, h)
            if part["summary"] and (part["summary"] != shown["summary"] or len(part["observations"]) > shown["n"]):
                shown["summary"], shown["n"] = part["summary"], len(part["observations"])
                self.state.publish_partial(part | {"ts": self._last_run, "frame_size": [w, h],
                                                   "elapsed_s": round(time.time() - self._last_run, 2)})
            # Enough detail, or the model started repeating itself: stop generating.
            return len(part["observations"]) >= prompts.MAX_OBSERVATIONS or part["repeats"] > 0

        gen = self.vlm.generate(to_pil(frame), prompts.SYSTEM, prompts.DESCRIBE,
                                prefill=prompts.DESCRIBE_PREFILL, on_text=on_text)
        parsed = parse_scene(gen.text, w, h)
        if not parsed.parsed:
            log.warning("Could not parse VLM output: %r", gen.text[:300])

        scene = parsed.to_dict() | {
            "changes": "",  # filled only by the before/after comparison below
            "id": self.state.next_id(), "ts": self._last_run, "reason": reason,
            "latency_s": round(gen.latency_s, 2), "tokens": gen.new_tokens,
            "frame_size": [w, h], "model": self.vlm.name,
        }
        self.state.set_scene(scene, self._event("start", parsed.summary, frame) if first else None)

        if compare:
            self.state.update_status(busy={"kind": "describe", "reason": "comparing with before",
                                           "since": time.time()})
            changes = self._what_changed(before, frame)
            if changes:
                self.state.set_scene(scene | {"changes": changes}, self._event("change", changes, frame))

        self.state.update_status(busy=None, last_latency_s=scene["latency_s"],
                                 analyses=self.state.status["analyses"] + 1, vram=self.vlm.vram())

    def _what_changed(self, before: np.ndarray, after: np.ndarray) -> str:
        gen = self.vlm.generate([to_pil(before), to_pil(after)], prompts.SYSTEM, prompts.CHANGES,
                                max_new_tokens=40)
        claim = " ".join(gen.text.split()).strip().strip('"')
        if not claim or claim.lower().rstrip(".") in ("none", "nothing", "no change", "no changes"):
            return ""
        # Verify: something "put on" must be absent before; something "taken off" absent after.
        removed = REMOVAL_RE.search(claim) is not None
        frame, when, question = ((after, "AFTER", "Is the item it mentions still being worn or held in this image?")
                                 if removed else
                                 (before, "BEFORE", "Is the item it mentions already being worn or held in this image?"))
        check = self.vlm.generate(to_pil(frame), prompts.SYSTEM,
                                  prompts.CHANGE_CHECK.format(claim=claim, when=when, question=question),
                                  max_new_tokens=3)
        if check.text.strip().lower().startswith("yes"):
            log.info("Rejected change claim %r (item visible %s)", claim, when.lower())
            return ""
        return claim

    def _event(self, kind: str, text: str, frame: np.ndarray) -> dict:
        return {"kind": kind, "text": text, "id": self.state.next_id(), "ts": self._last_run,
                "thumb": thumbnail(frame)}

    def _answer(self, q: dict, frame: np.ndarray):
        self.vlm.image_side = self.state.status["image_side"]
        self.state.update_status(busy={"kind": "ask", "reason": "question", "since": time.time()})
        gen = self.vlm.generate(to_pil(frame), prompts.SYSTEM, prompts.ask_prompt(q["question"]),
                                max_new_tokens=160)
        self.state.add_answer(q | {"answer": gen.text or "(no answer)", "answered_ts": time.time(),
                                   "latency_s": round(gen.latency_s, 2), "thumb": thumbnail(frame)})
        self.state.update_status(busy=None, last_latency_s=round(gen.latency_s, 2), vram=self.vlm.vram())
