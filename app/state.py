"""Shared state between the reasoner thread and the async web server."""
import asyncio
import itertools
import queue
import threading
import time
from collections import deque


class Broadcaster:
    """Fan out messages from worker threads to every connected WebSocket's asyncio queue."""

    def __init__(self):
        self._lock = threading.Lock()
        self._subs: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        with self._lock:
            self._subs.add((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, q: asyncio.Queue):
        with self._lock:
            self._subs = {s for s in self._subs if s[1] is not q}

    def publish(self, msg: dict):
        with self._lock:
            subs = list(self._subs)
        for loop, q in subs:
            loop.call_soon_threadsafe(_put_drop_oldest, q, msg)


def _put_drop_oldest(q: asyncio.Queue, msg: dict):
    if q.full():
        q.get_nowait()
    q.put_nowait(msg)


class AppState:
    def __init__(self, heartbeat_s: float, image_side: int):
        self._lock = threading.Lock()
        self.bus = Broadcaster()
        self.scene: dict | None = None
        self.events: deque[dict] = deque(maxlen=150)
        self.answers: deque[dict] = deque(maxlen=60)
        self.questions: queue.Queue = queue.Queue()
        self._ids = itertools.count(1)
        self.status: dict = {
            "phase": "starting",  # starting | loading | ready | error
            "message": "Starting up",
            "model": None,
            "busy": None,  # None or {"kind": "describe"|"ask", "reason": str, "since": ts}
            "paused": False,
            "heartbeat_s": heartbeat_s,
            "image_side": image_side,
            "camera": {"connected": False, "fps": 0.0},
            "last_latency_s": None,
            "analyses": 0,
            "vram": None,
        }
        self.analyze_now = threading.Event()

    def next_id(self) -> int:
        return next(self._ids)

    def update_status(self, **changes):
        with self._lock:
            self.status.update(changes)
            snapshot = dict(self.status)
        self.bus.publish({"type": "status", "status": snapshot})

    def set_scene(self, scene: dict, event: dict | None):
        with self._lock:
            self.scene = scene
            if event:
                self.events.append(event)
        self.bus.publish({"type": "scene", "scene": scene})
        if event:
            self.bus.publish({"type": "event", "event": event})

    def publish_partial(self, partial: dict):
        """A description that is still being generated; not stored, only shown."""
        self.bus.publish({"type": "scene_partial", "scene": partial})

    def add_answer(self, answer: dict):
        with self._lock:
            self.answers.append(answer)
        self.bus.publish({"type": "answer", "answer": answer})

    def ask(self, question: str) -> dict:
        item = {"id": self.next_id(), "question": question, "ts": time.time()}
        self.questions.put(item)
        self.bus.publish({"type": "question", "question": item})
        return item

    def snapshot(self) -> dict:
        with self._lock:
            return {"status": dict(self.status), "scene": self.scene,
                    "events": list(self.events), "answers": list(self.answers)}
