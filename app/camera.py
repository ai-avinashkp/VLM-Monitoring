"""Threaded webcam capture that always holds only the newest frame."""
import logging
import threading
import time

import cv2
import numpy as np

from .config import CameraConfig

log = logging.getLogger(__name__)

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


class Camera:
    def __init__(self, cfg: CameraConfig, jpeg_quality: int = 80):
        self.cfg = cfg
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._frame_ts = 0.0
        self._jpeg: bytes | None = None
        self._jpeg_id = -1
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self.connected = False
        self.fps = 0.0

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def start(self):
        if not self.running:
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def latest(self) -> tuple[np.ndarray | None, int, float]:
        with self._lock:
            return self._frame, self._frame_id, self._frame_ts

    def latest_jpeg(self) -> tuple[bytes | None, int]:
        """JPEG of the newest frame, encoded at most once per frame and shared by all viewers."""
        with self._lock:
            frame, fid = self._frame, self._frame_id
            if frame is None:
                return None, -1
            if self._jpeg_id == fid:
                return self._jpeg, fid
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return None, -1
        with self._lock:
            self._jpeg, self._jpeg_id = buf.tobytes(), fid
        return self._jpeg, fid

    def _open(self) -> cv2.VideoCapture | None:
        preferred = BACKENDS.get(self.cfg.backend.lower(), cv2.CAP_ANY)
        for backend in dict.fromkeys([preferred, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]):
            cap = cv2.VideoCapture(self.cfg.index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
                cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        return None

    def _run(self):
        cap = None
        n, t_window = 0, time.perf_counter()
        while not self._stop.is_set():
            if cap is None:
                cap = self._open()
                if cap is None:
                    self.connected = False
                    log.warning("Camera %s not available, retrying in 2 s", self.cfg.index)
                    self._stop.wait(2)
                    continue
                log.info("Camera %s opened", self.cfg.index)
            ok, frame = cap.read()
            if not ok:
                log.warning("Camera read failed, reconnecting")
                cap.release()
                cap, self.connected = None, False
                self._stop.wait(1)
                continue
            self.connected = True
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._frame_ts = time.time()
            n += 1
            elapsed = time.perf_counter() - t_window
            if elapsed >= 1.0:
                self.fps, n, t_window = n / elapsed, 0, time.perf_counter()
        if cap is not None:
            cap.release()
