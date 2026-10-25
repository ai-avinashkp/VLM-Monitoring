"""Start Sightline in order, and shut it down in reverse.

  startup:   system info -> load model -> warm up GPU -> open camera -> start web server
  Ctrl+C:    stop web server -> stop reasoner + camera -> offload model from GPU -> exit

Usage:  python run.py [--port 8000] [--host 127.0.0.1] [--model Qwen/Qwen3-VL-2B-Instruct | mock]
"""
import argparse
import asyncio
import contextlib
import logging
import os
import sys
import threading
import time

import uvicorn

T0 = time.time()


def say(msg: str = "") -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}" if msg else "", flush=True)


def setup_logging() -> None:
    """App log lines get the same elapsed-time prefix as the step messages."""
    handler = logging.StreamHandler(sys.stdout)

    def add_elapsed(record):
        record.elapsed = f"[{record.created - T0:7.1f}s]"
        return True

    handler.addFilter(add_elapsed)
    handler.setFormatter(logging.Formatter("%(elapsed)s   %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").addFilter(quiet_forced_close)


def quiet_forced_close(record: logging.LogRecord) -> bool:
    """A client that stopped reading (frozen tab, sleeping laptop) can hold a video stream open
    until uvicorn's grace period ends and it is cancelled. That's expected during shutdown, so
    replace uvicorn's traceback with one line."""
    from app import server
    if not server.stopping.is_set():
        return True
    msg = record.getMessage()
    if "timeout graceful shutdown exceeded" in msg:
        say("  closed a connection that stopped reading (after the 3 s grace period)")
        return False
    exc = record.exc_info[1] if record.exc_info else None
    return not (msg == "Exception in ASGI application" and isinstance(exc, asyncio.CancelledError))


def step(n: int, total: int, title: str) -> None:
    say()
    say(f"== [{n}/{total}] {title}")


class WebServer(uvicorn.Server):
    """uvicorn re-raises the stop signal after its graceful shutdown. For Ctrl+Break, or closing
    the console, that would hit Python's default handler and kill the process before the model is
    offloaded, so we keep the signal: shutdown continues below in main()."""

    @contextlib.contextmanager
    def capture_signals(self):
        with super().capture_signals():
            yield
            self._captured_signals.clear()

    def handle_exit(self, sig, frame):
        from app import server
        if not self.should_exit:
            say()
            say("== Stop requested: stopping web server ...")
        server.begin_shutdown()  # end video streams and websockets now instead of waiting them out
        super().handle_exit(sig, frame)


def wait_for_camera(camera, timeout_s: float = 8.0) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        frame, _, _ = camera.latest()
        if frame is not None:
            time.sleep(1.2)  # let the fps estimate settle
            h, w = frame.shape[:2]
            say(f"  camera {camera.cfg.index} open: {w}x{h} at ~{camera.fps:.0f} fps")
            return True
        time.sleep(0.1)
    say(f"  camera {camera.cfg.index} not available yet; the app keeps retrying in the background")
    return False


def announce_when_up(server: uvicorn.Server, url: str) -> None:
    while not server.started and not server.should_exit:
        time.sleep(0.1)
    if server.started:
        say(f"  web server running at {url}")
        say()
        say(f"Ready. Open {url}   (press Ctrl+C to stop)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--model", help="override model.id from config.yaml ('mock' runs without a GPU)")
    args = ap.parse_args()
    if args.model:
        os.environ["MONITOR_MODEL"] = args.model

    setup_logging()
    from app import lifecycle, server  # importing reads config.yaml; nothing starts yet
    from app.perf import boost_process

    cfg = server.cfg
    host, port = args.host or cfg.server.host, args.port or cfg.server.port
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"

    say("Sightline: live visual reasoning")
    vlm = None
    try:
        step(1, 5, "System")
        for line in lifecycle.system_report():
            say(f"  {line}")
        if cfg.server.boost_cpu:
            boost_process()
        say(f"  {lifecycle.memory_report()}")

        step(2, 5, "Load model")
        vlm = lifecycle.load_model(cfg.model, say)

        step(3, 5, "Warm up GPU")
        lifecycle.warm_up(vlm, say)

        step(4, 5, "Camera")
        server.camera.start()
        wait_for_camera(server.camera)

        step(5, 5, "Web server")
        server.use_preloaded_model(vlm)
        web = WebServer(uvicorn.Config(server.app, host=host, port=port, log_level="warning",
                                       access_log=False, timeout_graceful_shutdown=3))
        threading.Thread(target=announce_when_up, args=(web, url), daemon=True).start()
        try:
            web.run()  # blocks; on Ctrl+C it stops serving, then runs the app's shutdown (reasoner, camera)
        except KeyboardInterrupt:
            pass

        say()
        say("== Shutting down")
        say("  web server stopped; reasoner and camera released")
    except KeyboardInterrupt:
        say()
        say("== Interrupted during startup, shutting down")
        server.camera.stop()
    finally:
        lifecycle.offload(vlm, say)
        say("Shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
