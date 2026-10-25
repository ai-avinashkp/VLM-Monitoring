"""FastAPI app: MJPEG video, WebSocket updates, ask/control endpoints, and the static UI.

Run:  python run.py   (loads + warms the model first, then serves; Ctrl+C shuts down in order)
Also works as `python -m uvicorn app.server:app`, loading the model in the background instead.
"""
import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .camera import Camera
from .config import ROOT, load_config
from .perf import boost_process
from .reasoner import Reasoner
from .state import AppState
from .vlm import VLM, build_vlm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("monitor")

STATIC = ROOT / "app" / "static"
cfg = load_config(os.environ.get("MONITOR_CONFIG", ROOT / "config.yaml"))
if os.environ.get("MONITOR_MODEL"):
    cfg.model.id = os.environ["MONITOR_MODEL"]

state = AppState(heartbeat_s=cfg.reasoner.heartbeat_s, image_side=cfg.model.image_side)
camera = Camera(cfg.camera, jpeg_quality=cfg.server.jpeg_quality)
runtime: dict = {}
stopping = threading.Event()  # set as soon as shutdown begins, so endless streams end promptly


def begin_shutdown():
    stopping.set()


def use_preloaded_model(vlm: VLM):
    """run.py loads and warms the model before the web server starts, then hands it over here."""
    runtime["vlm"] = vlm


def _start_watching(vlm: VLM):
    runtime["reasoner"] = Reasoner(camera, vlm, state, cfg.reasoner).start()
    state.update_status(phase="ready", message="Watching", model=vlm.name, vram=vlm.vram())
    log.info("Reasoner started with %s", vlm.name)


def _load_model_in_background():
    """Fallback for `uvicorn app.server:app`: load while the UI is already up."""
    try:
        state.update_status(phase="loading", message=f"Loading {cfg.model.id}", model=cfg.model.id)
        vlm = build_vlm(cfg.model)
        state.update_status(phase="warming", message="Warming up (compiling on first run)", model=vlm.name)
        vlm.generate(Image.new("RGB", (320, 240), "gray"), "You are helpful.", "Say ok.", max_new_tokens=4)
        runtime["vlm"] = vlm
        _start_watching(vlm)
    except Exception as e:  # surface load errors in the UI instead of a dead server
        log.exception("Model load failed")
        state.update_status(phase="error", message=f"Model failed to load: {e}")


async def _camera_status_loop():
    last = None
    while True:
        cur = {"connected": camera.connected, "fps": round(camera.fps)}
        if cur != last:
            state.update_status(camera=cur)
            last = cur
        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if cfg.server.boost_cpu and "vlm" not in runtime:  # run.py has already done this
        boost_process()
    camera.start()
    if "vlm" in runtime:
        _start_watching(runtime["vlm"])
    else:
        threading.Thread(target=_load_model_in_background, name="model-loader", daemon=True).start()
    cam_task = asyncio.create_task(_camera_status_loop())
    yield
    # Shutdown: the web server has stopped taking requests; stop the workers that use the model.
    cam_task.cancel()
    if "reasoner" in runtime:
        log.info("Stopping reasoner (finishing any running model call) ...")
        runtime.pop("reasoner").stop()
    log.info("Releasing camera ...")
    camera.stop()


app = FastAPI(title="Sightline", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/stream")
async def stream():
    boundary = b"frame"
    delay = 1 / max(1, cfg.server.stream_fps)

    async def frames():
        last_id = -1
        while not stopping.is_set():
            jpeg, fid = await asyncio.to_thread(camera.latest_jpeg)
            if jpeg is not None and fid != last_id:
                last_id = fid
                yield (b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            await asyncio.sleep(delay)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/state")
async def get_state():
    return state.snapshot()


@app.get("/api/health")
async def health():
    s = state.status
    return {"phase": s["phase"], "camera": s["camera"], "model": s["model"]}


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@app.post("/api/ask")
async def ask(body: AskBody):
    if state.status["phase"] != "ready":
        raise HTTPException(503, "Model is not ready yet")
    return state.ask(body.question.strip())


class ControlBody(BaseModel):
    action: str | None = None  # pause | resume | analyze
    heartbeat_s: float | None = Field(default=None, ge=3, le=120)
    image_side: int | None = Field(default=None, ge=256, le=1280)


@app.post("/api/control")
async def control(body: ControlBody):
    changes = {}
    if body.action == "pause":
        changes["paused"] = True
    elif body.action == "resume":
        changes["paused"] = False
        state.analyze_now.set()
    elif body.action == "analyze":
        state.analyze_now.set()
    elif body.action is not None:
        raise HTTPException(400, f"Unknown action {body.action!r}")
    if body.heartbeat_s is not None:
        changes["heartbeat_s"] = body.heartbeat_s
    if body.image_side is not None:
        changes["image_side"] = body.image_side
    if changes:
        state.update_status(**changes)
    return {"ok": True, "status": state.status}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    q = state.bus.subscribe()
    try:
        await websocket.send_json({"type": "snapshot", **state.snapshot()})
        while not stopping.is_set():
            try:
                msg = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            await websocket.send_json(msg)
        await websocket.close(code=1001)  # "going away": the page shows it is reconnecting
    except WebSocketDisconnect:
        pass
    finally:
        state.bus.unsubscribe(q)
