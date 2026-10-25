"""Phase 1: grab one webcam frame (or load an image) and print what the VLM sees.

Usage:
  python scripts/describe_once.py                      # webcam, model from config.yaml
  python scripts/describe_once.py --image me.jpg --model Qwen/Qwen3.5-2B
  python scripts/describe_once.py --ask "Am I wearing glasses?"
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from app import prompts  # noqa: E402
from app.config import load_config  # noqa: E402
from app.parsing import parse_scene  # noqa: E402
from app.perf import boost_process  # noqa: E402
from app.reasoner import to_pil  # noqa: E402
from app.vlm import build_vlm  # noqa: E402


def grab_frame(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(15):  # let auto-exposure settle
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read from the webcam")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--model")
    ap.add_argument("--side", type=int)
    ap.add_argument("--ask")
    ap.add_argument("--save", help="save the captured frame here")
    args = ap.parse_args()

    cfg = load_config()
    if cfg.server.boost_cpu:
        boost_process()
    if args.model:
        cfg.model.id = args.model
    if args.side:
        cfg.model.image_side = args.side

    frame = cv2.imread(args.image) if args.image else grab_frame(cfg.camera.index)
    if args.save:
        cv2.imwrite(args.save, frame)

    t0 = time.perf_counter()
    vlm = build_vlm(cfg.model)
    print(f"loaded {cfg.model.id} in {time.perf_counter() - t0:.1f} s  vram={vlm.vram()}")
    vlm.generate(to_pil(frame), "You are helpful.", "Say ok.", max_new_tokens=4)  # warm-up

    prompt = prompts.ask_prompt(args.ask) if args.ask else prompts.DESCRIBE
    gen = vlm.generate(to_pil(frame), prompts.SYSTEM, prompt,
                       prefill="" if args.ask else prompts.DESCRIBE_PREFILL)
    print(f"\n--- raw ({gen.latency_s:.2f} s, {gen.new_tokens} tokens) ---\n{gen.text}\n")
    if not args.ask:
        h, w = frame.shape[:2]
        print("--- parsed ---")
        print(json.dumps(parse_scene(gen.text, w, h).to_dict() | {"raw": "..."}, indent=2))
    print(f"vram={vlm.vram()}")


if __name__ == "__main__":
    main()
