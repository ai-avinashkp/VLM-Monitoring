"""Phase 0: confirm the webcam opens and report what it delivers.

Usage: python scripts/check_camera.py [--index 0] [--save frame.jpg]
"""
import argparse
import time

import cv2

BACKENDS = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    for name, backend in BACKENDS:
        cap = cv2.VideoCapture(args.index, backend)
        if not cap.isOpened():
            print(f"[{name}] could not open camera {args.index}")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        # first frames are often dark while auto-exposure settles
        for _ in range(10):
            cap.read()
        n, t0, frame = 0, time.perf_counter(), None
        while n < 60:
            ok, f = cap.read()
            if not ok:
                break
            frame, n = f, n + 1
        dt = time.perf_counter() - t0
        cap.release()

        if frame is None:
            print(f"[{name}] opened but returned no frames")
            continue
        h, w = frame.shape[:2]
        print(f"[{name}] OK  {w}x{h}  ~{n / dt:.1f} fps  mean brightness {frame.mean():.0f}")
        if args.save:
            cv2.imwrite(args.save, frame)
            print(f"saved {args.save}")
        return
    raise SystemExit("No backend could read from the camera.")


if __name__ == "__main__":
    main()
