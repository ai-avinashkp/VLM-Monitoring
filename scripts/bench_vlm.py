"""Phase 0: compare candidate VLMs on the same images.

Usage:
  python scripts/bench_vlm.py --images a.jpg b.jpg --models Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3.5-2B
Each model is loaded in its own subprocess so GPU memory is fully released in between.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_one(model_id: str, images: list[str], sides: list[int], runs: int, quant: bool) -> dict:
    import time

    import torch
    from PIL import Image

    from app import prompts
    from app.config import ModelConfig
    from app.parsing import parse_scene
    from app.vlm import build_vlm

    t0 = time.perf_counter()
    vlm = build_vlm(ModelConfig(id=model_id, quantize_4bit=quant))
    load_s = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    vlm.generate(Image.new("RGB", (320, 240)), "You are helpful.", "Say ok.", max_new_tokens=4)

    result = {"model": model_id, "4bit": quant, "load_s": round(load_s, 1), "by_side": {}, "samples": []}
    for side in sides:
        vlm.image_side = side
        lat, parsed_ok = [], 0
        for path in images:
            img = Image.open(path).convert("RGB")
            for r in range(runs):
                gen = vlm.generate(img, prompts.SYSTEM, prompts.DESCRIBE)
                lat.append(gen.latency_s)
                scene = parse_scene(gen.text, *img.size)
                parsed_ok += scene.parsed
                if r == 0:
                    result["samples"].append({"image": Path(path).name, "side": side, "latency_s": round(gen.latency_s, 2),
                                              "summary": scene.summary,
                                              "observations": [(o.text, o.box) for o in scene.observations]})
        result["by_side"][side] = {"median_s": round(statistics.median(lat), 2), "max_s": round(max(lat), 2),
                                   "parsed": f"{parsed_ok}/{len(lat)}"}
    result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3.5-2B"])
    ap.add_argument("--sides", nargs="+", type=int, default=[448, 640])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--quant", action="store_true", help="load in 4-bit")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        print("RESULT " + json.dumps(run_one(args.child, args.images, args.sides, args.runs, args.quant)))
        return

    for model in args.models:
        cmd = [sys.executable, __file__, "--child", model, "--images", *args.images,
               "--sides", *map(str, args.sides), "--runs", str(args.runs)] + (["--quant"] if args.quant else [])
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            print(f"== {model}: FAILED\n{out.stderr[-2000:]}")
            continue
        res = json.loads(line[7:])
        print(f"\n== {model}  load {res['load_s']} s  peak VRAM {res['peak_vram_gb']} GB")
        for side, s in res["by_side"].items():
            print(f"   {side}px  median {s['median_s']} s  max {s['max_s']} s  parsed {s['parsed']}")
        for smp in res["samples"]:
            print(f"   [{smp['image']} @{smp['side']}px, {smp['latency_s']} s] {smp['summary']}")
            for text, box in smp["observations"]:
                print(f"      - {text}  {box}")


if __name__ == "__main__":
    main()
