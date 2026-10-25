"""Loading, warming up and offloading the model, with readable progress for the terminal."""
import gc
import platform
import time
from collections.abc import Callable

import psutil
from PIL import Image

from . import prompts
from .config import ModelConfig
from .vlm import VLM, build_vlm

Say = Callable[[str], None]


def _gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def memory_report() -> str:
    """One line: GPU memory (device-wide and ours), system RAM, and this process's RAM."""
    ram = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    parts = []
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            parts.append(f"VRAM {_gb(total - free)} / {_gb(total)} used "
                         f"(model tensors {_gb(torch.cuda.memory_allocated())}, "
                         f"reserved {_gb(torch.cuda.memory_reserved())})")
    except ImportError:
        pass
    parts.append(f"RAM {_gb(ram.used)} / {_gb(ram.total)} used, this process {_gb(rss)}")
    return " | ".join(parts)


def _os_name() -> str:
    if platform.system() == "Windows":
        build = int(platform.version().split(".")[-1] or 0)
        return f"Windows {'11' if build >= 22000 else '10'} (build {build})"  # release() says 10 on 11
    return f"{platform.system()} {platform.release()}"


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "unknown CPU"


def system_report() -> list[str]:
    lines = [f"Python {platform.python_version()} on {_os_name()}",
             f"CPU {_cpu_name()}, {psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads"]
    try:
        import torch
        lines.append(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            lines.append(f"GPU {p.name}, {_gb(p.total_memory)}, compute {p.major}.{p.minor}")
        else:
            lines.append("GPU not available (CUDA off)")
    except ImportError:
        lines.append("PyTorch not installed")
    return lines


def load_model(cfg: ModelConfig, say: Say = print) -> VLM:
    say(f"Loading model {cfg.id} ({cfg.dtype}{', 4-bit' if cfg.quantize_4bit else ''}) ...")
    t0 = time.perf_counter()
    vlm = build_vlm(cfg)
    say(f"  loaded {vlm.name} in {time.perf_counter() - t0:.1f} s")
    say(f"  {memory_report()}")
    return vlm


def warm_up(vlm: VLM, say: Say = print) -> None:
    """Run real-shaped passes so compilation and CUDA graph capture happen now, not on the
    first live frame. The very first run on a machine compiles for a long time; it's cached after."""
    say("Warming up GPU (compiling the decoder; the very first run on this machine can take many minutes) ...")
    side = vlm.image_side
    frame = Image.new("RGB", (side, side * 9 // 16), "gray")
    t0 = time.perf_counter()
    vlm.generate(frame, "You are helpful.", "Say ok.", max_new_tokens=4)
    t1 = time.perf_counter()
    vlm.generate(frame, prompts.SYSTEM, prompts.DESCRIBE, prefill=prompts.DESCRIBE_PREFILL, max_new_tokens=24)
    vlm.generate([frame, frame], prompts.SYSTEM, prompts.CHANGES, max_new_tokens=4)
    t2 = time.perf_counter()
    say(f"  compile/first pass {t1 - t0:.1f} s, describe + compare passes {t2 - t1:.1f} s")
    say(f"  {memory_report()}")


def offload(vlm: VLM | None, say: Say = print) -> None:
    if vlm is None:
        return
    say(f"Offloading model {vlm.name} ...")
    before = memory_report()
    vlm.close()
    gc.collect()
    say(f"  before: {before}")
    say(f"  after:  {memory_report()}")
