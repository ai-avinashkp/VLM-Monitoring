"""Any Hugging Face image-text-to-text chat model (Qwen3-VL, Qwen3.5, SmolVLM, ...)."""
import logging
import re
import threading
import time
from collections.abc import Callable

import torch
from PIL import Image
from transformers import (AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
                          StoppingCriteria, StoppingCriteriaList)
from transformers.generation.streamers import BaseStreamer

from .base import Generation, fit_image

log = logging.getLogger(__name__)

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class _CallbackStreamer(BaseStreamer):
    """Calls on_text(full reply so far) after every new token, on the generating thread.

    transformers' TextIteratorStreamer needs generate() on a second thread, which would replay
    the compiled CUDA graphs from a different thread than they were captured on.
    """

    def __init__(self, tokenizer, prefill: str, on_text: Callable[[str], bool | None]):
        self.tokenizer, self.prefill, self.on_text = tokenizer, prefill, on_text
        self.ids: list[int] = []
        self._prompt_seen = False
        self.stop = False  # set when on_text returns True

    def put(self, value):
        if not self._prompt_seen:  # the first call carries the prompt
            self._prompt_seen = True
            return
        self.ids.extend(value.reshape(-1).tolist())
        self.stop = bool(self.on_text(self.prefill + self.tokenizer.decode(self.ids, skip_special_tokens=True)))

    def end(self):
        pass


class _StopWhenAsked(StoppingCriteria):
    def __init__(self, streamer: _CallbackStreamer):
        self.streamer = streamer

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.streamer.stop, dtype=torch.bool, device=input_ids.device)


def _from_cache_first(cls, model_id: str, **kwargs):
    """Load from the local HF cache without touching the network; download only if missing."""
    try:
        return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
    except OSError:
        log.info("%s not in local cache, downloading", model_id)
        return cls.from_pretrained(model_id, **kwargs)


class HFVLM:
    def __init__(self, model_id: str, dtype: str = "bfloat16", quantize_4bit: bool = False,
                 image_side: int = 640, max_new_tokens: int = 260, static_cache: bool = True,
                 max_cache_len: int = 2048):
        self.name = model_id.split("/")[-1]
        self.image_side = image_side
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()  # one generate() at a time on the GPU
        torch_dtype = getattr(torch, dtype)

        kwargs = dict(dtype=torch_dtype, device_map="cuda", attn_implementation="sdpa")
        if quantize_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype)

        t0 = time.perf_counter()
        self.processor = _from_cache_first(AutoProcessor, model_id)
        self.model = _from_cache_first(AutoModelForImageTextToText, model_id, **kwargs).eval()
        log.info("Loaded %s in %.1f s", model_id, time.perf_counter() - t0)

        if static_cache and not quantize_4bit:
            # A fixed-size KV cache lets transformers compile the decode step into CUDA graphs.
            # Without it a 2B model is launch-bound (~150 ms/token on an RTX 3070 Ti laptop, GPU mostly
            # idle and down-clocked); with it ~17 ms/token. The first compile is slow but is
            # cached on disk by torch inductor, so later starts take ~25 s.
            gc = self.model.generation_config
            gc.cache_implementation = "static"
            gc.max_cache_len = max_cache_len
        # Hybrid-thinking models (e.g. Qwen3.5) must answer directly; only pass the flag to
        # templates that use it, otherwise transformers treats it as a processor argument.
        template = self.processor.chat_template or ""
        self._template_kwargs = {"enable_thinking": False} if "enable_thinking" in template else {}
        # Greedy decoding; clear sampling defaults so generate() doesn't warn every call.
        gc = self.model.generation_config
        gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None

    def generate(self, image: Image.Image | list[Image.Image], system: str, prompt: str,
                 max_new_tokens: int | None = None, prefill: str = "",
                 on_text: Callable[[str], bool | None] | None = None) -> Generation:
        """`image` may be a list (e.g. before/after frames).
        `prefill` starts the assistant's reply (e.g. '{"summary": "You'); it is included in the result.
        `on_text` receives the reply so far after each token, for streaming; returning True stops."""
        images = image if isinstance(image, list) else [image]
        content = [{"type": "image", "image": fit_image(im.convert("RGB"), self.image_side)} for im in images]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content + [{"type": "text", "text": prompt}]},
        ]
        if prefill:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": prefill}]})
        with self._lock, torch.inference_mode():
            t0 = time.perf_counter()
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=not prefill, continue_final_message=bool(prefill),
                **self._template_kwargs,
            ).to(self.model.device)
            streamer = _CallbackStreamer(self.processor.tokenizer, prefill, on_text) if on_text else None
            stopping = StoppingCriteriaList([_StopWhenAsked(streamer)]) if streamer else None
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                                      do_sample=False, streamer=streamer, stopping_criteria=stopping)
            new = out[:, inputs["input_ids"].shape[1]:]
            text = self.processor.batch_decode(new, skip_special_tokens=True)[0]
            latency = time.perf_counter() - t0
        return Generation(prefill + THINK_RE.sub("", text).strip(), latency, int(new.shape[1]))

    def close(self) -> None:
        """Free the GPU: drop the weights, compiled graphs and cached allocator blocks."""
        import gc
        with self._lock:  # never while a generate() is running
            self.model = self.processor = None
            gc.collect()
            torch.compiler.reset()  # releases compiled code and CUDA-graph memory pools
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def vram(self) -> dict:
        free, total = torch.cuda.mem_get_info()
        return {"used_gb": round((total - free) / 1e9, 2), "total_gb": round(total / 1e9, 2),
                "model_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}
