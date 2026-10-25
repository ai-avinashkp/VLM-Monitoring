"""Turn whatever the VLM wrote into a SceneState, without ever raising."""
import json
import re
from dataclasses import asdict, dataclass, field

FENCE_RE = re.compile(r"```(?:json)?", re.I)
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
NO_CHANGE = re.compile(r"^(none|n/?a|no( notable| meaningful| significant)? changes?\b.*|nothing\b.*|-|)$", re.I)


@dataclass
class Observation:
    text: str
    box: list[float] | None = None  # normalized [x1, y1, x2, y2] in 0-1


@dataclass
class SceneState:
    summary: str
    observations: list[Observation] = field(default_factory=list)
    changes: str = ""
    raw: str = ""
    parsed: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _first_json_object(text: str) -> dict | None:
    text = FENCE_RE.sub("", text)
    decoder = json.JSONDecoder()
    for candidate in (text, TRAILING_COMMA_RE.sub(r"\1", text)):
        for m in re.finditer(r"\{", candidate):
            try:
                obj, _ = decoder.raw_decode(candidate, m.start())
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def normalize_box(box, img_w: int | None = None, img_h: int | None = None) -> list[float] | None:
    """Accept 0-1, 0-1000, or absolute-pixel boxes and return a clamped 0-1 box."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    hi = max(x1, y1, x2, y2)
    if hi <= 1.0:
        sx = sy = 1.0
    elif hi <= 1000:
        sx = sy = 1000.0
    elif img_w and img_h:
        sx, sy = float(img_w), float(img_h)
    else:
        return None
    x1, x2 = sorted((x1 / sx, x2 / sx))
    y1, y2 = sorted((y1 / sy, y2 / sy))
    clamp = lambda v: min(1.0, max(0.0, v))
    x1, y1, x2, y2 = clamp(x1), clamp(y1), clamp(x2), clamp(y2)
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        return None
    if (x2 - x1) * (y2 - y1) > 0.9:  # "the whole frame" is not a useful place to point at
        return None
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def _clean(s) -> str:
    return " ".join(str(s).split()) if s is not None else ""


def parse_scene(text: str, img_w: int | None = None, img_h: int | None = None) -> SceneState:
    obj = _first_json_object(text)
    if obj is None:
        # A reply cut off on purpose (enough observations) or by the token limit: keep what's complete.
        part = parse_partial(text, img_w, img_h)
        if part["summary"]:
            return SceneState(summary=part["summary"], changes=_no_change(part["changes"]), raw=text,
                              observations=[Observation(o["text"], o["box"]) for o in part["observations"]])
        # Otherwise show the model's prose as-is rather than nothing.
        prose = _clean(FENCE_RE.sub("", text)) or "No description returned."
        return SceneState(summary=prose[:400], raw=text, parsed=False)

    observations = _observations(obj.get("observations") or [], img_w, img_h)
    changes = _no_change(_clean(obj.get("changes")))
    summary = _clean(obj.get("summary")) or (observations[0].text if observations else "")
    return SceneState(summary=summary, observations=observations[:6], changes=changes, raw=text)


def _observations(items: list, img_w: int | None = None, img_h: int | None = None,
                  stats: dict | None = None) -> list[Observation]:
    out, seen, repeats = [], set(), 0
    for item in items:
        # The model sometimes packs every detail into one list: ["a", "b", "c"].
        parts = item if isinstance(item, list) and len(item) > 1 and all(isinstance(x, str) for x in item) else [item]
        for part in parts:
            o = _observation(part, img_w, img_h)
            if not o:
                continue
            if o.text.lower() in seen:  # drop repeats from a looping model
                repeats += 1
                continue
            seen.add(o.text.lower())
            out.append(o)
    if stats is not None:
        stats["repeats"] = repeats
    return out


def _no_change(changes: str) -> str:
    return "" if NO_CHANGE.match(changes.rstrip(".")) else changes


def _observation(item, img_w: int | None = None, img_h: int | None = None) -> Observation | None:
    """Accept the compact ["text", x1, y1, x2, y2] form as well as {"text", "box"} and bare strings."""
    if isinstance(item, str):
        text, box = item, None
    elif isinstance(item, list) and item and isinstance(item[0], str):
        rest = item[1:]
        text, box = item[0], rest[0] if len(rest) == 1 and isinstance(rest[0], list) else rest
    elif isinstance(item, dict):
        text = item.get("text") or item.get("description")
        box = item.get("box", item.get("box_2d", item.get("bbox_2d")))
    else:
        return None
    text = _clean(text)
    return Observation(text, normalize_box(box, img_w, img_h)) if text else None


SUMMARY_RE = re.compile(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"')
CHANGES_RE = re.compile(r'"changes"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _complete_elements(text: str, start: int) -> list[str]:
    """Top-level elements of the JSON array opening at text[start] that are already closed."""
    out, depth, in_str, esc, elem_start = [], 0, False, False, None
    for i in range(start + 1, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
                if depth == 0:  # a bare-string element
                    out.append(text[elem_start:i + 1])
            continue
        if c == '"':
            in_str = True
            if depth == 0:
                elem_start = i
        elif c in "[{":
            if depth == 0:
                elem_start = i
            depth += 1
        elif c in "]}":
            if depth == 0:  # the array itself closed
                break
            depth -= 1
            if depth == 0:
                out.append(text[elem_start:i + 1])
    return out


def parse_partial(text: str, img_w: int | None = None, img_h: int | None = None) -> dict:
    """What can already be read from a reply that is still streaming: summary and changes once
    their strings are closed, and every observation that is complete."""
    summary, changes = (_closed_string(rx, text) for rx in (SUMMARY_RE, CHANGES_RE))
    observations, repeats = [], 0
    k = text.find('"observations"')
    bracket = text.find("[", k) if k >= 0 else -1
    if bracket >= 0:
        items = []
        for raw in _complete_elements(text, bracket):
            try:
                items.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        stats = {}
        observations = [asdict(o) for o in _observations(items, img_w, img_h, stats)]
        repeats = stats["repeats"]
    return {"summary": summary, "changes": changes, "observations": observations, "repeats": repeats}


def _closed_string(rx: re.Pattern, text: str) -> str:
    m = rx.search(text)
    if not m:
        return ""
    try:
        return _clean(json.loads(f'"{m.group(1)}"'))
    except json.JSONDecodeError:
        return _clean(m.group(1))
