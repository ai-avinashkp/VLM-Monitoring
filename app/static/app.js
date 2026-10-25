"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const SVG_NS = "http://www.w3.org/2000/svg";

const ui = {
  stage: $("#stage"), video: $("#video"), bubbles: $("#bubbles"), loose: $("#loose-notes"),
  leaders: $("#leaders"), regions: $("#regions"), assessment: $("#assessment"),
  assessmentText: $("#assessment-text"), assessmentMeta: $("#assessment-meta"),
  overlay: $("#overlay"), overlayTitle: $("#overlay-title"), overlaySub: $("#overlay-sub"),
  thinking: $("#thinking"), thinkingText: $("#thinking-text"), clock: $("#hud-clock"),
  meter: $("#meter-marker"), pillState: $("#pill-state"), pillModel: $("#pill-model"),
  latency: $("#m-latency"), vram: $("#m-vram"), camera: $("#m-camera"),
  btnPause: $("#btn-pause"), btnAnalyze: $("#btn-analyze"), heartbeat: $("#heartbeat"),
  heartbeatVal: $("#heartbeat-val"), detail: $("#detail"), optBubbles: $("#opt-bubbles"),
  optMirror: $("#opt-mirror"), nowEmpty: $("#now-empty"), nowBody: $("#now-body"),
  nowSummary: $("#now-summary"), nowChange: $("#now-change"), nowObs: $("#now-obs"),
  nowFacts: $("#now-facts"), nowRaw: $("#now-raw"), timeline: $("#timeline"),
  timelineEmpty: $("#timeline-empty"), timelineCount: $("#timeline-count"), thread: $("#thread"),
  threadEmpty: $("#thread-empty"), composer: $("#composer"), question: $("#question"),
  chips: $("#chips"), toast: $("#toast"),
};

const S = { status: null, scene: null, partial: null, events: [], answers: [], pending: new Map(), online: false };

// ------------------------------------------------------------------ prefs
const prefs = (() => {
  const defaults = { bubbles: true, mirror: true, tab: "now" };
  try { return { ...defaults, ...JSON.parse(localStorage.getItem("sightline-prefs") || "{}") }; }
  catch { return defaults; }
})();
function savePrefs() {
  try { localStorage.setItem("sightline-prefs", JSON.stringify(prefs)); } catch { /* private mode */ }
}

// ------------------------------------------------------------------ server
async function api(path, body) {
  const res = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `${res.status}`;
    try { msg = (await res.json()).detail || msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  return res.json();
}

let backoff = 1000;
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => {
    S.online = true;
    backoff = 1000;
  };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => {
    if (S.online) restartVideo();
    S.online = false;
    renderStatus();
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 1.6, 8000);
  };
}

function handle(m) {
  switch (m.type) {
    case "snapshot":
      Object.assign(S, { status: m.status, scene: m.scene, events: m.events, answers: m.answers });
      S.pending.clear();
      renderStatus(); renderScene(); renderTimeline(); renderThread();
      break;
    case "status": S.status = m.status; renderStatus(); break;
    case "scene_partial": S.partial = m.scene; renderPartial(); break;
    case "scene": S.partial = null; S.scene = m.scene; renderScene(); break;
    case "event": S.events.push(m.event); renderTimeline(true); break;
    case "question": S.pending.set(m.question.id, m.question); renderThread(); break;
    case "answer":
      S.pending.delete(m.answer.id);
      S.answers.push(m.answer);
      renderThread();
      break;
  }
}

function restartVideo() {
  ui.video.src = `/stream?t=${Date.now()}`;
}
ui.video.addEventListener("error", () => setTimeout(restartVideo, 1500));
ui.video.addEventListener("load", () => {
  if (ui.video.naturalWidth) setAspect(ui.video.naturalWidth, ui.video.naturalHeight);
});

function setAspect(w, h) {
  const ratio = `${w} / ${h}`;
  if (ui.stage.style.aspectRatio !== ratio) ui.stage.style.aspectRatio = ratio;
}

// ------------------------------------------------------------------ status
function stateOf(st) {
  if (!S.online) return ["offline", "Reconnecting"];
  if (!st) return ["loading", "Starting"];
  if (st.phase === "error") return ["error", "Error"];
  if (st.phase !== "ready") return ["loading", st.phase === "warming" ? "Warming up" : "Loading model"];
  if (st.busy) return ["thinking", st.busy.kind === "ask" ? "Answering" : "Looking"];
  if (st.paused) return ["paused", "Paused"];
  return ["live", "Live"];
}

function renderStatus() {
  const st = S.status;
  const [state, label] = stateOf(st);
  ui.pillState.dataset.state = state;
  $(".label", ui.pillState).textContent = label;
  if (!st) return;

  ui.pillModel.textContent = (st.model || "—").split("/").pop();
  ui.latency.textContent = st.last_latency_s != null ? `${st.last_latency_s.toFixed(1)} s` : "—";
  ui.vram.textContent = st.vram ? `${st.vram.used_gb.toFixed(1)} / ${st.vram.total_gb.toFixed(1)} GB` : "—";
  ui.camera.textContent = st.camera.connected ? `${Math.round(st.camera.fps)} fps` : "offline";

  const ready = st.phase === "ready";
  let overlay = null;
  if (!S.online) overlay = ["Connection lost", "Trying to reach the server again…", false];
  else if (st.phase === "error") overlay = ["Model failed to load", st.message, true];
  else if (!ready) overlay = [st.message || "Loading model",
    st.phase === "warming" ? "Running a first test pass on the GPU." :
      "The first run downloads the model, which can take a few minutes.", false];
  else if (!st.camera.connected) overlay = ["Camera not available",
    "Close other apps that use the webcam. Retrying automatically…", false];
  ui.overlay.hidden = !overlay;
  if (overlay) {
    ui.overlayTitle.textContent = overlay[0];
    ui.overlaySub.textContent = overlay[1];
    ui.overlay.classList.toggle("error", overlay[2]);
  }

  ui.stage.classList.toggle("busy", !!st.busy);
  ui.stage.classList.toggle("paused", !!st.paused);
  ui.thinking.hidden = !st.busy;
  if (st.busy) {
    ui.thinkingText.textContent = st.busy.kind === "ask" ? "Answering your question…" : `Looking · ${st.busy.reason}`;
  }

  ui.btnPause.disabled = !ready;
  ui.btnAnalyze.disabled = !ready || !!st.busy;
  $("span", ui.btnPause).textContent = st.paused ? "Resume" : "Pause";
  $("#pause-icon").setAttribute("d", st.paused ? "M8 5v14l11-7z" : "M8 5h3v14H8zM13 5h3v14h-3z");
  ui.btnPause.classList.toggle("primary", !st.paused);

  if (document.activeElement !== ui.heartbeat) {
    ui.heartbeat.value = st.heartbeat_s;
    ui.heartbeatVal.textContent = `${Math.round(st.heartbeat_s)} s`;
  }
  for (const b of ui.detail.querySelectorAll("button")) {
    b.setAttribute("aria-checked", String(Number(b.dataset.side) === st.image_side));
  }
}

// ------------------------------------------------------------------ scene
function fmtTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}
function ago(ts) {
  const s = Math.max(0, Math.round(Date.now() / 1000 - ts));
  return s < 60 ? `${s} s ago` : `${Math.floor(s / 60)} min ago`;
}
function agoSpan(ts) {
  const s = el("span", null, ago(ts));
  s.dataset.ago = ts;
  return s;
}

function renderScene() {
  const sc = S.scene;
  ui.assessment.hidden = !sc;
  ui.nowEmpty.hidden = !!sc;
  ui.nowBody.hidden = !sc;
  if (!sc) { clearOverlayLayers(); return; }
  if (sc.frame_size) setAspect(sc.frame_size[0], sc.frame_size[1]);

  ui.assessmentText.textContent = sc.summary;
  ui.assessmentMeta.replaceChildren(`${sc.latency_s.toFixed(1)} s · ${sc.reason} · `, agoSpan(sc.ts));

  ui.nowSummary.textContent = sc.summary;
  ui.nowChange.replaceChildren();
  const first = sc.reason === "first look";
  ui.nowChange.classList.toggle("none", !sc.changes);
  ui.nowChange.append(el("span", "lbl", sc.changes ? "What changed" : "Changes"),
    sc.changes || (first ? "First look, so nothing to compare yet." : "Nothing notable since the previous look."));

  ui.nowObs.replaceChildren(...sc.observations.map((o, i) => {
    const li = el("li");
    li.dataset.i = i;
    li.append(el("span", `num${o.box ? "" : " free"}`, String(i + 1)), el("span", null, o.text));
    linkHover(li, i);
    return li;
  }));
  if (!sc.observations.length) ui.nowObs.append(el("li", null, "The model didn't list separate observations."));

  const facts = [["Model", sc.model], ["Latency", `${sc.latency_s.toFixed(2)} s · ${sc.tokens} tokens`],
    ["Trigger", sc.reason], ["Detail", `${S.status?.image_side ?? "?"} px`]];
  ui.nowFacts.replaceChildren(...facts.flatMap(([k, v]) => [el("dt", null, k), el("dd", null, v)]));
  const updated = el("dd");
  updated.append(agoSpan(sc.ts));
  ui.nowFacts.append(el("dt", null, "Updated"), updated);
  ui.nowRaw.textContent = sc.raw + (sc.parsed ? "" : "\n\n(Could not read this as JSON; showing it as text.)");

  renderBubbles();
}

// A description still being written: show the summary and each bubble as soon as it exists.
function renderPartial() {
  const p = S.partial;
  if (p.frame_size) setAspect(p.frame_size[0], p.frame_size[1]);
  ui.assessment.hidden = false;
  ui.assessmentText.textContent = p.summary;
  ui.assessmentMeta.textContent = `writing… ${p.elapsed_s.toFixed(1)} s`;
  if (p.observations.length) renderBubbles();
}

// Until the new description has its first observation, keep showing the previous bubbles.
function overlayScene() {
  return S.partial && S.partial.observations.length ? S.partial : S.scene;
}

function clearOverlayLayers() {
  ui.bubbles.replaceChildren();
  ui.loose.replaceChildren();
  ui.leaders.replaceChildren();
  ui.regions.replaceChildren();
}

let shownBubbles = new Set(); // texts already on screen: don't replay their entrance animation

function makeBubble(o) {
  const b = el("div", `bubble glass${shownBubbles.has(o.text) ? " settled" : ""}`);
  b.dataset.i = o.i;
  b.append(el("span", "num", String(o.i + 1)), el("span", null, o.text));
  linkHover(b, o.i);
  return b;
}

function linkHover(node, i) {
  node.addEventListener("mouseenter", () => setHot(i, true));
  node.addEventListener("mouseleave", () => setHot(i, false));
}
function setHot(i, on) {
  document.querySelectorAll(`[data-i="${i}"]`).forEach((n) => n.classList.toggle("hot", on));
}

function stageRect(node) {
  const s = ui.stage.getBoundingClientRect();
  const r = node.getBoundingClientRect();
  return { x: r.left - s.left, y: r.top - s.top, w: r.width, h: r.height };
}

function renderBubbles() {
  clearOverlayLayers();
  const sc = overlayScene();
  if (!sc || !prefs.bubbles) { shownBubbles = new Set(); return; }
  const W = ui.stage.clientWidth, H = ui.stage.clientHeight;
  const obs = sc.observations.map((o, i) => ({ ...o, i }));

  obs.filter((o) => !o.box).forEach((o, k) => {
    const b = makeBubble(o);
    b.style.animationDelay = `${k * 70}ms`;
    ui.loose.append(b);
  });

  const placed = [...ui.loose.children].map(stageRect);
  if (!ui.assessment.hidden) placed.push(stageRect(ui.assessment));
  placed.push({ x: W - 180, y: 0, w: 180, h: 80 }); // clock + thinking chip

  const boxed = obs.filter((o) => o.box).map((o) => {
    let [x1, y1, x2, y2] = o.box;
    if (prefs.mirror) [x1, x2] = [1 - x2, 1 - x1];
    return { ...o, r: { x1: x1 * W, y1: y1 * H, x2: x2 * W, y2: y2 * H } };
  }).sort((a, b) => a.r.y1 - b.r.y1);

  boxed.forEach((o, k) => {
    const { r } = o;
    const region = el("div", "region");
    region.dataset.i = o.i;
    Object.assign(region.style, { left: `${r.x1}px`, top: `${r.y1}px`, width: `${r.x2 - r.x1}px`, height: `${r.y2 - r.y1}px` });
    ui.regions.append(region);

    const b = makeBubble(o);
    b.classList.add("measuring");
    ui.bubbles.append(b);
    const bw = b.offsetWidth, bh = b.offsetHeight;
    const spot = placeBubble(r, bw, bh, W, H, placed);
    placed.push({ x: spot.x, y: spot.y, w: bw, h: bh });
    // Lock the measured width: near the right edge an absolutely positioned box would
    // otherwise shrink, wrap to more lines, and grow into whatever is below it.
    Object.assign(b.style, { left: `${spot.x}px`, top: `${spot.y}px`, width: `${Math.ceil(bw) + 1}px`,
                             animationDelay: `${(k + 1) * 70}ms` });
    b.classList.remove("measuring");
    drawLeader(spot, bw, bh, r);
  });
  shownBubbles = new Set(obs.map((o) => o.text));
}

function placeBubble(r, bw, bh, W, H, placed) {
  // Candidates: beside the region (roomier side first), then centred on it; for each, scan
  // outward from the region's top edge. Take the first free spot, else the least-covered one.
  const gap = 30, pad = 14, top = 48, maxX = W - bw - pad, maxY = H - bh - pad;
  if (maxX < pad || maxY < top) return { x: pad, y: top };
  const besideLeft = clamp(r.x1 - gap - bw, pad, maxX), besideRight = clamp(r.x2 + gap, pad, maxX);
  const xs = [...(r.x1 > W - r.x2 ? [besideLeft, besideRight] : [besideRight, besideLeft]),
    clamp((r.x1 + r.x2 - bw) / 2, pad, maxX)];
  const y0 = clamp(r.y1, top, maxY);
  const ys = [y0];
  for (let d = 10; d <= H; d += 10) {
    if (y0 + d <= maxY) ys.push(y0 + d);
    if (y0 - d >= top) ys.push(y0 - d);
  }
  let best = null, bestCost = Infinity;
  for (const y of ys) {
    for (const x of xs) {
      const rect = { x, y, w: bw, h: bh };
      const cost = placed.reduce((sum, p) => sum + overlapArea(rect, p), 0);
      if (cost === 0) return { x, y };
      if (cost < bestCost) [best, bestCost] = [{ x, y }, cost];
    }
  }
  return best;
}

function overlapArea(a, b, m = 6) {
  const w = Math.min(a.x + a.w, b.x + b.w + m) - Math.max(a.x, b.x - m);
  const h = Math.min(a.y + a.h, b.y + b.h + m) - Math.max(a.y, b.y - m);
  return w > 0 && h > 0 ? w * h : 0;
}

function drawLeader(spot, bw, bh, r) {
  const midY = spot.y + bh / 2;
  const bubbleLeftOfBox = spot.x + bw <= r.x1 + 4;
  const bubbleRightOfBox = spot.x >= r.x2 - 4;
  let sx, ex;
  if (bubbleLeftOfBox) { sx = spot.x + bw; ex = r.x1; }
  else if (bubbleRightOfBox) { sx = spot.x; ex = r.x2; }
  else return; // bubble sits over its region; no leader needed
  const ey = clamp(midY, r.y1 + 6, r.y2 - 6);
  const line = document.createElementNS(SVG_NS, "line");
  line.setAttribute("x1", sx); line.setAttribute("y1", midY);
  line.setAttribute("x2", ex); line.setAttribute("y2", ey);
  const dot = document.createElementNS(SVG_NS, "circle");
  dot.setAttribute("cx", ex); dot.setAttribute("cy", ey); dot.setAttribute("r", 5);
  ui.leaders.append(line, dot);
}

new ResizeObserver(() => renderBubbles()).observe(ui.stage);

// ------------------------------------------------------------------ timeline
function renderTimeline(fresh = false) {
  ui.timelineEmpty.hidden = S.events.length > 0;
  ui.timelineCount.textContent = String(S.events.length);
  ui.timeline.replaceChildren(...[...S.events].reverse().map((ev) => {
    const li = el("li", ev.kind);
    const head = el("div", "tl-head");
    head.append(el("span", null, fmtTime(ev.ts)), el("span", "tag", ev.kind === "start" ? "Started" : "Change"));
    const body = el("div", "tl-body");
    if (ev.thumb) {
      const img = el("img");
      img.src = ev.thumb;
      img.alt = "";
      body.append(img);
    }
    body.append(el("p", null, ev.text));
    li.append(head, body);
    return li;
  }));
  if (fresh && prefs.tab !== "timeline" && S.events.at(-1)?.kind === "change") {
    toast(`Change: ${S.events.at(-1).text}`);
  }
}

// ------------------------------------------------------------------ ask
function renderThread() {
  const items = [];
  for (const a of S.answers) {
    items.push(el("div", "msg q", a.question));
    const msg = el("div", "msg a");
    if (a.thumb) {
      const img = el("img");
      img.src = a.thumb;
      img.alt = "Frame the answer is about";
      msg.append(img);
    }
    msg.append(el("span", null, a.answer), el("span", "meta", `${fmtTime(a.answered_ts)} · ${a.latency_s.toFixed(1)} s`));
    items.push(msg);
  }
  for (const q of S.pending.values()) {
    items.push(el("div", "msg q", q.question));
    const msg = el("div", "msg a");
    const dots = el("span", "typing");
    dots.append(el("i"), el("i"), el("i"));
    msg.append(dots);
    items.push(msg);
  }
  ui.threadEmpty.hidden = items.length > 0;
  ui.thread.replaceChildren(ui.threadEmpty, ...items);
  ui.thread.scrollTop = ui.thread.scrollHeight;
}

async function sendQuestion(text) {
  text = text.trim();
  if (!text) return;
  try {
    await api("/api/ask", { question: text });
    ui.question.value = "";
    autoGrow();
    selectTab("ask");
  } catch (e) {
    toast(e.message === "Model is not ready yet" ? "The model is still loading. Try again in a moment." : `Couldn't send: ${e.message}`);
  }
}

function autoGrow() {
  ui.question.style.height = "auto";
  ui.question.style.height = `${Math.min(ui.question.scrollHeight, 120)}px`;
}

ui.composer.addEventListener("submit", (e) => { e.preventDefault(); sendQuestion(ui.question.value); });
ui.question.addEventListener("input", autoGrow);
ui.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuestion(ui.question.value); }
});
ui.chips.addEventListener("click", (e) => {
  if (e.target instanceof HTMLButtonElement) sendQuestion(e.target.textContent);
});

// ------------------------------------------------------------------ controls
async function control(body) {
  try { await api("/api/control", body); } catch (e) { toast(`Couldn't update: ${e.message}`); }
}
ui.btnPause.addEventListener("click", () => control({ action: S.status?.paused ? "resume" : "pause" }));
ui.btnAnalyze.addEventListener("click", () => control({ action: "analyze" }));
ui.heartbeat.addEventListener("input", () => { ui.heartbeatVal.textContent = `${ui.heartbeat.value} s`; });
ui.heartbeat.addEventListener("change", () => control({ heartbeat_s: Number(ui.heartbeat.value) }));
ui.detail.addEventListener("click", (e) => {
  const side = e.target.closest("button")?.dataset.side;
  if (side) control({ image_side: Number(side) });
});

function applyToggles() {
  ui.optBubbles.checked = prefs.bubbles;
  ui.optMirror.checked = prefs.mirror;
  ui.stage.classList.toggle("mirror", prefs.mirror);
  document.body.classList.toggle("mirror", prefs.mirror);
  renderBubbles();
}
ui.optBubbles.addEventListener("change", () => { prefs.bubbles = ui.optBubbles.checked; savePrefs(); applyToggles(); });
ui.optMirror.addEventListener("change", () => { prefs.mirror = ui.optMirror.checked; savePrefs(); applyToggles(); });

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea") || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "a" && !ui.btnAnalyze.disabled) ui.btnAnalyze.click();
  if (e.key === "p" && !ui.btnPause.disabled) ui.btnPause.click();
});

// ------------------------------------------------------------------ tabs
function selectTab(name) {
  prefs.tab = name;
  savePrefs();
  for (const t of document.querySelectorAll("[role=tab]")) t.setAttribute("aria-selected", String(t.dataset.tab === name));
  for (const p of document.querySelectorAll("[data-panel]")) p.hidden = p.dataset.panel !== name;
  if (name === "ask") ui.thread.scrollTop = ui.thread.scrollHeight;
}
document.querySelector(".tabs").addEventListener("click", (e) => {
  const tab = e.target.closest("[role=tab]");
  if (tab) selectTab(tab.dataset.tab);
});

// ------------------------------------------------------------------ toast + ticker
let toastTimer;
function toast(text) {
  ui.toast.textContent = text;
  ui.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { ui.toast.hidden = true; }, 3500);
}

function tick() {
  ui.clock.textContent = new Date().toLocaleTimeString([], { hour12: false });
  document.querySelectorAll("[data-ago]").forEach((n) => { n.textContent = ago(Number(n.dataset.ago)); });
  const st = S.status;
  if (S.scene && st && !st.paused) {
    const p = clamp((Date.now() / 1000 - S.scene.ts) / st.heartbeat_s, 0, 1);
    ui.meter.style.left = `${p * 100}%`;
  }
}
setInterval(tick, 1000);

// ------------------------------------------------------------------ boot
const hashTab = location.hash.slice(1);
selectTab(["now", "timeline", "ask"].includes(hashTab) ? hashTab : prefs.tab);
applyToggles();
renderStatus();
tick();
connect();
