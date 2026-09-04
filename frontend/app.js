// ─── DOM refs ────────────────────────────────────────────────────────────────
const video          = document.querySelector("#video");
const overlay        = document.querySelector("#overlay");
const demoCanvas     = document.querySelector("#demoCanvas");
const chartCanvas    = document.querySelector("#kinematicsChart");
const sourceStatus   = document.querySelector("#sourceStatus");
const backendStatus  = document.querySelector("#backendStatus");
const modelStatus    = document.querySelector("#modelStatus");
const fpsStatus      = document.querySelector("#fpsStatus");
const scoreValue     = document.querySelector("#scoreValue");
const scoreMeter     = document.querySelector("#scoreMeter");
const objectCount    = document.querySelector("#objectCount");
const dominantAction = document.querySelector("#dominantAction");
const latencyValue   = document.querySelector("#latencyValue");
const objectList     = document.querySelector("#objectList");
const actionTimeline = document.querySelector("#actionTimeline");
const elapsedValue   = document.querySelector("#elapsedValue");
const chatLog        = document.querySelector("#chatLog");
const chatInput      = document.querySelector("#chatInput");
const chatSendBtn    = document.querySelector("#chatSendBtn");
const demoBtn        = document.querySelector("#demoBtn");
const cameraBtn      = document.querySelector("#cameraBtn");
const videoFile      = document.querySelector("#videoFile");
const playPauseBtn   = document.querySelector("#playPauseBtn");
const snapshotBtn    = document.querySelector("#snapshotBtn");
const SHOW_TIP_TRAJECTORY = false;

// ─── Config ──────────────────────────────────────────────────────────────────
const API_BASE = localStorage.getItem("visionApiBase") || "http://localhost:8000";
const colors   = ["#2ac779", "#4db5d8", "#f0c84b", "#d36adf", "#ef6961"];
const ACTION_INTERVAL_MS = Number(localStorage.getItem("actionIntervalMs") || 2000);

// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  source: "demo",
  running: true,
  startedAt: performance.now(),
  frameTimes: [],
  tracks: [],
  actions: [],
  history: new Map(),
  performance: 0,
  latency: 0,
  dominantAction: "Idle",
  stream: null,
  backendOnline: false,
  lastAnalysis: null,
  failureNotified: false,
};

// ─── ModelAdapter ────────────────────────────────────────────────────────────
class ModelAdapter {
  constructor(apiBase) {
    this.apiBase = apiBase.replace(/\/$/, "");
  }

  async health() {
    const response = await fetchWithTimeout(`${this.apiBase}/health`, { cache: "no-store" }, 1500);
    if (!response.ok) throw new Error(`health ${response.status}`);
    return response.json();
  }

  async detectFrame({ timestamp, width, height }) {
    if (state.source === "demo" || !state.backendOnline) {
      return this.demoAnalysis({ timestamp, width, height });
    }

    const frame = await captureCurrentFrame();
    const form  = new FormData();
    form.append("frame",          frame, `frame-${Math.round(timestamp)}.jpg`);
    form.append("source",         state.source);
    form.append("timestamp_ms",   String(Math.round(timestamp)));
    form.append("display_width",  String(Math.round(width)));
    form.append("display_height", String(Math.round(height)));

    const started  = performance.now();
    const response = await fetchWithTimeout(`${this.apiBase}/api/detect-frame`, {
      method: "POST",
      body:   form,
    }, 15000);
    if (!response.ok) throw new Error(`detect ${response.status}: ${await response.text()}`);

    const payload  = await response.json();
    const imgW     = payload.image_width  || width;
    const imgH     = payload.image_height || height;
    return normalizeAnalysis(payload, Math.round(performance.now() - started), imgW, imgH);
  }

  async analyzeAction({ timestamp }) {
    if (state.source === "demo" || !state.backendOnline) return null;

    const frame = await captureCurrentFrame();
    const form  = new FormData();
    form.append("frame",        frame, `action-${Math.round(timestamp)}.jpg`);
    form.append("source",       state.source);
    form.append("timestamp_ms", String(Math.round(timestamp)));

    const started  = performance.now();
    const response = await fetchWithTimeout(`${this.apiBase}/api/action-frame`, {
      method: "POST",
      body:   form,
    }, 45000);
    if (!response.ok) throw new Error(`action ${response.status}: ${await response.text()}`);

    const payload = await response.json();
    return normalizeAnalysis(payload, Math.round(performance.now() - started), 0, 0);
  }

  async chat(question, context) {
    if (!state.backendOnline) return localAnswerQuestion(question);
    const response = await fetchWithTimeout(`${this.apiBase}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, context }),
    }, 120000);
    if (!response.ok) throw new Error(`chat ${response.status}: ${await response.text()}`);
    const payload = await response.json();
    return payload.answer || localAnswerQuestion(question);
  }

  async demoAnalysis({ timestamp, width, height }) {
    const started = performance.now();
    const t = timestamp / 1000;
    const tracks = [
      makeTrack(1, "needle_driver_c", 0.5 + Math.sin(t * 0.75) * 0.25, 0.52, 0.15, 0.34, t, width, height),
      makeTrack(2, "scissors_s", 0.32 + Math.cos(t * 1.2) * 0.16, 0.35 + Math.sin(t) * 0.09, 0.09, 0.11, t * 1.4, width, height),
      makeTrack(3, "needle", 0.66 + Math.sin(t * 0.55) * 0.12, 0.72 + Math.cos(t * 0.8) * 0.07, 0.12, 0.1, t * 0.9, width, height),
    ];
    const actionNames = ["Approach", "Inspect", "Manipulate", "Transfer", "Idle"];
    const phase = Math.floor(t / 4) % actionNames.length;
    const actions = actionNames.map((name, index) => ({
      name,
      confidence: index === phase
        ? 0.72 + Math.sin(t) * 0.12
        : Math.max(0.05, 0.34 - Math.abs(index - phase) * 0.09),
      color: colors[index],
    }));
    const performanceScore = Math.round(
      clamp(78 + Math.sin(t * 0.4) * 12 - Math.abs(tracks[0].acceleration) * 2.4 + actions[phase].confidence * 7, 0, 100),
    );
    await sleep(8);
    return {
      tracks,
      actions,
      dominantAction: actionNames[phase],
      performanceScore,
      latency: Math.round(performance.now() - started),
      source: "demo",
    };
  }
}

const adapter = new ModelAdapter(API_BASE);

// ─── Utilities ───────────────────────────────────────────────────────────────
async function fetchWithTimeout(url, options = {}, timeoutMs = 5000) {
  const controller = new AbortController();
  const id = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    window.clearTimeout(id);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

// ─── Normalisation helpers ────────────────────────────────────────────────────
function normalizeAnalysis(payload, measuredLatency, imgW, imgH) {
  const rect     = overlay.getBoundingClientRect();
  const displayW = rect.width;
  const displayH = rect.height;
  // Scale factors: convert from image pixels → canvas CSS pixels.
  // If imgW/imgH not provided (demo path), scaleX/Y stay 1.
  const scaleX   = imgW && imgW > 1 ? displayW / imgW : 1;
  const scaleY   = imgH && imgH > 1 ? displayH / imgH : 1;

  const tracks = (payload.tracks || []).map((track, index) => {
    const bbox     = normalizeBox(track.bbox || track.box || [0, 0, 1, 1], displayW, displayH, scaleX, scaleY);
    const centroid = scalePt(track.centroid, scaleX, scaleY) || {
      x: bbox.x + bbox.w / 2,
      y: bbox.y + bbox.h / 2,
    };

    const rawTip = track.tip ?? null;
    const tip    = rawTip && Number.isFinite(rawTip.x) && Number.isFinite(rawTip.y)
      ? { x: rawTip.x * scaleX, y: rawTip.y * scaleY }
      : null;

    const tipTrail = Array.isArray(track.tip_trail)
      ? track.tip_trail
          .filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y))
          .map((p) => ({ x: p.x * scaleX, y: p.y * scaleY }))
      : [];

    return {
      id:           Number(track.id ?? track.track_id ?? index + 1),
      label:        String(track.label ?? track.class_name ?? track.class_id ?? "object"),
      confidence:   Number(track.confidence ?? track.conf ?? 0),
      bbox,
      centroid,
      speed:        Number(track.speed ?? 0),
      acceleration: Number(track.acceleration ?? 0),
      heading:      String(Math.round(Number(track.heading ?? 0))),
      tip,
      tipTrail,
    };
  });

  const actions = (payload.actions || [{ name: "Pending action model", confidence: 1 }])
    .map((action, index) => ({
      name:       String(action.name ?? action.label ?? "Action"),
      confidence: Number(action.confidence ?? action.score ?? 0),
      color:      action.color || colors[index % colors.length],
    }));

  return {
    tracks,
    actions,
    dominantAction:   payload.dominantAction || payload.dominant_action || actions[0]?.name || "Unknown",
    performanceScore: Number(payload.performanceScore ?? payload.performance_score ?? estimatePerformance(tracks, actions)),
    latency:          Number(payload.latency ?? payload.latency_ms ?? measuredLatency),
    source:           payload.source || "backend",
  };
}

function normalizeBox(box, displayW, displayH, scaleX = 1, scaleY = 1) {
  const values = Array.isArray(box)
    ? box
    : [box.x, box.y, box.w ?? box.width, box.h ?? box.height];

  let [a, b, c, d] = values.map(Number);

  const looksLikePixels = Math.max(Math.abs(a), Math.abs(b), Math.abs(c), Math.abs(d)) > 1.5;
  if (looksLikePixels) {
    if (c > a && d > b) {
      // xyxy image pixels → scale to CSS pixels
      return { x: a * scaleX, y: b * scaleY, w: (c - a) * scaleX, h: (d - b) * scaleY };
    }
    // xywh image pixels → scale to CSS pixels
    return { x: a * scaleX, y: b * scaleY, w: c * scaleX, h: d * scaleY };
  }

  // normalised 0-1 → scale to display pixels (scaleX/Y not needed)
  return { x: a * displayW, y: b * displayH, w: c * displayW, h: d * displayH };
}

/**
 * scalePt – scale a {x,y} point from image pixels to CSS pixels.
 * Returns null if the input is missing or invalid.
 */
function scalePt(pt, scaleX = 1, scaleY = 1) {
  if (!pt || !Number.isFinite(pt.x) || !Number.isFinite(pt.y)) return null;
  return { x: pt.x * scaleX, y: pt.y * scaleY };
}

function estimatePerformance(tracks, actions) {
  const meanConf = tracks.length
    ? tracks.reduce((s, t) => s + t.confidence, 0) / tracks.length
    : 0.5;
  const actionConf = actions[0]?.confidence ?? 0.5;
  return Math.round(clamp(35 + meanConf * 40 + actionConf * 25, 0, 100));
}

// ─── Frame capture ────────────────────────────────────────────────────────────
async function captureCurrentFrame() {
  const rect = overlay.getBoundingClientRect();
  const w    = Math.max(1, Math.round(rect.width));
  const h    = Math.max(1, Math.round(rect.height));

  const canvas = document.createElement("canvas");
  canvas.width  = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");

  if (state.source === "demo") {
    ctx.drawImage(demoCanvas, 0, 0, w, h);
  } else {
    // Guard: video must have actual pixel data (readyState >= 2 = HAVE_CURRENT_DATA)
    if (video.readyState < 2 || video.videoWidth === 0) {
      throw new Error("Video not ready yet");
    }
    ctx.drawImage(video, 0, 0, w, h);
  }

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Could not encode frame"))),
      "image/jpeg",
      0.82,
    );
  });
}

// ─── Demo scene helpers ───────────────────────────────────────────────────────
/**
 * makeTrack – all coordinates are in LOGICAL display pixels
 * (cx/cy/w/h are fractions of width/height, multiplied here).
 */
function makeTrack(id, label, cx, cy, w, h, phase, width, height) {
  const px    = cx * width;
  const py    = cy * height;
  const speed = 35 + Math.abs(Math.sin(phase)) * 105 + id * 8;
  const accel = Math.cos(phase * 0.8) * 18;

  // Simulate a tip at the leading edge of the bounding box.
  // The tip oscillates slightly within the box to mimic find_tip_in_roi output.
  const bx = (cx - w / 2) * width;
  const by = (cy - h / 2) * height;
  const bw = w * width;
  const bh = h * height;
  const tip = {
    x: bx + bw * (0.5 + Math.sin(phase * 1.3) * 0.22),
    y: by + bh * (0.18 + Math.abs(Math.cos(phase * 0.9)) * 0.14),
  };

  return {
    id,
    label,
    confidence: clamp(0.78 + Math.sin(phase) * 0.12, 0, 0.99),
    bbox: { x: bx, y: by, w: bw, h: bh },
    centroid:     { x: px, y: py },
    speed,
    acceleration: accel,
    heading:      ((Math.sin(phase) * 80 + 360) % 360).toFixed(0),
    tip,
    tipTrail:     [],  // trail is built live in updateHistory for demo
  };
}

// ─── Canvas sizing ────────────────────────────────────────────────────────────
/**
 * resizeCanvases – keeps backing-store at physical pixels (DPR-aware) while
 * the 2-D context transform is scaled so all drawing uses CSS logical pixels.
 */
function resizeCanvases() {
  const rect = overlay.getBoundingClientRect();
  const dpr  = window.devicePixelRatio || 1;
  const w    = Math.max(1, Math.round(rect.width));
  const h    = Math.max(1, Math.round(rect.height));
  for (const canvas of [overlay, demoCanvas]) {
    // Only resize if needed to avoid clearing unnecessarily every frame
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width  = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    canvas.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}

// ─── Backend health check ─────────────────────────────────────────────────────
let _healthFailStreak = 0;

async function checkBackend() {
  try {
    const response = await fetchWithTimeout(
      `${API_BASE}/health`,
      { cache: "no-store" },
      5000,   // 5 s — generous enough that a busy backend still responds
    );
    if (!response.ok) throw new Error(`health ${response.status}`);
    _healthFailStreak = 0;
    if (!state.backendOnline) {
      state.backendOnline = true;
      state.failureNotified = false;
      backendStatus.textContent = "Models online";
      backendStatus.classList.add("ok");
    }
  } catch {
    _healthFailStreak++;
    if (_healthFailStreak >= 3) {
      state.backendOnline = false;
      backendStatus.textContent = "Demo backend";
      backendStatus.classList.remove("ok");
    }
  }
}

// ─── Source management ────────────────────────────────────────────────────────
async function startDemo() {
  stopCamera();
  setSource("demo");
  video.pause();
  video.removeAttribute("src");
  video.style.display   = "none";
  demoCanvas.style.display = "block";
  addMessage("analyst", "Demo stream started. Use Camera or Video to send sampled frames to the backend.");
}

async function startCamera() {
  try {
    stopCamera();
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    state.stream = stream;
    video.srcObject = stream;
    video.style.display   = "block";
    demoCanvas.style.display = "none";
    await video.play();
    setSource("camera");
    addMessage("analyst", "Camera stream connected. Sampled frames will be sent to the backend when it is online.");
  } catch (error) {
    addMessage("analyst", `Camera could not start: ${error.message}`);
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
}

function setSource(source) {
  state.source      = source;
  state.lastAnalysis = null;
  sourceStatus.textContent = source === "camera" ? "Camera stream"
    : source === "file" ? "Video file" : "Demo stream";
  demoBtn.classList.toggle("active", source === "demo");
  cameraBtn.classList.toggle("active", source === "camera");
}

function loadVideoFile(file) {
  stopCamera();
  state.tracks      = [];
  state.actions     = [];
  state.lastAnalysis = null;

  const url = URL.createObjectURL(file);
  video.srcObject = null;
  video.src       = url;
  video.loop      = true;
  video.muted     = true;

  demoCanvas.style.display = "none";
  video.style.display      = "block";

  // Set source AFTER clearing state so analysisLoop doesn't race with stale demo data
  setSource("file");

  video.play().catch((err) => {
    addMessage("analyst", `Video play error: ${err.message}`);
  });

  addMessage("analyst", `Loaded ${file.name}. Detection and action frames are sent independently to ${API_BASE}.`);
}

// ─── Drawing ──────────────────────────────────────────────────────────────────
function drawDemoScene(now) {
  const ctx  = demoCanvas.getContext("2d");
  const rect = demoCanvas.getBoundingClientRect();
  const w    = rect.width;
  const h    = rect.height;
  ctx.clearRect(0, 0, w, h);

  // Only draw the animated grid + boxes in demo mode
  if (state.source !== "demo") return;

  ctx.fillStyle = "#080b0c";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "#1e2a2e";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 42) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = 0; y < h; y += 42) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  const t = now / 1000;
  state.tracks.forEach((track, index) => {
    const { x, y, w: bw, h: bh } = track.bbox;
    ctx.fillStyle   = `${colors[index]}33`;
    ctx.strokeStyle = colors[index];
    ctx.lineWidth   = 3;
    ctx.beginPath();
    ctx.roundRect(x, y, bw, bh, 8);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = colors[index];
    ctx.beginPath();
    ctx.arc(x + bw / 2, y + bh / 2, 5 + Math.sin(t * 4 + index) * 1.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawOverlay() {
  const ctx    = overlay.getContext("2d");
  const rect   = overlay.getBoundingClientRect();
  const width  = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);

  state.tracks.forEach((track, index) => {
    const color = colors[index % colors.length];
    const x = clamp(track.bbox.x, 0, width);
    const y = clamp(track.bbox.y, 0, height);
    const w = clamp(track.bbox.w, 1, width  - x);
    const h = clamp(track.bbox.h, 1, height - y);

    // ── Bounding box ──────────────────────────────────────────────────────────
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 8);
    ctx.stroke();

    // ── Label ─────────────────────────────────────────────────────────────────
    ctx.fillStyle = "rgba(0,0,0,0.66)";
    ctx.fillRect(x, Math.max(0, y - 25), Math.max(116, w), 24);
    ctx.fillStyle = color;
    ctx.font      = "13px Inter, sans-serif";
    ctx.fillText(
      `#${track.id} ${track.label} ${(track.confidence * 100).toFixed(0)}%`,
      x + 8,
      Math.max(16, y - 8),
    );

    // ── Motion vector arrow (from centroid) ───────────────────────────────────
    drawVector(ctx, track.centroid.x, track.centroid.y, track.heading, track.speed, color);

    // ── Tip trail (fading polyline) ───────────────────────────────────────────
    const trail = track.tipTrail;
    if (SHOW_TIP_TRAJECTORY && trail && trail.length > 1) {
      for (let i = 1; i < trail.length; i++) {
        const alpha = (i / trail.length);          // 0 = oldest, 1 = newest
        ctx.strokeStyle = color + Math.round(alpha * 200).toString(16).padStart(2, "0");
        ctx.lineWidth   = 1.5;
        ctx.beginPath();
        ctx.moveTo(trail[i - 1].x, trail[i - 1].y);
        ctx.lineTo(trail[i].x,     trail[i].y);
        ctx.stroke();
      }
    }

    // ── Tip circle ────────────────────────────────────────────────────────────
    if (track.tip) {
      const tx = track.tip.x;
      const ty = track.tip.y;
      // Filled circle in the track colour
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(tx, ty, 9, 0, Math.PI * 2);
      ctx.fill();
      // White border so it pops against any background
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth   = 2;
      ctx.stroke();
      // Small crosshair dot at the exact tip point
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(tx, ty, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

function drawVector(ctx, x, y, heading, speed, color) {
  const length = clamp(speed / 5, 12, 36);
  const angle  = (Number(heading) * Math.PI) / 180;
  const x2     = x + Math.cos(angle) * length;
  const y2     = y + Math.sin(angle) * length;
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x2, y2, 3, 0, Math.PI * 2);
  ctx.fill();
}

// ─── History & charts ─────────────────────────────────────────────────────────
function updateHistory(tracks) {
  const now = performance.now();
  for (const track of tracks) {
    if (!state.history.has(track.id)) state.history.set(track.id, []);
    const samples = state.history.get(track.id);
    samples.push({ t: now, speed: track.speed, acceleration: track.acceleration });
    while (samples.length > 90) samples.shift();

    // Build tip trail in demo mode (backend sends its own trail; demo builds here)
    if (state.source === "demo" && track.tip) {
      if (!Array.isArray(track.tipTrail)) track.tipTrail = [];
      track.tipTrail.push({ x: track.tip.x, y: track.tip.y });
      if (track.tipTrail.length > 64) track.tipTrail.shift();
    }
  }
}

function renderObjects() {
  objectCount.textContent = String(state.tracks.length);
  objectList.innerHTML    = "";
  for (const [index, track] of state.tracks.entries()) {
    const card = document.createElement("article");
    card.className = "object-card";
    card.innerHTML = `
      <i class="swatch" style="background:${colors[index % colors.length]}"></i>
      <div>
        <strong>#${track.id} ${track.label}</strong>
        <span>${track.speed.toFixed(1)} px/s · ${track.acceleration.toFixed(1)} px/s² · heading ${track.heading}°</span>
      </div>
      <output>${(track.confidence * 100).toFixed(0)}%</output>
    `;
    objectList.append(card);
  }
}

function renderChart() {
  const ctx    = chartCanvas.getContext("2d");
  const width  = chartCanvas.width;
  const height = chartCanvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101315";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "#283136";
  ctx.lineWidth   = 1;
  for (let y = 30; y < height; y += 40) {
    ctx.beginPath(); ctx.moveTo(36, y); ctx.lineTo(width - 12, y); ctx.stroke();
  }
  ctx.font      = "12px Inter, sans-serif";
  ctx.fillStyle = "#9eaaa7";
  ctx.fillText("speed", 14, 22);
  ctx.fillText("accel", 14, 42);

  for (const [index, track] of state.tracks.entries()) {
    const samples = state.history.get(track.id) || [];
    drawSeries(ctx, samples.map((s) => s.speed),              0, 170, colors[index % colors.length],         width, height);
    drawSeries(ctx, samples.map((s) => s.acceleration + 80), 0, 170, `${colors[index % colors.length]}88`, width, height);
  }
}

function drawSeries(ctx, values, min, max, color, width, height) {
  if (values.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = 38 + (index / 89) * (width - 56);
    const y = height - 18 - ((clamp(value, min, max) - min) / (max - min)) * (height - 46);
    if (index === 0) ctx.moveTo(x, y);
    else             ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderActions() {
  dominantAction.textContent = state.dominantAction;
  actionTimeline.innerHTML   = "";
  for (const action of state.actions) {
    const row = document.createElement("div");
    row.className = "timeline-row";
    row.innerHTML = `
      <span>${action.name}</span>
      <div class="timeline-track">
        <i class="timeline-fill" style="width:${Math.round(action.confidence * 100)}%; background:${action.color}"></i>
      </div>
      <strong>${Math.round(action.confidence * 100)}%</strong>
    `;
    actionTimeline.append(row);
  }
}

function updatePerformance() {
  scoreValue.textContent = `${state.performance}`;
  scoreMeter.value       = state.performance;
  latencyValue.textContent = `${state.latency} ms`;
  const usingBackend = state.backendOnline && state.source !== "demo";
  modelStatus.textContent = usingBackend ? "Real models" : "Demo models";
  modelStatus.classList.toggle("ok", state.latency <= 300 || state.source === "demo");
}

function updateClock(now) {
  const elapsed  = Math.floor((now - state.startedAt) / 1000);
  const minutes  = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const seconds  = String(elapsed % 60).padStart(2, "0");
  elapsedValue.textContent = `${minutes}:${seconds}`;
}

function updateFps(now) {
  state.frameTimes.push(now);
  while (state.frameTimes.length && now - state.frameTimes[0] > 1000) state.frameTimes.shift();
  fpsStatus.textContent = `${state.frameTimes.length} fps`;
}

// ─── Chat ─────────────────────────────────────────────────────────────────────
function addMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  chatLog.append(message);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function currentContext() {
  return {
    tracks:           state.tracks,
    actions:          state.actions,
    performanceScore: state.performance,
    latencyMs:        state.latency,
    dominantAction:   state.dominantAction,
    source:           state.source,
  };
}

function localAnswerQuestion(question) {
  const q       = question.toLowerCase();
  const fastest = [...state.tracks].sort((a, b) => b.speed - a.speed)[0];
  if (q.includes("performance") || q.includes("rate") || q.includes("score")) {
    return `Current performance is ${state.performance}/100 with ${state.latency} ms model latency. The main driver is ${state.dominantAction.toLowerCase()} confidence plus object motion stability.`;
  }
  if (q.includes("action")) {
    return `The dominant action is ${state.dominantAction}. The action segmentation model endpoint is currently a placeholder until your model is connected.`;
  }
  if (q.includes("speed") || q.includes("kinematic") || q.includes("motion")) {
    return fastest
      ? `Fastest object is #${fastest.id} ${fastest.label} at ${fastest.speed.toFixed(1)} px/s with ${fastest.acceleration.toFixed(1)} px/s² acceleration.`
      : "No tracked objects are available yet.";
  }
  if (q.includes("object") || q.includes("scene")) {
    return `I see ${state.tracks.length} tracked objects: ${state.tracks.map((t) => `#${t.id} ${t.label}`).join(", ")}. Motion vectors and bounding boxes are drawn on the live feed.`;
  }
  return `Scene summary: ${state.tracks.length} objects tracked, ${state.dominantAction.toLowerCase()} is dominant, performance is ${state.performance}/100, latency is ${state.latency} ms.`;
}

async function sendChat() {
  const question = chatInput.value.trim();
  if (!question) return;
  addMessage("user", question);
  chatInput.value = "";
  try {
    addMessage("analyst", await adapter.chat(question, currentContext()));
  } catch (error) {
    addMessage("analyst", `${localAnswerQuestion(question)} LLM backend error: ${error.message}`);
  }
}

// ─── Render loop (runs at ~60 fps, sync, draws only) ─────────────────────────
function renderLoop(now) {
  resizeCanvases();
  if (state.running) {
    drawDemoScene(now);
    drawOverlay();
    renderObjects();
    renderChart();
    renderActions();
    updatePerformance();
    updateClock(now);
    updateFps(now);
  }
  requestAnimationFrame(renderLoop);
}

// ─── Detection loop (one fast request at a time, never waits for action) ─────
let _detectionFailStreak = 0;

async function detectionLoop() {
  if (!state.running) {
    setTimeout(detectionLoop, 100);
    return;
  }

  const rect = overlay.getBoundingClientRect();
  const now  = performance.now();
  try {
    const analysis = await adapter.detectFrame({
      timestamp: now,
      width:     rect.width,
      height:    rect.height,
    });
    state.tracks         = analysis.tracks;
    state.latency        = analysis.latency;
    state.performance    = estimatePerformance(state.tracks, state.actions);
    if (state.source === "demo") {
      state.actions        = analysis.actions;
      state.performance    = analysis.performanceScore;
      state.dominantAction = analysis.dominantAction;
    }
    updateHistory(state.tracks);
    _detectionFailStreak  = 0;
    state.failureNotified = false;
  } catch (error) {
    if (error.message === "Video not ready yet") {
      setTimeout(detectionLoop, 200);
      return;
    }
    _detectionFailStreak++;
    if (_detectionFailStreak >= 3) {
      state.backendOnline = false;
      backendStatus.textContent = "Backend error";
      backendStatus.classList.remove("ok");
      if (!state.failureNotified) {
        addMessage("analyst", `failed attempts: ${error.message}`);
        state.failureNotified = true;
      }
    }
  }

  setTimeout(detectionLoop, state.source === "demo" ? 16 : 0);
}

// ─── Action loop (slow and independent; never blocks tracking updates) ───────
async function actionLoop() {
  if (!state.running || state.source === "demo" || !state.backendOnline) {
    setTimeout(actionLoop, 250);
    return;
  }

  try {
    const analysis = await adapter.analyzeAction({ timestamp: performance.now() });
    if (analysis) {
      state.actions        = analysis.actions;
      state.dominantAction = analysis.dominantAction;
      state.performance    = estimatePerformance(state.tracks, state.actions);
    }
  } catch (error) {
    if (error.message !== "Video not ready yet") {
      console.warn("Action analysis failed:", error);
    }
  }

  setTimeout(actionLoop, ACTION_INTERVAL_MS);
}

// ─── Event wiring ─────────────────────────────────────────────────────────────
demoBtn.addEventListener("click", startDemo);
cameraBtn.addEventListener("click", startCamera);
videoFile.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) loadVideoFile(file);
});
playPauseBtn.addEventListener("click", () => {
  state.running = !state.running;
  playPauseBtn.textContent = state.running ? "Pause" : "Resume";
});
snapshotBtn.addEventListener("click", async () => {
  addMessage("analyst", await adapter.chat("Summarize the current scene and performance.", currentContext()));
});
chatSendBtn.addEventListener("click", sendChat);
chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendChat(); }
});
window.addEventListener("resize", resizeCanvases);

// ─── Boot ─────────────────────────────────────────────────────────────────────
addMessage(
  "analyst",
  `Frontend ready. API target is ${API_BASE}. ` +
  `reload if needed.`,
);
checkBackend();
window.setInterval(checkBackend, 10000);  // every 10 s — backend busy under load
startDemo();
requestAnimationFrame(renderLoop);   // draw at 60 fps
detectionLoop();                     // update tracking as fast as detection allows
actionLoop();                        // update actions independently at a slower rate
