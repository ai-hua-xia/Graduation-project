const styleConfig = {
  night: { rot: "-35deg", label: "decoder: night" },
  fog: { rot: "0deg", label: "decoder: fog" },
  snow: { rot: "35deg", label: "decoder: snow" },
};

const realCanvas = document.getElementById("realCanvas");
const dreamCanvas = document.getElementById("dreamCanvas");
const diffCanvas = document.getElementById("diffCanvas");
const realCtx = realCanvas.getContext("2d");
const dreamCtx = dreamCanvas.getContext("2d");
const diffCtx = diffCanvas ? diffCanvas.getContext("2d") : null;

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const fpsValue = document.getElementById("fpsValue");
const styleNote = document.getElementById("styleNote");
const leftTitle = document.getElementById("leftTitle");
const leftMeta = document.getElementById("leftMeta");
const rightTitle = document.getElementById("rightTitle");
const rightMeta = document.getElementById("rightMeta");
const framesPerActionInput = document.getElementById("framesPerAction");
const videoFpsInput = document.getElementById("videoFps");

const streams = {
  style: {
    real: { video: null, mode: "demo" },
    dream: {
      night: { video: null, mode: "demo" },
      fog: { video: null, mode: "demo" },
      snow: { video: null, mode: "demo" },
    },
  },
  action: {
    base: { video: null, mode: "demo" },
    dream: { video: null, mode: "demo" },
  },
};

const diffBufferA = document.createElement("canvas");
const diffBufferB = document.createElement("canvas");
const diffCtxA = diffBufferA.getContext("2d");
const diffCtxB = diffBufferB.getContext("2d");
if (diffCanvas) {
  diffBufferA.width = diffCanvas.width;
  diffBufferA.height = diffCanvas.height;
  diffBufferB.width = diffCanvas.width;
  diffBufferB.height = diffCanvas.height;
}

let activeStyle = "night";
let activeMode = "style";
let lastFrameTime = performance.now();
let frameCount = 0;
let lastActionText = "";

const actionState = {
  steerFrames: [],
  cumulative: [],
  maxCumulative: 1,
};

function setStatus(text, active) {
  statusText.textContent = text;
  statusDot.style.background = active ? "var(--accent-2)" : "var(--accent-3)";
}

function setStyle(style) {
  const cfg = styleConfig[style] || styleConfig.night;
  activeStyle = style;
  document.documentElement.style.setProperty("--dial-rot", cfg.rot);
  styleNote.textContent = cfg.label;
  document.querySelectorAll(".style-btn").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.style === style);
  });
}

function setMode(mode) {
  activeMode = mode;
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.mode === mode);
  });
  document.querySelectorAll(".mode-style").forEach((el) => {
    el.classList.toggle("is-hidden", mode !== "style");
  });
  document.querySelectorAll(".mode-action").forEach((el) => {
    el.classList.toggle("is-hidden", mode !== "action");
  });

  if (mode === "action") {
    leftTitle.textContent = "Dream (No Action)";
    leftMeta.textContent = "baseline rollout";
    rightTitle.textContent = "Dream (With Action)";
    rightMeta.textContent = "action-conditioned";
    setStatus("action compare", true);
  } else {
    leftTitle.textContent = "Reality Feed";
    leftMeta.textContent = "MetaDrive observer";
    rightTitle.textContent = "Imagination Feed";
    rightMeta.textContent = "VQ-VAE + World Model";
    setStatus("style compare", true);
  }
}

function buildVideoFromFile(file) {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src = url;
  video.loop = true;
  video.muted = true;
  video.playsInline = true;
  video.play();
  return video;
}

function parseActionText(text) {
  lastActionText = text || "";
  const lines = text.split(/\r?\n/);
  const framesPerAction = Number.parseInt(framesPerActionInput?.value, 10) || 15;
  const steerFrames = [];
  lines.forEach((rawLine) => {
    const token = rawLine.trim().toLowerCase();
    if (!token || token.startsWith("#")) return;
    if (token === "space" || token === "brake") {
      for (let i = 0; i < framesPerAction; i += 1) {
        steerFrames.push(0);
      }
      return;
    }
    const letters = new Set(token);
    let steer = 0;
    if (letters.has("a")) steer -= 1;
    if (letters.has("d")) steer += 1;
    for (let i = 0; i < framesPerAction; i += 1) {
      steerFrames.push(steer);
    }
  });

  const cumulative = [];
  let sum = 0;
  steerFrames.forEach((val) => {
    sum += val;
    cumulative.push(sum);
  });
  const maxCumulative = Math.max(1, ...cumulative.map((v) => Math.abs(v)));

  actionState.steerFrames = steerFrames;
  actionState.cumulative = cumulative;
  actionState.maxCumulative = maxCumulative;
}

function drawDemoFrame(ctx, time, variant) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  const t = time * 0.0003;

  const grad = ctx.createLinearGradient(0, 0, w, h);
  if (variant === "dream") {
    grad.addColorStop(0, "#1b2d34");
    grad.addColorStop(1, "#20404a");
  } else {
    grad.addColorStop(0, "#394b4f");
    grad.addColorStop(1, "#5d6f6d");
  }
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.lineWidth = 2;
  for (let i = 0; i < 6; i += 1) {
    const y = (t * 80 + i * 60) % h;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y + 20);
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(242, 140, 43, 0.2)";
  ctx.beginPath();
  ctx.arc(w * 0.7, h * 0.3, 70 + 10 * Math.sin(t * 2), 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = variant === "dream" ? "rgba(47, 156, 149, 0.4)" : "rgba(248, 248, 240, 0.4)";
  ctx.beginPath();
  ctx.arc(w * 0.3, h * 0.6, 90 + 20 * Math.cos(t * 1.4), 0, Math.PI * 2);
  ctx.fill();
}

function drawVideoFrame(ctx, video) {
  ctx.drawImage(video, 0, 0, ctx.canvas.width, ctx.canvas.height);
}

function drawActionCurve(ctx, frameIndex) {
  if (!actionState.steerFrames.length) return;
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  const panelHeight = Math.min(70, Math.floor(h * 0.28));
  const y0 = h - panelHeight;

  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
  ctx.fillRect(0, y0, w, panelHeight);

  const windowSize = Math.min(140, actionState.steerFrames.length);
  const start = Math.max(0, frameIndex - windowSize + 1);
  const end = Math.min(actionState.steerFrames.length, start + windowSize);
  const denom = Math.max(1, end - start - 1);

  const steerColor = "rgba(96, 220, 255, 0.9)";
  const cumColor = "rgba(255, 166, 69, 0.9)";

  ctx.lineWidth = 2;
  ctx.strokeStyle = steerColor;
  ctx.beginPath();
  for (let i = start; i < end; i += 1) {
    const x = ((i - start) / denom) * w;
    const steer = actionState.steerFrames[i] || 0;
    const y = y0 + panelHeight / 2 - steer * (panelHeight * 0.35);
    if (i === start) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.strokeStyle = cumColor;
  ctx.beginPath();
  for (let i = start; i < end; i += 1) {
    const x = ((i - start) / denom) * w;
    const cum = (actionState.cumulative[i] || 0) / actionState.maxCumulative;
    const y = y0 + panelHeight / 2 - cum * (panelHeight * 0.35);
    if (i === start) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  const currentSteer = actionState.steerFrames[frameIndex] || 0;
  ctx.fillStyle = "#ffffff";
  ctx.font = "12px 'Space Grotesk', sans-serif";
  ctx.fillText(`steer: ${currentSteer.toFixed(2)}`, 10, y0 + 18);
  ctx.fillText(`cum: ${(actionState.cumulative[frameIndex] || 0).toFixed(1)}`, 10, y0 + 34);
  ctx.restore();
}

function drawDiffHeatmap(baseVideo, actionVideo, time) {
  if (!diffCanvas || !diffCtx) return;
  const w = diffCanvas.width;
  const h = diffCanvas.height;

  if (!baseVideo || !actionVideo || baseVideo.readyState < 2 || actionVideo.readyState < 2) {
    drawDemoFrame(diffCtx, time, "dream");
    return;
  }

  diffCtxA.drawImage(baseVideo, 0, 0, w, h);
  diffCtxB.drawImage(actionVideo, 0, 0, w, h);
  const dataA = diffCtxA.getImageData(0, 0, w, h).data;
  const dataB = diffCtxB.getImageData(0, 0, w, h).data;
  const output = diffCtx.createImageData(w, h);
  for (let i = 0; i < dataA.length; i += 4) {
    const dr = Math.abs(dataA[i] - dataB[i]);
    const dg = Math.abs(dataA[i + 1] - dataB[i + 1]);
    const db = Math.abs(dataA[i + 2] - dataB[i + 2]);
    const diff = (dr + dg + db) / 3;
    const r = Math.min(255, diff * 1.4);
    const g = Math.min(255, diff * 0.7);
    const b = Math.min(255, diff * 0.2);
    output.data[i] = r;
    output.data[i + 1] = g;
    output.data[i + 2] = b;
    output.data[i + 3] = 255;
  }
  diffCtx.putImageData(output, 0, 0);

  const fps = Number.parseFloat(videoFpsInput?.value) || 10;
  const t = actionVideo.currentTime || baseVideo.currentTime || 0;
  const frameIndex = Math.max(0, Math.floor(t * fps));
  drawActionCurve(diffCtx, frameIndex);
}

function tick(time) {
  if (activeMode === "style") {
    const realStream = streams.style.real;
    const dreamStream = streams.style.dream[activeStyle] || { mode: "demo" };

    if (realStream.mode === "video" && realStream.video.readyState >= 2) {
      drawVideoFrame(realCtx, realStream.video);
    } else {
      drawDemoFrame(realCtx, time, "real");
    }

    if (dreamStream.mode === "video" && dreamStream.video.readyState >= 2) {
      drawVideoFrame(dreamCtx, dreamStream.video);
    } else {
      drawDemoFrame(dreamCtx, time, "dream");
    }
  } else {
    const baseStream = streams.action.base;
    const actionStream = streams.action.dream;

    if (baseStream.mode === "video" && baseStream.video.readyState >= 2) {
      drawVideoFrame(realCtx, baseStream.video);
    } else {
      drawDemoFrame(realCtx, time, "real");
    }

    if (actionStream.mode === "video" && actionStream.video.readyState >= 2) {
      drawVideoFrame(dreamCtx, actionStream.video);
    } else {
      drawDemoFrame(dreamCtx, time, "dream");
    }

    drawDiffHeatmap(baseStream.video, actionStream.video, time);
  }

  frameCount += 1;
  if (time - lastFrameTime > 1000) {
    fpsValue.textContent = frameCount.toString();
    frameCount = 0;
    lastFrameTime = time;
  }

  requestAnimationFrame(tick);
}

function handleFileInput(event) {
  const file = event.target.files[0];
  if (!file) return;
  const target = event.target.dataset.target;

  if (target === "action-file") {
    const reader = new FileReader();
    reader.onload = () => {
      parseActionText(reader.result || "");
      setStatus(`action frames: ${actionState.steerFrames.length}`, true);
    };
    reader.readAsText(file);
    return;
  }

  const video = buildVideoFromFile(file);
  if (target === "real") {
    streams.style.real = { video, mode: "video" };
  } else if (target && target.startsWith("dream-")) {
    const style = target.split("-")[1];
    if (streams.style.dream[style]) {
      streams.style.dream[style] = { video, mode: "video" };
    }
  } else if (target === "action-base") {
    streams.action.base = { video, mode: "video" };
  } else if (target === "action-dream") {
    streams.action.dream = { video, mode: "video" };
  }
  setStatus("local files", true);
}

function initListeners() {
  document.querySelectorAll(".style-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      setStyle(btn.dataset.style);
    });
  });

  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      setMode(btn.dataset.mode);
    });
  });

  document.querySelectorAll('input[type="file"]').forEach((input) => {
    input.addEventListener("change", handleFileInput);
  });

  if (framesPerActionInput) {
    framesPerActionInput.addEventListener("change", () => {
      if (lastActionText) {
        parseActionText(lastActionText);
        setStatus(`action frames: ${actionState.steerFrames.length}`, true);
      }
    });
  }
}

setStyle("night");
setMode("style");
setStatus("demo mode", true);
initListeners();
requestAnimationFrame(tick);
