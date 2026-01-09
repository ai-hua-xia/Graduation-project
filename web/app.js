const styleConfig = {
  night: { rot: "-35deg", label: "decoder: night" },
  fog: { rot: "0deg", label: "decoder: fog" },
  snow: { rot: "35deg", label: "decoder: snow" },
};

const realCanvas = document.getElementById("realCanvas");
const dreamCanvas = document.getElementById("dreamCanvas");
const realCtx = realCanvas.getContext("2d");
const dreamCtx = dreamCanvas.getContext("2d");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const fpsValue = document.getElementById("fpsValue");
const styleNote = document.getElementById("styleNote");
const throttleFill = document.getElementById("throttleFill");
const steerFill = document.getElementById("steerFill");
const contextFill = document.getElementById("contextFill");
const tempFill = document.getElementById("tempFill");

const streams = {
  real: { video: null, mode: "demo" },
  dream: {
    night: { video: null, mode: "demo" },
    fog: { video: null, mode: "demo" },
    snow: { video: null, mode: "demo" },
  },
};

let activeStyle = "night";

let lastFrameTime = performance.now();
let frameCount = 0;

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

document.querySelectorAll(".style-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    setStyle(btn.dataset.style);
  });
});

document.querySelectorAll('input[type="file"]').forEach((input) => {
  input.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const target = event.target.dataset.target;
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.src = url;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.play();
    if (target === "real") {
      streams.real = { video, mode: "video" };
    } else if (target && target.startsWith("dream-")) {
      const style = target.split("-")[1];
      if (streams.dream[style]) {
        streams.dream[style] = { video, mode: "video" };
      }
    }
    setStatus("local files", true);
  });
});

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

function updateMeters(time) {
  const throttle = (Math.sin(time * 0.001) + 1) / 2;
  const steer = (Math.cos(time * 0.0013) + 1) / 2;
  const context = (Math.sin(time * 0.0007) + 1) / 2;
  const temp = (Math.cos(time * 0.0009) + 1) / 2;
  throttleFill.style.width = `${30 + throttle * 70}%`;
  steerFill.style.width = `${20 + steer * 80}%`;
  contextFill.style.width = `${40 + context * 60}%`;
  tempFill.style.width = `${35 + temp * 65}%`;
}

function tick(time) {
  const realStream = streams.real;
  const dreamStream = streams.dream[activeStyle] || { mode: "demo" };

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

  updateMeters(time);

  frameCount += 1;
  if (time - lastFrameTime > 1000) {
    fpsValue.textContent = frameCount.toString();
    frameCount = 0;
    lastFrameTime = time;
  }

  requestAnimationFrame(tick);
}

setStyle("night");
setStatus("demo mode", true);
requestAnimationFrame(tick);
