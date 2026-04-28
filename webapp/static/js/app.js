/**
 * DeepSceneLoc — Production App JS
 * Handles: image upload/drag-drop, API calls, result rendering,
 *          history, map links, model status polling
 */

const API = {
  predict : "/api/predict",
  analyze : "/api/analyze",
  status  : "/api/status",
  models  : "/api/models",
};

const CLASS_COLORS = {
  Coastal:  "var(--coastal)",
  Forest:   "var(--forest)",
  Mountain: "var(--mountain)",
  Rural:    "var(--rural)",
  Urban:    "var(--urban)",
};

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────
let currentFile  = null;
let analyzeMode  = true;  // true = full hybrid, false = stage-1 only
let history      = [];    // [{thumb, top_class, confidence}]
let modelInfo    = {};

// ─────────────────────────────────────────────────────────────
// DOM refs
// ─────────────────────────────────────────────────────────────
const dropZone    = document.getElementById("drop-zone");
const fileInput   = document.getElementById("file-input");
const previewImg  = document.getElementById("preview-img");
const dzPlaceholder = document.querySelector(".dz-placeholder");
const geminiToggle  = document.getElementById("gemini-toggle");
const analyzeBtn    = document.getElementById("analyze-btn");
const resultsArea   = document.getElementById("results-area");
const historyGrid   = document.getElementById("history-grid");
const modelBanner   = document.getElementById("model-banner");
const toastCont     = document.getElementById("toast-container");

// ─────────────────────────────────────────────────────────────
// Toast
// ─────────────────────────────────────────────────────────────
function toast(msg, type = "info", duration = 3500) {
  const icons = { info: "ℹ️", success: "✅", error: "❌", warning: "⚠️" };
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type]}</span><span>${msg}</span>`;
  toastCont.appendChild(el);
  setTimeout(() => {
    el.style.transition = "opacity 0.3s";
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 300);
  }, duration);
}

// ─────────────────────────────────────────────────────────────
// Image loading
// ─────────────────────────────────────────────────────────────
function loadFile(file) {
  if (!file?.type?.startsWith("image/")) {
    toast("Please upload an image file (JPG, PNG, WEBP)", "error");
    return;
  }
  currentFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.classList.add("visible");
  dzPlaceholder.style.display = "none";
  dropZone.classList.add("has-image");
  analyzeBtn.disabled = false;

  // Auto-analyze on load
  runAnalysis();
}

// ─────────────────────────────────────────────────────────────
// Drag & Drop
// ─────────────────────────────────────────────────────────────
dropZone.addEventListener("click", (e) => {
  if (e.target !== dropZone && !e.target.classList.contains("dz-btn") &&
      !dzPlaceholder.contains(e.target)) return;
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});

// Paste from clipboard
document.addEventListener("paste", (e) => {
  const item = [...e.clipboardData.items].find(i => i.type.startsWith("image/"));
  if (item) loadFile(item.getAsFile());
});

// Gemini toggle
geminiToggle.addEventListener("change", () => { analyzeMode = geminiToggle.checked; });

// Analyze button
analyzeBtn.addEventListener("click", runAnalysis);

// ─────────────────────────────────────────────────────────────
// Core: Run analysis
// ─────────────────────────────────────────────────────────────
async function runAnalysis() {
  if (!currentFile) { toast("Upload an image first", "warning"); return; }

  // Loading state
  analyzeBtn.classList.add("loading");
  analyzeBtn.disabled = true;
  resultsArea.innerHTML = renderLoading();

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const endpoint = analyzeMode ? API.analyze : API.predict;
    const res      = await fetch(endpoint, { method: "POST", body: formData });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResults(data, analyzeMode);
    addToHistory(currentFile, data);
    toast("Analysis complete", "success", 2000);
  } catch (err) {
    resultsArea.innerHTML = renderError(err.message);
    toast(`Error: ${err.message}`, "error");
  } finally {
    analyzeBtn.classList.remove("loading");
    analyzeBtn.disabled = false;
  }
}

// ─────────────────────────────────────────────────────────────
// Render: Loading skeleton
// ─────────────────────────────────────────────────────────────
function renderLoading() {
  return `
    <div class="stage-card">
      <div class="stage-header">
        <span class="sh-label">Stage 1 — Scene Classification</span>
        <span class="sh-badge warn">Analyzing…</span>
      </div>
      <div class="stage-body" style="text-align:center;padding:2rem;color:var(--text-muted)">
        <div style="font-size:2rem;margin-bottom:1rem;animation:spin 1s linear infinite;display:inline-block">⚙️</div>
        <div>Running deep learning inference…</div>
      </div>
    </div>`;
}

// ─────────────────────────────────────────────────────────────
// Render: Error
// ─────────────────────────────────────────────────────────────
function renderError(msg) {
  return `
    <div class="stage-card">
      <div class="stage-header">
        <span class="sh-label">Error</span>
        <span class="sh-badge err">Failed</span>
      </div>
      <div class="stage-body" style="color:var(--danger);font-size:0.875rem;padding:1rem">
        ❌ ${msg}
      </div>
    </div>`;
}

// ─────────────────────────────────────────────────────────────
// Render: Full results
// ─────────────────────────────────────────────────────────────
function renderResults(data, hybrid) {
  const s1 = hybrid ? data.stage1 : data;
  const s2 = hybrid ? data.stage2 : null;

  resultsArea.innerHTML = renderStage1(s1) + (s2 ? renderStage2(s2) : "");
}

function renderStage1(s1) {
  const top    = s1.top_class;
  const conf   = s1.confidence;
  const color  = CLASS_COLORS[top] || "var(--accent)";
  const probs  = s1.probabilities || {};
  const sorted = Object.entries(probs).sort((a,b) => b[1]-a[1]);
  const mock   = s1.mock ? '<span style="color:var(--warning);font-size:0.7rem">⚠️ demo weights</span>' : "";

  const probRows = sorted.map(([cls, p]) => {
    const isTop = cls === top;
    const fill  = CLASS_COLORS[cls] || "var(--accent)";
    return `
      <div class="prob-row ${isTop ? "top" : ""}">
        <span class="pr-name">${cls}</span>
        <div class="prob-track">
          <div class="prob-fill" style="width:${(p*100).toFixed(1)}%;background:${fill}"></div>
        </div>
        <span class="pr-val">${(p*100).toFixed(1)}%</span>
      </div>`;
  }).join("");

  const confLevel = conf >= 0.65 ? "HIGH" : conf >= 0.40 ? "MEDIUM" : "LOW";

  return `
    <div class="stage-card">
      <div class="stage-header">
        <span class="sh-label">🧠 Stage 1 — Scene Classification</span>
        <span class="sh-badge">${confLevel} confidence</span>
      </div>
      <div class="stage-body">
        <div class="cat-badge">
          <span class="cat-emoji">${s1.emoji || "🗺️"}</span>
          <div class="cat-info">
            <div class="cat-name" style="color:${color}">${top}</div>
            <div class="cat-desc">${s1.description || ""}</div>
          </div>
        </div>

        <div class="conf-bar-wrap">
          <div class="conf-label">
            <span class="conf-text">Confidence</span>
            <span class="conf-pct">${(conf*100).toFixed(1)}%</span>
          </div>
          <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width:${(conf*100).toFixed(1)}%"></div>
          </div>
        </div>

        <div class="prob-list">${probRows}</div>

        <div class="latency-pill">
          ⚡ ${s1.latency_ms || "—"} ms ·
          Model: ${s1.model || modelInfo.arch || "ResNet-50"} ${mock}
        </div>
      </div>
    </div>`;
}

function renderStage2(s2) {
  const loc   = s2.exact_location || "Unknown";
  const conf  = s2.confidence || "none";
  const lat   = s2.latitude;
  const lon   = s2.longitude;
  const cache = s2.from_cache ? '<span class="cache-tag">⚡ cached</span>' : "";
  const mapBtn = (lat && lon)
    ? `<a class="map-btn" href="https://www.google.com/maps?q=${lat},${lon}" target="_blank" rel="noopener">
        🗺️ Open in Google Maps
       </a>`
    : "";

  const landmarks = (s2.landmarks || []).join(", ") || "—";
  const coords    = (lat && lon) ? `${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E` : "—";

  const noGemini = loc.includes("not configured") || loc.includes("Gemini not");

  const descHtml = s2.description
    ? `<div class="gemini-desc">${s2.description}</div>`
    : "";

  return `
    <div class="stage-card">
      <div class="stage-header">
        <span class="sh-label">🌍 Stage 2 — Exact Location (Gemini AI)</span>
        <span class="sh-badge ${noGemini ? "warn" : ""}">${noGemini ? "API key needed" : "Complete"}</span>
      </div>
      <div class="stage-body">
        <div class="loc-exact">
          <div class="le-name">${loc} ${cache}</div>
          <div class="le-conf ${conf}">${conf.toUpperCase()} confidence</div>
        </div>

        <div class="loc-rows">
          <div class="loc-row">
            <span class="lr-icon">🌐</span>
            <span class="lr-key">Country</span>
            <span class="lr-val">${s2.country || "—"}</span>
          </div>
          <div class="loc-row">
            <span class="lr-icon">🏙️</span>
            <span class="lr-key">City</span>
            <span class="lr-val">${s2.city || "—"}</span>
          </div>
          <div class="loc-row">
            <span class="lr-icon">📍</span>
            <span class="lr-key">Coords</span>
            <span class="lr-val">${coords}</span>
          </div>
          <div class="loc-row">
            <span class="lr-icon">🏛️</span>
            <span class="lr-key">Landmarks</span>
            <span class="lr-val">${landmarks}</span>
          </div>
        </div>

        ${mapBtn}
        ${descHtml}

        <div class="latency-pill">⚡ ${s2.latency_ms || "—"} ms</div>
      </div>
    </div>`;
}

// ─────────────────────────────────────────────────────────────
// History
// ─────────────────────────────────────────────────────────────
function addToHistory(file, data) {
  const s1 = data.stage1 || data;
  const thumb = URL.createObjectURL(file);
  history.unshift({ thumb, top_class: s1.top_class, confidence: s1.confidence, file, data });
  if (history.length > 8) history.pop();
  renderHistory();
}

function renderHistory() {
  if (!historyGrid || history.length === 0) return;
  historyGrid.innerHTML = history.map((item, idx) => `
    <div class="history-item" data-idx="${idx}">
      <img src="${item.thumb}" alt="${item.top_class}">
      <div class="history-meta">
        <div class="hm-cat" style="color:${CLASS_COLORS[item.top_class]}">${item.top_class}</div>
        <div class="hm-conf">${(item.confidence*100).toFixed(1)}%</div>
      </div>
    </div>
  `).join("");

  historyGrid.querySelectorAll(".history-item").forEach(el => {
    el.addEventListener("click", () => {
      const item = history[+el.dataset.idx];
      currentFile = item.file;
      previewImg.src = item.thumb;
      previewImg.classList.add("visible");
      dzPlaceholder.style.display = "none";
      dropZone.classList.add("has-image");
      renderResults(item.data, !!item.data.stage1);
    });
  });
}

// ─────────────────────────────────────────────────────────────
// Model status banner
// ─────────────────────────────────────────────────────────────
async function loadModelStatus() {
  try {
    const res = await fetch(API.status);
    const data = await res.json();
    modelInfo = data.model || {};

    const arch    = (modelInfo.arch || "unknown").replace("_", "-").toUpperCase();
    const acc     = modelInfo.val_acc
      ? `${parseFloat(modelInfo.val_acc).toFixed(2)}%`
      : "—";
    const epoch   = modelInfo.epoch || "—";
    const gpu     = data.gpu?.name?.split(" ").slice(-3).join(" ") || "CPU";
    const gemini  = data.gemini?.available ? "✅ Ready" : "⚠️ Set GEMINI_API_KEY";
    const status  = modelInfo.status === "loaded" ? "✅ Loaded" : "⚠️ No checkpoint";

    if (modelBanner) {
      modelBanner.innerHTML = `
        <div class="mb-item"><span class="mb-key">Model</span><span class="mb-val">${arch}</span></div>
        <span class="mb-sep">·</span>
        <div class="mb-item"><span class="mb-key">Val Acc</span><span class="mb-val">${acc}</span></div>
        <span class="mb-sep">·</span>
        <div class="mb-item"><span class="mb-key">Epoch</span><span class="mb-val">${epoch}</span></div>
        <span class="mb-sep">·</span>
        <div class="mb-item"><span class="mb-key">GPU</span><span class="mb-val">${gpu}</span></div>
        <span class="mb-sep">·</span>
        <div class="mb-item"><span class="mb-key">Gemini</span><span class="mb-val">${gemini}</span></div>
        <span class="mb-sep">·</span>
        <div class="mb-item"><span class="mb-key">Status</span><span class="mb-val">${status}</span></div>`;
    }

    // Update nav status dot colour
    const navStatus = document.querySelector(".status-dot");
    if (navStatus) {
      navStatus.style.background = modelInfo.status === "loaded" ? "var(--success)" : "var(--warning)";
    }
    const navStatusText = document.getElementById("nav-status-text");
    if (navStatusText) {
      navStatusText.textContent = modelInfo.status === "loaded" ? "System Online" : "Demo Mode";
    }
  } catch (_) {
    /* silently fail */
  }
}

// Init
loadModelStatus();
