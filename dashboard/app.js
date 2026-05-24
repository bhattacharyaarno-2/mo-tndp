const dashboardData = window.MOTNDP_DASHBOARD_DATA || { models: [] };
const allModels = dashboardData.models.map((model) => ({ ...model, originalTitle: model.title }));
const runManagementKey = "motndp-dashboard-run-management";
let runManagement = { renames: {}, hidden: {} };

const state = {
  modelIndex: 0,
  comparisonModelId: null,
  overlay: "socioeconomic_groups",
  view: "playback",
  showBaseMap: false,
  mapOpacity: 0.45,
  mapScale: 1,
  mapOffsetX: 0,
  mapOffsetY: 0,
  step: 0,
  playing: false,
  speed: 700,
  timer: null,
};

const palette = ["#0f6c7a", "#ef7d57", "#f6bd60", "#84a59d", "#9f86c0", "#4d908e", "#bc4749", "#277da1", "#f94144", "#577590"];
const socioeconomicScale = ["#fdd0c0", "#fca082", "#fb6a4a", "#ef3b2c", "#b80d1a"];

const els = {
  modelSelect: document.getElementById("model-select"),
  overlaySelect: document.getElementById("overlay-select"),
  mapToggle: document.getElementById("map-toggle"),
  mapCalibrationBar: document.getElementById("map-calibration-bar"),
  mapOpacitySlider: document.getElementById("map-opacity-slider"),
  mapScaleSlider: document.getElementById("map-scale-slider"),
  mapOffsetXSlider: document.getElementById("map-offset-x-slider"),
  mapOffsetYSlider: document.getElementById("map-offset-y-slider"),
  stepSlider: document.getElementById("step-slider"),
  playButton: document.getElementById("play-button"),
  resetButton: document.getElementById("reset-button"),
  speedSlider: document.getElementById("speed-slider"),
  comparisonSelect: document.getElementById("comparison-select"),
  playbackView: document.getElementById("playback-view"),
  libraryView: document.getElementById("library-view"),
  playbackButton: document.getElementById("view-playback"),
  libraryButton: document.getElementById("view-library"),
  manageButton: document.getElementById("view-manage"),
  libraryGrid: document.getElementById("library-grid"),
  manageView: document.getElementById("manage-view"),
  manageGrid: document.getElementById("manage-grid"),
  resetRunManagement: document.getElementById("reset-run-management"),
  wandbPanel: document.getElementById("wandb-panel"),
  wandbTitle: document.getElementById("wandb-title"),
  wandbSummaryGrid: document.getElementById("wandb-summary-grid"),
  wandbMetaGrid: document.getElementById("wandb-meta-grid"),
  wandbChartsGrid: document.getElementById("wandb-charts-grid"),
  wandbMediaGrid: document.getElementById("wandb-media-grid"),
  heroMetrics: document.getElementById("hero-metrics"),
  panelTitle: document.getElementById("panel-title"),
  stepPill: document.getElementById("step-pill"),
  comparisonTitle: document.getElementById("comparison-title"),
  comparisonCopy: document.getElementById("comparison-copy"),
  decisionTitle: document.getElementById("decision-title"),
  decisionCopy: document.getElementById("decision-copy"),
  allowedActions: document.getElementById("allowed-actions"),
  fairnessSummary: document.getElementById("fairness-summary"),
  fairnessBreakdown: document.getElementById("fairness-breakdown"),
  topStarts: document.getElementById("top-starts"),
  stepTable: document.getElementById("step-table"),
  configGrid: document.getElementById("config-grid"),
  legendRow: document.getElementById("legend-row"),
  comparisonLegend: document.getElementById("comparison-legend"),
  canvas: document.getElementById("map-canvas"),
  comparisonCanvas: document.getElementById("comparison-canvas"),
};

const ctx = els.canvas.getContext("2d");
const comparisonCtx = els.comparisonCanvas.getContext("2d");
const baseMapImage = new Image();
baseMapImage.src = "./assets/amsterdam-reference-map.png";
baseMapImage.decoding = "async";

const storageKey = "motndp-dashboard-map-calibration";

function loadMapSettings() {
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) {
      return;
    }
    const saved = JSON.parse(raw);
    state.showBaseMap = Boolean(saved.showBaseMap);
    state.mapOpacity = Number.isFinite(saved.mapOpacity) ? saved.mapOpacity : state.mapOpacity;
    state.mapScale = Number.isFinite(saved.mapScale) ? saved.mapScale : state.mapScale;
    state.mapOffsetX = Number.isFinite(saved.mapOffsetX) ? saved.mapOffsetX : state.mapOffsetX;
    state.mapOffsetY = Number.isFinite(saved.mapOffsetY) ? saved.mapOffsetY : state.mapOffsetY;
  } catch (error) {
    console.warn("Could not restore map calibration settings.", error);
  }
}

function saveMapSettings() {
  try {
    window.localStorage.setItem(
      storageKey,
      JSON.stringify({
        showBaseMap: state.showBaseMap,
        mapOpacity: state.mapOpacity,
        mapScale: state.mapScale,
        mapOffsetX: state.mapOffsetX,
        mapOffsetY: state.mapOffsetY,
      })
    );
  } catch (error) {
    console.warn("Could not save map calibration settings.", error);
  }
}

function loadRunManagement() {
  try {
    const raw = window.localStorage.getItem(runManagementKey);
    if (!raw) {
      return;
    }
    const parsed = JSON.parse(raw);
    runManagement = {
      renames: parsed.renames || {},
      hidden: parsed.hidden || {},
    };
  } catch (error) {
    console.warn("Could not restore run management settings.", error);
  }
}

function saveRunManagement() {
  try {
    window.localStorage.setItem(runManagementKey, JSON.stringify(runManagement));
  } catch (error) {
    console.warn("Could not save run management settings.", error);
  }
}

function applyRunManagement({ preserveCurrent = true } = {}) {
  const currentId = preserveCurrent ? currentModel()?.id : null;
  allModels.forEach((model) => {
    model.title = runManagement.renames[model.id] || model.originalTitle;
  });
  dashboardData.models = allModels.filter((model) => !runManagement.hidden[model.id]);

  if (!dashboardData.models.length) {
    state.modelIndex = 0;
    state.comparisonModelId = null;
    return;
  }

  const nextIndex = currentId ? dashboardData.models.findIndex((model) => model.id === currentId) : state.modelIndex;
  state.modelIndex = nextIndex >= 0 ? nextIndex : Math.min(state.modelIndex, dashboardData.models.length - 1);
  state.comparisonModelId = dashboardData.comparison_defaults?.[currentModel().id] || null;
}

function populateModelSelect() {
  if (!dashboardData.models.length) {
    els.modelSelect.innerHTML = `<option value="">No visible models</option>`;
    return;
  }
  els.modelSelect.innerHTML = dashboardData.models
    .map((model, index) => `<option value="${index}">${model.kind === "deep_rl" ? "Deep RL" : "Tabular"}: ${model.title}</option>`)
    .join("");
  els.modelSelect.value = String(state.modelIndex);
}

function currentModel() {
  return dashboardData.models[state.modelIndex];
}

function currentFrame() {
  return currentModel().timeline[state.step];
}

function modelById(modelId) {
  return dashboardData.models.find((model) => model.id === modelId) || null;
}

function currentComparisonModel() {
  return modelById(state.comparisonModelId);
}

function formatValue(value) {
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }
  return String(value);
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatNumber(value) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatPercent(value) {
  return `${(Number(value) * 100).toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;
}

function formatCompactNumber(value) {
  if (!Number.isFinite(Number(value))) {
    return formatValue(value);
  }
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: Math.abs(Number(value)) >= 100 ? 1 : 4,
  });
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function giniCoefficient(values) {
  const finite = values.filter((value) => Number.isFinite(value) && value >= 0).sort((a, b) => a - b);
  if (!finite.length) {
    return 0;
  }
  const total = finite.reduce((sum, value) => sum + value, 0);
  if (total <= 0) {
    return 0;
  }
  let weightedSum = 0;
  finite.forEach((value, index) => {
    weightedSum += (index + 1) * value;
  });
  return (2 * weightedSum) / (finite.length * total) - (finite.length + 1) / finite.length;
}

function interpolateColor(start, end, t) {
  const s = start.match(/\w\w/g).map((chunk) => parseInt(chunk, 16));
  const e = end.match(/\w\w/g).map((chunk) => parseInt(chunk, 16));
  const mixed = s.map((value, index) => Math.round(value + (e[index] - value) * t));
  return `rgb(${mixed[0]}, ${mixed[1]}, ${mixed[2]})`;
}

function heatColor(value, min, max) {
  if (!Number.isFinite(value)) {
    return "#efe6d6";
  }
  const normalized = max > min ? (value - min) / (max - min) : 0;
  if (normalized < 0.5) {
    return interpolateColor("f6efe4", "f6bd60", normalized / 0.5);
  }
  return interpolateColor("f6bd60", "bb4d24", (normalized - 0.5) / 0.5);
}

function socioeconomicGradientColor(value, min, max) {
  if (!Number.isFinite(value)) {
    return "#c8ccd1";
  }
  const normalized = max > min ? (value - min) / (max - min) : 0;
  if (normalized <= 0.25) {
    return interpolateColor("fdd0c0", "fca082", normalized / 0.25);
  }
  if (normalized <= 0.5) {
    return interpolateColor("fca082", "fb6a4a", (normalized - 0.25) / 0.25);
  }
  if (normalized <= 0.75) {
    return interpolateColor("fb6a4a", "ef3b2c", (normalized - 0.5) / 0.25);
  }
  return interpolateColor("ef3b2c", "b80d1a", (normalized - 0.75) / 0.25);
}

function socioeconomicLegendLabels(model) {
  return model.socioeconomic_group_labels && model.socioeconomic_group_labels.length
    ? model.socioeconomic_group_labels
    : model.group_labels;
}

function describeSeriesLabel(key) {
  const labels = {
    reward: "Reward",
    average_reward: "Average reward",
    best_episode_reward: "Best episode reward",
    epsilon: "Epsilon",
    training_step: "Training step",
    training_phase_code: "Training phase",
    train_q_network: "Action net active",
    train_start_network: "Start net active",
    bonus_existing_connection: "Existing connection bonus",
    penalty_existing_overlap: "Existing overlap penalty",
    episode: "Episode",
    "gpu.0.powerWatts": "GPU power",
    "gpu.0.gpu": "GPU utilization",
    "gpu.0.memory": "GPU memory usage",
    "gpu.0.memoryAllocated": "GPU memory allocated",
    "gpu.0.temp": "GPU temperature",
    "gpu.0.smClock": "GPU SM clock",
    "gpu.0.memoryClock": "GPU memory clock",
    cpu: "CPU utilization",
    memory_percent: "RAM utilization",
    "proc.memory.rssMB": "Process memory",
    "proc.memory.percent": "Process memory share",
    "proc.cpu.threads": "Process threads",
  };
  return labels[key] || key;
}

function describeMetricUnit(key) {
  if (key.includes("powerWatts") || key.endsWith("_power")) {
    return "W";
  }
  if (key.includes("Clock")) {
    return "MHz";
  }
  if (key.includes("temp")) {
    return "C";
  }
  if (key.includes("memory") && key.includes("rssMB")) {
    return "MB";
  }
  if (key.includes("percent") || key === "cpu" || key === "gpu.0.gpu" || key === "gpu.0.memory") {
    return "%";
  }
  if (key.includes("energy")) {
    return "kWh";
  }
  if (key === "emissions") {
    return "kg CO2e";
  }
  if (key === "emissions_rate") {
    return "kg CO2e/s";
  }
  if (key === "duration") {
    return "s";
  }
  if (key === "water_consumed") {
    return "L";
  }
  if (key === "ram_used_gb") {
    return "GB";
  }
  return "";
}

function formatMetricValue(key, value) {
  const unit = describeMetricUnit(key);
  const formatted = formatCompactNumber(value);
  return unit ? `${formatted} ${unit}` : formatted;
}

function buildSeriesChart(series, stroke = "#1b6b72") {
  if (!Array.isArray(series) || !series.length) {
    return "";
  }

  const width = 320;
  const height = 120;
  const paddingX = 10;
  const paddingY = 10;
  const xs = series.map((point) => Number(point[0]));
  const ys = series.map((point) => Number(point[1]));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1;
  const gridYs = [paddingY, height / 2, height - paddingY];

  const polyline = series
    .map(([rawX, rawY]) => {
      const x = paddingX + ((Number(rawX) - minX) / xSpan) * (width - paddingX * 2);
      const y = height - paddingY - ((Number(rawY) - minY) / ySpan) * (height - paddingY * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const lastX = paddingX + ((xs[xs.length - 1] - minX) / xSpan) * (width - paddingX * 2);
  const lastY = height - paddingY - ((ys[ys.length - 1] - minY) / ySpan) * (height - paddingY * 2);

  return `
    <svg class="wandb-chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">
      <rect class="wandb-chart-frame" x="0.5" y="0.5" width="${width - 1}" height="${height - 1}" rx="14" />
      ${gridYs.map((y) => `<line class="wandb-chart-gridline" x1="${paddingX}" y1="${y}" x2="${width - paddingX}" y2="${y}" />`).join("")}
      <polyline class="wandb-chart-line" points="${polyline}" style="stroke:${stroke}" />
      <circle class="wandb-chart-point" cx="${lastX.toFixed(1)}" cy="${lastY.toFixed(1)}" r="3.5" />
    </svg>
  `;
}

function renderWandbCharts(wandb) {
  const history = { ...(wandb?.history_series || {}), ...(wandb?.system_series || {}) };
  const chartOrder = [
    "reward",
    "average_reward",
    "best_episode_reward",
    "epsilon",
    "training_step",
    "training_phase_code",
    "train_q_network",
    "train_start_network",
    "bonus_existing_connection",
    "penalty_existing_overlap",
    "gpu.0.powerWatts",
    "gpu.0.gpu",
    "gpu.0.memoryAllocated",
    "gpu.0.smClock",
    "gpu.0.memoryClock",
    "cpu",
    "memory_percent",
    "proc.memory.rssMB",
  ];
  const colors = {
    reward: "#1b6b72",
    average_reward: "#ed6a3a",
    best_episode_reward: "#bb4d24",
    epsilon: "#4d908e",
    training_step: "#577590",
    training_phase_code: "#7a3e9d",
    train_q_network: "#0e7c66",
    train_start_network: "#c2553d",
    bonus_existing_connection: "#2f855a",
    penalty_existing_overlap: "#b83232",
    "gpu.0.powerWatts": "#b83232",
    "gpu.0.gpu": "#7a3e9d",
    "gpu.0.memoryAllocated": "#277da1",
    "gpu.0.smClock": "#0e7c66",
    "gpu.0.memoryClock": "#4d908e",
    cpu: "#ed6a3a",
    memory_percent: "#577590",
    "proc.memory.rssMB": "#9f86c0",
  };

  const cards = chartOrder
    .filter((key) => Array.isArray(history[key]) && history[key].length > 1)
    .map((key) => {
      const series = history[key];
      const yValues = series.map((point) => Number(point[1]));
      const latest = yValues[yValues.length - 1];
      const min = Math.min(...yValues);
      const max = Math.max(...yValues);
      const firstX = Number(series[0][0]);
      const lastX = Number(series[series.length - 1][0]);
      const rangeLabel = key in (wandb?.system_series || {}) ? "s" : "ep";
      return `
        <article class="wandb-chart-card">
          <div class="wandb-chart-head">
            <div>
              <div class="wandb-chart-label">${describeSeriesLabel(key)}</div>
              <div class="wandb-chart-value">${formatMetricValue(key, latest)}</div>
            </div>
            <div class="muted">${rangeLabel} ${formatCompactNumber(lastX)}</div>
          </div>
          ${buildSeriesChart(series, colors[key] || "#1b6b72")}
          <div class="wandb-chart-range">
            <span>start ${rangeLabel} ${formatCompactNumber(firstX)}</span>
            <span>min ${formatMetricValue(key, min)} / max ${formatMetricValue(key, max)}</span>
          </div>
        </article>
      `;
    });

  els.wandbChartsGrid.innerHTML = cards.length
    ? cards.join("")
    : `<div class="config-card"><span class="config-value library-value">No scalar history was found for this run.</span></div>`;
}

function canShowBaseMap(model) {
  return state.showBaseMap && model.city_name === "amsterdam" && baseMapImage.complete;
}

function drawBaseMap(canvasContext, model, layout) {
  if (!canShowBaseMap(model)) {
    return;
  }

  const { cellSize, offsetX, offsetY, rows, cols } = layout;
  const targetWidth = cols * cellSize;
  const targetHeight = rows * cellSize;
  const sourceAspect = baseMapImage.naturalWidth / Math.max(baseMapImage.naturalHeight, 1);
  const targetAspect = targetWidth / Math.max(targetHeight, 1);

  let drawWidth = targetWidth;
  let drawHeight = targetHeight;
  if (sourceAspect > targetAspect) {
    drawHeight = targetHeight * state.mapScale;
    drawWidth = drawHeight * sourceAspect;
  } else {
    drawWidth = targetWidth * state.mapScale;
    drawHeight = drawWidth / sourceAspect;
  }

  const drawX = offsetX + (targetWidth - drawWidth) / 2 + state.mapOffsetX;
  const drawY = offsetY + (targetHeight - drawHeight) / 2 + state.mapOffsetY;

  canvasContext.save();
  canvasContext.globalAlpha = state.mapOpacity;
  canvasContext.drawImage(baseMapImage, drawX, drawY, drawWidth, drawHeight);
  canvasContext.restore();
}

function getOverlayCellColor(model, row, col) {
  const layers = model.base_layers;
  const key = state.overlay;
  const source =
    key === "socioeconomic_groups"
      ? model.socioeconomic_groups
      : key === "raw_house_prices"
        ? model.socioeconomic_values
        : layers[key];
  const value = source[row][col];

  if (key === "socioeconomic_groups") {
    if (!Number.isFinite(value)) {
      return "#c8ccd1";
    }
    const colorIndex = Math.max(0, Math.round(value) - 1) % socioeconomicScale.length;
    return socioeconomicScale[colorIndex];
  }

  if (key === "raw_house_prices") {
    return socioeconomicGradientColor(
      value,
      model.socioeconomic_value_min,
      model.socioeconomic_value_max
    );
  }

  const flattened = layers[key].flat().filter((item) => Number.isFinite(item));
  return heatColor(value, Math.min(...flattened), Math.max(...flattened));
}

function drawArrow(row, col, actionIndex, cellSize, offsetX, offsetY) {
  const directions = [
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, -1],
  ];
  const [dr, dc] = directions[actionIndex];
  const cx = offsetX + col * cellSize + cellSize / 2;
  const cy = offsetY + row * cellSize + cellSize / 2;
  const scale = cellSize * 0.22;
  const tx = cx + dc * scale;
  const ty = cy + dr * scale;

  ctx.save();
  ctx.strokeStyle = "rgba(22, 37, 33, 0.28)";
  ctx.lineWidth = Math.max(1, cellSize * 0.06);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(tx, ty);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(tx, ty);
  ctx.lineTo(tx - dc * scale * 0.45 - dr * scale * 0.25, ty - dr * scale * 0.45 + dc * scale * 0.25);
  ctx.lineTo(tx - dc * scale * 0.45 + dr * scale * 0.25, ty - dr * scale * 0.45 - dc * scale * 0.25);
  ctx.closePath();
  ctx.fillStyle = "rgba(22, 37, 33, 0.34)";
  ctx.fill();
  ctx.restore();
}

function drawMap(canvasContext, model, frame, options = {}) {
  const canvas = options.canvas;
  const rows = model.grid.rows;
  const cols = model.grid.cols;
  const padding = 28;
  const drawableWidth = canvas.width - padding * 2;
  const drawableHeight = canvas.height - padding * 2;
  const cellSize = Math.min(drawableWidth / cols, drawableHeight / rows);
  const offsetX = (canvas.width - cellSize * cols) / 2;
  const offsetY = (canvas.height - cellSize * rows) / 2;
  const routeColor = options.routeColor || "#162521";
  const startColor = options.startColor || "#1b6b72";
  const agentColor = options.agentColor || "#ed6a3a";
  const metroColor = options.metroColor || "rgba(9, 35, 79, 0.88)";
  const showAgentHints = options.showAgentHints ?? true;
  const showBaseMap = options.showBaseMap ?? canShowBaseMap(model);

  canvasContext.clearRect(0, 0, canvas.width, canvas.height);
  canvasContext.fillStyle = "#fffaf2";
  canvasContext.fillRect(0, 0, canvas.width, canvas.height);

  if (showBaseMap) {
    drawBaseMap(canvasContext, model, { cellSize, offsetX, offsetY, rows, cols });
  }

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      canvasContext.save();
      if (showBaseMap && (state.overlay === "socioeconomic_groups" || state.overlay === "raw_house_prices")) {
        canvasContext.globalAlpha = 0.58;
      }
      canvasContext.fillStyle = getOverlayCellColor(model, row, col);
      canvasContext.fillRect(offsetX + col * cellSize, offsetY + row * cellSize, cellSize, cellSize);
      canvasContext.restore();
      canvasContext.strokeStyle = "rgba(22, 37, 33, 0.08)";
      canvasContext.strokeRect(offsetX + col * cellSize, offsetY + row * cellSize, cellSize, cellSize);
    }
  }

  canvasContext.save();
  canvasContext.lineCap = "round";
  canvasContext.lineJoin = "round";

  model.existing_lines.forEach((line) => {
    canvasContext.strokeStyle = metroColor;
    canvasContext.lineWidth = Math.max(2, cellSize * 0.18);
    canvasContext.beginPath();
    line.forEach(([row, col], index) => {
      const x = offsetX + col * cellSize + cellSize / 2;
      const y = offsetY + row * cellSize + cellSize / 2;
      if (index === 0) {
        canvasContext.moveTo(x, y);
      } else {
        canvasContext.lineTo(x, y);
      }
    });
    canvasContext.stroke();
  });

  canvasContext.strokeStyle = routeColor;
  canvasContext.lineWidth = Math.max(3, cellSize * 0.24);
  canvasContext.beginPath();
  frame.path.forEach(([row, col], index) => {
    const x = offsetX + col * cellSize + cellSize / 2;
    const y = offsetY + row * cellSize + cellSize / 2;
    if (index === 0) {
      canvasContext.moveTo(x, y);
    } else {
      canvasContext.lineTo(x, y);
    }
  });
  canvasContext.stroke();

  frame.path.forEach(([row, col], index) => {
    const x = offsetX + col * cellSize + cellSize / 2;
    const y = offsetY + row * cellSize + cellSize / 2;
    canvasContext.beginPath();
    canvasContext.arc(x, y, Math.max(4, cellSize * 0.26), 0, Math.PI * 2);
    canvasContext.fillStyle = index === 0 ? startColor : routeColor;
    canvasContext.fill();
  });

  const [agentRow, agentCol] = frame.location;
  const agentX = offsetX + agentCol * cellSize + cellSize / 2;
  const agentY = offsetY + agentRow * cellSize + cellSize / 2;
  const directionMap = [
    [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1],
  ];

  if (showAgentHints) {
    frame.allowed_actions.forEach((allowed, actionIndex) => {
      if (!allowed) {
        return;
      }
      const [dr, dc] = directionMap[actionIndex];
      const nextRow = agentRow + dr;
      const nextCol = agentCol + dc;
      if (nextRow < 0 || nextCol < 0 || nextRow >= rows || nextCol >= cols) {
        return;
      }

      canvasContext.strokeStyle = "rgba(237, 106, 58, 0.38)";
      canvasContext.lineWidth = Math.max(2, cellSize * 0.12);
      canvasContext.setLineDash([cellSize * 0.16, cellSize * 0.12]);
      canvasContext.beginPath();
      canvasContext.moveTo(agentX, agentY);
      canvasContext.lineTo(offsetX + nextCol * cellSize + cellSize / 2, offsetY + nextRow * cellSize + cellSize / 2);
      canvasContext.stroke();
      canvasContext.setLineDash([]);
    });
  }

  canvasContext.fillStyle = agentColor;
  canvasContext.beginPath();
  canvasContext.moveTo(agentX, agentY - cellSize * 0.42);
  canvasContext.lineTo(agentX + cellSize * 0.34, agentY);
  canvasContext.lineTo(agentX, agentY + cellSize * 0.42);
  canvasContext.lineTo(agentX - cellSize * 0.34, agentY);
  canvasContext.closePath();
  canvasContext.fill();
  canvasContext.restore();
}

function renderCanvas() {
  const model = currentModel();
  const frame = currentFrame();
  drawMap(ctx, model, frame, { canvas: els.canvas });
}

function renderComparisonCanvas() {
  const comparisonModel = currentComparisonModel();
  if (!comparisonModel) {
    comparisonCtx.clearRect(0, 0, els.comparisonCanvas.width, els.comparisonCanvas.height);
    els.comparisonTitle.textContent = "No comparison model";
    els.comparisonCopy.textContent = "No same-city comparison model is available for the current selection yet.";
    els.comparisonLegend.innerHTML = "";
    return;
  }

  const comparisonFrame = comparisonModel.timeline[comparisonModel.timeline.length - 1];
  els.comparisonTitle.textContent = `${comparisonModel.kind === "deep_rl" ? "Deep RL" : "Tabular"} comparison`;
  els.comparisonCopy.textContent =
    state.overlay === "raw_house_prices"
      ? `${comparisonModel.title} shown on the Amsterdam house-price gradient with the existing metro network overlaid.`
      : `${comparisonModel.title} shown on the Amsterdam 5-group socio-economic map with the existing metro network overlaid.`;
  drawMap(comparisonCtx, comparisonModel, comparisonFrame, {
    canvas: els.comparisonCanvas,
    routeColor: comparisonModel.kind === "deep_rl" ? "#0e7c66" : "#7a3e9d",
    startColor: comparisonModel.kind === "deep_rl" ? "#38a38b" : "#a25dc2",
    agentColor: comparisonModel.kind === "deep_rl" ? "#0e7c66" : "#7a3e9d",
    metroColor: comparisonModel.kind === "deep_rl" ? "rgba(110, 43, 138, 0.8)" : "rgba(9, 35, 79, 0.88)",
    showAgentHints: false,
    showBaseMap: canShowBaseMap(comparisonModel),
  });

  if (state.overlay === "raw_house_prices") {
    els.comparisonLegend.innerHTML = `
      <div class="legend-item"><span class="swatch" style="background:#c8ccd1"></span><span>No data</span></div>
      <div class="legend-item legend-gradient-item">
        <span class="legend-gradient" style="background:linear-gradient(90deg, ${socioeconomicScale.join(", ")})"></span>
        <span>Average house price: ${formatNumber(comparisonModel.socioeconomic_value_min)}k to ${formatNumber(comparisonModel.socioeconomic_value_max)}k</span>
      </div>
      <div class="legend-item"><span class="swatch" style="background:${comparisonModel.kind === "deep_rl" ? "#0e7c66" : "#7a3e9d"}"></span><span>${comparisonModel.kind === "deep_rl" ? "Deep RL route" : "Tabular route"}</span></div>
    `;
    return;
  }

  const labels = socioeconomicLegendLabels(comparisonModel);
  const groupLegend = labels
    .map((label, index) => {
      const color = socioeconomicScale[index % socioeconomicScale.length];
      return `<div class="legend-item"><span class="swatch" style="background:${color}"></span><span>${label}</span></div>`;
    })
    .join("");
  els.comparisonLegend.innerHTML =
    `<div class="legend-item"><span class="swatch" style="background:#c8ccd1"></span><span>No data</span></div>` +
    groupLegend +
    `<div class="legend-item"><span class="swatch" style="background:${comparisonModel.kind === "deep_rl" ? "#0e7c66" : "#7a3e9d"}"></span><span>${comparisonModel.kind === "deep_rl" ? "Deep RL route" : "Tabular route"}</span></div>`;
}

function renderHeroMetrics() {
  const model = currentModel();
  const summary = model.summary;
  const cards = [
    ["City", model.city_name],
    ["Algorithm", model.kind === "deep_rl" ? "Deep RL" : "Tabular RL"],
    ["Grid", `${model.grid.rows} x ${model.grid.cols}`],
    ["Stations", model.nr_stations],
    ["Reward Groups", model.nr_groups],
    ["Final Reward", formatNumber(summary.final_reward_total)],
  ];

  els.heroMetrics.innerHTML = cards
    .map(
      ([label, value]) => `
        <div class="metric-card">
          <span class="metric-label">${label}</span>
          <span class="metric-value">${value}</span>
        </div>
      `
    )
    .join("");
}

function renderLegend() {
  const model = currentModel();
  if (state.overlay === "socioeconomic_groups") {
    const labels = socioeconomicLegendLabels(model);
    const dataLegend = labels
      .map((label, index) => {
        const color = socioeconomicScale[index % socioeconomicScale.length];
        return `<div class="legend-item"><span class="swatch" style="background:${color}"></span><span>${label}</span></div>`;
      })
      .join("");
    els.legendRow.innerHTML =
      `<div class="legend-item"><span class="swatch" style="background:#c8ccd1"></span><span>No data</span></div>` +
      dataLegend;
    return;
  }

  if (state.overlay === "raw_house_prices") {
    els.legendRow.innerHTML = `
      <div class="legend-item"><span class="swatch" style="background:#c8ccd1"></span><span>No data</span></div>
      <div class="legend-item legend-gradient-item">
        <span class="legend-gradient" style="background:linear-gradient(90deg, ${socioeconomicScale.join(", ")})"></span>
        <span>Average house price: ${formatNumber(model.socioeconomic_value_min)}k to ${formatNumber(model.socioeconomic_value_max)}k</span>
      </div>
    `;
    return;
  }

  const labelMap = {
    aggregate_od: "Origin-destination demand",
    q_best_value: "Best action value",
    q_start: "Starting preference",
  };

  els.legendRow.innerHTML = `
    <div class="legend-item"><span class="swatch" style="background:#f6efe4"></span><span>Lower ${labelMap[state.overlay]}</span></div>
    <div class="legend-item"><span class="swatch" style="background:#f6bd60"></span><span>Mid range</span></div>
    <div class="legend-item"><span class="swatch" style="background:#bb4d24"></span><span>Higher ${labelMap[state.overlay]}</span></div>
  `;
}

function renderDecisionPanel() {
  const model = currentModel();
  const frame = currentFrame();
  const previous = state.step > 0 ? model.timeline[state.step - 1] : null;
  const arrivingReward = frame.reward_total || 0;

  els.panelTitle.textContent = `${model.city_name} route playback`;
  els.stepPill.textContent = `Step ${frame.step} of ${model.timeline.length - 1}`;
  els.decisionTitle.textContent =
    frame.kind === "start"
      ? `Starting at (${frame.location[0]}, ${frame.location[1]})`
      : `${frame.action_label} to (${frame.location[0]}, ${frame.location[1]})`;
  els.decisionCopy.textContent =
    frame.kind === "start"
      ? "The agent begins on the highest-valued starting cell from the learned Q-start table. Dashed rays show the valid first moves."
      : `${frame.description} This move added ${formatNumber(arrivingReward)} reward${previous ? ` and brought the cumulative total to ${formatNumber(frame.cumulative_reward.reduce((sum, value) => sum + value, 0))}.` : "."}`;

  els.allowedActions.innerHTML = frame.allowed_action_labels.length
    ? frame.allowed_action_labels.map((label) => `<div class="chip">${label}</div>`).join("")
    : `<div class="chip">No moves left</div>`;
}

function renderFairnessBreakdown() {
  const model = currentModel();
  const frame = currentFrame();
  const cumulativeVector = frame.cumulative_reward || frame.reward_vector;
  const groupTotals = model.group_totals || [];
  const totalCityDemand = groupTotals.reduce((sum, value) => sum + value, 0);
  const totalServed = cumulativeVector.reduce((sum, value) => sum + value, 0);
  const serviceRates = cumulativeVector.map((value, index) => {
    const denominator = groupTotals[index] || 0;
    return denominator > 0 ? value / denominator : 0;
  });
  const averageServiceRate =
    serviceRates.length > 0 ? serviceRates.reduce((sum, value) => sum + value, 0) / serviceRates.length : 0;
  const gini = giniCoefficient(serviceRates);

  const mostHelpedIndex = serviceRates.reduce(
    (best, value, index, array) => (value > array[best] ? index : best),
    0
  );
  const leastHelpedIndex = serviceRates.reduce(
    (worst, value, index, array) => (value < array[worst] ? index : worst),
    0
  );

  els.fairnessSummary.innerHTML = `
    <div class="metric-card">
      <span class="metric-label">Gini Across Group Service Rates</span>
      <span class="metric-value">${formatNumber(gini)}</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Most Helped Right Now</span>
      <span class="metric-value">${model.group_labels[mostHelpedIndex] || "N/A"}</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Least Helped Right Now</span>
      <span class="metric-value">${model.group_labels[leastHelpedIndex] || "N/A"}</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Total Reward Captured</span>
      <span class="metric-value">${formatNumber(totalServed)}</span>
    </div>
  `;

  els.fairnessBreakdown.innerHTML = cumulativeVector
    .map((value, index) => {
      const cityShare = totalCityDemand > 0 ? groupTotals[index] / totalCityDemand : 0;
      const servedShare = totalServed > 0 ? value / totalServed : 0;
      const serviceRate = serviceRates[index];
      const relativeLift = averageServiceRate > 0 ? serviceRate / averageServiceRate - 1 : 0;
      const balance = servedShare - cityShare;
      const direction =
        balance > 0.01 ? "Advantaged" : balance < -0.01 ? "Underserved" : "Near proportional";
      const fill = averageServiceRate > 0 ? clamp01(serviceRate / Math.max(...serviceRates, averageServiceRate)) : 0;

      return `
        <div class="fairness-card">
          <div class="reward-row">
            <div class="fairness-group-label">
              <span class="swatch" style="background:${socioeconomicScale[index % socioeconomicScale.length]}"></span>
              <strong>${model.group_labels[index]}</strong>
            </div>
            <span>${direction}</span>
          </div>
          <div class="fairness-stats">
            <span>City share: ${formatPercent(cityShare)}</span>
            <span>Served share: ${formatPercent(servedShare)}</span>
            <span>Service rate: ${formatPercent(serviceRate)}</span>
            <span>Lift vs avg: ${relativeLift >= 0 ? "+" : ""}${formatPercent(relativeLift)}</span>
          </div>
          <div class="reward-bar">
            <div class="fairness-fill ${balance >= 0 ? "positive" : "negative"}" style="width:${fill * 100}%"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderTopStarts() {
  const model = currentModel();
  els.topStarts.innerHTML = model.top_starting_cells
    .map(
      (item, index) => `
        <div class="start-row">
          <strong>${index + 1}.</strong>
          <span>(${item.location[0]}, ${item.location[1]})</span>
          <span class="muted">${formatNumber(item.value)}</span>
        </div>
      `
    )
    .join("");
}

function renderStepTable() {
  const model = currentModel();
  els.stepTable.innerHTML = model.timeline
    .slice(1)
    .map((frame) => {
      const isActive = frame.step === state.step ? "active-row" : "";
      return `
        <tr class="${isActive}">
          <td>${frame.step}</td>
          <td>${frame.action_label}</td>
          <td>(${frame.from[0]}, ${frame.from[1]})</td>
          <td>(${frame.to[0]}, ${frame.to[1]})</td>
          <td>${formatNumber(frame.reward_total)}</td>
        </tr>
      `;
    })
    .join("");
}

function renderConfig() {
  const model = currentModel();
  const configItems = [
    ["Environment", model.env_id],
    ["Algorithm", model.kind === "deep_rl" ? "Deep RL" : "Tabular RL"],
    ["Grid size", `${model.grid.rows} rows x ${model.grid.cols} columns`],
    ["OD type", model.od_type],
    ["Reward objective", model.config?.reward_type || "N/A"],
    ["Chained reward", model.chained_reward ? "On" : "Off"],
    ["No-data penalty", model.config?.no_data_penalty ?? "N/A"],
    ["Existing connection bonus", model.config?.existing_connection_bonus ?? "N/A"],
    ["Existing overlap penalty", model.config?.existing_overlap_penalty ?? "N/A"],
    ["Random-start warmup", model.config?.random_start_episodes ?? "N/A"],
    ["Freeze cycle", model.config?.freeze_cycle_episodes ?? "N/A"],
    ["Start cell", `(${model.starting_location[0]}, ${model.starting_location[1]})`],
    ["Steps taken", model.summary.steps_taken],
    ["Final cell", `(${model.summary.final_location[0]}, ${model.summary.final_location[1]})`],
  ];

  els.configGrid.innerHTML = configItems
    .map(
      ([label, value]) => `
        <div class="config-card">
          <span class="config-label">${label}</span>
          <span class="config-value">${value}</span>
        </div>
      `
    )
    .join("");
}

function renderLibrary() {
  els.libraryGrid.innerHTML = dashboardData.models
    .map((model, index) => {
      const metadata = [
        ["Run name", model.id],
        ["City", model.city_name],
        ["Algorithm", model.kind === "deep_rl" ? "Deep RL" : "Tabular RL"],
        ["Reward objective", model.config?.reward_type || "N/A"],
        ["OD type", model.od_type],
        ["Groups", model.nr_groups],
        ["Stations", model.nr_stations],
        ["Chained reward", model.chained_reward],
        ["No-data penalty", model.config?.no_data_penalty ?? "N/A"],
        ["Existing connection bonus", model.config?.existing_connection_bonus ?? "N/A"],
        ["Existing overlap penalty", model.config?.existing_overlap_penalty ?? "N/A"],
        ["Train episodes", model.config?.train_episodes ?? "N/A"],
        ["Batch size", model.config?.batch_size ?? "N/A"],
        ["Learning rate", model.config?.learning_rate ?? "N/A"],
        ["Start LR", model.config?.start_learning_rate ?? "N/A"],
        ["Random-start warmup", model.config?.random_start_episodes ?? "N/A"],
        ["Freeze cycle", model.config?.freeze_cycle_episodes ?? "N/A"],
        ["Epsilon decay", model.config?.epsilon_decay_steps ?? "N/A"],
      ];

      return `
        <article class="library-card ${index === state.modelIndex ? "library-card-active" : ""}">
          <div class="panel-header compact">
            <div>
              <p class="panel-kicker">${model.kind === "deep_rl" ? "Deep RL Run" : "Tabular Run"}</p>
              <h2>${model.title}</h2>
            </div>
            <div class="button-row">
              <button type="button" class="secondary library-inspect-button" data-model-index="${index}">Inspect</button>
              <button type="button" class="secondary library-open-button" data-model-index="${index}">Open</button>
            </div>
          </div>
          <div class="library-meta">
            ${metadata
              .map(
                ([label, value]) => `
                  <div class="config-card">
                    <span class="config-label">${label}</span>
                    <span class="config-value library-value">${formatValue(value)}</span>
                  </div>
                `
              )
              .join("")}
          </div>
        </article>
      `;
    })
    .join("");

  Array.from(document.querySelectorAll(".library-inspect-button")).forEach((button) => {
    button.addEventListener("click", () => {
      stopPlayback();
      state.modelIndex = Number(button.dataset.modelIndex);
      state.comparisonModelId = dashboardData.comparison_defaults?.[currentModel().id] || "";
      populateComparisonSelect();
      renderAll();
      els.wandbPanel?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  Array.from(document.querySelectorAll(".library-open-button")).forEach((button) => {
    button.addEventListener("click", () => {
      stopPlayback();
      state.modelIndex = Number(button.dataset.modelIndex);
      state.step = 0;
      state.comparisonModelId = dashboardData.comparison_defaults?.[currentModel().id] || "";
      populateComparisonSelect();
      state.view = "playback";
      renderAll();
    });
  });
}

function renderManageRuns() {
  els.manageGrid.innerHTML = allModels
    .map((model) => {
      const hidden = Boolean(runManagement.hidden[model.id]);
      const title = runManagement.renames[model.id] || model.originalTitle;
      return `
        <article class="library-card ${hidden ? "library-card-muted" : ""}">
          <div class="manage-row">
            <div>
              <p class="panel-kicker">${model.kind === "deep_rl" ? "Deep RL Run" : "Tabular Run"}</p>
              <h2>${escapeHtml(model.originalTitle)}</h2>
              <p class="decision-copy">${hidden ? "Hidden from dashboard menus" : "Visible in dashboard menus"}</p>
            </div>
            <label class="control">
              <span>Display name</span>
              <input type="text" class="manage-rename-input" data-run-id="${escapeHtml(model.id)}" value="${escapeHtml(title)}" />
            </label>
            <div class="button-row">
              <button type="button" class="secondary manage-save-button" data-run-id="${escapeHtml(model.id)}">Save</button>
              <button type="button" class="${hidden ? "secondary" : "danger"} manage-hide-button" data-run-id="${escapeHtml(model.id)}">
                ${hidden ? "Show" : "Hide"}
              </button>
            </div>
          </div>
        </article>
      `;
    })
    .join("");

  Array.from(document.querySelectorAll(".manage-save-button")).forEach((button) => {
    button.addEventListener("click", () => {
      const runId = button.dataset.runId;
      const input = Array.from(document.querySelectorAll(".manage-rename-input")).find(
        (candidate) => candidate.dataset.runId === runId
      );
      const nextTitle = input.value.trim();
      if (nextTitle) {
        runManagement.renames[runId] = nextTitle;
      } else {
        delete runManagement.renames[runId];
      }
      saveRunManagement();
      applyRunManagement();
      populateModelSelect();
      populateComparisonSelect();
      renderAll();
    });
  });

  Array.from(document.querySelectorAll(".manage-hide-button")).forEach((button) => {
    button.addEventListener("click", () => {
      const runId = button.dataset.runId;
      if (runManagement.hidden[runId]) {
        delete runManagement.hidden[runId];
      } else {
        runManagement.hidden[runId] = true;
      }
      saveRunManagement();
      applyRunManagement();
      populateModelSelect();
      populateComparisonSelect();
      renderAll();
    });
  });
}

function renderWandbDetails() {
  const model = currentModel();
  const wandb = model.wandb;

  if (!wandb) {
    els.wandbTitle.textContent = `${model.title} has no local WandB artifacts`;
    els.wandbSummaryGrid.innerHTML = "";
    els.wandbMetaGrid.innerHTML = "";
    els.wandbChartsGrid.innerHTML = "";
    els.wandbMediaGrid.innerHTML = `<div class="config-card"><span class="config-value library-value">No local WandB summary or media was found for this run.</span></div>`;
    return;
  }

  els.wandbTitle.textContent = `${model.title} (${wandb.run_id})`;

  const summaryOrder = [
    "reward",
    "average_reward",
    "best_episode_reward",
    "epsilon",
    "training_step",
    "training_phase_code",
    "train_q_network",
    "train_start_network",
    "random_start_episodes",
    "freeze_cycle_episodes",
    "episode",
    "_runtime",
    "_step",
  ];
  const summaryEntries = summaryOrder
    .filter((key) => key in (wandb.summary_metrics || {}))
    .map((key) => [key, wandb.summary_metrics[key]]);

  const carbon = wandb.carbon_metrics || {};
  const carbonEntries = [
    ["CodeCarbon duration", "duration"],
    ["Energy consumed", "energy_consumed"],
    ["CO2 emissions", "emissions"],
    ["CPU mean power", "cpu_power"],
    ["GPU mean power", "gpu_power"],
    ["RAM mean power", "ram_power"],
    ["CPU utilization", "cpu_utilization_percent"],
    ["GPU utilization", "gpu_utilization_percent"],
    ["RAM utilization", "ram_utilization_percent"],
    ["RAM used", "ram_used_gb"],
  ]
    .filter(([, key]) => key in carbon)
    .map(([label, key]) => [label, formatMetricValue(key, carbon[key])]);

  const system = wandb.system_summary || {};
  const systemEntries = [
    ["GPU power avg", "gpu.0.powerWatts", "avg"],
    ["GPU power max", "gpu.0.powerWatts", "max"],
    ["GPU SM clock avg", "gpu.0.smClock", "avg"],
    ["GPU memory clock avg", "gpu.0.memoryClock", "avg"],
    ["GPU utilization avg", "gpu.0.gpu", "avg"],
    ["CPU utilization avg", "cpu", "avg"],
    ["RAM utilization avg", "memory_percent", "avg"],
    ["Process memory max", "proc.memory.rssMB", "max"],
  ]
    .filter(([, key, stat]) => system[key] && stat in system[key])
    .map(([label, key, stat]) => [label, formatMetricValue(key, system[key][stat])]);

  els.wandbSummaryGrid.innerHTML = [...summaryEntries, ...carbonEntries, ...systemEntries]
    .map(
      ([label, value]) => `
        <div class="config-card">
          <span class="config-label">${label}</span>
          <span class="config-value library-value">${formatValue(value)}</span>
        </div>
      `
    )
    .join("");

  const metaEntries = [
    ["Started", wandb.metadata?.startedAt],
    ["Program", wandb.metadata?.program],
    ["Python", wandb.metadata?.python],
    ["OS", wandb.metadata?.os],
    ["GPU", wandb.metadata?.gpu],
    ["Logical CPUs", wandb.metadata?.cpu_count_logical],
    ["Args", (wandb.metadata?.args || []).join(" ")],
    ["Output log", wandb.output_log_path || "N/A"],
  ];

  els.wandbMetaGrid.innerHTML = metaEntries
    .map(
      ([label, value]) => `
        <div class="config-card">
          <span class="config-label">${label}</span>
          <span class="config-value library-value">${formatValue(value)}</span>
        </div>
      `
    )
    .join("");

  renderWandbCharts(wandb);

  els.wandbMediaGrid.innerHTML = (wandb.media || []).length
    ? wandb.media
        .map(
          (item) => `
            <article class="wandb-media-card">
              <p class="panel-kicker">${item.label}</p>
              <img src="${item.path}" alt="${item.label}" class="wandb-media-image" loading="lazy" decoding="async" />
            </article>
          `
        )
        .join("")
    : `<div class="config-card"><span class="config-value library-value">No local media was found for this run.</span></div>`;
}

function renderViewState() {
  const playbackActive = state.view === "playback";
  const libraryActive = state.view === "library";
  const manageActive = state.view === "manage";
  els.playbackView.classList.toggle("hidden", !playbackActive);
  els.libraryView.classList.toggle("hidden", !libraryActive);
  els.manageView.classList.toggle("hidden", !manageActive);
  els.playbackButton.classList.toggle("secondary", !playbackActive);
  els.libraryButton.classList.toggle("secondary", !libraryActive);
  els.manageButton.classList.toggle("secondary", !manageActive);
}

function syncControls() {
  const model = currentModel();
  els.stepSlider.max = String(model.timeline.length - 1);
  els.stepSlider.value = String(state.step);
  els.overlaySelect.value = state.overlay;
  els.speedSlider.value = String(state.speed);
  els.mapToggle.checked = state.showBaseMap;
  els.mapOpacitySlider.value = String(Math.round(state.mapOpacity * 100));
  els.mapScaleSlider.value = String(Math.round(state.mapScale * 100));
  els.mapOffsetXSlider.value = String(Math.round(state.mapOffsetX));
  els.mapOffsetYSlider.value = String(Math.round(state.mapOffsetY));
  els.mapCalibrationBar.classList.toggle("hidden", !(state.showBaseMap && currentModel().city_name === "amsterdam"));
  els.playButton.textContent = state.playing ? "Pause" : "Play";
  if (els.comparisonSelect) {
    els.comparisonSelect.value = state.comparisonModelId || "";
  }
}

function renderAll() {
  if (!dashboardData.models.length) {
    state.view = "manage";
    renderViewState();
    renderManageRuns();
    els.heroMetrics.innerHTML = `<div class="metric-card"><span class="metric-label">Visible Runs</span><span class="metric-value">0</span></div>`;
    return;
  }
  renderViewState();
  renderHeroMetrics();
  if (state.view === "playback") {
    syncControls();
    renderLegend();
    renderDecisionPanel();
    renderFairnessBreakdown();
    renderTopStarts();
    renderStepTable();
    renderConfig();
    renderCanvas();
    renderComparisonCanvas();
    return;
  }

  renderLibrary();
  renderManageRuns();
  renderWandbDetails();
}

function stopPlayback() {
  state.playing = false;
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
}

function startPlayback() {
  stopPlayback();
  state.playing = true;
  state.timer = setInterval(() => {
    const maxStep = currentModel().timeline.length - 1;
    if (state.step >= maxStep) {
      stopPlayback();
      renderAll();
      return;
    }
    state.step += 1;
    renderAll();
  }, state.speed);
}

function bindEvents() {
  els.playbackButton.addEventListener("click", () => {
    state.view = "playback";
    renderAll();
  });

  els.libraryButton.addEventListener("click", () => {
    stopPlayback();
    state.view = "library";
    renderAll();
  });

  els.manageButton.addEventListener("click", () => {
    stopPlayback();
    state.view = "manage";
    renderAll();
  });

  els.resetRunManagement.addEventListener("click", () => {
    runManagement = { renames: {}, hidden: {} };
    saveRunManagement();
    applyRunManagement({ preserveCurrent: false });
    populateModelSelect();
    populateComparisonSelect();
    renderAll();
  });

  els.modelSelect.addEventListener("change", (event) => {
    if (!dashboardData.models.length) {
      return;
    }
    stopPlayback();
    state.modelIndex = Number(event.target.value);
    state.step = 0;
    state.comparisonModelId = dashboardData.comparison_defaults?.[currentModel().id] || "";
    populateComparisonSelect();
    renderAll();
  });

  els.overlaySelect.addEventListener("change", (event) => {
    state.overlay = event.target.value;
    renderAll();
  });

  els.mapToggle.addEventListener("change", (event) => {
    state.showBaseMap = event.target.checked;
    saveMapSettings();
    renderAll();
  });

  els.mapOpacitySlider.addEventListener("input", (event) => {
    state.mapOpacity = Number(event.target.value) / 100;
    saveMapSettings();
    renderAll();
  });

  els.mapScaleSlider.addEventListener("input", (event) => {
    state.mapScale = Number(event.target.value) / 100;
    saveMapSettings();
    renderAll();
  });

  els.mapOffsetXSlider.addEventListener("input", (event) => {
    state.mapOffsetX = Number(event.target.value);
    saveMapSettings();
    renderAll();
  });

  els.mapOffsetYSlider.addEventListener("input", (event) => {
    state.mapOffsetY = Number(event.target.value);
    saveMapSettings();
    renderAll();
  });

  els.stepSlider.addEventListener("input", (event) => {
    stopPlayback();
    state.step = Number(event.target.value);
    renderAll();
  });

  els.playButton.addEventListener("click", () => {
    if (state.playing) {
      stopPlayback();
      renderAll();
      return;
    }
    startPlayback();
    renderAll();
  });

  els.resetButton.addEventListener("click", () => {
    stopPlayback();
    state.step = 0;
    renderAll();
  });

  els.speedSlider.addEventListener("input", (event) => {
    state.speed = Number(event.target.value);
    if (state.playing) {
      startPlayback();
    }
  });

  els.comparisonSelect.addEventListener("change", (event) => {
    state.comparisonModelId = event.target.value || null;
    renderAll();
  });

  window.addEventListener("resize", () => {
    renderCanvas();
    renderComparisonCanvas();
  });
}

function populateComparisonSelect() {
  if (!dashboardData.models.length) {
    els.comparisonSelect.innerHTML = `<option value="">No comparison available</option>`;
    state.comparisonModelId = null;
    return;
  }
  const selected = currentModel();
  const candidates = dashboardData.models.filter((model) => model.city_name === selected.city_name && model.id !== selected.id);
  if (!candidates.length) {
    els.comparisonSelect.innerHTML = `<option value="">No comparison available</option>`;
    state.comparisonModelId = null;
    return;
  }

  if (!state.comparisonModelId || !candidates.some((model) => model.id === state.comparisonModelId)) {
    state.comparisonModelId = dashboardData.comparison_defaults?.[selected.id] || candidates[0].id;
  }

  els.comparisonSelect.innerHTML = candidates
    .map((model) => `<option value="${model.id}">${model.title}</option>`)
    .join("");
}

function init() {
  loadMapSettings();
  loadRunManagement();
  applyRunManagement({ preserveCurrent: false });

  bindEvents();
  populateModelSelect();
  populateComparisonSelect();
  baseMapImage.addEventListener("load", () => renderAll(), { once: true });
  renderAll();
}

init();
