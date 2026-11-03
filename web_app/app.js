// app.js - Full-Stack Urban Traffic GNN Presentation & Simulation Engine
let graphData = null;
let currentStep = 0;
let isPlaying = false;
let playInterval = null;
let selectedNode = 12;
let chartInstance = null;
let testPredictions = {};
let groundTruth = null;
let metricsSummary = null;

// Presentation State
let currentSlide = 1;
const totalSlides = 6;
const slideTitles = [
  "Introduction & Smart City Domain",
  "Why GNN vs. CNN / LSTM (Non-Euclidean Roads)",
  "The Real-World Class Imbalance Challenge",
  "Proposed Deep Learning Architecture (A-STGCN)",
  "Empirical Benchmark & Ablation Results",
  "Viva Voce Defense & Examiner Q&A Cheat Sheet"
];

// Node coordinates scaled for canvas
let canvasNodes = [];
const canvas = document.getElementById("graphCanvas");
const ctx = canvas.getContext("2d");

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  renderGraph();
}
window.addEventListener("resize", resizeCanvas);

// Main Initialization
async function initializeApp() {
  try {
    const resMeta = await fetch("/api/metadata");
    graphData = await resMeta.json();

    const resMetrics = await fetch("/api/metrics");
    metricsSummary = await resMetrics.json();

    const resPreds = await fetch("/api/predictions");
    const predPayload = await resPreds.json();
    testPredictions = predPayload.predictions;
    groundTruth = predPayload.ground_truth;

    setupKPIs();
    setupTable();
    setupGraphCoordinates();
    resizeCanvas();
    setupEventListeners();
    setupPresentationControls();
    updateSelectedNode(12);
  } catch (err) {
    console.error("Failed to load backend data:", err);
  }
}

function setupKPIs() {
  if (!metricsSummary || !metricsSummary.Improved_A_STGCN) return;
  const imp = metricsSummary.Improved_A_STGCN;
  const base = metricsSummary.Baseline_LSTM;
  const baseGnn = metricsSummary.Base_STGCN;

  document.getElementById("kpiMae").innerText = `${imp.test_metrics.MAE} mph`;
  const delta = (((base.test_metrics.MAE - imp.test_metrics.MAE) / base.test_metrics.MAE) * 100).toFixed(1);
  document.getElementById("kpiMaeDelta").innerText = `${delta}% error cut`;

  document.getElementById("kpiJamMae").innerText = `${imp.regime_metrics.severe_congestion_regime.MAE} mph`;
  
  if (graphData && graphData.imbalance_stats) {
    const st = graphData.imbalance_stats;
    document.getElementById("kpiImbalance").innerText = `${st.free_flow_pct}% / ${(st.moderate_congestion_pct + st.severe_gridlock_pct).toFixed(1)}%`;
  }
}

function setupTable() {
  const tbody = document.getElementById("metricsTableBody");
  if (!metricsSummary) return;

  const rows = [
    { key: "Baseline_LSTM", name: "Baseline LSTM", type: "Pure Temporal (Recurrent)", spatial: "❌ Blind to road topology" },
    { key: "Base_STGCN", name: "Base ST-GCN", type: "Spatio-Temporal (Normalized Laplacian)", spatial: "✓ Static Graph Convolutions" },
    { key: "Improved_A_STGCN", name: "Improved A-STGCN (Ours)", type: "Attention GNN + Congestion Focal Loss", spatial: "✓ Adaptive Graph Attn + Multi-Head Temporal" }
  ];

  tbody.innerHTML = rows.map(r => {
    const data = metricsSummary[r.key];
    if (!data) return "";
    const isWinner = r.key === "Improved_A_STGCN";
    return `
      <tr style="${isWinner ? 'background: rgba(0, 229, 255, 0.05); font-weight: 600;' : ''}">
        <td>${r.name} ${isWinner ? '<span class="winner-tag">BEST</span>' : ''}</td>
        <td style="color:var(--text-muted);">${r.type}</td>
        <td style="color:${isWinner ? '#00e676' : 'inherit'}; font-weight:700;">${data.test_metrics.MAE} mph</td>
        <td>${data.test_metrics.RMSE} mph</td>
        <td style="color:${isWinner ? '#00e676' : '#ff5252'}; font-weight:700;">${data.regime_metrics.severe_congestion_regime.MAE} mph</td>
        <td>${(data.regime_metrics.bottleneck_detection.Recall * 100).toFixed(1)}%</td>
        <td>${(data.regime_metrics.bottleneck_detection.F1_Score * 100).toFixed(1)}%</td>
        <td>${r.spatial}</td>
      </tr>
    `;
  }).join("");
}

function setupGraphCoordinates() {
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;

  canvasNodes = [];
  const rawPositions = graphData.node_positions;
  const numNodes = graphData.num_nodes;

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < numNodes; i++) {
    const [x, y] = rawPositions[i.toString()];
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }

  const paddingX = 70;
  const paddingY = 60;

  for (let i = 0; i < numNodes; i++) {
    const [x, y] = rawPositions[i.toString()];
    const cx = paddingX + ((x - minX) / (maxX - minX)) * (width - 2 * paddingX);
    const cy = paddingY + ((y - minY) / (maxY - minY)) * (height - 2 * paddingY);
    canvasNodes.push({ id: i, x: cx, y: cy });
  }
}

function getSpeedColor(speed) {
  if (speed >= 50.0) return "#00e676"; // Free flow green
  if (speed >= 30.0) return "#ffc400"; // Moderate amber
  return "#ff1744"; // Severe red
}

function renderGraph() {
  if (!graphData || canvasNodes.length === 0) return;
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);

  const modelKey = document.getElementById("modelSelect").value;
  const currentSpeeds = getCurrentStepSpeeds(modelKey, currentStep);

  // Draw Edges
  const adj = graphData.adj_matrix;
  const N = graphData.num_nodes;

  ctx.lineWidth = 2.0;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      if (adj[i][j] > 0.1) {
        ctx.strokeStyle = "rgba(41, 121, 255, 0.22)";
        ctx.beginPath();
        ctx.moveTo(canvasNodes[i].x, canvasNodes[i].y);
        ctx.lineTo(canvasNodes[j].x, canvasNodes[j].y);
        ctx.stroke();
      }
    }
  }

  // Draw Nodes
  for (let i = 0; i < N; i++) {
    const node = canvasNodes[i];
    const speed = currentSpeeds[i] || 60.0;
    const color = getSpeedColor(speed);
    const isSelected = (i === selectedNode);

    ctx.beginPath();
    ctx.arc(node.x, node.y, isSelected ? 18 : 12, 0, Math.PI * 2);
    ctx.fillStyle = isSelected ? "rgba(0, 229, 255, 0.25)" : (speed < 30 ? "rgba(255, 23, 68, 0.25)" : "rgba(0, 230, 118, 0.12)");
    ctx.fill();

    ctx.beginPath();
    ctx.arc(node.x, node.y, isSelected ? 11 : 8, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.strokeStyle = isSelected ? "#ffffff" : "#0f1726";
    ctx.lineWidth = isSelected ? 2.5 : 1.5;
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = isSelected ? "bold 11px sans-serif" : "9px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`${i}`, node.x, node.y - (isSelected ? 20 : 15));
  }
}

function getCurrentStepSpeeds(modelKey, step) {
  if (!groundTruth || !testPredictions) return [];
  const safeStep = Math.min(step, groundTruth.length - 1);
  if (modelKey === "Ground_Truth") {
    return groundTruth[safeStep][0];
  }
  const preds = testPredictions[modelKey];
  return preds ? preds[safeStep][0] : groundTruth[safeStep][0];
}

function setupEventListeners() {
  const slider = document.getElementById("timeSlider");
  const playBtn = document.getElementById("playBtn");
  const modelSelect = document.getElementById("modelSelect");

  if (groundTruth) {
    slider.max = Math.min(287, groundTruth.length - 1);
  }

  slider.addEventListener("input", (e) => {
    currentStep = parseInt(e.target.value);
    updateStepDisplay();
    renderGraph();
    updateNodeCard();
  });

  playBtn.addEventListener("click", () => {
    if (isPlaying) {
      clearInterval(playInterval);
      playBtn.innerText = "▶ Play";
      isPlaying = false;
    } else {
      playBtn.innerText = "⏸ Pause";
      isPlaying = true;
      playInterval = setInterval(() => {
        currentStep = (currentStep + 1) % (parseInt(slider.max) + 1);
        slider.value = currentStep;
        updateStepDisplay();
        renderGraph();
        updateNodeCard();
      }, 250);
    }
  });

  modelSelect.addEventListener("change", () => {
    renderGraph();
    updateNodeCard();
  });

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    for (let i = 0; i < canvasNodes.length; i++) {
      const node = canvasNodes[i];
      const dist = Math.hypot(node.x - clickX, node.y - clickY);
      if (dist < 18) {
        updateSelectedNode(i);
        break;
      }
    }
  });

  // View switchers
  const btnViewSim = document.getElementById("btnViewSim");
  const btnViewPres = document.getElementById("btnViewPres");
  const simView = document.getElementById("simulatorView");
  const presView = document.getElementById("presentationView");

  btnViewSim.addEventListener("click", () => {
    btnViewSim.classList.add("active");
    btnViewPres.classList.remove("active");
    simView.style.display = "block";
    presView.style.display = "none";
    resizeCanvas();
  });

  btnViewPres.addEventListener("click", () => {
    btnViewPres.classList.add("active");
    btnViewSim.classList.remove("active");
    simView.style.display = "none";
    presView.style.display = "block";
    updateSlideView();
  });
}

function updateStepDisplay() {
  const totalMins = currentStep * 5;
  const hours = Math.floor(totalMins / 60) % 24;
  const mins = totalMins % 60;
  const timeStr = `${hours.toString().padStart(2, "0")}:${mins.toString().padStart(2, "0")}`;
  document.getElementById("sliderValueText").innerText = timeStr;
  document.getElementById("currentStepLabel").innerText = `Time: ${timeStr} (Step ${currentStep} of 288)`;
}

function updateSelectedNode(nodeId) {
  selectedNode = nodeId;
  renderGraph();
  updateNodeCard();
  renderTrajectoryChart(nodeId);
}

function updateNodeCard() {
  document.getElementById("nodeCardTitle").innerText = `📍 Sensor Node #${selectedNode}`;
  
  const isBottleneck = (4 <= selectedNode && selectedNode <= 19);
  const badge = document.getElementById("nodeTypeBadge");
  badge.innerText = isBottleneck ? "Bottleneck Corridor" : "Highway Arterial";
  badge.className = isBottleneck ? "badge imbalance" : "badge";

  const modelKey = document.getElementById("modelSelect").value;
  const speeds = getCurrentStepSpeeds(modelKey, currentStep);
  const currentSpeed = speeds[selectedNode] ? speeds[selectedNode].toFixed(1) : "--";
  document.getElementById("nodeCurrentSpeed").innerText = `${currentSpeed} mph`;

  const risk = currentSpeed < 30 ? "CRITICAL CONGESTION" : (currentSpeed < 50 ? "MODERATE SLOWDOWN" : "OPTIMAL FREE-FLOW");
  const riskElem = document.getElementById("nodeRiskLevel");
  riskElem.innerText = risk;
  riskElem.style.color = currentSpeed < 30 ? "#ff1744" : (currentSpeed < 50 ? "#ffc400" : "#00e676");

  const adj = graphData.adj_matrix;
  const neighbors = [];
  for (let j = 0; j < graphData.num_nodes; j++) {
    if (adj[selectedNode][j] > 0.1) neighbors.push(j);
  }
  document.getElementById("nodeNeighbors").innerText = neighbors.length ? `Nodes ${neighbors.join(", ")}` : "None";
}

function renderTrajectoryChart(nodeId) {
  if (!groundTruth || !testPredictions) return;

  const ctxChart = document.getElementById("nodeTrajectoryChart").getContext("2d");
  const numSteps = Math.min(144, groundTruth.length);
  const labels = [];
  for (let s = 0; s < numSteps; s++) {
    const totalMins = s * 5;
    const h = Math.floor(totalMins / 60) % 24;
    const m = totalMins % 60;
    labels.push(`${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}`);
  }

  const truthSeries = [];
  const lstmSeries = [];
  const baseGnnSeries = [];
  const impGnnSeries = [];

  for (let s = 0; s < numSteps; s++) {
    truthSeries.push(groundTruth[s][0][nodeId]);
    lstmSeries.push(testPredictions.Baseline_LSTM[s][0][nodeId]);
    baseGnnSeries.push(testPredictions.Base_STGCN[s][0][nodeId]);
    impGnnSeries.push(testPredictions.Improved_A_STGCN[s][0][nodeId]);
  }

  if (chartInstance) {
    chartInstance.destroy();
  }

  chartInstance = new Chart(ctxChart, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        { label: "Actual Ground Truth", data: truthSeries, borderColor: "#ffffff", borderWidth: 2.2, pointRadius: 0 },
        { label: "Baseline LSTM", data: lstmSeries, borderColor: "#2979ff", borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0 },
        { label: "Base ST-GCN", data: baseGnnSeries, borderColor: "#ffc400", borderWidth: 1.5, borderDash: [2, 2], pointRadius: 0 },
        { label: "Improved A-STGCN (Ours)", data: impGnnSeries, borderColor: "#ff1744", borderWidth: 2.0, pointRadius: 0 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#8a99ad", font: { size: 11 } } },
        tooltip: { mode: "index", intersect: false }
      },
      scales: {
        x: { ticks: { color: "#8a99ad", maxTicksLimit: 8 }, grid: { color: "#1a2538" } },
        y: { ticks: { color: "#8a99ad" }, grid: { color: "#1a2538" }, min: 5, max: 75, title: { display: true, text: "Speed (mph)", color: "#8a99ad" } }
      }
    }
  });
}

// Presentation Deck Logic
function setupPresentationControls() {
  const prevBtn = document.getElementById("prevSlideBtn");
  const nextBtn = document.getElementById("nextSlideBtn");
  const fsBtn = document.getElementById("fullscreenBtn");

  prevBtn.addEventListener("click", () => {
    if (currentSlide > 1) {
      currentSlide--;
      updateSlideView();
    }
  });

  nextBtn.addEventListener("click", () => {
    if (currentSlide < totalSlides) {
      currentSlide++;
      updateSlideView();
    }
  });

  fsBtn.addEventListener("click", () => {
    const el = document.getElementById("slideContainer");
    if (!document.fullscreenElement) {
      el.requestFullscreen().catch(err => console.log(err));
    } else {
      document.exitFullscreen();
    }
  });

  // Keyboard navigation
  window.addEventListener("keydown", (e) => {
    const presView = document.getElementById("presentationView");
    if (presView.style.display !== "none") {
      if (e.key === "ArrowRight" || e.key === "PageDown" || e.key === " ") {
        if (currentSlide < totalSlides) {
          currentSlide++;
          updateSlideView();
        }
      } else if (e.key === "ArrowLeft" || e.key === "PageUp") {
        if (currentSlide > 1) {
          currentSlide--;
          updateSlideView();
        }
      }
    }
  });
}

function updateSlideView() {
  document.getElementById("currentSlideNum").innerText = currentSlide;
  document.getElementById("totalSlidesNum").innerText = totalSlides;
  document.getElementById("slideTitleHeading").innerText = slideTitles[currentSlide - 1];

  const slides = document.querySelectorAll(".slide-content");
  slides.forEach(s => {
    if (parseInt(s.getAttribute("data-slide")) === currentSlide) {
      s.classList.add("active");
    } else {
      s.classList.remove("active");
    }
  });

  // Button disabled state
  document.getElementById("prevSlideBtn").disabled = (currentSlide === 1);
  document.getElementById("nextSlideBtn").disabled = (currentSlide === totalSlides);
}

// Viva Accordion Toggle
window.toggleQA = function(cardElem) {
  cardElem.classList.toggle("open");
};

// Launch App
initializeApp();
