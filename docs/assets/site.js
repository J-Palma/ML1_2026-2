const metricLabels = {
  "F1 Macro": "F1 macro",
  "Accuracy (%)": "Accuracy (%)",
  "Balanced Acc (%)": "Balanced accuracy (%)",
};

function formatMetric(value, key) {
  return key === "F1 Macro" ? Number(value).toFixed(3) : `${Number(value).toFixed(1)}%`;
}

function renderModels(models, key) {
  const chart = document.querySelector("#model-chart");
  const table = document.querySelector("#model-table");
  const sorted = [...models].sort((a, b) => b[key] - a[key]);
  const maxValue = key === "F1 Macro" ? 1 : 100;
  chart.innerHTML = sorted.map((model, index) => {
    const width = Math.max(1, Math.min(100, (Number(model[key]) / maxValue) * 100));
    return `<div class="model-bar-row"><span class="model-bar-label">${model.Modelo}</span><div class="model-track" role="img" aria-label="${metricLabels[key]}: ${formatMetric(model[key], key)}"><span class="model-fill ${index === 0 ? "best" : ""}" style="width:${width}%"></span></div><span class="model-value">${formatMetric(model[key], key)}</span></div>`;
  }).join("");
  table.innerHTML = sorted.map((model) => `<tr><td>${model.Modelo}</td><td>${Number(model["Accuracy (%)"]).toFixed(2)}%</td><td>${Number(model["Balanced Acc (%)"]).toFixed(2)}%</td><td>${Number(model["F1 Macro"]).toFixed(4)}</td><td>${Number(model["Precision Macro"]).toFixed(4)}</td><td>${Number(model["Recall Macro"]).toFixed(4)}</td></tr>`).join("");
}

async function loadPageData() {
  try {
    const [metricsResponse, datasetResponse] = await Promise.all([
      fetch("data/model_metrics.json"),
      fetch("data/dataset_summary.json"),
    ]);
    if (!metricsResponse.ok || !datasetResponse.ok) throw new Error("No se pudieron cargar los datos del sitio.");
    const metrics = await metricsResponse.json();
    const dataset = await datasetResponse.json();
    const models = [...metrics.comparison_table].sort((a, b) => b["F1 Macro"] - a["F1 Macro"]);
    const winner = models[0];
    document.querySelector("#winner-name").textContent = winner.Modelo;
    document.querySelector("#winner-f1").textContent = Number(winner["F1 Macro"]).toFixed(3).replace(".", ",");
    renderModels(models, "F1 Macro");
    document.querySelector("#metric-select").addEventListener("change", (event) => renderModels(models, event.target.value));

    const classContainer = document.querySelector("#class-list");
    const largest = Math.max(...dataset.classes.map((item) => item.count));
    classContainer.innerHTML = dataset.classes.map((item) => `<div class="class-row"><span>${item.name}</span><div class="class-bar"><span style="width:${(item.count / largest) * 100}%"></span></div><span class="class-count">${item.count}</span></div>`).join("");
  } catch (error) {
    document.querySelector("#model-chart").textContent = `${error.message} Ejecutá el sitio desde el directorio docs.`;
  }
}

const menuButton = document.querySelector(".menu-toggle");
menuButton?.addEventListener("click", () => {
  const nav = document.querySelector(".nav-links");
  const open = nav.classList.toggle("open");
  menuButton.setAttribute("aria-expanded", String(open));
});
document.querySelectorAll(".nav-links a").forEach((link) => link.addEventListener("click", () => {
  document.querySelector(".nav-links").classList.remove("open");
  menuButton?.setAttribute("aria-expanded", "false");
}));

loadPageData();
