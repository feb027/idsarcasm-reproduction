const DATA_PATH = "./dashboard/data/dashboard-data.json";

const chartLabelPlugin = {
  id: "valueLabel",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    ctx.save();
    ctx.font = "600 11px 'Fira Code', monospace";
    ctx.fillStyle = "#1E293B";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";

    chart.data.datasets.forEach((dataset, datasetIndex) => {
      const meta = chart.getDatasetMeta(datasetIndex);
      if (meta.hidden) {
        return;
      }

      meta.data.forEach((element, index) => {
        const value = dataset.data[index];
        if (value === null || value === undefined) {
          return;
        }
        const yOffset = chart.config.type === "bar" ? 6 : 10;
        ctx.fillText(Number(value).toFixed(3), element.x, element.y - yOffset);
      });
    });

    ctx.restore();
  },
};

const state = {
  data: null,
  comparisonChart: null,
  rankingChart: null,
};

const formatF1 = (value) => {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  return Number(value).toFixed(4);
};

const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
}[char]));
const formatCompact = (value) => new Intl.NumberFormat("en-US").format(value);

function getStrategyThreshold(datasetData, strategy) {
  if (strategy === "default") {
    return 0.5;
  }
  if (strategy === "tuned") {
    return datasetData.selected_threshold;
  }
  return datasetData.selected_strategy === "default" ? 0.5 : datasetData.selected_threshold;
}

function iconSvg(type) {
  const icons = {
    database:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><ellipse cx="12" cy="5" rx="8" ry="3"></ellipse><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"></path><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"></path></svg>',
    target:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"></circle><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M22 12h-2M12 22v-2M2 12h2"></path></svg>',
    line:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m3 17 6-6 4 4 8-8"></path><path d="M14 7h7v7"></path></svg>',
    split:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 3h5v5"></path><path d="M4 20 21 3"></path><path d="M21 16v5h-5"></path><path d="M15 15 21 21"></path><path d="M4 4l5 5"></path></svg>',
    check:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 5 5L20 7"></path></svg>',
    alert:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 9v4"></path><path d="M12 17h.01"></path><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"></path></svg>',
  };
  return icons[type] || icons.database;
}

function createKpiCard({ title, value, meta, icon }) {
  return `
    <article class="kpi-card">
      <div class="kpi-top">
        <div>
          <dt>${escapeHtml(title)}</dt>
          <dd class="kpi-value">${escapeHtml(value)}</dd>
        </div>
        <span class="kpi-icon">${iconSvg(icon)}</span>
      </div>
      <p class="kpi-meta">${escapeHtml(meta)}</p>
    </article>
  `;
}

function renderHeroMeta(data) {
  const generatedAt = new Date(data.generated_at);
  document.getElementById("generated-at").textContent = generatedAt.toLocaleString("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC",
  }) + " UTC";
}

function renderKpis(data) {
  const kpiGrid = document.getElementById("kpi-grid");
  const twitterGap = data.gap_cards.find((card) => card.dataset === "twitter");
  const redditGap = data.gap_cards.find((card) => card.dataset === "reddit");

  kpiGrid.innerHTML = [
    createKpiCard({
      title: "Datasets",
      value: String(data.overview.dataset_count),
      meta: "Twitter and Reddit are both summarized in the final benchmark view.",
      icon: "database",
    }),
    createKpiCard({
      title: "Test Rows",
      value: formatCompact(data.overview.test_rows),
      meta: "Explorer rows come from the final selected runs only.",
      icon: "split",
    }),
    createKpiCard({
      title: "Best Optimized F1",
      value: formatF1(data.overview.best_overall_f1),
      meta: "Best overall test F1 across selected final strategies.",
      icon: "target",
    }),
    createKpiCard({
      title: "Paper Gap Snapshot",
      value: `${twitterGap.gap_f1 >= 0 ? "+" : ""}${twitterGap.gap_f1.toFixed(4)} / ${redditGap.gap_f1.toFixed(4)}`,
      meta: "Twitter first, Reddit second, measured against paper-reported best F1.",
      icon: "line",
    }),
  ].join("");
}

function buildComparisonChart(data) {
  const methodLabels = ["Classical ML", "Transformer baseline", "Optimized transformer", "Zero-shot paper LLM", "Modern local LLM", "Paper best"];
  const twitterRow = data.comparison.find((row) => row.dataset === "twitter");
  const redditRow = data.comparison.find((row) => row.dataset === "reddit");

  const twitterData = [
    twitterRow.methods.find((m) => m.key === "classical").f1,
    twitterRow.methods.find((m) => m.key === "transformer").f1,
    twitterRow.methods.find((m) => m.key === "optimized").f1,
    twitterRow.methods.find((m) => m.key === "zero_shot").f1,
    twitterRow.methods.find((m) => m.key === "modern_llm").f1,
    twitterRow.paper_best_f1,
  ];

  const redditData = [
    redditRow.methods.find((m) => m.key === "classical").f1,
    redditRow.methods.find((m) => m.key === "transformer").f1,
    redditRow.methods.find((m) => m.key === "optimized").f1,
    redditRow.methods.find((m) => m.key === "zero_shot").f1,
    redditRow.methods.find((m) => m.key === "modern_llm").f1,
    redditRow.paper_best_f1,
  ];

  const ctx = document.getElementById("comparison-chart");
  state.comparisonChart = new Chart(ctx, {
    type: "bar",
    plugins: [chartLabelPlugin],
    data: {
      labels: methodLabels,
      datasets: [
        {
          label: "Twitter",
          data: twitterData,
          backgroundColor: "#2563EB",
          borderRadius: 8,
        },
        {
          label: "Reddit",
          data: redditData,
          backgroundColor: "#93C5FD",
          borderRadius: 8,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      animation: !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
      scales: {
        y: {
          min: 0,
          max: 1,
          ticks: {
            callback: (value) => Number(value).toFixed(1),
          },
        },
      },
      plugins: {
        legend: {
          position: "bottom",
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${Number(context.raw).toFixed(4)}`,
          },
        },
      },
    },
  });

  const detail = document.getElementById("comparison-detail");
  detail.innerHTML = data.comparison
    .map(
      (row) => `
        <section class="metric-row">
          <header>
            <strong>${escapeHtml(row.dataset_label)}</strong>
            <span class="table-tag">${formatF1(row.paper_best_f1)} paper</span>
          </header>
          <ul class="mini-list">
            ${row.methods
              .map(
                (method) => `
                  <li>
                    <span>${escapeHtml(method.label)}</span>
                    <span>${method.f1 === null ? "n/a" : `${formatF1(method.f1)} · ${escapeHtml(method.model)}`}</span>
                  </li>
                `
              )
              .join("")}
          </ul>
        </section>
      `
    )
    .join("");
}

function buildRankingChart(data) {
  const labels = [];
  const values = [];
  const colors = [];

  ["twitter", "reddit"].forEach((dataset) => {
    data.ranking[dataset].forEach((row) => {
      labels.push(`${dataset.toUpperCase()} #${row.rank}`);
      values.push(row.f1);
      colors.push(dataset === "twitter" ? "#2563EB" : "#60A5FA");
    });
  });

  const ctx = document.getElementById("ranking-chart");
  state.rankingChart = new Chart(ctx, {
    type: "bar",
    plugins: [chartLabelPlugin],
    data: {
      labels,
      datasets: [
        {
          label: "F1",
          data: values,
          backgroundColor: colors,
          borderRadius: 8,
        },
      ],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      animation: !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
      scales: {
        x: {
          min: 0,
          max: 1,
        },
      },
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label: (context) => `${Number(context.raw).toFixed(4)}`,
          },
        },
      },
    },
  });

  document.getElementById("ranking-detail").innerHTML = ["twitter", "reddit"]
    .map(
      (dataset) => `
        <section class="ranking-row">
          <header>
            <strong>${dataset[0].toUpperCase()}${dataset.slice(1)}</strong>
            <span class="table-tag">${data.ranking[dataset].length} ranked methods</span>
          </header>
          <ul class="mini-list">
            ${data.ranking[dataset]
              .map(
                (row) => `
                  <li>
                    <span>#${row.rank} ${escapeHtml(row.method_family)}</span>
                    <span>${formatF1(row.f1)} · ${escapeHtml(row.model)}</span>
                  </li>
                `
              )
              .join("")}
          </ul>
        </section>
      `
    )
    .join("");
}

function renderGapCards(data) {
  document.getElementById("gap-grid").innerHTML = data.gap_cards
    .map((card) => {
      const isAbove = card.status === "above";
      return `
        <article class="gap-card">
          <header>
            <div>
              <p class="eyebrow">${escapeHtml(card.dataset_label)}</p>
              <h3>${escapeHtml(card.optimized_model)}</h3>
            </div>
            <span class="gap-status" data-status="${card.status}">
              <span class="status-icon">${iconSvg(isAbove ? "check" : "alert")}</span>
              ${isAbove ? "Above paper" : "Below paper"}
            </span>
          </header>
          <div class="summary-list">
            <span>Selected strategy</span>
            <code>${escapeHtml(card.selected_strategy)}</code>
          </div>
          <div class="summary-list">
            <span>Optimization method</span>
            <code>${escapeHtml(card.optimization_strategy)}</code>
          </div>
          <div class="summary-list">
            <span>Optimized F1</span>
            <strong>${formatF1(card.optimized_f1)}</strong>
          </div>
          <div class="summary-list">
            <span>Paper F1</span>
            <strong>${formatF1(card.paper_f1)}</strong>
          </div>
          <div class="summary-list gap-meta">
            <span>Gap</span>
            <strong>${card.gap_f1 >= 0 ? "+" : ""}${card.gap_f1.toFixed(4)}</strong>
          </div>
        </article>
      `;
    })
    .join("");
}

function setupFilters(data) {
  const datasetFilter = document.getElementById("dataset-filter");
  data.explorer.datasets.forEach((dataset) => {
    const option = document.createElement("option");
    option.value = dataset.dataset;
    option.textContent = dataset.dataset_label;
    datasetFilter.appendChild(option);
  });

  ["dataset-filter", "outcome-filter", "strategy-filter", "label-filter", "search-filter"].forEach((id) => {
    document.getElementById(id).addEventListener("input", () => renderExplorer(data));
  });

  renderExplorer(data);
}

function updateDatasetSummary(datasetData, strategy) {
  const selectedMetrics = datasetData.metrics[strategy];
  const correctness = datasetData.correctness_counts[strategy];
  const strategyText =
    strategy === "selected" ? `${strategy} (${datasetData.selected_strategy} for this dataset)` : strategy;
  const threshold = getStrategyThreshold(datasetData, strategy);

  document.getElementById("dataset-summary").innerHTML = `
    <article class="summary-card">
      <p class="eyebrow">Run</p>
      <strong>${escapeHtml(datasetData.run_id)}</strong>
      <p class="muted">${escapeHtml(datasetData.model_name)}</p>
    </article>
    <article class="summary-card">
      <p class="eyebrow">Strategy snapshot</p>
      <div class="summary-list"><span>Mode</span><code>${escapeHtml(strategyText)}</code></div>
      <div class="summary-list"><span>F1</span><strong>${formatF1(selectedMetrics.f1)}</strong></div>
      <div class="summary-list"><span>Threshold</span><code>${threshold.toFixed(2)}</code></div>
    </article>
    <article class="summary-card">
      <p class="eyebrow">Counts</p>
      <div class="summary-list"><span>Correct</span><strong>${formatCompact(correctness.correct)}</strong></div>
      <div class="summary-list"><span>Error</span><strong>${formatCompact(correctness.error)}</strong></div>
      <div class="summary-list"><span>Rows</span><strong>${formatCompact(datasetData.test_rows)}</strong></div>
    </article>
  `;
}

function renderExplorer(data) {
  const datasetValue = document.getElementById("dataset-filter").value || data.explorer.datasets[0].dataset;
  const outcomeValue = document.getElementById("outcome-filter").value;
  const strategyValue = document.getElementById("strategy-filter").value;
  const labelValue = document.getElementById("label-filter").value;
  const searchValue = document.getElementById("search-filter").value.trim().toLowerCase();

  const datasetData = data.explorer.datasets.find((entry) => entry.dataset === datasetValue);
  updateDatasetSummary(datasetData, strategyValue);

  const filteredRows = datasetData.rows.filter((row) => {
    const strategyCorrect = row.correctness[strategyValue];
    const strategyPredictionName = row.prediction_names[strategyValue];

    if (outcomeValue === "correct" && !strategyCorrect) {
      return false;
    }
    if (outcomeValue === "error" && strategyCorrect) {
      return false;
    }
    if (labelValue !== "all" && row.true_label_name !== labelValue) {
      return false;
    }
    if (searchValue && !row.text.toLowerCase().includes(searchValue)) {
      return false;
    }

    row._strategyPredictionName = strategyPredictionName;
    row._strategyCorrect = strategyCorrect;
    return true;
  });

  const body = document.getElementById("explorer-body");
  body.innerHTML = filteredRows
    .slice(0, 250)
    .map((row) => {
      const tone = row._strategyCorrect ? "correct" : "error";
      const threshold = getStrategyThreshold(datasetData, strategyValue);
      return `
        <tr>
          <td>${escapeHtml(row.dataset_label)}</td>
          <td><span class="table-tag">${escapeHtml(strategyValue)}</span></td>
          <td><span class="table-tag" data-tone="${tone}">${row._strategyCorrect ? "correct" : "error"}</span></td>
          <td>${escapeHtml(row.true_label_name)}</td>
          <td>${escapeHtml(row._strategyPredictionName)}</td>
          <td class="mono">${row.prob_sarcastic.toFixed(4)}</td>
          <td class="mono">${threshold.toFixed(2)}</td>
          <td class="table-text">${escapeHtml(row.text)}</td>
        </tr>
      `;
    })
    .join("");

  const limitMessage = filteredRows.length > 250 ? ` Showing first 250.` : "";
  document.getElementById("table-status").textContent =
    `${formatCompact(filteredRows.length)} matching rows.${limitMessage}`;
}

function renderNotes(data) {
  document.getElementById("notes-list").innerHTML = data.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("");
}

async function init() {
  const response = await fetch(DATA_PATH);
  if (!response.ok) {
    throw new Error(`Failed to load ${DATA_PATH}`);
  }

  const data = await response.json();
  state.data = data;

  renderHeroMeta(data);
  renderKpis(data);
  buildComparisonChart(data);
  buildRankingChart(data);
  renderGapCards(data);
  setupFilters(data);
  renderNotes(data);
}

window.addEventListener("DOMContentLoaded", () => {
  init().catch((error) => {
    const status = document.getElementById("table-status");
    status.textContent = "Dashboard failed to load data.";
    console.error(error);
  });
});
