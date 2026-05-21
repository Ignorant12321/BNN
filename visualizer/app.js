(function () {
  "use strict";

  const METRICS = ["mae", "rmse", "nmae", "nrmse", "picp_90", "pinaw_90", "picp_95", "pinaw_95"];
  const PALETTE = ["#0e6f68", "#c6512f", "#315f9b", "#8a5a21", "#7b3f76", "#3f7c35", "#9a3d47", "#5d6678"];

  const state = {
    activeSidebarTab: "read",
    comparisons: [],
    trainRuns: [],
    selectedTrainRunPaths: new Set(),
    currentComparison: null,
    runs: [],
    figures: [],
    split: "test",
    isCreating: false,
    query: "",
    trainQuery: "",
    figureQuery: "",
    diffOnly: true,
    selectedCell: null,
    selectedFigurePath: null,
  };

  const $ = (id) => document.getElementById(id);

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function formatValue(value) {
    if (value === undefined || value === null || value === "") return "-";
    if (typeof value === "number") {
      if (!Number.isFinite(value)) return String(value);
      if (Math.abs(value) >= 1000) return value.toFixed(2);
      return Number(value.toFixed(6)).toString();
    }
    return String(value);
  }

  function flatten(value, prefix = "", output = {}) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      for (const [key, child] of Object.entries(value)) flatten(child, prefix ? `${prefix}.${key}` : key, output);
      return output;
    }
    if (prefix) output[prefix] = value;
    return output;
  }

  async function fetchJson(path, options = {}) {
    const response = await fetch(path, {
      cache: "no-store",
      headers: options.body ? { "Content-Type": "application/json" } : undefined,
      ...options,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
    return payload;
  }

  function normalizeRun(run, index) {
    return {
      ...run,
      id: run.path || `${run.label}-${index}`,
      color: PALETTE[index % PALETTE.length],
      visible: true,
    };
  }

  function chooseSplit(runs) {
    for (const split of ["test", "val", "train"]) {
      if (runs.some((run) => Object.keys(run.metrics || {}).some((key) => key.startsWith(`${split}_`)))) return split;
    }
    return "test";
  }

  async function refreshLists() {
    const [comparisonPayload, trainPayload] = await Promise.all([
      fetchJson("/api/comparisons"),
      fetchJson("/api/train-runs"),
    ]);
    state.comparisons = comparisonPayload.comparisons || [];
    state.trainRuns = (trainPayload.runs || []).map(normalizeRun);
    render();
    showToast("列表已刷新。");
  }

  async function loadComparison(path) {
    if (!path) {
      showToast("请选择一个 comparison 文件夹。");
      return;
    }
    const payload = await fetchJson(`/api/comparison?path=${encodeURIComponent(path)}`);
    state.currentComparison = payload.comparison;
    state.runs = (payload.runs || []).map(normalizeRun);
    state.figures = payload.figures || [];
    state.split = chooseSplit(state.runs);
    $("splitSelect").value = state.split;
    switchToNavigation();
    render();
    focusSectionDirectory();
    showToast(`已读取 ${state.currentComparison.name}。`);
  }

  async function createComparison() {
    const selectedRuns = state.trainRuns
      .filter((run) => state.selectedTrainRunPaths.has(run.path))
      .map((run) => ({ label: run.label, path: run.path }));
    if (selectedRuns.length === 0) {
      showToast("请先选择至少一个训练 run。");
      return;
    }
    closeCurrentComparison();
    state.activeSidebarTab = "create";
    state.isCreating = true;
    render();
    renderCreateButton();
    try {
      const payload = await fetchJson("/api/comparisons", {
        method: "POST",
        body: JSON.stringify({
          name: $("comparisonNameInput").value || "visualizer",
          split: "test",
          note: $("comparisonNoteInput").value || null,
          runs: selectedRuns,
        }),
      });
      state.currentComparison = payload.comparison;
      state.runs = (payload.runs || []).map(normalizeRun);
      state.figures = payload.figures || [];
      state.split = chooseSplit(state.runs);
      $("splitSelect").value = state.split;
      state.comparisons = (await fetchJson("/api/comparisons")).comparisons || [];
      switchToNavigation();
      render();
      focusSectionDirectory();
      showToast(`已创建并读取 ${state.currentComparison.name}。`);
    } finally {
      state.isCreating = false;
      renderCreateButton();
    }
  }

  function clearPage() {
    closeCurrentComparison();
    render();
  }

  function closeCurrentComparison() {
    state.currentComparison = null;
    state.runs = [];
    state.figures = [];
    state.selectedCell = null;
    state.selectedFigurePath = null;
    state.query = "";
    state.figureQuery = "";
    $("searchInput").value = "";
    $("figureSearch").value = "";
  }

  function visibleRuns() {
    const query = state.query.trim().toLowerCase();
    return state.runs.filter((run) => run.visible).filter((run) => {
      if (!query) return true;
      return `${run.label} ${run.model} ${run.note} ${run.path}`.toLowerCase().includes(query);
    });
  }

  function filteredTrainRuns() {
    const query = state.trainQuery.trim().toLowerCase();
    if (!query) return state.trainRuns;
    return state.trainRuns.filter((run) => `${run.label} ${run.model} ${run.note} ${run.path}`.toLowerCase().includes(query));
  }

  function metricScore(metric, value) {
    if (typeof value !== "number" || Number.isNaN(value)) return Number.POSITIVE_INFINITY;
    if (metric === "picp_90") return Math.abs(value - 0.9);
    if (metric === "picp_95") return Math.abs(value - 0.95);
    return value;
  }

  function metricValue(run, metric) {
    return run.metrics?.[`${state.split}_${metric}`];
  }

  function renderComparisonSelect() {
    if (!state.comparisons.length) {
      $("comparisonSelect").innerHTML = '<option value="">没有 comparison</option>';
      return;
    }
    const selectedPath = state.currentComparison?.path || $("comparisonSelect").value || state.comparisons[0].path;
    $("comparisonSelect").innerHTML = state.comparisons.map((item) => `
      <option value="${escapeHtml(item.path)}" ${item.path === selectedPath ? "selected" : ""}>
        ${escapeHtml(item.name)} · ${item.runCount} runs${item.note ? ` · ${escapeHtml(item.note)}` : ""}
      </option>
    `).join("");
  }

  function renderSidebarTabs() {
    document.querySelectorAll("[data-sidebar-tab]").forEach((button) => {
      const active = button.dataset.sidebarTab === state.activeSidebarTab;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll("[data-sidebar-panel]").forEach((panel) => {
      const active = panel.dataset.sidebarPanel === state.activeSidebarTab;
      panel.classList.toggle("active", active);
      panel.hidden = !active;
    });
  }

  function switchToNavigation() {
    state.activeSidebarTab = "nav";
  }

  function focusSectionDirectory() {
    requestAnimationFrame(() => {
      const sidebar = document.querySelector(".sidebar");
      if (sidebar) sidebar.scrollTo({ top: 0, left: 0 });
    });
  }

  function scrollToMainSection(target) {
    const header = document.querySelector(".app-header");
    const headerOffset = header ? header.getBoundingClientRect().height + 16 : 0;
    const top = Math.max(0, target.getBoundingClientRect().top + window.scrollY - headerOffset);
    window.scrollTo({ top, left: 0, behavior: "smooth" });
  }

  function renderTrainRunList() {
    const runs = filteredTrainRuns();
    if (!runs.length) {
      $("trainRunList").innerHTML = '<p class="empty compact-empty">没有训练 run。</p>';
      return;
    }
    $("trainRunList").innerHTML = runs.map((run) => `
      <article class="run-item train-run" style="--run-color: ${run.color}">
        <label class="run-title">
          <input type="checkbox" data-train-run="${escapeHtml(run.path)}" ${state.selectedTrainRunPaths.has(run.path) ? "checked" : ""} />
          ${escapeHtml(run.label)}
        </label>
        <div class="run-meta">${escapeHtml(run.model)} · ${escapeHtml(run.path)}</div>
      </article>
    `).join("");
  }

  function renderStats() {
    const runs = visibleRuns();
    const bestRmse = runs.reduce((best, run) => {
      const value = metricValue(run, "rmse");
      return metricScore("rmse", value) < metricScore("rmse", best?.value) ? { run, value } : best;
    }, null);
    $("statsSection").innerHTML = [
      ["当前文件夹", state.currentComparison?.name || "-"],
      ["全部 run", state.runs.length],
      ["当前可见", runs.length],
      ["最佳 RMSE", bestRmse ? formatValue(bestRmse.value) : "-"],
    ].map(([label, value]) => `<article class="stat-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></article>`).join("");
  }

  function renderRunList() {
    $("runCount").textContent = String(state.runs.length);
    if (!state.runs.length) {
      $("runList").innerHTML = '<p class="empty">页面为空。请选择一个 comparison 读取，或创建新的 comparison。</p>';
      return;
    }
    $("runList").innerHTML = state.runs.map((run) => `
      <article class="run-item ${run.visible ? "" : "hidden"}" style="--run-color: ${run.color}">
        <label class="run-title">
          <input type="checkbox" data-toggle-run="${escapeHtml(run.id)}" ${run.visible ? "checked" : ""} />
          ${escapeHtml(run.label)}
        </label>
        <div class="run-meta">${escapeHtml(run.model)} · ${escapeHtml(run.note || "no note")}</div>
        <div class="run-meta">${escapeHtml(run.path)}</div>
      </article>
    `).join("");
  }

  function renderMetricBoard() {
    const runs = visibleRuns();
    if (!runs.length) {
      $("metricBoard").innerHTML = '<p class="empty">读取或创建 comparison 后显示指标。</p>';
      return;
    }
    const cards = METRICS.map((metric) => {
      const values = runs.map((run) => ({ run, value: metricValue(run, metric) })).filter((item) => typeof item.value === "number");
      if (!values.length) return "";
      const max = Math.max(...values.map((item) => Math.abs(item.value)), 1);
      const best = values.reduce((winner, item) => metricScore(metric, item.value) < metricScore(metric, winner.value) ? item : winner, values[0]);
      return `
        <article class="metric-card">
          <h3>${escapeHtml(state.split)}_${escapeHtml(metric)}</h3>
          ${values.map((item) => `
            <div class="bar-row ${item.run.id === best.run.id ? "best" : ""}">
              <span>${escapeHtml(item.run.label)}</span>
              <span class="bar-track"><i class="bar-fill" style="--bar-width: ${(Math.abs(item.value) / max) * 100}%; --run-color: ${item.run.color}"></i></span>
              <span>${formatValue(item.value)}</span>
            </div>
          `).join("")}
        </article>
      `;
    }).filter(Boolean);
    $("metricBoard").innerHTML = cards.length ? cards.join("") : '<p class="empty">当前 split 没有可显示指标。</p>';
  }

  function cellClass(tableId, row, col, extra = "") {
    if (!state.selectedCell || state.selectedCell.tableId !== tableId) return extra;
    const classes = [extra];
    if (state.selectedCell.row === row && state.selectedCell.col === col) classes.push("cell-selected");
    if (state.selectedCell.row === row) classes.push("row-selected");
    if (state.selectedCell.col === col) classes.push("col-selected");
    return classes.filter(Boolean).join(" ");
  }

  function tableCell(tag, tableId, row, col, value, extra = "") {
    return `<${tag} class="${cellClass(tableId, row, col, extra)}" data-table-id="${tableId}" data-row="${row}" data-col="${col}">${value}</${tag}>`;
  }

  function bestRunIdForMetric(runs, metric) {
    const values = runs.map((run) => ({ run, value: metricValue(run, metric) })).filter((item) => typeof item.value === "number");
    if (!values.length) return "";
    return values.reduce((winner, item) => (metricScore(metric, item.value) < metricScore(metric, winner.value) ? item : winner), values[0]).run.id;
  }

  function renderPredictionSummary() {
    const runs = visibleRuns();
    const rows = runs.filter((run) => run.predictionSummary);
    if (!rows.length) {
      $("predictionTable").innerHTML = '<p class="empty">当前 comparison 没有 predictions CSV 摘要。</p>';
      return;
    }
    const fields = [
      ["label", "Run"],
      ["rows", "预测行数"],
      ["horizons", "Horizon 数"],
      ["targetMean", "Target 均值"],
      ["predictionMean", "预测均值"],
      ["mae", "预测 MAE"],
      ["picp90", "PICP 90"],
      ["pinaw90", "PINAW 90"],
      ["picp95", "PICP 95"],
      ["pinaw95", "PINAW 95"],
    ];
    $("predictionTable").innerHTML = `
      <table>
        <thead>
          <tr>${fields.map(([, label], col) => tableCell("th", "predictions", -1, col, escapeHtml(label))).join("")}</tr>
        </thead>
        <tbody>
          ${rows.map((run, rowIndex) => `
            <tr>
              ${fields.map(([key], col) => {
                const value = key === "label" ? run.label : run.predictionSummary[key];
                return tableCell(col === 0 ? "th" : "td", "predictions", rowIndex, col, escapeHtml(formatValue(value)));
              }).join("")}
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  }

  function renderConfigTable() {
    const runs = visibleRuns();
    if (!runs.length) {
      $("configTable").innerHTML = '<p class="empty">读取或创建 comparison 后显示参数。</p>';
      return;
    }
    const flattened = runs.map((run) => flatten(run.config || {}));
    const keys = [...new Set(flattened.flatMap((item) => Object.keys(item)))].sort();
    const visibleKeys = keys.filter((key) => !state.diffOnly || new Set(flattened.map((item) => JSON.stringify(item[key]))).size > 1);
    if (!visibleKeys.length) {
      $("configTable").innerHTML = '<p class="empty">当前可见 run 没有配置差异。</p>';
      return;
    }
    $("configTable").innerHTML = `
      <table>
        <thead><tr>${tableCell("th", "config", -1, 0, "参数")}${runs.map((run, col) => tableCell("th", "config", -1, col + 1, escapeHtml(run.label))).join("")}</tr></thead>
        <tbody>
          ${visibleKeys.map((key, row) => `
            <tr>${tableCell("th", "config", row, 0, escapeHtml(key))}${flattened.map((item, col) => tableCell("td", "config", row, col + 1, escapeHtml(formatValue(item[key])))).join("")}</tr>
          `).join("")}
        </tbody>
      </table>
    `;
  }

  function filteredFigures() {
    const query = state.figureQuery.trim().toLowerCase();
    return (state.figures || []).filter((figure) => !query || figure.name.toLowerCase().includes(query));
  }

  function figureGroupClass(key) {
    if (key === "loss") return "figure-group figure-group-loss";
    if (key === "predict") return "figure-group figure-group-predict";
    return "figure-group";
  }

  function renderFigures() {
    const figures = filteredFigures();
    if (!figures.length) {
      $("figureGrid").innerHTML = '<p class="empty">读取或创建 comparison 后显示图片。</p>';
      return;
    }
    const groups = [
      ["loss", "Loss"],
      ["predict", "Prediction"],
    ].map(([key, label]) => [key, label, figures.filter((figure) => figure.group === key)]).filter(([, , items]) => items.length);
    $("figureGrid").innerHTML = `
      ${groups.map(([key, label, items]) => `
        <article class="${figureGroupClass(key)}">
          <h3>${escapeHtml(label)}</h3>
          <div class="single-figure-grid">
            ${items.map((figure) => `
              <article class="figure-card figure-card-wide ${state.selectedFigurePath === figure.path ? "selected" : ""}" data-figure-path="${escapeHtml(figure.path)}">
                <h4>${escapeHtml(figure.name)}</h4>
                <img src="${figure.url}" alt="${escapeHtml(figure.name)}" data-caption="${escapeHtml(figure.path)}" />
              </article>
            `).join("")}
          </div>
        </article>
      `).join("")}
    `;
  }

  function renderMetricTable() {
    const runs = visibleRuns();
    if (!runs.length) return;
    const metricRows = METRICS.filter((metric) => runs.some((run) => typeof metricValue(run, metric) === "number"));
    if (!metricRows.length) return;
    const table = `
      <div class="table-wrap metric-table-wrap">
        <table>
          <thead>
            <tr>${tableCell("th", "metrics", -1, 0, "指标")}${runs.map((run, col) => tableCell("th", "metrics", -1, col + 1, escapeHtml(run.label))).join("")}</tr>
          </thead>
          <tbody>
            ${metricRows.map((metric, row) => `
              <tr>
                ${tableCell("th", "metrics", row, 0, `${escapeHtml(state.split)}_${escapeHtml(metric)}`)}
                ${runs.map((run, col) => {
                  const isBest = run.id === bestRunIdForMetric(runs, metric);
                  return tableCell("td", "metrics", row, col + 1, escapeHtml(formatValue(metricValue(run, metric))), isBest ? "metric-best-soft" : "");
                }).join("")}
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    `;
    $("metricBoard").insertAdjacentHTML("beforeend", table);
  }

  function render() {
    renderSidebarTabs();
    renderComparisonSelect();
    renderTrainRunList();
    renderCreateButton();
    renderStats();
    renderRunList();
    renderMetricBoard();
    renderMetricTable();
    renderPredictionSummary();
    renderConfigTable();
    renderFigures();
  }

  function exportJson() {
    const blob = new Blob([JSON.stringify({ comparison: state.currentComparison, runs: visibleRuns() }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "bnn-visualizer-export.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function showToast(message) {
    const toast = $("toast");
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => toast.classList.remove("show"), 2600);
  }

  function renderCreateButton() {
    const button = $("createComparisonButton");
    button.disabled = state.isCreating;
    button.classList.toggle("loading", state.isCreating);
    button.textContent = state.isCreating ? "创建中..." : "创建并读取";
  }

  function selectFigureCard(card) {
    state.selectedFigurePath = card.dataset.figurePath;
    document.querySelectorAll(".figure-card.selected").forEach((selectedCard) => selectedCard.classList.remove("selected"));
    card.classList.add("selected");
  }

  function bindSidebarResize() {
    const handle = $("sidebarResizer");
    const storedWidth = localStorage.getItem("bnnVisualizer.sidebarWidth");
    if (storedWidth) document.documentElement.style.setProperty("--sidebar-width", storedWidth);
    let dragging = false;
    handle.addEventListener("mousedown", () => {
      dragging = true;
      document.body.classList.add("resizing-sidebar");
    });
    document.addEventListener("mousemove", (event) => {
      if (!dragging) return;
      const next = Math.max(240, Math.min(520, event.clientX - 22));
      const value = `${next}px`;
      document.documentElement.style.setProperty("--sidebar-width", value);
      localStorage.setItem("bnnVisualizer.sidebarWidth", value);
    });
    document.addEventListener("mouseup", () => {
      dragging = false;
      document.body.classList.remove("resizing-sidebar");
    });
  }

  function bindEvents() {
    bindSidebarResize();
    document.querySelectorAll("[data-sidebar-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        state.activeSidebarTab = button.dataset.sidebarTab;
        renderSidebarTabs();
      });
    });
    $("refreshButton").addEventListener("click", () => refreshLists().catch((error) => showToast(`刷新失败：${error.message}`)));
    $("clearPageButton").addEventListener("click", clearPage);
    $("exportButton").addEventListener("click", exportJson);
    $("loadComparisonButton").addEventListener("click", () => loadComparison($("comparisonSelect").value).catch((error) => showToast(`读取失败：${error.message}`)));
    $("createComparisonButton").addEventListener("click", () => createComparison().catch((error) => showToast(`创建失败：${error.message}`)));
    $("exitFolderButton").addEventListener("click", () => {
      clearPage();
      showToast("已退出当前文件夹。");
    });
    $("searchInput").addEventListener("input", (event) => { state.query = event.target.value; render(); });
    $("trainSearchInput").addEventListener("input", (event) => { state.trainQuery = event.target.value; renderTrainRunList(); });
    $("figureSearch").addEventListener("input", (event) => { state.figureQuery = event.target.value; renderFigures(); });
    $("splitSelect").addEventListener("change", (event) => {
      state.split = event.target.value;
      renderStats();
      renderMetricBoard();
      renderMetricTable();
    });
    $("diffOnly").addEventListener("change", (event) => { state.diffOnly = event.target.checked; renderConfigTable(); });
    $("sectionDirectory").addEventListener("click", (event) => {
      const button = event.target.closest("[data-nav-target]");
      if (!button) return;
      const target = $(button.dataset.navTarget);
      if (target) scrollToMainSection(target);
    });
    $("trainRunList").addEventListener("change", (event) => {
      const path = event.target.dataset.trainRun;
      if (!path) return;
      if (event.target.checked) {
        state.selectedTrainRunPaths.add(path);
      } else {
        state.selectedTrainRunPaths.delete(path);
      }
    });
    $("runList").addEventListener("change", (event) => {
      const id = event.target.dataset.toggleRun;
      const run = state.runs.find((item) => item.id === id);
      if (run) {
        run.visible = event.target.checked;
        render();
      }
    });
    $("figureGrid").addEventListener("click", (event) => {
      const card = event.target.closest(".figure-card[data-figure-path]");
      if (!card) return;
      selectFigureCard(card);
    });
    $("figureGrid").addEventListener("dblclick", (event) => {
      const image = event.target.closest("img");
      if (!image) return;
      openLightbox(image.src, image.dataset.caption || image.alt);
    });
    document.querySelector(".content").addEventListener("click", (event) => {
      const cell = event.target.closest("[data-table-id][data-row][data-col]");
      if (!cell) return;
      state.selectedCell = {
        tableId: cell.dataset.tableId,
        row: Number(cell.dataset.row),
        col: Number(cell.dataset.col),
      };
      renderMetricBoard();
      renderMetricTable();
      renderPredictionSummary();
      renderConfigTable();
    });
    $("lightboxClose").addEventListener("click", closeLightbox);
    $("lightbox").addEventListener("click", (event) => {
      if (event.target.id === "lightbox") closeLightbox();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeLightbox();
    });
  }

  function closeLightbox() {
    $("lightbox").classList.remove("open");
    $("lightbox").setAttribute("aria-hidden", "true");
    $("lightboxImage").removeAttribute("src");
  }

  function openLightbox(src, caption) {
    $("lightboxImage").src = src;
    $("lightboxCaption").textContent = caption;
    $("lightbox").classList.add("open");
    $("lightbox").setAttribute("aria-hidden", "false");
  }

  bindEvents();
  render();
  refreshLists().catch((error) => showToast(`初始化失败：${error.message}`));
})();
