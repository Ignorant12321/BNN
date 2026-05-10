(function (root) {
  "use strict";

  const METRIC_KEYS = ["mae", "rmse", "nrmse", "smape", "nll", "picp_90", "pinaw_90", "picp_95", "pinaw_95"];
  const TIMESTAMP_RE = /^\d{8}-\d{6}$/;
  const FIGURE_ORDER = [
    "loss_curve.png",
    "prediction_interval_90.png",
    "prediction_interval_95.png",
    "horizon_rmse.png",
    "picp_pinaw.png",
    "calibration_curve.png",
  ];
  const PALETTE = ["#0f766e", "#b45309", "#2563eb", "#9333ea", "#be123c", "#15803d", "#0e7490", "#a16207"];
  const PICP_TARGETS = {
    picp_90: 0.9,
    picp_95: 0.95,
  };

  const state = {
    runs: [],
    sidebarCollapsed: false,
    selectedCell: null,
    lightbox: {
      open: false,
      groupName: "",
      index: 0,
      items: [],
    },
    toastTimer: null,
  };

  function normalizePath(path) {
    return String(path || "").replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
  }

  function getFilePath(file) {
    return normalizePath(file.webkitRelativePath || file.relativePath || file.name);
  }

  function getStoredSidebarCollapsed() {
    if (typeof localStorage === "undefined") return false;
    return localStorage.getItem("bnnVisualizer.sidebarCollapsed") === "true";
  }

  function storeSidebarCollapsed(collapsed) {
    if (typeof localStorage === "undefined") return;
    localStorage.setItem("bnnVisualizer.sidebarCollapsed", String(collapsed));
  }

  function nextSidebarCollapsed(current) {
    return !current;
  }

  function dirname(path) {
    const clean = normalizePath(path);
    const idx = clean.lastIndexOf("/");
    return idx === -1 ? "" : clean.slice(0, idx);
  }

  function isRunRoot(path) {
    const clean = normalizePath(path);
    if (!clean) return false;
    const name = clean.split("/").pop();
    return TIMESTAMP_RE.test(name);
  }

  function inferRunRoot(path) {
    const clean = normalizePath(path);
    if (!clean) return null;

    if (clean.endsWith("/config.yaml")) return dirname(clean);
    if (clean.endsWith("/note.txt")) return dirname(clean);

    const markers = ["/metrics/", "/figures/", "/predictions/", "/artifacts/", "/logs/", "/checkpoints/"];
    for (const marker of markers) {
      const idx = clean.indexOf(marker);
      if (idx > 0) return clean.slice(0, idx);
    }

    return null;
  }

  function discoverRunsFromPaths(paths) {
    const roots = new Set();
    for (const path of paths) {
      const rootPath = inferRunRoot(path);
      if (rootPath && isRunRoot(rootPath)) roots.add(rootPath);
    }
    return [...roots]
      .sort()
      .map((relativePath) => ({
        relativePath,
        name: relativePath.split("/").pop(),
      }));
  }

  function parseScalar(rawValue) {
    const trimmed = String(rawValue).trim();
    if (trimmed === "") return "";
    if (trimmed === "true") return true;
    if (trimmed === "false") return false;
    if (trimmed === "null" || trimmed === "None") return null;
    if ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
      return trimmed.slice(1, -1);
    }
    if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(trimmed)) return Number(trimmed);
    return trimmed;
  }

  function parseSimpleYaml(text) {
    const rootObject = {};
    const stack = [{ indent: -1, value: rootObject }];
    const lines = String(text || "").split(/\r?\n/);

    for (const rawLine of lines) {
      const withoutComment = rawLine.replace(/\s+#.*$/, "");
      if (!withoutComment.trim()) continue;
      const indent = withoutComment.match(/^\s*/)[0].length;
      const trimmed = withoutComment.trim();

      while (stack.length > 1 && indent <= stack[stack.length - 1].indent) stack.pop();
      const parent = stack[stack.length - 1].value;

      if (trimmed.startsWith("- ")) {
        if (Array.isArray(parent)) parent.push(parseScalar(trimmed.slice(2)));
        continue;
      }

      const colonIndex = trimmed.indexOf(":");
      if (colonIndex === -1) continue;

      const key = trimmed.slice(0, colonIndex).trim();
      const valueText = trimmed.slice(colonIndex + 1).trim();
      if (valueText === "") {
        const nextObject = {};
        parent[key] = nextObject;
        stack.push({ indent, value: nextObject });
      } else {
        parent[key] = parseScalar(valueText);
      }
    }

    return rootObject;
  }

  function flattenObject(value, prefix = "", output = {}) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      for (const [key, childValue] of Object.entries(value)) {
        const nextPrefix = prefix ? `${prefix}.${key}` : key;
        flattenObject(childValue, nextPrefix, output);
      }
      return output;
    }
    output[prefix] = value;
    return output;
  }

  function parseCsv(text) {
    const rows = String(text || "")
      .trim()
      .split(/\r?\n/)
      .filter(Boolean);
    if (rows.length === 0) return [];

    const headers = splitCsvLine(rows[0]);
    return rows.slice(1).map((row) => {
      const values = splitCsvLine(row);
      return Object.fromEntries(headers.map((header, index) => [header, parseScalar(values[index] ?? "")]));
    });
  }

  function splitCsvLine(line) {
    const values = [];
    let current = "";
    let quoted = false;
    for (let index = 0; index < line.length; index += 1) {
      const char = line[index];
      if (char === '"') {
        if (quoted && line[index + 1] === '"') {
          current += '"';
          index += 1;
        } else {
          quoted = !quoted;
        }
      } else if (char === "," && !quoted) {
        values.push(current);
        current = "";
      } else {
        current += char;
      }
    }
    values.push(current);
    return values;
  }

  function formatValue(value) {
    if (value === undefined || value === null || value === "") return "-";
    if (typeof value === "number") {
      if (!Number.isFinite(value)) return String(value);
      if (Math.abs(value) >= 1000) return value.toFixed(2);
      if (Math.abs(value) > 0 && Math.abs(value) < 0.000001) return value.toExponential(3);
      return Number(value.toFixed(6)).toString();
    }
    if (typeof value === "boolean") return value ? "true" : "false";
    return String(value);
  }

  function getMetricDatasetGroups() {
    return [
      { label: "测试集", field: "testMetrics", description: "最终泛化表现" },
      { label: "验证集", field: "validationMetrics", description: "调参和模型选择" },
    ];
  }

  function filterRunsBySearch(runs, query) {
    const normalized = String(query || "").trim().toLowerCase();
    if (!normalized) return runs;
    return runs.filter((run) => `${run.name} ${run.relativePath} ${run.note || ""}`.toLowerCase().includes(normalized));
  }

  function getTableSelectionClasses(selection, tableId, row, col) {
    if (!selection || selection.tableId !== tableId) return "";
    const classes = [];
    if (selection.row === row && selection.col === col) classes.push("cell-selected");
    if (selection.row === row) classes.push("row-selected");
    if (selection.col === col) classes.push("col-selected");
    return classes.join(" ");
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  async function loadRunsFromFiles(files) {
    const fileArray = [...files];
    const pathToFile = new Map(fileArray.map((file) => [getFilePath(file), file]));
    const discovered = discoverRunsFromPaths([...pathToFile.keys()]);
    const newRuns = [];

    for (const item of discovered) {
      if (state.runs.some((run) => run.relativePath === item.relativePath)) continue;
      const runFiles = new Map();
      for (const [path, file] of pathToFile.entries()) {
        if (path === item.relativePath || path.startsWith(`${item.relativePath}/`)) {
          runFiles.set(path.slice(item.relativePath.length).replace(/^\//, ""), file);
        }
      }
      newRuns.push(await buildRun(item, runFiles, state.runs.length + newRuns.length));
    }

    state.runs.push(...newRuns);
    render();
    return newRuns;
  }

  async function readTextIfPresent(files, relativePath) {
    const file = files.get(relativePath);
    return file ? file.text() : null;
  }

  async function buildRun(item, files, index = 0) {
    const configText = await readTextIfPresent(files, "config.yaml");
    const testMetricsText = await readTextIfPresent(files, "metrics/metrics.json");
    const validationMetricsText = await readTextIfPresent(files, "metrics/validation_metrics.json");
    const pointMetricsText = await readTextIfPresent(files, "metrics/point_metrics.csv");
    const noteText = await readTextIfPresent(files, "note.txt");

    const config = configText ? parseSimpleYaml(configText) : {};
    const figures = [...files.entries()]
      .filter(([path]) => path.startsWith("figures/") && /\.(png|jpg|jpeg|webp|gif)$/i.test(path))
      .map(([path, file]) => ({
        name: path.split("/").pop(),
        path,
        url: URL.createObjectURL(file),
      }))
      .sort((a, b) => figureRank(a.name) - figureRank(b.name) || a.name.localeCompare(b.name));

    return {
      id: `${item.relativePath}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      name: item.name,
      relativePath: item.relativePath,
      visible: true,
      color: PALETTE[index % PALETTE.length],
      config,
      flatConfig: flattenObject(config),
      testMetrics: safeJsonParse(testMetricsText),
      validationMetrics: safeJsonParse(validationMetricsText),
      pointMetrics: pointMetricsText ? parseCsv(pointMetricsText) : [],
      figures,
      note: noteText ? noteText.trim() : "",
    };
  }

  function safeJsonParse(text) {
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch {
      return null;
    }
  }

  function figureRank(name) {
    const idx = FIGURE_ORDER.indexOf(name);
    return idx === -1 ? FIGURE_ORDER.length : idx;
  }

  function groupFiguresByName(runs, filterText = "") {
    const normalizedFilter = String(filterText || "").trim().toLowerCase();
    const groups = new Map();
    for (const run of runs) {
      for (const figure of run.figures || []) {
        if (normalizedFilter && !figure.name.toLowerCase().includes(normalizedFilter)) continue;
        if (!groups.has(figure.name)) groups.set(figure.name, { name: figure.name, items: [] });
        groups.get(figure.name).items.push({
          runName: run.name,
          runId: run.id,
          runColor: run.color,
          figure,
        });
      }
    }
    return [...groups.values()].sort((a, b) => figureRank(a.name) - figureRank(b.name) || a.name.localeCompare(b.name));
  }

  function summarizeFigureCoverage(runs) {
    const groups = groupFiguresByName(runs);
    const imageCount = groups.reduce((total, group) => total + group.items.length, 0);
    const missingSlots = groups.reduce((total, group) => total + Math.max(0, runs.length - group.items.length), 0);
    return {
      groupCount: groups.length,
      imageCount,
      missingSlots,
    };
  }

  function getLightboxItemsForGroup(groups, groupName) {
    const group = groups.find((item) => item.name === groupName);
    return group ? group.items : [];
  }

  function getMetricScore(metricKey, value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return Number.POSITIVE_INFINITY;
    if (Object.hasOwn(PICP_TARGETS, metricKey)) return Number(Math.abs(number - PICP_TARGETS[metricKey]).toFixed(12));
    return number;
  }

  function summarizeRuns(runs) {
    const includedRuns = runs.filter((run) => run.visible !== false);
    const bestMetrics = {};
    for (const key of METRIC_KEYS) {
      let bestRun = null;
      let bestScore = Number.POSITIVE_INFINITY;
      for (const run of includedRuns) {
        const score = getMetricScore(key, run.testMetrics?.[key]);
        if (score < bestScore) {
          bestRun = run;
          bestScore = score;
        }
      }
      if (bestRun) bestMetrics[key] = bestRun.name;
    }

    const horizonSeries = {};
    for (const metricKey of ["rmse", "mae"]) {
      horizonSeries[metricKey] = includedRuns
        .filter((run) => run.pointMetrics?.length)
        .map((run, index) => ({
          runName: run.name,
          runColor: run.color || PALETTE[index % PALETTE.length],
          points: run.pointMetrics
            .filter((row) => Number.isFinite(Number(row.horizon)) && Number.isFinite(Number(row[metricKey])))
            .map((row) => ({ x: Number(row.horizon), y: Number(row[metricKey]) })),
        }))
        .filter((series) => series.points.length);
    }

    return {
      bestMetrics,
      horizonSeries,
      runCount: includedRuns.length,
      figureGroupCount: groupFiguresByName(includedRuns).length,
      figureCoverage: summarizeFigureCoverage(includedRuns),
    };
  }

  function visibleRuns() {
    return state.runs.filter((run) => run.visible);
  }

  function render() {
    if (typeof document === "undefined") return;
    renderSidebarState();
    renderSummaryStats();
    renderRunList();
    renderVisualCharts();
    renderMetricsTable();
    renderParamsTable();
    renderPointMetrics();
    renderFigures();
  }

  function renderSidebarState() {
    document.body.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
    const button = document.getElementById("sidebarToggle");
    if (button) {
      button.textContent = state.sidebarCollapsed ? "显示侧栏" : "隐藏侧栏";
      button.setAttribute("aria-pressed", String(state.sidebarCollapsed));
    }
  }

  function renderSummaryStats() {
    const container = document.getElementById("summaryStats");
    const runs = visibleRuns();
    const summary = summarizeRuns(runs);
    const coverage = summary.figureCoverage;
    if (state.runs.length === 0) {
      container.innerHTML = `
        <article class="stat-card"><span>可见 run</span><strong>0</strong></article>
        <article class="stat-card"><span>最佳 RMSE</span><strong>-</strong></article>
        <article class="stat-card"><span>图像总数</span><strong>0</strong></article>
        <article class="stat-card"><span>缺失图像槽位</span><strong>0</strong></article>
      `;
      return;
    }
    const bestRmse = summary.bestMetrics.rmse || "-";
    const bestPicp = summary.bestMetrics.picp_90 || "-";
    container.innerHTML = `
      <article class="stat-card"><span>可见 / 总 run</span><strong>${runs.length} / ${state.runs.length}</strong></article>
      <article class="stat-card"><span>测试集 RMSE 最佳</span><strong>${escapeHtml(bestRmse)}</strong></article>
      <article class="stat-card"><span>图像总数 / 组</span><strong>${coverage.imageCount} / ${coverage.groupCount}</strong></article>
      <article class="stat-card"><span>缺失图像槽位</span><strong>${coverage.missingSlots}</strong></article>
    `;
  }

  function renderRunList() {
    const list = document.getElementById("runList");
    const count = document.getElementById("runCount");
    count.textContent = state.runs.length;

    const query = document.getElementById("runSearch")?.value || "";
    const runs = filterRunsBySearch(state.runs, query);

    if (state.runs.length === 0) {
      list.innerHTML = '<p class="empty">还没有添加实验文件夹。</p>';
      return;
    }

    if (runs.length === 0) {
      list.innerHTML = '<p class="empty">没有匹配的 run。</p>';
      return;
    }

    list.innerHTML = runs
      .map(
        (run) => `
          <article class="run-item ${run.visible ? "active" : ""}" style="--run-color: ${escapeHtml(run.color)}">
            <label>
              <input type="checkbox" data-run-toggle="${escapeHtml(run.id)}" ${run.visible ? "checked" : ""} />
              <span class="run-color" aria-hidden="true"></span>
              <span>${escapeHtml(run.name)}</span>
            </label>
            <div class="run-path">${escapeHtml(run.relativePath)}</div>
            ${run.note ? `<div class="run-note">${escapeHtml(run.note)}</div>` : ""}
          </article>
        `,
      )
      .join("");
  }

  function renderCell(content, tableId, row, col, extraClass = "") {
    const classes = [extraClass, getTableSelectionClasses(state.selectedCell, tableId, row, col)].filter(Boolean).join(" ");
    return `<td class="${classes}" data-table-id="${tableId}" data-row="${row}" data-col="${col}">${content}</td>`;
  }

  function renderHeaderCell(content, tableId, row, col, scope = "col", extraClass = "") {
    const classes = [extraClass, getTableSelectionClasses(state.selectedCell, tableId, row, col)].filter(Boolean).join(" ");
    return `<th class="${classes}" scope="${scope}" data-table-id="${tableId}" data-row="${row}" data-col="${col}">${content}</th>`;
  }

  function renderVisualCharts() {
    const container = document.getElementById("visualCharts");
    const runs = visibleRuns();
    if (runs.length === 0) {
      container.innerHTML = '<p class="empty">选择 run 后显示图表。</p>';
      return;
    }

    const metricKey = document.getElementById("metricSelect").value;
    const horizonMetricKey = document.getElementById("horizonMetricSelect").value;
    const summary = summarizeRuns(runs);
    container.innerHTML = `
      <article class="chart-card">
        <h3>测试集 ${escapeHtml(metricKey)} 对比</h3>
        <p>${Object.hasOwn(PICP_TARGETS, metricKey) ? "越接近目标覆盖率越好" : "数值越低越好"}</p>
        ${renderMetricBarChart(runs, metricKey)}
      </article>
      <article class="chart-card">
        <h3>Horizon ${escapeHtml(horizonMetricKey)} 趋势</h3>
        <p>每条线对应一个 run。</p>
        ${renderLineChart(summary.horizonSeries[horizonMetricKey])}
      </article>
    `;
  }

  function renderMetricBarChart(runs, metricKey) {
    const data = runs
      .map((run) => ({
        name: run.name,
        color: run.color,
        value: Number(run.testMetrics?.[metricKey]),
        score: getMetricScore(metricKey, run.testMetrics?.[metricKey]),
      }))
      .filter((item) => Number.isFinite(item.value));

    if (data.length === 0) return '<p class="empty">未找到该指标。</p>';

    const bestScore = Math.min(...data.map((item) => item.score));
    const maxValue = Math.max(...data.map((item) => item.value), 1);
    const width = 760;
    const height = 260;
    const plotTop = 24;
    const plotBottom = 214;
    const barGap = 16;
    const barWidth = Math.max(26, (width - 90 - barGap * (data.length - 1)) / data.length);

    const bars = data
      .map((item, index) => {
        const x = 56 + index * (barWidth + barGap);
        const barHeight = Math.max(2, (item.value / maxValue) * (plotBottom - plotTop));
        const y = plotBottom - barHeight;
        const label = item.name.length > 16 ? `${item.name.slice(0, 14)}...` : item.name;
        const isBest = item.score === bestScore;
        return `
          <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="5" fill="${item.color}" opacity="${isBest ? "1" : "0.72"}"></rect>
          ${isBest ? `<rect x="${x - 2}" y="${y - 2}" width="${barWidth + 4}" height="${barHeight + 4}" rx="7" fill="none" stroke="#114f3f" stroke-width="2"></rect>` : ""}
          <text class="chart-value" x="${x + barWidth / 2}" y="${Math.max(14, y - 7)}" text-anchor="middle">${formatValue(item.value)}</text>
          <text class="chart-label" x="${x + barWidth / 2}" y="239" text-anchor="middle">${escapeHtml(label)}</text>
        `;
      })
      .join("");

    return `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(metricKey)} bar chart">
        <line class="chart-axis" x1="44" y1="${plotBottom}" x2="${width - 20}" y2="${plotBottom}"></line>
        <line class="chart-axis" x1="44" y1="${plotTop}" x2="44" y2="${plotBottom}"></line>
        ${bars}
      </svg>
    `;
  }

  function renderLineChart(series) {
    if (!series || series.length === 0) return '<p class="empty">未找到 horizon 数据。</p>';

    const allPoints = series.flatMap((item) => item.points);
    const minX = Math.min(...allPoints.map((point) => point.x));
    const maxX = Math.max(...allPoints.map((point) => point.x));
    const minY = Math.min(...allPoints.map((point) => point.y));
    const maxY = Math.max(...allPoints.map((point) => point.y));
    const width = 760;
    const height = 260;
    const left = 50;
    const right = 22;
    const top = 22;
    const bottom = 214;
    const xRange = Math.max(1, maxX - minX);
    const yRange = Math.max(1, maxY - minY);
    const xScale = (x) => left + ((x - minX) / xRange) * (width - left - right);
    const yScale = (y) => bottom - ((y - minY) / yRange) * (bottom - top);

    const paths = series
      .map((item) => {
        const d = item.points.map((point, index) => `${index === 0 ? "M" : "L"} ${xScale(point.x).toFixed(2)} ${yScale(point.y).toFixed(2)}`).join(" ");
        const dots = item.points
          .map((point) => `<circle cx="${xScale(point.x)}" cy="${yScale(point.y)}" r="3" fill="${item.runColor || "#0f766e"}"></circle>`)
          .join("");
        return `<path d="${d}" fill="none" stroke="${item.runColor || "#0f766e"}" stroke-width="3"></path>${dots}`;
      })
      .join("");

    const legend = series
      .map(
        (item) => `
          <span style="--run-color: ${escapeHtml(item.runColor || "#0f766e")}">
            <i class="legend-dot"></i>${escapeHtml(item.runName)}
          </span>
        `,
      )
      .join("");

    return `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="horizon line chart">
        <line class="chart-axis" x1="${left}" y1="${bottom}" x2="${width - right}" y2="${bottom}"></line>
        <line class="chart-axis" x1="${left}" y1="${top}" x2="${left}" y2="${bottom}"></line>
        <text class="chart-label" x="${left}" y="238" text-anchor="middle">${minX}</text>
        <text class="chart-label" x="${width - right}" y="238" text-anchor="middle">${maxX}</text>
        <text class="chart-label" x="38" y="${bottom}" text-anchor="end">${formatValue(minY)}</text>
        <text class="chart-label" x="38" y="${top + 4}" text-anchor="end">${formatValue(maxY)}</text>
        ${paths}
      </svg>
      <div class="line-legend">${legend}</div>
    `;
  }

  function renderMetricsTable() {
    const container = document.getElementById("metricsTable");
    const runs = visibleRuns();
    if (runs.length === 0) {
      container.innerHTML = '<p class="empty">选择 run 后显示指标。</p>';
      return;
    }

    const rows = [];
    const bestByGroup = {
      testMetrics: bestRunIdsForMetrics(runs, "testMetrics"),
      validationMetrics: bestRunIdsForMetrics(runs, "validationMetrics"),
    };
    let rowIndex = 0;
    for (const group of getMetricDatasetGroups()) {
      rows.push(`
        <tr class="metric-section-row">
          <th scope="row" colspan="${runs.length + 1}">
            <span>${escapeHtml(group.label)}</span>
            <small>${escapeHtml(group.description)}</small>
          </th>
        </tr>
      `);
      rowIndex += 1;
      for (const key of METRIC_KEYS) {
        const currentRow = rowIndex;
        rows.push(`
          <tr class="metric-data-row">
            ${renderHeaderCell(key, "metrics", currentRow, 0, "row")}
            ${runs
              .map((run, runIndex) => {
                const isBest = bestByGroup[group.field][key] === run.id;
                return renderCell(formatValue(run[group.field]?.[key]), "metrics", currentRow, runIndex + 1, isBest ? "metric-best" : "");
              })
              .join("")}
          </tr>
        `);
        rowIndex += 1;
      }
    }

    container.innerHTML = `
      <table>
        <thead>
          <tr>
            ${renderHeaderCell("指标", "metrics", -1, 0)}
            ${runs.map((run, index) => renderHeaderCell(escapeHtml(run.name), "metrics", -1, index + 1)).join("")}
          </tr>
        </thead>
        <tbody>${rows.join("")}</tbody>
      </table>
    `;
  }

  function bestRunIdsForMetrics(runs, metricGroup) {
    const best = {};
    for (const key of METRIC_KEYS) {
      let bestRun = null;
      let bestScore = Number.POSITIVE_INFINITY;
      for (const run of runs) {
        const score = getMetricScore(key, run[metricGroup]?.[key]);
        if (score < bestScore) {
          bestRun = run;
          bestScore = score;
        }
      }
      if (bestRun) best[key] = bestRun.id;
    }
    return best;
  }

  function renderParamsTable() {
    const container = document.getElementById("paramsTable");
    const runs = visibleRuns();
    const diffOnly = document.getElementById("diffOnlyToggle").checked;
    const search = document.getElementById("paramSearch").value.trim().toLowerCase();
    if (runs.length === 0) {
      container.innerHTML = '<p class="empty">选择 run 后显示参数。</p>';
      return;
    }

    const keys = [...new Set(runs.flatMap((run) => Object.keys(run.flatConfig)))].sort();
    const visibleKeys = keys.filter((key) => {
      if (!diffOnly || runs.length < 2) return true;
      const values = new Set(runs.map((run) => JSON.stringify(run.flatConfig[key])));
      return values.size > 1;
    }).filter((key) => !search || key.toLowerCase().includes(search));

    if (visibleKeys.length === 0) {
      container.innerHTML = '<p class="empty">当前可见 run 的参数没有差异。</p>';
      return;
    }

    container.innerHTML = `
      <table>
        <thead>
          <tr>
            ${renderHeaderCell("参数", "params", -1, 0)}
            ${runs.map((run, index) => renderHeaderCell(escapeHtml(run.name), "params", -1, index + 1)).join("")}
          </tr>
        </thead>
        <tbody>
          ${visibleKeys
            .map(
              (key, rowIndex) => `
                <tr>
                  ${renderHeaderCell(escapeHtml(key), "params", rowIndex, 0, "row")}
                  ${runs.map((run, runIndex) => renderCell(escapeHtml(formatValue(run.flatConfig[key])), "params", rowIndex, runIndex + 1)).join("")}
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    `;
  }

  function renderPointMetrics() {
    const container = document.getElementById("pointMetrics");
    const runs = visibleRuns().filter((run) => run.pointMetrics.length > 0);
    if (runs.length === 0) {
      container.innerHTML = '<p class="empty">未找到 point_metrics.csv。</p>';
      return;
    }

    container.innerHTML = runs
      .map((run) => {
        const rows = run.pointMetrics
          .map(
            (row) => `
              <tr>
                <td>${formatValue(row.horizon)}</td>
                <td>${formatValue(row.mae)}</td>
                <td>${formatValue(row.rmse)}</td>
              </tr>
            `,
          )
          .join("");
        return `
          <article class="mini-card">
            <h3>${escapeHtml(run.name)}</h3>
            <div class="table-wrap">
              <table>
                <thead><tr><th>horizon</th><th>mae</th><th>rmse</th></tr></thead>
                <tbody>${rows}</tbody>
              </table>
            </div>
          </article>
        `;
      })
      .join("");
  }

  function renderFigures() {
    const container = document.getElementById("figureComparison");
    const runs = visibleRuns();
    const filterText = document.getElementById("figureSearch").value;
    const groups = groupFiguresByName(runs, filterText);
    if (groups.length === 0) {
      container.innerHTML = `<p class="empty">${filterText ? "没有匹配的图像组。" : "未找到 figures 图片。"}</p>`;
      return;
    }

    container.innerHTML = groups
      .map((group) => {
        const itemsByRun = new Map(group.items.map((item) => [item.runId || item.runName, item]));
        return `
          <article class="figure-group">
            <div class="figure-group-head">
              <h3>${escapeHtml(group.name)}</h3>
              <span class="coverage-pill">${group.items.length} 张 · 缺 ${Math.max(0, runs.length - group.items.length)} 张</span>
            </div>
            <div class="figure-row">
              ${runs
                .map((run) => {
                  const item = itemsByRun.get(run.id || run.name);
                  return `
                    <div class="figure-slot" style="--run-color: ${escapeHtml(run.color)}">
                      <h4><span class="run-color" aria-hidden="true"></span>${escapeHtml(run.name)}</h4>
                      ${
                        item
                          ? `<div class="figure-image-wrap">
                               <img src="${item.figure.url}" alt="${escapeHtml(run.name)} ${escapeHtml(group.name)}" loading="lazy" data-figure-group="${escapeHtml(group.name)}" data-run-id="${escapeHtml(run.id || run.name)}" />
                               <span class="zoom-hint">双击放大</span>
                             </div>
                             <div class="figure-meta">${escapeHtml(item.figure.path)}</div>`
                          : '<div class="missing-figure">未找到该图</div>'
                      }
                    </div>
                  `;
                })
                .join("")}
            </div>
          </article>
        `;
      })
      .join("");
  }

  function openLightbox(groupName, runId) {
    const groups = groupFiguresByName(visibleRuns());
    const items = getLightboxItemsForGroup(groups, groupName);
    if (items.length === 0) return;
    const index = Math.max(0, items.findIndex((item) => (item.runId || item.runName) === runId));
    state.lightbox = {
      open: true,
      groupName,
      index,
      items,
    };
    renderLightbox();
  }

  function closeLightbox() {
    state.lightbox.open = false;
    state.lightbox.items = [];
    renderLightbox();
  }

  function moveLightbox(step) {
    if (!state.lightbox.open || state.lightbox.items.length === 0) return;
    const length = state.lightbox.items.length;
    state.lightbox.index = (state.lightbox.index + step + length) % length;
    renderLightbox();
  }

  function renderLightbox() {
    const lightbox = document.getElementById("lightbox");
    if (!lightbox) return;
    const image = document.getElementById("lightboxImage");
    const title = document.getElementById("lightboxTitle");
    const run = document.getElementById("lightboxRun");
    const path = document.getElementById("lightboxPath");
    const counter = document.getElementById("lightboxCounter");
    const prev = document.getElementById("lightboxPrev");
    const next = document.getElementById("lightboxNext");

    if (!state.lightbox.open || state.lightbox.items.length === 0) {
      lightbox.classList.remove("open");
      lightbox.setAttribute("aria-hidden", "true");
      image.removeAttribute("src");
      image.alt = "";
      return;
    }

    const item = state.lightbox.items[state.lightbox.index];
    lightbox.classList.add("open");
    lightbox.setAttribute("aria-hidden", "false");
    image.src = item.figure.url;
    image.alt = `${item.runName} ${item.figure.name}`;
    title.textContent = item.figure.name;
    run.textContent = item.runName;
    path.textContent = item.figure.path;
    counter.textContent = `${state.lightbox.index + 1} / ${state.lightbox.items.length}`;
    prev.disabled = state.lightbox.items.length < 2;
    next.disabled = state.lightbox.items.length < 2;
  }

  function clearRuns() {
    for (const run of state.runs) {
      for (const figure of run.figures) URL.revokeObjectURL(figure.url);
    }
    state.runs = [];
    render();
    showToast("已清空当前对比。");
  }

  function exportCurrentComparison() {
    const payload = visibleRuns().map((run) => ({
      name: run.name,
      relativePath: run.relativePath,
      note: run.note,
      config: run.config,
      testMetrics: run.testMetrics,
      validationMetrics: run.validationMetrics,
      pointMetrics: run.pointMetrics,
      figures: run.figures.map((figure) => figure.path),
    }));
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "bnn-run-comparison.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function showToast(message, isError = false) {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.classList.toggle("error", isError);
    toast.classList.add("show");
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => toast.classList.remove("show"), 2600);
  }

  function bindEvents() {
    state.sidebarCollapsed = getStoredSidebarCollapsed();
    const input = document.getElementById("folderInput");
    input.addEventListener("change", async (event) => {
      const files = event.target.files;
      if (!files || files.length === 0) return;
      try {
        const added = await loadRunsFromFiles(files);
        showToast(added.length ? `已添加 ${added.length} 个 run。` : "没有发现新的时间戳 run。", added.length === 0);
      } catch (error) {
        showToast(`读取失败：${error.message}`, true);
      } finally {
        input.value = "";
      }
    });

    document.getElementById("clearButton").addEventListener("click", clearRuns);
    document.getElementById("exportButton").addEventListener("click", exportCurrentComparison);
    document.getElementById("sidebarToggle").addEventListener("click", () => {
      state.sidebarCollapsed = nextSidebarCollapsed(state.sidebarCollapsed);
      storeSidebarCollapsed(state.sidebarCollapsed);
      renderSidebarState();
    });
    document.getElementById("diffOnlyToggle").addEventListener("change", renderParamsTable);
    document.getElementById("paramSearch").addEventListener("input", renderParamsTable);
    document.getElementById("runSearch").addEventListener("input", renderRunList);
    document.getElementById("metricSelect").addEventListener("change", renderVisualCharts);
    document.getElementById("horizonMetricSelect").addEventListener("change", renderVisualCharts);
    document.getElementById("figureSearch").addEventListener("input", renderFigures);

    document.getElementById("runList").addEventListener("change", (event) => {
      const runId = event.target.dataset.runToggle;
      if (!runId) return;
      const run = state.runs.find((item) => item.id === runId);
      if (run) run.visible = event.target.checked;
      render();
    });

    document.querySelector(".content").addEventListener("click", (event) => {
      const cell = event.target.closest("[data-table-id][data-row][data-col]");
      if (!cell) {
        if (state.selectedCell) {
          state.selectedCell = null;
          renderMetricsTable();
          renderParamsTable();
        }
        return;
      }
      state.selectedCell = {
        tableId: cell.dataset.tableId,
        row: Number(cell.dataset.row),
        col: Number(cell.dataset.col),
      };
      renderMetricsTable();
      renderParamsTable();
    });

    document.getElementById("figureComparison").addEventListener("dblclick", (event) => {
      const image = event.target.closest("img[data-figure-group][data-run-id]");
      if (!image) return;
      openLightbox(image.dataset.figureGroup, image.dataset.runId);
    });

    document.getElementById("lightboxClose").addEventListener("click", closeLightbox);
    document.querySelector("[data-lightbox-close]").addEventListener("click", closeLightbox);
    document.getElementById("lightboxPrev").addEventListener("click", () => moveLightbox(-1));
    document.getElementById("lightboxNext").addEventListener("click", () => moveLightbox(1));
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && state.selectedCell) {
        state.selectedCell = null;
        renderMetricsTable();
        renderParamsTable();
      }
      if (!state.lightbox.open) return;
      if (event.key === "Escape") closeLightbox();
      if (event.key === "ArrowLeft") moveLightbox(-1);
      if (event.key === "ArrowRight") moveLightbox(1);
    });
  }

  function startBrowserApp() {
    if (typeof document === "undefined") return;
    bindEvents();
    render();
  }

  const api = {
    discoverRunsFromPaths,
    filterRunsBySearch,
    flattenObject,
    formatValue,
    getTableSelectionClasses,
    getMetricDatasetGroups,
    getMetricScore,
    groupFiguresByName,
    getLightboxItemsForGroup,
    nextSidebarCollapsed,
    parseCsv,
    parseSimpleYaml,
    summarizeFigureCoverage,
    summarizeRuns,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.VisualizerApp = api;

  startBrowserApp();
})(typeof window !== "undefined" ? window : globalThis);
