const assert = require("node:assert/strict");

const {
  flattenObject,
  parseSimpleYaml,
  discoverRunsFromPaths,
  formatValue,
  getMetricScore,
  groupFiguresByName,
  getLightboxItemsForGroup,
  getMetricDatasetGroups,
  getMetricKeys,
  filterRunsBySearch,
  filterRunsByVisibility,
  getBestRunForMetric,
  getRunNoteStorageKey,
  getTableSelectionClasses,
  nextSidebarCollapsed,
  getRunNoteFromSources,
  isRunHidden,
  normalizeRunPaths,
  updateHiddenRunPaths,
  summarizeFigureCoverage,
  summarizeRuns,
} = require("./app.js");

const config = parseSimpleYaml(`
seed: 42
model:
  name: improved_bnn
  hidden_dim: 256
training:
  lr: 0.00019544293665253097
  epochs: 100
prediction:
  mc_samples: 30
  plot:
    start_time: "2020-06-13 10:00:00"
`);

assert.equal(config.model.name, "improved_bnn");
assert.equal(config.model.hidden_dim, 256);
assert.equal(config.training.epochs, 100);
assert.equal(config.prediction.plot.start_time, "2020-06-13 10:00:00");

const flattened = flattenObject(config);
assert.equal(flattened["training.lr"], 0.00019544293665253097);
assert.equal(flattened["prediction.mc_samples"], 30);

const runs = discoverRunsFromPaths([
  "improved_bnn/20260510-193755/config.yaml",
  "improved_bnn/20260510-193755/metrics/metrics.json",
  "smoke/smoke_bnn/20260509-223920/config.yaml",
  "smoke/smoke_bnn/20260509-223920/figures/loss_curve.png",
  "smoke/not-a-run/readme.txt",
]);

assert.deepEqual(
  runs.map((run) => run.relativePath).sort(),
  ["improved_bnn/20260510-193755", "smoke/smoke_bnn/20260509-223920"],
);

assert.equal(formatValue(0.00019544293665253097), "0.000195");
assert.equal(formatValue(undefined), "-");
assert.equal(getMetricKeys().includes("crps"), true);

assert.equal(getRunNoteStorageKey("outputs/improved_bnn/20260510-193755"), "bnnVisualizer.note.outputs/improved_bnn/20260510-193755");
assert.equal(
  getBestRunForMetric(
    [
      { name: "run-a", testMetrics: { rmse: 120, picp_90: 0.88 }, validationMetrics: { rmse: 90 } },
      { name: "run-b", testMetrics: { rmse: 100, picp_90: 0.91 }, validationMetrics: { rmse: 110 } },
    ],
    "testMetrics",
    "rmse",
  ).name,
  "run-b",
);
assert.equal(
  getBestRunForMetric(
    [
      { name: "run-a", testMetrics: { picp_90: 0.88 } },
      { name: "run-b", testMetrics: { picp_90: 0.91 } },
    ],
    "testMetrics",
    "picp_90",
  ).name,
  "run-b",
);

assert.deepEqual(
  filterRunsBySearch(
    [
      { name: "20260510-193755", relativePath: "outputs/improved_bnn/20260510-193755" },
      { name: "20260509-223920", relativePath: "outputs/smoke/smoke_bnn/20260509-223920" },
    ],
    "smoke",
  ).map((run) => run.name),
  ["20260509-223920"],
);

assert.deepEqual(
  filterRunsByVisibility(
    [
      { name: "run-a", visible: true },
      { name: "run-b", visible: false },
      { name: "run-c", visible: true },
    ],
    "visible",
  ).map((run) => run.name),
  ["run-a", "run-c"],
);
assert.deepEqual(
  filterRunsByVisibility(
    [
      { name: "run-a", visible: true },
      { name: "run-b", visible: false },
      { name: "run-c", visible: true },
    ],
    "hidden",
  ).map((run) => run.name),
  ["run-b"],
);
assert.deepEqual(filterRunsByVisibility([{ name: "run-a", visible: false }], "all").map((run) => run.name), ["run-a"]);

assert.equal(nextSidebarCollapsed(false), true);
assert.equal(nextSidebarCollapsed(true), false);

assert.deepEqual(normalizeRunPaths([" outputs\\b\\20260510-193755 ", "", "outputs/b/20260510-193755"]), ["outputs/b/20260510-193755"]);
assert.equal(isRunHidden("outputs/b/20260510-193755", ["outputs/b/20260510-193755"]), true);
assert.equal(isRunHidden("outputs/b/20260510-193755", ["outputs/b/20260509-223920"]), false);
assert.deepEqual(updateHiddenRunPaths(["outputs/b/20260509-223920"], "outputs/b/20260510-193755", false), [
  "outputs/b/20260509-223920",
  "outputs/b/20260510-193755",
]);
assert.deepEqual(updateHiddenRunPaths(["outputs/b/20260510-193755"], "outputs/b/20260510-193755", true), []);
assert.equal(getRunNoteFromSources("outputs/b/20260510-193755", "note from file\n"), "note from file");

assert.equal(getTableSelectionClasses({ tableId: "metrics", row: 2, col: 3 }, "metrics", 2, 3), "cell-selected row-selected col-selected");
assert.equal(getTableSelectionClasses({ tableId: "metrics", row: 2, col: 3 }, "metrics", 2, 1), "row-selected");
assert.equal(getTableSelectionClasses({ tableId: "metrics", row: 2, col: 3 }, "params", 2, 3), "");

assert.deepEqual(
  getMetricDatasetGroups().map((group) => [group.label, group.field]),
  [
    ["测试集", "testMetrics"],
    ["验证集", "validationMetrics"],
  ],
);

const groupedFigures = groupFiguresByName([
  {
    id: "run-a-id",
    name: "run-a",
    figures: [
      { name: "loss_curve.png", path: "figures/loss_curve.png" },
      { name: "horizon_rmse.png", path: "figures/horizon_rmse.png" },
    ],
  },
  {
    id: "run-b-id",
    name: "run-b",
    figures: [{ name: "loss_curve.png", path: "figures/loss_curve.png" }],
  },
]);

assert.deepEqual(
  groupedFigures.map((group) => [group.name, group.items.map((item) => item.runName)]),
  [
    ["loss_curve.png", ["run-a", "run-b"]],
    ["horizon_rmse.png", ["run-a"]],
  ],
);

const coverage = summarizeFigureCoverage([
  {
    id: "run-a-id",
    name: "run-a",
    figures: [
      { name: "loss_curve.png", path: "figures/loss_curve.png" },
      { name: "horizon_rmse.png", path: "figures/horizon_rmse.png" },
    ],
  },
  {
    id: "run-b-id",
    name: "run-b",
    figures: [{ name: "loss_curve.png", path: "figures/loss_curve.png" }],
  },
]);

assert.deepEqual(coverage, {
  groupCount: 2,
  imageCount: 3,
  missingSlots: 1,
});

assert.deepEqual(
  groupFiguresByName(
    [
      { name: "run-a", figures: [{ name: "loss_curve.png", path: "figures/loss_curve.png" }] },
      { name: "run-b", figures: [{ name: "prediction_interval_90.png", path: "figures/prediction_interval_90.png" }] },
    ],
    "interval",
  ).map((group) => group.name),
  ["prediction_interval_90.png"],
);

const lightboxItems = getLightboxItemsForGroup(groupedFigures, "loss_curve.png");
assert.deepEqual(
  lightboxItems.map((item) => [item.runId, item.runName, item.figure.path]),
  [
    ["run-a-id", "run-a", "figures/loss_curve.png"],
    ["run-b-id", "run-b", "figures/loss_curve.png"],
  ],
);
assert.equal(lightboxItems.findIndex((item) => item.runId === "run-b-id"), 1);

assert.equal(getMetricScore("rmse", 100), 100);
assert.equal(getMetricScore("picp_90", 0.91), 0.01);

const summary = summarizeRuns([
  {
    name: "run-a",
    visible: true,
    testMetrics: { rmse: 100, picp_90: 0.91 },
    pointMetrics: [
      { horizon: 1, rmse: 80, mae: 40 },
      { horizon: 2, rmse: 90, mae: 50 },
    ],
  },
  {
    name: "run-b",
    visible: true,
    testMetrics: { rmse: 120, picp_90: 0.88 },
    pointMetrics: [{ horizon: 1, rmse: 95, mae: 55 }],
  },
]);

assert.equal(summary.bestMetrics.rmse, "run-a");
assert.equal(summary.bestMetrics.picp_90, "run-a");
assert.deepEqual(summary.horizonSeries.rmse[0].points, [{ x: 1, y: 80 }, { x: 2, y: 90 }]);
assert.equal(summary.horizonSeries.rmse[0].runName, "run-a");

console.log("visualizer app tests passed");
