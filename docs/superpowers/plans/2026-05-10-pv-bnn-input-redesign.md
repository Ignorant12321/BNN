# PV BNN Input Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the PV BNN inputs so history, weather, and direct match the approved paper-based semantics.

**Architecture:** Replace the current four-branch model with three semantic branches: past `AC_POWER` history, future weather plus hour, and previous-step `AC_POWER` direct input. The fused representation still uses Bayesian layers to produce mean and log variance for the 16-step horizon.

**Tech Stack:** Python, pandas, NumPy, scikit-learn scalers, PyTorch, pytest, YAML configs.

---

### Task 1: Tests For Correct Input Semantics

**Files:**
- Modify: `tests/test_features.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/test_leakage.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Update feature tests**

Assert `columns.history == ["AC_POWER"]`, `columns.weather == ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "hour"]`, `columns.time == []`, and `columns.direct == ["last_ac_power"]`.

- [ ] **Step 2: Update dataset tests**

Assert the first sample with `lookback=4, horizon=3` has `history[:, 0] == [0, 1, 2, 3]`, `weather` from rows `[4, 5, 6]`, `direct == [3]`, and `target == [4, 5, 6]`.

- [ ] **Step 3: Update model test**

Instantiate `ImprovedBayesianPVNet(history_features=1, weather_features=4, direct_features=1, horizon=16, hidden_dim=32)` and call it with only `history`, `weather`, and `direct`.

- [ ] **Step 4: Run failing tests**

Run: `pytest tests/test_features.py tests/test_dataset.py tests/test_leakage.py tests/test_model.py -q`

Expected before implementation: failures showing old feature groups and old model signature.

### Task 2: Feature And Window Refactor

**Files:**
- Modify: `src/features.py`
- Modify: `src/dataset.py`
- Modify: `src/train.py`

- [ ] **Step 1: Implement feature groups**

Add numeric `hour` in `add_basic_features`. Return one-column history, four-column weather, empty time list, and one-column direct.

- [ ] **Step 2: Simplify windows**

Make `weather` always use the target window rows. Keep `use_future_weather` accepted only for compatibility, but it no longer changes behavior.

- [ ] **Step 3: Preserve dataset API**

Keep `WindowArrays.time` and scaler handling compatible when `columns.time` is empty by returning arrays with shape `[samples, horizon, 0]` and skipping scaler fitting for empty time features.

- [ ] **Step 4: Run semantic tests**

Run: `pytest tests/test_features.py tests/test_dataset.py tests/test_leakage.py -q`

Expected after implementation: all pass.

### Task 3: Model Refactor

**Files:**
- Modify: `src/models/branches.py`
- Modify: `src/models/improved_bnn.py`
- Modify: `src/models/baselines.py`
- Modify: `src/train.py`

- [ ] **Step 1: Add history scalar sequence branch if needed**

Reuse `HistoryCNNBranch` for one-channel AC_POWER history, and keep `SequenceMLPBranch` for the weather plus hour sequence.

- [ ] **Step 2: Update main model signature**

Remove required `time_features`; forward should concatenate `history_branch(batch["history"])`, `weather_branch(batch["weather"])`, and `direct_branch(batch["direct"])`.

- [ ] **Step 3: Update train instantiation**

Pass `history_features`, `weather_features`, `direct_features`, `horizon`, `hidden_dim`, `branch_dim`, and `prior_sigma`.

- [ ] **Step 4: Run model tests**

Run: `pytest tests/test_model.py tests/test_training_runtime.py -q`

Expected after implementation: all pass.

### Task 4: Config And README Correction

**Files:**
- Modify: `configs/default.yaml`
- Modify: `configs/tuning.yaml`
- Modify: `configs/compare.yaml`
- Modify: `README.md`

- [ ] **Step 1: Update configs**

Set `data.lookback: 16` in all configs and leave `horizon: 16`.

- [ ] **Step 2: Update README**

Rewrite input descriptions, model structure, and limitations to match the new semantics.

- [ ] **Step 3: Search for stale claims**

Run: `rg "DC_POWER|last_dc_power|last_irradiation|dayofyear|month_sin|use_future_weather|8小时|8h|time 分支|四类" README.md src tests configs -n`

Expected: no stale architecture claims except raw data field explanations or compatibility comments.

### Task 5: Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_features.py tests/test_dataset.py tests/test_leakage.py tests/test_model.py -q`

Expected: pass.

- [ ] **Step 2: Run full test suite**

Run: `pytest -q`

Expected: pass.

- [ ] **Step 3: Report verification evidence**

Summarize changed files and exact test results in the final response.
