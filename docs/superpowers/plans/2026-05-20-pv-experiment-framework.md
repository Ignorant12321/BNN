# PV Experiment Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the project as a clean photovoltaic probabilistic forecasting experiment framework that supports single-model training and multi-model comparison.

**Architecture:** The first rebuild phase creates a small, testable core: config loading, model registry, shared probabilistic model interface, experiment spec parsing, and lightweight smoke training. Data, model, training, evaluation, and experiment orchestration live in separate packages.

**Tech Stack:** Python, PyYAML, NumPy, optional PyTorch, pytest.

---

### Task 1: Core Skeleton

**Files:**
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `src/models/registry.py`
- Create: `src/models/baselines.py`
- Create: `src/models/improved_bnn.py`
- Create: `src/experiments/compare.py`
- Create: `src/experiments/train.py`
- Create: `tests/test_config.py`
- Create: `tests/test_models.py`
- Create: `tests/test_compare.py`

- [x] Write tests for config loading, model registry, and compare spec parsing.
- [x] Run tests to verify the project fails because modules are missing.
- [x] Implement minimal modules to satisfy the tests.
- [x] Run tests until green.

### Task 2: Usable Entrypoints

**Files:**
- Create: `configs/data.yaml`
- Create: `configs/models/bnn_24h.yaml`
- Create: `configs/compare/main.yaml`
- Create: `README.md`
- Create: `requirements.txt`

- [x] Add single-model and compare YAML examples.
- [x] Add command examples to README.
- [x] Verify CLI help works.

### Task 3: Real CSV Windows and Trainable NumPy Models

**Files:**
- Create: `src/data/pv.py`
- Modify: `src/experiments/train.py`
- Modify: `src/experiments/compare.py`
- Modify: `src/models/baselines.py`
- Modify: `src/models/improved_bnn.py`
- Test: `tests/test_data_pipeline.py`
- Test: `tests/test_models.py`
- Test: `tests/test_compare.py`

- [x] Add tests for generation aggregation, weather merge, and sliding-window shapes.
- [x] Add tests requiring model `fit()` to reduce error on a simple direct-signal dataset.
- [x] Implement real CSV loading and window construction.
- [x] Implement NumPy ridge probabilistic models.
- [x] Update training to fit models before writing metrics.
- [x] Update comparison summary to include run metrics.
- [x] Run tests until green.

### Task 4: Full-Window Evaluation

**Files:**
- Modify: `src/experiments/train.py`
- Test: `tests/test_training.py`

- [x] Add a test proving evaluation uses every available window.
- [x] Extract `evaluate_model()`.
- [x] Update training to evaluate all windows after fitting.
- [x] Run tests and real single/compare commands.
