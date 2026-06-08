# Project_NNDL
This repository is created for completing capstone project for NNDL

# Predictive Maintenance — LSTM Classifier (NASA C-MAPSS)

Anomaly detection for industrial equipment: an LSTM classifier that flags turbofan engine failures **24 cycles ahead** on the NASA C-MAPSS (FD001) dataset.

Capstone project · HSE · Business Analytics & Big Data Systems.

## How to run (Kaggle)

The notebook is built to run on **Kaggle** with GPU.

1. Create a new Kaggle Notebook and upload `notebook.ipynb` (or copy the cells in).
2. In the right sidebar click **Add Input** and add the NASA C-MAPSS dataset:
   **`behrad3d/nasa-cmaps`** (search for "NASA C-MAPSS" / "behrad3d/nasa-cmaps").
3. (Recommended) Turn on a **GPU** accelerator: Settings → Accelerator → GPU.
4. Run all cells.

The first cell auto-detects the dataset path under `/kaggle/input`. If it is not attached, it falls back to downloading via `kagglehub`. If neither works, the notebook will stop with a clear message asking you to add the dataset.

> The notebook uses **FD001** (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) — single fault mode, single operating condition.

## What the notebook does

- Loads C-MAPSS FD001 and builds Remaining Useful Life (RUL) labels.
- EDA: drops 6 constant sensors, keeps 15 informative ones.
- Frames a supervised task: sliding windows of 30 cycles → label `1` if RUL ≤ 24.
- Trains a 2-layer LSTM classifier (~33k params) with class weighting, EarlyStopping and ReduceLROnPlateau.
- Selects the decision threshold **on validation** (max recall subject to FPR < 5%), then evaluates on test.
- Reports KPIs (Recall, FPR) at window and engine level, plus SHAP feature attributions.

## KPI targets

| Metric | Target | Result (test) |
|--------|--------|---------------|
| Recall | ≥ 85% | 88.9% ✓ |
| FPR    | < 5%  | 1.5% ✓ |

Engine level: 19/19 failing engines caught ≥24 cycles ahead.

## Requirements

Runs out of the box on the default Kaggle Python image (TensorFlow, scikit-learn, pandas, matplotlib, seaborn). For SHAP, add `shap` if it is not preinstalled:

```bash
pip install shap
```

## Team

Anastasia Zaporozhets · Vladimir Kuznetsov · Pavel Manuilov
