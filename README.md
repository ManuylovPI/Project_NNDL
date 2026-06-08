# Project_NNDL
This repository is created for completing capstone project for NNDL

# Predictive Maintenance — LSTM Classifier (NASA C-MAPSS)

Anomaly detection for industrial equipment: an LSTM classifier that flags turbofan engine failures **24 cycles ahead** on the NASA C-MAPSS (FD001) dataset.

Capstone project · HSE · Business Analytics & Big Data Systems.

## How to run (Kaggle)

The notebook is built to run on **Kaggle** with GPU.

1. Create a new Kaggle Notebook and upload `notebook.ipynb` (or copy the cells in).
2. In the right sidebar click **Add Input** and add the NASA C-MAPSS dataset:
   ```bash
   https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
   ```
4. (Recommended) Turn on a **GPU** accelerator: Settings → Accelerator → GPU.
5. Run all cells.

## KPI targets

| Metric | Target | Result (test) |
|--------|--------|---------------|
| Recall | ≥ 85% | 88.9% ✓ |
| FPR    | < 5%  | 1.5% ✓ |

Engine level: 19/19 failing engines caught ≥24 cycles ahead.

## Team

Anastasia Zaporozhets · Vladimir Kuznetsov · Pavel Manuilov
