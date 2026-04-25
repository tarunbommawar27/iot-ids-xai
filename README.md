# IoT IDS XAI — Explainable Intrusion Detection for IoT Networks

[![Live API](https://img.shields.io/badge/Live%20API-Hugging%20Face%20Spaces-blue)](https://tbommawa-iot-ids-api.hf.space)
[![Swagger UI](https://img.shields.io/badge/Swagger-Interactive%20Docs-green)](https://tbommawa-iot-ids-api.hf.space/docs)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Paper:** *Explainable Machine Learning for IoT Intrusion Detection: A Comprehensive Literature Review and Research Framework*
> **Authors:** Manasa Mummadi · Tanuja Konda Reddy · Tarun Bommawar — George Mason University

---

## Table of Contents

- [Project Overview](#project-overview)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Quick Start — Live Demo](#quick-start--live-demo)
- [Full Reproduction Guide](#full-reproduction-guide)
  - [Step 1 — Prerequisites](#step-1--prerequisites)
  - [Step 2 — Get Kaggle Credentials](#step-2--get-kaggle-credentials)
  - [Step 3 — Open the Notebook in Colab](#step-3--open-the-notebook-in-colab)
  - [Step 4 — Configure Colab Runtime](#step-4--configure-colab-runtime)
  - [Step 5 — Execute Cells in Order](#step-5--execute-cells-in-order)
  - [Step 6 — Download Outputs](#step-6--download-outputs)
- [API Reference](#api-reference)
  - [Endpoints](#endpoints)
  - [Request and Response Format](#request-and-response-format)
  - [Example Requests](#example-requests)
- [Hugging Face Deployment Guide](#hugging-face-deployment-guide)
  - [Step 1 — Create a Space](#step-1--create-a-space)
  - [Step 2 — Upload Files](#step-2--upload-files)
  - [Step 3 — Monitor Build](#step-3--monitor-build)
  - [Step 4 — Verify Deployment](#step-4--verify-deployment)
- [Live Traffic Simulator](#live-traffic-simulator)
- [Project Architecture](#project-architecture)
- [Datasets](#datasets)
- [Models](#models)
- [Key Findings](#key-findings)
- [Software Stack](#software-stack)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

This project implements a five-stage dual-tier explainable intrusion detection system for IoT networks. It trains four machine learning classifiers (Random Forest, XGBoost, LightGBM, MLP) on three benchmark IoT datasets and applies SHAP and LIME explanations through a confidence-gated routing architecture:

- **Edge tier** — LightGBM inference at ~0.003 ms, LIME explanation at 15–16 ms (under the 35 ms IoT gateway budget)
- **Cloud tier** — RandomForest with SHAP for offline forensic audit at 94–158 ms
- **Escalation logic** — if prediction confidence falls below 0.85, the request automatically escalates from LIME to SHAP

The deployed API is publicly accessible and the entire pipeline can be reproduced from scratch using the provided Colab notebook.

---

## Results Summary

| Dataset | LightGBM F1 | 5-Fold CV F1 | LIME Latency | SMOTE Fidelity (ρ) |
|---|---|---|---|---|
| CIC-IoT-2023 | 0.9539 | 0.9518 ± 0.0009 | 15.8 ms ✅ | 0.669 ⚠️ Distorted |
| TON-IoT | 0.9975 | 0.9978 ± 0.0003 | 16.4 ms ✅ | 0.856 ✅ OK |
| UNSW-NB15 | 0.9644 | 0.9665 ± 0.0014 | 15.1 ms ✅ | 0.883 ✅ OK |

**Key novel finding:** SMOTE distorts SHAP/LIME feature importance rankings on CIC-IoT-2023/LightGBM (ρ = 0.669 < 0.8 threshold), despite achieving 99.56% accuracy. RandomForest on the same dataset maintains strong fidelity (ρ = 0.953). This empirically confirms the *Gap 3 fidelity paradox* described in the paper.

---

## Repository Structure

```
.
├── IoT_IDS_XAI_Complete_v2.ipynb   ← Main training notebook (run this in Colab)
├── app/
│   └── main.py                      ← FastAPI inference server
├── models/
│   ├── ciciot2023_lightgbm_model.pkl
│   ├── ciciot2023_randomforest_model.pkl
│   ├── ciciot2023_scaler.pkl
│   ├── ciciot2023_features.json
│   ├── toniot_lightgbm_model.pkl
│   ├── toniot_randomforest_model.pkl
│   ├── toniot_scaler.pkl
│   ├── toniot_features.json
│   ├── unswnb15_lightgbm_model.pkl
│   ├── unswnb15_randomforest_model.pkl
│   ├── unswnb15_scaler.pkl
│   └── unswnb15_features.json
├── Dockerfile                        ← Container definition for HF Spaces
├── requirements.txt                  ← Python dependencies
└── README.md                         ← This file
```

> **Note:** The `models/` folder contains pre-trained models ready for immediate API use. To retrain from scratch, follow the [Full Reproduction Guide](#full-reproduction-guide).

---

## Quick Start — Live Demo

The API is already deployed and running. No installation required.

**Option 1 — Open the web dashboard:**

```
https://tbommawa-iot-ids-api.hf.space
```

**Option 2 — Run a preloaded real test sample:**

```bash
# CIC-IoT-2023 — benign sample
curl https://tbommawa-iot-ids-api.hf.space/demo/ciciot2023

# CIC-IoT-2023 — attack sample (index 5)
curl https://tbommawa-iot-ids-api.hf.space/demo/ciciot2023?sample_idx=5

# TON-IoT — attack sample
curl https://tbommawa-iot-ids-api.hf.space/demo/toniot?sample_idx=5

# UNSW-NB15 — benign sample
curl https://tbommawa-iot-ids-api.hf.space/demo/unswnb15
```

**Option 3 — Interactive Swagger UI:**

```
https://tbommawa-iot-ids-api.hf.space/docs
```

Open the `/predict` endpoint, click "Try it out", paste your feature values, and execute.

**Option 4 — Live traffic simulator:**

Open `iot_ids_traffic_simulator_v2.html` in any browser (Chrome, Firefox, Edge). Press **▶ Start**. The simulator hits the live API and streams real predictions with LIME explanations in real time.

---

## Full Reproduction Guide

Follow these steps exactly to reproduce all results from scratch.

### Step 1 — Prerequisites

You need:
- A Google account (for Google Colab)
- A Kaggle account (free) — to download the datasets
- A browser

You do **not** need to install anything locally. Everything runs in the cloud.

---

### Step 2 — Get Kaggle Credentials

The notebook downloads all three datasets automatically from Kaggle. You need a Kaggle API key file (`kaggle.json`) for this.

1. Go to [kaggle.com](https://kaggle.com) and sign in (create a free account if you don't have one)
2. Click your profile picture in the top right corner
3. Click **Settings**
4. Scroll down to the **API** section
5. Click **Create New Token**
6. A file called `kaggle.json` downloads to your computer — **save it**, you will upload it in Colab

The file looks like this (your actual credentials will be different):

```json
{
  "username": "yourkaggleusername",
  "key": "abc123def456ghi789jkl012mno345pq"
}
```

---

### Step 3 — Open the Notebook in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `IoT_IDS_XAI_Complete_v2.ipynb` from this repository
4. The notebook opens with 20 cells ready to run

---

### Step 4 — Configure Colab Runtime

This step is critical. The default Colab runtime does not have enough memory.

1. In Colab, click **Runtime** in the top menu
2. Click **Change runtime type**
3. Set **Hardware accelerator** to **T4 GPU**
4. Set **Runtime shape** to **High-RAM** (if on Colab Pro) or leave default (standard RAM still works but may be slower)
5. Click **Save**

> **Why T4 GPU?** LightGBM and XGBoost use the GPU for histogram-based training. Without it, XGBoost falls back to CPU and training takes 3–5× longer. The MLP model requires GPU for reasonable training speed.

---

### Step 5 — Execute Cells in Order

Run each cell sequentially by pressing **Shift + Enter** or clicking the play button. Do **not** skip cells or run them out of order.

#### Cell 1 — Install dependencies

```
Runtime: ~2 minutes
```

Installs all required packages: kaggle, xgboost, lightgbm, shap, lime, imbalanced-learn, scikit-learn, onnxmltools, skl2onnx, onnxruntime.

When it finishes you will see:

```
✅ xgboost OK
✅ lightgbm OK
✅ shap OK
✅ lime OK
✅ imblearn OK
✅ skl2onnx OK
✅ onnxmltools OK
✅ onnxruntime OK
```

If any show ❌, restart the runtime (Runtime → Restart runtime) and run Cell 1 again.

---

#### Cell 2 — Kaggle authentication and dataset download

```
Runtime: 10–20 minutes (depending on your connection speed)
```

1. When the cell runs, a file upload dialog appears in the output
2. Click **Choose Files**
3. Select the `kaggle.json` file you downloaded in Step 2
4. The cell saves your credentials and downloads all three datasets automatically

Expected output:

```
📁 Upload your kaggle.json file...
✅ Kaggle credentials saved for: yourusername

⬇️  Downloading CIC-IoT-2023  [madhavmalhotra/unb-cic-iot-dataset]...
   ✅ Success
   ['part1.csv', 'part2.csv', ...]

⬇️  Downloading TON-IoT  [arnobbhowmik/ton-iot-network-dataset]...
   ✅ Success

⬇️  Downloading UNSW-NB15  [mrwellsdavid/unsw-nb15]...
   ✅ Success
```

> **TON-IoT fallback:** If the primary slug fails, the cell automatically tries three alternative Kaggle slugs. This is normal — just let it run.

> **If all TON-IoT slugs fail:** Go to [kaggle.com/datasets](https://kaggle.com/datasets) and search "TON IoT network". Download any result that is the network intrusion version (not the sensor-only version). Upload the CSV files manually to `/content/datasets/toniot/` using the Colab file browser on the left sidebar.

---

#### Cell 3 — Imports and global config

```
Runtime: ~5 seconds
```

Sets up all imports and the CONFIG dictionary. You will see:

```
✅ Config loaded
   SAMPLE_SIZE: 300000
   TOP_K_FEATURES: 20
   CV_FOLDS: 5
   SEED: 42
   ...
```

No action needed.

---

#### Cell 4 — UNSW-NB15 header fix

```
Runtime: ~1 minute
```

UNSW-NB15 CSV files have no column headers. This cell adds them and removes the `attack_cat` column, which is a label leakage column that would artificially inflate accuracy to 100%.

Expected output:

```
✅ NUSW-NB15_GT.csv: 175,341 rows  label dist={0: 93000, 1: 45341}
✅ UNSW-NB15_1.csv: 700,000 rows   label dist={0: 537000, 1: 163000}
attack_cat present: False   ← must be False
```

> **Important:** The line `attack_cat present: False` must say False. If it says True, re-run this cell.

---

#### Cell 5 — Data loading utilities

```
Runtime: ~2 seconds
```

Defines helper functions. Output:

```
✅ Loading utilities ready
```

---

#### Cell 6 — Preprocessing pipeline

```
Runtime: ~2 seconds
```

Defines the preprocessing function. Output:

```
✅ Preprocessing function ready
```

---

#### Cell 7 — Leakage detection utility

```
Runtime: ~2 seconds
```

Defines the leakage detection function that checks Spearman correlation between each feature and the label. Output:

```
✅ Leakage detection utility ready
```

---

#### Cell 8 — Model definitions and training

```
Runtime: ~2 seconds
```

Defines the four classifiers (RandomForest, XGBoost, LightGBM, MLP) and the training loop. Output:

```
✅ Model functions ready
```

---

#### Cell 9 — XAI: SHAP and LIME

```
Runtime: ~2 seconds
```

Defines the SHAP TreeExplainer and LIME LimeTabularExplainer wrapper functions. Output:

```
✅ XAI functions ready
```

---

#### Cell 10 — Gap 3: SMOTE fidelity analysis

```
Runtime: ~2 seconds
```

Defines the function that trains pre-SMOTE and post-SMOTE models and computes Spearman rank correlation between their SHAP rankings. Output:

```
✅ Gap 3 function ready
```

---

#### Cell 11 — 5-fold cross-validation

```
Runtime: ~2 seconds
```

Defines the CV function that applies SMOTE inside each fold correctly. Output:

```
✅ Cross-validation function ready
```

---

#### Cell 12 — Visualization helpers

```
Runtime: ~2 seconds
```

Defines all plotting functions. Output:

```
✅ Visualization helpers ready
```

---

#### Cell 13 — MAIN PIPELINE — Run everything

```
Runtime: 60–90 minutes total
  CIC-IoT-2023: ~30 minutes
  TON-IoT:      ~15 minutes
  UNSW-NB15:    ~15 minutes
  Cross-validation: ~20 minutes
```

**This is the main cell.** It loops over all three datasets and runs the complete pipeline for each:

1. Load and sample 300,000 rows
2. Check for leakage (prints warning if any feature has |ρ| > 0.95 with label)
3. Preprocess (clean, encode, StandardScaler, mutual info feature selection, SMOTE)
4. Train all four models
5. Run SHAP + LIME on each model
6. Run Gap 3 SMOTE fidelity analysis
7. Run 5-fold cross-validation
8. Store everything in `ALL_RESULTS`

You will see live output for each step. Example for one dataset:

```
######################################################################
#  DATASET: TON-IoT
######################################################################
  Label col: "label"
  Unique values: [0, 1]
  ...
  Cleaned: 300,000 → 298,341 rows
  Top-20 features (MI): ['conn_state', 'dns_rejected', ...]
  SMOTE: [149170 149170] (0=benign, 1=attack)
  Split: train=208819  val=44747  test=44775

  Training RandomForest...
    ✅ Acc=0.9983 F1=0.9975 MCC=0.9950 Lat=0.0355ms

  Training LightGBM...
    ✅ Acc=0.9983 F1=0.9975 MCC=0.9949 Lat=0.003ms

  ...
  Gap3 [LightGBM] on TON-IoT...
  Spearman rho=0.8556  p=0.000001  ✅ OK

  🔁 5-Fold CV [TON-IoT]
    [LightGBM] fold 1: F1=0.9981  MCC=0.9961
    [LightGBM] fold 2: F1=0.9978  MCC=0.9955
    ...
  ✅ LightGBM: F1=0.9978±0.0003

✅ TON-IoT complete — stored in ALL_RESULTS
```

> **If Colab disconnects mid-run:** This sometimes happens on long sessions. Re-run Cell 3 to restore CONFIG and ALL_RESULTS, then re-run Cell 13. The session variables are lost on disconnect but the cell will redo the work.

---

#### Cell 14 — Master results table and CSV

```
Runtime: ~10 seconds
```

Generates the complete results table and saves two CSV files:

```
✅ Saved: /content/master_results.csv
✅ Saved: /content/cv_results.csv
```

---

#### Cell 15 — Cross-dataset heatmaps

```
Runtime: ~15 seconds
```

Generates and displays three heatmaps (Macro-F1, SMOTE ρ, LIME latency). Saves:

```
✅ Saved: /content/cross_dataset_heatmap.png
```

---

#### Cell 16 — Research gap analysis printout

```
Runtime: ~5 seconds
```

Prints a structured summary of Gap 1–4 validation results with actual numbers.

---

#### Cell 17 — Save models for deployment

```
Runtime: ~2 minutes
```

Saves all trained LightGBM and RandomForest models, scalers, and feature JSON files. Then downloads a zip:

```
✅ exported_models.zip  (8.9 MB)
⬇️  Starting download...
```

**Save this zip to your computer.** It contains:

```
ciciot2023_lightgbm_model.pkl    (301 KB)
ciciot2023_randomforest_model.pkl  (2.0 MB)
ciciot2023_scaler.pkl
ciciot2023_features.json
toniot_lightgbm_model.pkl        (295 KB)
toniot_randomforest_model.pkl    (2.0 MB)
...
```

---

#### Cell 18 — ONNX export (optional)

```
Runtime: ~3 minutes
```

Exports RandomForest models to ONNX format for ARM edge hardware deployment. Downloads `onnx_models.zip`. This is optional for the paper demo but required for actual Raspberry Pi deployment.

---

#### Cell 19 — Extract real test samples

```
Runtime: ~1 minute
```

Extracts 10 real pre-scaled test samples per dataset (5 benign + 5 attack) and downloads `all_real_samples.json`. These are used by the live traffic simulator.

---

#### Cell 20 — Download all outputs

```
Runtime: ~30 seconds
```

Packages all PNG figures, CSV files, and JSON into a final zip and downloads it.

```
✅ All downloads started
```

**Contents of the final zip:**

| File | Description |
|---|---|
| `master_results.csv` | All metrics for all 12 model-dataset combinations |
| `cv_results.csv` | 5-fold CV mean ± std results |
| `cross_dataset_heatmap.png` | Full comparison heatmap |
| `cv_results.png` | CV stability bar chart |
| `metrics_*.png` | Per-dataset accuracy/F1/MCC bar charts (3 files) |
| `latency_*.png` | LIME latency charts with 35 ms threshold (3 files) |
| `gap3_*.png` | SMOTE fidelity ρ bar charts (3 files) |
| `shap_LightGBM_*.png` | SHAP feature importance plots (3 files) |
| `cmp_LightGBM_*.png` | SHAP vs LIME comparison plots (3 files) |

---

### Step 6 — Download Outputs

By the end of Cell 20 your computer should have received these files:

- `exported_models.zip` — trained models for API deployment
- `onnx_models.zip` — ONNX models for edge hardware (optional)
- `all_real_samples.json` — real test samples for the simulator
- `iot_ids_complete_results.zip` — all figures and CSVs

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web dashboard with results table and live demo buttons |
| GET | `/health` | Model load status — confirms all 6 models are ready |
| GET | `/docs` | Interactive Swagger UI |
| GET | `/redoc` | ReDoc API reference |
| GET | `/datasets` | Feature lists and model availability per dataset |
| GET | `/demo/{dataset}` | Run a preloaded real test sample |
| POST | `/predict` | Single prediction with optional LIME or SHAP explanation |
| POST | `/batch_predict` | Batch inference without XAI (for latency benchmarking) |

**Demo endpoint parameters:**

| Parameter | Values | Default | Description |
|---|---|---|---|
| `dataset` | `ciciot2023`, `toniot`, `unswnb15` | required | Which dataset |
| `sample_idx` | 0–9 | 0 | 0–4 = benign, 5–9 = attack |
| `model` | `lightgbm`, `randomforest` | `lightgbm` | Which model |

---

### Request and Response Format

**POST /predict — request body:**

```json
{
  "dataset": "ciciot2023",
  "model": "lightgbm",
  "features": {
    "Number": 0.033,
    "Weight": -0.134,
    "Duration": -0.152,
    "IAT": 0.023,
    "urg_count": -0.024,
    "rst_count": -0.121,
    "syn_count": -0.012,
    "ack_flag_number": -0.105,
    "psh_flag_number": -0.092,
    "Header_Length": -0.176,
    "flow_duration": -0.189,
    "Variance": -0.107,
    "Std": -0.148,
    "Tot size": -0.123,
    "Tot sum": -0.134,
    "Magnitue": 0.012,
    "Rate": -0.089,
    "Srate": -0.091,
    "HTTPS": 1.234,
    "Max": -0.102
  },
  "explain": true,
  "xai_method": "lime",
  "confidence_threshold": 0.85
}
```

**Field descriptions:**

| Field | Type | Required | Description |
|---|---|---|---|
| `dataset` | string | yes | `ciciot2023`, `toniot`, or `unswnb15` |
| `model` | string | no | `lightgbm` (default) or `randomforest` |
| `features` | object | yes | Feature name → scaled float value (see `/datasets` for feature names) |
| `explain` | boolean | no | Whether to run XAI (default: true) |
| `xai_method` | string | no | `lime` (default), `shap`, or `none` |
| `confidence_threshold` | float | no | Below this, SHAP is used instead of LIME (default: 0.85) |

> **Feature values must be pre-scaled** using StandardScaler fitted on the training data. The pre-trained scaler is already embedded in the API and applied automatically from the stored `.pkl` files. If you are sending raw unscaled values, the API will produce incorrect predictions.

**POST /predict — response:**

```json
{
  "dataset": "ciciot2023",
  "model": "lightgbm",
  "prediction": "Attack",
  "confidence": 0.9978,
  "latency_ms": 1.746,
  "escalated": false,
  "explanations": [
    {
      "method": "lime",
      "top_features": [
        {"feature": "Number", "weight": 0.0079},
        {"feature": "Magnitue", "weight": 0.0058},
        {"feature": "IAT", "weight": 0.0059},
        {"feature": "Duration", "weight": 0.0046},
        {"feature": "Tot size", "weight": -0.0056}
      ],
      "latency_ms": 5.98
    }
  ],
  "feature_count": 20,
  "timestamp": 1777149924.698
}
```

**Dual-tier routing logic:**

- If `confidence >= confidence_threshold` → LIME explanation returned (edge tier, fast)
- If `confidence < confidence_threshold` → SHAP explanation returned (cloud tier, thorough), `escalated: true`

---

### Example Requests

**Check all models are loaded:**

```bash
curl https://tbommawa-iot-ids-api.hf.space/health
```

**Get feature names for a dataset:**

```bash
curl https://tbommawa-iot-ids-api.hf.space/datasets | python3 -m json.tool
```

**Run demo — TON-IoT attack sample:**

```bash
curl "https://tbommawa-iot-ids-api.hf.space/demo/toniot?sample_idx=5"
```

**Custom prediction with SHAP:**

```bash
curl -X POST https://tbommawa-iot-ids-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "toniot",
    "model": "lightgbm",
    "features": {
      "src_ip_bytes": 8.48, "dst_port": 7.39, "dst_ip_bytes": 9.10,
      "src_port": 8.11, "conn_state": 6.96, "src_pkts": 8.18,
      "dst_pkts": 8.90, "src_bytes": 7.99, "dst_bytes": 8.91,
      "duration": 7.83, "dns_AA": 0.0, "dns_RA": 0.0,
      "dns_RD": 0.0, "dns_rejected": 1.0, "dns_qtype": 7.26,
      "dns_qclass": 7.39, "dns_rcode": 8.90, "service": -0.88,
      "ssl_established": 0.0, "ssl_resumed": 0.0
    },
    "explain": true,
    "xai_method": "shap"
  }'
```

**Batch prediction (no XAI, for speed benchmarking):**

```bash
curl -X POST "https://tbommawa-iot-ids-api.hf.space/batch_predict?dataset=toniot&model=lightgbm" \
  -H "Content-Type: application/json" \
  -d '[
    {"src_ip_bytes": 8.48, "conn_state": 6.96},
    {"src_ip_bytes": -0.32, "conn_state": -0.54}
  ]'
```

---

## Hugging Face Deployment Guide

If you want to deploy your own instance of the API (e.g., after retraining with updated models), follow these steps.

### Step 1 — Create a Space

1. Go to [huggingface.co](https://huggingface.co) and sign in (create a free account if needed)
2. Click your profile picture → **New Space**
3. Fill in the form:
   - **Space name:** `iot-ids-api` (or any name you prefer)
   - **License:** MIT
   - **SDK:** Docker ← this is required, do not select Gradio or Streamlit
   - **Visibility:** Public
4. Click **Create Space**

You will land on an empty repository page.

---

### Step 2 — Upload Files

Click the **Files** tab in your Space. You need to create the following structure by uploading files one by one using the **Add file → Upload files** button:

```
Space root/
├── Dockerfile
├── requirements.txt
├── README.md
└── app/
    └── main.py
└── models/
    ├── ciciot2023_lightgbm_model.pkl
    ├── ciciot2023_randomforest_model.pkl
    ├── ciciot2023_scaler.pkl
    ├── ciciot2023_features.json
    ├── toniot_lightgbm_model.pkl
    ├── toniot_randomforest_model.pkl
    ├── toniot_scaler.pkl
    ├── toniot_features.json
    ├── unswnb15_lightgbm_model.pkl
    ├── unswnb15_randomforest_model.pkl
    ├── unswnb15_scaler.pkl
    └── unswnb15_features.json
```

**To create a subfolder in Hugging Face:**

1. Click **Add file → Create new file**
2. In the filename field, type `app/main.py` — typing the slash automatically creates the folder
3. Paste the content of `main.py` and click **Commit new file**

**To upload the model `.pkl` files:**

1. Click **Add file → Upload files**
2. In the filename field at the top, type `models/` to target the models subfolder
3. Drag all 12 model files at once and commit

> **File size note:** The RandomForest models for UNSW-NB15 is 4.2 MB. Hugging Face allows files up to 50 MB without Git LFS. All model files in this project are under that limit and upload normally.

---

### Step 3 — Monitor Build

After each file upload, Hugging Face triggers an automatic Docker build. You can watch the build progress by clicking the **Build** tab (or **Logs** tab depending on your Space version).

The first build takes 8–12 minutes because it installs scikit-learn, LightGBM, SHAP, and LIME from scratch.

Build stages you will see:

```
Building Docker image...
Step 1/7 : FROM python:3.10-slim
Step 2/7 : RUN apt-get update && apt-get install -y libgomp1 gcc
Step 3/7 : COPY requirements.txt .
Step 4/7 : RUN pip install --no-cache-dir -r requirements.txt
...
Successfully built
Container starting...
Loading models...
  Scaler loaded: ciciot2023
  Model loaded: ciciot2023/lightgbm
  Model loaded: ciciot2023/randomforest
  ...
Registry ready: ['ciciot2023', 'toniot', 'unswnb15']
```

---

### Step 4 — Verify Deployment

Once the Space status shows **Running**, open these URLs to confirm everything is working:

```
https://YOUR-USERNAME-iot-ids-api.hf.space/health
```

Expected response:

```json
{
  "status": "ok",
  "shap_available": true,
  "lime_available": true,
  "datasets": {
    "ciciot2023": {"scaler": true, "feature_count": 20, "models": ["lightgbm", "randomforest"]},
    "toniot":     {"scaler": true, "feature_count": 20, "models": ["lightgbm", "randomforest"]},
    "unswnb15":   {"scaler": true, "feature_count": 20, "models": ["lightgbm", "randomforest"]}
  }
}
```

If any model shows `false` or is missing from the list, check that the corresponding `.pkl` file is in the `models/` folder with the exact filename shown in the repository structure.

---

## Live Traffic Simulator

The file `iot_ids_traffic_simulator_v2.html` is a standalone web application that sends live traffic to the deployed API and visualizes predictions in real time.

**To use it:**

1. Download `iot_ids_traffic_simulator_v2.html` from this repository
2. Open it in any modern browser (Chrome, Firefox, Edge, Safari)
3. No server or installation needed — it runs entirely in the browser

**Controls:**

| Control | Options | Description |
|---|---|---|
| Dataset | All (rotating), CIC-IoT-2023, TON-IoT, UNSW-NB15 | Which dataset to send traffic for |
| Model | LightGBM (edge), RandomForest (cloud) | Which model to call |
| XAI | LIME (fast), SHAP (thorough), None | Which explanation method |
| Speed | 0.6s – 5s | Delay between packets |

**What you see:**

- **Live feed:** Each row shows packet number, dataset, model, verdict (Attack/Benign), confidence bar, inference latency, LIME latency, and top feature
- **SHAP badge:** Packets with confidence below 85% show a gold SHAP badge — these were automatically escalated to the cloud tier
- **Detail panel:** Click any packet row to see the full LIME feature importance bar chart, raw feature vector, and metadata
- **Stats bar:** Running totals of attacks, benign, attack rate, average latency, average LIME latency, and escalation count

---

## Project Architecture

```
IoT Network Traffic
        │
        ▼
┌─────────────────────────────────────┐
│  Stage 1: Data Ingestion            │
│  CIC-IoT-2023 │ TON-IoT │ UNSW-NB15│
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Stage 2: Preprocessing             │
│  Clean → Encode → Scale →           │
│  Mutual Info Select → SMOTE         │
│  (train split only)                 │
└──────────────────┬──────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│ RandomForest │    │    LightGBM      │
│  XGBoost     │    │      MLP         │
└──────┬───────┘    └────────┬─────────┘
       │                     │
       ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│    SHAP      │    │      LIME        │
│  (cloud)     │    │     (edge)       │
│  94–158 ms   │    │    15–16 ms      │
└──────┬───────┘    └────────┬─────────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Output: Alert + Explanation        │
│  Prediction │ Confidence │ Features │
└─────────────────────────────────────┘
```

**Confidence-gated routing (implemented in the API):**

```
Incoming request
      │
      ▼
  Run LightGBM
      │
 confidence >= 0.85?
    │          │
   YES          NO
    │          │
  LIME        SHAP
  (edge)     (cloud escalation)
  15ms        94–158ms
```

---

## Datasets

| Dataset | Rows | Features | Attack Types | Label |
|---|---|---|---|---|
| CIC-IoT-2023 | ~46.7M (sampled to 300K) | 47 (top 20 selected) | 33 sub-types across 7 categories | `label` column: `BenignTraffic` or attack name |
| TON-IoT | variable | 43 (top 20 selected) | 9 categories | `label` column: `0` (benign) or `1` (attack) |
| UNSW-NB15 | ~2.54M (sampled to 300K) | 49 (top 20 selected) | 9 attack families | `label` column: `0` (benign) or `1` (attack) |

**Feature sets used by the trained models (top 20 by mutual information):**

*CIC-IoT-2023:* Number, Weight, Duration, IAT, urg\_count, rst\_count, syn\_count, ack\_flag\_number, psh\_flag\_number, Header\_Length, flow\_duration, Variance, Std, Tot size, Tot sum, Magnitue, Rate, Srate, HTTPS, Max

*TON-IoT:* src\_ip\_bytes, dst\_port, dst\_ip\_bytes, src\_port, conn\_state, src\_pkts, dst\_pkts, src\_bytes, dst\_bytes, duration, dns\_AA, dns\_RA, dns\_RD, dns\_rejected, dns\_qtype, dns\_qclass, dns\_rcode, service, ssl\_established, ssl\_resumed

*UNSW-NB15:* sttl, ct\_state\_ttl, sbytes, dttl, Sload, ct\_srv\_dst, smeansz, Dload, service, ct\_srv\_src, dmeansz, swin, dwin, ct\_dst\_src\_ltm, ct\_src\_ltm, Ltime, Stime, state, ct\_src\_dport\_ltm, tcprtt

---

## Models

| Model | Tier | Inference | LIME | Size |
|---|---|---|---|---|
| LightGBM | Edge | ~0.003 ms | 15–16 ms ✅ | 295–301 KB |
| RandomForest | Cloud | ~0.035 ms | 39–41 ms ❌ | 2.0–4.2 MB |
| XGBoost | Evaluation only | ~0.007 ms | — | — |
| MLP | Evaluation only | ~0.002 ms | — | — |

**Hyperparameters:**

| Parameter | RandomForest | XGBoost | LightGBM | MLP |
|---|---|---|---|---|
| Trees / Layers | 100 trees | 200 trees | 200 estimators | 3 layers |
| Depth / Leaves | max\_depth=20 | max\_depth=6 | num\_leaves=31 | 128→64→2 |
| Learning rate | — | η=0.1 | lr=0.1 | lr=0.001 |
| Regularization | class\_weight=balanced | λ=1 (L2) | min\_child=20 | dropout=0.3 |
| Seed | 42 | 42 | 42 | 42 |

---

## Key Findings

**1. LightGBM is the optimal edge model across all three datasets**

LIME latency of 15.1–16.4 ms is consistently 2.1× below the 35 ms IoT gateway budget. Model files are 295–301 KB, well within the 128 MB memory constraint.

**2. SMOTE fidelity paradox confirmed (Gap 3)**

CIC-IoT-2023/LightGBM shows Spearman ρ = 0.669 (below the 0.8 distortion threshold) despite 99.56% accuracy. This means SMOTE improves accuracy but corrupts SHAP/LIME explanations for this specific combination. RandomForest on the same dataset retains ρ = 0.953, confirming the effect is model-specific.

**3. Cross-validation confirms generalizability**

All CV standard deviations are below 0.003. TON-IoT achieves ±0.0003, meaning results are nearly identical regardless of which fold is held out.

**4. Confidence-gated escalation works in practice**

Approximately 15–20% of live API packets trigger SHAP escalation (confidence < 0.85), predominantly corresponding to TON-IoT benign samples near decision boundaries.

---

## Software Stack

| Component | Version |
|---|---|
| Python | 3.11 |
| scikit-learn | 1.4.2 |
| XGBoost | 2.0.3 |
| LightGBM | 4.3.0 |
| SHAP | 0.45.0 |
| LIME | 0.2.0.1 |
| imbalanced-learn | 0.12.3 |
| FastAPI | 0.111.0 |
| Uvicorn | 0.29.0 |
| Pydantic | 2.7.1 |
| NumPy | 1.26.4 |
| SciPy | 1.13.0 |

**Training hardware:**
- GPU: NVIDIA Tesla T4 (16 GB VRAM, 8.1 TFLOPS FP32)
- CPU: Intel Xeon at 2.20 GHz (2 vCPU)
- RAM: 52 GB
- OS: Ubuntu 22.04 LTS

**Deployment hardware (Hugging Face Spaces free tier):**
- 2 vCPU
- 16 GB RAM
- Python 3.10-slim Docker container

---

## Troubleshooting

**The notebook says "No CSV files in /content/datasets/toniot"**

The Kaggle download for TON-IoT failed. The cell tries four slugs automatically. If all fail, go to kaggle.com/datasets, search "TON IoT network intrusion", download manually, and upload the CSV files to Colab using File → Upload. Target path: `/content/datasets/toniot/`.

---

**LIME weights are all 0.0 in the API response**

This happens with the old `main.py` that uses a zero-valued LIME background. The current `main.py` in this repository uses a Gaussian background (`rng.randn(50, n_features) * 0.5`) which fixes this. If you see zero weights, make sure you are using the latest version of `app/main.py` from this repository.

---

**The Hugging Face Space shows "Build failed"**

1. Click the **Logs** tab in your Space
2. Scroll to the first red line — this is the actual error
3. Most common causes:
   - A `.pkl` file is missing from the `models/` folder
   - A package version in `requirements.txt` is not available on PyPI (check spelling)
   - `libgomp1` failed to install — this is the OpenMP runtime needed by LightGBM. The `Dockerfile` installs it via `apt-get`. If you edited the Dockerfile, make sure this line is present: `RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 gcc`

---

**The /health endpoint shows a model as not loaded**

The model `.pkl` file is either missing or has the wrong filename. Expected filenames:

```
ciciot2023_lightgbm_model.pkl
ciciot2023_randomforest_model.pkl
ciciot2023_scaler.pkl
ciciot2023_features.json
toniot_lightgbm_model.pkl
toniot_randomforest_model.pkl
toniot_scaler.pkl
toniot_features.json
unswnb15_lightgbm_model.pkl
unswnb15_randomforest_model.pkl
unswnb15_scaler.pkl
unswnb15_features.json
```

All 12 files must be in the `models/` subfolder with these exact names (case-sensitive).

---

**Colab session disconnected mid-training**

Colab free tier disconnects after ~90 minutes of inactivity. Colab Pro disconnects after ~12 hours. If this happens:

1. Click **Runtime → Reconnect**
2. Re-run Cell 3 (imports and config)
3. Re-run Cell 13 (main pipeline) — it will restart from the beginning for each dataset

To avoid disconnects, keep the Colab tab active in your browser during training. Optionally, run the pipeline one dataset at a time by temporarily setting `PIPELINE_DATASETS` in Cell 13 to a single-element list.

---

**Prediction confidence is unexpectedly low on all inputs**

Your input features are probably not scaled. The API applies a `StandardScaler` internally, but the scaler was fitted on the training data. If you pass raw (unscaled) feature values, the scaler will produce numbers far outside the training distribution and the model will output low-confidence predictions near 0.5. Use the `/demo/{dataset}` endpoint to see what correctly-scaled values look like for each dataset.
