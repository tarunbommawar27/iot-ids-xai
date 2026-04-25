---
title: IoT IDS XAI API
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# IoT IDS XAI — Explainable Intrusion Detection

**Paper:** *Explainable Machine Learning for IoT Intrusion Detection*

| Dataset | LightGBM F1 | CV F1 (5-fold) | LIME latency |
|---|---|---|---|
| CIC-IoT-2023 | 0.9539 | 0.9518 ± 0.0009 | 15.8ms ✅ |
| TON-IoT | 0.9975 | 0.9978 ± 0.0003 | 16.4ms ✅ |
| UNSW-NB15 | 0.9644 | 0.9665 ± 0.0014 | 15.1ms ✅ |

## Endpoints
- `GET /` — Interactive dashboard
- `GET /docs` — Swagger UI
- `GET /health` — Model status
- `GET /demo/{dataset}` — Run a real pre-loaded test sample
- `POST /predict` — Custom prediction + LIME/SHAP
- `POST /batch_predict` — Batch inference
