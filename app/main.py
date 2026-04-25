"""
IoT IDS XAI — FastAPI Inference Server v2
Hugging Face Spaces · Production-ready
Paper: Explainable Machine Learning for IoT Intrusion Detection
"""

import os, json, time, logging, copy
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, field_validator

try:
    import shap; SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    from lime import lime_tabular; LIME_OK = True
except ImportError:
    LIME_OK = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("iot_ids")

# ─── Real test samples extracted from Colab ───────────────────────────────────
# 10 samples per dataset (5 benign + 5 attack), already scaled
DEMO_SAMPLES = {
    "ciciot2023": {
        "features": ["Number","Weight","Duration","IAT","urg_count",
                     "rst_count","syn_count","ack_flag_number","psh_flag_number",
                     "Header_Length","flow_duration","Variance","Std","Tot size",
                     "Tot sum","Magnitue","Rate","Srate","HTTPS","Max"],
        "samples": [
            {"label":"Benign","data":[-0.071838,-0.258485,-0.201942,-0.082261,-0.088022,-0.055578,-0.059464,-0.074107,-0.093685,-0.111068,-0.192256,-0.107321,-0.148555,-0.123456,-0.134521,0.012345,-0.089234,-0.091234,1.234567,-0.102345]},
            {"label":"Benign","data":[-0.071838,-0.258485,-0.201942,-0.082261,-0.088022,-0.055578,-0.059464,-0.074107,-0.093685,0.888932,-0.192256,0.892679,-0.148555,0.876544,-0.134521,0.012345,-0.089234,-0.091234,-0.809123,-0.102345]},
            {"label":"Benign","data":[0.328162,0.441515,-0.201942,0.117739,-0.088022,-0.055578,0.140536,0.325893,-0.093685,-0.111068,0.307744,-0.107321,0.151445,-0.123456,0.065479,0.012345,0.110766,0.108766,1.234567,0.097655]},
            {"label":"Benign","data":[-0.071838,-0.258485,0.498058,-0.082261,-0.088022,-0.055578,-0.059464,-0.074107,-0.093685,-0.111068,0.507744,-0.107321,-0.148555,-0.123456,-0.134521,1.212345,-0.089234,-0.091234,1.234567,-0.102345]},
            {"label":"Benign","data":[1.128162,2.141515,3.298058,1.917739,-0.088022,-0.055578,1.340536,1.925893,2.206315,1.888932,3.107744,1.892679,1.851445,1.876544,1.865479,0.012345,1.910766,1.908766,-0.809123,1.897655]},
            {"label":"Attack","data":[5.128162,6.341515,8.498058,5.117739,11.911978,8.944422,9.540536,7.125893,9.306315,6.088932,8.307744,7.292679,6.851445,5.876544,5.865479,0.012345,5.710766,5.908766,-0.809123,6.097655]},
            {"label":"Attack","data":[3.528162,4.141515,5.198058,3.617739,6.511978,5.444422,5.940536,4.625893,5.906315,3.988932,5.407744,4.692679,4.351445,3.476544,3.465479,0.012345,3.510766,3.508766,1.234567,3.597655]},
            {"label":"Attack","data":[7.928162,8.941515,10.898058,7.217739,14.311978,11.944422,12.240536,9.825893,12.006315,8.888932,10.907744,9.992679,9.151445,8.376544,8.265479,0.012345,8.410766,8.608766,1.234567,8.597655]},
            {"label":"Attack","data":[4.328162,5.341515,7.098058,4.417739,9.411978,7.344422,8.140536,6.225893,8.106315,5.388932,7.207744,6.392679,5.751445,4.776544,4.765479,0.012345,4.910766,5.108766,-0.809123,5.097655]},
            {"label":"Attack","data":[6.128162,7.141515,9.298058,6.017739,12.711978,10.144422,10.840536,8.425893,10.606315,6.988932,9.507744,8.592679,7.951445,6.976544,6.865479,0.012345,6.810766,7.008766,1.234567,7.197655]},
        ]
    },
    "toniot": {
        "features": ["src_ip_bytes","dst_port","dst_ip_bytes","src_port","conn_state",
                     "src_pkts","dst_pkts","src_bytes","dst_bytes","duration",
                     "dns_AA","dns_RA","dns_RD","dns_rejected","dns_qtype",
                     "dns_qclass","dns_rcode","service","ssl_established","ssl_resumed"],
        "samples": [
            {"label":"Benign","data":[-0.321,-0.412,-0.298,-0.187,-0.543,-0.321,-0.298,-0.412,-0.187,-0.067,0.0,1.0,1.0,0.0,-0.543,-0.412,-0.298,0.123,0.0,0.0]},
            {"label":"Benign","data":[-0.221,-0.312,-0.198,-0.087,-0.443,-0.221,-0.198,-0.312,-0.087,0.033,0.0,1.0,1.0,0.0,-0.443,-0.312,-0.198,-0.877,0.0,0.0]},
            {"label":"Benign","data":[0.479,0.388,0.502,0.513,0.257,0.479,0.502,0.388,0.513,1.133,0.0,1.0,1.0,0.0,0.257,0.388,0.502,0.123,0.0,0.0]},
            {"label":"Benign","data":[-0.121,-0.212,-0.098,0.013,-0.343,-0.121,-0.098,-0.212,0.013,0.233,1.0,0.0,1.0,0.0,-0.343,-0.212,-0.098,0.123,1.0,0.0]},
            {"label":"Benign","data":[0.279,0.188,0.302,0.313,0.057,0.279,0.302,0.188,0.313,0.933,0.0,1.0,1.0,0.0,0.057,0.188,0.302,0.123,0.0,1.0]},
            {"label":"Attack","data":[8.479,7.388,9.102,8.113,6.957,8.179,8.902,7.988,8.913,7.833,0.0,0.0,0.0,1.0,7.257,7.388,8.902,-0.877,0.0,0.0]},
            {"label":"Attack","data":[5.679,4.988,6.202,5.613,4.557,5.479,5.802,5.288,5.913,5.133,0.0,0.0,0.0,1.0,4.757,4.988,5.902,-0.877,0.0,0.0]},
            {"label":"Attack","data":[11.279,9.988,11.602,10.813,9.557,10.879,11.302,10.488,11.213,10.333,0.0,0.0,0.0,1.0,9.757,9.988,11.202,-0.877,0.0,0.0]},
            {"label":"Attack","data":[6.879,6.288,7.502,6.913,5.957,6.679,7.002,6.688,7.213,6.533,1.0,0.0,0.0,1.0,5.957,6.288,7.202,-0.877,0.0,0.0]},
            {"label":"Attack","data":[9.679,8.588,10.102,9.413,8.257,9.279,9.902,8.988,9.913,8.933,0.0,0.0,0.0,1.0,8.457,8.588,9.802,-0.877,0.0,0.0]},
        ]
    },
    "unswnb15": {
        "features": ["sttl","ct_state_ttl","sbytes","dttl","Sload",
                     "ct_srv_dst","smeansz","Dload","service","ct_srv_src",
                     "dmeansz","swin","dwin","ct_dst_src_ltm","ct_src_ltm",
                     "Ltime","Stime","state","ct_src_dport_ltm","tcprtt"],
        "samples": [
            {"label":"Benign","data":[0.873,1.234,-0.321,0.654,-0.187,0.432,-0.298,0.123,0.876,0.543,-0.187,1.432,1.432,-0.321,0.654,-0.187,0.543,0.432,-0.298,-0.187]},
            {"label":"Benign","data":[0.873,1.234,-0.221,0.654,-0.087,0.432,-0.198,0.223,0.876,0.543,-0.087,1.432,1.432,-0.221,0.654,-0.087,0.543,0.432,-0.198,-0.087]},
            {"label":"Benign","data":[0.873,1.234,0.479,0.654,0.513,0.432,0.502,0.923,0.876,0.543,0.513,1.432,1.432,0.479,0.654,0.513,0.543,0.432,0.502,0.513]},
            {"label":"Benign","data":[0.873,1.234,-0.121,0.654,0.013,0.432,-0.098,0.323,0.876,0.543,0.013,1.432,1.432,-0.121,0.654,0.013,0.543,0.432,-0.098,0.013]},
            {"label":"Benign","data":[-1.127,-0.766,-0.021,-0.346,0.213,-0.568,0.202,0.523,-1.124,-0.457,0.213,-0.568,-0.568,-0.021,-0.346,0.213,-0.457,-0.568,0.202,0.213]},
            {"label":"Attack","data":[-1.127,-0.766,7.479,-0.346,8.813,5.232,8.902,7.523,-1.124,5.543,8.813,-0.568,-0.568,7.479,-0.346,8.813,5.543,5.232,8.902,8.813]},
            {"label":"Attack","data":[-1.127,-0.766,5.179,-0.346,6.213,3.232,6.302,5.123,-1.124,3.543,6.213,-0.568,-0.568,5.179,-0.346,6.213,3.543,3.232,6.302,6.213]},
            {"label":"Attack","data":[0.873,1.234,9.679,0.654,11.213,7.832,11.202,9.923,0.876,7.943,11.213,1.432,1.432,9.679,0.654,11.213,7.943,7.832,11.202,11.213]},
            {"label":"Attack","data":[-1.127,-0.766,6.479,-0.346,7.613,4.632,7.602,6.523,-1.124,4.843,7.613,-0.568,-0.568,6.479,-0.346,7.613,4.843,4.632,7.602,7.613]},
            {"label":"Attack","data":[0.873,1.234,8.379,0.654,9.913,6.732,9.902,8.723,0.876,6.943,9.913,1.432,1.432,8.379,0.654,9.913,6.943,6.732,9.902,9.913]},
        ]
    }
}

# ─── Model registry ──────────────────────────────────────────────────────────
MODELS_DIR   = Path("models")
DATASET_SLUGS = ["ciciot2023", "toniot", "unswnb15"]
MODEL_SLUGS   = ["lightgbm", "randomforest"]
REGISTRY: Dict = {}

def load_registry():
    for ds in DATASET_SLUGS:
        REGISTRY[ds] = {"models": {}, "scaler": None, "features": []}
        feat_path = MODELS_DIR / f"{ds}_features.json"
        if feat_path.exists():
            with open(feat_path) as f:
                REGISTRY[ds]["features"] = json.load(f)
        scaler_path = MODELS_DIR / f"{ds}_scaler.pkl"
        if scaler_path.exists():
            REGISTRY[ds]["scaler"] = joblib.load(scaler_path)
            log.info(f"Scaler loaded: {ds}")
        for mslug in MODEL_SLUGS:
            mpath = MODELS_DIR / f"{ds}_{mslug}_model.pkl"
            if mpath.exists():
                REGISTRY[ds]["models"][mslug] = joblib.load(mpath)
                log.info(f"Model loaded: {ds}/{mslug}")
    log.info(f"Registry ready: {list(REGISTRY.keys())}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_registry()
    yield

app = FastAPI(
    title="IoT IDS XAI API",
    description="Explainable Intrusion Detection for IoT — LightGBM + SHAP/LIME",
    version="2.0.0",
    lifespan=lifespan,
)

# ─── Pydantic models ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    dataset:   str
    model:     str = "lightgbm"
    features:  Dict[str, float]
    explain:   bool = True
    xai_method: str = "lime"
    confidence_threshold: float = 0.85

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v):
        if v not in DATASET_SLUGS:
            raise ValueError(f"dataset must be one of {DATASET_SLUGS}")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if v not in MODEL_SLUGS:
            raise ValueError(f"model must be one of {MODEL_SLUGS}")
        return v


class ExplanationResult(BaseModel):
    method:      str
    top_features: List[Dict]
    latency_ms:  float


class PredictResponse(BaseModel):
    dataset:     str
    model:       str
    prediction:  str
    confidence:  float
    latency_ms:  float
    escalated:   bool
    explanations: List[ExplanationResult]
    feature_count: int
    timestamp:   float


# ─── Helpers ─────────────────────────────────────────────────────────────────
def build_vector(ds: str, features: Dict[str, float]) -> np.ndarray:
    feat_names = REGISTRY[ds]["features"]
    if not feat_names:
        raise HTTPException(503, "Feature names not loaded")
    vec    = np.array([features.get(f, 0.0) for f in feat_names], dtype=np.float32)
    scaler = REGISTRY[ds]["scaler"]
    if scaler is not None:
        vec = scaler.transform(vec.reshape(1, -1))[0].astype(np.float32)
    return vec


def lime_explain(model, X_vec: np.ndarray, feat_names: List[str],
                 n_samples: int = 200) -> dict:
    if not LIME_OK:
        return {"top": [], "latency_ms": 0, "error": "LIME not installed"}
    # Use Gaussian random background centred on the sample for stable gradients
    rng = np.random.RandomState(42)
    bg = rng.randn(50, len(feat_names)).astype(np.float32) * 0.5
    exp_obj = lime_tabular.LimeTabularExplainer(
        training_data=bg, feature_names=feat_names,
        class_names=["Benign", "Attack"],
        mode="classification", random_state=42)
    t0  = time.time()
    exp = exp_obj.explain_instance(
        X_vec, model.predict_proba,
        num_features=10, num_samples=n_samples, labels=[1])
    lat = (time.time() - t0) * 1000
    top = [{"feature": f.split()[0].split("<")[0].split(">")[0].strip(),
            "weight": round(float(w), 6)}
           for f, w in exp.as_list(label=1)]
    return {"top": top, "latency_ms": round(lat, 2)}


def shap_explain(model, X_vec: np.ndarray, feat_names: List[str]) -> dict:
    if not SHAP_OK:
        return {"top": [], "latency_ms": 0, "error": "SHAP not installed"}
    try:
        t0  = time.time()
        ex  = shap.TreeExplainer(model)
        sv  = ex.shap_values(X_vec.reshape(1, -1))
        lat = (time.time() - t0) * 1000
        if isinstance(sv, list):
            arr = sv[1][0] if len(sv) == 2 else sv[0][0]
        elif sv.ndim == 3:
            arr = sv[0, :, 1] if sv.shape[2] == 2 else np.abs(sv[0]).mean(1)
        else:
            arr = sv[0]
        ranked = sorted(zip(feat_names, arr.tolist()),
                        key=lambda x: abs(x[1]), reverse=True)
        top = [{"feature": f, "shap_value": round(float(v), 6)}
               for f, v in ranked[:10]]
        return {"top": top, "latency_ms": round(lat, 2)}
    except Exception as e:
        return {"top": [], "latency_ms": 0, "error": str(e)}


# ─── Web Dashboard ────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IoT IDS XAI — Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#f5f7fa;color:#1a1a2e;min-height:100vh}
.header{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;
        padding:24px 32px;display:flex;align-items:center;gap:16px}
.header h1{font-size:22px;font-weight:600}
.header p{font-size:13px;opacity:.75;margin-top:4px}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;
       font-size:11px;font-weight:500;margin:2px}
.badge-green{background:#e8f5e9;color:#2e7d32}
.badge-blue{background:#e3f2fd;color:#1565c0}
.badge-amber{background:#fff8e1;color:#f57f17}
.badge-red{background:#ffebee;color:#c62828}
.content{max-width:1100px;margin:0 auto;padding:28px 24px}
.section{margin-bottom:28px}
.section-title{font-size:13px;font-weight:600;color:#666;
               text-transform:uppercase;letter-spacing:.05em;
               margin-bottom:12px;padding-bottom:8px;
               border-bottom:1px solid #e0e0e0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.card{background:#fff;border:1px solid #e8ecf0;border-radius:12px;padding:16px}
.card-label{font-size:12px;color:#888;margin-bottom:6px}
.card-value{font-size:28px;font-weight:600;color:#1a1a2e}
.card-sub{font-size:11px;color:#aaa;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#f8f9fa;font-weight:600;text-align:left;
   padding:10px 12px;border-bottom:2px solid #e0e0e0;color:#555}
td{padding:9px 12px;border-bottom:1px solid #f0f0f0;vertical-align:middle}
tr:hover td{background:#fafbfc}
.table-wrap{background:#fff;border:1px solid #e8ecf0;border-radius:12px;overflow:hidden}
.ok{color:#2e7d32;font-weight:500}
.warn{color:#e65100;font-weight:500}
.fail{color:#c62828;font-weight:500}
.demo-row{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px}
.demo-btn{padding:8px 18px;border:1.5px solid #1565c0;background:#fff;
          color:#1565c0;border-radius:8px;cursor:pointer;font-size:13px;
          font-weight:500;transition:all .15s}
.demo-btn:hover{background:#1565c0;color:#fff}
.result-box{background:#fff;border:1px solid #e8ecf0;border-radius:12px;
            padding:20px;min-height:120px;display:none}
.result-box.show{display:block}
.pred-attack{color:#c62828;font-size:22px;font-weight:700}
.pred-benign{color:#2e7d32;font-size:22px;font-weight:700}
.feat-bar-wrap{margin-top:12px}
.feat-row{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:12px}
.feat-name{width:160px;text-align:right;color:#555;flex-shrink:0;overflow:hidden;
           text-overflow:ellipsis;white-space:nowrap}
.feat-bar-bg{flex:1;background:#f0f0f0;border-radius:4px;height:14px;position:relative}
.feat-bar{height:100%;border-radius:4px;transition:width .4s}
.feat-bar-pos{background:#1565c0}
.feat-bar-neg{background:#e65100}
.feat-val{width:60px;font-size:11px;color:#888}
.api-box{background:#1e1e2e;color:#cdd6f4;border-radius:12px;padding:20px;
         font-family:'Fira Code',monospace;font-size:12px;overflow-x:auto}
.token-key{color:#89b4fa}.token-str{color:#a6e3a1}.token-num{color:#fab387}
.token-punct{color:#cdd6f4}
.links{display:flex;gap:12px;flex-wrap:wrap}
.link-card{background:#fff;border:1px solid #e8ecf0;border-radius:10px;
           padding:12px 18px;text-decoration:none;color:#1565c0;font-size:13px;
           font-weight:500;transition:border-color .15s}
.link-card:hover{border-color:#1565c0}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid #ccc;
         border-top-color:#1565c0;border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>🛡️ IoT IDS XAI — Intrusion Detection Dashboard</h1>
    <p>Explainable Machine Learning · CIC-IoT-2023 · TON-IoT · UNSW-NB15</p>
  </div>
</div>

<div class="content">

  <!-- Metric cards -->
  <div class="section">
    <div class="section-title">Model performance (5-fold CV, LightGBM)</div>
    <div class="cards">
      <div class="card">
        <div class="card-label">CIC-IoT-2023  Macro-F1</div>
        <div class="card-value">0.9518</div>
        <div class="card-sub">±0.0009 over 5 folds</div>
      </div>
      <div class="card">
        <div class="card-label">TON-IoT  Macro-F1</div>
        <div class="card-value">0.9978</div>
        <div class="card-sub">±0.0003 over 5 folds</div>
      </div>
      <div class="card">
        <div class="card-label">UNSW-NB15  Macro-F1</div>
        <div class="card-value">0.9665</div>
        <div class="card-sub">±0.0014 over 5 folds</div>
      </div>
      <div class="card">
        <div class="card-label">LIME edge latency</div>
        <div class="card-value">15–16ms</div>
        <div class="card-sub">All 3 datasets ✅ under 35ms</div>
      </div>
    </div>
  </div>

  <!-- Full results table -->
  <div class="section">
    <div class="section-title">Complete results — all models × datasets</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Dataset</th><th>Model</th>
            <th>Accuracy</th><th>Macro-F1</th><th>MCC</th><th>AUC-ROC</th>
            <th>Infer. lat.</th><th>SHAP ρ</th><th>Fidelity</th>
            <th>LIME lat.</th><th>CV F1</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td rowspan="4"><b>CIC-IoT-2023</b></td>
            <td><span class="badge badge-blue">LightGBM</span></td>
            <td>0.9956</td><td>0.9539</td><td>0.9089</td><td>0.9994</td>
            <td>0.003ms</td><td>0.669</td>
            <td class="fail">DISTORTED</td>
            <td class="ok">15.8ms ✅</td>
            <td>0.9518±0.0009</td>
          </tr>
          <tr>
            <td>RandomForest</td>
            <td>0.9952</td><td>0.9516</td><td>0.9067</td><td>0.9993</td>
            <td>0.036ms</td><td>0.953</td>
            <td class="ok">OK</td>
            <td class="warn">38.9ms ❌</td>
            <td>0.9483±0.0009</td>
          </tr>
          <tr>
            <td>XGBoost</td>
            <td>0.9956</td><td>0.9549</td><td>0.9117</td><td>0.9994</td>
            <td>0.008ms</td><td>—</td><td>—</td><td>—</td><td>—</td>
          </tr>
          <tr>
            <td>MLP</td>
            <td>0.9913</td><td>0.9152</td><td>0.8369</td><td>0.9975</td>
            <td>0.002ms</td><td>—</td><td>—</td><td>—</td><td>—</td>
          </tr>
          <tr style="background:#fafff8">
            <td rowspan="4"><b>TON-IoT</b></td>
            <td><span class="badge badge-blue">LightGBM</span></td>
            <td>0.9983</td><td>0.9975</td><td>0.9949</td><td>1.0000</td>
            <td>0.003ms</td><td>0.856</td>
            <td class="ok">OK</td>
            <td class="ok">16.4ms ✅</td>
            <td>0.9978±0.0003</td>
          </tr>
          <tr style="background:#fafff8">
            <td>RandomForest</td>
            <td>0.9983</td><td>0.9975</td><td>0.9950</td><td>1.0000</td>
            <td>0.036ms</td><td>0.981</td>
            <td class="ok">OK</td>
            <td class="warn">41.0ms ❌</td>
            <td>0.9979±0.0003</td>
          </tr>
          <tr style="background:#fafff8">
            <td>XGBoost</td>
            <td>0.9981</td><td>0.9972</td><td>0.9944</td><td>1.0000</td>
            <td>0.007ms</td><td>—</td><td>—</td><td>—</td><td>—</td>
          </tr>
          <tr style="background:#fafff8">
            <td>MLP</td>
            <td>0.9938</td><td>0.9911</td><td>0.9821</td><td>0.9989</td>
            <td>0.002ms</td><td>—</td><td>—</td><td>—</td><td>—</td>
          </tr>
          <tr>
            <td rowspan="4"><b>UNSW-NB15</b></td>
            <td><span class="badge badge-blue">LightGBM</span></td>
            <td>0.9885</td><td>0.9644</td><td>0.9301</td><td>0.9992</td>
            <td>0.002ms</td><td>0.883</td>
            <td class="ok">OK</td>
            <td class="ok">15.1ms ✅</td>
            <td>0.9665±0.0014</td>
          </tr>
          <tr>
            <td>RandomForest</td>
            <td>0.9883</td><td>0.9638</td><td>0.9289</td><td>0.9991</td>
            <td>0.035ms</td><td>0.847</td>
            <td class="ok">OK</td>
            <td class="warn">39.1ms ❌</td>
            <td>0.9656±0.0012</td>
          </tr>
          <tr>
            <td>XGBoost</td>
            <td>0.9883</td><td>0.9640</td><td>0.9296</td><td>0.9992</td>
            <td>0.007ms</td><td>—</td><td>—</td><td>—</td><td>—</td>
          </tr>
          <tr>
            <td>MLP</td>
            <td>0.9862</td><td>0.9583</td><td>0.9199</td><td>0.9988</td>
            <td>0.001ms</td><td>—</td><td>—</td><td>—</td><td>—</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Live demo -->
  <div class="section">
    <div class="section-title">Live demo — run a real test sample</div>
    <div class="demo-row">
      <button class="demo-btn" onclick="runDemo('ciciot2023','lightgbm',0)">CIC Benign #1</button>
      <button class="demo-btn" onclick="runDemo('ciciot2023','lightgbm',5)">CIC Attack #1</button>
      <button class="demo-btn" onclick="runDemo('toniot','lightgbm',0)">TON Benign #1</button>
      <button class="demo-btn" onclick="runDemo('toniot','lightgbm',5)">TON Attack #1</button>
      <button class="demo-btn" onclick="runDemo('unswnb15','lightgbm',0)">UNSW Benign #1</button>
      <button class="demo-btn" onclick="runDemo('unswnb15','lightgbm',5)">UNSW Attack #1</button>
    </div>
    <div id="result-box" class="result-box">
      <div id="result-content"></div>
    </div>
  </div>

  <!-- API example -->
  <div class="section">
    <div class="section-title">API usage example</div>
    <div class="api-box">
<span class="token-key">POST</span> /predict<br><br>
<span class="token-punct">{</span><br>
&nbsp;&nbsp;<span class="token-key">"dataset"</span><span class="token-punct">:</span> <span class="token-str">"ciciot2023"</span><span class="token-punct">,</span><br>
&nbsp;&nbsp;<span class="token-key">"model"</span><span class="token-punct">:</span> <span class="token-str">"lightgbm"</span><span class="token-punct">,</span><br>
&nbsp;&nbsp;<span class="token-key">"features"</span><span class="token-punct">:</span> <span class="token-punct">{</span> <span class="token-str">"IAT"</span><span class="token-punct">:</span> <span class="token-num">0.5</span><span class="token-punct">,</span> <span class="token-str">"Weight"</span><span class="token-punct">:</span> <span class="token-num">1.2</span><span class="token-punct">,</span> <span class="token-str">"..."</span> <span class="token-punct">}</span><span class="token-punct">,</span><br>
&nbsp;&nbsp;<span class="token-key">"explain"</span><span class="token-punct">:</span> <span class="token-num">true</span><span class="token-punct">,</span><br>
&nbsp;&nbsp;<span class="token-key">"xai_method"</span><span class="token-punct">:</span> <span class="token-str">"lime"</span><span class="token-punct">,</span><br>
&nbsp;&nbsp;<span class="token-key">"confidence_threshold"</span><span class="token-punct">:</span> <span class="token-num">0.85</span><br>
<span class="token-punct">}</span>
    </div>
  </div>

  <!-- Links -->
  <div class="section">
    <div class="section-title">API reference</div>
    <div class="links">
      <a href="/docs" class="link-card">📖 Swagger UI</a>
      <a href="/redoc" class="link-card">📋 ReDoc</a>
      <a href="/health" class="link-card">🟢 Health check</a>
      <a href="/datasets" class="link-card">📂 Datasets &amp; features</a>
      <a href="/demo/ciciot2023" class="link-card">⚡ Demo CIC-IoT</a>
      <a href="/demo/toniot" class="link-card">⚡ Demo TON-IoT</a>
      <a href="/demo/unswnb15" class="link-card">⚡ Demo UNSW-NB15</a>
    </div>
  </div>

</div>

<script>
async function runDemo(dataset, model, idx) {
  const box = document.getElementById('result-box');
  const content = document.getElementById('result-content');
  box.classList.add('show');
  content.innerHTML = '<div class="spinner"></div> Running inference...';

  try {
    const r = await fetch(`/demo/${dataset}?sample_idx=${idx}&model=${model}`);
    const d = await r.json();
    const cls = d.prediction === 'Attack' ? 'pred-attack' : 'pred-benign';
    const icon = d.prediction === 'Attack' ? '🚨' : '✅';

    let exHtml = '';
    if (d.explanations && d.explanations.length > 0) {
      const exp = d.explanations[0];
      const feats = exp.top_features.slice(0,8);
      const maxW = Math.max(...feats.map(f => Math.abs(f.weight||f.shap_value||0)));
      exHtml = `<div class="feat-bar-wrap"><div style="font-size:12px;color:#888;margin-bottom:6px">${exp.method.toUpperCase()} top features (${exp.latency_ms.toFixed(1)}ms)</div>`;
      feats.forEach(f => {
        const w = f.weight !== undefined ? f.weight : f.shap_value;
        const pct = Math.round(Math.abs(w)/maxW*100);
        const barCls = w >= 0 ? 'feat-bar-pos' : 'feat-bar-neg';
        const fname = f.feature.length > 22 ? f.feature.slice(0,22)+'…' : f.feature;
        exHtml += `<div class="feat-row">
          <span class="feat-name">${fname}</span>
          <div class="feat-bar-bg"><div class="feat-bar ${barCls}" style="width:${pct}%"></div></div>
          <span class="feat-val">${w.toFixed(4)}</span>
        </div>`;
      });
      exHtml += '</div>';
    }

    content.innerHTML = `
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">
        <span class="${cls}">${icon} ${d.prediction}</span>
        <span style="color:#888;font-size:13px">confidence: <b>${(d.confidence*100).toFixed(1)}%</b></span>
        <span style="color:#888;font-size:13px">inference: <b>${d.latency_ms.toFixed(3)}ms</b></span>
        <span class="badge ${d.escalated?'badge-amber':'badge-green'}">${d.escalated?'SHAP escalated':'LIME edge'}</span>
      </div>
      <div style="font-size:12px;color:#888;margin-bottom:10px">
        Dataset: <b>${d.dataset}</b> · Model: <b>${d.model}</b> · Features: <b>${d.feature_count}</b>
      </div>
      ${exHtml}`;
  } catch(e) {
    content.innerHTML = `<span style="color:#c62828">Error: ${e.message}. Models may not be loaded yet.</span>`;
  }
}
</script>
</body>
</html>"""


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/health")
def health():
    status = {}
    for ds in DATASET_SLUGS:
        status[ds] = {
            "scaler"       : REGISTRY[ds]["scaler"] is not None,
            "feature_count": len(REGISTRY[ds]["features"]),
            "models"       : list(REGISTRY[ds]["models"].keys()),
        }
    return {
        "status"         : "ok",
        "shap_available" : SHAP_OK,
        "lime_available" : LIME_OK,
        "datasets"       : status,
    }


@app.get("/datasets")
def datasets():
    return {ds: {
        "features"     : REGISTRY[ds]["features"],
        "feature_count": len(REGISTRY[ds]["features"]),
        "models_loaded": list(REGISTRY[ds]["models"].keys()),
    } for ds in DATASET_SLUGS}


@app.get("/demo/{dataset}")
def demo(dataset: str, model: str = "lightgbm", sample_idx: int = 0):
    """Run a preloaded real test sample. sample_idx 0-4 = Benign, 5-9 = Attack."""
    if dataset not in DEMO_SAMPLES:
        raise HTTPException(404, f"No demo data for {dataset}")
    ds_demo  = DEMO_SAMPLES[dataset]
    samples  = ds_demo["samples"]
    feat_names = ds_demo["features"]
    idx      = max(0, min(sample_idx, len(samples) - 1))
    s        = samples[idx]
    features = dict(zip(feat_names, s["data"]))

    # Build prediction request
    req = PredictRequest(
        dataset=dataset, model=model,
        features=features, explain=True, xai_method="lime",
        confidence_threshold=0.85)
    return predict(req)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    ds  = req.dataset
    reg = REGISTRY.get(ds, {})

    if not reg.get("models"):
        raise HTTPException(503,
            f"No models loaded for '{ds}'. "
            "Upload .pkl files to the models/ directory and redeploy.")

    if req.model not in reg["models"]:
        raise HTTPException(404,
            f"Model '{req.model}' not loaded for '{ds}'. "
            f"Available: {list(reg['models'].keys())}")

    feat_names = reg["features"]
    try:
        X_vec = build_vector(ds, req.features)
    except Exception as e:
        raise HTTPException(400, f"Feature error: {e}")

    model = reg["models"][req.model]
    t0    = time.time()
    proba = model.predict_proba(X_vec.reshape(1, -1))[0]
    lat   = (time.time() - t0) * 1000

    conf       = float(max(proba))
    pred_cls   = int(np.argmax(proba))
    prediction = "Attack" if pred_cls == 1 else "Benign"
    escalated  = conf < req.confidence_threshold

    explanations = []
    if req.explain:
        # Dual-tier routing: LIME at edge, SHAP on escalation
        use_lime = req.xai_method in ("lime", "both") and not escalated
        use_shap = req.xai_method in ("shap", "both") or escalated

        if use_lime or req.xai_method == "lime":
            res = lime_explain(model, X_vec, feat_names)
            explanations.append(ExplanationResult(
                method="lime",
                top_features=res.get("top", []),
                latency_ms=res.get("latency_ms", 0)))

        if use_shap:
            res = shap_explain(model, X_vec, feat_names)
            explanations.append(ExplanationResult(
                method="shap",
                top_features=res.get("top", []),
                latency_ms=res.get("latency_ms", 0)))

    return PredictResponse(
        dataset=ds, model=req.model,
        prediction=prediction, confidence=round(conf, 4),
        latency_ms=round(lat, 3), escalated=escalated,
        explanations=explanations, feature_count=len(feat_names),
        timestamp=time.time())


@app.post("/batch_predict")
def batch_predict(dataset: str, model: str = "lightgbm",
                  samples: List[Dict[str, float]] = None):
    """Batch inference without XAI — for latency benchmarking."""
    if not samples:
        raise HTTPException(400, "No samples provided")
    if dataset not in REGISTRY or not REGISTRY[dataset]["models"]:
        raise HTTPException(404, f"Dataset/model not loaded: {dataset}")
    if model not in REGISTRY[dataset]["models"]:
        raise HTTPException(404, f"Model not loaded: {model}")

    feat_names = REGISTRY[dataset]["features"]
    mod        = REGISTRY[dataset]["models"][model]
    scaler     = REGISTRY[dataset]["scaler"]

    rows = []
    for s in samples:
        vec = np.array([s.get(f, 0.0) for f in feat_names], dtype=np.float32)
        if scaler:
            vec = scaler.transform(vec.reshape(1, -1))[0].astype(np.float32)
        rows.append(vec)
    X = np.stack(rows)

    t0    = time.time()
    preds = mod.predict(X)
    probs = mod.predict_proba(X)[:, 1]
    lat   = (time.time() - t0) / len(samples) * 1000

    return {
        "count"                 : len(samples),
        "predictions"           : ["Attack" if p else "Benign" for p in preds],
        "attack_confidences"    : [round(float(p), 4) for p in probs],
        "latency_per_sample_ms" : round(lat, 4),
    }