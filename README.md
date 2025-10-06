This Project was done for NASA's 2025 Space Apps Challenge!

# Mission Control for Exoplanets 🚀
*A reproducible, cloud-hosted ML dashboard for classifying exoplanet candidates from K2/TESS/Kepler (or your own data).*

> From starlight to signal in five clicks — dataset → features → tuning → evaluation → artifact.

---

## Table of Contents
- [What it does](#what-it-does)
- [Key features](#key-features)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Quickstart (local)](#quickstart-local)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Deploy on Google Cloud Run](#deploy-on-google-cloud-run)
- [Model artifacts & downloads](#model-artifacts--downloads)
- [Make predictions outside the app](#make-predictions-outside-the-app)
- [Security notes](#security-notes)
- [Roadmap](#roadmap)
- [Citation / Acknowledgments](#citation--acknowledgments)

---

## What it does
This Flask web app guides researchers through an end-to-end, **reproducible** ML workflow for exoplanet candidate classification:

1. **Choose a dataset** — K2, TESS, Kepler, or upload CSV/Parquet.  
2. **Pick target & features** — live search, “Select/Unselect visible,” target auto-excluded from features.  
3. **Tune** — schema-driven hyperparameters with plain-English help; optional **L1/L2 Regularization**.  
4. **Evaluate** — optional stratified train/test split with **Accuracy, Precision, Recall, F1, ROC-AUC** and a **confusion matrix**.  
5. **Train all & ship** — train on all data and **download a model bundle (.zip)** containing a portable **joblib** artifact, metadata, and a README.  
6. **Predict** — in-app single-row form or batch CSV with a downloadable template.

---

## Key features
- **Reproducible by design** — one scikit-learn **Pipeline** (impute → encode → optional L1/L2 → model).  
- **Multiple model families** — Random Forest, Gradient Boosting, AdaBoost, MLP (Neural Net), Logistic Regression.  
- **Friendly UX** — space-themed yet professional UI; loading overlay with status lines; model summaries & pros/cons.  
- **Downloadable bundle** — one click exports `model.joblib + metadata.json + README.txt`.  
- **Cloud-ready** — stateless app; datasets/artifacts in Google Cloud Storage; easy to run on Cloud Run.

---

## Screenshots
```
docs/01-index.png
docs/02-columns.png
docs/03-tune.png
docs/04-split.png
docs/05-metrics.png
docs/06-final.png
```

---

## Architecture
- **App**: Flask + Jinja2 serve a 5-step wizard.
- **Data I/O**: pandas (CSV/Parquet).
- **ML**: scikit-learn Pipeline  
  - Numeric: `SimpleImputer(strategy="median")` (+ `StandardScaler` for some models)  
  - Categorical: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`  
  - Optional: `SelectFromModel(LogisticRegression(penalty=l1/l2, solver="saga"))`  
  - Estimator: RF / GB / AdaBoost / MLP / Logistic Regression
- **Artifacts**: joblib (`pipe + features + feature_types + metadata/versions`).  
- **Storage**: local filesystem and/or Google Cloud Storage.  
- **Hosting**: containerized; runs well on Cloud Run.

---

## Tech stack
**Languages**: Python, HTML/CSS, vanilla JS  
**Libraries**: Flask, Jinja2, pandas, scikit-learn, joblib, google-cloud-storage (optional)  
**Cloud**: Cloud Run, Cloud Storage, Secret Manager (recommended)

---

## Project structure
```
.
├─ app.py
├─ requirements.txt
├─ templates/
│  ├─ index.html      # dataset + model
│  ├─ columns.html    # target + features (search + select/unselect visible)
│  ├─ tune.html       # hyperparameters + L1/L2 selector
│  ├─ split.html      # optional stratified split
│  ├─ metrics.html    # metrics + confusion matrix + Train-all
│  └─ final.html      # summary + predict (single/csv) + download bundle
├─ static/
│  ├─ fun_background.svg
│  ├─ loader.gif
├─ docs/              # screenshots
└─ README.md
```

---

## Quickstart (local)

### 1) Create env & install deps
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` (example):
```
Flask>=2.3
pandas>=2.0
scikit-learn>=1.3
joblib>=1.3
google-cloud-storage>=2.13   # optional if using GCS
pyarrow>=15                  # for Parquet (optional but recommended)
python-dotenv>=1.0           # optional, for local env loading
```

### 2) Configure environment
Create `.env` or export in your shell:
```
FLASK_SECRET_KEY=change_me_in_production
MAX_UPLOAD_MB=50
UPLOAD_DIR=/tmp/uploads
MODEL_DIR=/tmp/models

# Optional built-in datasets (GCS paths or local file paths)
K2_URI=gs://my-bucket/data/k2.csv
TESS_URI=gs://my-bucket/data/tess.csv
KEPLER_URI=gs://my-bucket/data/kepler.csv

# Optional GCS integration for uploads/results
GCS_BUCKET=my-bucket-name                # uploads/
RESULTS_BUCKET=gs://my-bucket/results   # models/ artifacts
```

> Using GCS locally? Authenticate first: `gcloud auth application-default login`

### 3) Run
```bash
export FLASK_APP=app.py
python app.py
# visit http://127.0.0.1:5000
```

---

## Configuration
- **Secrets**: `FLASK_SECRET_KEY` (use Secret Manager in production).  
- **Built-in dataset URIs**: `K2_URI`, `TESS_URI`, `KEPLER_URI` (support `gs://` or local files).  
- **Uploads**: `UPLOAD_DIR` locally or `GCS_BUCKET` (saved under `uploads/`).  
- **Artifacts**: `MODEL_DIR` locally; uploaded to `RESULTS_BUCKET` if set.  
- **Max file size**: `MAX_UPLOAD_MB` (default 50 MB).

---

## Datasets
Works with:
- **K2 / TESS / Kepler** via env URIs (CSV/Parquet).  
- **Other** (upload your own) in CSV/Parquet.  
Categoricals are one-hot encoded with **unknowns ignored** at inference. Extra columns are safely dropped by the `ColumnTransformer`.

---

## Deploy on Google Cloud Run

**Prereqs**: GCP project, `gcloud` CLI, Container/Cloud Run enabled, and a GCS bucket.

### 1) Build & push container
```bash
gcloud builds submit --tag gcr.io/$(gcloud config get-value project)/exoplanet-app
```

### 2) Deploy
```bash
gcloud run deploy exoplanet-app   --image gcr.io/$(gcloud config get-value project)/exoplanet-app   --region us-central1   --allow-unauthenticated   --set-env-vars FLASK_SECRET_KEY=prod_secret,MAX_UPLOAD_MB=50,UPLOAD_DIR=/tmp/uploads,MODEL_DIR=/tmp/models   --set-env-vars RESULTS_BUCKET=gs://my-bucket/results,GCS_BUCKET=my-bucket-name   --set-env-vars K2_URI=gs://my-bucket/data/k2.csv,TESS_URI=gs://my-bucket/data/tess.csv,KEPLER_URI=gs://my-bucket/data/kepler.csv
```

### 3) Permissions
- The Cloud Run service account needs **Storage Object Admin** (or granular) on the buckets you use.  
- Use **Secret Manager** for secrets.


---

## Model artifacts & downloads
After training on all data, the app saves a compressed **joblib** bundle and exposes **Download bundle (.zip)**, which contains:

- `model_<uuid>.joblib` — full scikit-learn **Pipeline** (preprocessing + model)  
- `metadata.json` — features, params, created timestamp, and library versions  
- `README.txt` — quick usage instructions

Artifacts are saved locally and optionally uploaded to GCS; the download route can stream from either.

---

## Make predictions outside the app

```python
import joblib, pandas as pd

art = joblib.load("model_<uuid>.joblib")
pipe = art["pipe"]
features = art["features"]

df = pd.read_csv("new_data.csv")
missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

pred = pipe.predict(df[features])
proba = pipe.predict_proba(df[features])[:, 1] if hasattr(pipe, "predict_proba") else None

out = df.copy()
out["prediction"] = pred
if proba is not None:
    out["prediction_proba"] = proba
out.to_csv("predictions.csv", index=False)
```

> **Note:** `joblib` uses pickle; only load artifacts from **trusted sources**. Use versions in `metadata.json` for best compatibility.

---

## Security notes
- **Secrets**: use Secret Manager or env vars; never hard-code.  
- **URIs**: dataset URIs are not shown in the UI.  
- **Pickle safety**: only load joblib artifacts you trust.  
- **Uploads**: size-limited; processed via pandas within a controlled pipeline.

---

## Roadmap
- Cross-validation and lightweight hyperparameter search  
- Explainability (feature importances, SHAP)  
- Model registry & experiment tracking  
- BigQuery/GCS dataset pickers; Cloud Run Jobs for long trainings  
- Threshold tuning UX for binary classifiers

---


## Citation / Acknowledgments
- Mission datasets: K2, TESS, Kepler (NASA/MAST)  
- Built with Flask, pandas, scikit-learn, joblib, and Google Cloud.  

---

*We turn weeks of notebook wrangling into a five-step, reproducible launch sequence — from starlight to trustworthy signals.*
