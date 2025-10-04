
import os, io, json
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import pandas as pd

# Optional: enable GCS uploads (requires google-cloud-storage + IAM + GCS_BUCKET env)
USE_GCS = False
try:
    from google.cloud import storage
    storage_client = storage.Client()
    USE_GCS = True
except Exception:
    USE_GCS = False

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Preconfigured dataset locations (set these env vars in Cloud Run)
DATASET_URIS = {
    "K2": os.environ.get("K2_URI", ""),
    "TESS": os.environ.get("TESS_URI", ""),
    "Kepler": os.environ.get("KEPLER_URI", ""),
}

DATASETS = ["K2", "TESS", "Kepler", "Other"]
MODELS = ["RandomForest", "GradientBoosting", "AdaBoost", "NeuralNet", "LogisticRegression"]

# Parameter schemas for dynamic forms
PARAM_SCHEMAS = {
    "RandomForest": [
        {"name":"n_estimators","type":"int","default":100,"help":"Number of trees."},
        {"name":"max_depth","type":"int_or_none","default":None,"help":"Depth limit."},
        {"name":"min_samples_split","type":"int","default":2},
        {"name":"min_samples_leaf","type":"int","default":1},
        {"name":"max_features","type":"str_or_float","default":"sqrt","help":"'sqrt', 'log2', or a float (0-1)."},
        {"name":"bootstrap","type":"bool","default":True},
        {"name":"class_weight","type":"str_or_none","default":None,"help":"'balanced' or None"},
        {"name":"random_state","type":"int_or_none","default":None}
    ],
    "GradientBoosting": [
        {"name":"n_estimators","type":"int","default":100},
        {"name":"learning_rate","type":"float","default":0.1},
        {"name":"max_depth","type":"int","default":3},
        {"name":"subsample","type":"float","default":1.0},
        {"name":"random_state","type":"int_or_none","default":None}
    ],
    "AdaBoost": [
        {"name":"n_estimators","type":"int","default":50},
        {"name":"learning_rate","type":"float","default":1.0},
        {"name":"algorithm","type":"choice","choices":["SAMME.R","SAMME"],"default":"SAMME.R"}
    ],
    "NeuralNet": [
        {"name":"hidden_layer_sizes","type":"tuple_ints","default":"100","help":"e.g., '100' or '128,64'"},
        {"name":"activation","type":"choice","choices":["relu","tanh","logistic"],"default":"relu"},
        {"name":"alpha","type":"float","default":0.0001,"help":"L2 penalty."},
        {"name":"learning_rate_init","type":"float","default":0.001},
        {"name":"max_iter","type":"int","default":200},
        {"name":"random_state","type":"int_or_none","default":None}
    ],
    "LogisticRegression": [
        {"name":"penalty","type":"choice","choices":["l2","l1","elasticnet","none"],"default":"l2"},
        {"name":"C","type":"float","default":1.0,"help":"Inverse of regularization strength."},
        {"name":"solver","type":"choice","choices":["lbfgs","liblinear","saga","newton-cg"],"default":"lbfgs"},
        {"name":"max_iter","type":"int","default":200},
        {"name":"class_weight","type":"str_or_none","default":None,"help":"'balanced' or None"},
        {"name":"random_state","type":"int_or_none","default":None}
    ]
}

def _read_table(uri: str, nrows: int = None):
    # Pandas can read gs:// if gcsfs is installed (included in requirements)
    if uri.lower().endswith(".csv"):
        return pd.read_csv(uri, nrows=nrows, comment = "#")
    elif uri.lower().endswith(".parquet") or uri.lower().endswith(".pq"):
        return pd.read_parquet(uri, engine="pyarrow", columns=None)
    else:
        return pd.read_csv(uri, nrows=nrows)

@app.get("/")
def index():
    return render_template("index.html", datasets=DATASETS, models=MODELS)

@app.post("/submit")
def submit():
    dataset = request.form.get("dataset")
    model = request.form.get("model")
    if not dataset or not model:
        flash("Please choose both dataset and model.")
        return redirect(url_for("index"))

    # Resolve dataset_uri
    if dataset == "Other":
        file = request.files.get("datafile")
        if not file or file.filename == "":
            flash("Please upload a file for 'Other' dataset.")
            return redirect(url_for("index"))
        fname = secure_filename(file.filename)
        if USE_GCS and os.environ.get("GCS_BUCKET"):
            bucket_name = os.environ["GCS_BUCKET"]
            bucket_path = f"uploads/{fname}"
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(bucket_path)
            blob.upload_from_file(file.stream, rewind=True)
            dataset_uri = f"gs://{bucket_name}/{bucket_path}"
        else:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            local_path = os.path.join(UPLOAD_DIR, fname)
            file.save(local_path)
            dataset_uri = local_path
    else:
        dataset_uri = (os.environ.get(f"{dataset.upper()}_URI") or "").strip()
        if not dataset_uri:
            flash(f"No URI configured for {dataset}. Set the {dataset.upper()}_URI env var.")
            return redirect(url_for("index"))

    session["dataset"] = dataset
    session["dataset_uri"] = dataset_uri
    session["model"] = model
    return redirect(url_for("columns"))

@app.get("/columns")
def columns():
    dataset_uri = session.get("dataset_uri")
    model = session.get("model")
    if not dataset_uri or not model:
        return redirect(url_for("index"))
    try:
        df = _read_table(dataset_uri, nrows=200)
        cols = list(df.columns)
    except Exception as e:
        flash(f"Could not read dataset: {e}")
        return redirect(url_for("index"))
    return render_template("columns.html", columns=cols, model=model, dataset_uri=dataset_uri)

@app.post("/columns")
def columns_post():
    feature_cols = request.form.getlist("features")
    target_col = request.form.get("target")
    if not feature_cols or not target_col:
        flash("Please select at least one feature and a target column.")
        return redirect(url_for("columns"))
    session["feature_cols"] = feature_cols
    session["target_col"] = target_col
    return redirect(url_for("tune"))

@app.get("/tune")
def tune():
    model = session.get("model")
    if not model:
        return redirect(url_for("index"))
    schema = PARAM_SCHEMAS.get(model, [])
    return render_template("tune.html", model=model, schema=schema)

@app.post("/tune")
def tune_post():
    model = session.get("model")
    dataset_uri = session.get("dataset_uri")
    feature_cols = session.get("feature_cols", [])
    target_col = session.get("target_col")
    if not (model and dataset_uri and feature_cols and target_col):
        flash("Session expired or incomplete. Please start again.")
        return redirect(url_for("index"))
    params = {}
    for p in PARAM_SCHEMAS.get(model, []):
        name = p["name"]
        raw = request.form.get(name, "")
        t = p.get("type", "str")
        if raw == "" or raw is None:
            val = None
        else:
            try:
                if t == "int": val = int(raw)
                elif t == "float": val = float(raw)
                elif t == "bool": val = raw.lower() in ["1","true","yes","on"]
                elif t == "int_or_none": val = None if raw.lower()=="none" else int(raw)
                elif t == "str_or_none": val = None if raw.lower()=="none" else raw
                elif t == "str_or_float":
                    try: val = float(raw)
                    except: val = raw
                elif t == "tuple_ints": val = tuple(int(x.strip()) for x in raw.split(",")) if raw else None
                else: val = raw
            except Exception:
                val = raw
        params[name] = val
    session["params"] = params
    return redirect(url_for("review"))

@app.get("/review")
def review():
    return render_template("review.html",
        dataset=session.get("dataset"),
        dataset_uri=session.get("dataset_uri"),
        model=session.get("model"),
        feature_cols=session.get("feature_cols", []),
        target_col=session.get("target_col"),
        params=session.get("params", {})
    )

@app.get("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
