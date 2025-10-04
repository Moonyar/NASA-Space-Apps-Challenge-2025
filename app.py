
import os, json
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

USE_GCS = False
try:
    from google.cloud import storage
    storage_client = storage.Client()
    USE_GCS = True
except Exception:
    USE_GCS = False

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev")
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATASETS = ["K2", "TESS", "Kepler", "Other"]
MODELS = ["RandomForest", "GradientBoosting", "AdaBoost", "NeuralNet", "LogisticRegression"]

def param_schema():
    return {
        "RandomForest": [
            {"name":"n_estimators","type":"int","default":100},
            {"name":"max_depth","type":"int_or_none","default":None},
            {"name":"min_samples_split","type":"int","default":2},
            {"name":"min_samples_leaf","type":"int","default":1},
            {"name":"max_features","type":"str_or_float","default":"sqrt"},
            {"name":"bootstrap","type":"bool","default":True},
            {"name":"class_weight","type":"str_or_none","default":None},
            {"name":"random_state","type":"int_or_none","default":42}
        ],
        "GradientBoosting": [
            {"name":"n_estimators","type":"int","default":100},
            {"name":"learning_rate","type":"float","default":0.1},
            {"name":"max_depth","type":"int","default":3},
            {"name":"subsample","type":"float","default":1.0},
            {"name":"random_state","type":"int_or_none","default":42}
        ],
        "AdaBoost": [
            {"name":"n_estimators","type":"int","default":50},
            {"name":"learning_rate","type":"float","default":1.0},
            {"name":"algorithm","type":"choice","choices":["SAMME.R","SAMME"],"default":"SAMME.R"}
        ],
        "NeuralNet": [
            {"name":"hidden_layer_sizes","type":"tuple_ints","default":"100"},
            {"name":"activation","type":"choice","choices":["relu","tanh","logistic"],"default":"relu"},
            {"name":"alpha","type":"float","default":0.0001},
            {"name":"learning_rate_init","type":"float","default":0.001},
            {"name":"max_iter","type":"int","default":200},
            {"name":"random_state","type":"int_or_none","default":42}
        ],
        "LogisticRegression": [
            {"name":"penalty","type":"choice","choices":["l2","l1","elasticnet","none"],"default":"l2"},
            {"name":"C","type":"float","default":1.0},
            {"name":"solver","type":"choice","choices":["lbfgs","liblinear","saga","newton-cg"],"default":"lbfgs"},
            {"name":"max_iter","type":"int","default":200},
            {"name":"class_weight","type":"str_or_none","default":None},
            {"name":"random_state","type":"int_or_none","default":42}
        ]
    }

def read_table(uri: str, nrows: int = None):
    if uri.lower().endswith(".csv"):
        return pd.read_csv(uri, nrows=nrows, comment = "#")
    elif uri.lower().endswith(".parquet") or uri.lower().endswith(".pq"):
        return pd.read_parquet(uri)
    else:
        return pd.read_csv(uri, nrows=nrows)

def build_pipeline(model_name: str, params: dict, numeric_cols, cat_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name in ("NeuralNet", "LogisticRegression"):
        num_steps.append(("scaler", StandardScaler()))
    num_tf = Pipeline(steps=num_steps)

    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_tf, numeric_cols),
        ("cat", cat_tf, cat_cols)
    ], remainder="drop", sparse_threshold=0.3)

    if model_name == "RandomForest":
        model = RandomForestClassifier(**params)
    elif model_name == "GradientBoosting":
        model = GradientBoostingClassifier(**{k:v for k,v in params.items() if k in {"n_estimators","learning_rate","subsample","random_state","max_depth"}})
    elif model_name == "AdaBoost":
        model = AdaBoostClassifier(**params)
    elif model_name == "NeuralNet":
        model = MLPClassifier(**params)
    elif model_name == "LogisticRegression":
        model = LogisticRegression(**params)
    else:
        raise ValueError("Unsupported model")

    return Pipeline([("pre", pre), ("model", model)])

def parse_params(model_name: str, form):
    schema = param_schema()[model_name]
    params = {}
    for p in schema:
        name = p["name"]
        raw = form.get(name, "")
        t = p.get("type","str")
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
                elif t == "choice": val = raw
                else: val = raw
            except Exception:
                val = raw
        if val is not None:
            params[name] = val
    return params

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

    if dataset == "Other":
        file = request.files.get("datafile")
        if not file or file.filename == "":
            flash("Please upload a file for 'Other'.")
            return redirect(url_for("index"))
        fname = secure_filename(file.filename)
        if USE_GCS and os.environ.get("GCS_BUCKET"):
            bucket = storage_client.bucket(os.environ["GCS_BUCKET"])
            path = f"uploads/{fname}"
            blob = bucket.blob(path)
            blob.upload_from_file(file.stream, rewind=True)
            dataset_uri = f"gs://{os.environ['GCS_BUCKET']}/{path}"
        else:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            local_path = os.path.join(UPLOAD_DIR, fname)
            file.save(local_path)
            dataset_uri = local_path
    else:
        env_key = f"{dataset.upper()}_URI"
        dataset_uri = (os.environ.get(env_key) or "").strip()
        if not dataset_uri:
            flash(f"Set {env_key} to the gs:// path for {dataset}.")
            return redirect(url_for("index"))

    session["dataset"] = dataset
    session["dataset_uri"] = dataset_uri
    session["model"] = model
    return redirect(url_for("columns"))

@app.get("/columns")
def columns():
    uri = session.get("dataset_uri")
    if not uri: return redirect(url_for("index"))
    try:
        df = read_table(uri, nrows=500)
        cols = list(df.columns)
    except Exception as e:
        flash(f"Read error: {e}")
        return redirect(url_for("index"))
    return render_template("columns.html", columns=cols, dataset=session.get("dataset"), model=session.get("model"))

@app.post("/columns")
def columns_post():
    feats = request.form.getlist("features")
    target = request.form.get("target")
    if not feats or not target:
        flash("Pick at least one feature and a target.")
        return redirect(url_for("columns"))
    session["feature_cols"] = feats
    session["target_col"] = target
    return redirect(url_for("tune"))

@app.get("/tune")
def tune():
    model = session.get("model")
    if not model: return redirect(url_for("index"))
    return render_template("tune.html", model=model, schema=param_schema()[model])

@app.post("/tune")
def tune_post():
    model = session.get("model")
    if not model: return redirect(url_for("index"))
    session["params"] = parse_params(model, request.form)
    return redirect(url_for("split"))

@app.get("/split")
def split():
    return render_template("split.html")

def evaluate(df, features, target, model_name, params, do_split, train_size):
    X = df[features]
    y = df[target]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pipe = build_pipeline(model_name, params, num_cols, cat_cols)

    out = {}
    if do_split:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=(1-train_size), random_state=42, stratify=y if y.nunique()>=2 else None)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        out["n_train"] = int(len(X_tr)); out["n_test"] = int(len(X_te))
        avg = "binary" if y.nunique()==2 else "macro"
        out["accuracy"] = float(accuracy_score(y_te, y_pred))
        out["precision"] = float(precision_score(y_te, y_pred, average=avg, zero_division=0))
        out["recall"] = float(recall_score(y_te, y_pred, average=avg, zero_division=0))
        out["f1"] = float(f1_score(y_te, y_pred, average=avg, zero_division=0))
        if y.nunique()==2 and hasattr(pipe.named_steps["model"], "predict_proba"):
            try:
                proba = pipe.predict_proba(X_te)[:, 1]
                out["roc_auc"] = float(roc_auc_score(y_te, proba))
            except Exception: pass
        try:
            cm = confusion_matrix(y_te, y_pred).tolist()
            out["confusion_matrix"] = {"labels": sorted(y_te.unique().tolist()), "matrix": cm}
        except Exception: pass
    else:
        pipe.fit(X, y)
        out["n_train"] = int(len(X)); out["n_test"] = 0

    return pipe, out

@app.post("/run")
def run():
    uri = session.get("dataset_uri")
    features = session.get("feature_cols", [])
    target = session.get("target_col")
    model_name = session.get("model")
    params = session.get("params", {})
    if not (uri and features and target and model_name):
        flash("Session missing info. Start over.")
        return redirect(url_for("index"))

    do_split = request.form.get("do_split") == "on"
    try:
        train_size = float(request.form.get("train_size", "0.8"))
    except Exception:
        train_size = 0.8
    train_size = min(max(train_size, 0.5), 0.95)

    try:
        df = read_table(uri)
    except Exception as e:
        flash(f"Read error: {e}")
        return redirect(url_for("index"))

    pipe, res = evaluate(df, features, target, model_name, params, do_split, train_size)
    session["did_split"] = do_split
    session["train_size"] = train_size
    session["last_results"] = res
    return render_template("metrics.html", model=model_name, results=res, did_split=do_split, train_size=train_size)

@app.post("/train_all")
def train_all():
    uri = session.get("dataset_uri")
    features = session.get("feature_cols", [])
    target = session.get("target_col")
    model_name = session.get("model")
    params = session.get("params", {})
    try:
        df = read_table(uri)
    except Exception as e:
        flash(f"Read error: {e}")
        return redirect(url_for("index"))
    X = df[features]; y = df[target]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pipe = build_pipeline(model_name, params, num_cols, cat_cols)
    pipe.fit(X, y)
    summary = {"rows": int(len(df)), "model": model_name, "target": target, "features": features, "params": params}
    return render_template("final.html", summary=summary)

@app.get("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
