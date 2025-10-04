import os, io, json, uuid, tempfile
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, make_response
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
from sklearn.feature_selection import SelectFromModel
import joblib

USE_GCS = False
try:
    from google.cloud import storage
    storage_client = storage.Client()
    USE_GCS = True
except Exception:
    USE_GCS = False

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev_thingies")
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/models")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATASETS = ["K2", "TESS", "Kepler", "Other"]
MODELS = ["RandomForest", "GradientBoosting", "AdaBoost", "NeuralNet", "LogisticRegression"]

MODEL_DESCRIPTIONS = {
    "RandomForest": {
        "summary": "Ensemble of decision trees via bagging; robust, handles non-linearities and mixed features well.",
        "pros": ["Works well out-of-the-box", "Handles mixed dtypes", "Less prone to overfitting than single trees"],
        "cons": ["Larger models", "Less interpretable", "Not ideal for sparse very-high-dimensional data"]
    },
    "GradientBoosting": {
        "summary": "Sequentially adds shallow trees to correct previous errors; strong for structured/tabular data.",
        "pros": ["High accuracy on tabular data", "Good with non-linear relations"],
        "cons": ["Sensitive to hyperparams", "Longer training time than RF"]
    },
    "AdaBoost": {
        "summary": "Boosting method that reweights mistakes; fast and simple with weak learners.",
        "pros": ["Simple", "Often competitive on small datasets"],
        "cons": ["Sensitive to noisy data/outliers", "Can underfit without tuning"]
    },
    "NeuralNet": {
        "summary": "Multi-layer perceptron (feed-forward) with non-linear activations.",
        "pros": ["Captures complex patterns", "Flexible architecture"],
        "cons": ["Requires scaling", "Sensitive to hyperparams", "Longer training time"]
    },
    "LogisticRegression": {
        "summary": "Linear classifier with probabilistic outputs; strong baseline.",
        "pros": ["Fast", "Interpretable coefficients", "Works well with many samples/low noise"],
        "cons": ["Linear decision boundary", "Needs scaling", "Limited for complex feature interactions"]
    }
}

PARAM_SCHEMAS = {
    "RandomForest": [
        {"name":"n_estimators","type":"int","default":100,"help":"Number of trees. Larger can improve performance but increases time."},
        {"name":"max_depth","type":"int_or_none","default":None,"help":"Max depth of trees. None = expand until pure."},
        {"name":"min_samples_split","type":"int","default":2,"help":"Min samples required to split an internal node."},
        {"name":"min_samples_leaf","type":"int","default":1,"help":"Min samples at a leaf node."},
        {"name":"max_features","type":"str_or_float","default":"sqrt","help":"Features considered per split. 'sqrt', 'log2', or float fraction."},
        {"name":"bootstrap","type":"bool","default":True,"help":"Whether to bootstrap samples."},
        {"name":"class_weight","type":"str_or_none","default":None,"help":"'balanced' for class imbalance, or None."},
        {"name":"random_state","type":"int_or_none","default":42,"help":"Seed for reproducibility."}
    ],
    "GradientBoosting": [
        {"name":"n_estimators","type":"int","default":100,"help":"Number of boosting stages."},
        {"name":"learning_rate","type":"float","default":0.1,"help":"Shrinkage applied to each tree's contribution."},
        {"name":"max_depth","type":"int","default":3,"help":"Depth of individual trees (controls complexity)."},
        {"name":"subsample","type":"float","default":1.0,"help":"Fraction of samples for each tree (<1.0 introduces randomness)."},
        {"name":"random_state","type":"int_or_none","default":42,"help":"Seed for reproducibility."}
    ],
    "AdaBoost": [
        {"name":"n_estimators","type":"int","default":50,"help":"Number of weak learners."},
        {"name":"learning_rate","type":"float","default":1.0,"help":"Weight applied to each weak learner."},
        {"name":"algorithm","type":"choice","choices":["SAMME.R","SAMME"],"default":"SAMME.R","help":"Real (SAMME.R) uses probabilities; SAMME uses class labels."}
    ],
    "NeuralNet": [
        {"name":"hidden_layer_sizes","type":"tuple_ints","default":"100","help":"e.g., '128,64' for two hidden layers."},
        {"name":"activation","type":"choice","choices":["relu","tanh","logistic"],"default":"relu","help":"Activation function."},
        {"name":"alpha","type":"float","default":0.0001,"help":"L2 regularization term."},
        {"name":"learning_rate_init","type":"float","default":0.001,"help":"Initial learning rate."},
        {"name":"max_iter","type":"int","default":200,"help":"Max training iterations."},
        {"name":"random_state","type":"int_or_none","default":42,"help":"Seed for reproducibility."}
    ],
    "LogisticRegression": [
        {"name":"penalty","type":"choice","choices":["l2","l1","elasticnet","none"],"default":"l2","help":"Regularization type."},
        {"name":"C","type":"float","default":1.0,"help":"Inverse of regularization strength. Smaller = stronger reg."},
        {"name":"solver","type":"choice","choices":["lbfgs","liblinear","saga","newton-cg"],"default":"lbfgs","help":"Optimization algorithm."},
        {"name":"max_iter","type":"int","default":200,"help":"Max iterations."},
        {"name":"class_weight","type":"str_or_none","default":None,"help":"'balanced' for class imbalance, or None."},
        {"name":"random_state","type":"int_or_none","default":42,"help":"Seed for reproducibility."}
    ]
}

L1L2_HELP = "Lasso (L1) encourages sparsity—zeroing weaker features. Ridge (L2) shrinks coefficients smoothly. We use a Logistic Regression selector before your chosen model; smaller C = stronger selection."

def read_table(uri: str, nrows: int = None) -> pd.DataFrame:
    if uri.lower().endswith(".csv"):
        return pd.read_csv(uri, nrows=nrows, comment ="#")
    elif uri.lower().endswith(".parquet") or uri.lower().endswith(".pq"):
        return pd.read_parquet(uri)
    else:
        return pd.read_csv(uri, nrows=nrows)

def build_pipeline(model_name: str, params: dict, numeric_cols, cat_cols,
                   fs_method: str = "none", fs_C: float = 1.0):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name in ("NeuralNet", "LogisticRegression"):
        numeric_steps.append(("scaler", StandardScaler()))
    num_tf = Pipeline(steps=numeric_steps)
    cat_tf = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                             ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_tf, numeric_cols),
                             ("cat", cat_tf, cat_cols)], remainder="drop", sparse_threshold=0.3)
    # Base estimator
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
        raise ValueError(f"Unsupported model: {model_name}")

    steps = [("pre", pre)]
    if fs_method in ("lasso", "ridge"):
        penalty = "l1" if fs_method == "lasso" else "l2"
        fs_est = LogisticRegression(penalty=penalty, C=float(fs_C), solver="saga", max_iter=2000)
        steps.append(("fs", SelectFromModel(fs_est, threshold="median")))
    steps.append(("model", model))
    return Pipeline(steps=steps)

def parse_params(model_name: str, form) -> dict:
    schema = PARAM_SCHEMAS[model_name]
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
    return render_template(
        "index.html",
        datasets=DATASETS,
        models=MODELS,
        model_descriptions=MODEL_DESCRIPTIONS,
    )
@app.post("/submit")
def submit():
    dataset = request.form.get("dataset")
    model = request.form.get("model")
    if not dataset or not model:
        flash("Please choose both dataset and model.")
        return redirect(url_for("index"))

    # Resolve dataset uri
    if dataset == "Other":
        file = request.files.get("datafile")
        if not file or file.filename == "":
            flash("Please upload a file for 'Other' dataset.")
            return redirect(url_for("index"))
        fname = secure_filename(file.filename)
        if USE_GCS and os.environ.get("GCS_BUCKET"):
            bucket = storage_client.bucket(os.environ["GCS_BUCKET"])
            path = f"uploads/{fname}"
            blob = bucket.blob(path)
            blob.upload_from_file(file.stream, rewind=True)
            dataset_uri = f"gs://{os.environ['GCS_BUCKET']}/{path}"
        else:
            local_path = os.path.join(UPLOAD_DIR, fname)
            file.save(local_path)
            dataset_uri = local_path
    else:
        env_key = f"{dataset.upper()}_URI"
        dataset_uri = (os.environ.get(env_key) or "").strip()
        if not dataset_uri:
            flash(f"Set {env_key} with the gs:// path for {dataset}.")
            return redirect(url_for("index"))

    session["dataset"] = dataset
    session["dataset_uri"] = dataset_uri
    session["model"] = model
    return redirect(url_for("columns"))

@app.get("/columns")
def columns():
    uri = session.get("dataset_uri")
    if not uri:
        return redirect(url_for("index"))
    try:
        df = read_table(uri, nrows=500)
        cols = list(df.columns)
    except Exception as e:
        flash(f"Could not read dataset: {e}")
        return redirect(url_for("index"))
    help_text = "Pick your target (label) and the feature columns used to train. Non-numeric features will be one-hot encoded automatically."
    return render_template("columns.html", columns=cols, dataset=session.get("dataset"), model=session.get("model"), help_text=help_text)

@app.post("/columns")
def columns_post():
    feats = request.form.getlist("features")
    target = request.form.get("target")
    if not feats or not target:
        flash("Please select at least one feature and a target column.")
        return redirect(url_for("columns"))
    session["feature_cols"] = feats
    session["target_col"] = target
    return redirect(url_for("tune"))

@app.get("/tune")
def tune():
    model = session.get("model")
    if not model: return redirect(url_for("index"))
    return render_template("tune.html", model=model, schema=PARAM_SCHEMAS[model],
                           model_info=MODEL_DESCRIPTIONS[model], l1l2_help=L1L2_HELP)

@app.post("/tune")
def tune_post():
    model = session.get("model")
    if not model: return redirect(url_for("index"))
    session["params"] = parse_params(model, request.form)
    session["fs_method"] = request.form.get("fs_method", "none")
    try:
        session["fs_C"] = float(request.form.get("fs_C", "1.0"))
    except Exception:
        session["fs_C"] = 1.0
    return redirect(url_for("split"))

@app.get("/split")
def split():
    return render_template("split.html")

def evaluate(df, features, target, model_name, params, do_split, train_size, fs_method, fs_C):
    X = df[features]; y = df[target]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pipe = build_pipeline(model_name, params, num_cols, cat_cols, fs_method, fs_C)

    out = {}
    if do_split:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=(1-train_size),
                                                  random_state=42, stratify=y if y.nunique()>=2 else None)
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

    return pipe, out, num_cols, cat_cols

@app.post("/run")
def run():
    uri = session.get("dataset_uri")
    features = session.get("feature_cols", [])
    target = session.get("target_col")
    model_name = session.get("model")
    params = session.get("params", {})
    fs_method = session.get("fs_method", "none")
    fs_C = float(session.get("fs_C", 1.0))

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

    pipe, res, num_cols, cat_cols = evaluate(df, features, target, model_name, params, do_split, train_size, fs_method, fs_C)

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
    fs_method = session.get("fs_method", "none")
    fs_C = float(session.get("fs_C", 1.0))

    try:
        df = read_table(uri)
    except Exception as e:
        flash(f"Read error: {e}")
        return redirect(url_for("index"))

    X = df[features]; y = df[target]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pipe = build_pipeline(model_name, params, num_cols, cat_cols, fs_method, fs_C)
    pipe.fit(X, y)

    model_id = str(uuid.uuid4())
    local_path = os.path.join(MODEL_DIR, f"model_{model_id}.joblib")
    joblib.dump({"pipe": pipe, "features": features, "feature_types": {c: ("num" if c in num_cols else "cat") for c in features}}, local_path)
    session["model_path"] = local_path
    session["features"] = features

    if USE_GCS and os.environ.get("RESULTS_BUCKET"):
        try:
            bucket_uri = os.environ["RESULTS_BUCKET"].rstrip("/")
            if bucket_uri.startswith("gs://"):
                bucket_name = bucket_uri[5:].split("/", 1)[0]
                prefix = bucket_uri[5+len(bucket_name):].strip("/")
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(f"{prefix}/models/model_{model_id}.joblib" if prefix else f"models/model_{model_id}.joblib")
                blob.upload_from_filename(local_path)
        except Exception:
            pass

    summary = {"rows": int(len(df)), "model": model_name, "target": target, "features": features, "params": params,
               "fs_method": fs_method, "fs_C": fs_C}
    session["summary"] = summary
    return redirect(url_for("final"))

@app.get("/final")
def final():
    summary = session.get("summary")
    features = session.get("features", [])
    feature_types = {}
    try:
        bundle = joblib.load(session.get("model_path"))
        feature_types = bundle.get("feature_types", {})
    except Exception:
        feature_types = {f: "num" for f in features}
    return render_template("final.html", summary=summary, features=features, feature_types=feature_types)

@app.post("/predict_one")
def predict_one():
    model_path = session.get("model_path")
    features = session.get("features", [])
    if not model_path or not os.path.exists(model_path):
        flash("Model not found in this instance. Please retrain.")
        return redirect(url_for("final"))
    bundle = joblib.load(model_path)
    pipe = bundle["pipe"]
    data = {}
    for f in features:
        val = request.form.get(f, "")
        try:
            data[f] = [float(val)] if val not in ("", None) else [None]
        except Exception:
            data[f] = [val if val != "" else None]
    df = pd.DataFrame(data)
    pred = pipe.predict(df)[0]
    proba = None
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        try:
            proba = pipe.predict_proba(df).tolist()[0]
        except Exception:
            proba = None
    return render_template("final.html",
                           summary=session.get("summary"),
                           features=features,
                           feature_types=bundle.get("feature_types", {}),
                           single_pred=pred,
                           single_proba=proba)

@app.post("/predict_csv")
def predict_csv():
    model_path = session.get("model_path")
    features = session.get("features", [])
    if not model_path or not os.path.exists(model_path):
        flash("Model not found in this instance. Please retrain.")
        return redirect(url_for("final"))
    bundle = joblib.load(model_path)
    pipe = bundle["pipe"]

    f = request.files.get("predict_file")
    if not f or f.filename == "":
        flash("Please upload a CSV file to predict on.")
        return redirect(url_for("final"))

    try:
        df = pd.read_csv(f)
    except Exception as e:
        flash(f"Could not read CSV: {e}")
        return redirect(url_for("final"))

    missing = [c for c in features if c not in df.columns]
    if missing:
        flash(f"Missing columns in uploaded CSV: {missing}")
        return redirect(url_for("final"))

    preds = pipe.predict(df[features])
    out = df.copy()
    out["prediction"] = preds

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        try:
            out["prediction_proba"] = pipe.predict_proba(df[features])[:, 1]
        except Exception:
            pass

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    resp = make_response(buf.getvalue())
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    return resp

@app.get("/template.csv")
def template_csv():
    features = session.get("features", [])
    buf = io.StringIO()
    buf.write(",".join(features) + "\\n")
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8")),
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name="prediction_template.csv")

@app.get("/healthz")
def healthz():
    return "ok", 200



if __name__ == "__main__": app.run(host="127.0.0.1", port=5000, debug=True)
