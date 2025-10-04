from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = "change-me"  # needed for flash messages

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATASETS = ["K2", "TESS", "Kepler", "Other"]
MODELS = ["RandomForest", "GradientBoosting", "AdaBoost", "NeuralNet", "LogisticRegression"]

@app.get("/")
def index():
    return render_template("index.html", datasets=DATASETS, models=MODELS)

@app.post("/submit")
def submit():
    dataset = request.form.get("dataset")
    model = request.form.get("model")
    uploaded_filename = None

    # Handle file upload only if "Other" was chosen
    if dataset == "Other":
        file = request.files.get("datafile")
        if not file or file.filename == "":
            flash("Please upload a file for 'Other' dataset.")
            return redirect(url_for("index"))
        fname = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_DIR, fname)
        file.save(save_path)
        uploaded_filename = fname

    # Basic validation
    if not dataset or not model:
        flash("Please choose both dataset and model.")
        return redirect(url_for("index"))

    # In a real app, enqueue a job here with (dataset, uploaded file path, model)
    # For now, just show a summary page.
    return redirect(url_for("summary", dataset=dataset, model=model, upload=uploaded_filename or ""))

@app.get("/summary")
def summary():
    dataset = request.args.get("dataset")
    model = request.args.get("model")
    upload = request.args.get("upload") or None
    return render_template("summary.html", dataset=dataset, model=model, upload=upload)

if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=8080, debug=True)