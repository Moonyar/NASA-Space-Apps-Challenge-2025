from flask import Flask
import time

app = Flask(__name__)

@app.get("/")
def home():
    return "✅ Flask on Cloud Run — it works!"

@app.post("/run")
def run_model():
    time.sleep(1)  # pretend to do work
    return {"status": "DONE", "message": "stub run completed"}