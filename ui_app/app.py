# app.py – Flask UI for Telugu News Summarizer + Evaluator + Entity Highlights
from flask import Flask, render_template, request
import subprocess
import json
import os

app = Flask(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    url = request.form.get("url", "").strip()
    size_choice = request.form.get("choice", "3").strip()
    custom_n = request.form.get("custom_n", "5").strip()

    if not url:
        return render_template("results.html", error="No URL provided.")

    # Step 1: Run summarisation pipeline (test2.py)
    try:
        subprocess.run(
            ["python", os.path.join(PROJECT_ROOT, "test2.py")],
            input=f"{url}\n{size_choice}\n{custom_n}\n",
            text=True,
            capture_output=True,
            cwd=PROJECT_ROOT,
            timeout=600
        )
    except subprocess.TimeoutExpired:
        return render_template("results.html", error="Summarization timed out (too long).")

    # Step 2: Run evaluation
    try:
        subprocess.run(
            ["python", os.path.join(PROJECT_ROOT, "evaluate_summary.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300
        )
    except subprocess.TimeoutExpired:
        return render_template("results.html", error="Evaluation timed out.")

    # Step 3: Load outputs
    summary_file = os.path.join(PROJECT_ROOT, "hybrid_summary.txt")
    report_file = os.path.join(PROJECT_ROOT, "evaluation_summary_report.txt")
    metrics_file = os.path.join(PROJECT_ROOT, "evaluation_summary_metrics.json")
    entity_file = os.path.join(PROJECT_ROOT, "entity_highlights.txt")

    summary_text = ""
    evaluation_report = ""
    metrics_json = {}
    entities_text = ""

    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_text = f.read()
    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            evaluation_report = f.read()
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_json = json.load(f)
    if os.path.exists(entity_file):
        with open(entity_file, "r", encoding="utf-8") as f:
            entities_text = f.read()

    return render_template(
        "results.html",
        url=url,
        summary=summary_text,
        report=evaluation_report,
        metrics=metrics_json,
        entities=entities_text
    )

if __name__ == "__main__":
    app.run(debug=True)
