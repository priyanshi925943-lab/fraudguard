import os, sys, json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file

# Fix import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from predict import predict_risk

app = Flask(__name__)

# Define folders using absolute paths
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, 'data', 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'data', 'results')

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ============================================================
@app.route("/")
def index():
    return render_template("index.html")

# ============================================================
@app.route("/upload", methods=["POST"])
def upload():
    try:
        # Check file exists in request
        if "file" not in request.files:
            print("ERROR: No file in request")
            return redirect(url_for("index"))

        file = request.files["file"]

        # Check filename
        if file.filename == "":
            print("ERROR: Empty filename")
            return redirect(url_for("index"))

        # Check extension
        if not file.filename.endswith(".csv"):
            print("ERROR: Not a CSV file")
            return redirect(url_for("index"))

        # Save uploaded file
        upload_path = os.path.join(UPLOAD_FOLDER, "uploaded.csv")
        file.save(upload_path)
        print(f"File saved to: {upload_path}")

        # Run prediction
        print("Running prediction...")
        results_df = predict_risk(upload_path)
        print(f"Prediction done. Rows: {len(results_df)}")
        print(f"Columns: {list(results_df.columns)}")

        # Save full results
        results_path = os.path.join(RESULTS_FOLDER, "results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")

        # Calculate summary
        summary = {
            "total"  : len(results_df),
            "high"   : int((results_df["Risk_Label"] == "High").sum()),
            "medium" : int((results_df["Risk_Label"] == "Medium").sum()),
            "low"    : int((results_df["Risk_Label"] == "Low").sum())
        }
        print(f"Summary: {summary}")

        # Save summary
        summary_path = os.path.join(RESULTS_FOLDER, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f)
        print(f"Summary saved to: {summary_path}")

        # Save display rows
        display_path = os.path.join(RESULTS_FOLDER, "display.csv")
        results_df.head(100).to_csv(display_path, index=False)
        print(f"Display saved to: {display_path}")

        return redirect(url_for("results"))

    except Exception as e:
        print(f"UPLOAD ERROR: {e}")
        import traceback
        traceback.print_exc()
        return f"<h2>Error: {str(e)}</h2><a href='/'>Go Back</a>"

# ============================================================
@app.route("/results")
def results():
    try:
        summary_path = os.path.join(RESULTS_FOLDER, "summary.json")
        display_path = os.path.join(RESULTS_FOLDER, "display.csv")

        if not os.path.exists(summary_path):
            print("ERROR: summary.json not found")
            return redirect(url_for("index"))

        if not os.path.exists(display_path):
            print("ERROR: display.csv not found")
            return redirect(url_for("index"))

        with open(summary_path) as f:
            summary = json.load(f)

        display_df = pd.read_csv(display_path)
        results    = display_df.to_dict(orient="records")

        print(f"Rendering results. Summary: {summary}, Rows: {len(results)}")

        return render_template(
            "results.html",
            summary=summary,
            results=results
        )

    except Exception as e:
        print(f"RESULTS ERROR: {e}")
        return f"<h2>Error: {str(e)}</h2><a href='/'>Go Back</a>"

# ============================================================
@app.route("/metrics")
def metrics():
    return render_template("metrics.html", metrics={
        "precision" : 87.10,
        "recall"    : 82.65,
        "f1"        : 84.82,
        "fpr"       : 0.02,
        "tn"        : 56852,
        "fp"        : 12,
        "fn"        : 17,
        "tp"        : 81
    })

# ============================================================
@app.route("/about")
def about():
    return render_template("about.html")

# ============================================================
@app.route("/download")
def download():
    path = os.path.join(RESULTS_FOLDER, "results.csv")
    if not os.path.exists(path):
        return redirect(url_for("index"))
    return send_file(
        path,
        as_attachment=True,
        download_name="fraudguard_results.csv"
    )

# ============================================================
@app.route("/test")
def test():
    summary_path = os.path.join(RESULTS_FOLDER, "summary.json")
    display_path = os.path.join(RESULTS_FOLDER, "display.csv")
    msg  = f"Base dir: {BASE_DIR}<br>"
    msg += f"Summary exists: {os.path.exists(summary_path)}<br>"
    msg += f"Display exists: {os.path.exists(display_path)}<br>"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            msg += f"Summary: {f.read()}<br>"
    if os.path.exists(display_path):
        df   = pd.read_csv(display_path)
        msg += f"Display rows: {len(df)}<br>"
        msg += f"Columns: {list(df.columns)}<br>"
    return msg

# ============================================================
if __name__ == "__main__":
    print(f"Base dir: {BASE_DIR}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    app.run(debug=True)