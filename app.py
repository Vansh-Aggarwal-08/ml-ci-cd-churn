# app.py
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model.pkl")

app = Flask(__name__)

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")

# Define feature columns manually (must match training order in train.py)
FEATURE_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "input" not in data:
            return jsonify({"error": "Missing 'input'"}), 400

        x = data["input"]
        if isinstance(x, dict):  # single record
            x = [x]

        df = pd.DataFrame(x)
        df = df.reindex(columns=FEATURE_COLS, fill_value=0)  # align with training

        preds = model.predict(df).tolist()
        return jsonify({"prediction": preds})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
