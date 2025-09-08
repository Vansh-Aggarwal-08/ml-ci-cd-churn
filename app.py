# app.py
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model.pkl")

app = Flask(__name__)

# Load once at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    # Fail fast with a helpful message
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")


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
        if isinstance(x, dict):   # single record
            x = [x]

        df = pd.DataFrame(x)

        # encode categorical cols
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])

        # align columns
        df = df.reindex(columns=feature_cols, fill_value=0)

        preds = model.predict(df).tolist()
        return jsonify({"prediction": preds})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
