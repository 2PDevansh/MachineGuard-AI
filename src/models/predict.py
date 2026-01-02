# src/models/predict.py

import pandas as pd
import joblib
from pathlib import Path

# ✅ IMPORT FEATURE PIPELINE
from src.features.build_features import build_features


def load_model():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    model_path = PROJECT_ROOT / "artifacts" / "models" / "xgboost_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    return joblib.load(model_path)


def predict(input_data: dict):
    model = load_model()

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # ✅ APPLY IDENTICAL FEATURE ENGINEERING
    df_fe = build_features(df)

    # Ensure column order matches training
    df_fe = df_fe[model.feature_names_in_]

    prob_failure = model.predict_proba(df_fe)[:, 1][0]
    prediction = int(prob_failure >= 0.5)

    return {
        "failure_probability": round(float(prob_failure), 4),
        "prediction": prediction
    }


def main():
    sample_input = {
        "Air temperature": 300.0,
        "Process temperature": 311.0,
        "Rotational speed": 1400,
        "Torque": 45.0,
        "Tool wear": 180,
        "Type": "M"
    }

    result = predict(sample_input)

    print("\n🔍 Inference Result")
    print(f"Failure Probability: {result['failure_probability']}")
    print(f"Prediction (1=Fail, 0=Healthy): {result['prediction']}")


if __name__ == "__main__":
    main()
