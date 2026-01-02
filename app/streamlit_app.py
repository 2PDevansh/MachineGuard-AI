import streamlit as st
import pandas as pd
import joblib
import os
import sys

# -----------------------------
# Fix Python path
# -----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.features.build_features import build_features
from src.llm.maintenance_assistant import ai_engineer_assistant_rag

# -----------------------------
# Model feature order (LOCKED)
# -----------------------------
MODEL_FEATURE_ORDER = [
    'Air temperature',
    'Process temperature',
    'Rotational speed',
    'Torque',
    'Tool wear',
    'Type_L',
    'Type_M',
    'temp_diff',
    'thermal_load',
    'power',
    'speed_torque_ratio',
    'wear_torque_interaction',
    'normalized_wear',
    'log_power',
    'log_tool_wear'
]

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AI Predictive Maintenance System",
    layout="centered"
)

st.title("AI Predictive Maintenance System")
st.write("ML-based failure prediction with AI-powered maintenance explanation")

# -----------------------------
# Load ML model
# -----------------------------
MODEL_PATH = os.path.join(
    ROOT_DIR, "artifacts", "models", "xgboost_model.pkl"
)
model = joblib.load(MODEL_PATH)

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Machine Sensor Inputs")

air_temp = st.sidebar.number_input("Air temperature (K)", value=300.0)
process_temp = st.sidebar.number_input("Process temperature (K)", value=311.0)
rot_speed = st.sidebar.number_input("Rotational speed (rpm)", value=1400)
torque = st.sidebar.number_input("Torque (Nm)", value=45.0)
tool_wear = st.sidebar.number_input("Tool wear (min)", value=180)
machine_type = st.sidebar.selectbox("Machine type", ["L", "M", "H"])

# -----------------------------
# Run prediction
# -----------------------------
if st.sidebar.button("Run Prediction"):

    # Raw input dataframe
    raw_input = pd.DataFrame([{
        "Air temperature": air_temp,
        "Process temperature": process_temp,
        "Rotational speed": rot_speed,
        "Torque": torque,
        "Tool wear": tool_wear,
        "Type": machine_type
    }])

    # Feature engineering
    X_fe = build_features(raw_input)

    # Enforce training feature order (CRITICAL)
    X_fe = X_fe[MODEL_FEATURE_ORDER]

    # ML inference
    failure_prob = model.predict_proba(X_fe)[:, 1][0]

    # -----------------------------
    # Display ML results
    # -----------------------------
    st.subheader("ML Prediction")
    st.metric("Failure Probability", f"{failure_prob:.2f}")
    st.write(
        "Predicted Status: "
        f"**{'FAILURE RISK' if failure_prob >= 0.5 else 'HEALTHY'}**"
    )

    # -----------------------------
    # LLM explanation
    # -----------------------------
    features_for_llm = {
        "Air temperature": air_temp,
        "Process temperature": process_temp,
        "Rotational speed": rot_speed,
        "Torque": torque,
        "Tool wear": tool_wear,
        "Type": machine_type
    }

    prediction_for_llm = {
        "failure_probability": round(float(failure_prob), 2)
    }

    with st.spinner("Generating AI maintenance assessment..."):
        explanation = ai_engineer_assistant_rag(
            features_for_llm,
            prediction_for_llm
        )

    st.subheader("AI Maintenance Engineer Assessment")
    st.write(explanation)
