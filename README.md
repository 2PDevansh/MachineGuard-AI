# MachineGuard AI

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![RAG](https://img.shields.io/badge/GenAI-RAG-purple)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-lightgrey)
<img src="https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
![Frontend](https://img.shields.io/badge/UI-HTML%2FCSS%2FJS-e34c26)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

MachineGuard AI is a predictive maintenance project for industrial machine monitoring. It combines an XGBoost failure-risk model, a FastAPI backend, a browser-based control-room UI, and a lightweight maintenance assistant grounded in local maintenance rules — with an optional RAG/LLM mode for deeper explanations.

The application predicts machine failure probability from sensor readings and gives maintenance guidance for tool wear, torque, temperature, power, and overall risk.

---

## Table of Contents

- [Screenshots](#screenshots)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Run the FastAPI Backend](#run-the-fastapi-backend)
- [Run the Frontend](#run-the-frontend)
- [API Endpoints](#api-endpoints)
- [Optional LLM and RAG Modes](#optional-llm-and-rag-modes)
- [Model Inference Flow](#model-inference-flow)
- [Streamlit App](#streamlit-app)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

---

## Screenshots

### Diagnostic Dashboard

Add screenshot here.

### Assistant Page

Add screenshot here.

### Session Log

Add screenshot here.

---

## Key Features

- Predict machine failure probability from AI4I-style sensor inputs.
- Show risk level as `safe`, `watch`, or `crit`.
- Display confidence, predicted operating mode, recommendations, and session history.
- Provide a fast Assistant page for maintenance questions, backed by local rules by default.
- Keep diagnostics usable even when the local LLM stack is unavailable.
- Optional RAG/LLM mode using FAISS, sentence-transformers, and a local Hugging Face model, toggled by environment variable.
- Two interchangeable interfaces: a Streamlit app for quick single-process use, and a FastAPI + browser console for a proper client/server setup.
- No frontend build step required — plain HTML/CSS/JS.

---

## System Architecture

<img width="376" height="417" alt="image" src="https://github.com/user-attachments/assets/7f282024-2a2b-4673-8292-e54c345e3d89" />



---

## Tech Stack

**Machine Learning**
- Python 3.10+
- XGBoost
- Scikit-learn
- Pandas, NumPy

**Backend & API**
- FastAPI
- Uvicorn (ASGI server)
- Pydantic request/response models

**GenAI / RAG (optional)**
- Hugging Face Transformers
- `Qwen/Qwen2.5-1.5B-Instruct` (CPU-friendly default)
- Sentence-Transformers (`all-MiniLM-L6-v2`)
- FAISS (CPU) for vector retrieval

**Frontend**
- Vanilla HTML/CSS/JS, no build step
- SVG-based analog risk gauge
- Fetch-based API client with an offline demo mode

**Interfaces & Ops**
- Streamlit (original single-process dashboard)
- Pickle (model persistence)
- Modular, environment-variable-driven feature toggles
- Google Colab (T4 GPU compatible) for training/experimentation

---

## Dataset

- **Source**: UCI AI4I 2020 Predictive Maintenance Dataset
- **Type**: Industrial sensor telemetry (air temperature, process temperature, rotational speed, torque, tool wear, product type)
- **Target**: Machine failure (binary classification)

---

## Project Structure

```text
predictive_maintainance-ai/
|-- api/
|   |-- main.py                         # FastAPI backend
|-- app/
|   |-- streamlit_app.py                # Streamlit interface
|-- artifacts/
|   |-- models/
|       |-- xgboost_model.pkl           # Trained XGBoost model
|-- data/
|   |-- knowledge_base/
|       |-- maintenance_guidelines.txt  # Local assistant rules
|       |-- faiss_index.bin             # FAISS index
|       |-- chunks.pkl                  # Retrieved text chunks
|-- frontend/
|   |-- index.html                      # Browser UI
|   |-- css/
|   |   |-- style.css
|   |-- js/
|       |-- config.js                   # API URL config
|       |-- api.js                      # Frontend API client
|       |-- app.js                      # UI logic
|       |-- gauge.js                    # Risk gauge
|-- notebooks/
|   |-- ai_engineer_assistant_colab.ipynb
|-- src/
|   |-- data/
|   |   |-- load_data.py
|   |   |-- preprocess.py
|   |-- features/
|   |   |-- build_features.py           # Feature engineering
|   |-- llm/
|   |   |-- build_retriever.py          # One-time FAISS index builder
|   |   |-- maintenance_assistant.py    # Optional LLM assistant
|   |   |-- retriever.py                # FAISS retriever
|   |-- models/
|       |-- predict.py                  # Model inference
|       |-- train_xgboost.py            # Training script
|-- requirements.txt
|-- README.md
```

---

## Requirements

- Python 3.10 or newer
- A modern browser
- The Python packages listed in `requirements.txt`

Install dependencies from the project root:

```powershell
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]"
```

If `fastapi` and `uvicorn` are already installed, the second command is not needed.

---

## Run the FastAPI Backend

From the project root:

```powershell
python -m uvicorn api.main:app --reload --port 8000
```

Health check:

```text
http://localhost:8000/api/health
```

Expected response shape:

```json
{
  "status": "ok",
  "assistant_available": false,
  "assistant_loaded": false,
  "llm_chat_enabled": false,
  "assistant_error": null
}
```

`assistant_available` can be `false` and diagnostics will still work. The backend intentionally keeps the LLM optional so the main prediction flow does not fail because of Hugging Face, FAISS, GPU, or network issues.

---

## Run the Frontend

The frontend is a static website. It does not need npm or a build step.

From the project root:

```powershell
cd frontend
python -m http.server 5500
```

Open:

```text
http://localhost:5500
```

The frontend is configured in:

```text
frontend/js/config.js
```

For the FastAPI backend above, use:

```js
window.MG_CONFIG = {
  API_BASE_URL: "http://localhost:8000",

  ENDPOINTS: {
    predict: "/api/predict",
    assistant: "/api/assistant/chat",
    health: "/api/health"
  },

  REQUEST_TIMEOUT: 15000
};
```

If `API_BASE_URL` is empty, the frontend runs in demo mode and generates local browser-only predictions.

---

## API Endpoints

### GET `/api/health`

Returns backend status and optional assistant status.

Example:

```json
{
  "status": "ok",
  "assistant_available": false,
  "assistant_loaded": false,
  "llm_chat_enabled": false,
  "assistant_error": null
}
```

### POST `/api/predict`

Runs the trained XGBoost model and returns failure risk.

Request:

```json
{
  "type": "M",
  "airTemp": 300.0,
  "processTemp": 311.0,
  "rotSpeed": 1400,
  "torque": 45.0,
  "toolWear": 180
}
```

Response:

```json
{
  "failureProbability": 0.0007,
  "riskLevel": "safe",
  "predictedMode": "Healthy operation",
  "confidence": 0.9986,
  "recommendations": [
    "Machine is operating in a nominal range. Continue normal monitoring and scheduled preventive maintenance."
  ]
}
```

Risk levels:

```text
safe  - failure probability below 0.30
watch - failure probability from 0.30 to below 0.65
crit  - failure probability 0.65 or higher
```

### POST `/api/assistant/chat`

Answers maintenance questions from fast local maintenance rules by default.

Request:

```json
{
  "message": "How often should tool wear be inspected on a CNC spindle?"
}
```

Response:

```json
{
  "answer": "Inspect CNC spindle tool wear at the normal preventive-maintenance cadence...",
  "sources": ["maintenance_guidelines.txt"]
}
```

The default chat path avoids loading the large LLM so the Assistant page does not time out.

---

## Optional LLM and RAG Modes

The project includes an optional RAG assistant in:

```text
src/llm/maintenance_assistant.py
src/llm/retriever.py
```

It uses:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `sentence-transformers/all-MiniLM-L6-v2`
- FAISS index files from `data/knowledge_base/`

These components can be slow on CPU and may require the model files to be available locally. That is why they are disabled by default for the web console.

Enable LLM explanations for diagnostic recommendations:

```powershell
$env:MG_ENABLE_LLM_EXPLANATIONS="1"
python -m uvicorn api.main:app --reload --port 8000
```

Enable LLM chat:

```powershell
$env:MG_ENABLE_LLM_CHAT="1"
python -m uvicorn api.main:app --reload --port 8000
```

Enable both at once:

```powershell
$env:MG_ENABLE_LLM_EXPLANATIONS="1"
$env:MG_ENABLE_LLM_CHAT="1"
python -m uvicorn api.main:app --reload --port 8000
```

If the LLM fails, the backend falls back to fast local maintenance answers where possible.

---

## Model Inference Flow

1. The frontend sends sensor readings to `/api/predict`.
2. `api/main.py` converts browser field names to model feature names.
3. `src/models/predict.py` loads `artifacts/models/xgboost_model.pkl`.
4. `src/features/build_features.py` applies feature engineering.
5. The model returns failure probability and binary prediction.
6. The API maps probability to risk level and maintenance recommendations.

Input fields:

```text
type         - product type: L, M, or H
airTemp      - air temperature
processTemp  - process temperature
rotSpeed     - rotational speed
torque       - torque
toolWear     - tool wear
```

---

## Streamlit App

The repository also includes a Streamlit app:

```powershell
streamlit run app/streamlit_app.py
```

Use the FastAPI plus frontend flow for the browser control-room UI.

---

## Troubleshooting

### Run Diagnostic button fails

Make sure the backend is running:

```powershell
python -m uvicorn api.main:app --reload --port 8000
```

Also check that `frontend/js/config.js` points to:

```js
API_BASE_URL: "http://localhost:8000"
```

### Assistant page says "Request timed out"

Restart the backend after pulling or editing the latest code:

```powershell
python -m uvicorn api.main:app --reload --port 8000
```

The current backend answers Assistant questions without loading the heavy LLM unless `MG_ENABLE_LLM_CHAT=1` is set.

### Port 8000 is already in use

Start the API on another port:

```powershell
python -m uvicorn api.main:app --reload --port 8001
```

Then update `frontend/js/config.js`:

```js
API_BASE_URL: "http://localhost:8001"
```

### Hugging Face or sentence-transformers errors

Diagnostics do not require the LLM. Leave these environment variables unset unless you specifically want LLM features:

```text
MG_ENABLE_LLM_EXPLANATIONS
MG_ENABLE_LLM_CHAT
```

### XGBoost pickle warning

You may see a warning when loading `xgboost_model.pkl` with a newer XGBoost version. The current prediction path still works. For production, retrain or export the model using the current XGBoost version.

---

## Notes

- The frontend is dependency-free and can be served by any static file server.
- The backend allows CORS from all origins for local development. Tighten CORS settings before deploying publicly.
- Keep `artifacts/models/xgboost_model.pkl` available or predictions will fail.
- Keep `data/knowledge_base/maintenance_guidelines.txt` available for local assistant answers.
- Optional LLM features are opt-in via environment variables so the core prediction flow never depends on GPU, internet access, or large model downloads.

---

## License

This project is available under the MIT License. Add a `LICENSE` file at the repository root if one is not already present.
