"""
api/main.py

Thin FastAPI wrapper around the existing prediction + RAG assistant code
so the frontend/ console has something to talk to.

Run from the project root (the folder containing app/, src/, frontend/):

    pip install fastapi "uvicorn[standard]"
    python -m uvicorn api.main:app --reload --port 8000

NOTE: importing src.llm.maintenance_assistant loads microsoft/phi-2,
a sentence-transformer, and a FAISS index at import time. The first
request (in fact, the import itself) can take a while and needs a
reasonable amount of RAM. If you don't have the LLM stack set up yet,
you can still exercise /api/predict — only /api/assistant/chat and the
LLM explanation in /api/predict depend on it (see the try/except below).
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make sure the project root is importable regardless of where uvicorn
# is launched from.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.predict import predict as ml_predict  # noqa: E402

# The RAG/LLM assistant is optional and heavy, so keep it lazy. A diagnostic
# should not fail or time out just because local Hugging Face assets are not
# cached or the machine is offline.
ASSISTANT_AVAILABLE = None
ASSISTANT_IMPORT_ERROR = None
_assistant_rag = None
_retrieve_relevant_context = None


app = FastAPI(title="MachineGuard AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class SensorReading(BaseModel):
    type: str          # "L" | "M" | "H"
    airTemp: float
    processTemp: float
    rotSpeed: float
    torque: float
    toolWear: float


class ChatMessage(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def risk_level_for(probability: float) -> str:
    if probability >= 0.65:
        return "crit"
    if probability >= 0.30:
        return "watch"
    return "safe"


def assistant_enabled_for_diagnostics() -> bool:
    return os.getenv("MG_ENABLE_LLM_EXPLANATIONS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def assistant_enabled_for_chat() -> bool:
    return os.getenv("MG_ENABLE_LLM_CHAT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_assistant() -> bool:
    """
    Load the RAG/LLM stack only when a route actually needs it.
    """
    global ASSISTANT_AVAILABLE, ASSISTANT_IMPORT_ERROR
    global _assistant_rag, _retrieve_relevant_context

    if ASSISTANT_AVAILABLE is not None:
        return ASSISTANT_AVAILABLE

    try:
        from src.llm.maintenance_assistant import ai_engineer_assistant_rag
        from src.llm.retriever import retrieve_relevant_context

        _assistant_rag = ai_engineer_assistant_rag
        _retrieve_relevant_context = retrieve_relevant_context
        ASSISTANT_AVAILABLE = True
        ASSISTANT_IMPORT_ERROR = None
    except Exception as exc:  # broad on purpose: torch/transformers/faiss setup issues
        ASSISTANT_AVAILABLE = False
        ASSISTANT_IMPORT_ERROR = str(exc)

    return ASSISTANT_AVAILABLE


def to_model_input(reading: SensorReading) -> dict:
    # Matches the exact column names src/features/build_features.py expects
    return {
        "Air temperature": reading.airTemp,
        "Process temperature": reading.processTemp,
        "Rotational speed": reading.rotSpeed,
        "Torque": reading.torque,
        "Tool wear": reading.toolWear,
        "Type": reading.type,
    }


def fallback_recommendations(probability: float, reading: SensorReading) -> list[str]:
    temp_delta = reading.processTemp - reading.airTemp

    if probability >= 0.65:
        recommendations = [
            "Critical failure risk detected. Stop or reduce load and inspect the machine before the next production run."
        ]
    elif probability >= 0.30:
        recommendations = [
            "Elevated failure risk detected. Keep the asset under observation and repeat the diagnostic after the next operating cycle."
        ]
    else:
        recommendations = [
            "Machine is operating in a nominal range. Continue normal monitoring and scheduled preventive maintenance."
        ]

    if reading.toolWear >= 200:
        recommendations.append("Tool wear is high; inspect or replace the tool before extended operation.")
    if reading.torque >= 55:
        recommendations.append("Torque is elevated; verify load, alignment, and product-type operating limits.")
    if temp_delta <= 8:
        recommendations.append("Temperature margin is low; check cooling, airflow, and heat dissipation paths.")

    return recommendations


def fast_chat_answer(message: str) -> tuple[str, list[str]]:
    """
    Fast local assistant used by the web console. It avoids loading the LLM,
    so the Assistant page always responds inside the frontend timeout.
    """
    text = message.lower()
    sources = ["maintenance_guidelines.txt"]

    if "tool" in text or "wear" in text or "spindle" in text:
        return (
            "Inspect CNC spindle tool wear at the normal preventive-maintenance cadence, and increase the cadence when wear is high or torque is elevated. In this project, excessive tool wear combined with high torque is treated as an overstrain warning, so a practical rule is to check wear at the start of each shift for routine operation, after any abnormal torque/vibration event, and before continuing production when tool wear approaches the high range.",
            sources,
        )

    if "torque" in text or "overstrain" in text:
        return (
            "High torque combined with excessive tool wear suggests overstrain risk. Reduce load if possible, inspect the tool and spindle condition, verify alignment and workholding, and repeat the diagnostic before returning the machine to sustained operation.",
            sources,
        )

    if "temperature" in text or "heat" in text or "cool" in text:
        return (
            "A high process-to-air temperature difference together with high tool wear can indicate heat dissipation failure. Check coolant flow, ventilation, spindle load, and recent temperature trends before extending the run.",
            sources,
        )

    if "power" in text:
        return (
            "Power consumption outside the normal operating range may indicate power failure or mechanical resistance. Inspect drive load, torque demand, bearings, tool condition, and any recent process changes.",
            sources,
        )

    if "failure" in text or "probability" in text or "risk" in text:
        return (
            "A failure probability below 0.2 usually indicates healthy operation, while a probability above 0.6 should trigger preventive maintenance. For values between those bands, trend the reading and inspect the strongest contributors such as tool wear, torque, temperature margin, and power demand.",
            sources,
        )

    return (
        "Use the diagnostic result as the first decision point: low probability usually means normal monitoring, while high probability should trigger preventive maintenance. The key local indicators are tool wear, torque, temperature difference, and power consumption.",
        sources,
    )


def generate_chat_answer(message: str) -> tuple[str, list[str]]:
    """
    Free-form Q&A against the knowledge base. The repo's
    ai_engineer_assistant_rag() is built specifically around a
    prediction context, so for open questions we reuse the same
    retriever + model/tokenizer objects with a simpler prompt.
    """
    if not load_assistant():
        raise RuntimeError(f"Assistant not available: {ASSISTANT_IMPORT_ERROR}")

    import torch
    from src.llm.maintenance_assistant import model, tokenizer, DEVICE  # already loaded

    retrieved_context = _retrieve_relevant_context(message)

    system_msg = (
        "You are a senior industrial maintenance engineer. Answer the question in ONE "
        "professional paragraph, maximum 120 words, in plain English. Do not write code, "
        "and do not mention models, algorithms, or libraries."
    )
    user_msg = f"""Question: {message}

Maintenance reference:
{retrieved_context}"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    new_tokens = output[0][input_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    sources = [line.strip() for line in retrieved_context.split("\n") if line.strip()][:2]
    return answer, sources


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "assistant_available": bool(ASSISTANT_AVAILABLE),
        "assistant_loaded": ASSISTANT_AVAILABLE is not None,
        "llm_chat_enabled": assistant_enabled_for_chat(),
        "assistant_error": ASSISTANT_IMPORT_ERROR,
    }


@app.post("/api/predict")
def predict_endpoint(reading: SensorReading):
    try:
        model_input = to_model_input(reading)
        result = ml_predict(model_input)  # {"failure_probability": float, "prediction": 0|1}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    probability = result["failure_probability"]
    risk_level = risk_level_for(probability)
    confidence = round(abs(probability - 0.5) * 2, 4)

    recommendations = fallback_recommendations(probability, reading)
    if assistant_enabled_for_diagnostics() and load_assistant():
        try:
            explanation = _assistant_rag(
                model_input,
                {"failure_probability": probability},
            )
            recommendations = [explanation]
        except Exception as exc:
            import traceback
            traceback.print_exc()  # full traceback in the uvicorn console
            recommendations.append(f"AI explanation unavailable: {exc!r}")
    elif assistant_enabled_for_diagnostics():
        recommendations.append(f"AI explanation unavailable: {ASSISTANT_IMPORT_ERROR}")

    return {
        "failureProbability": probability,
        "riskLevel": risk_level,
        "predictedMode": "Failure risk" if result["prediction"] == 1 else "Healthy operation",
        "confidence": confidence,
        "recommendations": recommendations,
    }


@app.post("/api/assistant/chat")
def chat_endpoint(msg: ChatMessage):
    if not assistant_enabled_for_chat():
        answer, sources = fast_chat_answer(msg.message)
        return {"answer": answer, "sources": sources}

    if not load_assistant():
        answer, sources = fast_chat_answer(msg.message)
        sources.append(f"LLM unavailable: {ASSISTANT_IMPORT_ERROR}")
        return {"answer": answer, "sources": sources}

    try:
        answer, sources = generate_chat_answer(msg.message)
    except Exception as exc:
        answer, sources = fast_chat_answer(msg.message)
        sources.append(f"LLM failed: {exc}")

    return {"answer": answer, "sources": sources}
