import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .retriever import retrieve_relevant_context



# -----------------------------
# Device setup
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Load LLM (open-source)
# -----------------------------
MODEL_NAME = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

model.to(DEVICE)
model.eval()


# -----------------------------
# AI Maintenance Engineer RAG
# -----------------------------
def ai_engineer_assistant_rag(features: dict, prediction: dict) -> str:
    """
    Generates a professional maintenance assessment using RAG + LLM
    """

    # Build retrieval query
    query = f"""
    Machine failure probability is {prediction['failure_probability']}.
    Tool wear is {features['Tool wear']}.
    Temperature difference is {features['Process temperature'] - features['Air temperature']}.
    """

    retrieved_context = retrieve_relevant_context(query)

    # Prompt
    prompt = f"""
You are a senior industrial maintenance engineer.

IMPORTANT:
Your response MUST start with a sentence describing machine health.
If you write code, the answer is invalid.

INSTRUCTIONS (MANDATORY):
- Write ONE professional paragraph
- Do NOT explain your reasoning
- Do NOT analyze parameters individually
- Do NOT reference guidelines by number
- Do NOT write code
- Do NOT mention models, algorithms, or libraries
- Use plain English only
- Maximum 100 words

Machine condition summary:
{features}

ML model output:
Failure probability = {prediction['failure_probability']}

Maintenance reference:
{retrieved_context}

FINAL ANSWER (TEXT ONLY):
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    output = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.4,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    final_answer = decoded_output.split(
        "FINAL ANSWER (TEXT ONLY):"
    )[-1].strip()

    # Safety filter
    forbidden_tokens = ["import ", "def ", "class ", "pd.", "sklearn", "python"]

    for token in forbidden_tokens:
        if token in final_answer.lower():
            return (
                "The machine shows an elevated risk of failure and is not operating "
                "under optimal conditions. Preventive maintenance is strongly "
                "recommended to reduce the likelihood of unplanned downtime."
            )

    return final_answer
