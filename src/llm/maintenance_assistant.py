import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .retriever import retrieve_relevant_context


# -----------------------------
# Device setup
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Load LLM — smaller, instruct-tuned, much faster on CPU than phi-2
# -----------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
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

    query = f"""
    Machine failure probability is {prediction['failure_probability']}.
    Tool wear is {features['Tool wear']}.
    Temperature difference is {features['Process temperature'] - features['Air temperature']}.
    """

    retrieved_context = retrieve_relevant_context(query)

    system_msg = (
        "You are a senior industrial maintenance engineer. Respond with exactly ONE "
        "professional paragraph, maximum 100 words, in plain English. Do not write code, "
        "do not mention models/algorithms/libraries, do not analyze parameters individually, "
        "do not reference guidelines by number. Start with a sentence describing machine health."
    )
    user_msg = f"""Machine condition summary:
{features}

ML model output:
Failure probability = {prediction['failure_probability']}

Maintenance reference:
{retrieved_context}"""

    # Instruct models use a chat template rather than a raw prompt string
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
            max_new_tokens=100,       # was 120 — the task doesn't need more
            do_sample=False,          # greedy: faster + deterministic, fine for this task
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Only decode the newly generated tokens, not the whole prompt again
    input_len = inputs["input_ids"].shape[-1]
    new_tokens = output[0][input_len:]
    final_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

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