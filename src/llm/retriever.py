import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INDEX_PATH = os.path.join(
    BASE_DIR, "data", "knowledge_base", "faiss_index.bin"
)

CHUNKS_PATH = os.path.join(
    BASE_DIR, "data", "knowledge_base", "chunks.pkl"
)


# -----------------------------
# Load FAISS index & chunks
# -----------------------------
if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError("FAISS index or chunks file not found.")

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)


# -----------------------------
# Load embedding model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# RETRIEVER FUNCTION (THIS WAS MISSING)
# -----------------------------
def retrieve_relevant_context(query: str, top_k: int = 2) -> str:
    """
    Retrieve relevant maintenance knowledge using FAISS
    """

    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    retrieved_chunks = []
    for idx in indices[0]:
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])

    return "\n".join(retrieved_chunks)
