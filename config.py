"""Central configuration. Override any value with an environment variable."""
import os

# Which LLM backend the agents use: "mock" | "ollama" | "groq" | "gemini"
BACKEND = os.environ.get("LLM_BACKEND", "mock")

# Retriever: "bm25" (pure python, zero deps) | "dense" (E5 + FAISS, needs install)
RETRIEVER = os.environ.get("RETRIEVER", "bm25")
DENSE_MODEL = os.environ.get("DENSE_MODEL", "intfloat/e5-base-v2")

# Pipeline knobs (match the base paper's defaults where sensible)
TOP_K = int(os.environ.get("TOP_K", "3"))      # chunks retrieved per query
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "3"))  # max retrieval rounds
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "20"))  # global cap on evidence

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATA = os.path.join(ROOT, "data", "sample", "hotpot_sample.json")
INDEX_DIR = os.path.join(ROOT, "data", "index")
RESULTS_DIR = os.path.join(ROOT, "results")
