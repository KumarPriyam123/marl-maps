"""One-command sanity check. Runs the scripted MAPS baseline with the mock
backend + BM25 + offline sample data. Needs nothing but Python 3.8+.

    python run_smoke_test.py

If you see a RESULTS block with EM/F1 numbers, the whole pipeline works and you
can move on to plugging in a real backend (Groq/Ollama) and real data.
"""
import sys
import subprocess

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "eval.run_eval",
           "--backend", "mock", "--retriever", "bm25",
           "--data", "local", "--n", "10", "--verbose"]
    print("Running:", " ".join(cmd), "\n")
    raise SystemExit(subprocess.call(cmd))
