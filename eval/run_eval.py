"""
Run the scripted MAPS baseline over a dataset and report EM / F1.

Examples
--------
  # zero-dependency smoke test (mock LLM + BM25 + offline sample data)
  python -m eval.run_eval --backend mock --retriever bm25 --data local --n 8

  # real run on HotpotQA with free Groq backbone + dense retriever
  export GROQ_API_KEY=...
  python -m eval.run_eval --backend groq --retriever dense --data hotpotqa --n 50

  # offline real model via Ollama
  python -m eval.run_eval --backend ollama --retriever bm25 --data local --n 8
"""
from __future__ import annotations
import os
import sys
import json
import argparse

# allow running as `python -m eval.run_eval` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from llm_backend import get_backend
from retriever.retriever import get_retriever
from agents.agents import QueryExtensionAgent, JudgeAgent, AnswerAgent
from pipeline.maps_scripted import run_maps
from eval.metrics import score_one, aggregate
from data.load_data import load_dataset_by_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=config.BACKEND)
    ap.add_argument("--retriever", default=config.RETRIEVER)
    ap.add_argument("--data", default="local")
    ap.add_argument("--data-path", default=config.SAMPLE_DATA)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--max-rounds", type=int, default=config.MAX_ROUNDS)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"[setup] backend={args.backend}  retriever={args.retriever}  "
          f"data={args.data}  n={args.n}  top_k={args.top_k}  "
          f"max_rounds={args.max_rounds}")

    dataset = load_dataset_by_name(args.data, n=args.n, local_path=args.data_path)
    backend = get_backend(args.backend)
    retriever = get_retriever(dataset, kind=args.retriever)
    q_agent = QueryExtensionAgent(backend)
    j_agent = JudgeAgent(backend)
    a_agent = AnswerAgent(backend)

    scores, records = [], []
    total_rounds = 0
    for ex in dataset:
        result = run_maps(ex["question"], retriever, q_agent, j_agent, a_agent,
                          top_k=args.top_k, max_rounds=args.max_rounds,
                          max_chunks=config.MAX_CHUNKS)
        sc = score_one(result["answer"], ex["answer"])
        scores.append(sc)
        total_rounds += result["rounds"]
        records.append({
            "id": ex.get("id"),
            "question": ex["question"],
            "gold": ex["answer"],
            "pred": result["answer"],
            "rounds": result["rounds"],
            "queries": result["queries"],
            "em": sc["em"], "f1": round(sc["f1"], 3),
        })
        if args.verbose:
            print(f"\nQ: {ex['question']}")
            print(f"  queries : {result['queries']}")
            print(f"  pred    : {result['answer']!r}   gold: {ex['answer']!r}")
            print(f"  EM={sc['em']:.0f}  F1={sc['f1']:.2f}  rounds={result['rounds']}")

    agg = aggregate(scores)
    agg["avg_rounds"] = round(total_rounds / max(len(dataset), 1), 2)

    print("\n" + "=" * 48)
    print(f"  RESULTS  (n={agg['n']})")
    print(f"  EM         : {agg['EM']}")
    print(f"  F1         : {agg['F1']}")
    print(f"  avg rounds : {agg['avg_rounds']}")
    print("=" * 48)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        config.RESULTS_DIR,
        f"maps_{args.backend}_{args.retriever}_{args.data}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "aggregate": agg, "records": records},
                  f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
