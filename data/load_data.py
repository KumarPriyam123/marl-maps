"""
Dataset loading.

Week 1 default: a small local JSON in our own schema (works offline, no deps).
Real runs: HotpotQA distractor dev via HuggingFace `datasets`. 2Wiki / MuSiQue /
Bamboogle follow the same pattern (documented in the README).

Our schema (one list element per question):
  {
    "id": str,
    "question": str,
    "answer": str | [str, ...],
    "passages": [ {"title": str, "text": str, "is_gold": bool}, ... ]
  }
"""
from __future__ import annotations
import json
from typing import List, Dict


def load_local(path: str, n: int = None) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:n] if n else data


def load_hotpotqa(n: int = 50, split: str = "validation") -> List[Dict]:
    """HotpotQA distractor setting -> our schema. Needs `pip install datasets`.

    Each example ships with 10 context paragraphs (gold + distractors); we flatten
    them into passages. The union across questions becomes the retrieval corpus.
    """
    # Script-based loading (trust_remote_code) no longer works in current `datasets`;
    # read the auto-converted parquet instead (same file the Kaggle notebook uses).
    from datasets import load_dataset
    parquet = ("hf://datasets/hotpotqa/hotpot_qa@refs/convert/parquet/"
               f"distractor/{split}/0000.parquet")
    ds = load_dataset("parquet", data_files=parquet, split="train")
    out = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        gold_titles = set(ex["supporting_facts"]["title"])
        passages = []
        for t, sents in zip(titles, sentences):
            passages.append({
                "title": t,
                "text": " ".join(sents),
                "is_gold": t in gold_titles,
            })
        out.append({
            "id": ex.get("id", str(i)),
            "question": ex["question"],
            "answer": ex["answer"],
            "passages": passages,
        })
    return out


def load_dataset_by_name(name: str, n: int = 50, local_path: str = None) -> List[Dict]:
    name = name.lower()
    if name in ("local", "sample"):
        return load_local(local_path, n)
    if name in ("hotpotqa", "hotpot"):
        return load_hotpotqa(n)
    raise ValueError(f"Loader for '{name}' not wired yet. Use 'local' or 'hotpotqa', "
                     "or add a loader following the HotpotQA pattern.")
