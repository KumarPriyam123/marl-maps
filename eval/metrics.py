"""Exact Match (EM) and token-level F1 -- the metrics the base paper reports.

Normalization follows the standard SQuAD recipe: lowercase, strip punctuation,
remove articles, collapse whitespace. F1 is computed on the multiset of tokens.
"""
import re
import string
import collections


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)  # both empty -> 1, one empty -> 0
    common = collections.Counter(p_toks) & collections.Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def score_one(pred: str, gold) -> dict:
    """gold may be a string or a list of acceptable answers; take the best."""
    golds = gold if isinstance(gold, list) else [gold]
    em = max(exact_match(pred, g) for g in golds)
    f = max(f1(pred, g) for g in golds)
    return {"em": em, "f1": f}


def aggregate(scores) -> dict:
    n = max(len(scores), 1)
    return {
        "EM": round(100 * sum(s["em"] for s in scores) / n, 2),
        "F1": round(100 * sum(s["f1"] for s in scores) / n, 2),
        "n": len(scores),
    }
