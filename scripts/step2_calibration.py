"""STEP 2a — Calibrate the MAPS judge's confidence (CPU only, no LLM calls).

Input : results/step1_judge_calls.csv   (one row per judge call, from Kaggle Step 1)
        results/step1_questions.csv     (optional: adds a paired bootstrap on EM)
Output: results/step2_calibration.json

Label  : gold_covered_now = all gold passages were in the pool at that judge call
Score  : conf = sigmoid(z_Sufficient - z_Insufficient), first decision token
Methods: temperature scaling (1 param) and Platt scaling (2 params)
Honest eval: 2-fold split BY QUESTION (no question in both fit and test),
             repeated over 20 random splits; mean and std reported.

Run from the repo root:  python scripts/step2_calibration.py
Needs: numpy pandas scikit-learn
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

RES = "results"
SPLITS = 20


def ece(p, y, bins=10):
    edges, tot = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def sig(x):
    return 1 / (1 + np.exp(-x))


def metrics(p, y):
    return {"ECE": ece(p, y), "Brier": float(brier_score_loss(y, p)), "NLL": nll(p, y)}


def fit_temperature(z, y):
    Ts = np.exp(np.linspace(np.log(0.05), np.log(200), 600))
    return float(Ts[np.argmin([nll(sig(z / T), y) for T in Ts])])


def as_bool(s):
    return s.astype(str).str.strip().str.lower().eq("true")


def main():
    jc = pd.read_csv(f"{RES}/step1_judge_calls.csv")
    jc = jc[jc.conf.notna() & jc.gold_covered_now.notna()].copy()
    y = as_bool(jc.gold_covered_now).astype(float).values
    said = as_bool(jc.verdict).values
    c = jc.conf.astype(float).clip(1e-6, 1 - 1e-6).values
    z = np.log(c / (1 - c))
    q = jc.id.astype(str).values
    uniq = np.array(sorted(set(q)))

    runs = {"raw": [], "temperature": [], "platt": []}
    Ts, p_platt_last = [], None
    for seed in range(SPLITS):
        perm = uniq.copy()
        np.random.default_rng(seed).shuffle(perm)
        fold_of = {qq: i % 2 for i, qq in enumerate(perm)}
        fold = np.array([fold_of[qq] for qq in q])
        pt, pp = np.zeros_like(c), np.zeros_like(c)
        for k in (0, 1):
            tr, te = fold != k, fold == k
            T = fit_temperature(z[tr], y[tr])
            Ts.append(T)
            pt[te] = sig(z[te] / T)
            lr = LogisticRegression(C=1e6).fit(z[tr].reshape(-1, 1), y[tr])
            pp[te] = lr.predict_proba(z[te].reshape(-1, 1))[:, 1]
        runs["raw"].append(metrics(c, y))
        runs["temperature"].append(metrics(pt, y))
        runs["platt"].append(metrics(pp, y))
        p_platt_last = pp if seed == 0 else p_platt_last

    def agg(rows):
        out = {}
        for key in rows[0]:
            v = np.array([r[key] for r in rows])
            out[key] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4)}
        return out

    # early-stop usefulness: stop when calibrated P(covered) >= tau
    thr = []
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        stop = p_platt_last >= tau
        thr.append({"tau": tau, "stop_rate_pct": round(100 * stop.mean(), 1),
                    "precision_pct": round(100 * y[stop].mean(), 1) if stop.any() else None})

    by_round = {}
    for r, g in jc.groupby("round"):
        yy = as_bool(g.gold_covered_now).values
        ss = as_bool(g.verdict).values
        by_round[int(r)] = {"calls": int(len(g)), "covered_pct": round(100 * yy.mean(), 1),
                            "said_sufficient_pct": round(100 * ss.mean(), 1),
                            "verdict_acc_pct": round(100 * (ss == yy).mean(), 1)}

    out = {
        "n_calls": int(len(y)), "n_questions": int(len(uniq)), "splits": SPLITS,
        "base_rate_covered_pct": round(100 * y.mean(), 1),
        "verdict_accuracy_pct": round(100 * (said == y.astype(bool)).mean(), 1),
        "AUROC_raw": round(float(roc_auc_score(y, c)), 4),
        "conf_extreme_pct": round(100 * ((c < 0.02) | (c > 0.98)).mean(), 1),
        "temperature_T": {"median": round(float(np.median(Ts)), 3),
                          "min": round(float(min(Ts)), 3), "max": round(float(max(Ts)), 3)},
        "raw": agg(runs["raw"]),
        "temperature": agg(runs["temperature"]),
        "platt": agg(runs["platt"]),
        "stop_thresholds_platt": thr,
        "by_round": by_round,
    }

    qpath = f"{RES}/step1_questions.csv"
    if os.path.exists(qpath):
        qs = pd.read_csv(qpath, keep_default_na=False)
        a = qs[qs.system == "naive_rag"].set_index("id")
        b = qs[qs.system == "maps_scripted"].set_index("id")
        ids = a.index.intersection(b.index)
        rng = np.random.default_rng(0)
        for metric in ("em", "f1"):
            d = (b.loc[ids, metric].astype(float) - a.loc[ids, metric].astype(float)).values
            boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
            out[f"paired_{metric}_maps_minus_naive"] = {
                "mean": round(100 * float(d.mean()), 2),
                "ci95": [round(100 * float(np.percentile(boots, 2.5)), 2),
                         round(100 * float(np.percentile(boots, 97.5)), 2)],
                "p_le_0": round(float((boots <= 0).mean()), 4)}
    else:
        out["paired_test"] = "skipped: results/step1_questions.csv not found"

    with open(f"{RES}/step2_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    if not os.path.isdir(RES):
        sys.exit("Run from the repo root (results/ not found).")
    main()
