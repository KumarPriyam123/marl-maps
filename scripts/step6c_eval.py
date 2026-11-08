"""STEP 6c — full-system comparison on held-out questions 800-999 (CPU only).

For each 1.5B pipeline (untrained, GRPO-trained) a stop policy is learned on its OWN
trajectories for questions 400-799 (same recipe as Step 4: logistic regression on SGWM
state, process reward R(t) = F1_t - 0.05 t, theta by inner CV) and then frozen.
Everything is scored on questions 800-999, next to the 7B pipeline from Step 5.

Inputs : results/step6c_1.5B_{base,grpo}_{train,test}_steps.csv
         results/step5_n200_steps.csv + results/step4_policy.json   (7B reference)
Output : results/step6c_system.json

Run from the repo root:  python scripts/step6c_eval.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step4_policy as sp  # noqa: E402

LAM = sp.PRIMARY_LAMBDA
PARAMS_B = {"7B": 7.62, "1.5B_base": 1.54, "1.5B_grpo": 1.54}   # parameters, billions
TAG = os.environ.get("STEP6C_TAG", "step6c")


def frozen_7b_probs(d, pol):
    x = d[pol["features"]].astype(float).values
    z = ((x - np.array(pol["mean"])) / np.array(pol["scale"])) @ np.array(pol["coef"]) + pol["intercept"]
    return 1 / (1 + np.exp(-z))


def policies(test, model=None, theta=None, probs=None):
    tmax = int(test.t.max())
    p = probs if probs is not None else model.predict_proba(test[sp.CHEAP])[:, 1]
    th = theta
    return {"judge_stop": sp.run_policy(test, test.verdict == 1),
            "fixed_t1": sp.run_policy(test, test.t == 1),
            "always_full": sp.run_policy(test, test.t == tmax),
            "learned": sp.run_policy(test, p >= th),
            "oracle": sp.run_policy(test, test.y == 1)}


def score(picks, params):
    s = sp.score(picks, LAM, False)
    s["llm_calls_x_params_B"] = round(s["llm_calls"] * params, 2)   # rough compute proxy
    return s


def both(a, b):
    f = sp.paired(a, b)
    r = sp.paired(a, b, col="t")
    return {"f1_points": f, "extra_searches": {"mean": r["mean"] / 100,
                                               "ci95": [x / 100 for x in r["ci95"]],
                                               "p_le_0": r["p_le_0"]}}


def main():
    out = {"lambda": LAM, "params_B": PARAMS_B, "note_compute":
           "llm_calls x params is a rough proxy, not measured latency", "systems": {}}
    picks = {}
    for pipe in ("1.5B_base", "1.5B_grpo"):
        tr, _ = sp.load(f"results/{TAG}_{pipe}_train_steps.csv")
        te, _ = sp.load(f"results/{TAG}_{pipe}_test_steps.csv")
        tr, te = sp.add_labels(tr, LAM), sp.add_labels(te, LAM)
        model, theta = sp.fit_policy(tr, sp.CHEAP, "logreg", LAM)
        pk = policies(te, model, theta)
        picks[pipe] = pk
        out["systems"][pipe] = {"theta": theta, **{k: score(v, PARAMS_B[pipe]) for k, v in pk.items()},
                                "by_t_F1": {int(t): round(100 * g.f1.mean(), 2) for t, g in te.groupby("t")}}

    pol = json.load(open("results/step4_policy.json"))["deploy_policy"]
    s7, _ = sp.load("results/step5_n200_steps.csv")
    s7 = sp.add_labels(s7, LAM)
    pk7 = policies(s7, probs=frozen_7b_probs(s7, pol), theta=pol["theta"])
    picks["7B"] = pk7
    out["systems"]["7B"] = {"theta": pol["theta"], **{k: score(v, PARAMS_B["7B"]) for k, v in pk7.items()},
                            "by_t_F1": {int(t): round(100 * g.f1.mean(), 2) for t, g in s7.groupby("t")}}

    ids = set(picks["7B"]["learned"].id) & set(picks["1.5B_grpo"]["learned"].id)
    out["n_common_questions"] = len(ids)
    sel = {k: {p: v[v.id.isin(ids)] for p, v in d.items()} for k, d in picks.items()}
    out["paired"] = {
        "grpo_learned_vs_base_learned": both(sel["1.5B_base"]["learned"], sel["1.5B_grpo"]["learned"]),
        "grpo_learned_vs_7B_learned": both(sel["7B"]["learned"], sel["1.5B_grpo"]["learned"]),
        "grpo_learned_vs_7B_judge_stop": both(sel["7B"]["judge_stop"], sel["1.5B_grpo"]["learned"]),
        "grpo_full_vs_7B_full": both(sel["7B"]["always_full"], sel["1.5B_grpo"]["always_full"]),
    }
    with open("results/step6c_system.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"HELD-OUT questions 800-999 (n={len(ids)}), lambda={LAM}")
    print(f"{'system':10s} {'policy':12s}   F1     EM  searches  calls  calls*B")
    for sysn, d in out["systems"].items():
        for k in ("judge_stop", "fixed_t1", "learned", "always_full", "oracle"):
            v = d[k]
            print(f"{sysn:10s} {k:12s} {v['F1']:6.2f} {v['EM']:6.2f} {v['avg_extra_retrievals']:6.3f}"
                  f" {v['llm_calls']:6.2f} {v['llm_calls_x_params_B']:7.2f}")
    print(json.dumps(out["paired"], indent=1))
    print("saved results/step6c_system.json")


if __name__ == "__main__":
    main()
