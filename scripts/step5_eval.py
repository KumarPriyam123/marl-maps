"""STEP 5 — final HELD-OUT evaluation of the stop policies (CPU only).

Inputs : results/step5_n<N>_steps.csv   (Kaggle Step 5: questions 800-999, never used before)
         results/step4_policy.json      (frozen learned policy, trained on questions 400-799)
Output : results/step5_heldout.json

Confirmatory (pre-registered, nothing is refit on these questions):
    judge_stop | fixed_t1 (always one extra search) | learned_frozen | baselines | oracle
Exploratory (clearly labelled): does the answer agent's own token confidence help?
    5-fold CV *within* the held-out questions, so it is NOT a held-out result.

Run from the repo root:  python scripts/step5_eval.py results/step5_n200_steps.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step4_policy as sp  # noqa: E402  (shared loaders, scoring, bootstrap)


def frozen_probs(d, pol):
    x = d[pol["features"]].astype(float).values
    z = ((x - np.array(pol["mean"])) / np.array(pol["scale"])) @ np.array(pol["coef"]) + pol["intercept"]
    return 1 / (1 + np.exp(-z))


def main(path):
    pol = json.load(open("results/step4_policy.json"))["deploy_policy"]
    lam = pol["lambda"]
    base, tmax = sp.load(path)
    d = sp.add_labels(base, lam)
    p = frozen_probs(d, pol)

    picks = {"stop_at_t0": sp.run_policy(d, d.t == 0),
             "judge_stop": sp.run_policy(d, d.verdict == 1),
             **{f"fixed_t{k}": sp.run_policy(d, d.t == k) for k in range(1, tmax)},
             "always_full": sp.run_policy(d, d.t == tmax),
             "learned_frozen": sp.run_policy(d, p >= pol["theta"]),
             "oracle": sp.run_policy(d, d.y == 1)}
    out = {"source": path, "n_questions": int(d.id.nunique()), "lambda": lam,
           "frozen_policy_theta": pol["theta"],
           "confirmatory": {k: sp.score(v, lam, False) for k, v in picks.items()},
           "stop_distribution": {k: {int(t): int(c) for t, c in v.t.value_counts().sort_index().items()}
                                 for k, v in picks.items() if k in ("judge_stop", "learned_frozen")}}

    def both(a, b):
        f = sp.paired(picks[a], picks[b])
        r = sp.paired(picks[a], picks[b], col="t")
        r = {"mean": r["mean"] / 100, "ci95": [x / 100 for x in r["ci95"]], "p_le_0": r["p_le_0"]}
        return {"f1_points": f, "extra_searches": r}

    out["paired"] = {"learned_minus_judge": both("judge_stop", "learned_frozen"),
                     "learned_minus_fixed_t1": both("fixed_t1", "learned_frozen"),
                     "judge_minus_fixed_t1": both("fixed_t1", "judge_stop")}

    # ---------------- exploratory: answer-agent token confidence ----------------
    if "ans_logprob_mean" in base.columns and base.ans_logprob_mean.notna().mean() > 0.8:
        e = d.copy()
        e["ans_lp_mean"] = e.ans_logprob_mean.astype(float).fillna(e.ans_logprob_mean.astype(float).median())
        e["ans_lp_min"] = e.ans_logprob_min.astype(float).fillna(e.ans_logprob_min.astype(float).median())
        good = (e.f1 >= 0.5).astype(int)
        auc = lambda s: (round(float(roc_auc_score(good, s)), 4) if good.nunique() == 2 else None)
        expl = {"note": "exploratory: 5-fold CV inside the held-out questions",
                "AUROC_ans_lp_mean_for_f1_ge_0.5": auc(e.ans_lp_mean),
                "AUROC_ans_lp_min_for_f1_ge_0.5": auc(e.ans_lp_min),
                "AUROC_judge_conf_for_f1_ge_0.5": auc(e.conf_logit)}
        feats = sp.CHEAP + ["ans_lp_mean", "ans_lp_min"]
        cv = []
        for tr, te in GroupKFold(n_splits=5).split(e, e.y, e.id):
            train, test = e.iloc[tr].copy(), e.iloc[te].copy()
            m, th = sp.fit_policy(train, feats, "logreg", lam)
            cv.append(sp.run_policy(test, m.predict_proba(test[feats])[:, 1] >= th))
        cvp = pd.concat(cv)
        # answering at every step is required to read its confidence -> 3t+2 calls
        expl["policy_with_answer_conf_cv"] = sp.score(cvp, lam, True)
        expl["paired_vs_fixed_t1"] = sp.paired(picks["fixed_t1"], cvp)
        out["exploratory_answer_confidence"] = expl

    with open("results/step5_heldout.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"HELD-OUT questions={out['n_questions']}  lambda={lam}  (F1 / EM / searches / LLM calls / reward)")
    for k, v in out["confirmatory"].items():
        print(f"  {k:16s} {v['F1']:6.2f} {v['EM']:6.2f} {v['avg_extra_retrievals']:6.3f} "
              f"{v['llm_calls']:6.3f} {v['reward']:7.4f}")
    print(json.dumps(out["paired"], indent=1))
    if "exploratory_answer_confidence" in out:
        print(json.dumps(out["exploratory_answer_confidence"], indent=1))
    print("saved results/step5_heldout.json")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/step5_eval.py results/step5_n200_steps.csv")
    main(sys.argv[1])
