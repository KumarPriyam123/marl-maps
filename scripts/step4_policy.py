"""STEP 4b — learn WHEN to stop searching, from full trajectories (CPU only).

Input : results/step4_n<N>_steps.csv   (Kaggle Step 4: one row per question x step t)
Output: results/step4_policy.json      (offline results + deployable policy weights)

Process reward (HiPRAG-style efficiency penalty), per question and step t:
    R(t) = F1_t - lambda * t          t = extra retrievals used before answering
Label  : stop_now(t) = 1 if R(t) >= max over later steps of R(t')  (optimal stopping)
Policy : stop at the first t where P(stop_now | SGWM state at t) >= theta
Model  : logistic regression (deployable as a few numbers); gradient boosting is
         reported for comparison only.
Eval   : 5-fold cross-validation BY QUESTION. theta is picked with an inner CV on the
         training questions only. Every policy is scored on the same held-out
         questions, exactly (stopping at t returns the logged answer_t).

Run from the repo root:   python scripts/step4_policy.py results/step4_n400_steps.csv
Needs: numpy pandas scikit-learn
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAMBDAS = (0.02, 0.05, 0.10)
PRIMARY_LAMBDA = 0.05          # fixed before looking at results: 1 search = 5 F1 points
THETAS = np.round(np.linspace(0.05, 0.95, 19), 2)
CHEAP = ["t", "verdict", "conf_logit", "n_new_last", "top_score_last",
         "mean_score_last", "qsim_last", "dup_last", "pool_size"]
FULL = CHEAP + ["pred_words", "pred_changed", "pred_in_pool"]   # needs an answer every step


def load(path):
    d = pd.read_csv(path, keep_default_na=False, na_values=[""])
    tf = lambda s: s.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    d["verdict"] = tf(d.verdict).fillna(0.0)
    d["dup_last"] = tf(d.dup_last).fillna(0.0)
    d["gold_covered"] = tf(d.gold_covered)
    c = d.conf.astype(float).fillna(0.5).clip(1e-6, 1 - 1e-6)
    d["conf_logit"] = np.log(c / (1 - c))
    d["pred_changed"] = d.pred_changed.astype(float).fillna(1.0)   # t=0: no previous answer
    for col in ("f1", "em", "pred_words", "pred_in_pool", "n_new_last", "top_score_last",
                "mean_score_last", "qsim_last", "pool_size", "t"):
        d[col] = d[col].astype(float)
    d = d.sort_values(["id", "t"]).reset_index(drop=True)
    tmax = int(d.t.max())
    good = d.groupby("id").t.count() == tmax + 1
    return d[d.id.isin(good[good].index)].reset_index(drop=True), tmax


def add_labels(d, lam):
    d = d.copy()
    d["R"] = d.f1 - lam * d.t
    d["best_later"] = d.groupby("id").R.transform(lambda r: r[::-1].cummax()[::-1].shift(-1))
    d["y"] = (d.R >= d.best_later.fillna(-np.inf)).astype(int)
    return d


def run_policy(d, stop_mask):
    """stop_mask: boolean array aligned with d. Returns one picked row per question."""
    d = d.assign(_stop=stop_mask | (d.t == d.t.max()))
    return d[d._stop].groupby("id").head(1)


def score(picks, lam, full_feats):
    t = picks.t
    calls = (3 * t + 2) if full_feats else (2 * t + 2)
    return {"F1": round(100 * picks.f1.mean(), 2), "EM": round(100 * picks.em.mean(), 2),
            "avg_extra_retrievals": round(t.mean(), 3),
            "llm_calls": round(calls.mean(), 3),
            "reward": round(float((picks.f1 - lam * t).mean()), 4)}


def make_model(kind):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05)


def fit_policy(train, feats, kind, lam):
    """Fit on train rows (t < tmax); choose theta by inner CV on train questions."""
    rows = train[train.t < train.t.max()]
    oof = np.zeros(len(train))
    inner = GroupKFold(n_splits=3)
    for a, b in inner.split(rows, rows.y, rows.id):
        m = make_model(kind).fit(rows.iloc[a][feats], rows.iloc[a].y)
        idx = rows.index[b]
        oof[train.index.get_indexer(idx)] = m.predict_proba(train.loc[idx, feats])[:, 1]
    best = max(THETAS, key=lambda th: (run_policy(train, oof >= th).pipe(
        lambda p: (p.f1 - lam * p.t).mean()), -th))
    model = make_model(kind).fit(rows[feats], rows.y)
    return model, float(best)


def paired(a, b, n=10000, col="f1"):
    b = b.set_index("id")
    d = (b[col] - a.set_index("id").loc[b.index, col]).values
    rng = np.random.default_rng(0)
    bo = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"mean": round(100 * d.mean(), 2),
            "ci95": [round(100 * np.percentile(bo, 2.5), 2), round(100 * np.percentile(bo, 97.5), 2)],
            "p_le_0": round(float((bo <= 0).mean()), 4)}


def main(path):
    base, tmax = load(path)
    out = {"source": path, "n_questions": int(base.id.nunique()), "max_t": tmax,
           "primary_lambda": PRIMARY_LAMBDA, "features_cheap": CHEAP, "features_full": FULL,
           "by_lambda": {}}
    for lam in LAMBDAS:
        d = add_labels(base, lam)
        res = {"stop_at_t0": score(run_policy(d, d.t == 0), lam, False),
               "judge_stop": score(run_policy(d, d.verdict == 1), lam, False),
               "always_full": score(run_policy(d, d.t == tmax), lam, False),
               **{f"fixed_t{k}": score(run_policy(d, d.t == k), lam, False)
                  for k in range(1, tmax)},
               "oracle": score(run_policy(d, d.y == 1), lam, False)}
        picks_all = {}
        for name, feats, kind in (("learned_cheap_logreg", CHEAP, "logreg"),
                                  ("learned_full_logreg", FULL, "logreg"),
                                  ("learned_cheap_gbm", CHEAP, "gbm")):
            picks, thetas = [], []
            for tr, te in GroupKFold(n_splits=5).split(d, d.y, d.id):
                train, test = d.iloc[tr].copy(), d.iloc[te].copy()
                model, th = fit_policy(train, feats, kind, lam)
                p = model.predict_proba(test[feats])[:, 1]
                picks.append(run_policy(test, p >= th))
                thetas.append(th)
            pk = pd.concat(picks)
            picks_all[name] = pk
            res[name] = {**score(pk, lam, feats is FULL), "theta_per_fold": thetas}
        judge = run_policy(d, d.verdict == 1)
        fixed1 = run_policy(d, d.t == 1)
        res["paired_f1_vs_judge_stop"] = {k: paired(judge, v) for k, v in picks_all.items()}
        res["paired_f1_vs_fixed_t1"] = {k: paired(fixed1, v) for k, v in picks_all.items()}
        res["paired_f1_vs_fixed_t1"]["judge_stop"] = paired(fixed1, judge)
        # retrievals are counts, not percentages: report raw difference
        res["paired_retrievals_vs_judge_stop"] = {
            k: {kk: (vv / 100 if kk == "mean" else [x / 100 for x in vv] if kk == "ci95" else vv)
                for kk, vv in paired(judge, v, col="t").items()} for k, v in picks_all.items()}
        out["by_lambda"][str(lam)] = res

    # deployable policy for Step 5: cheap features, primary lambda, fit on ALL questions
    d = add_labels(base, PRIMARY_LAMBDA)
    model, th = fit_policy(d, CHEAP, "logreg", PRIMARY_LAMBDA)
    sc, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    out["deploy_policy"] = {"features": CHEAP, "lambda": PRIMARY_LAMBDA, "theta": th,
                            "mean": sc.mean_.round(6).tolist(), "scale": sc.scale_.round(6).tolist(),
                            "coef": lr.coef_[0].round(6).tolist(),
                            "intercept": round(float(lr.intercept_[0]), 6)}
    out["deploy_policy"]["coef_by_feature"] = dict(zip(CHEAP, out["deploy_policy"]["coef"]))

    with open("results/step4_policy.json", "w") as f:
        json.dump(out, f, indent=2)
    lam = str(PRIMARY_LAMBDA)
    print(f"lambda={lam}  (F1 / EM / extra retrievals / LLM calls / reward)")
    for k, v in out["by_lambda"][lam].items():
        if not k.startswith("paired"):
            print(f"  {k:22s} {v['F1']:6.2f} {v['EM']:6.2f} {v['avg_extra_retrievals']:6.3f} "
                  f"{v['llm_calls']:6.3f} {v['reward']:7.4f}")
    for key in ("paired_f1_vs_judge_stop", "paired_f1_vs_fixed_t1", "paired_retrievals_vs_judge_stop"):
        print(key, json.dumps(out["by_lambda"][lam][key]))
    print("saved results/step4_policy.json")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/step4_policy.py results/step4_n400_steps.csv")
    main(sys.argv[1])
