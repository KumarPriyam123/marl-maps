# =============================================================================
# MARL-MAPS · STEP 6c — FULL-SYSTEM TEST: trained 1.5B multi-agent pipeline
#
# Two pipelines, identical except for the trained agents:
#   1.5B_base : judge / query / answer = Qwen2.5-1.5B-Instruct, untrained
#   1.5B_grpo : judge = untrained 1.5B, query = Step 6b LoRA, answer = Step 6a LoRA
# Both use the Step 3-5 orchestration (surgical retrieval + duplicate-query guard) and
# run to the FULL budget (t = 0..3) so any stop policy can be scored exactly offline,
# as in Steps 4-5. Trajectories are collected on:
#   questions 400-799 -> train each pipeline's own stop policy (CPU, scripts/step6c_eval.py)
#   questions 800-999 -> the held-out test, same questions as Step 5 (7B reference)
# Generation is batched across questions (all questions advance one step together).
# Kaggle: GPU T4 x2 | Internet ON | secret GITHUB_TOKEN attached.
# Smoke test: SMOKE = True (8+8 questions) -> checks PASS -> SMOKE = False (~45 min).
# Split at the "# %% CELL" markers to make notebook cells.
# =============================================================================


# %% CELL 1 — installs (once per fresh session) ===============================
import subprocess, sys
pkgs = ["peft", "faiss-cpu", "sentence-transformers", "datasets"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
print("installed:", pkgs, "| removed: torchao")


# %% CELL 2 — config ===========================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import re, json, time, math, string, random, collections
import numpy as np, pandas as pd, faiss

SMOKE = True
RUN_TAG = "step6c"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"
ADAPTERS = {"query": "adapters/step6b/query", "answer": "adapters/step6a"}
PIPELINES = {"1.5B_base": {"query": None, "answer": None},
             "1.5B_grpo": {"query": "query", "answer": "answer"}}
SLICES = {"train": (400, 8 if SMOKE else 400), "test": (800, 8 if SMOKE else 200)}
CORPUS_QUESTIONS = 1000
TOP_K, SEARCH_DEPTH, MAX_ROUNDS = 3, 20, 3
DUP_QUERY_SIM = 0.92
MAX_PASSAGE_WORDS = 250
BATCH = 16
PARQUET = ("hf://datasets/hotpotqa/hotpot_qa@refs/convert/parquet/"
           "distractor/validation/0000.parquet")
REPO = "KumarPriyam123/marl-maps"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
TAG = f"{RUN_TAG}{'_smoke' if SMOKE else ''}"
print("SMOKE =", SMOKE, "| slices:", SLICES)


# %% CELL 3 — validation corpus + questions (identical to Steps 1-5) ===========
from datasets import load_dataset
ds = load_dataset("parquet", data_files=PARQUET, split="train")
rows = ds.select(range(CORPUS_QUESTIONS)).to_list()
passages, keys, qrows = [], set(), []
for ex in rows:
    ctx = ex["context"]
    for title, sents in zip(list(ctx["title"]), list(ctx["sentences"])):
        text = " ".join(str(s) for s in list(sents)).strip()
        if text and (str(title), text) not in keys:
            keys.add((str(title), text)); passages.append({"title": str(title), "text": text})
    qrows.append({"id": str(ex["id"]), "question": str(ex["question"]), "answer": str(ex["answer"]),
                  "qtype": str(ex["type"]),
                  "gold_titles": sorted({str(t) for t in list(ex["supporting_facts"]["title"])})})
queries = pd.DataFrame(qrows)
del rows, ds
print("passages:", len(passages), "| questions:", len(queries))


# %% CELL 4 — retriever + adapters from GitHub =================================
import base64, requests, torch
from sentence_transformers import SentenceTransformer
from kaggle_secrets import UserSecretsClient

embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

def embed(texts):
    return embedder.encode(texts, batch_size=128, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype("float32")

E = embed([f"{p['title']}: {p['text']}" for p in passages])
index = faiss.IndexFlatIP(E.shape[1]); index.add(E)

def search_unseen(qvecs, seens, k=TOP_K):
    scores, idx = index.search(qvecs, SEARCH_DEPTH)
    out = []
    for sc, ix, seen in zip(scores, idx, seens):
        hits = [(int(i), float(s)) for s, i in zip(sc, ix) if i >= 0 and int(i) not in seen]
        out.append(hits[:k])
    return out

token = UserSecretsClient().get_secret("GITHUB_TOKEN")      # never print this
GH = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
RAW = {**GH, "Accept": "application/vnd.github.raw+json"}
for name, path in ADAPTERS.items():
    os.makedirs(f"{OUT}/adapters/{name}", exist_ok=True)
    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        r = requests.get(f"https://api.github.com/repos/{REPO}/contents/{path}/{fn}", headers=RAW)
        assert r.status_code == 200, f"could not fetch {path}/{fn}: {r.status_code}"
        open(f"{OUT}/adapters/{name}/{fn}", "wb").write(r.content)
print("index:", index.ntotal, "| adapters fetched:", list(ADAPTERS))


# %% CELL 5 — model with both LoRAs; base = adapters disabled ==================
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from contextlib import nullcontext

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
DT = torch.float16 if AMP else torch.float32
try:
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DT)
except TypeError:
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DT)
base.to(DEVICE)
model = PeftModel.from_pretrained(base, f"{OUT}/adapters/query", adapter_name="query")
model.load_adapter(f"{OUT}/adapters/answer", adapter_name="answer")
model.eval()
SUFF_ID = tok.encode("Sufficient", add_special_tokens=False)[0]
INSUF_ID = tok.encode("Insufficient", add_special_tokens=False)[0]
assert SUFF_ID != INSUF_ID

@torch.no_grad()
def generate(prompts, adapter, max_new, scores=False):
    """Greedy batch generation with one adapter active (None = untrained base)."""
    ctx = model.disable_adapter() if adapter is None else nullcontext()
    if adapter is not None:
        model.set_adapter(adapter)
    with ctx, torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
        enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                             output_scores=scores, return_dict_in_generate=True,
                             pad_token_id=tok.pad_token_id)
    gen = out.sequences[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    return texts, gen, (out.scores if scores else None)

print("model ready on", DEVICE)


# %% CELL 6 — MAPS prompts + batched agents ====================================
QUERY_SYS = ("You are a retrieval query generation agent.\n"
    "Generate exactly one NEW QUERY that helps answer the QUESTION or fills gaps "
    "not covered by EXISTING QUERIES and RETRIEVED DOCUMENTS, and differs from all "
    "existing queries.\nOutput rules: keywords only; no stopwords or connectors; "
    "max 8 keywords separated by single spaces. Output only the query text.")
JUDGE_SYS = ("Determine whether the EXISTING QUERIES and RETRIEVED DOCUMENTS provide "
    "sufficient information to answer the QUESTION correctly. Output only one of: "
    "'Sufficient Information' or 'Insufficient Information'. No explanations.")
ANSWER_SYS = ("Answer the question based on the given document. Output only the final "
    "answer with no explanations or additional text.\nQuery Pool:\n{pool}")

def clip(text, n=MAX_PASSAGE_WORDS):
    w = text.split()
    return text if len(w) <= n else " ".join(w[:n]) + " ..."

def format_pool(pool):
    lines = []
    for qi, e in enumerate(pool):
        lines.append(f"Query {qi}: {e['query']}")
        lines.append(f"Retrieved Chunks for Query {qi}:")
        for ci, c in enumerate(e["chunks"], 1):
            lines.append(f"({ci}) Title: {c['title']}; Content: {clip(c['text'])}")
    return "\n".join(lines)

def chat(system, user):
    return tok.apply_chat_template([{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)

def batched(fn, items, *a):
    out = []
    for b in range(0, len(items), BATCH):
        out += fn(items[b:b + BATCH], *a)
    return out

def judge_batch(states, adapter):
    texts, gen, sc = generate([chat(JUDGE_SYS, f"Question: {s['question']}\nQuery Pool:\n"
                                    f"{format_pool(s['pool'])}") for s in states],
                              adapter, 8, scores=True)
    res = []
    for i, tx in enumerate(texts):
        t = tx.lower()
        verdict = False if "insufficient" in t else (True if "sufficient" in t else None)
        conf = None
        for j in range(min(3, gen.shape[1])):
            if int(gen[i, j]) in (SUFF_ID, INSUF_ID):
                lp = torch.log_softmax(sc[j][i].float(), -1)
                d = float(lp[SUFF_ID] - lp[INSUF_ID])
                conf = 1 / (1 + math.exp(-max(min(d, 50), -50)))
                break
        res.append((tx.strip(), verdict, conf))
    return res

def answer_batch(states, adapter):
    texts, _, _ = generate([chat(ANSWER_SYS.format(pool=format_pool(s["pool"])),
                                 f"Question: {s['question']}") for s in states], adapter, 24)
    return [next((ln.strip() for ln in tx.splitlines() if ln.strip()), "") for tx in texts]

def query_batch(states, adapter, extra=None):
    prompts = []
    for i, s in enumerate(states):
        pool = s["pool"] + ([{"query": extra[i], "chunks": []}] if extra and extra[i] else [])
        prompts.append(chat(QUERY_SYS, f"Question: {s['question']}\nQuery Pool:\n{format_pool(pool)}"))
    texts, _, _ = generate(prompts, adapter, 20)
    out = []
    for s, tx in zip(states, texts):
        kws = re.findall(r"[A-Za-z0-9']+", tx)
        out.append(" ".join(kws[:8]) or s["question"])
    return out

def normalize(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def em(p, g): return float(normalize(p) == normalize(g))

def f1(p, g):
    pt, gt = normalize(p).split(), normalize(g).split()
    if not pt or not gt:
        return float(pt == gt)
    common = collections.Counter(pt) & collections.Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pr, rc = n / len(pt), n / len(gt)
    return 2 * pr * rc / (pr + rc)


# %% CELL 7 — collect full trajectories, batched (checkpointed per slice) =======
def collect(pipe, start, n):
    ad = PIPELINES[pipe]
    sub = queries.iloc[start:start + n].to_dict("records")
    qv = embed([q["question"] for q in sub])
    first = search_unseen(qv, [set()] * len(sub))
    S = []
    for q, v, hits in zip(sub, qv, first):
        S.append({**q, "seen": {i for i, _ in hits}, "titles": {passages[i]["title"] for i, _ in hits},
                  "pool": [{"query": q["question"], "chunks": [passages[i] for i, _ in hits]}],
                  "psi": [v], "last": {"query": q["question"], "n_new": len(hits), "dup": False,
                                       "qsim": 0.0, "scores": [s for _, s in hits]},
                  "prev_pred": None})
    rows = []
    for t in range(MAX_ROUNDS + 1):
        judged = batched(judge_batch, S, None)        # judge = untrained 1.5B in both pipelines
        preds = batched(answer_batch, S, ad["answer"])
        for s, (raw, verdict, conf), pred in zip(S, judged, preds):
            npred = normalize(pred)
            sc = s["last"]["scores"] or [0.0]
            pool_text = normalize(" ".join(c["title"] + " " + c["text"]
                                           for e in s["pool"] for c in e["chunks"]))
            rows.append({"pipeline": pipe, "id": s["id"], "qtype": s["qtype"],
                         "question": s["question"], "gold": s["answer"], "t": t,
                         "query": s["last"]["query"], "judge_raw": raw, "verdict": verdict,
                         "conf": conf, "n_new_last": s["last"]["n_new"],
                         "top_score_last": max(sc), "mean_score_last": float(np.mean(sc)),
                         "qsim_last": s["last"]["qsim"], "dup_last": s["last"]["dup"],
                         "pool_size": len(s["seen"]), "pred": pred,
                         "pred_words": len(npred.split()),
                         "pred_changed": None if s["prev_pred"] is None else float(npred != s["prev_pred"]),
                         "pred_in_pool": float(bool(npred) and npred in pool_text),
                         "em": em(pred, s["answer"]), "f1": round(f1(pred, s["answer"]), 4),
                         "gold_covered": all(g in s["titles"] for g in s["gold_titles"]),
                         "gold_any": any(g in s["titles"] for g in s["gold_titles"])})
            s["prev_pred"] = npred
        if t == MAX_ROUNDS:
            break
        nq = batched(query_batch, S, ad["query"])
        v = embed(nq)
        dup = [any(float(v[i] @ p) >= DUP_QUERY_SIM for p in s["psi"]) for i, s in enumerate(S)]
        if any(dup):                                   # duplicate guard: one retry
            idx = [i for i, d in enumerate(dup) if d]
            retry = batched(query_batch, [S[i] for i in idx], ad["query"],
                            [nq[i] + "  (DUPLICATE - must differ)" for i in idx])
            for i, q2 in zip(idx, retry):
                nq[i] = q2
            v = embed(nq)
            dup = [any(float(v[i] @ p) >= DUP_QUERY_SIM for p in s["psi"]) for i, s in enumerate(S)]
        hits = search_unseen(v, [s["seen"] for s in S])
        for s, q, vec, d, h in zip(S, nq, v, dup, hits):
            qsim = max(float(vec @ p) for p in s["psi"])
            s["psi"].append(vec)
            s["seen"] |= {i for i, _ in h}
            s["titles"] |= {passages[i]["title"] for i, _ in h}
            s["pool"].append({"query": q, "chunks": [passages[i] for i, _ in h]})
            s["last"] = {"query": q, "n_new": len(h), "dup": d, "qsim": qsim,
                         "scores": [sc for _, sc in h]}
        print(f"    {pipe} t={t + 1} done")
    return pd.DataFrame(rows)

DATA = {}
t_run = time.time()
for pipe in PIPELINES:
    for sl, (start, n) in SLICES.items():
        path = f"{OUT}/{TAG}_{pipe}_{sl}.pkl"
        if os.path.exists(path):
            DATA[(pipe, sl)] = pd.read_pickle(path); print("loaded", path); continue
        t0 = time.time()
        DATA[(pipe, sl)] = collect(pipe, start, n)
        DATA[(pipe, sl)].to_pickle(path)
        d = DATA[(pipe, sl)]
        print(f"  {pipe} {sl}: {d.id.nunique()} questions | F1 by t: " +
              " ".join(f"t{t}={100 * g.f1.mean():.1f}" for t, g in d.groupby("t")) +
              f" | {time.time() - t0:.0f}s")
print(f"collection done in {time.time() - t_run:.0f}s")


# %% CELL 8 — quick offline view, checks, save =================================
def first_stop(d, rule):
    picks = []
    for _, g in d.groupby("id"):
        g = g.sort_values("t")
        hit = g[g.apply(rule, axis=1) | (g.t == MAX_ROUNDS)]
        picks.append(hit.iloc[0])
    p = pd.DataFrame(picks)
    return {"F1": round(100 * p.f1.mean(), 2), "EM": round(100 * p.em.mean(), 2),
            "avg_extra_searches": round(p.t.mean(), 3)}

summary = {"meta": {"run_tag": RUN_TAG, "smoke": SMOKE, "model": MODEL, "adapters": ADAPTERS,
                    "slices": SLICES, "judge": "untrained 1.5B in both pipelines"}, "results": {}}
for (pipe, sl), d in DATA.items():
    summary["results"][f"{pipe}/{sl}"] = {
        "by_t_F1": {int(t): round(100 * g.f1.mean(), 2) for t, g in d.groupby("t")},
        "by_t_gold_covered": {int(t): round(100 * g.gold_covered.mean(), 2) for t, g in d.groupby("t")},
        "judge_said_sufficient_pct": round(100 * (d.verdict == True).mean(), 2),
        "judge_stop": first_stop(d, lambda r: r.verdict == True),
        "fixed_t1": first_stop(d, lambda r: r.t == 1),
        "always_full": first_stop(d, lambda r: False)}
allr = pd.concat(DATA.values())
checks = {
    "all questions have 4 steps": bool((allr.groupby(["pipeline", "id"]).t.count() == MAX_ROUNDS + 1).all()),
    "no empty answers (<5%)": (allr.pred.str.strip() == "").mean() < 0.05,
    "judge parseable (>=90%)": allr.verdict.notna().mean() >= 0.90,
    "judge confidence captured (>=80%)": allr.conf.notna().mean() >= 0.80,
    "trained pipeline differs from base": bool(
        (DATA[("1.5B_grpo", "test")].pred.values != DATA[("1.5B_base", "test")].pred.values).any()),
}
checks = {k: bool(v) for k, v in checks.items()}
summary["meta"]["checks"] = checks
for k, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {k}")
for (pipe, sl), d in DATA.items():
    d.to_csv(f"{OUT}/{TAG}_{pipe}_{sl}_steps.csv", index=False)
with open(f"{OUT}/{TAG}_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n----- PASTE THIS BACK -----")
print(json.dumps(summary))
print("---------------------------")


# %% CELL 9 — push to GitHub ===================================================
def push(local, remote):
    url = f"https://api.github.com/repos/{REPO}/contents/{remote}"
    old = requests.get(url, headers=GH)
    body = {"message": f"{TAG}: {remote}",
            "content": base64.b64encode(open(local, "rb").read()).decode()}
    if old.status_code == 200:
        body["sha"] = old.json()["sha"]
    r = requests.put(url, headers=GH, json=body)
    print(remote, "->", r.status_code)
    assert r.status_code in (200, 201), r.text[:300]

push(f"{OUT}/{TAG}_summary.json", f"results/{TAG}_summary.json")
for (pipe, sl) in DATA:
    fn = f"{TAG}_{pipe}_{sl}_steps.csv"
    push(f"{OUT}/{fn}", f"results/{fn}")
print(f"pushed to github.com/{REPO}")
