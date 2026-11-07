# =============================================================================
# MARL-MAPS · STEP 1 — BUILD CORPUS + MEASURED BASELINES (Kaggle, self-contained)
#
# Needs nothing from your laptop. In one notebook it:
#   1. downloads HotpotQA (distractor, validation) from Hugging Face
#   2. builds the passage corpus + FAISS index from the first 1,000 questions
#   3. runs two systems on the same questions, retriever and LLM:
#        A) NAIVE RAG      : retrieve top-k once -> answer
#        B) MAPS-SCRIPTED  : retrieve -> judge -> (query-extend -> retrieve)* -> answer
#                            (base paper Eq. 8-11, prompt-only, no training)
#   4. checks the run is valid and saves everything to /kaggle/working
#
# Every later MARL-MAPS number is compared against B. Every number printed
# here is measured on your run; nothing is projected.
#
# Kaggle settings (right panel):  Accelerator = GPU T4 x1   |   Internet = ON
# Run order:
#   1. CELL 1 once per fresh session (installs).
#   2. CELL 2..9 with N_QUESTIONS = 10  -> smoke test.
#   3. All 6 checks PASS -> N_QUESTIONS = 200, re-run CELL 8 and CELL 9 only.
#   4. Click "Save Version" (Quick Save) so the outputs are kept.
# Split at the "# %% CELL" markers to make notebook cells.
# =============================================================================


# %% CELL 1 — installs (once per fresh session) ===============================
BACKEND = "vllm"   # "vllm" (fast) | "hf" (fallback: transformers + 4-bit). Set the same in CELL 2.

import subprocess, sys
pkgs = ["faiss-cpu", "sentence-transformers", "datasets"]
pkgs += ["vllm"] if BACKEND == "vllm" else ["bitsandbytes", "accelerate"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
print("installed:", pkgs)
# pip "dependency resolver" warnings here are normal. If CELL 5 then fails to
# import torch/vllm: Run -> Restart session, skip CELL 1, run from CELL 2.


# %% CELL 2 — config ===========================================================
import os, re, json, time, math, string, collections
import numpy as np, pandas as pd, faiss

BACKEND     = "vllm"   # must match CELL 1 ("vllm" or "hf")
MODEL_VLLM  = "Qwen/Qwen2.5-7B-Instruct-AWQ"
MODEL_HF    = "Qwen/Qwen2.5-7B-Instruct"       # loaded in 4-bit for the fallback
EMBED_MODEL = "all-MiniLM-L6-v2"               # 384-d, cosine via normalized vectors
CORPUS_QUESTIONS = 1000   # corpus = all context paragraphs of these questions
N_QUESTIONS = 10          # smoke test first; then 200
START_AT    = 0           # which slice of the CORPUS_QUESTIONS to evaluate
TOP_K       = 3
MAX_ROUNDS  = 3           # max judge calls; max extra retrievals = MAX_ROUNDS
MAX_PASSAGE_WORDS = 250   # keeps the longest prompt well under max_model_len
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

PARQUET_HF   = ("hf://datasets/hotpotqa/hotpot_qa@refs/convert/parquet/"
                "distractor/validation/0000.parquet")
PARQUET_HTTP = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
                "refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet")
assert START_AT + N_QUESTIONS <= CORPUS_QUESTIONS, "eval slice must lie inside the corpus questions"


# %% CELL 3 — download HotpotQA + build corpus and question set ================
def load_hotpot_rows(n):
    """First n rows of HotpotQA distractor/validation as plain dicts."""
    try:
        from datasets import load_dataset
        ds = load_dataset("parquet", data_files=PARQUET_HF, split="train")
        return ds.select(range(min(n, len(ds)))).to_list()
    except Exception as e:
        print("datasets loader failed, trying direct parquet URL:", repr(e)[:150])
        return pd.read_parquet(PARQUET_HTTP).head(n).to_dict("records")

rows = load_hotpot_rows(CORPUS_QUESTIONS)
ctx0 = rows[0]["context"]
assert isinstance(ctx0, dict) and "title" in ctx0 and "sentences" in ctx0, \
    f"unexpected context schema: {type(ctx0)}"

passages, seen, qrows = [], set(), []
for ex in rows:
    ctx = ex["context"]
    for title, sents in zip(list(ctx["title"]), list(ctx["sentences"])):
        text = " ".join(str(s) for s in list(sents)).strip()
        key = (str(title), text)
        if text and key not in seen:          # O(1) de-duplication
            seen.add(key)
            passages.append({"title": str(title), "text": text})
    qrows.append({"id": str(ex["id"]), "question": str(ex["question"]),
                  "answer": str(ex["answer"]), "level": str(ex["level"]),
                  "qtype": str(ex["type"]),
                  "gold_titles": sorted({str(t) for t in list(ex["supporting_facts"]["title"])})})
queries = pd.DataFrame(qrows)
HAVE_GOLD = True

corpus_titles = {p["title"] for p in passages}
in_corpus = queries.gold_titles.map(lambda ts: all(t in corpus_titles for t in ts)).mean()
print(f"questions={len(queries)} | unique passages={len(passages)} | "
      f"gold titles in corpus: {100*in_corpus:.1f}%")
assert in_corpus == 1.0, "every gold passage must be in the corpus"
os.makedirs(f"{OUT}/data", exist_ok=True)
pd.DataFrame(passages).to_csv(f"{OUT}/data/passages.csv", index=False)
queries.assign(gold_titles=queries.gold_titles.map(json.dumps)) \
       .to_csv(f"{OUT}/data/queries.csv", index=False)


# %% CELL 4 — embed corpus, build FAISS index, retriever ======================
from sentence_transformers import SentenceTransformer

# CPU on purpose: the LLM takes most of the T4's memory (takes ~1-3 min here)
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
t0 = time.time()
emb = embedder.encode([f"{p['title']}: {p['text']}" for p in passages], batch_size=128,
                      normalize_embeddings=True, convert_to_numpy=True,
                      show_progress_bar=True).astype("float32")
index = faiss.IndexFlatIP(emb.shape[1])     # inner product on unit vectors == cosine
index.add(emb)
faiss.write_index(index, f"{OUT}/data/index.faiss")
assert index.ntotal == len(passages), "index and passages are misaligned"
print(f"index: {index.ntotal} x {emb.shape[1]} built in {time.time()-t0:.0f}s")

class Retriever:
    def __init__(self, index, meta, embedder):
        self.index, self.meta, self.embedder = index, meta, embedder

    def retrieve(self, query, k=TOP_K):
        qv = self.embedder.encode([query], normalize_embeddings=True,
                                  convert_to_numpy=True).astype("float32")
        scores, idx = self.index.search(qv, k)
        return [{"pid": int(i), "title": self.meta[i]["title"],
                 "text": self.meta[i]["text"], "score": float(s)}
                for s, i in zip(scores[0], idx[0]) if i >= 0]

retriever = Retriever(index, passages, embedder)
# self-test: the first question's gold titles should show up in its top results
_q = queries.iloc[0]
_hits = retriever.retrieve(_q.question, 5)
print("self-test Q:", _q.question)
print("  gold:", _q.gold_titles, "| top-5:", [h["title"] for h in _hits])


# %% CELL 5 — LLM backend (vLLM or HF 4-bit, same interface) ===================
# Interface used by the agents:
#   generate(prompt, system, max_tokens)  -> str
#   judge_generate(prompt, system)        -> (str, conf)
# conf = P("Sufficient") / (P("Sufficient") + P("Insufficient")) on the first
# decision token. Training-free proxy only (Self-REF trains special tokens; this does not).
import torch
from transformers import AutoTokenizer

MODEL = MODEL_VLLM if BACKEND == "vllm" else MODEL_HF
tok = AutoTokenizer.from_pretrained(MODEL)

SUFF_ID  = tok.encode("Sufficient",   add_special_tokens=False)[0]
INSUF_ID = tok.encode("Insufficient", add_special_tokens=False)[0]
assert SUFF_ID != INSUF_ID, "decision tokens collide; confidence would be meaningless"

def chat_text(prompt, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def conf_from(ls, li):
    """sigmoid(z_S - z_I); None if neither token was seen."""
    if ls is None and li is None:
        return None
    ls = -1e9 if ls is None else ls
    li = -1e9 if li is None else li
    d = max(min(ls - li, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-d))

LLM_ERRORS = collections.Counter()

if BACKEND == "vllm":
    from vllm import LLM, SamplingParams
    engine = LLM(model=MODEL, dtype="float16", max_model_len=8192,
                 gpu_memory_utilization=0.85, enforce_eager=True, seed=0)

    def generate(prompt, system=None, max_tokens=64):
        out = engine.generate([chat_text(prompt, system)],
                              SamplingParams(temperature=0.0, max_tokens=max_tokens),
                              use_tqdm=False)
        return out[0].outputs[0].text.strip()

    def judge_generate(prompt, system=None):
        o = engine.generate([chat_text(prompt, system)],
                            SamplingParams(temperature=0.0, max_tokens=8, logprobs=20),
                            use_tqdm=False)[0].outputs[0]
        conf = None
        # first generated position whose token is the decision word (skips a leading quote)
        for i, pos in enumerate((o.logprobs or [])[:3]):
            if o.token_ids[i] in (SUFF_ID, INSUF_ID):
                lp = lambda t: getattr(pos[t], "logprob", pos[t])
                floor = min(getattr(v, "logprob", v) for v in pos.values())  # bound for a token outside top-20
                ls = lp(SUFF_ID) if SUFF_ID in pos else floor
                li = lp(INSUF_ID) if INSUF_ID in pos else floor
                conf = conf_from(ls, li)
                break
        return o.text.strip(), conf

else:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                    device_map="cuda:0")
    hf_model.eval()

    @torch.no_grad()
    def _hf(prompt, system, max_tokens, scores=False):
        ids = tok(chat_text(prompt, system), return_tensors="pt").to("cuda:0")
        out = hf_model.generate(**ids, max_new_tokens=max_tokens, do_sample=False,
                                output_scores=scores, return_dict_in_generate=True,
                                pad_token_id=tok.eos_token_id)
        text = tok.decode(out.sequences[0, ids["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        return text, (out.scores if scores else None)

    def generate(prompt, system=None, max_tokens=64):
        return _hf(prompt, system, max_tokens)[0]

    def judge_generate(prompt, system=None):
        text, sc = _hf(prompt, system, 8, scores=True)
        conf = None
        for step in (sc or [])[:3]:
            lp = torch.log_softmax(step[0].float(), dim=-1)
            top = int(lp.argmax())
            if top in (SUFF_ID, INSUF_ID):
                conf = conf_from(float(lp[SUFF_ID]), float(lp[INSUF_ID]))
                break
        return text, conf

print("backend:", BACKEND, "| model:", MODEL)
print("smoke:", repr(generate("What is the capital of France? Answer in one word.")))


# %% CELL 6 — MAPS prompts + the three agents =================================
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

def parse_judge(text):
    """True = sufficient, False = insufficient, None = unparseable (counted)."""
    t = text.lower()
    if "insufficient" in t:
        return False
    if "sufficient" in t:
        return True
    return None

def query_agent(q, pool):
    out = generate(f"Question: {q}\nQuery Pool:\n{format_pool(pool)}", QUERY_SYS, 32)
    kws = re.findall(r"[A-Za-z0-9']+", out)
    return " ".join(kws[:8]) or q

def judge_agent(q, pool):
    return judge_generate(f"Question: {q}\nQuery Pool:\n{format_pool(pool)}", JUDGE_SYS)

def answer_agent(q, pool):
    out = generate(f"Question: {q}", ANSWER_SYS.format(pool=format_pool(pool)), 64)
    first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    return first


# %% CELL 7 — the two systems, with full traces ===============================
def covered(titles, gold_titles):
    """All gold supporting titles present in the pool? None if no gold labels."""
    if not gold_titles:
        return None
    return all(g in titles for g in gold_titles)

def any_gold(titles, gold_titles):
    """At least one gold supporting title in the pool? None if no gold labels."""
    if not gold_titles:
        return None
    return any(g in titles for g in gold_titles)

def naive_rag(q, gold_titles=None):
    t0 = time.time()
    chunks = retriever.retrieve(q, TOP_K)
    pool = [{"query": q, "chunks": chunks}]
    pred = answer_agent(q, pool)
    return {"pred": pred, "n_retrievals": 1, "n_redundant": 0, "n_post_coverage": 0,
            "gold_covered": covered({c["title"] for c in chunks}, gold_titles),
            "gold_any": any_gold({c["title"] for c in chunks}, gold_titles),
            "queries": [q], "judge_calls": [], "latency_s": time.time() - t0}

def maps_scripted(q, gold_titles=None):
    t0 = time.time()
    first = retriever.retrieve(q, TOP_K)                          # Eq. 8
    pool = [{"query": q, "chunks": first}]
    seen = {c["pid"] for c in first}
    titles = {c["title"] for c in first}
    n_ret, n_redundant, n_post_cov, judge_calls = 1, 0, 0, []
    for r in range(MAX_ROUNDS):
        raw, conf = judge_agent(q, pool)                          # Eq. 10
        verdict = parse_judge(raw)
        judge_calls.append({"round": r, "raw": raw, "verdict": verdict, "conf": conf,
                            "gold_covered_now": covered(titles, gold_titles)})
        if verdict is True:
            break
        cov_before = covered(titles, gold_titles)
        nq = query_agent(q, pool)                                 # Eq. 9
        chunks = retriever.retrieve(nq, TOP_K)
        new = [c for c in chunks if c["pid"] not in seen]
        n_ret += 1
        n_redundant += int(len(new) == 0)       # retrieval added nothing new
        n_post_cov  += int(cov_before is True)  # retrieved although gold was already in the pool
        seen |= {c["pid"] for c in chunks}
        titles |= {c["title"] for c in chunks}
        pool.append({"query": nq, "chunks": chunks})
    pred = answer_agent(q, pool)                                  # Eq. 11
    return {"pred": pred, "n_retrievals": n_ret, "n_redundant": n_redundant,
            "n_post_coverage": n_post_cov, "gold_covered": covered(titles, gold_titles),
            "gold_any": any_gold(titles, gold_titles),
            "queries": [e["query"] for e in pool], "judge_calls": judge_calls,
            "latency_s": time.time() - t0}

SYSTEMS = {"naive_rag": naive_rag, "maps_scripted": maps_scripted}


# %% CELL 8 — run both systems + compute metrics ===============================
def normalize(s):
    s = s.lower()
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

def ece(confs, labels, bins=10):
    """Expected calibration error of conf vs binary label."""
    c, y = np.asarray(confs, float), np.asarray(labels, float)
    if len(c) == 0:
        return None
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c >= lo) & ((c < hi) if hi < 1 else (c <= hi))
        if m.any():
            total += m.mean() * abs(c[m].mean() - y[m].mean())
    return float(total)

def run_system(name, fn, subset):
    q_rows, j_rows = [], []
    for k, row in enumerate(subset.itertuples(index=False), 1):
        gold_titles = row.gold_titles if HAVE_GOLD else None
        try:
            res, err = fn(row.question, gold_titles), ""
        except Exception as e:                      # never let one question kill a long run
            res = {"pred": "", "n_retrievals": 0, "n_redundant": 0, "n_post_coverage": 0,
                   "gold_covered": None, "gold_any": None, "queries": [], "judge_calls": [],
                   "latency_s": 0.0}
            err = repr(e)[:300]
        q_rows.append({
            "system": name, "id": row.id, "qtype": row.qtype, "level": getattr(row, "level", ""),
            "question": row.question, "gold": row.answer, "pred": res["pred"],
            "em": em(res["pred"], row.answer), "f1": round(f1(res["pred"], row.answer), 4),
            "n_retrievals": res["n_retrievals"], "n_redundant": res["n_redundant"],
            "n_post_coverage": res["n_post_coverage"], "gold_covered": res["gold_covered"],
            "gold_any": res["gold_any"],
            "n_judge_calls": len(res["judge_calls"]), "latency_s": round(res["latency_s"], 3),
            "queries": " || ".join(res["queries"]), "error": err})
        for jc in res["judge_calls"]:
            j_rows.append({"system": name, "id": row.id, **jc})
        if k % 10 == 0 or k == len(subset):
            d = pd.DataFrame(q_rows)
            print(f"  [{name}] {k}/{len(subset)} | EM={100*d.em.mean():.1f} "
                  f"F1={100*d.f1.mean():.1f} | avg retrievals={d.n_retrievals.mean():.2f}")
    return pd.DataFrame(q_rows), pd.DataFrame(j_rows)

def summarize(qdf, jdf):
    # EM/F1 over ALL questions (a crash counts as wrong); cost stats over completed ones
    ok = qdf[qdf.error == ""]
    extra = (ok.n_retrievals - 1).clip(lower=0).sum()
    s = {
        "n": int(len(qdf)),
        "EM": round(100 * qdf.em.mean(), 2),
        "F1": round(100 * qdf.f1.mean(), 2),
        "empty_pred_rate": round(100 * (qdf.pred.str.strip() == "").mean(), 2),
        "error_rate": round(100 * (qdf.error != "").mean(), 2),
        "avg_retrievals": round(ok.n_retrievals.mean(), 3),
        "redundant_retrieval_pct": round(100 * ok.n_redundant.sum() / extra, 2) if extra else 0.0,
        "post_coverage_retrieval_pct": (round(100 * ok.n_post_coverage.sum() / extra, 2)
                                        if extra and HAVE_GOLD else None),
        "gold_coverage_pct": (round(100 * ok.gold_covered.astype(float).mean(), 2)
                              if HAVE_GOLD else None),   # ALL gold passages in the pool
        "gold_any_pct": (round(100 * ok.gold_any.astype(float).mean(), 2)
                         if HAVE_GOLD else None),        # at least one gold passage
        "latency_mean_s": round(ok.latency_s.mean(), 3),
        "latency_p50_s": round(ok.latency_s.quantile(0.5), 3),
        "latency_p90_s": round(ok.latency_s.quantile(0.9), 3),
        "by_type": {t: {"n": int(len(g)), "EM": round(100 * g.em.mean(), 2),
                        "F1": round(100 * g.f1.mean(), 2)}
                    for t, g in qdf.groupby("qtype")},
    }
    if len(jdf):
        parsed = jdf[jdf.verdict.notna()]
        s["judge"] = {
            "calls": int(len(jdf)),
            "unparseable_pct": round(100 * jdf.verdict.isna().mean(), 2),
            "said_sufficient_pct": round(100 * (jdf.verdict == True).mean(), 2),
            "conf_missing_pct": round(100 * jdf.conf.isna().mean(), 2),
        }
        if HAVE_GOLD and len(parsed):
            lab = parsed.gold_covered_now.astype(bool)
            said = parsed.verdict.astype(bool)
            s["judge"].update({
                "agree_with_gold_coverage_pct": round(100 * (said == lab).mean(), 2),
                "false_sufficient_pct": round(100 * (said & ~lab).mean(), 2),
                "false_insufficient_pct": round(100 * (~said & lab).mean(), 2),
            })
            cj = jdf[jdf.conf.notna()]
            if len(cj):
                s["judge"]["ECE_conf_vs_gold_coverage"] = round(
                    ece(cj.conf, cj.gold_covered_now.astype(bool)), 4)
    return s

subset = queries.iloc[START_AT:START_AT + N_QUESTIONS].reset_index(drop=True)
all_q, all_j, summary = [], [], {}
t_run = time.time()
for name, fn in SYSTEMS.items():
    print(f"\n=== {name} on {len(subset)} questions ===")
    qdf, jdf = run_system(name, fn, subset)
    all_q.append(qdf); all_j.append(jdf)
    summary[name] = summarize(qdf, jdf)
qdf_all = pd.concat(all_q, ignore_index=True)
jdf_all = pd.concat(all_j, ignore_index=True) if any(len(j) for j in all_j) else pd.DataFrame()
print(f"\nrun time: {time.time() - t_run:.0f}s")


# %% CELL 9 — sanity checks, save, print the block to paste back ==============
m = summary["maps_scripted"]
checks = {
    "no empty predictions (<5%)":        all(summary[k]["empty_pred_rate"] < 5 for k in summary),
    "no crashed questions (0%)":         all(summary[k]["error_rate"] == 0 for k in summary),
    "judge parseable (>=95%)":           m.get("judge", {}).get("unparseable_pct", 100) <= 5,
    "judge sometimes asks for more":     m["avg_retrievals"] > 1.0,
    "judge confidence captured (>=80%)": m.get("judge", {}).get("conf_missing_pct", 100) <= 20,
    "retriever finds gold (>=50% any)":  (summary["naive_rag"]["gold_any_pct"] or 0) >= 50,
}
checks = {k: bool(v) for k, v in checks.items()}
for k, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {k}")

meta = {"backend": BACKEND, "model": MODEL, "embedder": EMBED_MODEL, "top_k": TOP_K,
        "max_rounds": MAX_ROUNDS, "start_at": START_AT, "n_questions": len(subset),
        "corpus_passages": int(index.ntotal), "checks": checks}
with open(f"{OUT}/step1_summary.json", "w") as f:
    json.dump({"meta": meta, "results": summary}, f, indent=2, default=str)
qdf_all.to_csv(f"{OUT}/step1_questions.csv", index=False)
if len(jdf_all):
    jdf_all.to_csv(f"{OUT}/step1_judge_calls.csv", index=False)

print("\nfirst 3 MAPS traces:")
for _, r in qdf_all[qdf_all.system == "maps_scripted"].head(3).iterrows():
    print(f"  Q: {r.question}\n    queries: {r.queries}\n    pred: {r.pred!r} | gold: {r.gold!r} | "
          f"retrievals={r.n_retrievals} covered={r.gold_covered}")

print("\n----- PASTE THIS BACK -----")
print(json.dumps({"meta": meta, "results": summary}, default=str))
print("---------------------------")
print(f"saved in {OUT}: step1_summary.json, step1_questions.csv, step1_judge_calls.csv, data/")
print("Now click Save Version (Quick Save) so these outputs are kept.")
