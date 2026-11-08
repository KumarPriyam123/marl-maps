# =============================================================================
# MARL-MAPS · STEP 4 — FULL TRAJECTORIES + PROCESS REWARDS (Kaggle, data collection)
#
# Goal: learn WHEN to stop searching. Step 3 showed the judge-based stop rule is
# the bottleneck (30% of searches happen after the gold is already in the pool,
# 22% of questions burn the full budget).
#
# For each question (400-799) this notebook runs the orchestrator to the FULL
# budget (t = 0..3 extra retrievals, surgical retrieval + duplicate guard) and at
# EVERY step t logs:
#   state  : SGWM features (judge verdict + calibrated confidence, new passages,
#            retrieval scores, query similarity, answer stability, ...)
#   outcome: the answer the answer agent would give if we stopped at t, its EM/F1,
#            and whether all gold passages are in the pool
# Because earlier steps never depend on later ones, these logs let us evaluate ANY
# stopping policy exactly, offline: stop at t -> answer_t, cost t retrievals.
# Process reward (HiPRAG-style efficiency penalty): R(t) = F1_t - lambda * t
#
# The policy itself is trained on CPU by scripts/step4_policy.py (no GPU).
# Kaggle: GPU T4 x2 | Internet ON | secret GITHUB_TOKEN attached.
# Smoke test N_QUESTIONS = 10 -> all checks PASS -> 400 -> Run All (about 45 min).
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
import os
# vLLM starts its engine in a child process. With the default "fork", any CUDA use
# in this notebook before CELL 5 breaks it ("CUDA error: initialization error").
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import re, json, time, math, string, collections
import numpy as np, pandas as pd, faiss

BACKEND     = "vllm"   # must match CELL 1 ("vllm" or "hf")
MODEL_VLLM  = "Qwen/Qwen2.5-7B-Instruct-AWQ"
MODEL_HF    = "Qwen/Qwen2.5-7B-Instruct"       # loaded in 4-bit for the fallback
EMBED_MODEL = "all-MiniLM-L6-v2"               # 384-d, cosine via normalized vectors
CORPUS_QUESTIONS = 1000   # corpus = all context paragraphs of these questions
N_QUESTIONS = 10          # smoke test first; then 400
START_AT    = 400         # Step 4 slice: questions 400-799 (TRAINING data for the policy)
RUN_TAG     = "step4"     # prefix of every output file
# Steps 1-3 used 0-399. Questions 800-999 stay untouched for the final test (Step 5).
# Orchestrator constants (fixed before the run, never tuned on this slice)
SEARCH_DEPTH = 20         # FAISS depth searched to find k UNSEEN passages
DUP_QUERY_SIM = 0.92      # cosine >= this to an earlier query = duplicate query
PLATT_A, PLATT_B = 0.10338, -0.40947   # Step 2: fitted on Step 1 judge calls (Q 0-199)
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

    def retrieve_unseen(self, query, k, seen, depth=SEARCH_DEPTH):
        """SURGICAL_MICRO_RETRIEVAL: best k passages NOT already in the pool."""
        hits = self.retrieve(query, depth)
        return [h for h in hits if h["pid"] not in seen][:k]

    def embed(self, text):
        return self.embedder.encode([text], normalize_embeddings=True,
                                    convert_to_numpy=True).astype("float32")[0]

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
    if "engine" not in globals():   # Run All twice in one session: reuse the loaded model
        engine = LLM(model=MODEL, dtype="float16", max_model_len=8192,
                     gpu_memory_utilization=0.85, enforce_eager=True, seed=0)

    def generate(prompt, system=None, max_tokens=64):
        out = engine.generate([chat_text(prompt, system)],
                              SamplingParams(temperature=0.0, max_tokens=max_tokens),
                              use_tqdm=False)
        return out[0].outputs[0].text.strip()

    def judge_generate(prompt, system=None):
        o = engine.generate([chat_text(prompt, system)],
                            SamplingParams(temperature=0.0, max_tokens=8,
                                           logprobs=20),
                            use_tqdm=False)[0].outputs[0]
        conf = None
        for i, pos in enumerate((o.logprobs or [])[:3]):
            if o.token_ids[i] not in (SUFF_ID, INSUF_ID):
                continue
            vals = {t: getattr(v, "logprob", v) for t, v in pos.items()}
            floor = min(vals.values())
            ls = vals.get(SUFF_ID, floor)
            li = vals.get(INSUF_ID, floor)
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

from dataclasses import dataclass, field

@dataclass
class SGWM:
    """Shared working memory, read/written by every agent (names from the proposal)."""
    q_emb: np.ndarray                                     # Q_emb: question embedding
    h_trace: list = field(default_factory=list)          # H_trace: (action, detail) log
    c_currency: list = field(default_factory=list)       # C_currency: calibrated judge conf
    omega_ctx: list = field(default_factory=list)        # Omega_ctx: query pool
    psi_query: list = field(default_factory=list)        # Psi_query: embeddings of queries
    gamma_ops: collections.Counter = field(default_factory=collections.Counter)  # op counts
    seen: set = field(default_factory=set)               # pids already in the pool
    titles: set = field(default_factory=set)

    def add(self, query, chunks):
        self.omega_ctx.append({"query": query, "chunks": chunks})
        self.seen |= {c["pid"] for c in chunks}
        self.titles |= {c["title"] for c in chunks}

def platt(conf):
    if conf is None:
        return None
    c = min(max(conf, 1e-6), 1 - 1e-6)
    return 1 / (1 + math.exp(-(PLATT_A * math.log(c / (1 - c)) + PLATT_B)))

def is_dup(m, qv):
    return any(float(qv @ p) >= DUP_QUERY_SIM for p in m.psi_query)

def orch_rules(q, gold_titles=None):
    t0 = time.time()
    m = SGWM(q_emb=retriever.embed(q))
    m.psi_query.append(m.q_emb)
    m.add(q, retriever.retrieve_unseen(q, TOP_K, m.seen))
    m.gamma_ops["retrieve"] += 1
    n_ret, n_redundant, n_post_cov, n_dup_blocked = 1, 0, 0, 0
    judge_calls, stop_reason = [], "budget"
    for r in range(MAX_ROUNDS):
        raw, conf = judge_agent(q, m.omega_ctx)
        m.gamma_ops["judge"] += 1
        verdict, p = parse_judge(raw), platt(conf)
        m.c_currency.append(p)
        judge_calls.append({"round": r, "raw": raw, "verdict": verdict, "conf": conf,
                            "conf_platt": p, "gold_covered_now": covered(m.titles, gold_titles)})
        if verdict is True:
            stop_reason = "judge_sufficient"; m.h_trace.append(("EARLY_STOP", stop_reason)); break
        # SPECIALIST_ACTIVATE with duplicate-query guard (one retry)
        nq = query_agent(q, m.omega_ctx); m.gamma_ops["query_agent"] += 1
        qv = retriever.embed(nq)
        if is_dup(m, qv):
            n_dup_blocked += 1
            nq = query_agent(q, m.omega_ctx + [{"query": nq + "  (DUPLICATE - must differ)",
                                                "chunks": []}])
            m.gamma_ops["query_agent"] += 1
            qv = retriever.embed(nq)
            if is_dup(m, qv):
                n_dup_blocked += 1
                stop_reason = "duplicate_query"; m.h_trace.append(("EARLY_STOP", stop_reason)); break
        m.psi_query.append(qv)
        cov_before = covered(m.titles, gold_titles)
        chunks = retriever.retrieve_unseen(nq, TOP_K, m.seen)    # SURGICAL_MICRO_RETRIEVAL
        m.gamma_ops["retrieve"] += 1
        n_ret += 1
        n_redundant += int(len(chunks) == 0)
        n_post_cov += int(cov_before is True)
        m.h_trace.append(("SURGICAL_MICRO_RETRIEVAL", nq))
        m.add(nq, chunks)
    pred = answer_agent(q, m.omega_ctx)
    return {"pred": pred, "n_retrievals": n_ret, "n_redundant": n_redundant,
            "n_post_coverage": n_post_cov, "gold_covered": covered(m.titles, gold_titles),
            "gold_any": any_gold(m.titles, gold_titles),
            "queries": [e["query"] for e in m.omega_ctx], "judge_calls": judge_calls,
            "latency_s": time.time() - t0, "stop_reason": stop_reason,
            "n_dup_blocked": n_dup_blocked, "n_llm_calls": m.gamma_ops["judge"]
            + m.gamma_ops["query_agent"] + 1}

SYSTEMS = {"naive_rag": naive_rag, "maps_scripted": maps_scripted, "orch_rules": orch_rules}


# %% CELL 8 — collect full trajectories (one row per question x step) =========
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

def pool_text(m):
    return normalize(" ".join(c["title"] + " " + c["text"]
                              for e in m.omega_ctx for c in e["chunks"]))

def trajectory(q, gold, gold_titles):
    m = SGWM(q_emb=retriever.embed(q))
    m.psi_query.append(m.q_emb)
    chunks = retriever.retrieve_unseen(q, TOP_K, m.seen)
    m.add(q, chunks)
    last = {"query": q, "n_new": len(chunks), "dup": False, "qsim": 0.0,
            "scores": [c["score"] for c in chunks]}
    rows, prev_pred = [], None
    for t in range(MAX_ROUNDS + 1):
        t0 = time.time()
        raw, conf = judge_agent(q, m.omega_ctx)
        verdict = parse_judge(raw)
        pred = answer_agent(q, m.omega_ctx)
        npred = normalize(pred)
        sc = last["scores"] or [0.0]
        rows.append({
            "t": t, "query": last["query"],
            "judge_raw": raw, "verdict": verdict, "conf": conf, "conf_platt": platt(conf),
            "n_new_last": last["n_new"], "top_score_last": max(sc),
            "mean_score_last": float(np.mean(sc)), "qsim_last": last["qsim"],
            "dup_last": last["dup"], "pool_size": len(m.seen),
            "pred": pred, "pred_words": len(npred.split()),
            "pred_changed": None if prev_pred is None else float(npred != prev_pred),
            "pred_in_pool": float(bool(npred) and npred in pool_text(m)),
            "em": em(pred, gold), "f1": round(f1(pred, gold), 4),
            "gold_covered": covered(m.titles, gold_titles),
            "gold_any": any_gold(m.titles, gold_titles),
            "step_latency_s": None})
        prev_pred = npred
        if t < MAX_ROUNDS:                       # prepare the next state
            nq = query_agent(q, m.omega_ctx)
            qv = retriever.embed(nq)
            dup = is_dup(m, qv)
            if dup:                              # same guard as Step 3, one retry
                nq = query_agent(q, m.omega_ctx + [{"query": nq + "  (DUPLICATE - must differ)",
                                                    "chunks": []}])
                qv = retriever.embed(nq)
                dup = is_dup(m, qv)
            qsim = max(float(qv @ p) for p in m.psi_query)
            m.psi_query.append(qv)
            chunks = retriever.retrieve_unseen(nq, TOP_K, m.seen)   # surgical retrieval
            m.add(nq, chunks)
            last = {"query": nq, "n_new": len(chunks), "dup": dup, "qsim": qsim,
                    "scores": [c["score"] for c in chunks]}
        rows[-1]["step_latency_s"] = round(time.time() - t0, 3)
    return rows

subset = queries.iloc[START_AT:START_AT + N_QUESTIONS].reset_index(drop=True)
all_rows, errors, t_run = [], [], time.time()
for k, row in enumerate(subset.itertuples(index=False), 1):
    try:
        for r in trajectory(row.question, row.answer, row.gold_titles):
            all_rows.append({"id": row.id, "qtype": row.qtype, "question": row.question,
                             "gold": row.answer, **r})
    except Exception as e:                       # log and keep going
        errors.append({"id": row.id, "error": repr(e)[:300]})
    if k % 20 == 0 or k == len(subset):
        d = pd.DataFrame(all_rows)
        f = d.groupby("t").f1.mean() * 100
        print(f"  {k}/{len(subset)} | F1 by step t: " +
              " ".join(f"t{t}={v:.1f}" for t, v in f.items()) +
              f" | errors={len(errors)} | {time.time()-t_run:.0f}s")
steps = pd.DataFrame(all_rows)
print(f"\ncollected {len(steps)} step rows from {steps.id.nunique()} questions "
      f"in {time.time()-t_run:.0f}s")


# %% CELL 9 — offline policy evaluation, checks, save =========================
# Every policy below is evaluated EXACTLY on the same trajectories:
# stopping at step t gives answer_t and costs t extra retrievals.
LAMBDAS = (0.02, 0.05, 0.10)

def first_stop(g, rule):
    for _, r in g.sort_values("t").iterrows():
        if r.t == MAX_ROUNDS or rule(r):
            return r
    return g.iloc[-1]

def evaluate(rule, name):
    picks = pd.DataFrame([first_stop(g, rule) for _, g in steps.groupby("id")])
    t = picks.t.astype(float)
    return {"policy": name, "F1": round(100 * picks.f1.mean(), 2),
            "EM": round(100 * picks.em.mean(), 2),
            "avg_extra_retrievals": round(t.mean(), 3),
            "gold_coverage_pct": round(100 * picks.gold_covered.astype(float).mean(), 2),
            # judge + query agent each step, answer once at the end
            "llm_calls_judge_policy": round((2 * t + 2).mean(), 3)}

pol = [evaluate(lambda r: True, "stop_at_t0"),
       evaluate(lambda r: r.verdict == True, "judge_stop (= Step 3 orchestrator)"),
       evaluate(lambda r: False, "always_full_budget")]
oracle = {}
for lam in LAMBDAS:
    best = steps.assign(R=steps.f1 - lam * steps.t).sort_values(["id", "R", "t"],
                        ascending=[True, False, True]).groupby("id").head(1)
    oracle[str(lam)] = {"F1": round(100 * best.f1.mean(), 2),
                        "avg_extra_retrievals": round(best.t.mean(), 3)}

by_t = {int(t): {"F1": round(100 * g.f1.mean(), 2), "EM": round(100 * g.em.mean(), 2),
                 "gold_coverage_pct": round(100 * g.gold_covered.astype(float).mean(), 2),
                 "said_sufficient_pct": round(100 * (g.verdict == True).mean(), 2),
                 "answer_changed_pct": (round(100 * g.pred_changed.mean(), 2)
                                        if g.pred_changed.notna().any() else None)}
        for t, g in steps.groupby("t")}

n_q = len(subset)
checks = {
    "no crashed questions (0%)":       len(errors) == 0,
    "4 steps per question":            bool((steps.groupby("id").t.count() == MAX_ROUNDS + 1).all()),
    "no empty answers (<5%)":          (steps.pred.str.strip() == "").mean() < 0.05,
    "judge parseable (>=95%)":         steps.verdict.notna().mean() >= 0.95,
    "confidence captured (>=80%)":     steps.conf.notna().mean() >= 0.80,
    "answers do change across steps":  (steps.pred_changed.fillna(0) > 0).any(),
}
checks = {k: bool(v) for k, v in checks.items()}
for k, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {k}")

summary = {"meta": {"run_tag": RUN_TAG, "model": MODEL, "embedder": EMBED_MODEL,
                    "start_at": START_AT, "n_questions": n_q, "top_k": TOP_K,
                    "max_rounds": MAX_ROUNDS, "search_depth": SEARCH_DEPTH,
                    "dup_query_sim": DUP_QUERY_SIM, "platt": [PLATT_A, PLATT_B],
                    "lambdas": LAMBDAS, "checks": checks, "errors": errors[:20]},
           "by_step": by_t, "policies": pol, "oracle_stop": oracle}
steps.to_csv(f"{OUT}/{RUN_TAG}_steps.csv", index=False)
with open(f"{OUT}/{RUN_TAG}_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\n----- PASTE THIS BACK -----")
print(json.dumps(summary, default=str))
print("---------------------------")
print(f"saved in {OUT}: {RUN_TAG}_steps.csv, {RUN_TAG}_summary.json")


# %% CELL 10 — push results to GitHub (works even if you're away) ============
# Needs Kaggle secret GITHUB_TOKEN (fine-grained, marl-maps only, Contents: Read and write).
import base64, requests
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")   # never print this
REPO = "KumarPriyam123/marl-maps"
H = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
tag = f"{RUN_TAG}_n{len(subset)}"                      # smoke test and full run don't overwrite
for kind in ("summary.json", "steps.csv"):
    src_path = f"{OUT}/{RUN_TAG}_{kind}"
    url = f"https://api.github.com/repos/{REPO}/contents/results/{tag}_{kind}"
    old = requests.get(url, headers=H)
    body = {"message": f"{tag}: {kind}",
            "content": base64.b64encode(open(src_path, "rb").read()).decode()}
    if old.status_code == 200:
        body["sha"] = old.json()["sha"]
    r = requests.put(url, headers=H, json=body)
    print(f"{tag}_{kind} ->", r.status_code)
    assert r.status_code in (200, 201), r.text[:300]
print(f"pushed to github.com/{REPO}/tree/main/results")
