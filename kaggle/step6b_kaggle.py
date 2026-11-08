# =============================================================================
# MARL-MAPS · STEP 6b — TRAIN THE QUERY AGENT WITH GRPO (Kaggle, LoRA, T4)
#
# Agent   : Qwen2.5-1.5B-Instruct + a NEW LoRA ("query"), same prompt as the pipeline's
#           query-extension agent (MAPS Appendix A).
# State   : question + evidence pool after the first retrieval, only where some gold
#           passage is still MISSING (otherwise there is nothing to learn).
# Reward  : process reward from retrieval, per sampled query q':
#             R = (# missing gold passages that q' brings in) / (# missing gold)
#                 - 0.2 if q' duplicates the question (cosine >= 0.92)
#                 - 0.2 if q' is empty
#           (a cheap, dense stand-in for MAPS's F1-gain reward Rq)
# Data    : a retrieval corpus built ONLY from HotpotQA TRAIN-split questions.
# GRPO    : G = 8 samples per prompt, group-normalised advantage, KL to the base model.
# Eval    : before AND after, greedy, holding the ANSWER agent fixed (Step 6a adapter):
#           E2 = Step 5 questions 800-999 on the validation corpus. Query sources compared:
#                1.5B untrained | 1.5B GRPO | the 7B's logged query (Step 5, t=1).
#           E1 = 200 held-out train-split questions on the train corpus.
# Safety  : config-checked checkpoint every 25 steps; Run All resumes after a crash.
# Kaggle  : GPU T4 x2 | Internet ON | secret GITHUB_TOKEN attached.
# Smoke test: SMOKE = True -> checks PASS -> SMOKE = False -> Run All (~1.5-2 h).
# Split at the "# %% CELL" markers to make notebook cells.
# =============================================================================


# %% CELL 1 — installs (once per fresh session) ===============================
import subprocess, sys
pkgs = ["peft", "faiss-cpu", "sentence-transformers", "datasets"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
# Kaggle's preinstalled torchao is too old for current peft; nothing here needs it.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
print("installed:", pkgs, "| removed: torchao")


# %% CELL 2 — config ===========================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import re, json, time, math, string, random, collections, inspect
import numpy as np, pandas as pd, faiss

SMOKE = True               # True: 4 steps + tiny evals. False: full run.
RUN_TAG = "step6b"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"
ANSWER_ADAPTER = "adapters/step6a"     # trained answer agent from Step 6a (kept frozen)
SEED = 0

N_STEPS = 4 if SMOKE else 200
PROMPTS_PER_STEP = 8
G = 8
MICRO_BATCH = 8
LR = 3e-5
KL_BETA = 0.04             # 6a drifted to KL 0.38 with 0.02; a bit firmer here
MAX_NEW_TOKENS = 20
TEMPERATURE = 1.0
LORA_R, LORA_ALPHA = 16, 32
CKPT_EVERY = 25
DUP_SIM = 0.92

TRAIN_PARQUET = ("hf://datasets/hotpotqa/hotpot_qa@refs/convert/parquet/"
                 "distractor/train/0000.parquet")
VAL_PARQUET = ("hf://datasets/hotpotqa/hotpot_qa@refs/convert/parquet/"
               "distractor/validation/0000.parquet")
N_TRAIN_Q = 2000           # questions whose paragraphs form the TRAIN corpus
N_E1 = 20 if SMOKE else 200
CORPUS_QUESTIONS = 1000    # validation corpus, identical to Steps 1-5
E2_FILE = "results/step5_n200_steps.csv"
E2_LIMIT = 10 if SMOKE else 200
TOP_K, SEARCH_DEPTH, MAX_PASSAGE_WORDS = 3, 20, 250
EVAL_BATCH = 8

REPO = "KumarPriyam123/marl-maps"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
TAG = f"{RUN_TAG}{'_smoke' if SMOKE else ''}"
CKPT = f"{OUT}/{TAG}_ckpt.pt"
random.seed(SEED); np.random.seed(SEED)
print("SMOKE =", SMOKE, "| steps:", N_STEPS)


# %% CELL 3 — data: validation corpus + TRAIN questions and TRAIN corpus =========
from datasets import load_dataset

def load_rows(path, n):
    ds = load_dataset("parquet", data_files=path, split="train")
    return ds.select(range(min(n, len(ds)))).to_list()

def paragraphs(ex):
    ctx = ex["context"]
    out = []
    for t, s in zip(list(ctx["title"]), list(ctx["sentences"])):
        text = " ".join(str(x) for x in list(s)).strip()
        if text:
            out.append({"title": str(t), "text": text})
    return out

def build_corpus(rows):
    ps, keys = [], set()
    for ex in rows:
        for p in paragraphs(ex):
            k = (p["title"], p["text"])
            if k not in keys:
                keys.add(k); ps.append(p)
    return ps

val_rows = load_rows(VAL_PARQUET, CORPUS_QUESTIONS)
val_passages = build_corpus(val_rows)
val_by_id = {str(ex["id"]): ex for ex in val_rows}
del val_rows

train_rows = load_rows(TRAIN_PARQUET, 20000)
random.Random(SEED + 1).shuffle(train_rows)
train_rows = train_rows[:N_TRAIN_Q + N_E1]
train_passages = build_corpus(train_rows)
train_qs = [{"id": str(ex["id"]), "question": str(ex["question"]), "answer": str(ex["answer"]),
             "gold_titles": sorted(set(str(t) for t in list(ex["supporting_facts"]["title"])))}
            for ex in train_rows]
del train_rows
print(f"val corpus: {len(val_passages)} | train corpus: {len(train_passages)} | "
      f"train questions: {len(train_qs)}")


# %% CELL 4 — retrievers, starting states, Step 5 data and the answer adapter ===
import base64, requests, torch
from sentence_transformers import SentenceTransformer
from kaggle_secrets import UserSecretsClient

EMB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL, device=EMB_DEVICE)

def embed(texts):
    return embedder.encode(texts, batch_size=128, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype("float32")

def make_index(ps):
    e = embed([f"{p['title']}: {p['text']}" for p in ps])
    ix = faiss.IndexFlatIP(e.shape[1])
    ix.add(e)
    return ix

CORPORA = {"val": (val_passages, make_index(val_passages)),
           "train": (train_passages, make_index(train_passages))}

def retrieve_unseen_batch(corpus, queries, seens, k=TOP_K):
    ps, ix = CORPORA[corpus]
    _, idx = ix.search(embed(queries), SEARCH_DEPTH)
    return [[int(i) for i in row if i >= 0 and int(i) not in seen][:k]
            for row, seen in zip(idx, seens)]

def make_state(corpus, item, first_query=None):
    """Pool after the first retrieval (query = the question itself)."""
    ps = CORPORA[corpus][0]
    hits = retrieve_unseen_batch(corpus, [first_query or item["question"]], [set()])[0]
    titles = {ps[i]["title"] for i in hits}
    missing = [t for t in item["gold_titles"] if t not in titles]
    return {**item, "corpus": corpus, "seen": set(hits),
            "pool": [{"query": first_query or item["question"],
                      "chunks": [ps[i] for i in hits]}],
            "missing": missing}

all_train_states = [make_state("train", q) for q in train_qs]
e1_states = [s for s in all_train_states[:N_E1] if s["missing"]]
train_states = [s for s in all_train_states[N_E1:] if s["missing"]]
print(f"train states with missing gold: {len(train_states)} / {N_TRAIN_Q} | "
      f"E1 states: {len(e1_states)}")

token = UserSecretsClient().get_secret("GITHUB_TOKEN")      # never print this
GH = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
RAW = {**GH, "Accept": "application/vnd.github.raw+json"}

def fetch(path, local):
    r = requests.get(f"https://api.github.com/repos/{REPO}/contents/{path}", headers=RAW)
    assert r.status_code == 200, f"could not fetch {path}: {r.status_code}"
    open(local, "wb").write(r.content)

fetch(E2_FILE, f"{OUT}/e2_steps.csv")
os.makedirs(f"{OUT}/answer_adapter", exist_ok=True)
for fn in ("adapter_config.json", "adapter_model.safetensors"):
    fetch(f"{ANSWER_ADAPTER}/{fn}", f"{OUT}/answer_adapter/{fn}")

s5 = pd.read_csv(f"{OUT}/e2_steps.csv", keep_default_na=False, na_values=[""])
e2_states = []
for qid, g in list(s5.groupby("id", sort=False))[:E2_LIMIT]:
    g = g.sort_values("t")
    ex = val_by_id[str(qid)]
    item = {"id": str(qid), "question": str(ex["question"]), "answer": str(ex["answer"]),
            "gold_titles": sorted(set(str(t) for t in list(ex["supporting_facts"]["title"])))}
    st = make_state("val", item, first_query=str(g.iloc[0]["query"]))
    st["query_7b"] = str(g.iloc[1]["query"])                  # the 7B's logged t=1 query
    st["covered_7b_logged"] = str(g.iloc[1]["gold_covered"]).lower() == "true"
    e2_states.append(st)
print(f"E2 states: {len(e2_states)} (of which gold still missing: "
      f"{sum(bool(s['missing']) for s in e2_states)})")


# %% CELL 5 — model: base + trainable "query" LoRA + frozen "answer" LoRA ======
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

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
_fwd = inspect.signature(base.forward).parameters
KEEP_ARG = "logits_to_keep" if "logits_to_keep" in _fwd else (
    "num_logits_to_keep" if "num_logits_to_keep" in _fwd else None)
base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
base.config.use_cache = False
model = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                        task_type="CAUSAL_LM"), adapter_name="query")
model.load_adapter(f"{OUT}/answer_adapter", adapter_name="answer", is_trainable=False)

def use(adapter):
    """Activate one adapter. peft marks the active one trainable, so re-freeze 'answer'."""
    model.set_adapter(adapter)
    for n, p in model.named_parameters():
        p.requires_grad = (".query." in n or n.endswith(".query.weight")) and "lora_" in n

use("query")
QUERY_PARAMS = [p for n, p in model.named_parameters() if p.requires_grad]
for p in QUERY_PARAMS:                               # fp32 LoRA weights for stable Adam
    p.data = p.data.float()
print("trainable query-LoRA tensors:", len(QUERY_PARAMS),
      "| params:", sum(p.numel() for p in QUERY_PARAMS))
EOS_IDS = set(model.generation_config.eos_token_id
              if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
EOS_IDS.add(tok.eos_token_id)

QUERY_SYS = ("You are a retrieval query generation agent.\n"
    "Generate exactly one NEW QUERY that helps answer the QUESTION or fills gaps "
    "not covered by EXISTING QUERIES and RETRIEVED DOCUMENTS, and differs from all "
    "existing queries.\nOutput rules: keywords only; no stopwords or connectors; "
    "max 8 keywords separated by single spaces. Output only the query text.")
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

def query_prompt(st):
    return chat(QUERY_SYS, f"Question: {st['question']}\nQuery Pool:\n{format_pool(st['pool'])}")

def answer_prompt(question, pool):
    return chat(ANSWER_SYS.format(pool=format_pool(pool)), f"Question: {question}")

def clean_query(text, fallback):
    kws = re.findall(r"[A-Za-z0-9']+", text)
    return " ".join(kws[:8]), bool(kws)

def first_line(text):
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")

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

@torch.no_grad()
def generate(prompts, sample, n=1, max_new=MAX_NEW_TOKENS):
    model.eval()
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=sample,
                             temperature=TEMPERATURE if sample else None,
                             top_p=1.0 if sample else None, top_k=0 if sample else None,
                             repetition_penalty=1.0, num_return_sequences=n,
                             pad_token_id=tok.pad_token_id, use_cache=True)
    return enc["input_ids"].shape[1], out, enc["attention_mask"]

def completion_mask(comp):
    m = torch.ones_like(comp)
    for i in range(comp.shape[0]):
        for j, t in enumerate(comp[i].tolist()):
            if t in EOS_IDS:
                m[i, j + 1:] = 0
                break
    return m

def token_logprobs(seq, attn, comp_len):
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    kw = {KEEP_ARG: comp_len + 1} if KEEP_ARG else {}
    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
        logits = model(input_ids=seq, attention_mask=attn, position_ids=pos, **kw).logits
    logits = logits[:, -(comp_len + 1):-1, :].float()
    tgt = seq[:, -comp_len:]
    return torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

def query_rewards(states, queries_raw):
    """Retrieval process reward for each (state, raw query)."""
    qs, ok = zip(*[clean_query(q, st["question"]) for st, q in zip(states, queries_raw)])
    qv = embed([q or "empty" for q in qs])
    qq = embed([st["question"] for st in states])
    hits = retrieve_unseen_batch(states[0]["corpus"], [q or "empty" for q in qs],
                                 [st["seen"] for st in states])
    out = []
    for st, q, good, h, v, u in zip(states, qs, ok, hits, qv, qq):
        ps = CORPORA[st["corpus"]][0]
        new_titles = {ps[i]["title"] for i in h}
        gain = (sum(t in new_titles for t in st["missing"]) / len(st["missing"])
                if st["missing"] else 0.0)
        dup = float(v @ u) >= DUP_SIM
        out.append({"query": q, "hits": h, "gain": gain, "dup": dup,
                    "reward": gain - 0.2 * dup - 0.2 * (not good)})
    return out

print("device:", DEVICE, "| logits arg:", KEEP_ARG)


# %% CELL 6 — evaluation: query quality AND downstream answer quality ===========
def eval_states(states, source, queries=None):
    """source: 'model' (query adapter, greedy) or 'given' (queries supplied)."""
    rows = []
    for b in range(0, len(states), EVAL_BATCH):
        chunk = states[b:b + EVAL_BATCH]
        if source == "model":
            use("query")
            plen, out, _ = generate([query_prompt(st) for st in chunk], sample=False)
            raw = [tok.decode(s[plen:], skip_special_tokens=True) for s in out]
        else:
            raw = queries[b:b + EVAL_BATCH]
        res = query_rewards(chunk, raw)
        pools = []
        for st, r in zip(chunk, res):
            ps = CORPORA[st["corpus"]][0]
            pools.append(st["pool"] + [{"query": r["query"], "chunks": [ps[i] for i in r["hits"]]}])
        use("answer")
        plen, out, _ = generate([answer_prompt(st["question"], p) for st, p in zip(chunk, pools)],
                                sample=False, max_new=24)
        for st, r, s, pool in zip(chunk, res, out, pools):
            pred = first_line(tok.decode(s[plen:], skip_special_tokens=True))
            titles = {c["title"] for e in pool for c in e["chunks"]}
            rows.append({"id": st["id"], "query": r["query"], "gain": r["gain"], "dup": r["dup"],
                         "covered_after": all(t in titles for t in st["gold_titles"]),
                         "had_missing": bool(st["missing"]), "pred": pred, "gold": st["answer"],
                         "f1": f1(pred, st["answer"]), "em": em(pred, st["answer"])})
    use("query")
    return pd.DataFrame(rows)

def summarize(d):
    m = d[d.had_missing]
    return {"n": int(len(d)), "covered_after_pct": round(100 * d.covered_after.mean(), 2),
            "gain_on_missing": round(float(m.gain.mean()), 4) if len(m) else None,
            "dup_query_pct": round(100 * d.dup.mean(), 2),
            "answer_F1": round(100 * d.f1.mean(), 2), "answer_EM": round(100 * d.em.mean(), 2)}

BEFORE_FILE = f"{OUT}/{TAG}_eval_before.pkl"
if os.path.exists(BEFORE_FILE):
    before = pd.read_pickle(BEFORE_FILE)
else:
    before = {"E1": eval_states(e1_states, "model"),
              "E2": eval_states(e2_states, "model"),
              "E2_7b_query": eval_states(e2_states, "given", [s["query_7b"] for s in e2_states])}
    pd.to_pickle(before, BEFORE_FILE)
res_before = {k: summarize(v) for k, v in before.items()}
rebuild_7b = float(np.mean(before["E2_7b_query"].covered_after.values ==
                           np.array([s["covered_7b_logged"] for s in e2_states])))
print("BEFORE:", json.dumps(res_before))
print(f"7B-query pools match Step 5 logged coverage: {100 * rebuild_7b:.1f}%")


# %% CELL 7 — GRPO training of the query agent (checkpointed) =================
use("query")
opt = torch.optim.AdamW(QUERY_PARAMS, lr=LR, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda", enabled=AMP)
CFG = {"smoke": SMOKE, "n_e1": N_E1, "n_train_q": N_TRAIN_Q, "prompts": PROMPTS_PER_STEP,
       "G": G, "lr": LR, "kl_beta": KL_BETA, "lora": [LORA_R, LORA_ALPHA], "model": MODEL}
log, start_step = [], 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    if ck.get("cfg") != CFG:
        print("IGNORING checkpoint from a different configuration:", ck.get("cfg"))
    else:
        set_peft_model_state_dict(model, ck["lora"], adapter_name="query")
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        log, start_step = ck["log"], ck["step"]
        print(f"resuming from step {start_step}")

def save_ckpt(step):
    torch.save({"lora": get_peft_model_state_dict(model, adapter_name="query"),
                "opt": opt.state_dict(), "scaler": scaler.state_dict(), "log": log,
                "step": step, "cfg": CFG}, CKPT)

t_run = time.time()
for step in range(start_step, N_STEPS):
    t0 = time.time()
    rng = random.Random(SEED * 100003 + step)
    batch = rng.sample(train_states, PROMPTS_PER_STEP)
    use("query")
    plen, seqs, pmask = generate([query_prompt(st) for st in batch], sample=True, n=G)
    comp = seqs[:, plen:]
    cmask = completion_mask(comp)
    attn = torch.cat([pmask.repeat_interleave(G, 0), cmask], dim=1)
    texts = tok.batch_decode(comp, skip_special_tokens=True)
    rs = query_rewards([batch[i // G] for i in range(len(texts))], texts)
    R = torch.tensor([r["reward"] for r in rs])
    Rg = R.view(PROMPTS_PER_STEP, G)
    adv = ((Rg - Rg.mean(1, keepdim=True)) / (Rg.std(1, keepdim=True) + 1e-4)).view(-1)
    live = (Rg.std(1) > 1e-6).repeat_interleave(G)
    comp_len = comp.shape[1]

    model.train()
    opt.zero_grad(set_to_none=True)
    idx = torch.nonzero(live).view(-1).tolist()
    kl_sum, n_seq = 0.0, max(len(idx), 1)
    for b in range(0, len(idx), MICRO_BATCH):
        mb = idx[b:b + MICRO_BATCH]
        s, a, m = seqs[mb], attn[mb], cmask[mb].float()
        with torch.no_grad(), model.disable_adapter():
            ref = token_logprobs(s, a, comp_len)
        lp = token_logprobs(s, a, comp_len)
        diff = ref - lp
        kl = (torch.exp(diff) - diff - 1) * m
        ntok = m.sum(1).clamp(min=1)
        pg = -(adv[mb].to(lp.device) * (lp * m).sum(1) / ntok)
        loss = (pg + KL_BETA * kl.sum(1) / ntok).sum() / n_seq
        scaler.scale(loss).backward()
        kl_sum += float((kl.sum(1) / ntok).sum())
    if idx:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(QUERY_PARAMS, 1.0)
        scaler.step(opt)
        scaler.update()
    log.append({"step": step, "reward_mean": float(R.mean()), "gain_mean":
                float(np.mean([r["gain"] for r in rs])), "dup_pct": 100 * float(np.mean(
                [r["dup"] for r in rs])), "live_groups": int(live.sum()) // G,
                "kl": kl_sum / n_seq, "comp_tokens": float(cmask.sum(1).float().mean()),
                "sec": round(time.time() - t0, 1)})
    if (step + 1) % CKPT_EVERY == 0 or step + 1 == N_STEPS:
        save_ckpt(step + 1)
    if step % 5 == 0 or step + 1 == N_STEPS:
        L = log[-25:]
        print(f"  step {step + 1}/{N_STEPS} | reward(last25)={np.mean([x['reward_mean'] for x in L]):.3f}"
              f" | gain(last25)={np.mean([x['gain_mean'] for x in L]):.3f}"
              f" | live={np.mean([x['live_groups'] for x in L]):.1f}/{PROMPTS_PER_STEP}"
              f" | kl={log[-1]['kl']:.4f} | {log[-1]['sec']}s/step | {time.time()-t_run:.0f}s")
print("training done")


# %% CELL 8 — evaluate after training, compare, checks, save ===================
after = {"E1": eval_states(e1_states, "model"), "E2": eval_states(e2_states, "model")}
res_after = {k: summarize(v) for k, v in after.items()}

def paired(a, b, n=10000, scale=100):
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(0)
    bo = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"mean": round(scale * float(d.mean()), 2),
            "ci95": [round(scale * float(np.percentile(bo, 2.5)), 2),
                     round(scale * float(np.percentile(bo, 97.5)), 2)],
            "p_le_0": round(float((bo <= 0).mean()), 4)}

B, A, Q7 = before["E2"], after["E2"], before["E2_7b_query"]
cmp = {"E2_covered_after_minus_before": paired(B.covered_after, A.covered_after),
       "E2_covered_after_minus_7b": paired(Q7.covered_after, A.covered_after),
       "E2_answerF1_after_minus_before": paired(B.f1, A.f1),
       "E2_answerF1_after_minus_7b": paired(Q7.f1, A.f1),
       "E1_covered_after_minus_before": paired(before["E1"].covered_after,
                                               after["E1"].covered_after),
       "E1_answerF1_after_minus_before": paired(before["E1"].f1, after["E1"].f1)}

L = pd.DataFrame(log)
k = max(1, min(25, len(L) // 2))
checks = {
    "training ran all steps":               len(L) == N_STEPS,
    "no NaN in log":                        bool(np.isfinite(L[["reward_mean", "kl"]].values).all()),
    "groups carry signal (>=40% live)":     L.live_groups.mean() >= 0.40 * PROMPTS_PER_STEP,
    "7B-query pools rebuilt (>=95%)":       rebuild_7b >= 0.95,
    "queries not empty after training":     (A["query"].str.strip() == "").mean() < 0.05,
}
checks = {a: bool(b) for a, b in checks.items()}
for a, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {a}")

summary = {"meta": {"run_tag": RUN_TAG, "smoke": SMOKE, "model": MODEL, "cfg": CFG,
                    "n_train_states": len(train_states), "n_e1": len(e1_states),
                    "n_e2": len(e2_states), "rebuild_7b": round(rebuild_7b, 4),
                    "answer_agent": ANSWER_ADAPTER + " (frozen)", "checks": checks},
           "train": {"reward_first": round(float(L.reward_mean.head(k).mean()), 4),
                     "reward_last": round(float(L.reward_mean.tail(k).mean()), 4),
                     "gain_first": round(float(L.gain_mean.head(k).mean()), 4),
                     "gain_last": round(float(L.gain_mean.tail(k).mean()), 4),
                     "live_groups_mean": round(float(L.live_groups.mean()), 2),
                     "kl_last": round(float(L.kl.tail(k).mean()), 5),
                     "sec_per_step": round(float(L.sec.mean()), 1)},
           "before": res_before, "after": res_after, "paired": cmp}
L.to_csv(f"{OUT}/{TAG}_train_log.csv", index=False)
pd.concat([before["E1"].assign(set="E1", src="1.5B_untrained"),
           before["E2"].assign(set="E2", src="1.5B_untrained"),
           before["E2_7b_query"].assign(set="E2", src="7B_logged_query"),
           after["E1"].assign(set="E1", src="1.5B_grpo"),
           after["E2"].assign(set="E2", src="1.5B_grpo")]).to_csv(f"{OUT}/{TAG}_eval.csv", index=False)
with open(f"{OUT}/{TAG}_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
model.save_pretrained(f"{OUT}/{TAG}_adapter", selected_adapters=["query"])
print("\n----- PASTE THIS BACK -----")
print(json.dumps(summary))
print("---------------------------")


# %% CELL 9 — push results + query adapter to GitHub ==========================
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

for kind in ("summary.json", "train_log.csv", "eval.csv"):
    push(f"{OUT}/{TAG}_{kind}", f"results/{TAG}_{kind}")
for root, _, files in os.walk(f"{OUT}/{TAG}_adapter"):
    for fn in files:
        rel = os.path.relpath(os.path.join(root, fn), f"{OUT}/{TAG}_adapter")
        push(os.path.join(root, fn), f"adapters/{TAG}/{rel}")
print(f"pushed to github.com/{REPO}")
