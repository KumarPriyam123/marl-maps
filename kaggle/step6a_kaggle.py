# =============================================================================
# MARL-MAPS · STEP 6a — TRAIN THE ANSWER AGENT WITH GRPO (Kaggle, LoRA, T4)
#
# Model   : Qwen2.5-1.5B-Instruct + LoRA (attention projections), fp16 base.
#           7B GRPO does not fit a T4; the fair baseline is the SAME 1.5B untrained.
# Data    : HotpotQA TRAIN split (distractor). Evaluation questions are all from
#           validation, so training never sees them.
# Prompt  : exactly the pipeline's answer-agent prompt (MAPS Appendix A) and pool
#           format; training pools mix gold + distractor paragraphs, sometimes
#           with a gold paragraph missing (insufficient evidence).
# GRPO    : G samples per prompt, reward R = F1 + 0.5*EM (empty answer: -0.1),
#           advantage = (R - group mean) / group std, one policy-gradient update per
#           batch, KL penalty to the untrained model (adapter switched off).
# Eval    : before AND after training, greedy decoding, on
#           E1 = 300 held-out TRAIN-split questions (in-distribution)
#           E2 = the exact Step 5 evidence pools (questions 800-999) rebuilt from the
#                logged queries; the 7B's logged answers on the same pools are the
#                reference.
# Safety  : checkpoint every 25 steps; Run All after a crash resumes training.
# Kaggle  : GPU T4 x2 (one used) | Internet ON | secret GITHUB_TOKEN attached.
# Smoke test: SMOKE = True (5 steps, tiny evals) -> checks PASS -> SMOKE = False.
# Split at the "# %% CELL" markers to make notebook cells.
# =============================================================================


# %% CELL 1 — installs (once per fresh session) ===============================
import subprocess, sys
pkgs = ["peft", "faiss-cpu", "sentence-transformers", "datasets"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
# Kaggle's preinstalled torchao is too old for current peft ("incompatible version of
# torchao") and nothing here needs it.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
print("installed:", pkgs, "| removed: torchao")


# %% CELL 2 — config ===========================================================
import os
# less GPU memory fragmentation; only takes effect if set before CUDA starts
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import re, json, time, math, string, random, collections, inspect
import numpy as np, pandas as pd, faiss

SMOKE = True               # True: 5 steps + tiny evals (~15 min). False: full run.
RUN_TAG = "step6a"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"
SEED = 0

# GRPO
N_STEPS = 5 if SMOKE else 250
PROMPTS_PER_STEP = 8
G = 4                      # samples per prompt
MICRO_BATCH = 4            # sequences per backward pass (8 came close to OOM on T4)
LR = 3e-5
KL_BETA = 0.02
MAX_NEW_TOKENS = 24
TEMPERATURE = 1.0
LORA_R, LORA_ALPHA = 16, 32
CKPT_EVERY = 25

# data / eval
TRAIN_PARQUET = ("hf://datasets/hotpotqa/hotpot_qa@refs/convert/parquet/"
                 "distractor/train/0000.parquet")
VAL_PARQUET = ("hf://datasets/hotpotqa/hotpot_qa@refs/convert/parquet/"
               "distractor/validation/0000.parquet")
N_TRAIN_Q = 4000
N_E1 = 20 if SMOKE else 300
CORPUS_QUESTIONS = 1000    # same corpus as Steps 1-5 (needed to rebuild Step 5 pools)
E2_FILE = "results/step5_n200_steps.csv"
E2_LIMIT = 10 if SMOKE else 200
E2_STEPS = (0, 1, 3)       # evidence pools after 0, 1 and 3 extra searches
TOP_K, SEARCH_DEPTH, MAX_PASSAGE_WORDS = 3, 20, 250
EVAL_BATCH = 8

REPO = "KumarPriyam123/marl-maps"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
CKPT = f"{OUT}/{RUN_TAG}{'_smoke' if SMOKE else ''}_ckpt.pt"   # smoke never resumes a full run
random.seed(SEED); np.random.seed(SEED)
print("SMOKE =", SMOKE, "| steps:", N_STEPS)


# %% CELL 3 — data: HotpotQA validation corpus + TRAIN questions ===============
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

# validation corpus: identical construction to Steps 1-5
val_rows = load_rows(VAL_PARQUET, CORPUS_QUESTIONS)
passages, seen_keys = [], set()
for ex in val_rows:
    for p in paragraphs(ex):
        key = (p["title"], p["text"])
        if key not in seen_keys:
            seen_keys.add(key); passages.append(p)
val_by_id = {str(ex["id"]): ex for ex in val_rows}
del val_rows
print("validation corpus passages:", len(passages))

# TRAIN split questions (never used for evaluation in Steps 1-5)
train_rows = load_rows(TRAIN_PARQUET, 20000)
random.Random(SEED).shuffle(train_rows)
train_items = []
for ex in train_rows:
    gold_t = set(str(t) for t in list(ex["supporting_facts"]["title"]))
    paras = paragraphs(ex)
    gold = [p for p in paras if p["title"] in gold_t]
    dis = [p for p in paras if p["title"] not in gold_t]
    if len(gold) >= 1 and len(dis) >= 3:
        train_items.append({"id": str(ex["id"]), "question": str(ex["question"]),
                            "answer": str(ex["answer"]), "gold": gold, "dis": dis})
    if len(train_items) >= N_TRAIN_Q + N_E1:
        break
del train_rows
e1_items, train_items = train_items[:N_E1], train_items[N_E1:]
print(f"train questions: {len(train_items)} | E1 held-out train questions: {len(e1_items)}")


# %% CELL 4 — retriever + rebuild the exact Step 5 evidence pools (E2) ==========
import base64, requests
from sentence_transformers import SentenceTransformer
from kaggle_secrets import UserSecretsClient

embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
emb = embedder.encode([f"{p['title']}: {p['text']}" for p in passages], batch_size=128,
                      normalize_embeddings=True, convert_to_numpy=True,
                      show_progress_bar=True).astype("float32")
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)

def retrieve_unseen(query, k, seen):
    qv = embedder.encode([query], normalize_embeddings=True,
                         convert_to_numpy=True).astype("float32")
    _, idx = index.search(qv, SEARCH_DEPTH)
    hits = [int(i) for i in idx[0] if i >= 0 and int(i) not in seen]
    return hits[:k]

token = UserSecretsClient().get_secret("GITHUB_TOKEN")      # never print this
GH = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
r = requests.get(f"https://api.github.com/repos/{REPO}/contents/{E2_FILE}",
                 headers={**GH, "Accept": "application/vnd.github.raw+json"})
assert r.status_code == 200, f"could not fetch {E2_FILE}: {r.status_code}"
open(f"{OUT}/e2_steps.csv", "w").write(r.text)
s5 = pd.read_csv(f"{OUT}/e2_steps.csv", keep_default_na=False, na_values=[""])

e2_items, match = [], []
for qid, g in list(s5.groupby("id", sort=False))[:E2_LIMIT]:
    g = g.sort_values("t")
    ex = val_by_id[str(qid)]
    gold_titles = set(str(t) for t in list(ex["supporting_facts"]["title"]))
    pool, seen = [], set()
    for _, row in g.iterrows():                 # replay the logged queries
        hits = retrieve_unseen(str(row["query"]), TOP_K, seen)
        seen |= set(hits)
        pool.append({"query": str(row["query"]), "chunks": [passages[i] for i in hits]})
        titles = {passages[i]["title"] for i in seen}
        cov = all(t in titles for t in gold_titles)
        match.append(cov == (str(row["gold_covered"]).lower() == "true"))
        if int(row["t"]) in E2_STEPS:
            e2_items.append({"id": str(qid), "t": int(row["t"]), "question": str(ex["question"]),
                             "answer": str(ex["answer"]), "pool": [dict(e) for e in pool],
                             "f1_7b": float(row["f1"]), "em_7b": float(row["em"])})
REBUILD_MATCH = float(np.mean(match))
print(f"E2 items: {len(e2_items)} | pool rebuild matches Step 5 gold coverage: "
      f"{100 * REBUILD_MATCH:.1f}%")


# %% CELL 5 — model, LoRA, prompts, generation and log-probs ===================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
try:
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16 if AMP else torch.float32)
except TypeError:                                   # older transformers
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if AMP else torch.float32)
base.to(DEVICE)
_fwd = inspect.signature(base.forward).parameters
KEEP_ARG = "logits_to_keep" if "logits_to_keep" in _fwd else (
    "num_logits_to_keep" if "num_logits_to_keep" in _fwd else None)
base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
base.config.use_cache = False
model = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                        task_type="CAUSAL_LM"))
for p in model.parameters():                        # LoRA weights in fp32 for stable Adam
    if p.requires_grad:
        p.data = p.data.float()
model.print_trainable_parameters()
EOS_IDS = set(model.generation_config.eos_token_id
              if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
EOS_IDS.add(tok.eos_token_id)

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

def prompt_text(question, pool):
    msgs = [{"role": "system", "content": ANSWER_SYS.format(pool=format_pool(pool))},
            {"role": "user", "content": f"Question: {question}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

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

def reward(completion, gold):
    pred = first_line(completion)
    if not pred:
        return -0.1
    return f1(pred, gold) + 0.5 * em(pred, gold)

@torch.no_grad()
def generate(prompts, sample, n=1):
    model.eval()
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=sample,
                             temperature=TEMPERATURE if sample else None,
                             top_p=1.0 if sample else None, top_k=0 if sample else None,
                             repetition_penalty=1.0,   # sample from the policy itself
                             num_return_sequences=n, pad_token_id=tok.pad_token_id,
                             use_cache=True)
    return enc["input_ids"].shape[1], out, enc["attention_mask"]

def completion_mask(comp):
    """1 for generated tokens up to and including the first EOS, else 0."""
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

print("device:", DEVICE, "| logits arg:", KEEP_ARG)


# %% CELL 6 — evaluation (greedy) on E1 and E2 ================================
def train_pool(item, rng):
    """Pipeline-shaped pool: question + keyword-style follow-up 'queries'."""
    n = rng.choice([3, 6, 9])
    gold = [p for p in item["gold"] if rng.random() < 0.85] or []
    dis = rng.sample(item["dis"], min(len(item["dis"]), max(n - len(gold), 0)))
    chunks = gold + dis
    rng.shuffle(chunks)
    pool = []
    for i in range(0, len(chunks), 3):
        grp = chunks[i:i + 3]
        q = item["question"] if i == 0 else " ".join(grp[0]["title"].split()[:6])
        pool.append({"query": q, "chunks": grp})
    return pool

E1_POOLS = [train_pool(it, random.Random(1000 + i)) for i, it in enumerate(e1_items)]

def evaluate(tag):
    rows = []
    jobs = ([("E1", it["id"], -1, it["question"], it["answer"], pool, None)
             for it, pool in zip(e1_items, E1_POOLS)] +
            [("E2", it["id"], it["t"], it["question"], it["answer"], it["pool"], it["f1_7b"])
             for it in e2_items])
    for b in range(0, len(jobs), EVAL_BATCH):
        chunk = jobs[b:b + EVAL_BATCH]
        plen, out, _ = generate([prompt_text(j[3], j[5]) for j in chunk], sample=False)
        for j, seq in zip(chunk, out):
            pred = first_line(tok.decode(seq[plen:], skip_special_tokens=True))
            rows.append({"set": j[0], "id": j[1], "t": j[2], "gold": j[4], "pred": pred,
                         "em": em(pred, j[4]), "f1": f1(pred, j[4]), "f1_7b": j[6],
                         "model": tag})
    d = pd.DataFrame(rows)
    res = {"E1": {"F1": round(100 * d[d.set == "E1"].f1.mean(), 2),
                  "EM": round(100 * d[d.set == "E1"].em.mean(), 2)}}
    for t, g in d[d.set == "E2"].groupby("t"):
        res[f"E2_t{t}"] = {"F1": round(100 * g.f1.mean(), 2), "EM": round(100 * g.em.mean(), 2),
                           "F1_7b_logged": round(100 * g.f1_7b.mean(), 2)}
    return d, res

BEFORE_FILE = f"{OUT}/{RUN_TAG}{'_smoke' if SMOKE else ''}_eval_before.pkl"
if os.path.exists(BEFORE_FILE):                 # resume after a crash: don't redo it
    eval_before, res_before = pd.read_pickle(BEFORE_FILE)
else:
    eval_before, res_before = evaluate("1.5B_untrained")
    pd.to_pickle((eval_before, res_before), BEFORE_FILE)
print("BEFORE training:", json.dumps(res_before))


# %% CELL 7 — GRPO training (checkpointed; Run All resumes) ===================
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR,
                        weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda", enabled=AMP)
log, start_step = [], 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model, ck["lora"])
    opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
    log, start_step = ck["log"], ck["step"]
    print(f"resuming from step {start_step}")

def save_ckpt(step):
    torch.save({"lora": get_peft_model_state_dict(model), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "log": log, "step": step}, CKPT)

t_run = time.time()
for step in range(start_step, N_STEPS):
    t0 = time.time()
    rng = random.Random(SEED * 100003 + step)          # same batch on resume
    batch = rng.sample(train_items, PROMPTS_PER_STEP)
    prompts = [prompt_text(it["question"], train_pool(it, rng)) for it in batch]
    plen, seqs, pmask = generate(prompts, sample=True, n=G)
    comp = seqs[:, plen:]
    cmask = completion_mask(comp)
    attn = torch.cat([pmask.repeat_interleave(G, 0), cmask], dim=1)
    texts = tok.batch_decode(comp, skip_special_tokens=True)
    R = torch.tensor([reward(tx, batch[i // G]["answer"]) for i, tx in enumerate(texts)])
    Rg = R.view(PROMPTS_PER_STEP, G)
    adv = ((Rg - Rg.mean(1, keepdim=True)) / (Rg.std(1, keepdim=True) + 1e-4)).view(-1)
    live = (Rg.std(1) > 1e-6).repeat_interleave(G)       # groups with no signal are skipped
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
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt)                       # skipped internally if grads are inf/NaN
        scaler.update()
    log.append({"step": step, "reward_mean": float(R.mean()), "reward_max": float(R.max()),
                "f1_mean": float(np.mean([f1(first_line(tx), batch[i // G]["answer"])
                                          for i, tx in enumerate(texts)])),
                "live_groups": int(live.sum()) // G, "kl": kl_sum / n_seq,
                "comp_tokens": float(cmask.sum(1).float().mean()),
                "sec": round(time.time() - t0, 1)})
    if (step + 1) % CKPT_EVERY == 0 or step + 1 == N_STEPS:
        save_ckpt(step + 1)
    if step % 5 == 0 or step + 1 == N_STEPS:
        L = log[-25:]
        print(f"  step {step + 1}/{N_STEPS} | reward(last25)={np.mean([x['reward_mean'] for x in L]):.3f}"
              f" | F1(last25)={100*np.mean([x['f1_mean'] for x in L]):.1f}"
              f" | kl={log[-1]['kl']:.4f} | {log[-1]['sec']}s/step | {time.time()-t_run:.0f}s")
print("training done")


# %% CELL 8 — evaluate after training, compare, checks, save ===================
eval_after, res_after = evaluate("1.5B_grpo")
ev = pd.concat([eval_before, eval_after], ignore_index=True)

def paired(a, b, n=10000):
    d = b.values - a.values
    rng = np.random.default_rng(0)
    bo = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"mean": round(100 * float(d.mean()), 2),
            "ci95": [round(100 * float(np.percentile(bo, 2.5)), 2),
                     round(100 * float(np.percentile(bo, 97.5)), 2)],
            "p_le_0": round(float((bo <= 0).mean()), 4)}

key = ["set", "id", "t"]
A = eval_before.set_index(key); B = eval_after.set_index(key).loc[A.index]
paired_res = {"E1_f1_after_minus_before": paired(A[A.index.get_level_values(0) == "E1"].f1,
                                                 B[B.index.get_level_values(0) == "E1"].f1)}
for t in E2_STEPS:
    sel = (A.index.get_level_values(0) == "E2") & (A.index.get_level_values(2) == t)
    if sel.any():
        paired_res[f"E2_t{t}_f1_after_minus_before"] = paired(A[sel].f1, B[sel].f1)
        paired_res[f"E2_t{t}_f1_after_minus_7b"] = paired(A[sel].f1_7b.astype(float), B[sel].f1)

L = pd.DataFrame(log)
k = max(1, min(25, len(L) // 2))
checks = {
    "training ran all steps":          len(L) == N_STEPS,
    "no NaN in log":                   bool(np.isfinite(L[["reward_mean", "kl"]].values).all()),
    "groups carry signal (>=25% live)": L.live_groups.mean() >= 0.25 * PROMPTS_PER_STEP,
    "E2 pools rebuilt faithfully (>=95%)": REBUILD_MATCH >= 0.95,
    "answers not empty after training (<5%)": (eval_after.pred.str.strip() == "").mean() < 0.05,
}
checks = {a: bool(b) for a, b in checks.items()}
for a, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {a}")

summary = {"meta": {"run_tag": RUN_TAG, "smoke": SMOKE, "model": MODEL, "n_steps": N_STEPS,
                    "prompts_per_step": PROMPTS_PER_STEP, "G": G, "lr": LR, "kl_beta": KL_BETA,
                    "lora": [LORA_R, LORA_ALPHA], "n_train_q": len(train_items),
                    "n_e1": len(e1_items), "n_e2_items": len(e2_items),
                    "e2_rebuild_match": round(REBUILD_MATCH, 4), "checks": checks},
           "train": {"reward_first": round(float(L.reward_mean.head(k).mean()), 4),
                     "reward_last": round(float(L.reward_mean.tail(k).mean()), 4),
                     "f1_first": round(100 * float(L.f1_mean.head(k).mean()), 2),
                     "f1_last": round(100 * float(L.f1_mean.tail(k).mean()), 2),
                     "kl_last": round(float(L.kl.tail(k).mean()), 5),
                     "sec_per_step": round(float(L.sec.mean()), 1)},
           "before": res_before, "after": res_after, "paired": paired_res}
tag = f"{RUN_TAG}{'_smoke' if SMOKE else ''}"
L.to_csv(f"{OUT}/{tag}_train_log.csv", index=False)
ev.to_csv(f"{OUT}/{tag}_eval.csv", index=False)
with open(f"{OUT}/{tag}_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
model.save_pretrained(f"{OUT}/{tag}_adapter")
print("\n----- PASTE THIS BACK -----")
print(json.dumps(summary))
print("---------------------------")


# %% CELL 9 — push results + adapter to GitHub ================================
def push(local, remote):
    url = f"https://api.github.com/repos/{REPO}/contents/{remote}"
    old = requests.get(url, headers=GH)
    body = {"message": f"{tag}: {remote}",
            "content": base64.b64encode(open(local, "rb").read()).decode()}
    if old.status_code == 200:
        body["sha"] = old.json()["sha"]
    r = requests.put(url, headers=GH, json=body)
    print(remote, "->", r.status_code)
    assert r.status_code in (200, 201), r.text[:300]

for kind in ("summary.json", "train_log.csv", "eval.csv"):
    push(f"{OUT}/{tag}_{kind}", f"results/{tag}_{kind}")
for fn in os.listdir(f"{OUT}/{tag}_adapter"):
    push(f"{OUT}/{tag}_adapter/{fn}", f"adapters/{tag}/{fn}")
print(f"pushed to github.com/{REPO}")
