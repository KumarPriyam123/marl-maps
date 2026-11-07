# MARL-MAPS

An extension of MAPS (a multi-agent RAG system): a learned orchestrator, a shared memory and process rewards on top of three specialised agents. Team: Kumar Priyam, Vinayak, Twinkle (NIT Delhi).

## Where things run

| Where | What runs there | Why |
|---|---|---|
| Laptop (i7, no GPU) | Code, smoke tests, Claude Code development | Plumbing needs no GPU |
| Kaggle (GPU T4 x1) | **Every reported number** | The 7B model needs a GPU |
| GitHub (private repo) | The single copy of truth | So work is never lost again |

**Rule:** a number goes into the report only if a `results/stepN_summary.json` file in this repo produced it.

## Steps

| # | Step | Done when |
|---|---|---|
| 0 | Repo on GitHub + local smoke test | `python run_smoke_test.py` prints a RESULTS block; first commit pushed |
| 1 | Baselines on Kaggle (`kaggle/step1_kaggle.py`) | 200 questions, all 6 checks PASS, summary committed to `results/` |
| 2 | Judge confidence + calibration | ECE measured before/after temperature scaling |
| 3 | SGWM + rule-based orchestrator | Fewer retrievals than Step 1 at equal or better F1 |
| 4 | Process rewards on logged runs | Reward tracks F1 and redundant retrievals |
| 5 | GRPO-trained orchestrator (LoRA) | Trained policy beats Step 3 |
| 6 | MuSiQue + ablations + write-up | Every claim traceable to a results file |

## Step 0 — local setup (Windows, Git Bash)

```bash
cd marl-maps
py -m venv venv                 # use py, not python (avoids the Store/MSYS Python)
source venv/Scripts/activate    # check: `which python` points inside venv/
python run_smoke_test.py        # no installs needed: mock LLM + BM25 + 10 sample questions
git init && git add . && git commit -m "Step 0: starter repo"
```

Then create an empty **private** repo on GitHub named `marl-maps` and push:

```bash
git remote add origin https://github.com/KumarPriyam123/marl-maps.git
git branch -M main && git push -u origin main
```

Keep the repo folder **outside OneDrive** (e.g. `C:\dev\marl-maps`): OneDrive sync fights with `venv/` and `.git/`.

## Step 1 — Kaggle baselines

Open `kaggle/step1_kaggle.py` and follow its header. It downloads HotpotQA and builds the index on Kaggle itself, so nothing needs uploading. When it finishes, download `step1_summary.json` and `step1_questions.csv` into `results/`, commit, push.

## Layout

```
config.py            backend / retriever / top_k / max_rounds
llm_backend.py       one interface: mock | ollama | groq | gemini
run_smoke_test.py    one-command sanity check (no deps)
agents/              MAPS prompts (paper Appendix A) + the 3 agents
retriever/           BM25 (zero-dep) + dense (FAISS)
memory/sgwm.py       shared working memory (used from Step 3)
pipeline/            scripted MAPS loop (Eq. 8-11)
eval/                EM / F1 + eval runner
data/                loaders + 10-question offline sample
kaggle/              notebooks that produce the real numbers
results/             summaries from Kaggle runs (committed)
```
