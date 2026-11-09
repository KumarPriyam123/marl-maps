# MARL-MAPS

An extension of MAPS (a multi-agent RAG system): a learned orchestrator, a shared memory and process rewards on top of three specialised agents. Team: Kumar Priyam, Vinayak, Twinkle (NIT Delhi).

## Where things run

| Where | What runs there | Why |
|---|---|---|
| Laptop (i7, no GPU) | Code, smoke tests, Claude Code development | Plumbing needs no GPU |
| Kaggle (GPU T4) | **Every reported number** | The models need a GPU |
| GitHub (private repo) | The single copy of truth | So work is never lost again |

**Rule:** a number goes into the report only if a file in `results/` produced it.

## Steps (all done; numbers in [RESULTS.md](RESULTS.md))

| # | Step | Notebook / script | Result file |
|---|---|---|---|
| 1 | Naive RAG vs scripted MAPS baselines | `kaggle/step1_kaggle.py` | `results/step1_summary.json` |
| 2 | Judge confidence calibration | `scripts/step2_calibration.py` | `results/step2_calibration.json` |
| 3 | SGWM + rule-based orchestrator | `kaggle/step3_kaggle.py` | `results/step3_n200_summary.json` |
| 4 | Full trajectories + learned stop policy | `kaggle/step4_kaggle.py`, `scripts/step4_policy.py` | `results/step4_policy.json` |
| 5 | Held-out test of the stop policy | `kaggle/step5_kaggle.py`, `scripts/step5_eval.py` | `results/step5_heldout.json` |
| 6a | GRPO answer agent (1.5B + LoRA) | `kaggle/step6a_kaggle.py` | `results/step6a_summary.json`, `adapters/step6a/` |
| 6b | GRPO query agent (1.5B + LoRA) | `kaggle/step6b_kaggle.py` | `results/step6b_summary.json`, `adapters/step6b/` |
| 6c | Full-system test vs 7B | `kaggle/step6c_kaggle.py`, `scripts/step6c_eval.py` | `results/step6c_system.json` |

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
