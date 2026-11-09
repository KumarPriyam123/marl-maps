# MARL-MAPS — Results

Every number below was measured in this repo and comes from the file named next to it in `results/`. Nothing is projected or borrowed from other papers.

**Setup (all steps).** HotpotQA (distractor, validation). The corpus holds the 9,769 unique context paragraphs of the first 1,000 questions, retrieved with all-MiniLM-L6-v2 and FAISS (cosine), top-3 per query, at most 3 extra searches. Prompts are MAPS Appendix A. Greedy decoding. Paired bootstrap (10k) gives 95% CIs. Question slices never overlap across purposes:

| Questions | Used for |
|---|---|
| 0–199 | Step 1 baselines, Step 2 calibration fit |
| 200–399 | Step 3 orchestrator test |
| 400–799 | Step 4 stop-policy training (and 6c policy training) |
| 800–999 | **Held-out test** (Steps 5, 6a E2, 6b E2, 6c) |
| HotpotQA *train* split | GRPO training of the 1.5B agents (6a, 6b) |

---

## Headline (Step 6c, held-out questions 800–999, n = 200)

| System | F1 | EM | Extra searches | LLM calls | Calls × params (B)* |
|---|---|---|---|---|---|
| 1.5B pipeline, untrained | 36.8 | 26.0 | 0.15 | 2.29 | 3.5 |
| **1.5B pipeline, GRPO-trained query + answer agents** | **56.4** | **46.0** | **0.75** | **3.49** | **5.4** |
| 7B pipeline (Qwen2.5-7B-Instruct-AWQ) | 57.0 | 44.0 | 0.84 | 3.67 | 28.0 |

Each system uses its own stop policy, learned on questions 400–799. *Calls × params is a rough compute proxy, not measured latency.

| Paired comparison | F1 (95% CI) | Extra searches (95% CI) |
|---|---|---|
| Trained 1.5B − untrained 1.5B | **+19.6** [+13.8, +25.4] | +0.60 [+0.52, +0.69] |
| Trained 1.5B − 7B | −0.6 [−5.7, +4.4] (tie) | −0.09 [−0.19, +0.01] (tie) |
| Trained 1.5B − 7B with judge-stop | +0.6 [−4.8, +5.9] (tie) | **−0.48** [−0.66, −0.30] |

**Claim:** a 1.5B multi-agent RAG pipeline with GRPO-trained query and answer agents and a learned stop policy matches the 7B pipeline on held-out questions, with a model 4.9× smaller and about 5× less estimated compute per question. Source: `step6c_system.json`.

---

## Step by step

### 1. Baselines (Q 0–199) — `step1_summary.json`, `step2_calibration.json`
| | EM | F1 | Retrievals |
|---|---|---|---|
| Naive RAG | 38.5 | 48.5 | 1.00 |
| MAPS (scripted, Eq. 8–11) | 43.5 | 53.7 | 2.05 |

MAPS − naive: EM **+5.0** [+0.5, +9.5], F1 **+5.3** [+1.0, +9.8]. MAPS waste: 17.6% of extra retrievals return nothing new, and 26.2% happen after all gold passages are already in the pool.

### 2. Judge calibration (Q 0–199 judge calls) — `step2_calibration.json`
| | ECE | Brier |
|---|---|---|
| Raw judge confidence | 0.286 | 0.296 |
| Temperature scaling | 0.092 | 0.213 |
| Platt scaling | **0.061** | 0.209 |

AUROC is only 0.74, so even calibrated confidence is too weak to be a safe stop rule (best stop precision ≈ 65%).

### 3. Rule-based orchestrator + SGWM (Q 200–399) — `step3_n200_summary.json`
| | EM | F1 | Gold coverage | Redundant retrievals | LLM calls |
|---|---|---|---|---|---|
| Naive | 32.5 | 46.1 | 36.5% | — | 1.00 |
| MAPS | 42.5 | 57.3 | 54.5% | 14.1% | 3.76 |
| Orch-rules | 43.0 | 57.7 | 59.5% | 0% (by design) | 3.98 |

MAPS − naive EM **+10.0** [+6.0, +14.5] (replicates Step 1). Orch − MAPS EM +0.5 [−3.5, +4.5], a tie. The rules find more gold but do **not** cut cost; stopping stays the bottleneck.

### 4–5. Learned stop policy — `step4_policy.json`, `step5_heldout.json`
Process reward R(t) = F1ₜ − 0.05·t on full trajectories; logistic policy on SGWM state; trained on Q 400–799, **frozen**, tested on Q 800–999 (7B pipeline).

| Policy (held out) | F1 | Extra searches | LLM calls |
|---|---|---|---|
| Judge-stop (MAPS) | 55.8 | 1.22 | 4.44 |
| Fixed: one search | 56.9 | 1.00 | 4.00 |
| **Learned** | **57.0** | **0.84** | **3.67** |
| Always full budget | 60.5 | 3.00 | 8.00 |

Learned − judge-stop: F1 +1.2 (tie), searches **−0.39** [−0.53, −0.25] (**−32%**). Learned − fixed-one-search: F1 tie, searches −0.17 [−0.25, −0.09]. The same direction was seen in Step 4 cross-validation (−18%).

### 6a. GRPO answer agent, Qwen2.5-1.5B + LoRA — `step6a_summary.json`
Trained on the HotpotQA train split, reward F1 + 0.5·EM, G = 4, 250 steps. Evaluated on the **exact** Step 5 evidence pools (rebuilt with a 100% match).

| Pools after | Untrained 1.5B | GRPO 1.5B | Δ (95% CI) | 7B on same pools |
|---|---|---|---|---|
| 0 searches | 35.7 | 43.3 | +7.6 [+2.9, +12.3] | 44.1 |
| 1 search | 43.8 | 54.1 | +10.3 [+5.1, +15.8] | 56.9 |
| 3 searches | 47.9 | 58.1 | +10.2 [+5.1, +15.5] | 60.5 |

GRPO vs 7B: every CI includes 0. Mostly learned **answer format and precision** (answer length 4.0 → 2.3 words; gold 2.5).

### 6b. GRPO query agent, Qwen2.5-1.5B + LoRA — `step6b_summary.json`
Reward = share of missing gold passages retrieved (−0.2 for a duplicate or empty query), G = 8, 200 steps, train-split corpus. Answer agent held fixed (6a).

| Query source | All gold after 1 search | Answer F1 |
|---|---|---|
| Untrained 1.5B | 52.0% | 45.9 |
| **GRPO 1.5B** | **67.0%** | **57.4** |
| 7B (logged) | 67.5% | 54.1 |

GRPO − untrained: coverage **+15.0** [+8.0, +22.0], F1 **+11.5** [+6.0, +17.0]. GRPO − 7B: coverage −0.5 [−6.0, +5.0], a tie. The agent learned to **name the bridge entity** ("David Dunn", "Key West") instead of copying the question.

### 6c. Full system — see Headline.

---

## Limitations (state these)
- **One dataset, small corpus:** HotpotQA only, 9.8k passages, 200 test questions. Not comparable to papers that retrieve from full Wikipedia.
- **The judge is untrained** in the 1.5B pipelines (AUROC 0.62; says "sufficient" 6% of the time). A *perfect* judge is worth at most +0.6 F1 at equal cost, or +4.6 F1 at 65% more searches (simulated from logged gold coverage).
- **Accuracy is a tie, not a win.** The gains are efficiency and model size. With unlimited evidence the 7B is still ahead (60.5 vs 56.5, CI includes 0).
- **The "learned" orchestration is a stop policy trained offline** (contextual bandit with process rewards), not online multi-agent RL. Two agents are trained independently with GRPO toward the task reward.
- **Compute is a proxy** (calls × parameters). The 1.5B and 7B ran on different serving stacks, so wall-clock times aren't comparable.
- **Training notes:** 6a resumed from a 5-step smoke checkpoint (about 3 E1 questions were possibly seen; E2 unaffected). KL drift was 0.38 (6a) and 0.72 (6b).
