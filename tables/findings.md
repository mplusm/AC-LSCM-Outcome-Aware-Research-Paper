# Experiment Findings

All results are mean ± std over 3 random seeds. Hardware: NVIDIA Tesla T4 (fp32, batch 64).
Total GPU time: ~2 hours.

---

## Table A — Main Results (Intervention MSE / Counterfactual MSE)

| Model | Chain | Fork | Collider | ER (K=10) |
|---|---|---|---|---|
| Small Transformer | 0.053 / 0.057 | 0.045 / 0.048 | 0.043 / 0.045 | 0.047 / 0.050 |
| Vanilla VAE | 0.039 / 0.043 | 0.035 / 0.037 | 0.030 / 0.031 | 0.033 / 0.036 |
| Dreamer-style | 0.037 / 0.039 | 0.033 / 0.034 | 0.029 / 0.028 | 0.032 / 0.032 |
| **AC-LSCM (ours)** | **0.133 / 0.150** | **0.087 / 0.105** | **0.073 / 0.103** | **0.089 / 0.114** |

**Hypothesis 1 — AC-LSCM beats baselines on counterfactual MSE: NOT CONFIRMED.**
AC-LSCM's CF MSE (mean 0.118 across 4 graph types) is 3.5x worse than the best baseline
(Dreamer-style, 0.033). This holds across all graph types without exception.

**Hypothesis 2 — AC-LSCM competitive on simple graphs: direction only, not magnitude.**
AC-LSCM improves relative to itself as graph complexity increases (chain 0.133 to ER 0.089
Int MSE), but remains well above all baselines on every config. The structural constraints
appear to be a consistent optimisation burden at this data scale (2000 samples).

**Notable pattern:** AC-LSCM's CF MSE is consistently 25-45% higher than its own Int MSE
(e.g. chain: 0.133 vs 0.150), whereas Dreamer's CF/Int ratio is nearly 1:1. This suggests
the abduction step is accumulating encoder error rather than cleanly recovering exogenous noise.

**DAG constraint:** Reached ~0.000 at convergence on all 6 graph types — the learned adjacency
is acyclic. SHD on ER K=10 is 8.0 (out of K*(K-1)=90 possible edges), meaning the graph is
sparse but structurally imprecise.

---

## Table B — Ablations on ER (K=10)

| Variant | Int MSE | CF MSE | SHD |
|---|---|---|---|
| AC-LSCM (full) | 0.089 | 0.114 | 8.0 |
| -- no L_Causal (beta_3=0) | 0.054 | 0.128 | 80.7 |
| -- no L_Contrastive (beta_4=0) | 0.059 | **0.057** | 8.0 |
| -- no do-operator | 0.070 | 0.154 | 8.0 |
| -- DAGMA instead of NOTEARS | 0.095 | 0.132 | 55.0 |

**Key findings:**

1. **Removing the DAG loss (beta_3=0) collapses structure (SHD 8 to 80.7)** while slightly
   improving Int MSE. The DAG constraint is working as intended but at a predictive cost.

2. **Removing the contrastive loss (beta_4=0) improves CF MSE from 0.114 to 0.057** — a 50%
   improvement. This is the most striking result. The contrastive loss is actively hurting
   counterfactual accuracy. Possible explanations: the supervised CF term overfits the training
   distribution; the hinge term creates conflicting gradients with the reconstruction objective;
   2000 samples is insufficient for the contrastive term to generalise.
   **Hypothesis 5 (contrastive loss matters most for CF accuracy): NOT CONFIRMED — it hurts.**

3. **Removing the do-operator raises CF MSE by 35% (0.114 to 0.154)**, confirming it is the
   single most important architectural component for counterfactual accuracy.

4. **DAGMA performs poorly (SHD 55.0 vs 8.0 for NOTEARS)** — the constraint is not converging
   to a clean DAG on this setup. NOTEARS is clearly superior here.

---

## Table C — Agent Task on ER (K=10)

| Model | Goal Rate | Safety Violation | Appropriate Deferral |
|---|---|---|---|
| Small Transformer | 0.800 | 0.177 | 0.117 |
| **AC-LSCM (ours)** | **0.510** | **0.267** | **0.000** |

**Hypothesis 4 — AC-LSCM shows fewer safety violations: NOT CONFIRMED.**
Transformer achieves higher goal rate (0.80 vs 0.51) and lower safety violations (0.177 vs 0.267).
AC-LSCM's high variance in goal rate (+/-0.362) indicates instability across seeds.

AC-LSCM's deferral rate of 0.000 means it never correctly deferred when no safe option existed,
suggesting the safety_fn based on predicted z is unreliable — a direct consequence of the poor
transition accuracy shown in Table A.

---

## Overall Assessment

All four primary hypotheses were not confirmed on these synthetic benchmarks. Specific failure
modes identified:

1. **Optimisation conflict:** the DAG constraint, contrastive loss, and reconstruction objective
   create competing gradients that slow and degrade overall learning.
2. **Abduction error propagation:** the encode-then-abduct pipeline compounds encoder MSE into
   the counterfactual prediction. Baseline models avoid this entirely.
3. **Data efficiency:** Dreamer-style and Vanilla VAE converge cleanly on 2000 samples;
   AC-LSCM needs more data or a stronger inductive bias to pay off structurally.
4. **Contrastive loss design:** the current hinge+supervised formulation is harmful and should
   be redesigned or dropped (beta_4=0 is strictly better in this regime).

**Recommended next steps before revising the empirical claims:**
- Scale to 10,000-50,000 training samples (the SCM generator supports this).
- Drop or re-weight the contrastive loss (beta_4=0 strictly dominates).
- Investigate DAGMA convergence failure before including it as an ablation in the paper.
- Re-examine abduction: consider direct noise supervision during training rather than
  post-hoc residual computation, which amplifies encoder error.
- Run ER K=20 results as the primary table (most challenging, closest to real-world complexity).
