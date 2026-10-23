# AC-LSCM: Action-Conditioned Latent Structural Causal Models

> Outcome-Aware Agents: From Token Prediction to Action Consequence Modeling

A research codebase for training agents that simulate the outcomes of their actions before executing them, instead of predicting the next token.

[![Paper](https://img.shields.io/badge/paper-Zenodo-blue)](https://doi.org/10.5281/zenodo.20379456)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-red)](https://pytorch.org)

---

## The Problem

LLM agents are increasingly deployed on production systems. They write code that executes, run database migrations, navigate browsers. Their training objective is next-token prediction. Nothing in that objective rewards an accurate model of what their actions actually do.

A code agent that proposes `DROP TABLE` has not modeled the resulting database state. It has predicted a plausible token sequence.

This repository implements one architectural response to that gap.

## The Idea

Give the agent an internal forward model. Before executing a candidate action, simulate the resulting state. If the simulated outcome violates the goal or safety predicates, reject the action.

AC-LSCM is the architecture proposed in the paper. It combines:

- A disentangled latent representation of world state
- A learned sparse directed acyclic graph over latent factors
- A transition operator that implements Pearl's `do`-operator as graph surgery, not context concatenation
- A three-step counterfactual inference loop (abduction, intervention, prediction)
- A planning loop that filters actions through simulated outcomes before execution

## Headline Results (third version)

> **Correction.** The first version of the paper (Zenodo, May 2026) reported ~36x fewer safety
> violations than a Transformer. That result does not hold up: the originally specified model learned
> an **empty causal graph in every run**, and the agent task only tested the direct effect of an action
> with unsafe values outside the training range. See Section 6 of the current paper.

The current paper is a controlled study: 7 random SCMs (5 with 10 variables, 2 with 20), a consequence
task configured by a fixed rule (safety/goal variables downstream of the actions, 3-step horizon),
3 seeds per model and environment, 408 training runs. Excess violations are normalized per environment:
0 = oracle (true SCM), 1 = a planner that predicts no effect.

| Forward model (N = 10,000) | Rollout MSE, ER-10 | Excess viol., ER-10 (1 var / 2 vars) | Excess viol., ER-20 (1 var / 2 vars) |
|---|---:|---:|---:|
| **AC-LSCM, factored** (factors = observed variables, hard do) | **0.009** | **0.25 / 0.18** | **0.41 / 0.40** |
| AC-LSCM, latent (learned encoder) | 0.032 | 0.68 / 0.73 | 0.90 / 0.95 |
| Vanilla VAE + do-wrapper | 0.011 | 0.31 / 0.28 | 0.57 / 0.54 |
| Dreamer-style + do-wrapper | 0.011 | 0.38 / 0.36 | 0.53 / 0.56 |
| Small Transformer + do-wrapper | 0.023 | 0.36 / 0.37 | 0.59 / 0.59 |
| AC-LSCM, factored, true graph (reference) | 0.009 | 0.24 / 0.16 | 0.34 / 0.36 |

- The factored AC-LSCM recovers the exact graph on all ER-10 graphs and has lower 3-step rollout error
  than every baseline in **all 396 paired comparisons**. Its safety margin is largest for two-variable
  actions never seen in training, on 20-variable graphs, and with little data.
- The same model with learned latent factors is worse than the baselines: interventions land on latent
  coordinates that do not correspond to the observed variables.
- The do-wrapper (impose the correct action semantics on any model) removes much of the baselines' deficit;
  the explicit do-operator adds little once the factors are right.
- Counterfactual training data, given to every model, helps structured models only.
- Every planner remains far from the oracle.

## Honest Caveats

1. **Synthetic, matched setting.** Lagged additive-noise SCMs with K <= 20, observations = noisy state,
   known action targets: the setting most favourable to the factored model. No language-domain evaluation.
2. **Single-variable actions on ER-10:** the factored model's margin over the best baseline (VAE + do-wrapper)
   is within noise; the margins on two-variable actions and ER-20 are clear.
3. **ER-20 rests on two graphs**; baselines keep the first version's hyperparameters.
4. **First-version claims withdrawn:** the 36x safety result, precise structure recovery (SHD 8 was the
   empty graph), and the do-operator "carrying the result".

## Quick Start

### Install

```bash
git clone https://github.com/mplusm/AC-LSCM-Outcome-Aware-Research-Paper.git
cd AC-LSCM-Outcome-Aware-Research-Paper

# For NVIDIA T4 / CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Smoke test (5 minutes)

```bash
python -m src.train --config configs/synthetic_chain.yaml --seed 0 --epochs 1
```

If this completes without errors, your environment is set up correctly.

### Reproduce the current results

```bash
# One (environment, variant, seed, training-set size) per process; see VARIANTS in scripts/run_v3.py
python scripts/run_v3.py --env er10_s1 --variant fact_sparse --seed 0
python scripts/run_v3.py --env er20_s0 --variant transformer --seed 2 --n-train 10000
python scripts/v3_references.py er10_s0 er10_s1 er10_s2 er10_s3 er10_s4 er20_s0 er20_s1

# Aggregate and write the paper tables
python scripts/aggregate_v3.py > tables/v3_summary.json
python scripts/make_v3_tables.py results_v3 paper/source/generated
python scripts/make_revision_tables.py tables/revision_summary.json paper/source/generated --v1-only
```

The intermediate (second) revision's runs are in `results_v2/` (`scripts/run_revision.py`).

### Reproduce the first-version results

```bash
# All Tier 1 experiments + ablations + agent task (~40 GPU-hours on T4)
bash scripts/run_all.sh

# Generate paper tables from results
python scripts/make_tables.py --results-dir results/ --output tables/
```

### Single experiment

```bash
python -m src.train \
  --config configs/synthetic_er_k10.yaml \
  --seed 0
```

### Resume after preemption

```bash
python -m src.train \
  --config configs/synthetic_er_k10.yaml \
  --seed 0 \
  --resume
```

## Repository Layout

```
ac-lscm/
├── README.md                  # this file
├── LICENSE                    # Apache 2.0
├── requirements.txt
├── paper/                     # preprint PDF + LaTeX source
├── configs/                   # experiment configs (YAML)
│   ├── synthetic_chain.yaml
│   ├── synthetic_fork.yaml
│   ├── synthetic_collider.yaml
│   ├── synthetic_er_k10.yaml
│   ├── synthetic_er_k20.yaml
│   └── ablations.yaml
├── src/
│   ├── scm.py                 # ground-truth SCM data generator
│   ├── models.py              # AC-LSCM and three baselines
│   ├── losses.py              # four-term composite loss
│   ├── train.py               # training loop with checkpointing
│   ├── eval.py                # evaluation metrics
│   ├── planning.py            # Algorithm 1 (first-version agent task)
│   ├── agent_tasks.py         # direct-effect and consequence tasks, do-wrapper, rollout error
│   └── utils.py               # seeds, logging, DAG metrics
├── scripts/
│   ├── run_all.sh             # reproduces first-version experiments
│   ├── make_tables.py         # first-version LaTeX tables
│   ├── run_v3.py              # current experiments (7 environments, all variants)
│   ├── v3_references.py       # oracle and no-model references per environment
│   ├── aggregate_v3.py, make_v3_tables.py
│   ├── run_revision.py        # intermediate revision (single environment)
│   ├── aggregate_revision.py
│   └── make_revision_tables.py  # first-version diagnosis tables
├── results/                   # first-version per-seed JSON outputs
├── results_v2/                # intermediate revision per-seed JSON outputs
├── results_v3/                # current per-seed JSON outputs (408 runs)
└── tables/                    # generated LaTeX tables
```

## Architecture Components

The architecture is documented in detail in the paper. A brief summary:

**Encoder** `q_phi(z_t | x_t)`: 2-layer MLP, maps observations to K disentangled latent factors.

**Decoder** `p_theta(x_t | z_t)`: 2-layer MLP, reconstructs observations from latents.

**Transition operator** `p_psi(z_{t+1} | z_t, do(a_t))`: per-factor MLPs `f_i` with structural semantics. When the action targets factor `i`, the parent contribution is severed and the intervention value substituted. This is the `do`-operator made explicit at the computation graph level.

**Adjacency** `A`: learned `K x K` matrix, Gumbel-Sigmoid during training, thresholded at evaluation. The original model uses soft samples with the NOTEARS constraint plus L1 sparsity (this learns an empty graph); the simplified model uses straight-through hard samples, no acyclicity term (parents are lagged, so the graph is acyclic by construction) and a smaller L1 weight.

**Counterfactual inference**: three steps.
1. Abduction: recover `epsilon_t` from observed `(x_t, a_t, x_{t+1})`.
2. Intervention: substitute alternative action `a_tilde`.
3. Prediction: roll structural equations forward with the same `epsilon_t`, decode.

**Planning loop** (Algorithm 1 in paper): for each candidate action, simulate the next state, filter by safety predicate, rank survivors by goal achievement and value function, defer if no safe candidate exists.

## Baselines

The repo includes three baselines for comparison.

- **VanillaVAE**: same encoder/decoder, single-MLP transition with action concatenated.
- **DreamerStyle**: same as VanillaVAE but with a GRU transition cell.
- **SmallTransformer**: treats `(x_t, a_t)` as a token sequence, predicts `x_{t+1}`. It is a sequence-model forward model used in the same planner, not a language-model agent.

## Reproducibility

All experiments use deterministic seeding (`torch.manual_seed`, `numpy.random.seed`, `random.seed`) and `torch.use_deterministic_algorithms(True, warn_only=True)`. CuDNN is set to deterministic mode.

Per-seed result JSONs are produced for every run and validated against a schema. The `results/summary.json` file aggregates all metrics.

Compute environment:
- NVIDIA Tesla T4 (16 GB VRAM)
- Ubuntu 22.04, CUDA 12.1
- PyTorch 2.1+, Python 3.10+
- fp32 throughout (no mixed precision; T4 fp16 dynamic range is too narrow for the NOTEARS matrix exponential)

## Citing

If you use this code or build on the ideas, please cite:

```bibtex
@misc{madapathi2026aclscm,
  title  = {Outcome-Aware Agents: When Does Causal Structure Help a Forward Model Foresee the Consequences of Actions?},
  author = {Madapathi, Mallesh},
  year   = {2026},
  doi    = {10.5281/zenodo.20379456},
  url    = {https://doi.org/10.5281/zenodo.20379456},
  note   = {Preprint}
}
```

## Roadmap

- [x] Consequence task with downstream effects, in-range actions, fixed configuration rule, 7 environments.
- [x] Fair counterfactual-data comparison and do-wrapper control.
- [ ] Uncertainty-aware planning (calibrated rollouts, conservative safety margins).
- [ ] Misspecified mechanisms, partial observability, unknown intervention targets.
- [ ] Connect to a language-model agent (variables and actions extracted from tool calls).
- [ ] Scale beyond `K=20`.

## Contributing

This is a research codebase, not a library. If you find a bug, an issue is welcome. If you replicate or refute a result, a PR with your numbers and configs is very welcome. If you have a strong opinion about the architecture, the paper's Limitations section is where to start a discussion.

The most useful contributions right now are:

1. Replication on different hardware. The published results are from a single T4.
2. Trying the simplified architecture from the ablation section.
3. Evaluating on language benchmarks (CLadder, CounterBench).

## License

Apache 2.0. See [LICENSE](LICENSE).

## Acknowledgments

This work builds on causal representation learning (Schölkopf et al.), continuous DAG optimization (NOTEARS, DAGMA), action-conditioned world models (Ha and Schmidhuber, Dreamer), and Pearl's structural causal framework. References in the paper.

Thanks to the open research community whose published artifacts, benchmarks, and code made this work possible.

## Contact

Mallesh Madapathi
thinkingdbx pvt. ltd., Hyderabad, India
`mallesh@thinkingdbx.com`

For methodology questions, open a GitHub issue. For collaboration or research discussion, email is faster.
