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

## Headline Results

On a safety-critical agent planning task (ER K=10, 13 seeds):

| Model | Goal Rate | Safety Violations | Appropriate Deferral |
|-------|-----------|-------------------|----------------------|
| Small Transformer (n=3) | 0.800 ± 0.000 | 0.180 ± 0.022 | 0.100 ± 0.108 |
| AC-LSCM, all seeds (n=13) | 0.554 ± 0.369 | **0.005 ± 0.019** | **0.985 ± 0.053** |
| AC-LSCM, working seeds (n=9) | 0.800 ± 0.000 | **0.000 ± 0.000** | **1.000 ± 0.000** |

Roughly 36x fewer safety violations than the Transformer baseline on average. 12 of 13 seeds produce zero safety violations. The Transformer's 0.800 goal rate is the task ceiling, which working AC-LSCM seeds tie.

See the paper for the full attribution control (architectural fixes do the work, not data volume), the ablation results (three of four ablations improve the architecture), and the failure-mode analysis (about a third of seeds fail to produce a usable planner despite normal training-time MSE).

## Honest Caveats

Reading this before running the code will save you time.

1. **Training is unstable.** Roughly 31% of seeds (4 of 13) failed to produce a usable planner despite normal MSE. Two distinct failure modes: a passive policy that never picks the goal action, and a value mis-rank that occasionally picks unsafe.
2. **Three of four ablations improve the architecture.** The DAG loss and contrastive hinge are net-negative at the scales tested. The paper recommends a simplified follow-up. This codebase reproduces the original architecture as documented; the simpler version is on the roadmap.
3. **No language-domain evaluation yet.** All experiments are on synthetic structural causal models with `K` up to 20. CLadder and CounterBench evaluation is future work.
4. **Small experiments.** Total compute for the published results was under 50 GPU-hours on a single T4. Do not expect frontier-scale claims.

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

### Reproduce the headline result

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
│   ├── planning.py            # Algorithm 1: outcome-aware action selection
│   └── utils.py               # seeds, logging, DAG metrics
├── scripts/
│   ├── run_all.sh             # reproduces all experiments
│   └── make_tables.py         # generates LaTeX tables
├── results/                   # per-seed JSON outputs (gitignored except summary)
└── tables/                    # generated LaTeX tables
```

## Architecture Components

The architecture is documented in detail in the paper. A brief summary:

**Encoder** `q_phi(z_t | x_t)`: 2-layer MLP, maps observations to K disentangled latent factors.

**Decoder** `p_theta(x_t | z_t)`: 2-layer MLP, reconstructs observations from latents.

**Transition operator** `p_psi(z_{t+1} | z_t, do(a_t))`: per-factor MLPs `f_i` with structural semantics. When the action targets factor `i`, the parent contribution is severed and the intervention value substituted. This is the `do`-operator made explicit at the computation graph level.

**Adjacency** `A`: learned `K x K` matrix, soft during training (Gumbel-Sigmoid), hard at evaluation. Trained with the NOTEARS continuous DAG constraint plus L1 sparsity.

**Counterfactual inference**: three steps.
1. Abduction: recover `epsilon_t` from observed `(x_t, a_t, x_{t+1})`.
2. Intervention: substitute alternative action `a_tilde`.
3. Prediction: roll structural equations forward with the same `epsilon_t`, decode.

**Planning loop** (Algorithm 1 in paper): for each candidate action, simulate the next state, filter by safety predicate, rank survivors by goal achievement and value function, defer if no safe candidate exists.

## Baselines

The repo includes three baselines for comparison.

- **VanillaVAE**: same encoder/decoder, single-MLP transition with action concatenated.
- **DreamerStyle**: same as VanillaVAE but with a GRU transition cell.
- **SmallTransformer**: treats `(x_t, a_t)` as a token sequence, predicts `x_{t+1}`. This is the closest analog to a token-prediction agent.

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
  title  = {Outcome-Aware Agents: From Token Prediction to Action Consequence Modeling},
  author = {Madapathi, Mallesh},
  year   = {2026},
  doi    = {10.5281/zenodo.20379456},
  url    = {https://doi.org/10.5281/zenodo.20379456},
  note   = {Preprint}
}
```

(Replace `20379456` with the actual DOI from Zenodo after upload.)

## Roadmap

What is queued for the next version, in priority order.

- [ ] Retrain the simplified architecture (no contrastive hinge, soft DAGMA regularizer or no structural loss) and report whether the training instability persists.
- [ ] Make the agent task harder: raise `no_safe_frac` from 0.2 to 0.5, add upstream-effect distractors, lower the goal-action margin so the goal rate metric discriminates above the current 0.800 ceiling.
- [ ] Add a degenerate-seed detector during training (rank correlation between predicted and ground-truth `z_{K-1}` on a validation planning task; flag below threshold).
- [ ] Track `inappropriate_deferral_rate` and `safe_non_goal_pick_rate` as headline metrics so failure modes are visible without per-seed reverse-engineering.
- [ ] Evaluate on CLadder and CounterBench.
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
