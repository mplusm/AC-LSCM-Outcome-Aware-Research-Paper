# AC-LSCM: Action-Conditioned Latent Structural Causal Model

Experiment code for the paper *AC-LSCM: Outcome-Aware Action Selection via Learned Structural Causal Models*.

## Hardware target
Single NVIDIA Tesla T4 (16 GB VRAM, fp32 only). Designed for ~36–48 GPU-hours on a GCP spot instance.

## Setup

```bash
# T4 / CUDA 12.x
pip install -r requirements.txt
```

## Smoke test (run this first)
```bash
python -m src.train --config configs/synthetic_chain.yaml --seed 0 --epochs 1
```

## Single experiment
```bash
python -m src.train --config configs/synthetic_chain.yaml --seed 0
```

## Resume after preemption
```bash
python -m src.train --config configs/synthetic_chain.yaml --seed 0 --resume
```

## Full run (~36–48 GPU-hours on T4)
```bash
bash scripts/run_all.sh
```

## Generate paper tables
```bash
python scripts/make_tables.py --results-dir results/ --output tables/
```

## Repo layout
```
ac-lscm/
├── configs/           YAML experiment configs
├── src/
│   ├── scm.py         Ground-truth SCM data generator
│   ├── models.py      AC-LSCM and baselines
│   ├── losses.py      Four-term composite loss
│   ├── train.py       Training loop
│   ├── eval.py        Evaluation metrics
│   ├── planning.py    Algorithm 1: outcome-aware action selection
│   └── utils.py       Utilities
├── scripts/
│   ├── run_all.sh     Full experiment runner
│   └── make_tables.py LaTeX table generator
└── results/           Output directory (created at runtime)
```
