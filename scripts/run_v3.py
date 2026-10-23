"""Experiments added in the third version of the paper.

Every (environment, variant, training-set size, seed) is one process:

    python scripts/run_v3.py --env er10_s1 --variant fact_sparse --seed 0 [--n-train 2000]

Environments are Erdos-Renyi SCMs ``er<K>_s<graph seed>``. The consequence task
of each environment is configured by a fixed rule (src.agent_tasks.auto_task),
not by hand. Each run trains one model and evaluates

  * one-step intervention / counterfactual MSE (src.eval),
  * three-step rollout error after single- and two-factor interventions,
  * the consequence task with single-factor (in-distribution) and two-factor
    (never seen in training) candidate actions,

with the model's own action handling ("native") and with the do-wrapper
(src.agent_tasks.predict_rollout), so the effect of the action semantics can
be separated from the quality of the learned dynamics. Results go to
results_v3/<env>/<variant>_n<N>_seed<seed>/eval.json.
"""
import argparse
import json
import os
import re
import sys
from copy import deepcopy

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_tasks import (auto_task, make_consequence_episodes,  # noqa: E402
                             rollout_mse, run_consequence_task)
from src.models import ACLSCM, get_model  # noqa: E402
from src.scm import SyntheticSCM  # noqa: E402
from src.train import load_config, train_model  # noqa: E402

LATENT = {"model.acyclicity": "none", "model.init_logit": 2.0, "model.use_hinge": False,
          "model.exact_abduction": True, "model.hard_gumbel": True, "training.lambda_1": 0.01}
FACT = {**LATENT, "model.identity_obs": True, "model.hard_do": True}
NO_CF = {"training.beta_4": 0.0}
OBS_CF = {"training.cf_target": "obs"}
DENSE = {"training.lambda_1": 0.0}

VARIANTS = {
    # Unstructured baselines, without and with observed counterfactual pairs.
    "vae": ("vanilla_vae", {}),
    "dreamer": ("dreamer_style", {}),
    "transformer": ("small_transformer", {}),
    "vae_obscf": ("vanilla_vae", OBS_CF),
    "dreamer_obscf": ("dreamer_style", OBS_CF),
    "transformer_obscf": ("small_transformer", OBS_CF),
    # AC-LSCM with a learned latent space (the model of the second version).
    "aclscm_st": ("aclscm", LATENT),  # ground-truth latent CF supervision
    "aclscm_st_nocf": ("aclscm", {**LATENT, **NO_CF}),
    "aclscm_dense_nocf": ("aclscm", {**LATENT, **DENSE, **NO_CF}),
    "aclscm_st_obscf": ("aclscm", {**LATENT, **OBS_CF}),
    "aclscm_dense_obscf": ("aclscm", {**LATENT, **DENSE, **OBS_CF}),
    # Factored AC-LSCM: factors are the observed coordinates, hard do-operator.
    "fact_sparse": ("aclscm", {**FACT, **NO_CF}),
    "fact_dense": ("aclscm", {**FACT, **DENSE, **NO_CF}),
    "fact_sparse_obscf": ("aclscm", {**FACT, **OBS_CF}),
    "fact_sparse_softdo": ("aclscm", {**FACT, **NO_CF, "model.hard_do": False}),
    "fact_true": ("aclscm", {**FACT, **NO_CF, "model.fixed_adjacency": "true"}),
}


def parse_env(env):
    m = re.fullmatch(r"er(\d+)_s(\d+)", env)
    if not m:
        raise ValueError(f"bad env {env}")
    return int(m.group(1)), int(m.group(2))


def build_model(model_name, K, model_cfg):
    if model_name == "aclscm":
        return ACLSCM(K=K, obs_dim=K, use_do_operator=model_cfg.get("use_do_operator", True),
                      acyclicity=model_cfg.get("acyclicity", "notears"),
                      init_logit=model_cfg.get("init_logit", 0.0),
                      exact_abduction=model_cfg.get("exact_abduction", False),
                      hard_gumbel=model_cfg.get("hard_gumbel", False),
                      fixed_adjacency=model_cfg.get("fixed_adjacency"),
                      identity_obs=model_cfg.get("identity_obs", False),
                      hard_do=model_cfg.get("hard_do", False))
    return get_model(model_name, K=K, obs_dim=K)


def task_config(scm):
    t = auto_task(scm)
    return t, {k: t[k] for k in ("horizon", "safety_node", "goal_node", "tau_s", "tau_g")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--results-dir", default="results_v3")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=300)
    args = ap.parse_args()
    torch.set_num_threads(1)

    K, graph_seed = parse_env(args.env)
    model_name, overrides = VARIANTS[args.variant]
    cfg = deepcopy(load_config("configs/synthetic_er_k10.yaml"))
    cfg["name"] = args.env
    cfg["scm"].update(K=K, seed=graph_seed, n_train=args.n_train)
    cfg["training"]["v1_agent_eval"] = False
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    scm = SyntheticSCM("er_random", K, graph_seed, cfg["scm"]["edge_prob"])

    model_cfg = {}
    for k, v in overrides.items():
        if v == "true":
            v = scm.true_adjacency.tolist()
        section, key = k.split(".", 1)
        if section == "model":
            model_cfg[key] = v
        else:
            cfg[section][key] = v

    env_dir = os.path.join(args.results_dir, args.env)
    run_name = f"{args.variant}_n{args.n_train}"
    metrics = train_model(cfg, model_name, args.seed, env_dir,
                          ablation_name=run_name, model_cfg_overrides=model_cfg)
    run_dir = os.path.join(env_dir, f"ablation_{run_name}_seed{args.seed}")

    model = build_model(model_name, K, model_cfg)
    ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    task, conseq = task_config(scm)
    out = {"env": args.env, "variant": args.variant, "seed": args.seed, "n_train": args.n_train,
           "task": task,
           "mse": {k: metrics.get(k) for k in ("intervention_mse", "counterfactual_mse", "shd",
                                                "epochs_run")}}
    if isinstance(model, ACLSCM):
        out["mse"]["n_learned_edges"] = int(model.get_adjacency(hard=True).sum().item())
        out["mse"]["n_true_edges"] = int(scm.true_adjacency.sum())

    for space in ("native", "do_wrapper"):
        sp = "decoded" if space == "native" else space
        for nt in (1, 2):
            out[f"rollout_{space}_t{nt}"] = rollout_mse(model, scm, horizon=conseq["horizon"],
                                                        n_targets=nt, space=sp)
    for nt in (1, 2):
        episodes = make_consequence_episodes(scm, n_episodes=args.episodes, rng_seed=12345,
                                             n_targets=nt, **conseq)
        for space in ("native", "do_wrapper"):
            sp = "decoded" if space == "native" else space
            out[f"consequence_{space}_t{nt}"] = run_consequence_task(model, episodes, space=sp,
                                                                     **conseq)

    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k.startswith(("mse", "consequence"))}, indent=1))


if __name__ == "__main__":
    main()
