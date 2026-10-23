"""Experiments for the revised paper (v2).

Trains every model variant on ER K=10 and evaluates it on
  * one-step intervention / counterfactual MSE (src.eval),
  * the direct-effect agent task in three settings, and
  * the multi-step consequence agent task,
all through the same planner. One process per (variant, seed); results go to
results_v2/<variant>_seed<seed>/eval_v2.json.

    python scripts/run_revision.py --variant aclscm_simplified --seed 0
"""
import argparse
import json
import os
import sys
from copy import deepcopy

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_tasks import (make_consequence_episodes, oracle_predictor,  # noqa: E402
                             run_consequence_task, run_direct_task)
from src.models import ACLSCM, get_model  # noqa: E402
from src.scm import SyntheticSCM  # noqa: E402
from src.train import load_config, train_model  # noqa: E402

SIMPLIFIED = {"model.acyclicity": "none", "model.init_logit": 2.0, "model.use_hinge": False,
              "model.exact_abduction": True, "training.lambda_1": 0.01}

VARIANTS = {
    # name: (model, overrides)
    "aclscm_original": ("aclscm", {}),
    "aclscm_original_no_do": ("aclscm", {"model.use_do_operator": False}),
    "aclscm_simplified": ("aclscm", SIMPLIFIED),
    "aclscm_simplified_no_do": ("aclscm", {**SIMPLIFIED, "model.use_do_operator": False}),
    "aclscm_simplified_no_cf": ("aclscm", {**SIMPLIFIED, "training.beta_4": 0.0}),
    "aclscm_simplified_st": ("aclscm", {**SIMPLIFIED, "model.hard_gumbel": True}),
    "aclscm_simplified_st_no_do": ("aclscm", {**SIMPLIFIED, "model.hard_gumbel": True,
                                              "model.use_do_operator": False}),
    "aclscm_simplified_st_no_cf": ("aclscm", {**SIMPLIFIED, "model.hard_gumbel": True,
                                              "training.beta_4": 0.0}),
    "aclscm_simplified_st_dense": ("aclscm", {**SIMPLIFIED, "model.hard_gumbel": True,
                                              "training.lambda_1": 0.0}),
    "aclscm_simplified_st_dense_no_cf": ("aclscm", {**SIMPLIFIED, "model.hard_gumbel": True,
                                                    "training.lambda_1": 0.0, "training.beta_4": 0.0}),
    "aclscm_simplified_st_true_graph": ("aclscm", {**SIMPLIFIED, "model.hard_gumbel": True,
                                                   "model.fixed_adjacency": "true"}),
    "vanilla_vae": ("vanilla_vae", {}),
    "dreamer_style": ("dreamer_style", {}),
    "small_transformer": ("small_transformer", {}),
}

CONSEQ = dict(horizon=3, safety_node=9, goal_node=8, tau_s=0.4, tau_g=0.3)


def build_model(model_name, K, model_cfg):
    if model_name == "aclscm":
        return ACLSCM(K=K, obs_dim=K, use_do_operator=model_cfg.get("use_do_operator", True),
                      acyclicity=model_cfg.get("acyclicity", "notears"),
                      init_logit=model_cfg.get("init_logit", 0.0),
                      exact_abduction=model_cfg.get("exact_abduction", False),
                       hard_gumbel=model_cfg.get("hard_gumbel", False),
                       fixed_adjacency=model_cfg.get("fixed_adjacency"))
    return get_model(model_name, K=K, obs_dim=K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/synthetic_er_k10.yaml")
    ap.add_argument("--results-dir", default="results_v2")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--mse-only", action="store_true",
                    help="skip the agent tasks (they are defined for ER K=10 only)")
    args = ap.parse_args()
    torch.set_num_threads(1)

    model_name, overrides = VARIANTS[args.variant]
    cfg = deepcopy(load_config(args.config))
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    scm_cfg = cfg["scm"]
    K = scm_cfg["K"]
    scm = SyntheticSCM(scm_cfg["graph_type"], K, scm_cfg.get("seed", 0), scm_cfg.get("edge_prob", 0.3))
    model_cfg = {}
    for k, v in overrides.items():
        if v == "true":
            v = scm.true_adjacency.tolist()
        section, key = k.split(".", 1)
        if section == "model":
            model_cfg[key] = v
        else:
            cfg[section][key] = v

    metrics = train_model(cfg, model_name, args.seed, args.results_dir,
                          ablation_name=args.variant, model_cfg_overrides=model_cfg)
    run_dir = os.path.join(args.results_dir, f"ablation_{args.variant}_seed{args.seed}")

    model = build_model(model_name, K, model_cfg)
    ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out = {"variant": args.variant, "seed": args.seed,
           "mse": {k: metrics.get(k) for k in ("intervention_mse", "counterfactual_mse", "shd",
                                                "dag_constraint", "epochs_run")}}
    if isinstance(model, ACLSCM):
        A = model.get_adjacency(hard=True)
        out["mse"]["n_learned_edges"] = int(A.sum().item())
    if args.mse_only:
        with open(os.path.join(run_dir, "eval_v2.json"), "w") as f:
            json.dump(out, f, indent=2)
        return

    spaces = ["decoded"] if model_name == "small_transformer" else ["latent", "decoded"]
    for space in spaces:
        out[f"direct_original_{space}"] = run_direct_task(model, scm, K, space=space, rng_seed=args.seed)
    out["direct_in_range_decoded"] = run_direct_task(model, scm, K, tau_g=0.5, tau_s=1.0,
                                                     space="decoded", rng_seed=args.seed)

    # The same consequence episodes for every model and seed.
    episodes = make_consequence_episodes(scm, n_episodes=args.episodes, rng_seed=12345, legacy_rng=True, **{
        k: CONSEQ[k] for k in ("horizon", "safety_node", "goal_node", "tau_s", "tau_g")})
    out["consequence_decoded"] = run_consequence_task(model, episodes, space="decoded", **CONSEQ)
    if args.seed == 0 and args.variant == "aclscm_simplified":
        out["consequence_oracle"] = run_consequence_task(
            None, episodes, predictor=oracle_predictor(scm, CONSEQ["horizon"]), **CONSEQ)

    with open(os.path.join(run_dir, "eval_v2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
