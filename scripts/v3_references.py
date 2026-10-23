"""Model-free references for every v3 environment: the oracle (noise-free
ground-truth SCM from the observed state) and the no-effect predictor (nothing
changes after the action). Writes results_v3/<env>/references.json.

    python scripts/v3_references.py er10_s0 er10_s1 ...
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_v3 import parse_env, task_config  # noqa: E402
from src.agent_tasks import (make_consequence_episodes, oracle_predictor,  # noqa: E402
                             rollout_mse, run_consequence_task)
from src.scm import SyntheticSCM  # noqa: E402


def no_effect_predictor(horizon):
    def predict(e):
        x = torch.tensor(e["x_t"]).expand(len(e["actions"]), -1)
        return x.unsqueeze(0).repeat(horizon, 1, 1)
    return predict


def main():
    for env in sys.argv[1:]:
        K, graph_seed = parse_env(env)
        scm = SyntheticSCM("er_random", K, graph_seed, 0.3)
        task, conseq = task_config(scm)
        H = conseq["horizon"]
        out = {"env": env, "task": task, "n_true_edges": int(scm.true_adjacency.sum())}
        for nt in (1, 2):
            out[f"rollout_no_effect_t{nt}"] = rollout_mse(None, scm, horizon=H, n_targets=nt)
            episodes = make_consequence_episodes(scm, n_episodes=300, rng_seed=12345,
                                                 n_targets=nt, **conseq)
            out[f"consequence_oracle_t{nt}"] = run_consequence_task(
                None, episodes, predictor=oracle_predictor(scm, H), **conseq)
            out[f"consequence_no_effect_t{nt}"] = run_consequence_task(
                None, episodes, predictor=no_effect_predictor(H), **conseq)
        os.makedirs(os.path.join("results_v3", env), exist_ok=True)
        with open(os.path.join("results_v3", env, "references.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(env, {k: round(v["safety_violation_rate"], 3) for k, v in out.items()
                    if k.startswith("consequence")})


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
