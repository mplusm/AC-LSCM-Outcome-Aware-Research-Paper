"""Aggregate results_v3 into tables/v3_summary.json.

    python scripts/aggregate_v3.py [results_v3] > tables/v3_summary.json

Per (env, variant, n_train): mean / sample std / n over seeds of every scalar
metric. Per (variant, n_train) across environments: the mean over
environments of the per-environment means, and the normalised excess
violation rate

    (violations - oracle) / (no-effect reference - oracle),

which is 0 for the oracle and 1 for a predictor that ignores the action.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
            out[key] = float(v)
    return out


def stat(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return None
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
            "n": len(xs)}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results_v3"
    runs = defaultdict(list)
    for p in glob.glob(os.path.join(root, "*", "*", "eval.json")):
        r = json.load(open(p))
        runs[(r["env"], r["variant"], r["n_train"])].append(flatten(r))
    refs = {}
    for p in glob.glob(os.path.join(root, "*", "references.json")):
        r = json.load(open(p))
        refs[r["env"]] = flatten(r)

    per_env = {}
    for (env, var, n), rs in runs.items():
        keys = sorted({k for r in rs for k in r})
        per_env.setdefault(env, {}).setdefault(str(n), {})[var] = {
            k: stat([r.get(k) for r in rs]) for k in keys}

    # Normalised excess violations per run, then the same aggregation.
    for (env, var, n), rs in runs.items():
        if env not in refs:
            continue
        for t in (1, 2):
            o = refs[env][f"consequence_oracle_t{t}.safety_violation_rate"]
            z = refs[env][f"consequence_no_effect_t{t}.safety_violation_rate"]
            for space in ("native", "do_wrapper"):
                vals = [(r[f"consequence_{space}_t{t}.safety_violation_rate"] - o) / (z - o) for r in rs]
                per_env[env][str(n)][var][f"excess_{space}_t{t}"] = stat(vals)

    across = {}
    for env, by_n in per_env.items():
        for n, by_var in by_n.items():
            for var, metrics in by_var.items():
                for k, s in metrics.items():
                    if s:
                        across.setdefault(n, {}).setdefault(var, {}).setdefault(k, {})[env] = s["mean"]
    summary = {}
    for n, by_var in across.items():
        for var, metrics in by_var.items():
            for k, by_env in metrics.items():
                for group, envs in (("er10", [e for e in by_env if e.startswith("er10")]),
                                    ("er20", [e for e in by_env if e.startswith("er20")])):
                    if envs:
                        s = stat([by_env[e] for e in envs])
                        summary.setdefault(group, {}).setdefault(n, {}).setdefault(var, {})[k] = s

    json.dump({"per_env": per_env, "across_envs": summary, "references": refs}, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
