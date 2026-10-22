"""Aggregate results_v2/ (and the original results/) into the numbers used in the revised paper.

Reports mean +- sample standard deviation (ddof=1) over seeds.

    python scripts/aggregate_revision.py > tables/revision_summary.json
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": len(a), "values": [round(float(v), 4) for v in a]}


def collect_v2(config):
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "results_v2", config, "*", "eval_v2.json"))):
        d = json.load(open(path))
        out.setdefault(d["variant"], []).append(d)
    summary = {}
    for variant, runs in out.items():
        s = {}
        for section in runs[0]:
            if not isinstance(runs[0][section], dict):
                continue
            keys = [k for k, v in runs[0][section].items() if isinstance(v, (int, float))]
            s[section] = {k: ms([r[section].get(k) for r in runs if section in r]) for k in keys}
        summary[variant] = s
    return summary


def collect_v1(config):
    """Original (first-version) runs shipped in results/."""
    out = {}
    for path in glob.glob(os.path.join(ROOT, "results", f"synthetic_{config}_*_seed*", "results.json")):
        d = json.load(open(path))
        out.setdefault(d["model"], []).append(d)
    return {m: {k: ms([r.get(k) for r in runs]) for k in ("intervention_mse", "counterfactual_mse", "shd")}
            for m, runs in out.items()}


def main():
    res = {"v2": {c: collect_v2(c) for c in ("er_k10", "er_k20", "chain", "fork", "collider")},
           "v1": {c: collect_v1(c) for c in ("er_k10", "er_k20", "chain", "fork", "collider")}}
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
