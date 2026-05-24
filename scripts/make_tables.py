"""Generate LaTeX tables from results JSONs."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load_results(results_dir: str) -> list[dict]:
    results = []
    for path in glob.glob(os.path.join(results_dir, "**", "results.json"), recursive=True):
        with open(path) as f:
            r = json.load(f)
        results.append(r)
    return results


def fmt(values: list[float], decimals: int = 4) -> str:
    if not values:
        return "---"
    m = np.mean(values)
    s = np.std(values)
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


MODEL_DISPLAY = {
    "small_transformer": "Small Transformer",
    "vanilla_vae": "Vanilla VAE",
    "dreamer_style": "Dreamer-style",
    "aclscm": "AC-LSCM (ours)",
}

CONFIG_DISPLAY = {
    "synthetic_chain": "Chain",
    "synthetic_fork": "Fork",
    "synthetic_collider": "Collider",
    "synthetic_er_k5": "ER (K=5)",
    "synthetic_er_k10": "ER (K=10)",
    "synthetic_er_k20": "ER (K=20)",
}

ABLATION_DISPLAY = {
    "no_causal_loss": r"-- no $\mathcal{L}_{\text{Causal}}$ ($\beta_3=0$)",
    "no_contrastive_loss": r"-- no $\mathcal{L}_{\text{Contrastive}}$ ($\beta_4=0$)",
    "no_do_operator": "-- no do-operator",
    "dagma_instead_of_notears": "-- DAGMA instead of NOTEARS",
}


def build_table_a(results: list[dict]) -> str:
    """Table A: rows=models, columns=chain/fork/collider/er_k10, cells=Int/CF MSE."""
    configs = ["synthetic_chain", "synthetic_fork", "synthetic_collider", "synthetic_er_k10"]
    models = ["small_transformer", "vanilla_vae", "dreamer_style", "aclscm"]

    # Index: config → model → [int_mse, cf_mse] lists
    data = defaultdict(lambda: defaultdict(lambda: {"int": [], "cf": []}))
    for r in results:
        cfg = r.get("config", "")
        mdl = r.get("model", "")
        if cfg in configs and mdl in models and "ablation" not in r.get("run_name", ""):
            data[cfg][mdl]["int"].append(r.get("intervention_mse", np.nan))
            data[cfg][mdl]["cf"].append(r.get("counterfactual_mse", np.nan))

    col_headers = " & ".join(CONFIG_DISPLAY[c] for c in configs)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Main results: Intervention MSE / Counterfactual MSE (mean $\pm$ std over 3 seeds). Lower is better.}",
        r"\label{tab:main}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(configs) + "}",
        r"\toprule",
        r"Model & " + col_headers + r" \\",
        r"& " + " & ".join([r"\small{Int / CF}"] * len(configs)) + r" \\",
        r"\midrule",
    ]
    for mdl in models:
        cells = []
        for cfg in configs:
            ints = [v for v in data[cfg][mdl]["int"] if not np.isnan(v)]
            cfs = [v for v in data[cfg][mdl]["cf"] if not np.isnan(v)]
            cells.append(f"{fmt(ints, 3)} / {fmt(cfs, 3)}")
        lines.append(MODEL_DISPLAY[mdl] + " & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build_table_b(results: list[dict]) -> str:
    """Table B: Ablations on ER K=10."""
    ablations = ["no_causal_loss", "no_contrastive_loss", "no_do_operator", "dagma_instead_of_notears"]

    # Full AC-LSCM on er_k10
    full_data = {"int": [], "cf": [], "shd": []}
    for r in results:
        if r.get("config") == "synthetic_er_k10" and r.get("model") == "aclscm" \
                and "ablation" not in r.get("run_name", ""):
            full_data["int"].append(r.get("intervention_mse", np.nan))
            full_data["cf"].append(r.get("counterfactual_mse", np.nan))
            full_data["shd"].append(r.get("shd", np.nan))

    # Ablation data
    abl_data = defaultdict(lambda: {"int": [], "cf": [], "shd": []})
    for r in results:
        run_name = r.get("run_name", "")
        for abl in ablations:
            if f"ablation_{abl}" in run_name:
                abl_data[abl]["int"].append(r.get("intervention_mse", np.nan))
                abl_data[abl]["cf"].append(r.get("counterfactual_mse", np.nan))
                abl_data[abl]["shd"].append(r.get("shd", np.nan))

    def row(label, d):
        ints = [v for v in d["int"] if not np.isnan(v)]
        cfs = [v for v in d["cf"] if not np.isnan(v)]
        shds = [v for v in d["shd"] if not np.isnan(v)]
        return f"{label} & {fmt(ints, 3)} & {fmt(cfs, 3)} & {fmt(shds, 1)} \\\\"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation study on ER (K=10). Lower is better.}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Variant & Int MSE & CF MSE & SHD \\",
        r"\midrule",
        row("AC-LSCM (full)", full_data),
    ]
    for abl in ablations:
        lines.append(row(ABLATION_DISPLAY[abl], abl_data[abl]))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build_table_c(results: list[dict]) -> str:
    """Table C: Agent task on ER K=10."""
    models = ["small_transformer", "aclscm"]
    data = defaultdict(lambda: {"goal": [], "safety": [], "deferral": []})
    for r in results:
        if r.get("config") == "synthetic_er_k10" and "ablation" not in r.get("run_name", ""):
            mdl = r.get("model", "")
            if mdl in models:
                if "agent_goal_rate" in r:
                    data[mdl]["goal"].append(r["agent_goal_rate"])
                if "agent_safety_violation_rate" in r:
                    data[mdl]["safety"].append(r["agent_safety_violation_rate"])
                if "agent_appropriate_deferral_rate" in r:
                    data[mdl]["deferral"].append(r["agent_appropriate_deferral_rate"])

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Agent-task evaluation on ER (K=10). Goal/Deferral: higher is better. Safety violation: lower is better.}",
        r"\label{tab:agent}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Goal Rate & Safety Violation & Appropriate Deferral \\",
        r"\midrule",
    ]
    for mdl in models:
        d = data[mdl]
        lines.append(
            f"{MODEL_DISPLAY[mdl]} & {fmt(d['goal'], 3)} & {fmt(d['safety'], 3)} & {fmt(d['deferral'], 3)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build_summary(results: list[dict]) -> dict:
    """Build summary.json with all metrics."""
    summary = {}
    for r in results:
        key = r.get("run_name", "unknown")
        summary[key] = {k: v for k, v in r.items()}
    return summary


def interpret_findings(results: list[dict]) -> str:
    """One paragraph per table interpreting numbers."""
    lines = ["# Findings\n"]

    # Table A analysis
    models = ["small_transformer", "vanilla_vae", "dreamer_style", "aclscm"]
    configs = ["synthetic_chain", "synthetic_fork", "synthetic_collider", "synthetic_er_k10"]
    cf_by_model = defaultdict(list)
    int_by_model = defaultdict(list)
    for r in results:
        if r.get("config") in configs and r.get("model") in models \
                and "ablation" not in r.get("run_name", ""):
            cf_by_model[r["model"]].append(r.get("counterfactual_mse", np.nan))
            int_by_model[r["model"]].append(r.get("intervention_mse", np.nan))

    ac_cf = np.nanmean(cf_by_model["aclscm"])
    best_baseline_cf = min(np.nanmean(cf_by_model[m]) for m in ["small_transformer", "vanilla_vae", "dreamer_style"])
    ac_int = np.nanmean(int_by_model["aclscm"])
    best_baseline_int = min(np.nanmean(int_by_model[m]) for m in ["small_transformer", "vanilla_vae", "dreamer_style"])

    hyp1 = "CONFIRMED" if ac_cf < best_baseline_cf else "NOT CONFIRMED"
    hyp2 = "CONFIRMED" if ac_int >= best_baseline_int * 0.95 else "NOT CONFIRMED"

    lines.append(f"## Table A — Main Results\n")
    lines.append(
        f"AC-LSCM achieves a mean counterfactual MSE of {ac_cf:.4f} vs best baseline {best_baseline_cf:.4f} "
        f"(hypothesis 1: {hyp1}). "
        f"Interventional MSE: AC-LSCM {ac_int:.4f} vs best baseline {best_baseline_int:.4f} "
        f"(hypothesis 2 — AC-LSCM competitive on simple graphs: {hyp2}).\n"
    )

    # Table B analysis
    lines.append("## Table B — Ablations\n")
    full_cf = [r.get("counterfactual_mse", np.nan) for r in results
               if r.get("config") == "synthetic_er_k10" and r.get("model") == "aclscm"
               and "ablation" not in r.get("run_name", "")]
    abl_cf = defaultdict(list)
    for r in results:
        run = r.get("run_name", "")
        for abl in ["no_causal_loss", "no_contrastive_loss", "no_do_operator", "dagma_instead_of_notears"]:
            if f"ablation_{abl}" in run:
                abl_cf[abl].append(r.get("counterfactual_mse", np.nan))

    full_mean = np.nanmean(full_cf) if full_cf else np.nan
    lines.append(f"Full AC-LSCM CF MSE: {full_mean:.4f}.")
    for abl, vals in abl_cf.items():
        m = np.nanmean(vals)
        delta = m - full_mean
        direction = "worse" if delta > 0 else "better"
        lines.append(f"  - {abl}: {m:.4f} ({delta:+.4f} {direction} than full model).")
    lines.append(
        "\nHypothesis 5 (contrastive loss matters most for CF accuracy): "
        + ("CONFIRMED" if np.nanmean(abl_cf.get("no_contrastive_loss", [np.nan])) >
           np.nanmean(abl_cf.get("no_causal_loss", [np.nan])) else "NOT CONFIRMED")
        + ".\n"
    )

    # Table C analysis
    lines.append("## Table C — Agent Task\n")
    agent_aclscm = [r for r in results if r.get("model") == "aclscm"
                    and r.get("config") == "synthetic_er_k10"
                    and "agent_goal_rate" in r and "ablation" not in r.get("run_name", "")]
    agent_tf = [r for r in results if r.get("model") == "small_transformer"
                and r.get("config") == "synthetic_er_k10"
                and "agent_goal_rate" in r]
    if agent_aclscm:
        goal_ac = np.mean([r["agent_goal_rate"] for r in agent_aclscm])
        safety_ac = np.mean([r["agent_safety_violation_rate"] for r in agent_aclscm])
        hyp4 = "CONFIRMED" if (agent_tf and np.mean([r["agent_safety_violation_rate"] for r in agent_tf]) > safety_ac) else "NOT CONFIRMED"
        lines.append(
            f"AC-LSCM goal rate: {goal_ac:.3f}, safety violations: {safety_ac:.3f}. "
            f"Hypothesis 4 (fewer safety violations for AC-LSCM): {hyp4}.\n"
        )
    else:
        lines.append("Agent task results not available.\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="tables")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} result files from {args.results_dir}")

    if not results:
        print("No results found. Run experiments first.")
        return

    table_a = build_table_a(results)
    table_b = build_table_b(results)
    table_c = build_table_c(results)
    summary = build_summary(results)
    findings = interpret_findings(results)

    with open(os.path.join(args.output, "table_a.tex"), "w") as f:
        f.write(table_a)
    with open(os.path.join(args.output, "table_b.tex"), "w") as f:
        f.write(table_b)
    with open(os.path.join(args.output, "table_c.tex"), "w") as f:
        f.write(table_c)
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output, "findings.md"), "w") as f:
        f.write(findings)

    print(f"Tables written to {args.output}/")
    print(f"  table_a.tex  — main results")
    print(f"  table_b.tex  — ablations")
    print(f"  table_c.tex  — agent task")
    print(f"  findings.md  — interpretation")
    print(f"Summary written to {args.results_dir}/summary.json")


if __name__ == "__main__":
    main()
