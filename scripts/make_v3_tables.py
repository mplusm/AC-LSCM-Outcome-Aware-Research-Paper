"""Write the LaTeX tables of the third version from results_v3.

    python scripts/make_v3_tables.py results_v3 <out_dir>

Writes v3_main.tex, v3_paired_a.tex, v3_paired_b.tex, v3_data.tex, v3_ablation.tex, v3_envs.tex
(row files for tables whose headers live in the paper source) and
v3_numbers.json (every number quoted in the text).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ER10 = [f"er10_s{i}" for i in range(5)]
ER20 = ["er20_s0", "er20_s1"]
BASE = [("vae", "Vanilla VAE"), ("dreamer", "Dreamer-style"), ("transformer", "Small Transformer")]


def load(root):
    refs = {}
    for p in glob.glob(os.path.join(root, "*", "references.json")):
        r = json.load(open(p))
        refs[r["env"]] = r
    runs = defaultdict(dict)
    for p in glob.glob(os.path.join(root, "*", "*", "eval.json")):
        r = json.load(open(p))
        runs[(r["env"], r["n_train"], r["variant"])][r["seed"]] = r
    return refs, runs


class Metrics:
    def __init__(self, refs, runs):
        self.refs, self.runs = refs, runs

    def excess(self, r, space, t):
        ref = self.refs[r["env"]]
        o = ref[f"consequence_oracle_t{t}"]["safety_violation_rate"]
        z = ref[f"consequence_no_effect_t{t}"]["safety_violation_rate"]
        return (r[f"consequence_{space}_t{t}"]["safety_violation_rate"] - o) / (z - o)

    def value(self, r, key):
        kind, *rest = key
        if kind == "excess":
            return self.excess(r, *rest)
        if kind == "roll":
            space, t = rest
            return r[f"rollout_{space}_t{t}"]["mse"]
        if kind == "goal":
            space, t = rest
            return r[f"consequence_{space}_t{t}"]["goal_rate"]
        if kind == "mse":
            return r["mse"].get(rest[0])
        raise KeyError(key)

    def env_mean(self, env, n, var, key):
        rs = self.runs.get((env, n, var), {})
        vals = [self.value(r, key) for r in rs.values()]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    def across(self, envs, n, var, key):
        """Mean and std over environments of the per-environment seed means."""
        vals = [self.env_mean(e, n, var, key) for e in envs]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    def paired(self, envs, n, var, space, key_f):
        """Per-(env, seed) differences baseline - factored for key_f(space)."""
        d = []
        for e in envs:
            fs, bs = self.runs.get((e, n, "fact_sparse"), {}), self.runs.get((e, n, var), {})
            for s in sorted(set(fs) & set(bs)):
                d.append(self.value(bs[s], key_f(space)) - self.value(fs[s], key_f("native")))
        return np.array(d)


def fmt(x, digits=3, std=True):
    if x is None:
        return "--"
    m, s = x
    if std:
        return f"{m:.{digits}f}\\,{{\\tiny$\\pm$\\,{s:.{digits}f}}}"
    return f"{m:.{digits}f}"


def bold_min(cells, values):
    finite = [v for v in values if v is not None]
    if not finite:
        return cells
    best = min(finite)
    return [r"\textbf{" + c + "}" if v is not None and abs(v - best) < 1e-12 else c
            for c, v in zip(cells, values)]


MAIN_ROWS = [
    ("fact_sparse", "native", "AC-LSCM, factored"),
    ("fact_dense", "native", r"AC-LSCM, factored, dense ($\lambda_1{=}0$)"),
    ("aclscm_st_nocf", "native", "AC-LSCM, latent"),
    ("aclscm_st_nocf", "do_wrapper", "AC-LSCM, latent + do-wrapper"),
    ("vae", "native", "Vanilla VAE"),
    ("vae", "do_wrapper", "Vanilla VAE + do-wrapper"),
    ("dreamer", "native", "Dreamer-style"),
    ("dreamer", "do_wrapper", "Dreamer-style + do-wrapper"),
    ("transformer", "native", "Small Transformer"),
    ("transformer", "do_wrapper", "Small Transformer + do-wrapper"),
]
REF_ROWS = [("fact_true", "native", "AC-LSCM, factored, true graph")]


def main_table(M, n=10000):
    keys = [("roll", None, 1), ("roll", None, 2), ("excess", None, 1), ("excess", None, 2),
            ("goal", None, 1)]
    lines = []
    for group, envs in (("ER-10 (5 graphs)", ER10), ("ER-20 (2 graphs)", ER20)):
        lines += [r"\midrule", rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{group}}}}} \\"]
        cols = []
        for var, space, _ in MAIN_ROWS + REF_ROWS:
            row = []
            for kind, _, t in keys:
                row.append(M.across(envs, n, var, (kind, space, t)))
            cols.append(row)
        # bold the best (lowest; highest for goal) among non-reference rows
        nmain = len(MAIN_ROWS)
        best = []
        for j, (kind, _, _) in enumerate(keys):
            vals = [cols[i][j][0] for i in range(nmain) if cols[i][j]]
            best.append(max(vals) if kind == "goal" else min(vals))
        for i, (var, space, name) in enumerate(MAIN_ROWS + REF_ROWS):
            if i == nmain:
                lines.append(r"\cmidrule(l){1-6}")
            cells = []
            for j, (kind, _, _) in enumerate(keys):
                c = cols[i][j]
                cell = fmt(c, 3)
                if i < nmain and c and abs(c[0] - best[j]) < 1e-12:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
            lines.append(name + " & " + " & ".join(cells) + r" \\")
    return lines


def paired_table(M):
    """Two panels of row files: (ER-10, ER-20) at N=10000 and ER-10 at N=2000, 500."""
    settings = [("ER-10", ER10, 10000), ("ER-20", ER20, 10000), ("ER-10, $N{=}2000$", ER10[:3], 2000),
                ("ER-10, $N{=}500$", ER10[:3], 500)]
    panels = []
    totals = {"roll": [0, 0]}
    for panel in (settings[:2], settings[2:]):
        lines, t = _paired_rows(M, panel)
        panels.append(lines)
        totals["roll"][0] += t[0]
        totals["roll"][1] += t[1]
    return panels, totals


def _paired_rows(M, settings):
    lines = []
    totals = [0, 0]
    for var, name in BASE:
        for space in ("native", "do_wrapper"):
            cells = []
            for _, envs, n in settings:
                for t in (1, 2):
                    d = M.paired(envs, n, var, space, lambda sp, t=t: ("excess", sp, t))
                    se = d.std(ddof=1) / np.sqrt(len(d))
                    cells.append(f"${d.mean():+.2f}$\\,{{\\tiny$\\pm$\\,{se:.2f}}} ({(d > 0).sum()}/{len(d)})")
                for t in (1, 2):
                    r = M.paired(envs, n, var, space, lambda sp, t=t: ("roll", sp, t))
                    totals[0] += int((r > 0).sum())
                    totals[1] += len(r)
            label = name + (" + do-wrapper" if space == "do_wrapper" else "")
            lines.append(label + " & " + " & ".join(cells) + r" \\")
    return lines, totals


def data_table(M):
    rows = [("fact_sparse", "native", "AC-LSCM, factored"), ("fact_dense", "native", "AC-LSCM, factored, dense"),
            ("aclscm_st_nocf", "native", "AC-LSCM, latent"),
            ("aclscm_dense_nocf", "native", "AC-LSCM, latent, dense"),
            ("vae", "native", "Vanilla VAE"), ("dreamer", "native", "Dreamer-style"),
            ("transformer", "native", "Small Transformer"),
            ("fact_true", "native", "AC-LSCM, factored, true graph")]
    envs = ER10[:3]
    Ns = (500, 2000, 10000)
    keys = [("roll", 1), ("excess", 1), ("excess", 2)]
    table = [[M.across(envs, n, v, (k, sp, t)) for n in Ns for k, t in keys] for v, sp, _ in rows]
    lines = []
    for i, (v, sp, name) in enumerate(rows):
        if i == len(rows) - 1:
            lines.append(r"\midrule")
        cells = []
        for j, c in enumerate(table[i]):
            cell = fmt(c, 3, std=False)
            col = [table[r][j][0] for r in range(len(rows) - 1) if table[r][j]]
            if i < len(rows) - 1 and c and abs(c[0] - min(col)) < 1e-12:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    return lines


ABL = [
    ("fact_sparse", "Factored (default)", "none"),
    ("fact_sparse_softdo", "Factored, soft do-operator", "none"),
    ("fact_dense", r"Factored, dense ($\lambda_1{=}0$)", "none"),
    ("fact_sparse_obscf", "Factored", "observed"),
    ("fact_true", "Factored, true graph", "none"),
    ("aclscm_st_nocf", "Latent", "none"),
    ("aclscm_dense_nocf", r"Latent, dense ($\lambda_1{=}0$)", "none"),
    ("aclscm_st_obscf", "Latent", "observed"),
    ("aclscm_dense_obscf", r"Latent, dense ($\lambda_1{=}0$)", "observed"),
    ("aclscm_st", "Latent", "true latents$^\\dagger$"),
    ("vae_obscf", "Vanilla VAE", "observed"),
    ("dreamer_obscf", "Dreamer-style", "observed"),
    ("transformer_obscf", "Small Transformer", "observed"),
]


def ablation_table(M):
    lines = []
    for group, envs in (("ER-10 (5 graphs)", ER10), ("ER-20 (2 graphs)", ER20)):
        lines += [r"\midrule", rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{group}}}}} \\"]
        for var, name, cf in ABL:
            if not M.runs.get((envs[0], 10000, var)):
                continue
            shd = M.across(envs, 10000, var, ("mse", "shd"))
            cells = [cf, fmt(shd, 1, std=False) if shd else "--"]
            for key in (("mse", "intervention_mse"), ("mse", "counterfactual_mse"),
                        ("roll", "native", 1), ("excess", "native", 1), ("excess", "native", 2)):
                cells.append(fmt(M.across(envs, 10000, var, key), 3, std=False))
            lines.append(name + " & " + " & ".join(cells) + r" \\")
    return lines


def env_table(M):
    lines = []
    for e in ER10 + ER20:
        ref = M.refs[e]
        t = ref["task"]
        cells = [e.replace("_s", "/").upper().replace("ER", "ER-"),
                 str(ref["n_true_edges"]), f"$z^{{{t['safety_node']}}}$ ({t['n_ancestors_safety']})",
                 f"{t['tau_s']:.2f}", f"{ref['consequence_oracle_t1']['safety_violation_rate']:.3f}",
                 f"{ref['consequence_no_effect_t1']['safety_violation_rate']:.3f}"]
        vals = []
        for var, space in (("fact_sparse", "native"), ("vae", "native"), ("vae", "do_wrapper"),
                           ("dreamer", "do_wrapper"), ("transformer", "do_wrapper"),
                           ("aclscm_st_nocf", "native")):
            vals.append(M.env_mean(e, 10000, var, ("excess", space, 1)))
        raw = []
        for var, space in (("fact_sparse", "native"), ("vae", "native"), ("vae", "do_wrapper"),
                           ("dreamer", "do_wrapper"), ("transformer", "do_wrapper"),
                           ("aclscm_st_nocf", "native")):
            rs = M.runs[(e, 10000, var)].values()
            raw.append(float(np.mean([r[f"consequence_{space}_t1"]["safety_violation_rate"] for r in rs])))
        cells += bold_min([f"{v:.3f}" for v in raw], raw)
        lines.append(" & ".join(cells) + r" \\")
    return lines


def main():
    root, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    M = Metrics(*load(root))

    def write(name, lines):
        with open(os.path.join(out_dir, name), "w") as fh:
            fh.write("% generated by scripts/make_v3_tables.py\n" + "\n".join(lines) + "\n")

    write("v3_main.tex", main_table(M))
    (panel_a, panel_b), totals = paired_table(M)
    write("v3_paired_a.tex", panel_a)
    write("v3_paired_b.tex", panel_b)
    write("v3_data.tex", data_table(M))
    write("v3_ablation.tex", ablation_table(M))
    write("v3_envs.tex", env_table(M))

    numbers = {"rollout_wins": totals["roll"],
               "n_runs": sum(len(v) for v in M.runs.values())}
    for envs, tag in ((ER10, "er10"), (ER20, "er20")):
        for var in {v for _, _, v in []} | {k[2] for k in M.runs}:
            for key in (("excess", "native", 1), ("excess", "do_wrapper", 1), ("excess", "native", 2),
                        ("excess", "do_wrapper", 2), ("roll", "native", 1), ("roll", "do_wrapper", 1),
                        ("roll", "native", 2), ("mse", "shd"), ("mse", "n_learned_edges"),
                        ("mse", "counterfactual_mse"), ("mse", "intervention_mse")):
                for n in (500, 2000, 10000):
                    a = M.across(envs if n == 10000 else ER10[:3], n, var, key)
                    if a:
                        numbers[f"{tag}/{n}/{var}/{'_'.join(map(str, key))}"] = [round(a[0], 4), round(a[1], 4)]
    with open(os.path.join(out_dir, "v3_numbers.json"), "w") as fh:
        json.dump(numbers, fh, indent=1, sort_keys=True)
    print(totals)


if __name__ == "__main__":
    main()
