# Findings

## Table A — Main Results

AC-LSCM achieves a mean counterfactual MSE of 0.1179 vs best baseline 0.0332 (hypothesis 1: NOT CONFIRMED). Interventional MSE: AC-LSCM 0.0956 vs best baseline 0.0326 (hypothesis 2 — AC-LSCM competitive on simple graphs: CONFIRMED).

## Table B — Ablations

Full AC-LSCM CF MSE: 0.1140.
  - no_causal_loss: 0.1281 (+0.0141 worse than full model).
  - dagma_instead_of_notears: 0.1324 (+0.0184 worse than full model).
  - no_do_operator: 0.1543 (+0.0403 worse than full model).
  - no_contrastive_loss: 0.0570 (-0.0569 better than full model).

Hypothesis 5 (contrastive loss matters most for CF accuracy): NOT CONFIRMED.

## Table C — Agent Task

AC-LSCM goal rate: 0.510, safety violations: 0.267. Hypothesis 4 (fewer safety violations for AC-LSCM): NOT CONFIRMED.
