"""Training loop with checkpointing, early stopping, and CLI."""
import argparse
import json
import logging
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.eval import evaluate_model
from src.losses import composite_loss
from src.models import ACLSCM, get_model
from src.planning import run_agent_episodes
from src.scm import SyntheticSCM
from src.utils import get_device, load_json, save_json, set_seed, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply flat override dict with dot-notation keys, e.g. 'model.use_do_operator': False."""
    cfg = deepcopy(cfg)
    for k, v in overrides.items():
        parts = k.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = v
    return cfg


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def make_tensors(data: dict, device=None):
    keys = ["x_t", "a_t", "x_tp1", "eps_t", "x_tp1_cf", "a_t_cf", "z_tp1_cf"]
    tensors = [torch.tensor(data[k], dtype=torch.float32) for k in keys]
    return tensors


def make_loader(data: dict, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = make_tensors(data)
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, cfg, model_cfg, epoch: int = 0) -> dict:
    model.train()
    K = model.K
    beta_1 = cfg["training"].get("beta_1", 1.0)
    beta_2 = cfg["training"].get("beta_2", 1.0)
    beta_3 = cfg["training"].get("beta_3", 0.5)
    beta_4 = cfg["training"].get("beta_4", 1.0)
    acyclicity = model_cfg.get("acyclicity", "notears")

    # DAG curriculum: ramp lambda2 from 0 → full over dag_curriculum_epochs
    curriculum_epochs = cfg["training"].get("dag_curriculum_epochs", 0)
    dag_lambda2_factor = min(1.0, (epoch + 1) / curriculum_epochs) if curriculum_epochs > 0 else 1.0

    totals = {"total": 0, "recon": 0, "transition": 0, "dag": 0, "contrastive": 0}
    n_batches = 0

    for batch in loader:
        x_t, a_t, x_tp1, eps_t, x_tp1_cf, a_t_cf, z_tp1_cf_gt = [b.to(device) for b in batch]

        optimizer.zero_grad()
        out = model(x_t, a_t)

        # Posterior for KL term (VAE models)
        mu_q, logvar_q, mu_p = None, None, None
        if hasattr(model, "encode") and not isinstance(model, type(None)):
            try:
                mu_q, logvar_q = model.encode(x_tp1)
                mu_p = out["z_tp1_pred"]
            except Exception:
                pass

        # Counterfactual latent predictions (ACLSCM only)
        # Use ground-truth z_tp1_cf from SCM as supervision target (not encoded x_tp1_cf)
        z_tp1_cf_pred, z_tp1_cf_true, z_tp1_factual = None, None, None
        if isinstance(model, ACLSCM) and beta_4 > 0.0:
            _, z_cf_pred = model.counterfactual(x_t, a_t, x_tp1, a_t_cf)
            z_tp1_cf_pred = z_cf_pred
            z_tp1_cf_true = z_tp1_cf_gt  # ground-truth latent, no encoder noise
            z_tp1_factual = out["z_tp1_pred"]

        losses = composite_loss(
            x_tp1=x_tp1,
            x_tp1_recon=out["x_tp1_recon"],
            mu_q=mu_q,
            logvar_q=logvar_q,
            mu_p=mu_p,
            A=out.get("adjacency"),
            K=K,
            z_tp1_cf_pred=z_tp1_cf_pred,
            z_tp1_cf_true=z_tp1_cf_true,
            z_tp1_factual=z_tp1_factual,
            beta_1=beta_1,
            beta_2=beta_2,
            beta_3=beta_3 * dag_lambda2_factor,
            beta_4=beta_4,
            acyclicity=acyclicity,
        )

        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model, loader, device, cfg, model_cfg) -> dict:
    model.eval()
    K = model.K
    beta_1 = cfg["training"].get("beta_1", 1.0)
    beta_2 = cfg["training"].get("beta_2", 1.0)
    beta_3 = cfg["training"].get("beta_3", 0.5)
    beta_4 = cfg["training"].get("beta_4", 1.0)
    acyclicity = model_cfg.get("acyclicity", "notears")

    totals = {"total": 0, "recon": 0, "transition": 0, "dag": 0, "contrastive": 0}
    n_batches = 0

    for batch in loader:
        x_t, a_t, x_tp1, eps_t, x_tp1_cf, a_t_cf, z_tp1_cf_gt = [b.to(device) for b in batch]

        out = model(x_t, a_t)

        mu_q, logvar_q, mu_p = None, None, None
        if hasattr(model, "encode"):
            try:
                mu_q, logvar_q = model.encode(x_tp1)
                mu_p = out["z_tp1_pred"]
            except Exception:
                pass

        z_tp1_cf_pred, z_tp1_cf_true, z_tp1_factual = None, None, None
        if isinstance(model, ACLSCM) and beta_4 > 0.0:
            _, z_cf_pred = model.counterfactual(x_t, a_t, x_tp1, a_t_cf)
            z_tp1_cf_pred = z_cf_pred
            z_tp1_cf_true = z_tp1_cf_gt  # ground-truth latent
            z_tp1_factual = out["z_tp1_pred"]

        losses = composite_loss(
            x_tp1=x_tp1,
            x_tp1_recon=out["x_tp1_recon"],
            mu_q=mu_q,
            logvar_q=logvar_q,
            mu_p=mu_p,
            A=out.get("adjacency"),
            K=K,
            z_tp1_cf_pred=z_tp1_cf_pred,
            z_tp1_cf_true=z_tp1_cf_true,
            z_tp1_factual=z_tp1_factual,
            beta_1=beta_1,
            beta_2=beta_2,
            beta_3=beta_3,
            beta_4=beta_4,
            acyclicity=acyclicity,
        )

        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"], ckpt.get("best_val_metric", float("inf"))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(
    cfg: dict,
    model_name: str,
    seed: int,
    results_dir: str,
    resume: bool = False,
    smoke_test: bool = False,
    ablation_name: str = None,
    model_cfg_overrides: dict = None,
):
    config_name = cfg["name"]
    run_name = f"{config_name}_{model_name}_seed{seed}"
    if ablation_name:
        run_name = f"ablation_{ablation_name}_seed{seed}"

    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    final_json = os.path.join(run_dir, "results.json")
    if not smoke_test and os.path.exists(final_json):
        logger.info(f"Skipping {run_name} — results.json already exists.")
        return load_json(final_json)

    log_path = os.path.join(run_dir, "train.log")
    setup_logging(log_path)
    set_seed(seed)
    device = get_device()
    logger.info(f"Starting {run_name} on {device}")

    # Build SCM and data
    scm_cfg = cfg["scm"]
    scm = SyntheticSCM(
        graph_type=scm_cfg["graph_type"],
        K=scm_cfg["K"],
        seed=scm_cfg.get("seed", 0),
        edge_prob=scm_cfg.get("edge_prob", 0.3),
    )
    K = scm_cfg["K"]
    obs_dim = K

    train_data = scm.sample(scm_cfg.get("n_train", 2000), rng_seed=seed * 10)
    test_data = scm.sample(scm_cfg.get("n_test", 500), rng_seed=seed * 10 + 999)
    test_data["true_adjacency"] = scm.true_adjacency

    train_loader = make_loader(train_data, cfg["training"]["batch_size"], shuffle=True)
    # Use a subset of train as val (last 20%)
    n_val = int(len(train_data["x_t"]) * 0.2)
    val_data = {k: v[-n_val:] for k, v in train_data.items() if isinstance(v, np.ndarray)}
    val_loader = make_loader(val_data, cfg["training"]["batch_size"], shuffle=False)

    # Build model
    model_cfg = {}
    if model_cfg_overrides:
        model_cfg.update(model_cfg_overrides)

    if model_name == "aclscm":
        use_do = model_cfg.get("use_do_operator", True)
        acyclicity = model_cfg.get("acyclicity", "notears")
        model = ACLSCM(K=K, obs_dim=obs_dim, use_do_operator=use_do, acyclicity=acyclicity)
    else:
        model = get_model(model_name, K=K, obs_dim=obs_dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    epochs = 1 if smoke_test else cfg["training"]["epochs"]
    checkpoint_every = cfg["training"].get("checkpoint_every", 10)
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")

    start_epoch = 0
    best_val_metric = float("inf")
    patience = 20
    patience_counter = 0

    if resume and os.path.exists(ckpt_path):
        start_epoch, best_val_metric = load_checkpoint(ckpt_path, model, optimizer, device)
        logger.info(f"Resumed from epoch {start_epoch}, best_val={best_val_metric:.4f}")

    best_model_state = deepcopy(model.state_dict())
    epoch_log = []

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_losses = train_epoch(model, train_loader, optimizer, device, cfg, model_cfg, epoch=epoch)
        val_losses = val_epoch(model, val_loader, device, cfg, model_cfg)

        val_metric = val_losses["total"]
        improved = val_metric < best_val_metric
        if improved:
            best_val_metric = val_metric
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        dag_val = 0.0
        if isinstance(model, ACLSCM):
            A = torch.sigmoid(model.A_logits).float()
            diag = 1.0 - torch.eye(K, device=device)
            A = A * diag
            dag_val = (torch.trace(torch.matrix_exp(A * A)) - K).item()

        row = {
            "epoch": epoch + 1,
            "train_total": train_losses["total"],
            "train_recon": train_losses["recon"],
            "train_transition": train_losses["transition"],
            "train_dag": train_losses["dag"],
            "train_contrastive": train_losses["contrastive"],
            "val_total": val_losses["total"],
            "val_recon": val_losses["recon"],
            "dag_constraint": dag_val,
            "time_s": time.time() - t0,
        }
        epoch_log.append(row)
        logger.info(
            f"Epoch {epoch+1}/{epochs}  "
            f"train={train_losses['total']:.4f}  val={val_losses['total']:.4f}  "
            f"dag={dag_val:.4f}  {'*' if improved else ''}"
        )

        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_metric": best_val_metric,
                },
                ckpt_path,
            )

        if patience_counter >= patience and not smoke_test:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model and evaluate
    model.load_state_dict(best_model_state)
    model.eval()

    metrics = evaluate_model(model, test_data, device)
    metrics["run_name"] = run_name
    metrics["model"] = model_name
    metrics["config"] = config_name
    metrics["seed"] = seed
    metrics["epochs_run"] = len(epoch_log)

    # Agent-task evaluation (all models, but meaningful for ACLSCM + SmallTransformer)
    if not smoke_test and model_name in ("aclscm", "small_transformer"):
        try:
            agent_metrics = run_agent_episodes(model, scm, K, n_episodes=100,
                                               device=device, rng_seed=seed)
            metrics.update({f"agent_{k}": v for k, v in agent_metrics.items()})
        except Exception as e:
            logger.warning(f"Agent task failed: {e}")

    if not smoke_test:
        save_json(metrics, final_json)
        save_json(epoch_log, os.path.join(run_dir, "epoch_log.json"))
        # Save final checkpoint
        save_checkpoint(
            {
                "epoch": len(epoch_log),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_metric": best_val_metric,
            },
            ckpt_path,
        )

    logger.info(f"Done {run_name}: int_mse={metrics.get('intervention_mse', 'N/A'):.4f}  "
                f"cf_mse={metrics.get('counterfactual_mse', 'N/A'):.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablations(ablation_cfg: dict, base_configs: dict, seeds: list,
                  results_dir: str, resume: bool = False):
    base_cfg_name = ablation_cfg["base"]
    base_cfg = base_configs[base_cfg_name]

    for abl in ablation_cfg["ablations"]:
        abl_name = abl["name"]
        overrides = abl.get("overrides", {})

        cfg = apply_overrides(base_cfg, {
            k: v for k, v in overrides.items() if not k.startswith("model.")
        })
        model_cfg_overrides = {
            k[len("model."):]: v for k, v in overrides.items() if k.startswith("model.")
        }

        for seed in seeds:
            train_model(
                cfg=cfg,
                model_name="aclscm",
                seed=seed,
                results_dir=results_dir,
                resume=resume,
                ablation_name=abl_name,
                model_cfg_overrides=model_cfg_overrides,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train AC-LSCM experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None, help="Override model (for single-model runs)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (smoke test: 1)")
    parser.add_argument("--ablations-base-config", default=None,
                        help="Path to base config YAML for ablations")
    args = parser.parse_args()

    cfg = load_config(args.config)
    smoke_test = (args.epochs == 1)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    # Handle ablations config
    if "ablations" in cfg:
        if args.ablations_base_config is None:
            raise ValueError("--ablations-base-config required for ablations config")
        base_cfg = load_config(args.ablations_base_config)
        base_configs = {base_cfg["name"]: base_cfg}
        run_ablations(cfg, base_configs, seeds=cfg.get("seeds", [args.seed]),
                      results_dir=args.results_dir, resume=args.resume)
        return

    seeds = cfg.get("seeds", [args.seed])
    models = cfg.get("models", ["aclscm"])
    if args.model:
        models = [args.model]

    for model_name in models:
        for seed in seeds:
            train_model(
                cfg=cfg,
                model_name=model_name,
                seed=seed,
                results_dir=args.results_dir,
                resume=args.resume,
                smoke_test=smoke_test,
            )


if __name__ == "__main__":
    main()
