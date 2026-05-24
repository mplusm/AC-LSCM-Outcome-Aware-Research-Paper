"""Evaluation metrics for AC-LSCM and baselines."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import compute_shd


@torch.no_grad()
def evaluate_model(model, data: dict, device: torch.device, batch_size: int = 256) -> dict:
    """
    Compute all evaluation metrics on a held-out data dict.
    Returns a dict with metric names → float values.
    """
    model.eval()
    K = model.K

    x_t = torch.tensor(data["x_t"], dtype=torch.float32)
    a_t = torch.tensor(data["a_t"], dtype=torch.float32)
    x_tp1 = torch.tensor(data["x_tp1"], dtype=torch.float32)
    eps_t = torch.tensor(data["eps_t"], dtype=torch.float32)
    x_tp1_cf = torch.tensor(data["x_tp1_cf"], dtype=torch.float32)
    a_t_cf = torch.tensor(data["a_t_cf"], dtype=torch.float32)

    dataset = TensorDataset(x_t, a_t, x_tp1, eps_t, x_tp1_cf, a_t_cf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    int_mses, cf_mses, abs_errors = [], [], []

    for batch in loader:
        bx_t, ba_t, bx_tp1, beps_t, bx_tp1_cf, ba_t_cf = [b.to(device) for b in batch]

        # 1. Interventional MSE
        out = model(bx_t, ba_t)
        x_pred = out["x_tp1_recon"]
        int_mse = torch.mean((x_pred - bx_tp1) ** 2).item()
        int_mses.append(int_mse)

        # 2. Counterfactual MSE
        x_cf_pred, z_cf_pred = model.counterfactual(bx_t, ba_t, bx_tp1, ba_t_cf)
        cf_mse = torch.mean((x_cf_pred - bx_tp1_cf) ** 2).item()
        cf_mses.append(cf_mse)

        # 5. Abduction recovery error (AC-LSCM only)
        from src.models import ACLSCM
        if isinstance(model, ACLSCM):
            z_tp1_true = model.encode_mean(bx_tp1)
            z_tp1_base = model.transition_base(model.encode_mean(bx_t), ba_t)
            eps_hat = z_tp1_true - z_tp1_base
            # True eps is in the structural equation space (z space), not obs space.
            # We report the L2 norm of the recovered noise.
            abs_err = torch.mean(torch.norm(eps_hat, dim=-1)).item()
            abs_errors.append(abs_err)

    metrics = {
        "intervention_mse": float(np.mean(int_mses)),
        "counterfactual_mse": float(np.mean(cf_mses)),
    }

    # 3. SHD and 4. DAG constraint (AC-LSCM only)
    from src.models import ACLSCM
    if isinstance(model, ACLSCM):
        A_learned = model.get_adjacency(hard=True).cpu().numpy()
        # true_adjacency needs to be passed in via data
        if "true_adjacency" in data:
            metrics["shd"] = compute_shd(A_learned, data["true_adjacency"])
        A_soft = torch.sigmoid(model.A_logits).float()
        diag_mask = 1.0 - torch.eye(K, device=model.A_logits.device)
        A_soft = A_soft * diag_mask
        dag_val = (torch.trace(torch.matrix_exp(A_soft * A_soft)) - K).item()
        metrics["dag_constraint"] = float(dag_val)
        metrics["abduction_recovery_error"] = float(np.mean(abs_errors)) if abs_errors else float("nan")

    return metrics
