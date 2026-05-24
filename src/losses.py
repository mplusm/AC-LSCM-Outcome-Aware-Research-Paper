"""Four-term composite loss for AC-LSCM."""
import torch
import torch.nn.functional as F


def reconstruction_loss(x_tp1: torch.Tensor, x_tp1_recon: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_tp1_recon, x_tp1)


def kl_divergence_gaussians(
    mu_q: torch.Tensor, logvar_q: torch.Tensor,
    mu_p: torch.Tensor, logvar_p: torch.Tensor,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians, averaged over latent dim."""
    # KL = 0.5 * sum[ logvar_p - logvar_q + (exp(logvar_q) + (mu_q-mu_p)^2) / exp(logvar_p) - 1 ]
    var_q = logvar_q.exp()
    var_p = logvar_p.exp().clamp(min=1e-8)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0)
    return kl.sum(dim=-1)  # sum over latent dim, shape (B,)


def transition_loss(
    mu_q: torch.Tensor, logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
) -> torch.Tensor:
    """KL between encoder posterior q(z_tp1|x_tp1) and transition prior p(z_tp1|z_t,a_t)=N(mu_p, I)."""
    logvar_p = torch.zeros_like(mu_p)  # unit variance prior
    return kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p).mean()


def dag_loss(
    A: torch.Tensor,
    K: int,
    lambda1: float = 0.1,
    lambda2: float = 1.0,
    acyclicity: str = "notears",
) -> torch.Tensor:
    """DAG acyclicity constraint.
    NOTEARS: tr(exp(A ∘ A)) - K  (zero iff DAG)
    DAGMA:   -logdet(sI - A ∘ A) + K*log(s)  (zero iff DAG)
    CRITICAL: keep in fp32.
    """
    A_fp32 = A.float()
    sparsity = A_fp32.abs().sum()

    if acyclicity == "notears":
        h = torch.trace(torch.matrix_exp(A_fp32 * A_fp32)) - K
    elif acyclicity == "dagma":
        s = 1.0
        eye = torch.eye(K, device=A.device, dtype=torch.float32)
        M = s * eye - A_fp32 * A_fp32
        sign, logdet = torch.linalg.slogdet(M)
        h = -logdet + K * torch.log(torch.tensor(s, dtype=torch.float32, device=A.device))
    else:
        raise ValueError(f"Unknown acyclicity: {acyclicity}")

    return lambda1 * sparsity + lambda2 * h


def contrastive_loss(
    z_tp1_cf_pred: torch.Tensor,
    z_tp1_cf_true: torch.Tensor,
    z_tp1_factual: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    supervised = F.mse_loss(z_tp1_cf_pred, z_tp1_cf_true)
    hinge = F.relu(margin - (z_tp1_cf_pred - z_tp1_factual).pow(2).sum(dim=-1)).mean()
    return supervised + hinge


def composite_loss(
    x_tp1: torch.Tensor,
    x_tp1_recon: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    A: torch.Tensor | None,
    K: int,
    z_tp1_cf_pred: torch.Tensor | None,
    z_tp1_cf_true: torch.Tensor | None,
    z_tp1_factual: torch.Tensor | None,
    beta_1: float = 1.0,
    beta_2: float = 1.0,
    beta_3: float = 0.5,
    beta_4: float = 1.0,
    acyclicity: str = "notears",
) -> dict:
    losses = {}

    losses["recon"] = reconstruction_loss(x_tp1, x_tp1_recon)

    if mu_q is not None and logvar_q is not None and mu_p is not None:
        losses["transition"] = transition_loss(mu_q, logvar_q, mu_p)
    else:
        losses["transition"] = torch.tensor(0.0, device=x_tp1.device)

    if A is not None and beta_3 > 0.0:
        losses["dag"] = dag_loss(A, K, acyclicity=acyclicity)
    else:
        losses["dag"] = torch.tensor(0.0, device=x_tp1.device)

    if z_tp1_cf_pred is not None and z_tp1_cf_true is not None and z_tp1_factual is not None and beta_4 > 0.0:
        losses["contrastive"] = contrastive_loss(z_tp1_cf_pred, z_tp1_cf_true, z_tp1_factual)
    else:
        losses["contrastive"] = torch.tensor(0.0, device=x_tp1.device)

    losses["total"] = (
        beta_1 * losses["recon"]
        + beta_2 * losses["transition"]
        + beta_3 * losses["dag"]
        + beta_4 * losses["contrastive"]
    )
    return losses
