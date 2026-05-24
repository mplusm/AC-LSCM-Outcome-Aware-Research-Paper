"""AC-LSCM and baseline models."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 0.5, hard: bool = False) -> torch.Tensor:
    u = torch.zeros_like(logits).uniform_().clamp(1e-8, 1 - 1e-8)
    gumbels = -torch.log(-torch.log(u))
    y_soft = torch.sigmoid((logits + gumbels) / tau)
    if hard:
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft  # straight-through
    return y_soft


# ---------------------------------------------------------------------------
# AC-LSCM
# ---------------------------------------------------------------------------

class ACLSCM(nn.Module):
    """
    Action-Conditioned Latent Structural Causal Model.
    Observation dim = K (x = z + small noise).
    """

    def __init__(
        self,
        K: int,
        obs_dim: int,
        hidden_dim: int = 128,
        factor_hidden: int = 32,
        use_do_operator: bool = True,
        acyclicity: str = "notears",
        gumbel_tau: float = 0.5,
    ):
        super().__init__()
        self.K = K
        self.obs_dim = obs_dim
        self.use_do_operator = use_do_operator
        self.acyclicity = acyclicity
        self.gumbel_tau = gumbel_tau

        # Encoder: obs_dim → hidden → 2*K (mean + log-var)
        self.encoder = _mlp(obs_dim, hidden_dim, 2 * K)

        # Decoder: K → hidden → obs_dim
        self.decoder = _mlp(K, hidden_dim, obs_dim)

        # Learnable adjacency logits (K×K), diagonal zeroed out
        self.A_logits = nn.Parameter(torch.zeros(K, K))
        self._diag_mask = None  # lazy init on device

        # Per-factor transition MLPs: input = K parent values + 1 action value → 1 output
        self.transition_mlps = nn.ModuleList([
            _mlp(K + 1, factor_hidden, 1) for _ in range(K)
        ])

    def _get_diag_mask(self, device):
        if self._diag_mask is None or self._diag_mask.device != device:
            self._diag_mask = 1.0 - torch.eye(self.K, device=device)
        return self._diag_mask

    def get_adjacency(self, hard: bool = False) -> torch.Tensor:
        """Returns soft (training) or hard (eval) adjacency, diagonal zeroed."""
        diag_mask = self._get_diag_mask(self.A_logits.device)
        if self.training and not hard:
            A = gumbel_sigmoid(self.A_logits, tau=self.gumbel_tau, hard=False) * diag_mask
        else:
            A = (torch.sigmoid(self.A_logits) > 0.5).float() * diag_mask
        return A

    def encode(self, x: torch.Tensor):
        """Returns (mu, logvar) each shape (B, K)."""
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        logvar = logvar.clamp(-4, 4)
        return mu, logvar

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def transition_base(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """
        Deterministic transition (no noise).
        z_t: (B, K), a_t: (B, K) with NaN for non-intervened.
        Returns z_tp1_pred: (B, K).
        """
        B, K = z_t.shape
        A = self.get_adjacency()  # (K, K)

        intervention_mask = ~torch.isnan(a_t)  # (B, K), True where intervened
        a_clean = torch.nan_to_num(a_t, nan=0.0)  # (B, K)

        z_tp1_parts = []
        for i in range(K):
            # Weighted parent values for factor i: A[i] * z_t  shape (B, K)
            parent_vals = A[i].unsqueeze(0) * z_t  # (B, K)

            if self.use_do_operator:
                # Zero parent contribution for factors that are intervened on
                int_i = intervention_mask[:, i].float().unsqueeze(-1)  # (B, 1)
                parent_vals = parent_vals * (1.0 - int_i)
                action_val = a_clean[:, i:i+1] * int_i  # (B, 1)
            else:
                # No do-operator: just concatenate action
                action_val = a_clean[:, i:i+1]  # (B, 1)

            f_input = torch.cat([parent_vals, action_val], dim=-1)  # (B, K+1)
            z_i = self.transition_mlps[i](f_input)  # (B, 1)
            z_tp1_parts.append(z_i)

        return torch.cat(z_tp1_parts, dim=-1)  # (B, K)

    def forward(self, x_t: torch.Tensor, a_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_t = self.encode_mean(x_t)
        z_tp1_pred = self.transition_base(z_t, a_t)
        x_tp1_recon = self.decode(z_tp1_pred)
        A = self.get_adjacency()
        return {
            "z_t": z_t,
            "z_tp1_pred": z_tp1_pred,
            "x_tp1_recon": x_tp1_recon,
            "adjacency": A,
        }

    def counterfactual(
        self,
        x_t: torch.Tensor,
        a_t: torch.Tensor,
        x_tp1: torch.Tensor,
        a_t_cf: torch.Tensor,
    ) -> torch.Tensor:
        """Abduction → intervention → prediction."""
        z_t = self.encode_mean(x_t)
        z_tp1_factual = self.encode_mean(x_tp1)
        z_tp1_pred_factual = self.transition_base(z_t, a_t)
        # Abduction: recover noise
        eps_t = z_tp1_factual - z_tp1_pred_factual
        # Counterfactual prediction with same noise
        z_tp1_cf = self.transition_base(z_t, a_t_cf) + eps_t
        return self.decode(z_tp1_cf), z_tp1_cf


# ---------------------------------------------------------------------------
# Vanilla VAE
# ---------------------------------------------------------------------------

class VanillaVAE(nn.Module):
    def __init__(self, K: int, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.K = K
        self.obs_dim = obs_dim
        self.encoder = _mlp(obs_dim, hidden_dim, 2 * K)
        self.decoder = _mlp(K, hidden_dim, obs_dim)
        # Transition: (z_t, a_t_clean) → z_tp1; action NaN replaced with 0
        self.transition_net = _mlp(K + K, hidden_dim, K)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        logvar = logvar.clamp(-4, 4)
        return mu, logvar

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def transition_base(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        a_clean = torch.nan_to_num(a_t, nan=0.0)
        return self.transition_net(torch.cat([z_t, a_clean], dim=-1))

    def forward(self, x_t: torch.Tensor, a_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_t = self.encode_mean(x_t)
        z_tp1_pred = self.transition_base(z_t, a_t)
        x_tp1_recon = self.decode(z_tp1_pred)
        return {
            "z_t": z_t,
            "z_tp1_pred": z_tp1_pred,
            "x_tp1_recon": x_tp1_recon,
            "adjacency": None,
        }

    def counterfactual(
        self,
        x_t: torch.Tensor,
        a_t: torch.Tensor,
        x_tp1: torch.Tensor,
        a_t_cf: torch.Tensor,
    ) -> torch.Tensor:
        """No true counterfactual mechanism; forward pass with different action."""
        z_t = self.encode_mean(x_t)
        z_tp1_cf = self.transition_base(z_t, a_t_cf)
        return self.decode(z_tp1_cf), z_tp1_cf


# ---------------------------------------------------------------------------
# Dreamer-style (GRU transition)
# ---------------------------------------------------------------------------

class DreamerStyle(nn.Module):
    def __init__(self, K: int, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.K = K
        self.obs_dim = obs_dim
        self.encoder = _mlp(obs_dim, hidden_dim, 2 * K)
        self.decoder = _mlp(K, hidden_dim, obs_dim)
        # GRU: input = (z_t, a_t_clean), hidden_size = K
        self.gru = nn.GRUCell(input_size=K + K, hidden_size=K)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        logvar = logvar.clamp(-4, 4)
        return mu, logvar

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def transition_base(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        a_clean = torch.nan_to_num(a_t, nan=0.0)
        inp = torch.cat([z_t, a_clean], dim=-1)
        return self.gru(inp, z_t)

    def forward(self, x_t: torch.Tensor, a_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_t = self.encode_mean(x_t)
        z_tp1_pred = self.transition_base(z_t, a_t)
        x_tp1_recon = self.decode(z_tp1_pred)
        return {
            "z_t": z_t,
            "z_tp1_pred": z_tp1_pred,
            "x_tp1_recon": x_tp1_recon,
            "adjacency": None,
        }

    def counterfactual(
        self,
        x_t: torch.Tensor,
        a_t: torch.Tensor,
        x_tp1: torch.Tensor,
        a_t_cf: torch.Tensor,
    ) -> torch.Tensor:
        z_t = self.encode_mean(x_t)
        z_tp1_cf = self.transition_base(z_t, a_t_cf)
        return self.decode(z_tp1_cf), z_tp1_cf


# ---------------------------------------------------------------------------
# Small Transformer
# ---------------------------------------------------------------------------

class SmallTransformer(nn.Module):
    """
    Treats (x_t, a_t) as a sequence of 2K scalar tokens,
    each embedded to d_model=128. Predicts x_tp1 via causal transformer.
    """

    def __init__(self, K: int, obs_dim: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.K = K
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.seq_len = 2 * K  # x_t tokens + a_t tokens

        # Per-position scalar → d_model embedding
        self.input_proj = nn.Linear(1, d_model)
        # Learnable position embeddings
        self.pos_emb = nn.Embedding(self.seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Pool and project to obs_dim
        self.out_proj = _mlp(d_model, d_model, obs_dim)

    def _tokenize(self, x_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """Concatenate x_t and a_t (NaN→0) tokens, each scalar projected to d_model."""
        a_clean = torch.nan_to_num(a_t, nan=0.0)
        tokens = torch.cat([x_t, a_clean], dim=-1)  # (B, 2K)
        B, S = tokens.shape
        tokens = tokens.unsqueeze(-1)  # (B, 2K, 1)
        tok_emb = self.input_proj(tokens)  # (B, 2K, d_model)
        pos = torch.arange(S, device=tokens.device)
        tok_emb = tok_emb + self.pos_emb(pos).unsqueeze(0)
        return tok_emb  # (B, 2K, d_model)

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        # Transformer has no separate encoder; return x directly as "latent"
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def transition_base(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        return z_t  # unused for transformer

    def forward(self, x_t: torch.Tensor, a_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        tok = self._tokenize(x_t, a_t)  # (B, 2K, d_model)
        h = self.transformer(tok)  # (B, 2K, d_model)
        pooled = h.mean(dim=1)  # (B, d_model)
        x_tp1_recon = self.out_proj(pooled)  # (B, obs_dim)
        return {
            "z_t": x_t,  # no meaningful latent
            "z_tp1_pred": x_tp1_recon,
            "x_tp1_recon": x_tp1_recon,
            "adjacency": None,
        }

    def counterfactual(
        self,
        x_t: torch.Tensor,
        a_t: torch.Tensor,
        x_tp1: torch.Tensor,
        a_t_cf: torch.Tensor,
    ) -> torch.Tensor:
        out = self.forward(x_t, a_t_cf)
        x_cf = out["x_tp1_recon"]
        return x_cf, x_cf


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(name: str, K: int, obs_dim: int) -> nn.Module:
    name = name.lower()
    if name == "aclscm":
        return ACLSCM(K=K, obs_dim=obs_dim)
    elif name == "vanilla_vae":
        return VanillaVAE(K=K, obs_dim=obs_dim)
    elif name == "dreamer_style":
        return DreamerStyle(K=K, obs_dim=obs_dim)
    elif name == "small_transformer":
        return SmallTransformer(K=K, obs_dim=obs_dim)
    else:
        raise ValueError(f"Unknown model: {name}")
