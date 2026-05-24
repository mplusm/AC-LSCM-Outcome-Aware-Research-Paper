"""Ground-truth synthetic SCM data generator."""
import numpy as np
import networkx as nx


class SyntheticSCM:
    def __init__(self, graph_type: str, K: int, seed: int, edge_prob: float = 0.3):
        self.graph_type = graph_type
        self.K = K
        self.seed = seed
        self.edge_prob = edge_prob
        self.noise_sigma = 0.1
        self.obs_noise_sigma = 0.1

        rng = np.random.RandomState(seed)
        self._build_dag(rng)
        self._sample_weights(rng)

    # ------------------------------------------------------------------
    # DAG construction
    # ------------------------------------------------------------------

    def _build_dag(self, rng: np.random.RandomState):
        K = self.K
        A = np.zeros((K, K), dtype=np.float32)

        if self.graph_type == "chain":
            for i in range(K - 1):
                A[i + 1, i] = 1.0

        elif self.graph_type == "fork":
            # z0 → {z1, ..., zK-1}
            for i in range(1, K):
                A[i, 0] = 1.0

        elif self.graph_type == "collider":
            # {z0, ..., zK-2} → zK-1
            for i in range(K - 1):
                A[K - 1, i] = 1.0

        elif self.graph_type == "er_random":
            # Erdős–Rényi DAG: upper-triangular then topologically sort
            adj = np.zeros((K, K), dtype=np.float32)
            for i in range(K):
                for j in range(i + 1, K):
                    if rng.rand() < self.edge_prob:
                        adj[i, j] = 1.0
            # Permute node order randomly to break trivial ordering
            perm = rng.permutation(K)
            adj = adj[perm][:, perm]
            # Topological sort via DFS to produce a valid DAG adjacency
            G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
            order = list(nx.topological_sort(G))
            inv = np.argsort(order)
            adj = adj[order][:, order]
            # A[i,j]=1 means j is parent of i (j → i)
            A = adj.T  # convert to child,parent convention
        else:
            raise ValueError(f"Unknown graph_type: {self.graph_type}")

        self._A = A  # A[child, parent] = 1 if parent→child

    def _sample_weights(self, rng: np.random.RandomState):
        K = self.K
        # Random weights in [-2, -0.5] ∪ [0.5, 2]
        W = rng.uniform(0.5, 2.0, size=(K, K)).astype(np.float32)
        signs = rng.choice([-1, 1], size=(K, K)).astype(np.float32)
        self._W = W * signs * self._A  # zero out non-edges

    # ------------------------------------------------------------------
    # Structural equations
    # ------------------------------------------------------------------

    def _compute_z_from_parents(
        self, z: np.ndarray, eps: np.ndarray, intervention: dict | None = None
    ) -> np.ndarray:
        """
        Compute next z using additive non-linear structural equations.
        z_i = tanh(sum_j W_ij * z_j) + eps_i  for j in Pa(i)
        With do(z_i = v): z_i = v, ignoring parents.
        """
        K = self.K
        z_out = np.zeros(K, dtype=np.float32)
        for i in range(K):
            if intervention is not None and i in intervention:
                z_out[i] = intervention[i]
            else:
                parent_sum = np.sum(self._W[i] * z)  # W[i,j]*z[j] for all j (non-edges are 0)
                z_out[i] = np.tanh(parent_sum) + eps[i]
        return z_out

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n_samples: int, action_prob: float = 0.5, rng_seed: int = None):
        """
        Returns dict with keys:
            x_t, a_t, x_tp1, eps_t, x_tp1_cf, a_t_cf
        All arrays shape (n_samples, K).
        a_t has NaN for non-intervened factors.
        """
        K = self.K
        rng = np.random.RandomState(rng_seed if rng_seed is not None else self.seed + 1000)

        x_t_list, a_t_list, x_tp1_list = [], [], []
        eps_t_list, x_tp1_cf_list, a_t_cf_list, z_tp1_cf_list = [], [], [], []

        # Initialise first z from standard normal
        z = rng.randn(K).astype(np.float32) * 0.5

        for _ in range(n_samples):
            # Observation: z + small Gaussian noise
            x_t = z + rng.randn(K).astype(np.float32) * self.obs_noise_sigma

            # Sample noise for this step
            eps = rng.randn(K).astype(np.float32) * self.noise_sigma

            # Sample action: intervene on one factor with probability action_prob
            a = np.full(K, np.nan, dtype=np.float32)
            if rng.rand() < action_prob:
                idx = rng.randint(K)
                val = rng.uniform(-1.5, 1.5)
                a[idx] = val

            # Build intervention dict for factual action
            intervention = {}
            for i in range(K):
                if not np.isnan(a[i]):
                    intervention[i] = a[i]

            # Compute factual next state
            z_tp1 = self._compute_z_from_parents(z, eps, intervention if intervention else None)
            x_tp1 = z_tp1 + rng.randn(K).astype(np.float32) * self.obs_noise_sigma

            # Counterfactual: different action, same noise eps
            a_cf = np.full(K, np.nan, dtype=np.float32)
            cf_idx = rng.randint(K)
            cf_val = rng.uniform(-1.5, 1.5)
            # Ensure counterfactual is actually different
            while cf_idx == (list(intervention.keys())[0] if intervention else -1) and abs(cf_val - a[cf_idx]) < 0.1:
                cf_val = rng.uniform(-1.5, 1.5)
            a_cf[cf_idx] = cf_val

            cf_intervention = {cf_idx: cf_val}
            z_tp1_cf = self._compute_z_from_parents(z, eps, cf_intervention)
            x_tp1_cf = z_tp1_cf + rng.randn(K).astype(np.float32) * self.obs_noise_sigma

            x_t_list.append(x_t)
            a_t_list.append(a)
            x_tp1_list.append(x_tp1)
            eps_t_list.append(eps)
            x_tp1_cf_list.append(x_tp1_cf)
            a_t_cf_list.append(a_cf)
            z_tp1_cf_list.append(z_tp1_cf)

            # Advance state
            z = z_tp1

        return {
            "x_t": np.stack(x_t_list),
            "a_t": np.stack(a_t_list),
            "x_tp1": np.stack(x_tp1_list),
            "eps_t": np.stack(eps_t_list),
            "x_tp1_cf": np.stack(x_tp1_cf_list),
            "a_t_cf": np.stack(a_t_cf_list),
            "z_tp1_cf": np.stack(z_tp1_cf_list),  # ground-truth CF latent (no obs noise)
        }

    @property
    def true_adjacency(self) -> np.ndarray:
        return self._A.copy()
