"""GPU memory estimation for MCMC runs.

Provides conservative memory estimates for NumPyro/JAX MCMC inference.
The formula intentionally overestimates to avoid OOM surprises during
long-running sampling jobs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryEstimate:
    """Memory estimation breakdown for MCMC run.

    This dataclass holds the components of GPU memory estimation,
    allowing inspection of where memory is expected to be used.

    Attributes:
        base_model_gb: Memory for parameters, priors, and feature matrix.
        per_chain_gb: Memory per chain for samples, gradients, and HMC state.
        jit_buffer_gb: Buffer for JIT compilation overhead.
        num_chains: Number of MCMC chains.
    """

    base_model_gb: float
    per_chain_gb: float
    jit_buffer_gb: float
    num_chains: int

    @property
    def total_gb(self) -> float:
        """Total estimated memory in GB.

        Returns:
            Sum of base model, chain memory, and JIT buffer.
        """
        return self.base_model_gb + self.chain_memory_gb + self.jit_buffer_gb

    @property
    def chain_memory_gb(self) -> float:
        """Memory for all chains combined in GB.

        Returns:
            Per-chain memory multiplied by number of chains.
        """
        return self.per_chain_gb * self.num_chains


def estimate_memory_gb(
    n_observations: int,
    n_features: int,
    n_artists: int,
    max_seq: int,
    num_chains: int,
    num_samples: int,
    num_warmup: int,
    jit_buffer_percent: float = 0.40,
) -> MemoryEstimate:
    """Estimate GPU memory for MCMC run.

    This is a CONSERVATIVE estimate - designed to overestimate rather than
    underestimate to avoid OOM surprises during long sampling runs.

    The formula accounts for:
    1. Base model memory: Parameters, priors, and feature matrix
    2. Per-chain memory: Sample storage, gradients, and HMC state
    3. JIT buffer: Compilation overhead (30-50% of above)

    The JIT buffer accounts for JAX/XLA compilation caches during initial
    tracing, which causes memory spikes before stabilization. NumPyro
    issue #936 documents "excessive memory usage at the beginning of NUTS
    inference" due to XLA compilation.

    Args:
        n_observations: Number of observations in dataset.
        n_features: Number of features in model.
        n_artists: Number of unique artists (for hierarchical effects).
        max_seq: Maximum sequence length for time-varying effects.
        num_chains: Number of parallel MCMC chains.
        num_samples: Number of samples per chain (post-warmup).
        num_warmup: Number of warmup samples per chain.
        jit_buffer_percent: JIT compilation buffer as fraction (default 0.40).
            Range 0.30-0.50 is recommended based on observed behavior.

    Returns:
        MemoryEstimate with breakdown of memory components.

    Example:
        >>> estimate = estimate_memory_gb(
        ...     n_observations=1000,
        ...     n_features=20,
        ...     n_artists=50,
        ...     max_seq=10,
        ...     num_chains=4,
        ...     num_samples=1000,
        ...     num_warmup=1000,
        ... )
        >>> print(f"Estimated: {estimate.total_gb:.2f} GB")
    """
    bytes_per_float = 4  # float32

    # Model parameters (approximate for hierarchical time-varying model)
    # - init_artist_effect: n_artists
    # - rw_raw: n_artists * (max_seq - 1)
    # - beta: n_features
    # - hyperpriors: ~10 scalars (sigma_obs, sigma_artist, etc.)
    n_params = n_artists + n_artists * max(0, max_seq - 1) + n_features + 10

    # Base model memory (parameters + feature matrix)
    base_bytes = (
        n_params * bytes_per_float  # Parameters
        + n_observations * n_features * bytes_per_float  # Feature matrix X
    )

    # Per-chain memory
    # - Samples: n_params * (num_warmup + num_samples)
    # - Gradients: n_params (during computation)
    # - HMC state: momentum, energy, etc. (~3x n_params)
    samples_per_chain = num_warmup + num_samples
    per_chain_bytes = (
        n_params * samples_per_chain * bytes_per_float  # Samples
        + n_params * bytes_per_float * 3  # Gradients + momentum + state
    )

    # Convert to GB (1024^3 to match nvidia-smi GiB output)
    base_gb = base_bytes / (1024**3)
    per_chain_gb = per_chain_bytes / (1024**3)

    # JIT buffer (applied to subtotal of base + chains)
    subtotal = base_gb + per_chain_gb * num_chains
    jit_buffer_gb = subtotal * jit_buffer_percent

    return MemoryEstimate(
        base_model_gb=base_gb,
        per_chain_gb=per_chain_gb,
        jit_buffer_gb=jit_buffer_gb,
        num_chains=num_chains,
    )
