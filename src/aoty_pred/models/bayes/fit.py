"""MCMC fitting orchestration with GPU acceleration.

This module provides the infrastructure for fitting NumPyro models using MCMC
with GPU acceleration via JAX. Key features:
- NUTS kernel with configurable chain_method (sequential default for stability)
- Automatic GPU detection and logging
- Divergence tracking (logged but not failing - Phase 7 handles thresholds)
- ArviZ InferenceData conversion with observed/constant data groups
"""

import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import arviz as az
import jax
from jax import random
from numpyro.infer import MCMC, NUTS

__all__ = [
    "MCMCConfig",
    "FitResult",
    "fit_model",
    "get_gpu_info",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCMCConfig:
    """MCMC configuration for reproducibility.

    All parameters are frozen to ensure immutability during model fitting.

    Attributes:
        num_warmup: Number of warmup (burn-in) iterations per chain.
            Default 1000 is standard for publication.
        num_samples: Number of post-warmup samples per chain.
            Default 1000 provides 4000 total samples with 4 chains.
        num_chains: Number of parallel chains.
            Default 4 is standard for Rhat convergence assessment.
        chain_method: How to parallelize chains.
            "sequential" runs chains one at a time (default, most stable).
            "vectorized" runs chains on single GPU (faster but uses more memory).
            "parallel" uses pmap across multiple devices.
        seed: Random seed for reproducibility.
            Default 0 for consistent results.
        max_tree_depth: Maximum tree depth for NUTS.
            Default 10 is standard; increase if hitting tree depth limits.
        target_accept_prob: Target acceptance probability for adaptation.
            Default 0.8 is standard; increase to 0.9-0.95 if divergences occur.
    """

    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    chain_method: str = "sequential"
    seed: int = 0
    max_tree_depth: int = 10
    target_accept_prob: float = 0.8

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class FitResult:
    """Result container for MCMC fitting.

    This is not frozen because it contains large mutable objects (MCMC, InferenceData).

    Attributes:
        mcmc: NumPyro MCMC object with samples and extra fields.
        idata: ArviZ InferenceData with posterior, observed_data, constant_data groups.
        divergences: Total count of divergent transitions across all chains.
        runtime_seconds: Total wall-clock time for fitting.
        gpu_info: String describing GPU used (or "CPU only").
    """

    mcmc: MCMC
    idata: az.InferenceData
    divergences: int
    runtime_seconds: float
    gpu_info: str


def get_gpu_info() -> str:
    """Get GPU device info for reproducibility logging.

    Attempts to get detailed GPU information via nvidia-smi.
    Falls back to JAX device info if nvidia-smi is unavailable.

    Returns:
        String describing GPU (e.g., "NVIDIA GeForce RTX 5090, 32GB") or "CPU only".

    Example:
        >>> info = get_gpu_info()
        >>> print(info)  # "NVIDIA GeForce RTX 5090, 32768 MiB" or "CPU only"
    """
    # Check JAX devices first
    devices = jax.devices()
    has_gpu = any(d.platform == "gpu" for d in devices)

    if not has_gpu:
        return "CPU only"

    # Try nvidia-smi for detailed info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Format: "NVIDIA GeForce RTX 5090, 32768"
            gpu_infos = []
            for line in lines:
                parts = line.split(", ")
                if len(parts) == 2:
                    name, memory_mb = parts
                    gpu_infos.append(f"{name.strip()}, {memory_mb.strip()} MiB")
                else:
                    gpu_infos.append(line.strip())
            return "; ".join(gpu_infos)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # Fall back to JAX device kind
    gpu_devices = [d.device_kind for d in devices if d.platform == "gpu"]
    return ", ".join(gpu_devices) if gpu_devices else "GPU (unknown type)"


def fit_model(
    model: Callable,
    model_args: dict,
    config: MCMCConfig | None = None,
    progress_bar: bool = True,
) -> FitResult:
    """Fit NumPyro model via MCMC with GPU acceleration.

    Runs NUTS sampling with the specified configuration, converts results to
    ArviZ InferenceData, and logs basic statistics.

    Parameters
    ----------
    model : Callable
        NumPyro model function to fit (e.g., user_score_model).
    model_args : dict
        Arguments to pass to the model. Must include:
        - artist_idx: Integer array mapping observations to artists
        - album_seq: Integer array with album sequence numbers
        - prev_score: Float array with previous album scores
        - X: Feature matrix
        - y: Target scores
        - n_artists: Number of unique artists
        - max_seq: Maximum album sequence number
        Optional (for heteroscedastic models):
        - n_reviews: Array of per-observation review counts
        - n_exponent: Fixed exponent for noise scaling
        - learn_n_exponent: Whether to sample exponent from prior
    config : MCMCConfig, optional
        MCMC configuration. If None, uses default MCMCConfig().
    progress_bar : bool, default True
        Whether to display NumPyro's progress bar during sampling.

    Returns
    -------
    FitResult
        Container with MCMC object, InferenceData, divergence count,
        runtime, and GPU info.

    Example
    -------
    >>> from aoty_pred.models.bayes import user_score_model, fit_model, MCMCConfig
    >>> config = MCMCConfig(num_warmup=100, num_samples=100)
    >>> result = fit_model(user_score_model, model_args, config=config)
    >>> print(f"Divergences: {result.divergences}")
    >>> print(f"Runtime: {result.runtime_seconds:.1f}s")

    Notes
    -----
    Divergences are logged but do not cause failure. Phase 7 handles
    diagnostic thresholds and convergence assessment.
    """
    if config is None:
        config = MCMCConfig()

    # Get GPU info before fitting
    gpu_info = get_gpu_info()
    logger.info(f"GPU info: {gpu_info}")
    logger.info(f"JAX default backend: {jax.default_backend()}")

    # Create NUTS kernel with specified settings
    kernel = NUTS(
        model,
        max_tree_depth=config.max_tree_depth,
        target_accept_prob=config.target_accept_prob,
    )

    # Create MCMC runner
    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        chain_method=config.chain_method,
        progress_bar=progress_bar,
    )

    # Generate random key (using modern JAX API)
    rng_key = random.key(config.seed)

    # Run MCMC with timing
    logger.info(
        f"Starting MCMC: {config.num_chains} chains, "
        f"{config.num_warmup} warmup, {config.num_samples} samples"
    )
    start_time = time.perf_counter()

    mcmc.run(
        rng_key,
        extra_fields=("diverging", "num_steps"),
        **model_args,
    )

    runtime_seconds = time.perf_counter() - start_time

    # Count divergences
    diverging = mcmc.get_extra_fields()["diverging"]
    divergences = int(diverging.sum())

    logger.info(f"MCMC completed in {runtime_seconds:.1f}s")
    logger.info(f"Divergences: {divergences}")
    if divergences > 0:
        logger.warning(
            f"Found {divergences} divergent transitions. "
            "Consider increasing target_accept_prob or checking model specification."
        )

    # Convert to ArviZ InferenceData
    idata = az.from_numpyro(mcmc)

    # Prepare data groups for InferenceData
    observed_data = {"y": model_args["y"]}
    constant_data = {
        "X": model_args["X"],
        "artist_idx": model_args["artist_idx"],
        "album_seq": model_args["album_seq"],
        "prev_score": model_args["prev_score"],
    }
    # Include n_reviews for heteroscedastic models (if present)
    if "n_reviews" in model_args:
        constant_data["n_reviews"] = model_args["n_reviews"]

    # Add groups only if they don't already exist (az.from_numpyro may create observed_data)
    existing_groups = set(idata.groups())
    groups_to_add = {}
    if "observed_data" not in existing_groups:
        groups_to_add["observed_data"] = observed_data
    if "constant_data" not in existing_groups:
        groups_to_add["constant_data"] = constant_data

    if groups_to_add:
        idata.add_groups(**groups_to_add)

    logger.info(f"InferenceData groups: {list(idata.groups())}")

    return FitResult(
        mcmc=mcmc,
        idata=idata,
        divergences=divergences,
        runtime_seconds=runtime_seconds,
        gpu_info=gpu_info,
    )
