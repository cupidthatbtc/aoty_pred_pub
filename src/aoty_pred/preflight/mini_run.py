"""Mini-MCMC run for GPU memory measurement.

This module is designed to be run as a subprocess entry point for measuring
actual peak GPU memory usage during MCMC. It runs a minimal MCMC with
configurable warmup and sample counts and reports peak memory via JSON to stdout.

Usage:
    python -m aoty_pred.preflight.mini_run /path/to/model_args.json
    python -m aoty_pred.preflight.mini_run /path/to/model_args.json --num-warmup 10 --num-samples 50

The model_args.json file should contain:
    - artist_idx: List of integers mapping observations to artists
    - album_seq: List of integers with album sequence numbers
    - prev_score: List of floats with previous album scores
    - X: 2D list of floats (feature matrix)
    - y: List of floats (target scores)
    - n_artists: Integer count of unique artists
    - max_seq: Integer maximum album sequence number
    - n_reviews: Optional list of floats (per-observation review counts)
    - n_exponent: Optional float (heteroscedastic exponent)
    - learn_n_exponent: Optional bool (whether to sample exponent)

Output (stdout, JSON):
    {
        "success": true,
        "exit_code": 0,
        "peak_memory_bytes": 4567890123,
        "runtime_seconds": 45.2
    }

On error:
    {
        "success": false,
        "exit_code": 1,
        "error": "Error message here",
        "peak_memory_bytes": 0,
        "runtime_seconds": 0.0
    }

Note: All JSON output goes to stdout. Logs/warnings go to stderr.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["run_and_measure"]

logger = logging.getLogger(__name__)


def run_and_measure(
    model_args_path: Path,
    num_warmup: int = 10,
    num_samples: int = 1,
) -> dict[str, Any]:
    """Run mini-MCMC and measure peak GPU memory.

    Loads model arguments from JSON, runs MCMC with specified warmup and
    sample counts (default: 1 chain, 10 warmup, 1 sample), and returns
    peak memory usage from JAX device stats.

    Args:
        model_args_path: Path to JSON file with model arguments.
        num_warmup: Number of warmup iterations (default 10).
        num_samples: Number of post-warmup samples (default 1).

    Returns:
        Dictionary with:
            - success: True if measurement completed
            - peak_memory_bytes: Peak GPU memory in bytes
            - runtime_seconds: Wall-clock time for mini-run
    """
    # Import JAX/NumPyro at function level for subprocess isolation
    import jax.numpy as jnp
    from jax import random
    from numpyro.infer import MCMC, NUTS

    from aoty_pred.gpu_memory.measure import get_jax_memory_stats
    from aoty_pred.models.bayes.model import make_score_model

    # Load model args from JSON
    logger.info(f"Loading model args from {model_args_path}")
    with open(model_args_path, encoding="utf-8") as f:
        args_json = json.load(f)

    # Validate required keys exist before building model_args
    required_keys = ["artist_idx", "album_seq", "prev_score", "X", "y", "n_artists", "max_seq"]
    missing_keys = [k for k in required_keys if k not in args_json]
    if missing_keys:
        raise ValueError(
            f"Missing required keys in model args JSON: {missing_keys}. "
            f"Present keys: {list(args_json.keys())}"
        )

    # Convert lists to JAX arrays where needed
    model_args: dict[str, Any] = {
        # Integer arrays
        "artist_idx": jnp.array(args_json["artist_idx"], dtype=jnp.int32),
        "album_seq": jnp.array(args_json["album_seq"], dtype=jnp.int32),
        # Float arrays
        "prev_score": jnp.array(args_json["prev_score"], dtype=jnp.float32),
        "X": jnp.array(args_json["X"], dtype=jnp.float32),
        "y": jnp.array(args_json["y"], dtype=jnp.float32),
        # Scalar integers
        "n_artists": args_json["n_artists"],
        "max_seq": args_json["max_seq"],
    }

    # Optional heteroscedastic parameters
    if "n_reviews" in args_json:
        model_args["n_reviews"] = jnp.array(
            args_json["n_reviews"], dtype=jnp.float32
        )
    if "n_exponent" in args_json:
        model_args["n_exponent"] = args_json["n_exponent"]
    if "learn_n_exponent" in args_json:
        model_args["learn_n_exponent"] = args_json["learn_n_exponent"]

    # Create NUTS kernel with user score model
    # Using "user" model: consistent with CLI default behavior, most common use case
    model = make_score_model("user")
    kernel = NUTS(model)

    # Configure MCMC: mini-run parameters for memory measurement
    # 1 chain, configurable warmup (captures JIT overhead), configurable samples
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
    )

    logger.info("Starting mini-MCMC run")
    start_time = time.perf_counter()

    # Run MCMC
    rng_key = random.key(0)
    mcmc.run(rng_key, **model_args)

    runtime = time.perf_counter() - start_time
    logger.info(f"Mini-MCMC completed in {runtime:.2f}s")

    # Get peak memory from JAX device stats
    stats = get_jax_memory_stats()
    peak_bytes = stats.peak_bytes_in_use

    logger.info(f"Peak memory: {stats.peak_gb:.2f} GB")

    return {
        "success": True,
        "exit_code": 0,
        "peak_memory_bytes": peak_bytes,
        "runtime_seconds": runtime,
    }


def _parse_args(args: list[str]) -> tuple[Path, int, int]:
    """Parse command line arguments.

    Args:
        args: List of command line arguments (excluding script name).

    Returns:
        Tuple of (model_args_path, num_warmup, num_samples).

    Raises:
        ValueError: If arguments are invalid.
    """
    if not args:
        raise ValueError(
            "Usage: python -m aoty_pred.preflight.mini_run <model_args.json> "
            "[--num-warmup N] [--num-samples N]"
        )

    model_args_path = Path(args[0])
    num_warmup = 10
    num_samples = 1

    # Parse optional arguments
    i = 1
    while i < len(args):
        if args[i] == "--num-warmup":
            if i + 1 >= len(args):
                raise ValueError("--num-warmup requires an integer value")
            try:
                num_warmup = int(args[i + 1])
            except ValueError:
                raise ValueError(
                    f"--num-warmup requires an integer value, got: {args[i + 1]}"
                ) from None
            i += 2
        elif args[i] == "--num-samples":
            if i + 1 >= len(args):
                raise ValueError("--num-samples requires an integer value")
            try:
                num_samples = int(args[i + 1])
            except ValueError:
                raise ValueError(
                    f"--num-samples requires an integer value, got: {args[i + 1]}"
                ) from None
            i += 2
        else:
            raise ValueError(f"Unknown argument: {args[i]}")

    # Validate positive values
    if num_warmup <= 0:
        raise ValueError(f"--num-warmup must be positive, got: {num_warmup}")
    if num_samples <= 0:
        raise ValueError(f"--num-samples must be positive, got: {num_samples}")

    return model_args_path, num_warmup, num_samples


if __name__ == "__main__":
    # Configure logging to stderr only (stdout reserved for JSON output)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        model_args_path, num_warmup, num_samples = _parse_args(sys.argv[1:])
    except ValueError as e:
        result = {
            "success": False,
            "exit_code": 1,
            "error": str(e),
            "peak_memory_bytes": 0,
            "runtime_seconds": 0.0,
        }
        print(json.dumps(result))
        sys.exit(1)

    try:
        result = run_and_measure(model_args_path, num_warmup, num_samples)
        print(json.dumps(result))
    except Exception as e:
        logger.exception("Mini-run failed")
        result = {
            "success": False,
            "exit_code": 1,
            "error": str(e),
            "peak_memory_bytes": 0,
            "runtime_seconds": 0.0,
        }
        print(json.dumps(result))
        sys.exit(1)
