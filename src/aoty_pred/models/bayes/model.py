"""Bayesian hierarchical model for album score prediction.

This module defines the core NumPyro models with:
- Hierarchical artist effects with partial pooling
- Time-varying artist effects via random walk (career trajectory modeling)
- AR(1) structure for album-to-album score dependencies
- Non-centered parameterization to avoid funnel geometry
- Fixed effects for covariates (genre PCA, release year, etc.)
- Factory pattern for user_score and critic_score model variants

The extended model structure:
    y_ij ~ Normal(mu_ij, sigma_obs)
    mu_ij = artist_effect_jt + X_ij @ beta + rho * prev_score_ij

    # Time-varying artist effect via random walk
    artist_effect_j1 ~ Normal(mu_artist, sigma_artist)  # initial effect
    artist_effect_jt = artist_effect_j(t-1) + N(0, sigma_rw)  # random walk

    # Hyperpriors
    mu_artist ~ Normal(mu_artist_loc, mu_artist_scale)
    sigma_artist ~ HalfNormal(sigma_artist_scale)
    sigma_rw ~ HalfNormal(sigma_rw_scale)
    rho ~ TruncatedNormal(rho_loc, rho_scale, -0.99, 0.99)

    beta ~ Normal(beta_loc, beta_scale)  # fixed effects
    sigma_obs ~ HalfNormal(sigma_obs_scale)  # observation noise

Non-centered parameterization is applied via LocScaleReparam to transform
the init_artist_effect sampling for efficient NUTS sampling.

Model variants:
- user_score_model: For user score prediction (prefix: "user_")
- critic_score_model: For critic score prediction (prefix: "critic_")
"""

from typing import Callable

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

from aoty_pred.models.bayes.priors import PriorConfig, get_default_priors

__all__ = [
    "compute_sigma_scaled",
    "make_score_model",
    "user_score_model",
    "critic_score_model",
    "album_score_model",
]


def compute_sigma_scaled(
    sigma_obs: float,
    n_reviews: jnp.ndarray,
    exponent: float,
    single_review_multiplier: float = 2.0,
    min_sigma: float = 0.01,
) -> jnp.ndarray:
    """Compute per-observation sigma scaled by review count.

    Implements heteroscedastic observation noise where albums with more reviews
    have lower noise (more reliable scores). Uses log-space arithmetic to
    avoid numerical underflow/overflow with extreme review counts.

    The scaling formula is:
        sigma_scaled = sigma_obs / n_reviews^exponent

    Special cases:
        - n_reviews=1: Applies multiplier (default 2x) for unreliable single reviews
        - exponent=0: Returns sigma_obs unchanged (homoscedastic mode)
        - Large n_reviews: Floored at min_sigma to prevent numerical issues

    Args:
        sigma_obs: Base observation noise scale (scalar).
        n_reviews: Array of review counts per observation. Values < 1 are
            clamped to 1.0 defensively.
        exponent: Power-law exponent for scaling. Typically in [0.3, 0.7].
            exponent=0 gives homoscedastic (constant) noise.
            exponent=0.5 gives square-root scaling.
        single_review_multiplier: Multiplier applied to sigma_obs when
            n_reviews=1. Default 2.0 reflects that single reviews are
            unreliable indicators of true album quality.
        min_sigma: Minimum sigma floor for numerical stability. Default 0.01
            prevents underflow with very large review counts.

    Returns:
        Array of scaled sigma values, same shape as n_reviews.

    Notes:
        Log-space arithmetic is used to compute sigma_obs / n^exponent as:
            exp(log(sigma_obs) - exponent * log(n))
        This avoids overflow/underflow for extreme n values (e.g., n=100,000).

    Example:
        >>> import jax.numpy as jnp
        >>> sigma = compute_sigma_scaled(1.0, jnp.array([100.0]), 0.5)
        >>> print(f"{sigma[0]:.2f}")  # ~0.10 (1.0 / sqrt(100))
        0.10
        >>> sigma = compute_sigma_scaled(1.0, jnp.array([1.0]), 0.5)
        >>> print(f"{sigma[0]:.2f}")  # 2.0 (single review penalty)
        2.00
    """
    # Clamp n_reviews to minimum of 1.0 (defensive against invalid data)
    n_safe = jnp.maximum(n_reviews, 1.0)

    # Log-space computation: sigma_obs / n^exponent = exp(log(sigma_obs) - exponent * log(n))
    log_sigma = jnp.log(sigma_obs) - exponent * jnp.log(n_safe)
    sigma_scaled = jnp.exp(log_sigma)

    # Apply single-review penalty (n=1 is unreliable)
    # Use robust comparison instead of exact float equality
    is_single_review = jnp.isclose(n_safe, 1.0, rtol=1e-6, atol=1e-6)
    # Only apply single-review penalty in heteroscedastic mode (exponent > 0)
    # In homoscedastic mode, sigma_scaled already equals sigma_obs, so no penalty needed
    apply_penalty = jnp.logical_and(is_single_review, exponent > 0)
    sigma_scaled = jnp.where(
        apply_penalty, sigma_obs * single_review_multiplier, sigma_scaled
    )

    # Apply minimum floor for numerical stability
    sigma_scaled = jnp.maximum(sigma_scaled, min_sigma)

    return sigma_scaled


def make_score_model(score_type: str) -> Callable:
    """Factory function to create score prediction models.

    Creates a NumPyro model function with score-type-specific parameter prefixes.
    This allows fitting separate models for user scores and critic scores with
    distinct posterior distributions.

    Parameters
    ----------
    score_type : str
        Either "user" or "critic" to create score-specific models.
        The score_type becomes a prefix for all sample site names
        (e.g., "user_beta", "critic_rho").

    Returns
    -------
    Callable
        NumPyro model function with non-centered parameterization.
        The returned function has signature:
            model(artist_idx, album_seq, prev_score, X, y=None, n_artists=None, max_seq=None, priors=None)

    Example
    -------
    >>> user_model = make_score_model("user")
    >>> critic_model = make_score_model("critic")
    >>>
    >>> # User model samples will have prefixes like "user_beta", "user_rho"
    >>> # Critic model samples will have prefixes like "critic_beta", "critic_rho"
    """
    if score_type not in ("user", "critic"):
        raise ValueError(f"score_type must be 'user' or 'critic', got {score_type!r}")

    prefix = f"{score_type}_"

    def _score_model_centered(
        artist_idx: jnp.ndarray,
        album_seq: jnp.ndarray,
        prev_score: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray | None = None,
        n_artists: int | None = None,
        max_seq: int | None = None,
        priors: PriorConfig | None = None,
        n_reviews: jnp.ndarray | None = None,
        n_exponent: float = 0.0,
        learn_n_exponent: bool = False,
        n_exponent_prior: str = "logit-normal",
    ) -> None:
        """Centered parameterization of score model (internal).

        This function defines the model in centered form for the initial
        artist effect. Use the reparameterized version returned by
        make_score_model for actual inference.

        Args:
            artist_idx: Integer array of shape (n_obs,) mapping each observation
                to an artist index in [0, n_artists).
            album_seq: Integer array of shape (n_obs,) with album sequence numbers
                for each observation (1 = first album, 2 = second, etc.).
                Used for time-varying artist effects.
            prev_score: Float array of shape (n_obs,) with previous album scores.
                Should be 0.0 for debut albums (first album by artist).
            X: Feature matrix of shape (n_obs, n_features) containing covariates
                such as genre PCA components, release year, etc.
            y: Optional target scores of shape (n_obs,). Pass None for prior
                predictive sampling or posterior predictive on new data.
            n_artists: Number of unique artists. Must be provided.
            max_seq: Maximum album sequence number in the data. Must be provided
                for JAX tracing. Compute as int(album_seq.max()) before calling.
            priors: Prior configuration. If None, uses get_default_priors().
            n_reviews: Optional array of shape (n_obs,) with per-observation
                review counts. Used for heteroscedastic noise scaling. If None,
                homoscedastic noise is used (scalar sigma_obs for all observations).
            n_exponent: Fixed exponent for heteroscedastic noise scaling.
                Default 0.0 gives homoscedastic (constant) noise. Higher values
                give more noise reduction for albums with many reviews.
            learn_n_exponent: If True, sample the exponent from a prior distribution
                instead of using the fixed n_exponent value.
            n_exponent_prior: Prior type for learned n_exponent. Options:
                - "logit-normal" (default): Uses TransformedDistribution with
                  Normal(loc, scale) and SigmoidTransform. Avoids funnel geometry.
                - "beta" (legacy): Uses Beta(alpha, beta) distribution. May cause
                  divergences due to funnel geometry in the likelihood.

        Model structure:
            - Population-level hyperpriors for artist effect distribution
            - Time-varying artist effects via random walk
            - AR(1) term for album-to-album dependency
            - Fixed effects for covariates
            - Observation-level noise (optionally heteroscedastic)
        """
        # Get prior configuration
        if priors is None:
            priors = get_default_priors()

        if n_artists is None:
            raise ValueError("n_artists must be provided")

        if max_seq is None:
            raise ValueError("max_seq must be provided (compute as int(album_seq.max()) before calling)")

        n_features = X.shape[1]

        # === Population-level hyperpriors ===
        # Mean of artist quality distribution
        mu_artist = numpyro.sample(
            f"{prefix}mu_artist",
            dist.Normal(priors.mu_artist_loc, priors.mu_artist_scale),
        )

        # Between-artist standard deviation (controls pooling strength)
        sigma_artist = numpyro.sample(
            f"{prefix}sigma_artist",
            dist.HalfNormal(priors.sigma_artist_scale),
        )

        # Random walk innovation scale (controls career trajectory smoothness)
        sigma_rw = numpyro.sample(
            f"{prefix}sigma_rw",
            dist.HalfNormal(priors.sigma_rw_scale),
        )

        # AR(1) coefficient for album-to-album dependency
        # Truncated to ensure stationarity
        rho = numpyro.sample(
            f"{prefix}rho",
            dist.TruncatedNormal(
                loc=priors.rho_loc,
                scale=priors.rho_scale,
                low=-0.99,
                high=0.99,
            ),
        )

        # === Initial artist effects (partial pooling) ===
        with numpyro.plate(f"{prefix}artists", n_artists):
            init_artist_effect = numpyro.sample(
                f"{prefix}init_artist_effect",
                dist.Normal(mu_artist, sigma_artist),
            )

        # === Time-varying artist effects via random walk ===
        # Use GaussianRandomWalk distribution for memory efficiency:
        # - Samples entire trajectory as single tensor instead of max_seq separate parameters
        # - Shape: (n_artists, max_seq-1) for innovations, then cumsum to get trajectories
        # - This reduces parameter count from max_seq*n_artists separate sites to ONE site
        if max_seq > 1:
            # Sample random walk innovations as a single tensor
            # Shape: (n_artists, max_seq - 1) representing innovations for each artist over time
            rw_innovations = numpyro.sample(
                f"{prefix}rw_innovations",
                dist.Normal(0, sigma_rw).expand([n_artists, max_seq - 1]).to_event(2),
            )
            # Cumulative sum to get random walk trajectory from innovations
            # Shape: (n_artists, max_seq - 1)
            rw_trajectory = jnp.cumsum(rw_innovations, axis=1)
            # Full artist effects: init_effect + trajectory
            # Shape: (max_seq, n_artists) - transpose for [time, artist] indexing
            artist_effects = jnp.vstack([
                init_artist_effect[None, :],  # Shape: (1, n_artists)
                (init_artist_effect[None, :] + rw_trajectory.T)  # Shape: (max_seq-1, n_artists)
            ])
        else:
            # Only one time step, no random walk needed
            artist_effects = init_artist_effect[None, :]

        # Index artist effects by album sequence and artist
        # album_seq is 1-indexed, convert to 0-indexed
        seq_idx = jnp.clip(album_seq - 1, 0, max_seq - 1).astype(jnp.int32)
        obs_artist_effect = artist_effects[seq_idx, artist_idx]

        # === Fixed effects for covariates ===
        beta = numpyro.sample(
            f"{prefix}beta",
            dist.Normal(priors.beta_loc, priors.beta_scale).expand([n_features]).to_event(1),
        )

        # === AR(1) term for album-to-album dependency ===
        # prev_score should be 0 for debuts (handled by feature pipeline)
        ar_term = rho * prev_score

        # === Mean prediction ===
        mu = obs_artist_effect + X @ beta + ar_term

        # === Observation-level noise ===
        sigma_obs = numpyro.sample(
            f"{prefix}sigma_obs",
            dist.HalfNormal(priors.sigma_obs_scale),
        )

        # === Exponent for heteroscedastic noise (fixed or learned) ===
        if learn_n_exponent:
            if n_exponent_prior == "logit-normal":
                # Logit-normal: sample in unbounded space, transform via sigmoid
                # This avoids funnel geometry that causes divergences with Beta prior
                n_exp = numpyro.sample(
                    f"{prefix}n_exponent",
                    dist.TransformedDistribution(
                        dist.Normal(priors.n_exponent_loc, priors.n_exponent_scale),
                        [dist.transforms.SigmoidTransform()]
                    )
                )
            else:  # beta (legacy)
                n_exp = numpyro.sample(
                    f"{prefix}n_exponent",
                    dist.Beta(priors.n_exponent_alpha, priors.n_exponent_beta),
                )
        else:
            n_exp = n_exponent  # Use fixed value from config

        # === Per-observation noise scaling ===
        # Note: When learn_n_exponent=True, n_exp is a traced JAX value and cannot
        # be used in Python conditionals. We use the Python-level learn_n_exponent
        # flag to determine if we should apply heteroscedastic scaling.
        # When learning, we always apply scaling (that's why we're learning it).
        # When fixed, we can check if n_exp != 0 to skip unnecessary computation.

        # Validate: if heteroscedastic mode is requested, n_reviews must be provided
        heteroscedastic_requested = learn_n_exponent or n_exponent != 0
        if heteroscedastic_requested and n_reviews is None:
            raise ValueError(
                f"Heteroscedastic noise scaling requires n_reviews data. "
                f"Got learn_n_exponent={learn_n_exponent}, n_exponent={n_exponent}, "
                f"but n_reviews is None. Either provide n_reviews or set both "
                f"learn_n_exponent=False and n_exponent=0 for homoscedastic mode."
            )
        use_heteroscedastic = heteroscedastic_requested  # n_reviews guaranteed non-None here
        if use_heteroscedastic:
            sigma_scaled = compute_sigma_scaled(sigma_obs, n_reviews, n_exp)
        else:
            # Homoscedastic mode: use scalar sigma_obs for all observations
            sigma_scaled = sigma_obs

        # === Likelihood ===
        with numpyro.plate(f"{prefix}obs", len(artist_idx)):
            numpyro.sample(f"{prefix}y", dist.Normal(mu, sigma_scaled), obs=y)

    # Apply non-centered reparameterization to init_artist_effect
    reparam_config = {
        f"{prefix}init_artist_effect": LocScaleReparam(centered=0),
    }
    reparameterized_model = reparam(_score_model_centered, config=reparam_config)

    # Add docstring to reparameterized model
    reparameterized_model.__doc__ = f"""Non-centered hierarchical model for {score_type} scores.

This model includes:
- Time-varying artist effects via random walk (career trajectory)
- AR(1) structure for album-to-album score dependencies
- Hierarchical partial pooling of artist effects
- Non-centered parameterization via LocScaleReparam
- Optional heteroscedastic observation noise (per-observation sigma)

All sample site names are prefixed with "{prefix}" (e.g., "{prefix}beta", "{prefix}rho").

Args:
    artist_idx: Integer array of shape (n_obs,) mapping each observation
        to an artist index in [0, n_artists).
    album_seq: Integer array of shape (n_obs,) with album sequence numbers
        (1 = first album, 2 = second, etc.). Used for time-varying effects.
    prev_score: Float array of shape (n_obs,) with previous album scores.
        Should be 0.0 for debut albums.
    X: Feature matrix of shape (n_obs, n_features) containing covariates.
        Features should be standardized for the default priors to be appropriate.
    y: Optional target scores of shape (n_obs,). Pass None for prior
        predictive sampling or posterior predictive on new data.
    n_artists: Number of unique artists. Required.
    max_seq: Maximum album sequence number. Required for JAX tracing.
        Compute as int(album_seq.max()) before calling the model.
    priors: Prior configuration. If None, uses get_default_priors().
    n_reviews: Optional array of shape (n_obs,) with per-observation
        review counts. Used for heteroscedastic noise scaling. If None,
        homoscedastic noise is used (scalar sigma_obs for all observations).
    n_exponent: Fixed exponent for heteroscedastic noise scaling.
        Default 0.0 gives homoscedastic (constant) noise.
    learn_n_exponent: If True, sample the exponent from a prior distribution
        instead of using the fixed n_exponent value.
    n_exponent_prior: Prior type for learned n_exponent. Options:
        - "logit-normal" (default): Uses TransformedDistribution with
          Normal(loc, scale) and SigmoidTransform. Recommended to avoid
          funnel geometry that causes divergences.
        - "beta" (legacy): Uses Beta(alpha, beta) distribution. May cause
          divergences due to challenging posterior geometry.

Returns:
    None. Model samples are tracked by NumPyro internally.

Sample sites (all prefixed with "{prefix}"):
    - {prefix}mu_artist: Population mean of artist effects
    - {prefix}sigma_artist: Between-artist standard deviation
    - {prefix}sigma_rw: Random walk innovation scale
    - {prefix}rho: AR(1) coefficient
    - {prefix}init_artist_effect: Initial artist effects (partial pooling)
    - {prefix}rw_innovations: Random walk innovations tensor (n_artists x max_seq-1)
    - {prefix}beta: Fixed effect coefficients
    - {prefix}sigma_obs: Observation noise (base scale for heteroscedastic)
    - {prefix}n_exponent: Heteroscedastic scaling exponent (only when learn_n_exponent=True)
    - {prefix}y: Observed/predicted scores

Example:
    >>> import jax.numpy as jnp
    >>> from jax import random
    >>> from numpyro.infer import MCMC, NUTS
    >>> from aoty_pred.models.bayes.model import {score_type}_score_model
    >>>
    >>> # Prepare data
    >>> n_obs, n_features, n_artists = 100, 5, 20
    >>> artist_idx = jnp.array([i % n_artists for i in range(n_obs)])
    >>> album_seq = jnp.array([(i // n_artists) + 1 for i in range(n_obs)])
    >>> max_seq = int(album_seq.max())  # Compute before tracing
    >>> prev_score = jnp.zeros(n_obs)  # 0 for debuts
    >>> X = random.normal(random.PRNGKey(0), (n_obs, n_features))
    >>> y = random.normal(random.PRNGKey(1), (n_obs,)) * 10 + 70
    >>>
    >>> # Run MCMC
    >>> mcmc = MCMC(NUTS({score_type}_score_model), num_warmup=100, num_samples=100)
    >>> mcmc.run(
    ...     random.PRNGKey(2),
    ...     artist_idx=artist_idx,
    ...     album_seq=album_seq,
    ...     prev_score=prev_score,
    ...     X=X,
    ...     y=y,
    ...     n_artists=n_artists,
    ...     max_seq=max_seq
    ... )
    >>> samples = mcmc.get_samples()
    >>> print("{prefix}rho" in samples)
    True
"""

    return reparameterized_model


# Create the two exported model functions
user_score_model = make_score_model("user")
critic_score_model = make_score_model("critic")

# Backwards compatibility alias
album_score_model = user_score_model
