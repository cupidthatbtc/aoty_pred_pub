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
from numpyro.contrib.control_flow import scan
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

from aoty_pred.models.bayes.priors import PriorConfig, get_default_priors

__all__ = [
    "make_score_model",
    "user_score_model",
    "critic_score_model",
    "album_score_model",
]


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

        Model structure:
            - Population-level hyperpriors for artist effect distribution
            - Time-varying artist effects via random walk
            - AR(1) term for album-to-album dependency
            - Fixed effects for covariates
            - Observation-level noise
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
        def rw_step(carry, t):
            """Single step of random walk for artist effects."""
            prev_effects = carry
            innovation = numpyro.sample(
                f"{prefix}rw_innov_{t}",
                dist.Normal(0, sigma_rw).expand([n_artists]).to_event(1),
            )
            curr_effects = prev_effects + innovation
            return curr_effects, curr_effects

        # Run random walk for (max_seq - 1) steps starting from initial effects
        if max_seq > 1:
            _, effects_over_time = scan(
                rw_step, init_artist_effect, jnp.arange(max_seq - 1)
            )
            # Stack: shape (max_seq, n_artists)
            # First row is initial effects, subsequent rows are random walk
            artist_effects = jnp.vstack([init_artist_effect[None, :], effects_over_time])
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

        # === Likelihood ===
        with numpyro.plate(f"{prefix}obs", len(artist_idx)):
            numpyro.sample(f"{prefix}y", dist.Normal(mu, sigma_obs), obs=y)

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

Returns:
    None. Model samples are tracked by NumPyro internally.

Sample sites (all prefixed with "{prefix}"):
    - {prefix}mu_artist: Population mean of artist effects
    - {prefix}sigma_artist: Between-artist standard deviation
    - {prefix}sigma_rw: Random walk innovation scale
    - {prefix}rho: AR(1) coefficient
    - {prefix}init_artist_effect: Initial artist effects (partial pooling)
    - {prefix}rw_innov_{{t}}: Random walk innovations at time t
    - {prefix}beta: Fixed effect coefficients
    - {prefix}sigma_obs: Observation noise
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
