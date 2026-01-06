"""Bayesian hierarchical model for album score prediction.

This module defines the core NumPyro model with:
- Hierarchical artist effects with partial pooling
- Non-centered parameterization to avoid funnel geometry
- Fixed effects for covariates (genre PCA, release year, etc.)

The model structure:
    y_ij ~ Normal(mu_ij, sigma_obs)
    mu_ij = artist_effect_j + X_ij @ beta

    artist_effect_j ~ Normal(mu_artist, sigma_artist)  # partial pooling
    mu_artist ~ Normal(mu_artist_loc, mu_artist_scale)  # hyperprior
    sigma_artist ~ HalfNormal(sigma_artist_scale)       # hyperprior

    beta ~ Normal(beta_loc, beta_scale)  # fixed effects
    sigma_obs ~ HalfNormal(sigma_obs_scale)  # observation noise

Non-centered parameterization is applied via LocScaleReparam to transform
the artist_effect sampling from:
    artist_effect ~ Normal(mu_artist, sigma_artist)
to:
    artist_effect_decentered ~ Normal(0, 1)
    artist_effect = mu_artist + sigma_artist * artist_effect_decentered

This avoids the "funnel" geometry that makes NUTS sampling inefficient
when sigma_artist is small.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

from aoty_pred.models.bayes.priors import PriorConfig, get_default_priors


def _album_score_model_centered(
    artist_idx: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray | None = None,
    n_artists: int | None = None,
    priors: PriorConfig | None = None,
) -> None:
    """Centered parameterization of album score model (internal).

    This function defines the model in centered form. Use album_score_model
    (the reparameterized version) for actual inference.

    Args:
        artist_idx: Integer array of shape (n_obs,) mapping each observation
            to an artist index in [0, n_artists).
        X: Feature matrix of shape (n_obs, n_features) containing covariates
            such as genre PCA components, release year, etc.
        y: Optional target scores of shape (n_obs,). Pass None for prior
            predictive sampling or posterior predictive on new data.
        n_artists: Number of unique artists. Must be provided.
        priors: Prior configuration. If None, uses get_default_priors().

    Model structure:
        - Population-level hyperpriors for artist effect distribution
        - Artist-level effects sampled from population (partial pooling)
        - Fixed effects for covariates
        - Observation-level noise
    """
    # Get prior configuration
    if priors is None:
        priors = get_default_priors()

    if n_artists is None:
        raise ValueError("n_artists must be provided")

    n_obs = artist_idx.shape[0]
    n_features = X.shape[1]

    # === Population-level hyperpriors ===
    # Mean of artist quality distribution
    mu_artist = numpyro.sample(
        "mu_artist",
        dist.Normal(priors.mu_artist_loc, priors.mu_artist_scale),
    )

    # Between-artist standard deviation (controls pooling strength)
    sigma_artist = numpyro.sample(
        "sigma_artist",
        dist.HalfNormal(priors.sigma_artist_scale),
    )

    # === Artist-level effects (partial pooling) ===
    with numpyro.plate("artists", n_artists):
        artist_effect = numpyro.sample(
            "artist_effect",
            dist.Normal(mu_artist, sigma_artist),
        )

    # Map observations to their artist's effect
    obs_artist_effect = artist_effect[artist_idx]

    # === Fixed effects for covariates ===
    beta = numpyro.sample(
        "beta",
        dist.Normal(priors.beta_loc, priors.beta_scale).expand([n_features]).to_event(1),
    )

    # === Observation-level noise ===
    sigma_obs = numpyro.sample(
        "sigma_obs",
        dist.HalfNormal(priors.sigma_obs_scale),
    )

    # === Mean prediction ===
    # Artist effect + covariate effects
    mu = obs_artist_effect + X @ beta

    # === Likelihood ===
    with numpyro.plate("obs", n_obs):
        numpyro.sample("y", dist.Normal(mu, sigma_obs), obs=y)


# Apply non-centered reparameterization to artist_effect
# This transforms: artist_effect ~ Normal(mu_artist, sigma_artist)
# To: artist_effect_decentered ~ Normal(0, 1)
#     artist_effect = mu_artist + sigma_artist * artist_effect_decentered
#
# The non-centered form avoids "funnel" geometry in the posterior that
# causes NUTS to have many divergences when sigma_artist is small.
album_score_model = reparam(
    _album_score_model_centered,
    config={"artist_effect": LocScaleReparam(centered=0)},
)
album_score_model.__doc__ = """Non-centered hierarchical model for album scores.

This is the main model function for inference. It applies LocScaleReparam
to the artist_effect parameter for efficient NUTS sampling.

Args:
    artist_idx: Integer array of shape (n_obs,) mapping each observation
        to an artist index in [0, n_artists).
    X: Feature matrix of shape (n_obs, n_features) containing covariates.
        Features should be standardized (zero mean, unit variance) for
        the default priors to be appropriate.
    y: Optional target scores of shape (n_obs,). Pass None for prior
        predictive sampling or posterior predictive on new data.
    n_artists: Number of unique artists. Required.
    priors: Prior configuration. If None, uses get_default_priors().

Returns:
    None. Model samples are tracked by NumPyro internally.

Example:
    >>> import jax.numpy as jnp
    >>> from jax import random
    >>> from numpyro.infer import MCMC, NUTS
    >>> from aoty_pred.models.bayes.model import album_score_model
    >>>
    >>> # Prepare data
    >>> n_obs, n_features, n_artists = 100, 5, 20
    >>> artist_idx = jnp.array([i % n_artists for i in range(n_obs)])
    >>> X = random.normal(random.PRNGKey(0), (n_obs, n_features))
    >>> y = random.normal(random.PRNGKey(1), (n_obs,)) * 10 + 70
    >>>
    >>> # Run MCMC
    >>> mcmc = MCMC(NUTS(album_score_model), num_warmup=100, num_samples=100)
    >>> mcmc.run(random.PRNGKey(2), artist_idx=artist_idx, X=X, y=y, n_artists=n_artists)
    >>> samples = mcmc.get_samples()

Note:
    The samples will contain 'artist_effect_decentered' (the raw samples)
    and may be transformed back via:
        artist_effect = mu_artist + sigma_artist * artist_effect_decentered
"""
