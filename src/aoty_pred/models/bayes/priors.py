"""Prior configuration for Bayesian hierarchical model.

This module defines the hyperparameters for the album score model priors.
All priors are configurable to support sensitivity analysis.

Prior Roles:
- mu_artist: Population mean of artist quality (centering artist effects)
- sigma_artist: Between-artist variance controlling partial pooling strength
  - Large sigma_artist -> less pooling, artists estimated independently
  - Small sigma_artist -> more pooling, artists shrunk toward population mean
- beta: Fixed effect coefficients for covariates (genre PCA, release year, etc.)
- sigma_obs: Observation-level noise (unexplained variance per album)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PriorConfig:
    """Hyperparameter configuration for album score model priors.

    All parameters are frozen to ensure immutability during model fitting.

    Attributes:
        mu_artist_loc: Location (mean) for artist effect population mean prior.
            Default 0.0 centers artist effects around zero (deviations from baseline).
        mu_artist_scale: Scale (std) for artist effect population mean prior.
            Default 1.0 allows moderate uncertainty in population center.
        sigma_artist_scale: Scale for HalfNormal prior on artist effect dispersion.
            Default 0.5 encourages moderate pooling. Lower values -> more pooling.
        beta_loc: Location for fixed effect coefficient priors.
            Default 0.0 centers effects at zero (no effect assumption).
        beta_scale: Scale for fixed effect coefficient priors.
            Default 1.0 is weakly informative for standardized features.
        sigma_obs_scale: Scale for HalfNormal prior on observation noise.
            Default 1.0 allows moderate observation-level variance.
    """

    mu_artist_loc: float = 0.0
    mu_artist_scale: float = 1.0
    sigma_artist_scale: float = 0.5
    beta_loc: float = 0.0
    beta_scale: float = 1.0
    sigma_obs_scale: float = 1.0


def get_default_priors() -> PriorConfig:
    """Return default prior configuration.

    The defaults are designed to be weakly informative:
    - Artist effects centered at 0 with moderate pooling (sigma_artist_scale=0.5)
    - Fixed effects centered at 0 with unit scale (appropriate for standardized features)
    - Observation noise with unit scale HalfNormal

    Returns:
        PriorConfig with sensible default hyperparameters.

    Example:
        >>> priors = get_default_priors()
        >>> priors.mu_artist_loc
        0.0
        >>> priors.sigma_artist_scale
        0.5
    """
    return PriorConfig()
