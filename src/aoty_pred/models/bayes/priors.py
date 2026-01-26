"""Prior configuration for Bayesian hierarchical model.

This module defines the hyperparameters for the album score model priors.
All priors are configurable to support sensitivity analysis.

Prior Roles:
- mu_artist: Population mean of artist quality (centering artist effects)
- sigma_artist: Between-artist variance controlling partial pooling strength
  - Large sigma_artist -> less pooling, artists estimated independently
  - Small sigma_artist -> more pooling, artists shrunk toward population mean
- sigma_rw: Innovation scale for time-varying artist effects (random walk)
  - Controls how much an artist's quality changes between albums
  - Smaller values -> smoother career trajectories
- rho: AR(1) coefficient for album-to-album score dependency
  - Captures momentum: positive rho -> hot streaks, negative -> regression to mean
- beta: Fixed effect coefficients for covariates (genre PCA, release year, etc.)
- sigma_obs: Observation-level noise (unexplained variance per album)
- n_exponent: Scaling exponent for heteroscedastic observation noise
  - sigma_scaled = sigma_obs / n_reviews^exponent
  - Higher exponent -> more noise reduction for albums with many reviews
  - exponent=0 -> homoscedastic (constant noise)
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
        sigma_rw_scale: Scale for HalfNormal prior on random walk innovation.
            Default 0.1 produces smooth career trajectories. Controls how much
            an artist's quality can change between consecutive albums.
            Smaller values -> more stable artist effects over time.
        rho_loc: Location (mean) for AR(1) coefficient prior.
            Default 0.0 centers the autoregressive coefficient at zero,
            expressing no prior belief about momentum direction.
        rho_scale: Scale for AR(1) coefficient prior.
            Default 0.3 allows moderate uncertainty while keeping most prior
            mass on reasonable AR coefficients (roughly -0.6 to 0.6).
        beta_loc: Location for fixed effect coefficient priors.
            Default 0.0 centers effects at zero (no effect assumption).
        beta_scale: Scale for fixed effect coefficient priors.
            Default 1.0 is weakly informative for standardized features.
        sigma_obs_scale: Scale for HalfNormal prior on observation noise.
            Default 1.0 allows moderate observation-level variance.
        n_exponent_alpha: Alpha (concentration1) parameter for Beta prior on
            learned n_exponent. Default 2.0. (Legacy - use logit-normal instead)
        n_exponent_beta: Beta (concentration0) parameter for Beta prior on
            learned n_exponent. Default 4.0. (Legacy - use logit-normal instead)
            Note: Beta(2, 4) has mode at 0.25 and mean at 0.33, centering
            prior mass on cube-root-like scaling for heteroscedastic noise.
        n_exponent_loc: Location parameter for logit-normal prior on n_exponent.
            Default 0.0 maps to mode of 0.5 via sigmoid transform. This is the
            recommended prior type as it avoids funnel geometry issues.
        n_exponent_scale: Scale parameter for logit-normal prior on n_exponent.
            Default 1.0 gives reasonable spread in [0,1] after sigmoid transform.
    """

    mu_artist_loc: float = 0.0
    mu_artist_scale: float = 1.0
    sigma_artist_scale: float = 0.5
    sigma_rw_scale: float = 0.1
    rho_loc: float = 0.0
    rho_scale: float = 0.3
    beta_loc: float = 0.0
    beta_scale: float = 1.0
    sigma_obs_scale: float = 1.0
    n_exponent_alpha: float = 2.0
    n_exponent_beta: float = 4.0
    # Logit-normal prior parameters for n_exponent (new default)
    n_exponent_loc: float = 0.0  # maps to ~0.5 via sigmoid
    n_exponent_scale: float = 1.0  # reasonable spread in logit space


def get_default_priors() -> PriorConfig:
    """Return default prior configuration.

    The defaults are designed to be weakly informative:
    - Artist effects centered at 0 with moderate pooling (sigma_artist_scale=0.5)
    - Time-varying effects with small innovation (sigma_rw_scale=0.1 for smooth careers)
    - AR(1) coefficient centered at 0 with moderate uncertainty (rho_scale=0.3)
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
        >>> priors.sigma_rw_scale
        0.1
        >>> priors.rho_loc
        0.0
    """
    return PriorConfig()
