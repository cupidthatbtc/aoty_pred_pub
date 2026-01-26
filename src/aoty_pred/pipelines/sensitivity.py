"""Sensitivity analysis pipeline for robustness assessment.

This module provides orchestration for sensitivity analyses required for publication:
- SENS-01: Prior sensitivity (diffuse, default, informative priors)
- SENS-02: Threshold sensitivity (min-ratings 5, 10, 25)
- SENS-03: Feature ablation (remove feature groups to measure importance)
- SENS-04: Coefficient stability across analyses

Key outputs:
- SensitivityResult dataclass for each analysis variant
- Aggregation functions for comparison DataFrames
- Coefficient extraction for forest plot visualization

Usage:
    >>> from aoty_pred.pipelines.sensitivity import run_prior_sensitivity, PRIOR_CONFIGS
    >>> results = run_prior_sensitivity(model, model_args, mcmc_config)
    >>> comparison = aggregate_sensitivity_results(results, metric="elpd")
"""

import logging
from dataclasses import dataclass, field
from typing import Callable

import arviz as az
import numpy as np
import pandas as pd

from aoty_pred.evaluation.cv import (
    LOOResult,
    add_log_likelihood_to_idata,
    compute_log_likelihood,
    compute_loo,
)
from aoty_pred.evaluation.metrics import CRPSResult
from aoty_pred.models.bayes.diagnostics import ConvergenceDiagnostics, check_convergence
from aoty_pred.models.bayes.fit import MCMCConfig, fit_model
from aoty_pred.models.bayes.priors import PriorConfig, get_default_priors

__all__ = [
    "SensitivityResult",
    "PRIOR_CONFIGS",
    "run_prior_sensitivity",
    "run_threshold_sensitivity",
    "run_feature_ablation",
    "aggregate_sensitivity_results",
    "create_coefficient_comparison_df",
    "extract_coefficient_summary",
]

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Container for sensitivity analysis results.

    Each sensitivity analysis variant produces one SensitivityResult containing
    the configuration used, diagnostics, evaluation metrics, and coefficient
    estimates for comparison.

    Attributes
    ----------
    name : str
        Descriptive name for this variant (e.g., "diffuse_priors", "threshold_10", "no_genre").
    config : dict
        Configuration used for this variant. Contains priors, threshold, or feature mask
        depending on the analysis type.
    idata : az.InferenceData | None
        Fitted model's InferenceData. May be None if memory conservation is needed
        (e.g., after extracting coefficients and metrics).
    convergence : ConvergenceDiagnostics | None
        Convergence diagnostics (R-hat, ESS, divergences). None if not computed.
    loo : LOOResult | None
        LOO-CV results with ELPD and Pareto-k diagnostics. None if not computed.
    crps : CRPSResult | None
        CRPS (probabilistic prediction quality) result. None if not computed.
    coefficients : pd.DataFrame
        Posterior summary for key parameters. Contains columns:
        mean, sd, hdi_3%, hdi_97% (or similar HDI bounds).

    Example
    -------
    >>> result = SensitivityResult(
    ...     name="diffuse_priors",
    ...     config={"priors": {"mu_artist_scale": 5.0}},
    ...     idata=idata,
    ...     convergence=check_convergence(idata),
    ...     loo=None,
    ...     crps=None,
    ...     coefficients=extract_coefficient_summary(idata, ["user_beta", "user_rho"]),
    ... )
    """

    name: str
    config: dict
    idata: az.InferenceData | None = None
    convergence: ConvergenceDiagnostics | None = None
    loo: LOOResult | None = None
    crps: CRPSResult | None = None
    coefficients: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())


# Prior configurations for sensitivity analysis (SENS-01)
# These span from highly regularized (informative) to minimally regularized (diffuse)
PRIOR_CONFIGS: dict[str, PriorConfig] = {
    "default": get_default_priors(),
    "diffuse": PriorConfig(
        mu_artist_scale=5.0,  # Much wider (default: 1.0)
        sigma_artist_scale=2.0,  # Allow more variance (default: 0.5)
        sigma_rw_scale=0.5,  # More flexible time-varying (default: 0.1)
        rho_scale=0.5,  # Wider AR(1) prior (default: 0.3)
        beta_scale=5.0,  # Weaker regularization (default: 1.0)
        sigma_obs_scale=2.0,  # Wider observation noise (default: 1.0)
    ),
    "informative": PriorConfig(
        mu_artist_scale=0.5,  # Tighter (default: 1.0)
        sigma_artist_scale=0.25,  # Encourage pooling (default: 0.5)
        sigma_rw_scale=0.05,  # Smoother careers (default: 0.1)
        rho_scale=0.2,  # Tighter AR(1) (default: 0.3)
        beta_scale=0.5,  # Stronger regularization (default: 1.0)
        sigma_obs_scale=0.5,  # Tighter observation noise (default: 1.0)
    ),
}


def extract_coefficient_summary(
    idata: az.InferenceData,
    var_names: list[str] | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Extract posterior summary for specified parameters.

    Uses ArviZ summary to compute mean, standard deviation, and HDI bounds
    for specified parameters. Useful for comparing coefficient estimates
    across sensitivity analyses.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData with posterior samples.
    var_names : list[str] | None, optional
        Parameter names to summarize. If None, summarizes all posterior variables.
        Can include prefixed names (e.g., "user_beta", "user_rho").
    prefix : str, optional
        Optional prefix to add to all var_names. Useful when you want to
        specify generic names and add model-specific prefixes.

    Returns
    -------
    pd.DataFrame
        Summary with columns: mean, sd, hdi_3%, hdi_97% (default HDI bounds).
        Index is parameter names (potentially multi-indexed for arrays).

    Example
    -------
    >>> summary = extract_coefficient_summary(idata, ["user_beta", "user_rho"])
    >>> print(summary[["mean", "hdi_3%", "hdi_97%"]])
    """
    if prefix and var_names:
        var_names = [f"{prefix}{name}" for name in var_names]

    # Use ArviZ summary with stats and diagnostics
    # hdi_prob=0.94 gives 3% and 97% bounds
    try:
        summary = az.summary(
            idata,
            var_names=var_names,
            kind="stats",
            hdi_prob=0.94,
        )
    except KeyError as e:
        # Handle case where var_names don't exist
        logger.warning(f"Some variables not found in posterior: {e}")
        if var_names is None:
            raise
        # Try each variable individually
        valid_vars = []
        for var in var_names:
            try:
                az.summary(idata, var_names=[var], kind="stats")
                valid_vars.append(var)
            except KeyError:
                pass
        if not valid_vars:
            return pd.DataFrame()
        summary = az.summary(idata, var_names=valid_vars, kind="stats", hdi_prob=0.94)

    return summary


def run_prior_sensitivity(
    model: Callable,
    model_args: dict,
    mcmc_config: MCMCConfig | None = None,
    configs: dict[str, PriorConfig] | None = None,
    compute_loo_cv: bool = True,
    obs_name: str = "user_y",
    coefficient_vars: list[str] | None = None,
) -> dict[str, SensitivityResult]:
    """Run prior sensitivity analysis (SENS-01).

    Fits the model with multiple prior configurations to assess sensitivity
    of inference to prior choices. For publication, results should show
    that conclusions are robust across reasonable prior specifications.

    Parameters
    ----------
    model : Callable
        NumPyro model function (e.g., user_score_model).
    model_args : dict
        Arguments to pass to the model. Must include all required arrays
        (artist_idx, album_seq, prev_score, X, y, n_artists, max_seq).
    mcmc_config : MCMCConfig | None, optional
        MCMC configuration. If None, uses default MCMCConfig().
    configs : dict[str, PriorConfig] | None, optional
        Dictionary mapping config names to PriorConfig objects.
        If None, uses PRIOR_CONFIGS (default, diffuse, informative).
    compute_loo_cv : bool, default True
        Whether to compute LOO-CV for each variant. Set to False for
        faster execution when only comparing coefficients.
    obs_name : str, default "user_y"
        Name of the observed variable site in the model. Use "user_y"
        for user_score_model, "critic_y" for critic_score_model.
    coefficient_vars : list[str] | None, optional
        Parameter names to extract for coefficient comparison.
        If None, extracts all posterior variables.

    Returns
    -------
    dict[str, SensitivityResult]
        Mapping from config name to SensitivityResult.

    Example
    -------
    >>> from aoty_pred.models.bayes import user_score_model
    >>> results = run_prior_sensitivity(user_score_model, model_args)
    >>> for name, result in results.items():
    ...     print(f"{name}: ELPD={result.loo.elpd_loo:.1f}")

    Notes
    -----
    Each model fit is logged with progress information. For large datasets,
    consider using reduced MCMC iterations for initial sensitivity checks.
    """
    if configs is None:
        configs = PRIOR_CONFIGS

    if mcmc_config is None:
        mcmc_config = MCMCConfig()

    results = {}

    for name, prior_config in configs.items():
        logger.info(f"Prior sensitivity: fitting '{name}' configuration")

        # Add priors to model args
        args_with_priors = {**model_args, "priors": prior_config}

        # Fit model
        fit_result = fit_model(model, args_with_priors, config=mcmc_config, progress_bar=True)

        # Check convergence (allow divergences for sensitivity analysis)
        convergence = check_convergence(fit_result.idata, allow_divergences=True)

        # Extract coefficient summary
        coefficients = extract_coefficient_summary(fit_result.idata, var_names=coefficient_vars)

        # Optionally compute LOO-CV
        loo_result = None
        if compute_loo_cv:
            try:
                log_lik = compute_log_likelihood(
                    model, fit_result.mcmc, model_args, obs_name=obs_name
                )
                idata_with_ll = add_log_likelihood_to_idata(fit_result.idata, log_lik)
                loo_result = compute_loo(idata_with_ll)
            except Exception as e:
                logger.warning(f"LOO computation failed for '{name}': {e}")

        # Store result
        results[name] = SensitivityResult(
            name=name,
            config={"priors": prior_config.__dict__},
            idata=fit_result.idata,
            convergence=convergence,
            loo=loo_result,
            crps=None,  # CRPS requires posterior predictive, compute separately if needed
            coefficients=coefficients,
        )

        # Log summary
        status = "PASSED" if convergence.passed else "FAILED"
        elpd_str = f", ELPD={loo_result.elpd_loo:.1f}" if loo_result else ""
        logger.info(
            f"  '{name}': convergence {status}, "
            f"R-hat max={convergence.rhat_max:.4f}, "
            f"divergences={convergence.divergences}{elpd_str}"
        )

    return results


def run_threshold_sensitivity(
    model: Callable,
    data_loader: Callable[[int], tuple[pd.DataFrame, dict]],
    thresholds: tuple[int, ...] = (5, 10, 25),
    mcmc_config: MCMCConfig | None = None,
    compute_loo_cv: bool = True,
    obs_name: str = "user_y",
    coefficient_vars: list[str] | None = None,
) -> dict[int, SensitivityResult]:
    """Run threshold sensitivity analysis (SENS-02).

    Tests model inference across different minimum ratings thresholds.
    Each threshold produces a different subset of the data (e.g., threshold=10
    requires albums to have at least 10 user ratings).

    Parameters
    ----------
    model : Callable
        NumPyro model function (e.g., user_score_model).
    data_loader : Callable[[int], tuple[pd.DataFrame, dict]]
        Function that takes a threshold integer and returns:
        - DataFrame with the filtered data
        - dict of model_args ready for fitting (artist_idx, X, y, etc.)
        This abstraction allows flexibility in data loading.
    thresholds : tuple of int, default (5, 10, 25)
        Minimum ratings thresholds to test.
    mcmc_config : MCMCConfig | None, optional
        MCMC configuration. If None, uses default MCMCConfig().
    compute_loo_cv : bool, default True
        Whether to compute LOO-CV for each variant.
    obs_name : str, default "user_y"
        Name of the observed variable site in the model.
    coefficient_vars : list[str] | None, optional
        Parameter names to extract for coefficient comparison.

    Returns
    -------
    dict[int, SensitivityResult]
        Mapping from threshold to SensitivityResult.

    Example
    -------
    >>> def load_threshold_data(threshold):
    ...     df = pd.read_parquet(f"data/processed/user_score_minratings_{threshold}.parquet")
    ...     model_args = prepare_model_args(df)  # User-defined function
    ...     return df, model_args
    >>> results = run_threshold_sensitivity(user_score_model, load_threshold_data)

    Notes
    -----
    Threshold sensitivity demonstrates that conclusions hold across different
    data quality filters. Higher thresholds have fewer albums but more reliable
    scores; lower thresholds have more albums but potentially noisier scores.
    """
    if mcmc_config is None:
        mcmc_config = MCMCConfig()

    results = {}

    for threshold in thresholds:
        logger.info(f"Threshold sensitivity: fitting threshold={threshold}")

        # Load data for this threshold
        df, model_args = data_loader(threshold)
        n_obs = len(df) if hasattr(df, "__len__") else model_args.get("y", []).shape[0]
        logger.info(f"  Loaded {n_obs} observations for threshold={threshold}")

        # Fit model
        fit_result = fit_model(model, model_args, config=mcmc_config, progress_bar=True)

        # Check convergence
        convergence = check_convergence(fit_result.idata, allow_divergences=True)

        # Extract coefficient summary
        coefficients = extract_coefficient_summary(fit_result.idata, var_names=coefficient_vars)

        # Optionally compute LOO-CV
        loo_result = None
        if compute_loo_cv:
            try:
                log_lik = compute_log_likelihood(
                    model, fit_result.mcmc, model_args, obs_name=obs_name
                )
                idata_with_ll = add_log_likelihood_to_idata(fit_result.idata, log_lik)
                loo_result = compute_loo(idata_with_ll)
            except Exception as e:
                logger.warning(f"LOO computation failed for threshold={threshold}: {e}")

        # Store result
        results[threshold] = SensitivityResult(
            name=f"threshold_{threshold}",
            config={"threshold": threshold, "n_obs": n_obs},
            idata=fit_result.idata,
            convergence=convergence,
            loo=loo_result,
            crps=None,
            coefficients=coefficients,
        )

        # Log summary
        status = "PASSED" if convergence.passed else "FAILED"
        elpd_str = f", ELPD={loo_result.elpd_loo:.1f}" if loo_result else ""
        logger.info(
            f"  threshold={threshold}: convergence {status}, "
            f"R-hat max={convergence.rhat_max:.4f}, "
            f"divergences={convergence.divergences}{elpd_str}"
        )

    return results


def run_feature_ablation(
    model: Callable,
    model_args: dict,
    feature_groups: dict[str, list[int]],
    mcmc_config: MCMCConfig | None = None,
    compute_loo_cv: bool = True,
    obs_name: str = "user_y",
    coefficient_vars: list[str] | None = None,
) -> dict[str, SensitivityResult]:
    """Run feature ablation study (SENS-03).

    Measures the importance of each feature group by zeroing out those
    features and measuring the impact on model performance. Includes
    a "full" baseline with all features.

    Parameters
    ----------
    model : Callable
        NumPyro model function (e.g., user_score_model).
    model_args : dict
        Arguments to pass to the model. Must include "X" (feature matrix).
    feature_groups : dict[str, list[int]]
        Mapping from group name to column indices in X to ablate.
        E.g., {"genre": [0,1,2,3,4], "temporal": [5,6,7], "album_type": [8,9,10,11]}
    mcmc_config : MCMCConfig | None, optional
        MCMC configuration. If None, uses default MCMCConfig().
    compute_loo_cv : bool, default True
        Whether to compute LOO-CV for each variant.
    obs_name : str, default "user_y"
        Name of the observed variable site in the model.
    coefficient_vars : list[str] | None, optional
        Parameter names to extract for coefficient comparison.

    Returns
    -------
    dict[str, SensitivityResult]
        Mapping from ablation name to SensitivityResult.
        Includes "full" (baseline) and "no_{group}" for each ablated group.

    Example
    -------
    >>> feature_groups = {
    ...     "genre": [0, 1, 2, 3, 4],  # Genre PCA columns
    ...     "temporal": [5, 6, 7],      # Temporal features
    ...     "album_type": [8, 9, 10],   # Album type one-hot
    ... }
    >>> results = run_feature_ablation(user_score_model, model_args, feature_groups)
    >>> for name, result in results.items():
    ...     print(f"{name}: ELPD={result.loo.elpd_loo:.1f}")

    Notes
    -----
    Feature ablation reveals which feature groups contribute most to
    predictive performance. Larger ELPD drops indicate more important features.

    Features are ablated by setting their values to zero, which assumes
    features are standardized (zero = mean). For non-standardized features,
    consider using the feature mean instead.
    """
    if mcmc_config is None:
        mcmc_config = MCMCConfig()

    # Get original feature matrix
    X_original = model_args["X"]

    results = {}

    # First, fit the full model as baseline
    logger.info("Feature ablation: fitting full model (baseline)")
    fit_result = fit_model(model, model_args, config=mcmc_config, progress_bar=True)
    convergence = check_convergence(fit_result.idata, allow_divergences=True)
    coefficients = extract_coefficient_summary(fit_result.idata, var_names=coefficient_vars)

    loo_result = None
    if compute_loo_cv:
        try:
            log_lik = compute_log_likelihood(model, fit_result.mcmc, model_args, obs_name=obs_name)
            idata_with_ll = add_log_likelihood_to_idata(fit_result.idata, log_lik)
            loo_result = compute_loo(idata_with_ll)
        except Exception as e:
            logger.warning(f"LOO computation failed for full model: {e}")

    results["full"] = SensitivityResult(
        name="full",
        config={"ablated_features": None, "n_features": X_original.shape[1]},
        idata=fit_result.idata,
        convergence=convergence,
        loo=loo_result,
        crps=None,
        coefficients=coefficients,
    )

    status = "PASSED" if convergence.passed else "FAILED"
    elpd_str = f", ELPD={loo_result.elpd_loo:.1f}" if loo_result else ""
    logger.info(
        f"  full: convergence {status}, "
        f"R-hat max={convergence.rhat_max:.4f}, "
        f"divergences={convergence.divergences}{elpd_str}"
    )

    # Now ablate each feature group
    for group_name, col_indices in feature_groups.items():
        ablation_name = f"no_{group_name}"
        logger.info(f"Feature ablation: fitting '{ablation_name}' (removing columns {col_indices})")

        # Create modified feature matrix with ablated columns set to zero
        X_ablated = np.array(X_original, copy=True)
        X_ablated[:, col_indices] = 0.0

        # Create modified model args
        ablated_args = {**model_args, "X": X_ablated}

        # Fit model
        fit_result = fit_model(model, ablated_args, config=mcmc_config, progress_bar=True)
        convergence = check_convergence(fit_result.idata, allow_divergences=True)
        coefficients = extract_coefficient_summary(fit_result.idata, var_names=coefficient_vars)

        # Optionally compute LOO-CV
        loo_result = None
        if compute_loo_cv:
            try:
                log_lik = compute_log_likelihood(
                    model, fit_result.mcmc, ablated_args, obs_name=obs_name
                )
                idata_with_ll = add_log_likelihood_to_idata(fit_result.idata, log_lik)
                loo_result = compute_loo(idata_with_ll)
            except Exception as e:
                logger.warning(f"LOO computation failed for '{ablation_name}': {e}")

        results[ablation_name] = SensitivityResult(
            name=ablation_name,
            config={"ablated_features": group_name, "ablated_columns": col_indices},
            idata=fit_result.idata,
            convergence=convergence,
            loo=loo_result,
            crps=None,
            coefficients=coefficients,
        )

        status = "PASSED" if convergence.passed else "FAILED"
        elpd_str = f", ELPD={loo_result.elpd_loo:.1f}" if loo_result else ""
        logger.info(
            f"  '{ablation_name}': convergence {status}, "
            f"R-hat max={convergence.rhat_max:.4f}, "
            f"divergences={convergence.divergences}{elpd_str}"
        )

    return results


def aggregate_sensitivity_results(
    results: dict[str, SensitivityResult],
    metric: str = "elpd",
) -> pd.DataFrame:
    """Aggregate sensitivity results into a comparison DataFrame.

    Creates a summary table comparing all sensitivity variants on the
    specified metric.

    Parameters
    ----------
    results : dict[str, SensitivityResult]
        Dictionary mapping variant names to SensitivityResult objects.
    metric : str, default "elpd"
        Metric to aggregate. Options:
        - "elpd": ELPD from LOO-CV (higher is better)
        - "crps": Mean CRPS (lower is better)
        - "convergence": Convergence diagnostics summary
        - "coefficients": Coefficient estimates (mean, se)

    Returns
    -------
    pd.DataFrame
        Comparison table with rows for each variant and columns for:
        - name: Variant name
        - metric value(s)
        - convergence_passed: Whether convergence passed
        - divergences: Number of divergent transitions
        - For coefficients: additional columns for each parameter

    Example
    -------
    >>> results = run_prior_sensitivity(model, model_args)
    >>> comparison = aggregate_sensitivity_results(results, metric="elpd")
    >>> print(comparison)
    #                   elpd  elpd_se  convergence_passed  divergences
    # default        -1234.5     45.2               True            0
    # diffuse        -1256.3     48.1               True            3
    # informative    -1240.1     44.8               True            0
    """
    rows = []

    for name, result in results.items():
        row = {"name": name}

        # Add convergence info
        if result.convergence is not None:
            row["convergence_passed"] = result.convergence.passed
            row["divergences"] = result.convergence.divergences
            row["rhat_max"] = result.convergence.rhat_max
            row["ess_bulk_min"] = result.convergence.ess_bulk_min
        else:
            row["convergence_passed"] = None
            row["divergences"] = None
            row["rhat_max"] = None
            row["ess_bulk_min"] = None

        if metric == "elpd":
            if result.loo is not None:
                row["elpd"] = result.loo.elpd_loo
                row["elpd_se"] = result.loo.se_elpd
                row["p_loo"] = result.loo.p_loo
                row["n_high_pareto_k"] = result.loo.n_high_pareto_k
            else:
                row["elpd"] = None
                row["elpd_se"] = None
                row["p_loo"] = None
                row["n_high_pareto_k"] = None

        elif metric == "crps":
            if result.crps is not None:
                row["mean_crps"] = result.crps.mean_crps
                row["n_obs"] = result.crps.n_obs
            else:
                row["mean_crps"] = None
                row["n_obs"] = None

        elif metric == "convergence":
            # Already added convergence info above
            pass

        elif metric == "coefficients":
            # Add coefficient estimates from the summary
            if not result.coefficients.empty:
                for param in result.coefficients.index:
                    row[f"{param}_mean"] = result.coefficients.loc[param, "mean"]
                    if "sd" in result.coefficients.columns:
                        row[f"{param}_sd"] = result.coefficients.loc[param, "sd"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Handle empty results
    if df.empty:
        return df

    df = df.set_index("name")

    # Sort by ELPD (descending) if available
    if metric == "elpd" and "elpd" in df.columns:
        df = df.sort_values("elpd", ascending=False)

    return df


def create_coefficient_comparison_df(
    results: dict[str, SensitivityResult],
    params: list[str],
) -> pd.DataFrame:
    """Create coefficient comparison DataFrame for forest plots.

    Extracts specified parameter estimates from each sensitivity variant
    into a format suitable for forest plot visualization.

    Parameters
    ----------
    results : dict[str, SensitivityResult]
        Dictionary mapping variant names to SensitivityResult objects.
    params : list[str]
        Parameter names to compare (e.g., ["user_beta[0]", "user_rho"]).
        Must match the index in each result's coefficients DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - variant: Name of the sensitivity variant
        - param: Parameter name
        - mean: Posterior mean
        - lower: Lower HDI bound (e.g., 3%)
        - upper: Upper HDI bound (e.g., 97%)

        This format is suitable for plotting with:
        >>> for param in params:
        ...     subset = df[df["param"] == param]
        ...     plt.errorbar(subset["mean"], subset["variant"],
        ...                  xerr=[subset["mean"]-subset["lower"],
        ...                        subset["upper"]-subset["mean"]])

    Example
    -------
    >>> results = run_prior_sensitivity(model, model_args, coefficient_vars=["user_rho"])
    >>> forest_df = create_coefficient_comparison_df(results, ["user_rho"])
    >>> print(forest_df)
    #        variant        param   mean  lower  upper
    # 0      default     user_rho  0.150  0.085  0.215
    # 1      diffuse     user_rho  0.142  0.078  0.206
    # 2  informative     user_rho  0.155  0.090  0.220
    """
    rows = []

    for variant_name, result in results.items():
        if result.coefficients.empty:
            continue

        for param in params:
            if param not in result.coefficients.index:
                logger.warning(f"Parameter '{param}' not found in {variant_name} coefficients")
                continue

            coef_row = result.coefficients.loc[param]

            # Extract HDI bounds - try common column names
            lower = None
            upper = None

            # ArviZ summary uses hdi_X% format
            for col in coef_row.index:
                if "hdi" in col.lower() and "%" in col:
                    pct = float(col.replace("hdi_", "").replace("%", ""))
                    if pct < 50:
                        lower = coef_row[col]
                    else:
                        upper = coef_row[col]

            rows.append(
                {
                    "variant": variant_name,
                    "param": param,
                    "mean": coef_row["mean"],
                    "lower": lower,
                    "upper": upper,
                }
            )

    return pd.DataFrame(rows)
