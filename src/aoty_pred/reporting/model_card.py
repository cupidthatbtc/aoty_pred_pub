"""Model card generation for AOTY Artist Score Prediction.

This module generates model cards following Hugging Face conventions, adapted
for Bayesian hierarchical models. It produces both Markdown and LaTeX versions
suitable for documentation and academic publication.

Model cards serve two audiences:
- Academic reviewers: Methodology details, evaluation metrics, limitations
- Practitioners: Usage examples, intended use, code snippets

Usage:
    >>> from aoty_pred.reporting.model_card import (
    ...     create_default_model_card_data,
    ...     generate_model_card,
    ...     write_model_card,
    ... )
    >>> data = create_default_model_card_data()
    >>> card = generate_model_card(data, format="markdown")
    >>> write_model_card(data, Path("reports/model_card"))
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

__all__ = [
    "ModelCardData",
    "generate_model_card",
    "write_model_card",
    "create_default_model_card_data",
    "update_model_card_with_results",
]


@dataclass
class ModelCardData:
    """Container for all model card metadata.

    This dataclass captures comprehensive model documentation following
    Hugging Face model card conventions, adapted for Bayesian models.

    Attributes
    ----------
    model_name : str
        Human-readable model name.
    model_version : str
        Semantic version string (e.g., "0.1.0").
    model_type : str
        Model architecture description.
    authors : list[str]
        List of author names.
    created_date : str
        Creation date in ISO format (YYYY-MM-DD).
    last_updated : str
        Last update date in ISO format.
    dataset_name : str
        Name of training dataset.
    dataset_size : int
        Number of training observations.
    dataset_description : str
        Brief dataset description.
    data_preprocessing : str
        Description of preprocessing steps.
    architecture_summary : str
        Technical model architecture description.
    priors_description : str
        Description of prior distributions.
    hyperparameters : dict[str, Any]
        Key hyperparameters with values.
    convergence_summary : str
        Summary of MCMC convergence diagnostics.
    calibration_summary : str
        Summary of calibration metrics.
    predictive_summary : str
        Summary of predictive performance.
    loo_elpd : float | None
        LOO-CV expected log pointwise predictive density.
    limitations : list[str]
        List of known limitations.
    ethical_considerations : list[str]
        Ethical considerations and potential misuse.
    intended_use : str
        Description of intended use cases.
    out_of_scope_use : str
        Description of out-of-scope or discouraged uses.
    load_example : str
        Python code example for loading the model.
    predict_example : str
        Python code example for making predictions.
    interpret_example : str
        Python code example for interpreting results.
    """

    # Model identity
    model_name: str
    model_version: str
    model_type: str

    # Authors and dates
    authors: list[str]
    created_date: str
    last_updated: str

    # Training data
    dataset_name: str
    dataset_size: int
    dataset_description: str
    data_preprocessing: str

    # Model details
    architecture_summary: str
    priors_description: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Evaluation metrics
    convergence_summary: str = "Not yet evaluated"
    calibration_summary: str = "Not yet evaluated"
    predictive_summary: str = "Not yet evaluated"
    loo_elpd: float | None = None

    # Limitations and ethics
    limitations: list[str] = field(default_factory=list)
    ethical_considerations: list[str] = field(default_factory=list)
    intended_use: str = ""
    out_of_scope_use: str = ""

    # Code examples
    load_example: str = ""
    predict_example: str = ""
    interpret_example: str = ""


def generate_model_card(data: ModelCardData, format: str = "markdown") -> str:
    """Generate model card content from ModelCardData.

    Parameters
    ----------
    data : ModelCardData
        Model card data container.
    format : str, default "markdown"
        Output format: "markdown" or "latex".

    Returns
    -------
    str
        Formatted model card content.

    Raises
    ------
    ValueError
        If format is not "markdown" or "latex".

    Example
    -------
    >>> data = create_default_model_card_data()
    >>> card = generate_model_card(data, format="markdown")
    >>> print("## Model Details" in card)
    True
    """
    if format not in ("markdown", "latex"):
        raise ValueError(f"format must be 'markdown' or 'latex', got {format!r}")

    if format == "markdown":
        return _generate_markdown(data)
    else:
        return _generate_latex(data)


def _generate_markdown(data: ModelCardData) -> str:
    """Generate Markdown format model card."""
    lines = []

    # Header
    lines.append(f"# Model Card: {data.model_name}")
    lines.append("")

    # Model Details
    lines.append("## Model Details")
    lines.append("")
    lines.append(f"- **Model type:** {data.model_type}")
    lines.append(f"- **Version:** {data.model_version}")
    lines.append(f"- **Authors:** {', '.join(data.authors)}")
    lines.append(f"- **Created:** {data.created_date}")
    lines.append(f"- **Last updated:** {data.last_updated}")
    lines.append("")

    # Intended Use
    lines.append("## Intended Use")
    lines.append("")
    lines.append(data.intended_use)
    lines.append("")
    lines.append("### Out-of-Scope Use")
    lines.append("")
    lines.append(data.out_of_scope_use)
    lines.append("")

    # Training Data
    lines.append("## Training Data")
    lines.append("")
    lines.append(f"- **Dataset:** {data.dataset_name}")
    lines.append(f"- **Size:** {data.dataset_size:,} albums")
    lines.append(f"- **Description:** {data.dataset_description}")
    lines.append(f"- **Preprocessing:** {data.data_preprocessing}")
    lines.append("")

    # Model Architecture
    lines.append("## Model Architecture")
    lines.append("")
    lines.append(data.architecture_summary)
    lines.append("")

    # Prior Distributions
    lines.append("### Prior Distributions")
    lines.append("")
    lines.append(data.priors_description)
    lines.append("")

    # Hyperparameters
    lines.append("### Hyperparameters")
    lines.append("")
    if data.hyperparameters:
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for param, value in data.hyperparameters.items():
            lines.append(f"| {param} | {value} |")
    else:
        lines.append("No hyperparameters specified.")
    lines.append("")

    # Evaluation Results
    lines.append("## Evaluation Results")
    lines.append("")

    # Convergence
    lines.append("### Convergence Diagnostics")
    lines.append("")
    lines.append(data.convergence_summary)
    lines.append("")

    # Calibration
    lines.append("### Calibration")
    lines.append("")
    lines.append(data.calibration_summary)
    lines.append("")

    # Predictive Performance
    lines.append("### Predictive Performance")
    lines.append("")
    lines.append(data.predictive_summary)
    if data.loo_elpd is not None:
        lines.append("")
        lines.append(f"- **ELPD (LOO-CV):** {data.loo_elpd:.1f}")
    lines.append("")

    # Limitations
    lines.append("## Limitations")
    lines.append("")
    if data.limitations:
        for limitation in data.limitations:
            lines.append(f"- {limitation}")
    else:
        lines.append("No limitations documented.")
    lines.append("")

    # Ethical Considerations
    lines.append("## Ethical Considerations")
    lines.append("")
    if data.ethical_considerations:
        for consideration in data.ethical_considerations:
            lines.append(f"- {consideration}")
    else:
        lines.append("No ethical considerations documented.")
    lines.append("")

    # How to Use
    lines.append("## How to Use")
    lines.append("")

    # Loading
    lines.append("### Loading the Model")
    lines.append("")
    lines.append("```python")
    lines.append(data.load_example)
    lines.append("```")
    lines.append("")

    # Predictions
    lines.append("### Making Predictions")
    lines.append("")
    lines.append("```python")
    lines.append(data.predict_example)
    lines.append("```")
    lines.append("")

    # Interpretation
    lines.append("### Interpreting Results")
    lines.append("")
    lines.append("```python")
    lines.append(data.interpret_example)
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _generate_latex(data: ModelCardData) -> str:
    """Generate LaTeX format model card."""
    lines = []

    # Document setup
    lines.append("\\documentclass{article}")
    lines.append("\\usepackage{booktabs}")
    lines.append("\\usepackage{listings}")
    lines.append("\\usepackage{hyperref}")
    lines.append("")
    lines.append("\\lstset{")
    lines.append("  language=Python,")
    lines.append("  basicstyle=\\ttfamily\\small,")
    lines.append("  breaklines=true,")
    lines.append("  frame=single")
    lines.append("}")
    lines.append("")
    lines.append("\\begin{document}")
    lines.append("")

    # Title
    lines.append(f"\\section*{{Model Card: {_latex_escape(data.model_name)}}}")
    lines.append("")

    # Model Details
    lines.append("\\subsection*{Model Details}")
    lines.append("\\begin{itemize}")
    lines.append(f"  \\item \\textbf{{Model type:}} {_latex_escape(data.model_type)}")
    lines.append(f"  \\item \\textbf{{Version:}} {data.model_version}")
    lines.append(f"  \\item \\textbf{{Authors:}} {_latex_escape(', '.join(data.authors))}")
    lines.append(f"  \\item \\textbf{{Created:}} {data.created_date}")
    lines.append(f"  \\item \\textbf{{Last updated:}} {data.last_updated}")
    lines.append("\\end{itemize}")
    lines.append("")

    # Intended Use
    lines.append("\\subsection*{Intended Use}")
    lines.append(_latex_escape(data.intended_use))
    lines.append("")
    lines.append("\\subsubsection*{Out-of-Scope Use}")
    lines.append(_latex_escape(data.out_of_scope_use))
    lines.append("")

    # Training Data
    lines.append("\\subsection*{Training Data}")
    lines.append("\\begin{itemize}")
    lines.append(f"  \\item \\textbf{{Dataset:}} {_latex_escape(data.dataset_name)}")
    lines.append(f"  \\item \\textbf{{Size:}} {data.dataset_size:,} albums")
    lines.append(f"  \\item \\textbf{{Description:}} {_latex_escape(data.dataset_description)}")
    lines.append(f"  \\item \\textbf{{Preprocessing:}} {_latex_escape(data.data_preprocessing)}")
    lines.append("\\end{itemize}")
    lines.append("")

    # Model Architecture
    lines.append("\\subsection*{Model Architecture}")
    lines.append(_latex_escape(data.architecture_summary))
    lines.append("")

    # Prior Distributions
    lines.append("\\subsubsection*{Prior Distributions}")
    lines.append(_latex_escape(data.priors_description))
    lines.append("")

    # Hyperparameters
    lines.append("\\subsubsection*{Hyperparameters}")
    if data.hyperparameters:
        lines.append("\\begin{tabular}{ll}")
        lines.append("\\toprule")
        lines.append("Parameter & Value \\\\")
        lines.append("\\midrule")
        for param, value in data.hyperparameters.items():
            lines.append(f"{_latex_escape(param)} & {value} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
    else:
        lines.append("No hyperparameters specified.")
    lines.append("")

    # Evaluation Results
    lines.append("\\subsection*{Evaluation Results}")
    lines.append("")
    lines.append("\\subsubsection*{Convergence Diagnostics}")
    lines.append(_latex_escape(data.convergence_summary))
    lines.append("")
    lines.append("\\subsubsection*{Calibration}")
    lines.append(_latex_escape(data.calibration_summary))
    lines.append("")
    lines.append("\\subsubsection*{Predictive Performance}")
    lines.append(_latex_escape(data.predictive_summary))
    if data.loo_elpd is not None:
        lines.append("")
        lines.append(f"ELPD (LOO-CV): {data.loo_elpd:.1f}")
    lines.append("")

    # Limitations
    lines.append("\\subsection*{Limitations}")
    if data.limitations:
        lines.append("\\begin{itemize}")
        for limitation in data.limitations:
            lines.append(f"  \\item {_latex_escape(limitation)}")
        lines.append("\\end{itemize}")
    else:
        lines.append("No limitations documented.")
    lines.append("")

    # Ethical Considerations
    lines.append("\\subsection*{Ethical Considerations}")
    if data.ethical_considerations:
        lines.append("\\begin{itemize}")
        for consideration in data.ethical_considerations:
            lines.append(f"  \\item {_latex_escape(consideration)}")
        lines.append("\\end{itemize}")
    else:
        lines.append("No ethical considerations documented.")
    lines.append("")

    # How to Use
    lines.append("\\subsection*{How to Use}")
    lines.append("")
    lines.append("\\subsubsection*{Loading the Model}")
    lines.append("\\begin{lstlisting}")
    lines.append(data.load_example)
    lines.append("\\end{lstlisting}")
    lines.append("")
    lines.append("\\subsubsection*{Making Predictions}")
    lines.append("\\begin{lstlisting}")
    lines.append(data.predict_example)
    lines.append("\\end{lstlisting}")
    lines.append("")
    lines.append("\\subsubsection*{Interpreting Results}")
    lines.append("\\begin{lstlisting}")
    lines.append(data.interpret_example)
    lines.append("\\end{lstlisting}")
    lines.append("")

    # End document
    lines.append("\\end{document}")

    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in text."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def write_model_card(
    data: ModelCardData,
    output_path: Path,
    formats: tuple[str, ...] = ("md", "tex"),
) -> list[Path]:
    """Write model card to files in specified formats.

    Parameters
    ----------
    data : ModelCardData
        Model card data container.
    output_path : Path
        Base path for output files (without extension).
        Files will be named {output_path}.md, {output_path}.tex, etc.
    formats : tuple[str, ...], default ("md", "tex")
        File formats to generate. Supported: "md" (Markdown), "tex" (LaTeX).

    Returns
    -------
    list[Path]
        List of paths to created files.

    Example
    -------
    >>> data = create_default_model_card_data()
    >>> paths = write_model_card(data, Path("reports/model_card"))
    >>> print(paths)
    [PosixPath('reports/model_card.md'), PosixPath('reports/model_card.tex')]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    created_paths = []

    for fmt in formats:
        if fmt == "md":
            content = generate_model_card(data, format="markdown")
            file_path = output_path.with_suffix(".md")
        elif fmt == "tex":
            content = generate_model_card(data, format="latex")
            file_path = output_path.with_suffix(".tex")
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        file_path.write_text(content, encoding="utf-8")
        created_paths.append(file_path)

    return created_paths


def create_default_model_card_data() -> ModelCardData:
    """Create ModelCardData with AOTY project defaults.

    Returns pre-filled ModelCardData with sensible defaults for the
    AOTY Artist Score Prediction model. Evaluation metrics are left
    as placeholders to be filled after model fitting.

    Returns
    -------
    ModelCardData
        Pre-filled model card data for AOTY project.

    Example
    -------
    >>> data = create_default_model_card_data()
    >>> print(data.model_name)
    'AOTY Artist Score Prediction Model'
    >>> print(data.model_type)
    'Bayesian Hierarchical Regression with Time-Varying Effects'
    """
    today = date.today().isoformat()

    return ModelCardData(
        # Model identity
        model_name="AOTY Artist Score Prediction Model",
        model_version="0.1.0",
        model_type="Bayesian Hierarchical Regression with Time-Varying Effects",
        # Authors and dates
        authors=["AOTY Prediction Project"],
        created_date=today,
        last_updated=today,
        # Training data
        dataset_name="Album of the Year (AOTY)",
        dataset_size=0,  # To be filled with actual training data size
        dataset_description=(
            "Music album metadata and scores from Album of the Year, "
            "including artist information, release dates, genres, and "
            "both critic and user scores."
        ),
        data_preprocessing=(
            "Data filtered to artists with 3+ albums, "
            "within-artist temporal split for leakage prevention, "
            "features standardized to zero mean and unit variance."
        ),
        # Model details
        architecture_summary=(
            "Bayesian hierarchical regression with four key components:\n\n"
            "1. **Hierarchical artist effects**: Partial pooling across artists "
            "for robust estimation of artist quality. Non-centered parameterization "
            "via LocScaleReparam avoids funnel geometry.\n\n"
            "2. **Time-varying slopes**: Artist quality modeled as a random walk, "
            "allowing career trajectories to evolve over time.\n\n"
            "3. **AR(1) structure**: Album-to-album dependencies captured via "
            "autoregressive term, modeling momentum effects where consecutive "
            "albums tend to have correlated scores.\n\n"
            "4. **Heteroscedastic observation noise** (sigma_ref parameterization): "
            "Albums with more reviews have lower observation noise. The model samples "
            "sigma_ref (noise at the median review count n_ref) and derives per-observation "
            "noise as: sigma_obs = sigma_ref * n_ref^n_exponent, then "
            "sigma_i = sigma_obs / n_reviews_i^n_exponent. This reparameterization breaks "
            "the multiplicative funnel between sigma_obs and n_exponent that causes "
            "divergent transitions in MCMC sampling.\n\n"
            "Mathematical form:\n"
            "- y_ij ~ Normal(mu_ij, sigma_i)\n"
            "- mu_ij = artist_effect_jt + X_ij @ beta + rho * prev_score_ij\n"
            "- artist_effect_jt evolves via random walk from initial effect\n"
            "- sigma_i = sigma_obs / n_reviews_i^n_exponent (heteroscedastic mode)"
        ),
        priors_description=(
            "Default weakly informative priors:\n\n"
            "- **mu_artist** ~ Normal(0, 1): Population mean of artist effects\n"
            "- **sigma_artist** ~ HalfNormal(0.5): Between-artist variation (encourages pooling)\n"
            "- **sigma_rw** ~ HalfNormal(0.1): Random walk innovation (smooth trajectories)\n"
            "- **rho** ~ TruncatedNormal(0, 0.3, -0.99, 0.99): AR(1) coefficient (stationary)\n"
            "- **beta** ~ Normal(0, 1): Fixed effect coefficients\n"
            "- **sigma_obs** ~ HalfNormal(1): Observation noise\n"
            "- **sigma_ref** ~ HalfNormal(1): Observation noise at the reference review count "
            "(n_ref = median of training n_reviews). When heteroscedastic mode is active, "
            "sigma_ref replaces sigma_obs as the sampled parameter. sigma_obs is derived as "
            "sigma_ref * n_ref^n_exponent.\n"
            "- **n_exponent** ~ LogitNormal(0, 1) mapped to (0, 1): Power-law exponent "
            "controlling how observation noise decreases with review count. "
            "Only sampled in --learn-n-exponent mode."
        ),
        hyperparameters={
            "mu_artist_loc": 0.0,
            "mu_artist_scale": 1.0,
            "sigma_artist_scale": 0.5,
            "sigma_rw_scale": 0.1,
            "rho_loc": 0.0,
            "rho_scale": 0.3,
            "beta_loc": 0.0,
            "beta_scale": 1.0,
            "sigma_obs_scale": 1.0,
            "sigma_ref_scale": 1.0,
            "n_exponent_default": 0.0,
        },
        # Evaluation metrics (placeholders)
        convergence_summary="Model not yet fitted. Run MCMC first.",
        calibration_summary="Model not yet fitted. Run MCMC first.",
        predictive_summary="Model not yet fitted. Run MCMC first.",
        loo_elpd=None,
        # Limitations
        limitations=[
            "Trained on English-language reviews; may not generalize to other markets",
            "Requires artists to have at least 3 prior releases for reliable predictions",
            "Less reliable for genre-crossing artists due to sparse data",
            "Historical biases in music criticism may be reflected in predictions",
            "Does not account for album-specific factors (production, label influence)",
            "Assumes gradual career evolution; sudden style changes poorly predicted",
            "Score predictions are probabilistic and should not be treated as ground truth",
        ],
        # Ethical considerations
        ethical_considerations=[
            "Predictions should not gatekeep artists or influence career decisions",
            "Aggregated scores may not reflect artistic merit or listener preferences",
            "Care should be taken when interpreting genre-based effects",
            "Model may perpetuate historical biases present in music criticism",
            "Predictions are for research and exploration, not commercial evaluation",
            "Artists and labels should not be ranked solely based on predicted scores",
        ],
        # Intended use
        intended_use=(
            "This model is intended for:\n\n"
            "- Academic research on music industry trends and career trajectories\n"
            "- Personal exploration of album score patterns and artist development\n"
            "- Understanding factors that influence critical and user reception\n"
            "- Educational demonstration of Bayesian hierarchical modeling\n"
            "- Reproducibility research in music information retrieval"
        ),
        out_of_scope_use=(
            "This model should NOT be used for:\n\n"
            "- Commercial artist evaluation or signing decisions\n"
            "- Real-time prediction systems in production environments\n"
            "- Automated content moderation or recommendation without human review\n"
            "- High-stakes decisions affecting artists' careers or livelihoods\n"
            "- Marketing claims about album quality or artist potential"
        ),
        # Code examples
        load_example=(
            "from aoty_pred.models.bayes import load_model\n"
            "\n"
            "# Load fitted model from NetCDF file\n"
            'idata = load_model("models/user_score_model.nc")'
        ),
        predict_example=(
            "from aoty_pred.models.bayes import predict_new_artist\n"
            "import jax.numpy as jnp\n"
            "\n"
            "# Prepare features for new prediction\n"
            "artist_features = {\n"
            "    'prev_score': jnp.array([72.5]),\n"
            "    'career_years': jnp.array([5.0]),\n"
            "    'album_sequence': jnp.array([4]),\n"
            "}\n"
            "\n"
            "# Generate predictions with uncertainty\n"
            "predictions = predict_new_artist(\n"
            "    model, idata, artist_features, n_samples=1000\n"
            ")"
        ),
        interpret_example=(
            "import numpy as np\n"
            "\n"
            "# Extract prediction statistics\n"
            "pred_mean = np.mean(predictions)\n"
            "pred_std = np.std(predictions)\n"
            "ci_95 = np.percentile(predictions, [2.5, 97.5])\n"
            "\n"
            'print(f"Predicted score: {pred_mean:.1f} +/- {pred_std:.1f}")\n'
            'print(f"95% CI: [{ci_95[0]:.1f}, {ci_95[1]:.1f}]")'
        ),
    )


def update_model_card_with_results(
    data: ModelCardData,
    idata=None,
    convergence=None,
    coverage_results: dict | None = None,
    loo_result=None,
    point_metrics=None,
) -> ModelCardData:
    """Update model card data with evaluation results from fitted model.

    Parameters
    ----------
    data : ModelCardData
        Base model card data to update.
    idata : az.InferenceData, optional
        Fitted model inference data.
    convergence : ConvergenceDiagnostics, optional
        Convergence diagnostic results from check_convergence().
    coverage_results : dict[float, CoverageResult], optional
        Coverage results at multiple probability levels from compute_multi_coverage().
    loo_result : LOOResult, optional
        LOO-CV results from compute_loo().
    point_metrics : PointMetrics, optional
        Point prediction metrics from compute_point_metrics().

    Returns
    -------
    ModelCardData
        New ModelCardData instance with updated evaluation fields.

    Example
    -------
    >>> from aoty_pred.models.bayes.diagnostics import check_convergence
    >>> from aoty_pred.evaluation import compute_multi_coverage, compute_loo
    >>>
    >>> diags = check_convergence(result.idata)
    >>> coverage = compute_multi_coverage(y_true, y_samples)
    >>> loo = compute_loo(idata_with_loglik)
    >>>
    >>> updated_data = update_model_card_with_results(
    ...     data, convergence=diags, coverage_results=coverage, loo_result=loo
    ... )
    """
    # Start with existing values
    new_convergence = data.convergence_summary
    new_calibration = data.calibration_summary
    new_predictive = data.predictive_summary
    new_loo_elpd = data.loo_elpd

    # Update convergence summary
    if convergence is not None:
        status = "PASSED" if convergence.passed else "FAILED"
        new_convergence = (
            f"Convergence status: {status}\n\n"
            f"- R-hat (max): {convergence.rhat_max:.4f} (threshold: < 1.01)\n"
            f"- ESS bulk (min): {convergence.ess_bulk_min:,}\n"
            f"- ESS tail (min): {convergence.ess_tail_min:,}\n"
            f"- Divergent transitions: {convergence.divergences}"
        )
        if convergence.failing_params:
            new_convergence += f"\n- Failing parameters: {', '.join(convergence.failing_params)}"

    # Update calibration summary
    if coverage_results is not None:
        cov_lines = ["Credible interval coverage:"]
        for prob, result in sorted(coverage_results.items()):
            nominal_pct = int(prob * 100)
            empirical_pct = result.empirical * 100
            cov_lines.append(f"- {nominal_pct}% CI: {empirical_pct:.1f}% empirical coverage")
        new_calibration = "\n".join(cov_lines)

    # Update predictive summary
    if point_metrics is not None:
        new_predictive = (
            f"Point prediction metrics:\n\n"
            f"- MAE: {point_metrics.mae:.2f}\n"
            f"- RMSE: {point_metrics.rmse:.2f}\n"
            f"- R-squared: {point_metrics.r2:.3f}"
        )

    # Update LOO ELPD
    if loo_result is not None:
        new_loo_elpd = loo_result.elpd_loo
        if new_predictive != data.predictive_summary:
            new_predictive += (
                f"\n- ELPD (LOO-CV): {loo_result.elpd_loo:.1f} (SE: {loo_result.se_elpd:.1f})"
            )
        else:
            new_predictive = (
                f"ELPD (LOO-CV): {loo_result.elpd_loo:.1f} (SE: {loo_result.se_elpd:.1f})"
            )

    # Create new instance with updated fields
    return ModelCardData(
        model_name=data.model_name,
        model_version=data.model_version,
        model_type=data.model_type,
        authors=data.authors,
        created_date=data.created_date,
        last_updated=date.today().isoformat(),
        dataset_name=data.dataset_name,
        dataset_size=data.dataset_size,
        dataset_description=data.dataset_description,
        data_preprocessing=data.data_preprocessing,
        architecture_summary=data.architecture_summary,
        priors_description=data.priors_description,
        hyperparameters=data.hyperparameters,
        convergence_summary=new_convergence,
        calibration_summary=new_calibration,
        predictive_summary=new_predictive,
        loo_elpd=new_loo_elpd,
        limitations=data.limitations,
        ethical_considerations=data.ethical_considerations,
        intended_use=data.intended_use,
        out_of_scope_use=data.out_of_scope_use,
        load_example=data.load_example,
        predict_example=data.predict_example,
        interpret_example=data.interpret_example,
    )
