"""Bayesian hierarchical models for album score prediction."""

from aoty_pred.models.bayes.fit import (
    FitResult,
    MCMCConfig,
    fit_model,
    get_gpu_info,
)
from aoty_pred.models.bayes.io import (
    ModelManifest,
    ModelsManifest,
    generate_model_filename,
    load_manifest,
    load_model,
    save_model,
)
from aoty_pred.models.bayes.model import (
    album_score_model,
    critic_score_model,
    make_score_model,
    user_score_model,
)
from aoty_pred.models.bayes.predict import (
    PredictionResult,
    generate_posterior_predictive,
    predict_new_artist,
    predict_out_of_sample,
)
from aoty_pred.models.bayes.priors import PriorConfig, get_default_priors

__all__ = [
    # Priors
    "PriorConfig",
    "get_default_priors",
    # Models
    "make_score_model",
    "user_score_model",
    "critic_score_model",
    "album_score_model",
    # Fitting
    "fit_model",
    "MCMCConfig",
    "FitResult",
    "get_gpu_info",
    # Prediction
    "PredictionResult",
    "generate_posterior_predictive",
    "predict_out_of_sample",
    "predict_new_artist",
    # I/O
    "ModelManifest",
    "ModelsManifest",
    "save_model",
    "load_model",
    "generate_model_filename",
    "load_manifest",
]
