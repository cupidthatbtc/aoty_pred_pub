# Model Card: AOTY Artist Score Prediction Model

## Model Details

- **Model type:** Bayesian Hierarchical Regression with Time-Varying Effects
- **Version:** 0.1.0
- **Authors:** AOTY Prediction Project
- **Created:** 2026-01-20
- **Last updated:** 2026-01-20

## Intended Use

This model is intended for:

- Academic research on music industry trends and career trajectories
- Personal exploration of album score patterns and artist development
- Understanding factors that influence critical and user reception
- Educational demonstration of Bayesian hierarchical modeling
- Reproducibility research in music information retrieval

### Out-of-Scope Use

This model should NOT be used for:

- Commercial artist evaluation or signing decisions
- Real-time prediction systems in production environments
- Automated content moderation or recommendation without human review
- High-stakes decisions affecting artists' careers or livelihoods
- Marketing claims about album quality or artist potential

## Training Data

- **Dataset:** Album of the Year (AOTY)
- **Size:** 0 albums
- **Description:** Music album metadata and scores from Album of the Year, including artist information, release dates, genres, and both critic and user scores.
- **Preprocessing:** Data filtered to artists with 3+ albums, within-artist temporal split for leakage prevention, features standardized to zero mean and unit variance.

## Model Architecture

Bayesian hierarchical regression with three key components:

1. **Hierarchical artist effects**: Partial pooling across artists for robust estimation of artist quality. Non-centered parameterization via LocScaleReparam avoids funnel geometry.

2. **Time-varying slopes**: Artist quality modeled as a random walk, allowing career trajectories to evolve over time.

3. **AR(1) structure**: Album-to-album dependencies captured via autoregressive term, modeling momentum effects where consecutive albums tend to have correlated scores.

Mathematical form:
- y_ij ~ Normal(mu_ij, sigma_obs)
- mu_ij = artist_effect_jt + X_ij @ beta + rho * prev_score_ij
- artist_effect_jt evolves via random walk from initial effect

### Prior Distributions

Default weakly informative priors:

- **mu_artist** ~ Normal(0, 1): Population mean of artist effects
- **sigma_artist** ~ HalfNormal(0.5): Between-artist variation (encourages pooling)
- **sigma_rw** ~ HalfNormal(0.1): Random walk innovation (smooth trajectories)
- **rho** ~ TruncatedNormal(0, 0.3, -0.99, 0.99): AR(1) coefficient (stationary)
- **beta** ~ Normal(0, 1): Fixed effect coefficients
- **sigma_obs** ~ HalfNormal(1): Observation noise

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| mu_artist_loc | 0.0 |
| mu_artist_scale | 1.0 |
| sigma_artist_scale | 0.5 |
| sigma_rw_scale | 0.1 |
| rho_loc | 0.0 |
| rho_scale | 0.3 |
| beta_loc | 0.0 |
| beta_scale | 1.0 |
| sigma_obs_scale | 1.0 |

## Evaluation Results

### Convergence Diagnostics

Model not yet fitted. Run MCMC and call update_model_card_with_results().

### Calibration

Model not yet fitted. Run MCMC and call update_model_card_with_results().

### Predictive Performance

Model not yet fitted. Run MCMC and call update_model_card_with_results().

## Limitations

- Trained primarily on English-language album reviews; may not generalize to other markets
- Requires artists to have at least 3 prior releases for reliable predictions
- Predictions less reliable for genre-crossing artists due to sparse data in novel combinations
- Historical biases in music criticism may be reflected in predictions
- Does not account for album-specific factors (production changes, label influence)
- Career trajectory model assumes gradual evolution; sudden style changes may be poorly predicted
- Score predictions are probabilistic and should not be treated as ground truth

## Ethical Considerations

- Model predictions should not be used to gatekeep artists or influence career decisions
- Aggregated scores may not reflect artistic merit or individual listener preferences
- Care should be taken when interpreting genre-based effects to avoid stereotyping
- Model may perpetuate historical biases present in music criticism
- Predictions are for research and personal exploration, not commercial evaluation
- Artists and labels should not be ranked solely based on predicted scores

## How to Use

### Loading the Model

```python
from aoty_pred.models.bayes import load_model

# Load fitted model from NetCDF file
idata = load_model("models/user_score_model.nc")
```

### Making Predictions

```python
from aoty_pred.models.bayes import predict_new_artist
import jax.numpy as jnp

# Prepare features for new prediction
artist_features = {
    'prev_score': jnp.array([72.5]),
    'career_years': jnp.array([5.0]),
    'album_sequence': jnp.array([4]),
}

# Generate predictions with uncertainty
predictions = predict_new_artist(
    model, idata, artist_features, n_samples=1000
)
```

### Interpreting Results

```python
import numpy as np

# Extract prediction statistics
pred_mean = np.mean(predictions)
pred_std = np.std(predictions)
ci_95 = np.percentile(predictions, [2.5, 97.5])

print(f"Predicted score: {pred_mean:.1f} +/- {pred_std:.1f}")
print(f"95% CI: [{ci_95[0]:.1f}, {ci_95[1]:.1f}]")
```
