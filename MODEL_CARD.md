# AOTY Artist Score Prediction Model

**Model type:** Bayesian Hierarchical Regression with Time-Varying Effects
**Version:** 0.1.0
**Last updated:** 2026-01-19

## Overview

This model predicts an artist's next album score on [Album of the Year](https://www.albumoftheyear.org/) based on their historical performance, genre, and career trajectory. It produces probabilistic predictions with calibrated uncertainty intervals.

## Quick Start

```python
from aoty_pred.models.bayes import load_model, predict_new_artist

# Load fitted model
model = load_model("models/user_score_model.nc")

# Predict for new artist data
predictions = predict_new_artist(model, artist_features)
print(f"Predicted score: {predictions.mean:.1f} [{predictions.lower:.1f}, {predictions.upper:.1f}]")
```

## Key Features

- **Hierarchical artist effects**: Partial pooling across artists for robust estimation
- **Time-varying slopes**: Career trajectories modeled as random walks
- **Autoregressive structure**: Album-to-album dependencies captured via AR(1)
- **Calibrated uncertainty**: 94% credible intervals with verified coverage

## Intended Use

- Academic research on music industry trends
- Personal exploration of album score patterns
- Understanding factors that influence critical reception

## Limitations

- Trained on English-language album reviews only
- Limited to artists with at least 3 prior releases
- Predictions less reliable for genre-crossing artists
- Not suitable for real-time or high-stakes decisions

## Documentation

- **Technical model card:** `reports/MODEL_CARD_TECHNICAL.md`
- **API documentation:** `docs/api/`
- **Research paper:** [link when published]

## Citation

If you use this model in research, please cite:

```bibtex
@software{aoty_pred,
  title = {AOTY Artist Score Prediction Model},
  author = {AOTY Prediction Project},
  year = {2026},
  url = {https://github.com/aoty-pred/aoty_pred_pub}
}
```
