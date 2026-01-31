# Evaluation Protocol

Metrics (implemented in evaluate.py)
- **R-squared (R2)**: Coefficient of determination on held-out test set
- **RMSE**: Root mean squared error of posterior predictive mean vs observed
- **MAE**: Mean absolute error of posterior predictive mean vs observed
- **Mean Bias**: Average signed prediction error (pred - obs)
- **CRPS**: Continuous Ranked Probability Score (proper scoring rule for full predictive distribution)
- **Calibration Coverage (90%)**: Fraction of test observations falling within 90% posterior predictive interval (target: 0.90)
- **Calibration Coverage (50%)**: Fraction of test observations falling within 50% posterior predictive interval (target: 0.50)
- **Interval Width**: Mean width of 90% and 50% credible intervals

Cross-validation
- Primary evaluation: within-artist temporal holdout (last album per artist)
- Secondary evaluation: artist-group split (no artist overlap)
- Group-aware CV by artist for hyperparameter selection
- Nested CV if tuning model structure

Diagnostics
- R-hat <= 1.01 for all key parameters
- ESS above threshold
- No divergent transitions after tuning

Model comparison
- WAIC/LOO with robust error handling
- Report uncertainty in comparison metrics
