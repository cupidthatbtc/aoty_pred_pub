# Evaluation Protocol

Metrics
- R2 on held-out test sets (primary and secondary splits)
- RMSE and MAE
- Calibration curves and coverage of credible intervals (80% and 95%)

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
