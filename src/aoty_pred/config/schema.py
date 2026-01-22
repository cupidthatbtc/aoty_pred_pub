"""Config schema definitions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    raw_csv: str
    encoding: str = "utf-8-sig"
    min_ratings: int = 30


class SplitConfig(BaseModel):
    strategy: str = "within_artist_last"
    group_col: str = "Artist"
    seed: int = 42
    train_frac: float = 0.65
    val_frac: float = 0.15
    test_frac: float = 0.20
    k_folds: int = 5
    time_holdout: bool = True

    @model_validator(mode="after")
    def validate_split_fractions(self) -> "SplitConfig":
        """Validate that split fractions sum to 1.0 (within tolerance)."""
        total = self.train_frac + self.val_frac + self.test_frac
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Split fractions must sum to 1.0, got {total:.3f} "
                f"(train={self.train_frac}, val={self.val_frac}, test={self.test_frac})"
            )
        return self


class ImputationMinCounts(BaseModel):
    artist: int = 2
    genre: int = 5
    decade: int = 20


class ImputationConfig(BaseModel):
    strategy: str = "hierarchical"
    hierarchy: list[str] = Field(default_factory=lambda: ["artist", "genre", "decade", "global"])
    min_counts: ImputationMinCounts = ImputationMinCounts()


class FeaturePCAConfig(BaseModel):
    core: bool = False
    genres_components: int = 30
    descriptors_components: int = 30


class FeatureBlockSpec(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class FeaturesConfig(BaseModel):
    include_genre: bool = True
    include_artist: bool = True
    include_temporal: bool = True
    include_album_type: bool = True
    pca: FeaturePCAConfig = FeaturePCAConfig()
    blocks: list[FeatureBlockSpec] = Field(default_factory=list)


class ModelPriorsConfig(BaseModel):
    intercept_sd: float = 1.0
    slope_sd: float = 1.0
    group_sd: float = 1.0
    sigma_sd: float = 1.0


class DynamicConfig(BaseModel):
    enabled: bool = True
    min_albums: int = 2


class ModelConfig(BaseModel):
    sampler: str = "pymc"
    tune: int = 2000
    draws: int = 2000
    chains: int = 4
    target_accept: float = 0.90
    max_treedepth: int = 12
    target_scale: str = "standardized"
    priors: ModelPriorsConfig = ModelPriorsConfig()
    dynamic: DynamicConfig = DynamicConfig()


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: ["r2", "rmse", "mae"])
    calibration_intervals: list[float] = Field(default_factory=lambda: [0.80, 0.95])
    coverage_tolerance: float = 0.03
    model_comparison: list[str] = Field(default_factory=lambda: ["waic", "loo"])


class SensitivityConfig(BaseModel):
    min_ratings: list[int] = Field(default_factory=lambda: [10, 20, 30, 40])
    dynamic_min_albums: list[int] = Field(default_factory=lambda: [2, 3])
    prior_slope_sd: list[float] = Field(default_factory=lambda: [0.5, 1.0])
    feature_ablations: list[str] = Field(
        default_factory=lambda: ["no_genre", "no_artist", "no_temporal"]
    )
    splits: list[str] = Field(default_factory=lambda: ["within_artist_last", "artist_group"])


class OutputsConfig(BaseModel):
    save_traces: bool = True
    run_dir: str = "runs"
    processed_path: str = "data/processed/regression_ready.csv"
    features_dir: str = "data/features"
    features_manifest: str = "data/features/features_manifest.json"


class AppConfig(BaseModel):
    dataset: DatasetConfig
    splits: SplitConfig
    imputation: ImputationConfig = ImputationConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig
    evaluation: EvaluationConfig = EvaluationConfig()
    sensitivity: SensitivityConfig = SensitivityConfig()
    outputs: OutputsConfig = OutputsConfig()
