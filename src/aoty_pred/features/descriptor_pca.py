"""Descriptor PCA feature block (stub)."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from .base import BaseFeatureBlock, FeatureContext, FeatureOutput


class DescriptorPCABlock(BaseFeatureBlock):
    name: ClassVar[str] = "descriptor_pca"
    requires: ClassVar[list[str]] = []

    def transform(self, df, ctx: FeatureContext) -> FeatureOutput:
        self.validate_columns(df)
        data = pd.DataFrame(index=df.index)
        metadata = {"block": self.name, "params": self.params}
        return FeatureOutput(data=data, feature_names=list(data.columns), metadata=metadata)
