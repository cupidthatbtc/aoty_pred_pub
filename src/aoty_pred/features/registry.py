"""Feature registry (stub).

Use to register and construct feature blocks by name from config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .base import FeatureBlock
from .album_type import AlbumTypeBlock
from .artist import ArtistHistoryBlock, ArtistReputationBlock
from .collaboration import CollaborationBlock
from .core import CoreNumericBlock
from .descriptor_pca import DescriptorPCABlock
from .genre import GenreBlock, GenrePCABlock
from .temporal import TemporalBlock


@dataclass
class FeatureSpec:
    name: str
    params: dict[str, Any]


class FeatureRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, Callable[[dict[str, Any]], FeatureBlock]] = {}

    def register(self, name: str, builder: Callable[[dict[str, Any]], FeatureBlock]) -> None:
        if name in self._builders:
            raise ValueError(f"Feature block already registered: {name}")
        self._builders[name] = builder

    def build(self, spec: FeatureSpec) -> FeatureBlock:
        if spec.name not in self._builders:
            raise KeyError(f"Unknown feature block: {spec.name}")
        return self._builders[spec.name](spec.params)

    def build_all(self, specs: list[FeatureSpec]) -> list[FeatureBlock]:
        return [self.build(spec) for spec in specs]


def parse_feature_specs(config: dict[str, Any]) -> list[FeatureSpec]:
    blocks = config.get("features", {}).get("blocks", [])
    specs: list[FeatureSpec] = []
    for block in blocks:
        name = block.get("name")
        params = block.get("params", {})
        if not name:
            raise ValueError("Feature block missing name")
        specs.append(FeatureSpec(name=name, params=params))
    return specs


def build_default_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    registry.register("core_numeric", lambda params: CoreNumericBlock(params))
    registry.register("temporal", lambda params: TemporalBlock(params))
    registry.register("artist_reputation", lambda params: ArtistReputationBlock(params))
    registry.register("artist_history", lambda params: ArtistHistoryBlock(params))
    registry.register("genre", lambda params: GenreBlock(params))
    registry.register("genre_pca", lambda params: GenrePCABlock(params))  # Alias for backwards compat
    registry.register("descriptor_pca", lambda params: DescriptorPCABlock(params))
    registry.register("album_type", lambda params: AlbumTypeBlock(params))
    registry.register("collaboration", lambda params: CollaborationBlock(params))
    return registry
