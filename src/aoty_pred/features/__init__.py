"""Feature blocks package."""

from .album_type import AlbumTypeBlock
from .artist import ArtistHistoryBlock, ArtistReputationBlock
from .collaboration import CollaborationBlock
from .core import CoreNumericBlock
from .descriptor_pca import DescriptorPCABlock
from .genre import GenreBlock, GenrePCABlock
from .registry import FeatureRegistry, FeatureSpec, build_default_registry, parse_feature_specs
from .temporal import TemporalBlock

__all__ = [
    "AlbumTypeBlock",
    "ArtistHistoryBlock",
    "ArtistReputationBlock",
    "CollaborationBlock",
    "CoreNumericBlock",
    "DescriptorPCABlock",
    "GenreBlock",
    "GenrePCABlock",
    "TemporalBlock",
    "FeatureRegistry",
    "FeatureSpec",
    "build_default_registry",
    "parse_feature_specs",
]
