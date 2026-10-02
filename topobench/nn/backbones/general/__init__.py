"""Backbones whose architecture is independent of the input topology domain."""

from .trawl import TRAWL
from .trawl_categorical import CategoricalTRAWL
from .trawl_continuous import ContinuousTRAWL

BACKBONE_CLASSES = {
    "TRAWL": TRAWL,
    "CategoricalTRAWL": CategoricalTRAWL,
    "ContinuousTRAWL": ContinuousTRAWL,
}

__all__ = ["BACKBONE_CLASSES", *BACKBONE_CLASSES]
