"""Iteration 5 Semi-Automated CAD Vectorization Package."""

from src.v5_semi_automated.rectification import (
    RectificationResult,
    SemanticGuide,
    SemiAutomatedRectifier,
    classify_guide_format,
    normalize_unit_key,
)
from src.v5_semi_automated.topology import (
    SemiAutomatedTopology,
    TopologicalRoomNode,
    TopologyResult,
)
from src.v5_semi_automated.geometry import (
    GeometryResult,
    RegularizedRoomNode,
    SemiAutomatedGeometry,
)
from src.v5_semi_automated.scale import (
    DimensionedRoom,
    FixtureScaleAnchor,
    ScaleResult,
    SemiAutomatedScale,
)

__all__ = [
    "DimensionedRoom",
    "FixtureScaleAnchor",
    "GeometryResult",
    "RectificationResult",
    "RegularizedRoomNode",
    "ScaleResult",
    "SemanticGuide",
    "SemiAutomatedGeometry",
    "SemiAutomatedRectifier",
    "SemiAutomatedScale",
    "SemiAutomatedTopology",
    "TopologicalRoomNode",
    "TopologyResult",
    "classify_guide_format",
    "normalize_unit_key",
]
