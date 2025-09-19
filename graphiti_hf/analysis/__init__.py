"""
Community Detection Module for Graphiti-HF

This module provides community detection algorithms and analysis capabilities
specifically designed to work with HuggingFace datasets.
"""

from .community_detector import (
    CommunityDetector,
    CommunityDetectionConfig,
    CommunityStats,
    CommunityMember,
    CommunitySimilarity
)

__all__ = [
    "CommunityDetector",
    "CommunityDetectionConfig", 
    "CommunityStats",
    "CommunityMember",
    "CommunitySimilarity"
]