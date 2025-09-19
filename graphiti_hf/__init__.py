"""
Graphiti-HF: Adapting Graphiti for Hugging Face Ecosystem

This package provides a fork of Graphiti that uses Hugging Face Datasets
instead of traditional graph databases for knowledge graph storage and operations.
"""

__version__ = "0.1.0"
__author__ = "Graphiti-HF Team"
__email__ = "team@graphiti-hf.org"

from .drivers.huggingface_driver import HuggingFaceDriver
from .search import (
    GraphTraversalEngine,
    TraversalConfig,
    TraversalAlgorithm,
    EdgeFilterType,
    TraversalResult,
    HybridSearchEngine,
    HybridSearchConfig
)
from .search.performance_optimizer import (
    SearchIndexManager,
    IndexConfig,
    IndexType,
    PerformanceMetrics,
    QueryPattern,
    extend_huggingface_driver,
    optimize_search_performance,
    get_performance_metrics,
    auto_rebuild_indices
)

__all__ = [
    # Core driver
    "HuggingFaceDriver",
    
    # Search engines
    "GraphTraversalEngine",
    "TraversalConfig",
    "TraversalAlgorithm",
    "EdgeFilterType",
    "TraversalResult",
    "HybridSearchEngine",
    "HybridSearchConfig",
    
    # Performance optimization
    "SearchIndexManager",
    "IndexConfig",
    "IndexType",
    "PerformanceMetrics",
    "QueryPattern",
    "extend_huggingface_driver",
    "optimize_search_performance",
    "get_performance_metrics",
    "auto_rebuild_indices"
]