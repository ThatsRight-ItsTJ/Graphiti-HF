"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .vector_search import VectorSearchEngine
from .graph_traversal import (
    GraphTraversalEngine,
    TraversalConfig,
    TraversalAlgorithm,
    EdgeFilterType,
    TraversalResult
)
from .hybrid_search import HybridSearchEngine, HybridSearchConfig
from .advanced_config import (
    AdvancedSearchConfig,
    SearchMethod,
    RankingStrategy,
    SemanticSearchConfig,
    KeywordSearchConfig,
    GraphSearchConfig,
    TemporalSearchConfig,
    HybridSearchConfig as AdvancedHybridSearchConfig,
    PerformanceConfig,
    DomainConfig,
    TemporalConfig
)
from .integration import (
    SearchEngineIntegrator,
    create_semantic_search_config,
    create_graph_search_config,
    create_hybrid_search_config,
    create_domain_specific_config
)

__all__ = [
    # Original components
    "VectorSearchEngine",
    "GraphTraversalEngine",
    "TraversalConfig",
    "TraversalAlgorithm",
    "EdgeFilterType",
    "TraversalResult",
    "HybridSearchEngine",
    "HybridSearchConfig",
    
    # Advanced search components
    "AdvancedSearchConfig",
    "SearchMethod",
    "RankingStrategy",
    "SemanticSearchConfig",
    "KeywordSearchConfig",
    "GraphSearchConfig",
    "TemporalSearchConfig",
    "AdvancedHybridSearchConfig",
    "PerformanceConfig",
    "DomainConfig",
    "TemporalConfig",
    
    # Integration components
    "SearchEngineIntegrator",
    "create_semantic_search_config",
    "create_graph_search_config",
    "create_hybrid_search_config",
    "create_domain_specific_config"
]