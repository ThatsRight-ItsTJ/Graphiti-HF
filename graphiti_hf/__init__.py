"""
Graphiti-HF: Adapting Graphiti for Hugging Face Ecosystem

This package provides a fork of Graphiti that uses Hugging Face Datasets
instead of traditional graph databases for knowledge graph storage and operations.
"""

__version__ = "0.1.0"
__author__ = "Graphiti-HF Team"
__email__ = "team@graphiti-hf.org"

from .drivers.huggingface_driver import HuggingFaceDriver
from .analysis.community_detector import CommunityDetector, CommunityDetectionConfig, CommunityStats
from .models import (
    CustomTypeManager,
    PersonEntity,
    CompanyEntity,
    ProjectEntity,
    DocumentEntity,
    EventEntity,
    WorksAtEdge,
    CollaboratesOnEdge,
    AuthoredByEdge,
    ParticipatesInEdge,
    RelatedToEdge,
    create_custom_entity,
    create_custom_edge,
    validate_entity_properties,
    validate_edge_properties,
    serialize_custom_type,
    deserialize_custom_type,
    type_converter,
    get_type_manager,
)
from .search import (
    GraphTraversalEngine,
    TraversalConfig,
    TraversalAlgorithm,
    EdgeFilterType,
    TraversalResult,
    HybridSearchEngine,
    HybridSearchConfig,
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
    TemporalConfig,
    SearchEngineIntegrator,
    create_semantic_search_config,
    create_graph_search_config,
    create_hybrid_search_config,
    create_domain_specific_config
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
    
    # Custom types
    "CustomTypeManager",
    "PersonEntity",
    "CompanyEntity",
    "ProjectEntity",
    "DocumentEntity",
    "EventEntity",
    "WorksAtEdge",
    "CollaboratesOnEdge",
    "AuthoredByEdge",
    "ParticipatesInEdge",
    "RelatedToEdge",
    "create_custom_entity",
    "create_custom_edge",
    "validate_entity_properties",
    "validate_edge_properties",
    "serialize_custom_type",
    "deserialize_custom_type",
    "type_converter",
    "get_type_manager",
    
    # Community detection
    "CommunityDetector",
    "CommunityDetectionConfig",
    "CommunityStats",
    
    # Search engines
    "GraphTraversalEngine",
    "TraversalConfig",
    "TraversalAlgorithm",
    "EdgeFilterType",
    "TraversalResult",
    "HybridSearchEngine",
    "HybridSearchConfig",
    
    # Advanced search configuration
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
    "SearchEngineIntegrator",
    "create_semantic_search_config",
    "create_graph_search_config",
    "create_hybrid_search_config",
    "create_domain_specific_config",
    
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