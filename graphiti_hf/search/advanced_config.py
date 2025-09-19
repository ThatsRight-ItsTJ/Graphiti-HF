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

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import numpy as np
from pathlib import Path

from graphiti_hf.search.vector_search import SearchConfig, IndexType
from graphiti_hf.search.graph_traversal import TraversalConfig, TraversalAlgorithm
from graphiti_hf.search.hybrid_search import HybridSearchConfig

logger = logging.getLogger(__name__)


class SearchMethod(Enum):
    """Supported search methods"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    GRAPH = "graph"
    TEMPORAL = "temporal"
    HYBRID = "hybrid"


class RankingStrategy(Enum):
    """Supported ranking strategies"""
    WEIGHTED_SUM = "weighted_sum"
    RANK_AGGREGATE = "rank_aggregate"
    LEARNING_TO_RANK = "learning_to_rank"
    HYBRID_RANKING = "hybrid_ranking"


class TemporalFilterType(Enum):
    """Supported temporal filter types"""
    RANGE = "range"
    SINCE = "since"
    UNTIL = "until"
    BETWEEN = "between"
    RECENT = "recent"


@dataclass
class DomainWeights:
    """Configuration for domain-specific search weights"""
    technical: float = 0.3
    business: float = 0.3
    personal: float = 0.2
    temporal: float = 0.2
    
    def normalize(self) -> 'DomainWeights':
        """Normalize weights to sum to 1.0"""
        total = sum([self.technical, self.business, self.personal, self.temporal])
        if total > 0:
            return DomainWeights(
                technical=self.technical / total,
                business=self.business / total,
                personal=self.personal / total,
                temporal=self.temporal / total
            )
        return DomainWeights()


@dataclass
class TypeFilterConfig:
    """Configuration for type-based filtering"""
    include_types: List[str] = field(default_factory=list)
    exclude_types: List[str] = field(default_factory=list)
    type_weights: Dict[str, float] = field(default_factory=dict)
    
    def is_type_allowed(self, entity_type: str) -> bool:
        """Check if entity type is allowed"""
        if self.exclude_types and entity_type in self.exclude_types:
            return False
        if self.include_types and entity_type not in self.include_types:
            return False
        return True


@dataclass
class GroupFilterConfig:
    """Configuration for group-based filtering"""
    include_groups: List[str] = field(default_factory=list)
    exclude_groups: List[str] = field(default_factory=list)
    group_weights: Dict[str, float] = field(default_factory=dict)
    
    def is_group_allowed(self, group_id: str) -> bool:
        """Check if group is allowed"""
        if self.exclude_groups and group_id in self.exclude_groups:
            return False
        if self.include_groups and group_id not in self.include_groups:
            return False
        return True


@dataclass
class CustomScoringFunction:
    """Configuration for custom scoring functions"""
    name: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    enabled: bool = True


@dataclass
class RelevanceTuning:
    """Configuration for relevance parameter tuning"""
    precision_weight: float = 0.5
    recall_weight: float = 0.3
    diversity_weight: float = 0.2
    freshness_weight: float = 0.0
    learning_rate: float = 0.01
    max_iterations: int = 100
    convergence_threshold: float = 0.001


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing"""
    batch_size: int = 100
    max_workers: int = 4
    timeout: int = 300
    retry_attempts: int = 3
    memory_limit_mb: int = 1024


@dataclass
class CacheConfig:
    """Configuration for search caching"""
    enabled: bool = True
    max_size: int = 10000
    ttl_seconds: int = 3600
    eviction_policy: str = "lru"
    compression: bool = True


@dataclass
class IndexOptimizationConfig:
    """Configuration for index optimization"""
    auto_optimize: bool = True
    optimization_interval: int = 86400  # 24 hours
    threshold_size_change: float = 0.1  # 10% change triggers optimization
    index_types: List[str] = field(default_factory=lambda: ["faiss", "bm25", "graph"])
    compression_enabled: bool = True
    shard_size: int = 10000


@dataclass
class ParallelProcessingConfig:
    """Configuration for parallel processing"""
    enabled: bool = True
    max_concurrent_queries: int = 10
    query_timeout: int = 60
    result_timeout: int = 30
    load_balancing: str = "round_robin"
    worker_affinity: bool = False


@dataclass
class MemoryConfig:
    """Configuration for memory usage limits"""
    max_memory_mb: int = 8192
    cache_memory_mb: int = 2048
    index_memory_mb: int = 4096
    query_memory_mb: int = 1024
    garbage_collection_interval: int = 300
    memory_monitoring: bool = True


class AdvancedSearchConfig:
    """
    Advanced search configuration system for Graphiti-HF.
    
    Provides comprehensive configuration capabilities for all search methods,
    performance optimization, and domain-specific tuning.
    """
    
    def __init__(self):
        # Search method configurations
        self.semantic_config = SearchConfig()
        self.keyword_config = SearchConfig()
        self.graph_config = TraversalConfig()
        self.temporal_config = {}  # Will be populated with temporal filter settings
        self.hybrid_config = HybridSearchConfig()
        
        # Search strategy configuration
        self.search_weights = {
            SearchMethod.SEMANTIC: 0.4,
            SearchMethod.KEYWORD: 0.3,
            SearchMethod.GRAPH: 0.2,
            SearchMethod.TEMPORAL: 0.1
        }
        self.similarity_thresholds = {
            SearchMethod.SEMANTIC: 0.7,
            SearchMethod.KEYWORD: 0.5,
            SearchMethod.GRAPH: 0.0,
            SearchMethod.TEMPORAL: 0.0
        }
        self.result_ranking = RankingStrategy.WEIGHTED_SUM
        self.search_depth = 3
        self.temporal_filters = {}
        
        # Performance configuration
        self.batch_config = BatchProcessingConfig()
        self.cache_config = CacheConfig()
        self.index_config = IndexOptimizationConfig()
        self.parallel_config = ParallelProcessingConfig()
        self.memory_config = MemoryConfig()
        
        # Domain-specific configuration
        self.domain_weights = DomainWeights()
        self.type_filters = TypeFilterConfig()
        self.group_filters = GroupFilterConfig()
        self.custom_scoring_functions: List[CustomScoringFunction] = []
        self.relevance_tuning = RelevanceTuning()
        
        # Configuration metadata
        self.created_at = datetime.now()
        self.last_modified = datetime.now()
        self.version = "1.0.0"
        self.name = "default"
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
    
    # 1. Advanced Search Method Configuration
    
    def configure_semantic_search(self, **kwargs) -> 'AdvancedSearchConfig':
        """
        Configure semantic search parameters.
        
        Args:
            **kwargs: Semantic search configuration parameters
            
        Returns:
            Self for method chaining
        """
        valid_params = [
            'index_type', 'n_lists', 'n_probes', 'ef_search', 'm', 'nbits',
            'k', 'similarity_threshold', 'batch_size', 'use_gpu', 'normalize_embeddings'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params and hasattr(self.semantic_config, key):
                setattr(self.semantic_config, key, value)
        
        self.last_modified = datetime.now()
        return self
    
    def configure_keyword_search(self, **kwargs) -> 'AdvancedSearchConfig':
        """
        Configure keyword search parameters.
        
        Args:
            **kwargs: Keyword search configuration parameters
            
        Returns:
            Self for method chaining
        """
        valid_params = [
            'k', 'similarity_threshold', 'batch_size', 'use_gpu', 'normalize_embeddings'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params and hasattr(self.keyword_config, key):
                setattr(self.keyword_config, key, value)
        
        self.last_modified = datetime.now()
        return self
    
    def configure_graph_search(self, **kwargs) -> 'AdvancedSearchConfig':
        """
        Configure graph traversal parameters.
        
        Args:
            **kwargs: Graph search configuration parameters
            
        Returns:
            Self for method chaining
        """
        valid_params = [
            'max_depth', 'max_path_length', 'algorithm', 'weighted', 'edge_filter',
            'edge_types', 'temporal_filter', 'early_termination_size', 'batch_size',
            'cache_enabled', 'max_cache_size'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params and hasattr(self.graph_config, key):
                setattr(self.graph_config, key, value)
        
        self.last_modified = datetime.now()
        return self
    
    def configure_temporal_search(self, **kwargs) -> 'AdvancedSearchConfig':
        """
        Configure temporal search parameters.
        
        Args:
            **kwargs: Temporal search configuration parameters
            
        Returns:
            Self for method chaining
        """
        valid_params = [
            'filter_type', 'start_time', 'end_time', 'time_range',
            'include_expired', 'include_future', 'time_weight'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params:
                self.temporal_filters[key] = value
        
        self.last_modified = datetime.now()
        return self
    
    def configure_hybrid_search(self, **kwargs) -> 'AdvancedSearchConfig':
        """
        Configure hybrid search parameters.
        
        Args:
            **kwargs: Hybrid search configuration parameters
            
        Returns:
            Self for method chaining
        """
        valid_params = [
            'semantic_weight', 'keyword_weight', 'graph_weight',
            'semantic_threshold', 'keyword_threshold', 'graph_distance_cutoff',
            'result_limit', 'center_node_uuid', 'temporal_filter',
            'edge_types', 'batch_size', 'cache_enabled', 'max_cache_size'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params and hasattr(self.hybrid_config, key):
                setattr(self.hybrid_config, key, value)
        
        self.last_modified = datetime.now()
        return self
    
    # 2. Search Strategy Configuration
    
    def set_search_weights(self, weights: Dict[SearchMethod, float]) -> 'AdvancedSearchConfig':
        """
        Configure search method weights.
        
        Args:
            weights: Dictionary mapping search methods to weights
            
        Returns:
            Self for method chaining
        """
        total_weight = sum(weights.values())
        if total_weight > 0:
            # Normalize weights
            for method, weight in weights.items():
                if method in self.search_weights:
                    self.search_weights[method] = weight / total_weight
        
        self.last_modified = datetime.now()
        return self
    
    def set_similarity_thresholds(self, thresholds: Dict[SearchMethod, float]) -> 'AdvancedSearchConfig':
        """
        Set similarity thresholds for different search methods.
        
        Args:
            thresholds: Dictionary mapping search methods to threshold values
            
        Returns:
            Self for method chaining
        """
        for method, threshold in thresholds.items():
            if method in self.similarity_thresholds:
                self.similarity_thresholds[method] = max(0.0, min(1.0, threshold))
        
        self.last_modified = datetime.now()
        return self
    
    def set_result_ranking(self, strategy: Union[str, RankingStrategy]) -> 'AdvancedSearchConfig':
        """
        Configure result ranking strategy.
        
        Args:
            strategy: Ranking strategy name or enum
            
        Returns:
            Self for method chaining
        """
        if isinstance(strategy, str):
            strategy = RankingStrategy(strategy)
        self.result_ranking = strategy
        
        self.last_modified = datetime.now()
        return self
    
    def set_search_depth(self, depth: int) -> 'AdvancedSearchConfig':
        """
        Configure graph traversal depth.
        
        Args:
            depth: Maximum traversal depth
            
        Returns:
            Self for method chaining
        """
        self.search_depth = max(1, min(10, depth))
        self.graph_config.max_depth = self.search_depth
        
        self.last_modified = datetime.now()
        return self
    
    def set_temporal_filters(self, filters: Dict[str, Any]) -> 'AdvancedSearchConfig':
        """
        Configure temporal filters.
        
        Args:
            filters: Dictionary of temporal filter configurations
            
        Returns:
            Self for method chaining
        """
        self.temporal_filters.update(filters)
        
        self.last_modified = datetime.now()
        return self
    
    # 3. Performance Configuration
    
    def set_batch_sizes(self, **kwargs) -> 'AdvancedSearchConfig':
        """
        Configure batch processing sizes.
        
        Args:
            **kwargs: Batch processing configuration parameters
            
        Returns:
            Self for method chaining
        """
        valid_params = [
            'batch_size', 'max_workers', 'timeout', 'retry_attempts', 'memory_limit_mb'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params and hasattr(self.batch_config, key):
                setattr(self.batch_config, key, value)
        
        self.last_modified = datetime.now()
        return self
    
    def set_cache_settings(self, **kwargs) -> 'AdvancedSearchConfig':
        """
        Configure search caching.
        
        Args:
            **kwargs: Cache configuration parameters
            
        Returns:
            Self for method chaining
        """
        valid_params = [
            'enabled', 'max_size', 'ttl_seconds', 'eviction_policy', 'compression'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params and hasattr(self.cache_config, key):
                setattr(self.cache_config, key, value)
        
        self.last_modified = datetime.now()
        return self
    
    def set_index_settings(self, **kwargs) -> 'AdvancedSearchConfig':
        """
        Configure search indices.
        
        Args:
            **kwargs: Index configuration parameters
            
        Returns:
            Self for method chaining
        """
        valid_params = [
            'auto_optimize', 'optimization_interval', 'threshold_size_change',
            'index_types', 'compression_enabled', 'shard_size'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params and hasattr(self.index_config, key):
                setattr(self.index_config, key, value)
        
        self.last_modified = datetime.now()
        return self
    
    def set_parallel_settings(self, **kwargs) -> 'AdvancedSearchConfig':
        """
        Configure parallel processing.
        
        Args:
            **kwargs: Parallel processing configuration parameters
            
        Returns:
            Self for method chaining
        """
        valid_params = [
            'enabled', 'max_concurrent_queries', 'query_timeout',
            'result_timeout', 'load_balancing', 'worker_affinity'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params and hasattr(self.parallel_config, key):
                setattr(self.parallel_config, key, value)
        
        self.last_modified = datetime.now()
        return self
    
    def set_memory_limits(self, **kwargs) -> 'AdvancedSearchConfig':
        """
        Configure memory usage limits.
        
        Args:
            **kwargs: Memory configuration parameters
            
        Returns:
            Self for method chaining
        """
        valid_params = [
            'max_memory_mb', 'cache_memory_mb', 'index_memory_mb',
            'query_memory_mb', 'garbage_collection_interval', 'memory_monitoring'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params and hasattr(self.memory_config, key):
                setattr(self.memory_config, key, value)
        
        self.last_modified = datetime.now()
        return self
    
    # 4. Domain-Specific Configuration
    
    def set_domain_weights(self, weights: Dict[str, float]) -> 'AdvancedSearchConfig':
        """
        Configure domain-specific weights.
        
        Args:
            weights: Dictionary mapping domain names to weights
            
        Returns:
            Self for method chaining
        """
        domain_mapping = {
            'technical': 'technical',
            'business': 'business', 
            'personal': 'personal',
            'temporal': 'temporal'
        }
        
        for domain_name, weight in weights.items():
            if domain_name in domain_mapping:
                setattr(self.domain_weights, domain_mapping[domain_name], weight)
        
        # Normalize weights
        self.domain_weights = self.domain_weights.normalize()
        
        self.last_modified = datetime.now()
        return self
    
    def set_type_filters(self, **kwargs) -> 'AdvancedSearchConfig':
        """
        Configure type-based filtering.
        
        Args:
            **kwargs: Type filter configuration parameters
            
        Returns:
            Self for method chaining
        """
        valid_params = [
            'include_types', 'exclude_types', 'type_weights'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params and hasattr(self.type_filters, key):
                setattr(self.type_filters, key, value)
        
        self.last_modified = datetime.now()
        return self
    
    def set_group_filters(self, **kwargs) -> 'AdvancedSearchConfig':
        """
        Configure group-based filtering.
        
        Args:
            **kwargs: Group filter configuration parameters
            
        Returns:
            Self for method chaining
        """
        valid_params = [
            'include_groups', 'exclude_groups', 'group_weights'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params and hasattr(self.group_filters, key):
                setattr(self.group_filters, key, value)
        
        self.last_modified = datetime.now()
        return self
    
    def set_custom_scoring(self, scoring_functions: List[CustomScoringFunction]) -> 'AdvancedSearchConfig':
        """
        Configure custom scoring functions.
        
        Args:
            scoring_functions: List of custom scoring functions
            
        Returns:
            Self for method chaining
        """
        self.custom_scoring_functions = scoring_functions
        
        self.last_modified = datetime.now()
        return self
    
    def add_custom_scoring_function(self, name: str, function: Callable, 
                                  weight: float = 1.0, parameters: Optional[Dict[str, Any]] = None) -> 'AdvancedSearchConfig':
        """
        Add a custom scoring function.
        
        Args:
            name: Name of the scoring function
            function: Scoring function
            weight: Weight for this scoring function
            parameters: Additional parameters
            
        Returns:
            Self for method chaining
        """
        scoring_func = CustomScoringFunction(
            name=name,
            function=function,
            weight=weight,
            parameters=parameters if parameters is not None else {}
        )
        self.custom_scoring_functions.append(scoring_func)
        
        self.last_modified = datetime.now()
        return self
    
    def set_relevance_tuning(self, **kwargs) -> 'AdvancedSearchConfig':
        """
        Configure relevance parameters.
        
        Args:
            **kwargs: Relevance tuning configuration parameters
            
        Returns:
            Self for method chaining
        """
        valid_params = [
            'precision_weight', 'recall_weight', 'diversity_weight',
            'freshness_weight', 'learning_rate', 'max_iterations',
            'convergence_threshold'
        ]
        
        for key, value in kwargs.items():
            if key in valid_params and hasattr(self.relevance_tuning, key):
                setattr(self.relevance_tuning, key, value)
        
        self.last_modified = datetime.now()
        return self
    
    # 5. Configuration Management
    
    def validate(self) -> List[str]:
        """
        Validate the configuration and return any validation errors.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate search weights
        total_weight = sum(self.search_weights.values())
        if not np.isclose(total_weight, 1.0, atol=0.01):
            errors.append(f"Search weights must sum to 1.0, current sum: {total_weight}")
        
        # Validate similarity thresholds
        for method, threshold in self.similarity_thresholds.items():
            if not (0.0 <= threshold <= 1.0):
                errors.append(f"Similarity threshold for {method.value} must be between 0.0 and 1.0")
        
        # Validate search depth
        if not (1 <= self.search_depth <= 10):
            errors.append("Search depth must be between 1 and 10")
        
        # Validate batch sizes
        if self.batch_config.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        # Validate cache settings
        if self.cache_config.max_size <= 0:
            errors.append("Cache max size must be positive")
        
        # Validate memory limits
        total_memory = (
            self.memory_config.cache_memory_mb + 
            self.memory_config.index_memory_mb + 
            self.memory_config.query_memory_mb
        )
        if total_memory > self.memory_config.max_memory_mb:
            errors.append("Total memory allocation exceeds maximum memory limit")
        
        return errors
    
    def optimize(self) -> 'AdvancedSearchConfig':
        """
        Automatically optimize configuration based on current settings.
        
        Returns:
            Self for method chaining
        """
        optimization_log = {
            'timestamp': datetime.now().isoformat(),
            'optimizations_applied': []
        }
        
        # Optimize search weights based on domain
        if self.domain_weights.technical > 0.5:
            # Technical domain favors semantic search
            self.search_weights[SearchMethod.SEMANTIC] = 0.6
            self.search_weights[SearchMethod.KEYWORD] = 0.2
            self.search_weights[SearchMethod.GRAPH] = 0.2
            optimization_log['optimizations_applied'].append({
                'type': 'domain_weight_optimization',
                'domain': 'technical',
                'new_weights': dict(self.search_weights)
            })
        
        # Optimize batch size based on memory limits
        optimal_batch_size = min(
            self.batch_config.batch_size,
            self.memory_config.query_memory_mb * 1024 * 1024 // 1024  # rough estimate
        )
        if optimal_batch_size != self.batch_config.batch_size:
            self.batch_config.batch_size = optimal_batch_size
            optimization_log['optimizations_applied'].append({
                'type': 'batch_size_optimization',
                'old_size': self.batch_config.batch_size,
                'new_size': optimal_batch_size
            })
        
        # Optimize cache settings
        if self.cache_config.enabled and self.cache_config.max_size > 5000:
            self.cache_config.max_size = min(self.cache_config.max_size, 20000)
            optimization_log['optimizations_applied'].append({
                'type': 'cache_size_optimization',
                'new_max_size': self.cache_config.max_size
            })
        
        # Log optimization
        if optimization_log['optimizations_applied']:
            self.optimization_history.append(optimization_log)
            logger.info(f"Applied {len(optimization_log['optimizations_applied'])} optimizations")
        
        self.last_modified = datetime.now()
        return self
    
    def save(self, filepath: str) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save the configuration
        """
        config_dict = self.to_dict()
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AdvancedSearchConfig':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            Loaded AdvancedSearchConfig instance
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Create instance and restore configuration
        config = cls()
        config.from_dict(config_dict)
        
        logger.info(f"Configuration loaded from {filepath}")
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'name': self.name,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'semantic_config': self.semantic_config.__dict__,
            'keyword_config': self.keyword_config.__dict__,
            'graph_config': self.graph_config.__dict__,
            'hybrid_config': self.hybrid_config.__dict__,
            'search_weights': {method.value: weight for method, weight in self.search_weights.items()},
            'similarity_thresholds': {method.value: threshold for method, threshold in self.similarity_thresholds.items()},
            'result_ranking': self.result_ranking.value,
            'search_depth': self.search_depth,
            'temporal_filters': self.temporal_filters,
            'batch_config': self.batch_config.__dict__,
            'cache_config': self.cache_config.__dict__,
            'index_config': self.index_config.__dict__,
            'parallel_config': self.parallel_config.__dict__,
            'memory_config': self.memory_config.__dict__,
            'domain_weights': self.domain_weights.__dict__,
            'type_filters': {
                'include_types': self.type_filters.include_types,
                'exclude_types': self.type_filters.exclude_types,
                'type_weights': self.type_filters.type_weights
            },
            'group_filters': {
                'include_groups': self.group_filters.include_groups,
                'exclude_groups': self.group_filters.exclude_groups,
                'group_weights': self.group_filters.group_weights
            },
            'custom_scoring_functions': [
                {
                    'name': func.name,
                    'weight': func.weight,
                    'enabled': func.enabled,
                    'parameters': func.parameters
                }
                for func in self.custom_scoring_functions
            ],
            'relevance_tuning': self.relevance_tuning.__dict__,
            'optimization_history': self.optimization_history
        }
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Dictionary representation of the configuration
        """
        self.name = config_dict.get('name', 'default')
        self.version = config_dict.get('version', '1.0.0')
        self.created_at = datetime.fromisoformat(config_dict.get('created_at', datetime.now().isoformat()))
        self.last_modified = datetime.fromisoformat(config_dict.get('last_modified', datetime.now().isoformat()))
        
        # Restore search configurations
        if 'semantic_config' in config_dict:
            for key, value in config_dict['semantic_config'].items():
                if hasattr(self.semantic_config, key):
                    setattr(self.semantic_config, key, value)
        
        if 'keyword_config' in config_dict:
            for key, value in config_dict['keyword_config'].items():
                if hasattr(self.keyword_config, key):
                    setattr(self.keyword_config, key, value)
        
        if 'graph_config' in config_dict:
            for key, value in config_dict['graph_config'].items():
                if hasattr(self.graph_config, key):
                    setattr(self.graph_config, key, value)
        
        if 'hybrid_config' in config_dict:
            for key, value in config_dict['hybrid_config'].items():
                if hasattr(self.hybrid_config, key):
                    setattr(self.hybrid_config, key, value)
        
        # Restore search weights and thresholds
        if 'search_weights' in config_dict:
            self.search_weights = {
                SearchMethod(method): weight for method, weight in config_dict['search_weights'].items()
            }
        
        if 'similarity_thresholds' in config_dict:
            self.similarity_thresholds = {
                SearchMethod(method): threshold for method, threshold in config_dict['similarity_thresholds'].items()
            }
        
        # Restore other configurations
        self.result_ranking = RankingStrategy(config_dict.get('result_ranking', 'weighted_sum'))
        self.search_depth = config_dict.get('search_depth', 3)
        self.temporal_filters = config_dict.get('temporal_filters', {})
        
        # Restore performance configurations
        if 'batch_config' in config_dict:
            for key, value in config_dict['batch_config'].items():
                if hasattr(self.batch_config, key):
                    setattr(self.batch_config, key, value)
        
        if 'cache_config' in config_dict:
            for key, value in config_dict['cache_config'].items():
                if hasattr(self.cache_config, key):
                    setattr(self.cache_config, key, value)
        
        if 'index_config' in config_dict:
            for key, value in config_dict['index_config'].items():
                if hasattr(self.index_config, key):
                    setattr(self.index_config, key, value)
        
        if 'parallel_config' in config_dict:
            for key, value in config_dict['parallel_config'].items():
                if hasattr(self.parallel_config, key):
                    setattr(self.parallel_config, key, value)
        
        if 'memory_config' in config_dict:
            for key, value in config_dict['memory_config'].items():
                if hasattr(self.memory_config, key):
                    setattr(self.memory_config, key, value)
        
        # Restore domain-specific configurations
        if 'domain_weights' in config_dict:
            for key, value in config_dict['domain_weights'].items():
                if hasattr(self.domain_weights, key):
                    setattr(self.domain_weights, key, value)
        
        if 'type_filters' in config_dict:
            type_filter_data = config_dict['type_filters']
            self.type_filters.include_types = type_filter_data.get('include_types', [])
            self.type_filters.exclude_types = type_filter_data.get('exclude_types', [])
            self.type_filters.type_weights = type_filter_data.get('type_weights', {})
        
        if 'group_filters' in config_dict:
            group_filter_data = config_dict['group_filters']
            self.group_filters.include_groups = group_filter_data.get('include_groups', [])
            self.group_filters.exclude_groups = group_filter_data.get('exclude_groups', [])
            self.group_filters.group_weights = group_filter_data.get('group_weights', {})
        
        # Restore custom scoring functions
        if 'custom_scoring_functions' in config_dict:
            self.custom_scoring_functions = []
            for func_data in config_dict['custom_scoring_functions']:
                scoring_func = CustomScoringFunction(
                    name=func_data['name'],
                    function=lambda x: x,  # Placeholder function
                    weight=func_data.get('weight', 1.0),
                    parameters=func_data.get('parameters', {})
                )
                scoring_func.enabled = func_data.get('enabled', True)
                self.custom_scoring_functions.append(scoring_func)
        
        # Restore relevance tuning
        if 'relevance_tuning' in config_dict:
            for key, value in config_dict['relevance_tuning'].items():
                if hasattr(self.relevance_tuning, key):
                    setattr(self.relevance_tuning, key, value)
        
        # Restore optimization history
        self.optimization_history = config_dict.get('optimization_history', [])
    
    def copy(self) -> 'AdvancedSearchConfig':
        """
        Create a copy of the configuration.
        
        Returns:
            New AdvancedSearchConfig instance with the same settings
        """
        new_config = AdvancedSearchConfig()
        new_config.from_dict(self.to_dict())
        return new_config
    
    def __str__(self) -> str:
        """String representation of the configuration"""
        return f"AdvancedSearchConfig(name='{self.name}', version='{self.version}', last_modified='{self.last_modified}')"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"AdvancedSearchConfig("
                f"name='{self.name}', "
                f"version='{self.version}', "
                f"search_weights={dict(self.search_weights)}, "
                f"search_depth={self.search_depth})")


# Factory functions for common configuration presets

def create_technical_domain_config() -> AdvancedSearchConfig:
    """Create a configuration optimized for technical domains"""
    config = AdvancedSearchConfig()
    config.set_domain_weights({
        'technical': 0.6,
        'business': 0.2,
        'personal': 0.1,
        'temporal': 0.1
    })
    config.set_search_weights({
        SearchMethod.SEMANTIC: 0.5,
        SearchMethod.KEYWORD: 0.2,
        SearchMethod.GRAPH: 0.2,
        SearchMethod.TEMPORAL: 0.1
    })
    config.configure_semantic_search(index_type=IndexType.HNSW, ef_search=128)
    config.name = "technical_domain"
    return config


def create_business_domain_config() -> AdvancedSearchConfig:
    """Create a configuration optimized for business domains"""
    config = AdvancedSearchConfig()
    config.set_domain_weights({
        'technical': 0.2,
        'business': 0.5,
        'personal': 0.2,
        'temporal': 0.1
    })
    config.set_search_weights({
        SearchMethod.SEMANTIC: 0.3,
        SearchMethod.KEYWORD: 0.4,
        SearchMethod.GRAPH: 0.2,
        SearchMethod.TEMPORAL: 0.1
    })
    config.configure_keyword_search(k=20)
    config.name = "business_domain"
    return config


def create_high_performance_config() -> AdvancedSearchConfig:
    """Create a configuration optimized for high performance"""
    config = AdvancedSearchConfig()
    config.set_batch_sizes(batch_size=500, max_workers=8)
    config.set_cache_settings(max_size=50000, ttl_seconds=1800)
    config.set_parallel_settings(max_concurrent_queries=20)
    config.set_memory_limits(max_memory_mb=16384)
    config.name = "high_performance"
    return config


def create_precision_config() -> AdvancedSearchConfig:
    """Create a configuration optimized for precision"""
    config = AdvancedSearchConfig()
    config.set_similarity_thresholds({
        SearchMethod.SEMANTIC: 0.85,
        SearchMethod.KEYWORD: 0.7,
        SearchMethod.GRAPH: 0.0,
        SearchMethod.TEMPORAL: 0.0
    })
    config.set_result_ranking(RankingStrategy.RANK_AGGREGATE)
    config.name = "precision"
    return config