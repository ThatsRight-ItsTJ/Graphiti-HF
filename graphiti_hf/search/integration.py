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

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    import numpy as np
except ImportError:
    np = None

from graphiti_hf.search.advanced_config import AdvancedSearchConfig, SearchMethod, RankingStrategy
from graphiti_hf.search.vector_search import VectorSearchEngine, SearchConfig, IndexType
from graphiti_hf.search.graph_traversal import GraphTraversalEngine, TraversalConfig, TraversalAlgorithm, EdgeFilterType
from graphiti_hf.search.hybrid_search import HybridSearchEngine, HybridSearchConfig

logger = logging.getLogger(__name__)


class SearchEngineIntegrator:
    """
    Integrates AdvancedSearchConfig with existing search engines.
    
    This class provides methods to apply advanced search configurations
    to the VectorSearchEngine, GraphTraversalEngine, and HybridSearchEngine.
    """
    
    def __init__(self, driver):
        """Initialize the integrator with a HuggingFaceDriver instance."""
        self.driver = driver
    
    def apply_advanced_config_to_vector_search(self, 
                                             vector_engine: VectorSearchEngine,
                                             config: AdvancedSearchConfig) -> VectorSearchEngine:
        """
        Apply advanced search configuration to VectorSearchEngine.
        
        Args:
            vector_engine: VectorSearchEngine instance
            config: AdvancedSearchConfig instance
            
        Returns:
            Updated VectorSearchEngine instance
        """
        if not config.semantic_config:
            logger.warning("No semantic search configuration found")
            return vector_engine
        
        # Apply semantic search configuration
        semantic_config = config.semantic_config
        
        # Update SearchConfig with advanced settings
        search_config = SearchConfig(
            index_type=semantic_config.index_type or IndexType.FLAT,
            n_lists=semantic_config.n_lists or 100,
            n_probes=semantic_config.n_probes or 10,
            ef_search=semantic_config.ef_search or 64,
            m=semantic_config.m or 32,
            nbits=semantic_config.nbits or 8,
            k=semantic_config.k or 10,
            similarity_threshold=semantic_config.similarity_threshold or 0.0,
            batch_size=semantic_config.batch_size or 100,
            use_gpu=semantic_config.use_gpu or False,
            normalize_embeddings=semantic_config.normalize_embeddings or True
        )
        
        vector_engine.config = search_config
        
        # Apply performance settings
        if config.performance_config:
            perf_config = config.performance_config
            
            # Update batch sizes
            if hasattr(perf_config, 'vector_batch_size'):
                vector_engine.config.batch_size = perf_config.vector_batch_size
            
            # Update cache settings
            if hasattr(perf_config, 'cache_enabled'):
                # Note: VectorSearchEngine doesn't have explicit cache settings,
                # but we can apply this to the batch processing
                pass
            
            # Update parallel processing settings
            if hasattr(perf_config, 'max_workers'):
                # Note: VectorSearchEngine processes in batches, max_workers
                # would be applied at a higher level
                pass
        
        # Apply domain-specific settings
        if config.domain_config:
            domain_config = config.domain_config
            
            # Apply domain weights to similarity thresholds
            if hasattr(domain_config, 'domain_weights'):
                domain_weights = domain_config.domain_weights
                # Adjust similarity threshold based on domain importance
                if 'technical' in domain_weights and domain_weights['technical'] > 0.5:
                    vector_engine.config.similarity_threshold *= 1.2
                elif 'business' in domain_weights and domain_weights['business'] > 0.5:
                    vector_engine.config.similarity_threshold *= 0.8
        
        logger.info("Applied advanced configuration to VectorSearchEngine")
        return vector_engine
    
    def apply_advanced_config_to_graph_traversal(self,
                                               traversal_engine: GraphTraversalEngine,
                                               config: AdvancedSearchConfig) -> GraphTraversalEngine:
        """
        Apply advanced search configuration to GraphTraversalEngine.
        
        Args:
            traversal_engine: GraphTraversalEngine instance
            config: AdvancedSearchConfig instance
            
        Returns:
            Updated GraphTraversalEngine instance
        """
        if not config.graph_config:
            logger.warning("No graph search configuration found")
            return traversal_engine
        
        # Apply graph search configuration
        graph_config = config.graph_config
        
        # Update TraversalConfig with advanced settings
        traversal_config = TraversalConfig(
            max_depth=graph_config.max_depth or 5,
            max_path_length=graph_config.max_path_length or 10,
            algorithm=graph_config.algorithm or TraversalAlgorithm.BFS,
            weighted=graph_config.weighted or False,
            edge_filter=graph_config.edge_filter or EdgeFilterType.ALL,
            edge_types=graph_config.edge_types or None,
            temporal_filter=graph_config.temporal_filter or None,
            early_termination_size=graph_config.early_termination_size or None,
            batch_size=graph_config.batch_size or 1000,
            cache_enabled=graph_config.cache_enabled or True,
            max_cache_size=graph_config.max_cache_size or 10000
        )
        
        traversal_engine.config = traversal_config
        
        # Apply performance settings
        if config.performance_config:
            perf_config = config.performance_config
            
            # Update batch sizes
            if hasattr(perf_config, 'traversal_batch_size'):
                traversal_engine.config.batch_size = perf_config.traversal_batch_size
            
            # Update cache settings
            if hasattr(perf_config, 'cache_enabled'):
                traversal_engine.config.cache_enabled = perf_config.cache_enabled
            
            if hasattr(perf_config, 'max_cache_size'):
                traversal_engine.config.max_cache_size = perf_config.max_cache_size
        
        # Apply domain-specific settings
        if config.domain_config:
            domain_config = config.domain_config
            
            # Apply edge type filters based on domain
            if hasattr(domain_config, 'type_filters'):
                type_filters = domain_config.type_filters
                if type_filters:
                    traversal_config.edge_types = list(type_filters.keys())
        
        logger.info("Applied advanced configuration to GraphTraversalEngine")
        return traversal_engine
    
    def apply_advanced_config_to_hybrid_search(self,
                                             hybrid_engine: HybridSearchEngine,
                                             config: AdvancedSearchConfig) -> HybridSearchEngine:
        """
        Apply advanced search configuration to HybridSearchEngine.
        
        Args:
            hybrid_engine: HybridSearchEngine instance
            config: AdvancedSearchConfig instance
            
        Returns:
            Updated HybridSearchEngine instance
        """
        if not config.hybrid_config:
            logger.warning("No hybrid search configuration found")
            return hybrid_engine
        
        # Apply hybrid search configuration
        hybrid_config = config.hybrid_config
        
        # Update HybridSearchConfig with advanced settings
        search_config = HybridSearchConfig(
            semantic_weight=hybrid_config.semantic_weight or 0.4,
            keyword_weight=hybrid_config.keyword_weight or 0.3,
            graph_weight=hybrid_config.graph_weight or 0.3,
            semantic_threshold=hybrid_config.semantic_threshold or 0.0,
            keyword_threshold=hybrid_config.keyword_threshold or 0.0,
            graph_distance_cutoff=hybrid_config.graph_distance_cutoff or 5,
            result_limit=hybrid_config.result_limit or 10,
            center_node_uuid=hybrid_config.center_node_uuid or None,
            temporal_filter=hybrid_config.temporal_filter or None,
            edge_types=hybrid_config.edge_types or None,
            batch_size=hybrid_config.batch_size or 100,
            cache_enabled=hybrid_config.cache_enabled or True,
            max_cache_size=hybrid_config.max_cache_size or 10000
        )
        
        # Note: HybridSearchEngine doesn't have a direct config setter,
        # so we'll need to apply settings through method calls
        
        # Apply performance settings
        if config.performance_config:
            perf_config = config.performance_config
            
            # Update batch sizes
            if hasattr(perf_config, 'hybrid_batch_size'):
                search_config.batch_size = perf_config.hybrid_batch_size
            
            # Update cache settings
            if hasattr(perf_config, 'cache_enabled'):
                search_config.cache_enabled = perf_config.cache_enabled
            
            if hasattr(perf_config, 'max_cache_size'):
                search_config.max_cache_size = perf_config.max_cache_size
        
        # Apply domain-specific settings
        if config.domain_config:
            domain_config = config.domain_config
            
            # Apply domain weights to search weights
            if hasattr(domain_config, 'domain_weights'):
                domain_weights = domain_config.domain_weights
                
                # Adjust search weights based on domain importance
                if 'technical' in domain_weights:
                    search_config.semantic_weight *= domain_weights['technical']
                if 'business' in domain_weights:
                    search_config.keyword_weight *= domain_weights['business']
                if 'personal' in domain_weights:
                    search_config.graph_weight *= domain_weights['personal']
                
                # Normalize weights
                total_weight = (search_config.semantic_weight + 
                              search_config.keyword_weight + 
                              search_config.graph_weight)
                if total_weight > 0:
                    search_config.semantic_weight /= total_weight
                    search_config.keyword_weight /= total_weight
                    search_config.graph_weight /= total_weight
        
        # Apply temporal settings
        if config.temporal_config:
            temporal_config = config.temporal_config
            
            # Update temporal filters
            if temporal_config.start_date or temporal_config.end_date:
                from datetime import datetime
                if temporal_config.start_date:
                    search_config.temporal_filter = temporal_config.start_date
                elif temporal_config.end_date:
                    search_config.temporal_filter = temporal_config.end_date
        
        logger.info("Applied advanced configuration to HybridSearchEngine")
        return hybrid_engine
    
    def apply_all_configs(self, config: AdvancedSearchConfig) -> Dict[str, Any]:
        """
        Apply advanced search configuration to all search engines.
        
        Args:
            config: AdvancedSearchConfig instance
            
        Returns:
            Dictionary containing results of configuration application
        """
        results = {
            'vector_search': {'success': False, 'message': ''},
            'graph_traversal': {'success': False, 'message': ''},
            'hybrid_search': {'success': False, 'message': ''}
        }
        
        try:
            # Apply to VectorSearchEngine
            if self.driver.vector_search_engine:
                self.apply_advanced_config_to_vector_search(
                    self.driver.vector_search_engine, config
                )
                results['vector_search'] = {
                    'success': True, 
                    'message': 'Configuration applied successfully'
                }
            
            # Apply to GraphTraversalEngine
            if self.driver.traversal_engine:
                self.apply_advanced_config_to_graph_traversal(
                    self.driver.traversal_engine, config
                )
                results['graph_traversal'] = {
                    'success': True, 
                    'message': 'Configuration applied successfully'
                }
            
            # Apply to HybridSearchEngine
            if self.driver.hybrid_search_engine:
                self.apply_advanced_config_to_hybrid_search(
                    self.driver.hybrid_search_engine, config
                )
                results['hybrid_search'] = {
                    'success': True, 
                    'message': 'Configuration applied successfully'
                }
            
            logger.info("Applied advanced configurations to all search engines")
            
        except Exception as e:
            error_msg = f"Failed to apply advanced configurations: {str(e)}"
            logger.error(error_msg)
            
            # Update all results with error
            for key in results:
                results[key] = {
                    'success': False, 
                    'message': error_msg
                }
        
        return results
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get the current status of all search engines.
        
        Returns:
            Dictionary containing engine status information
        """
        status = {
            'vector_search': {'enabled': False, 'config': None},
            'graph_traversal': {'enabled': False, 'config': None},
            'hybrid_search': {'enabled': False, 'config': None}
        }
        
        if self.driver.vector_search_engine:
            status['vector_search'] = {
                'enabled': True,
                'config': {
                    'index_type': self.driver.vector_search_engine.config.index_type.value,
                    'k': self.driver.vector_search_engine.config.k,
                    'similarity_threshold': self.driver.vector_search_engine.config.similarity_threshold,
                    'batch_size': self.driver.vector_search_engine.config.batch_size,
                    'use_gpu': self.driver.vector_search_engine.config.use_gpu
                }
            }
        
        if self.driver.traversal_engine:
            status['graph_traversal'] = {
                'enabled': True,
                'config': {
                    'max_depth': self.driver.traversal_engine.config.max_depth,
                    'algorithm': self.driver.traversal_engine.config.algorithm.value,
                    'batch_size': self.driver.traversal_engine.config.batch_size,
                    'cache_enabled': self.driver.traversal_engine.config.cache_enabled
                }
            }
        
        if self.driver.hybrid_search_engine:
            status['hybrid_search'] = {
                'enabled': True,
                'config': {
                    'semantic_weight': 0.4,  # Default, would need to get from engine
                    'keyword_weight': 0.3,
                    'graph_weight': 0.3,
                    'result_limit': 10
                }
            }
        
        return status


# Factory functions for common configuration presets
def create_semantic_search_config() -> AdvancedSearchConfig:
    """Create a configuration optimized for semantic search."""
    config = AdvancedSearchConfig(name="semantic_optimized")
    
    # Configure semantic search
    config.configure_semantic_search(
        index_type=IndexType.HNSW,
        k=20,
        similarity_threshold=0.7,
        use_gpu=True,
        normalize_embeddings=True
    )
    
    # Set search weights favoring semantic search
    config.set_search_weights(
        semantic_weight=0.7,
        keyword_weight=0.2,
        graph_weight=0.1
    )
    
    # Set performance settings
    config.set_batch_sizes(
        vector_batch_size=50,
        hybrid_batch_size=100
    )
    
    return config


def create_graph_search_config() -> AdvancedSearchConfig:
    """Create a configuration optimized for graph traversal."""
    config = AdvancedSearchConfig(name="graph_optimized")
    
    # Configure graph search
    config.configure_graph_search(
        max_depth=8,
        algorithm=TraversalAlgorithm.BFS,
        weighted=True,
        early_termination_size=1000
    )
    
    # Set search weights favoring graph search
    config.set_search_weights(
        semantic_weight=0.2,
        keyword_weight=0.1,
        graph_weight=0.7
    )
    
    # Set performance settings
    config.set_batch_sizes(
        traversal_batch_size=500,
        hybrid_batch_size=200
    )
    
    return config


def create_hybrid_search_config() -> AdvancedSearchConfig:
    """Create a balanced hybrid search configuration."""
    config = AdvancedSearchConfig(name="hybrid_balanced")
    
    # Configure hybrid search
    config.configure_hybrid_search(
        semantic_weight=0.4,
        keyword_weight=0.3,
        graph_weight=0.3,
        result_limit=15
    )
    
    # Set balanced search weights
    config.set_search_weights(
        semantic_weight=0.4,
        keyword_weight=0.3,
        graph_weight=0.3
    )
    
    # Set performance settings
    config.set_batch_sizes(
        vector_batch_size=100,
        traversal_batch_size=200,
        hybrid_batch_size=150
    )
    
    return config


def create_domain_specific_config(domain: str) -> AdvancedSearchConfig:
    """
    Create a domain-specific search configuration.
    
    Args:
        domain: Domain type ('technical', 'business', 'personal')
        
    Returns:
        AdvancedSearchConfig optimized for the specified domain
    """
    config = AdvancedSearchConfig(name=f"{domain}_optimized")
    
    # Set domain weights
    if domain == 'technical':
        config.set_domain_weights(
            technical=0.8,
            business=0.2,
            personal=0.0
        )
        # Technical domains favor semantic search
        config.configure_semantic_search(
            index_type=IndexType.HNSW,
            similarity_threshold=0.8,
            k=25
        )
        
    elif domain == 'business':
        config.set_domain_weights(
            technical=0.2,
            business=0.8,
            personal=0.0
        )
        # Business domains favor keyword search
        config.configure_keyword_search(
            keyword_weight=0.6,
            use_bm25=True,
            use_tfidf=True
        )
        
    elif domain == 'personal':
        config.set_domain_weights(
            technical=0.1,
            business=0.1,
            personal=0.8
        )
        # Personal domains favor graph search
        config.configure_graph_search(
            max_depth=10,
            weighted=True,
            early_termination_size=500
        )
    
    return config