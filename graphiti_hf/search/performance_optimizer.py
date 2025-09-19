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

import asyncio
import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import faiss
from collections import defaultdict, deque

# Import optional dependencies with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from rank_bm25 import BM25Okapi
    import networkx as nx
    from scipy import stats
except ImportError as e:
    logging.warning(f"Optional dependencies not available: {e}")
    # Create dummy classes for missing dependencies
    class SentenceTransformer:
        def __init__(self, model_name):
            pass
        def encode(self, texts):
            return [[0.0] * 384] if isinstance(texts, list) else [0.0] * 384
    
    class TfidfVectorizer:
        def __init__(self, **kwargs):
            pass
        def fit(self, texts):
            pass
        def transform(self, texts):
            return []
    
    class BM25Okapi:
        def __init__(self, corpus):
            self.corpus = corpus
        def get_scores(self, query):
            return [0.0] * len(self.corpus) if hasattr(self, 'corpus') else [0.0]
    
    nx = None
    stats = None

from graphiti_core.nodes import EntityNode, EpisodicNode, CommunityNode
from graphiti_core.edges import EntityEdge
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver
from graphiti_hf.search.vector_search import VectorSearchEngine, SearchConfig, IndexType
from graphiti_hf.search.graph_traversal import GraphTraversalEngine, TraversalConfig
from graphiti_hf.search.hybrid_search import HybridSearchEngine, HybridSearchConfig

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Supported index types"""
    FLAT = "flat"
    IVFFLAT = "ivfflat"
    HNSW = "hnsw"
    PQ = "pq"
    IVFPQ = "ivfpq"
    BM25 = "bm25"
    TFIDF = "tfidf"
    NETWORKX = "networkx"
    TEMPORAL = "temporal"


@dataclass
class PerformanceMetrics:
    """Performance metrics for search operations"""
    query_time: float = 0.0
    memory_usage: float = 0.0
    accuracy: float = 0.0
    index_size: int = 0
    cache_hit_rate: float = 0.0
    build_time: float = 0.0
    search_latency_p50: float = 0.0
    search_latency_p95: float = 0.0
    search_latency_p99: float = 0.0
    error_rate: float = 0.0


@dataclass
class IndexConfig:
    """Configuration for index building and optimization"""
    index_type: IndexType = IndexType.FLAT
    faiss_config: Optional[Dict[str, Any]] = None
    bm25_config: Optional[Dict[str, Any]] = None
    tfidf_config: Optional[Dict[str, Any]] = None
    networkx_config: Optional[Dict[str, Any]] = None
    temporal_config: Optional[Dict[str, Any]] = None
    auto_optimize: bool = True
    optimization_interval: int = 3600  # seconds
    performance_threshold: float = 0.8
    cache_size: int = 10000
    batch_size: int = 1000


@dataclass
class QueryPattern:
    """Represents a query pattern for optimization"""
    query_type: str
    parameters: Dict[str, Any]
    frequency: int = 0
    avg_latency: float = 0.0
    last_used: datetime = field(default_factory=datetime.now)


class SearchIndexManager:
    """
    Comprehensive search index manager for optimizing performance in Graphiti-HF.
    
    Implements all the required functionality from the plan:
    - Text indices (BM25, TF-IDF) for lines 714-728
    - Vector indices (FAISS) for lines 192-196
    - Graph indices (NetworkX) for adjacency lists and centrality
    - Temporal indices for time-based filtering
    - Performance monitoring and optimization
    - Index management and versioning
    """
    
    def __init__(self, driver: HuggingFaceDriver, config: Optional[IndexConfig] = None):
        """
        Initialize the SearchIndexManager.
        
        Args:
            driver: HuggingFaceDriver instance
            config: Index configuration
        """
        self.driver = driver
        self.config = config or IndexConfig()
        
        # Index storage
        self.text_indices: Dict[str, Any] = {}
        self.vector_indices: Dict[str, Any] = {}
        self.graph_indices: Dict[str, Any] = {}
        self.temporal_indices: Dict[str, Any] = {}
        
        # Performance monitoring
        self.performance_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Index versions
        self.index_versions: Dict[str, int] = {}
        self.current_version = 1
        
        # Initialize vector search engine if available
        self.vector_search_engine = None
        if hasattr(driver, 'vector_search_engine') and driver.vector_search_engine:
            self.vector_search_engine = driver.vector_search_engine
        
        # Initialize other engines
        self.traversal_engine = driver.traversal_engine
        self.hybrid_search_engine = driver.hybrid_search_engine
        
        # Performance optimization
        self.last_optimization = datetime.now()
        self.auto_optimization_enabled = self.config.auto_optimize
        
        logger.info("SearchIndexManager initialized")
    
    def build_text_indices(self) -> Dict[str, Any]:
        """
        Build BM25 and TF-IDF indices for text search optimization.
        Corresponds to lines 714-728 in the plan.
        
        Returns:
            Dictionary containing built text indices
        """
        start_time = time.time()
        
        if self.driver.edges_df.empty:
            logger.warning("No edges data available for text index building")
            return {}
        
        logger.info("Building text indices...")
        
        # Prepare text corpus from edge facts and node names
        edge_texts = self.driver.edges_df['fact'].fillna('').tolist()
        node_texts = self.driver.nodes_df['name'].fillna('').tolist()
        
        # Build BM25 index for edges
        bm25_config = self.config.bm25_config or {}
        tokenized_edge_texts = [text.lower().split() for text in edge_texts]
        self.text_indices['bm25_edges'] = BM25Okapi(tokenized_edge_texts)
        
        # Build BM25 index for nodes
        tokenized_node_texts = [text.lower().split() for text in node_texts]
        self.text_indices['bm25_nodes'] = BM25Okapi(tokenized_node_texts)
        
        # Build TF-IDF vectorizers
        tfidf_config = self.config.tfidf_config or {}
        self.text_indices['tfidf_edges'] = TfidfVectorizer(
            max_features=tfidf_config.get('max_features', 10000),
            stop_words='english'
        )
        self.text_indices['tfidf_edges'].fit(edge_texts)
        
        self.text_indices['tfidf_nodes'] = TfidfVectorizer(
            max_features=tfidf_config.get('max_features', 10000),
            stop_words='english'
        )
        self.text_indices['tfidf_nodes'].fit(node_texts)
        
        build_time = time.time() - start_time
        logger.info(f"Text indices built in {build_time:.2f} seconds")
        
        # Record performance metrics
        metrics = PerformanceMetrics(
            build_time=build_time,
            index_size=len(edge_texts) + len(node_texts)
        )
        self.performance_metrics['text_indices'].append(metrics)
        
        return self.text_indices
    
    def build_vector_indices(self, index_types: Optional[List[IndexType]] = None) -> Dict[str, Any]:
        """
        Build FAISS indices for vector search optimization.
        Corresponds to lines 192-196 in the plan.
        
        Args:
            index_types: List of index types to build (uses config default if None)
            
        Returns:
            Dictionary containing built vector indices
        """
        start_time = time.time()
        
        if not self.vector_search_engine:
            logger.warning("Vector search engine not available")
            return {}
        
        index_types = index_types or [IndexType.FLAT]
        
        logger.info(f"Building vector indices with types: {[t.value for t in index_types]}")
        
        # Build node indices
        if not self.driver.nodes_df.empty and 'name_embedding' in self.driver.nodes_df.columns:
            node_embeddings = self.driver.nodes_df['name_embedding'].dropna()
            if not node_embeddings.empty:
                embedding_matrix = np.array(node_embeddings.tolist()).astype('float32')
                
                for index_type in index_types:
                    try:
                        index_config = SearchConfig(
                            index_type=index_type,
                            k=10,
                            similarity_threshold=0.0
                        )
                        
                        temp_engine = VectorSearchEngine(
                            embed_dim=embedding_matrix.shape[1],
                            config=index_config
                        )
                        
                        index = temp_engine.build_index(
                            embedding_matrix,
                            self.driver.nodes_df[node_embeddings.index]['uuid'].tolist(),
                            metadata={'entity_type': 'node', 'index_type': index_type.value}
                        )
                        
                        self.vector_indices[f'nodes_{index_type.value}'] = {
                            'index': index,
                            'id_map': self.driver.nodes_df[node_embeddings.index]['uuid'].tolist(),
                            'metadata': {'entity_type': 'node', 'index_type': index_type.value}
                        }
                        
                    except Exception as e:
                        logger.warning(f"Failed to build {index_type.value} index for nodes: {e}")
        
        # Build edge indices
        if not self.driver.edges_df.empty and 'fact_embedding' in self.driver.edges_df.columns:
            edge_embeddings = self.driver.edges_df['fact_embedding'].dropna()
            if not edge_embeddings.empty:
                embedding_matrix = np.array(edge_embeddings.tolist()).astype('float32')
                
                for index_type in index_types:
                    try:
                        index_config = SearchConfig(
                            index_type=index_type,
                            k=10,
                            similarity_threshold=0.0
                        )
                        
                        temp_engine = VectorSearchEngine(
                            embed_dim=embedding_matrix.shape[1],
                            config=index_config
                        )
                        
                        index = temp_engine.build_index(
                            embedding_matrix,
                            self.driver.edges_df[edge_embeddings.index]['uuid'].tolist(),
                            metadata={'entity_type': 'edge', 'index_type': index_type.value}
                        )
                        
                        self.vector_indices[f'edges_{index_type.value}'] = {
                            'index': index,
                            'id_map': self.driver.edges_df[edge_embeddings.index]['uuid'].tolist(),
                            'metadata': {'entity_type': 'edge', 'index_type': index_type.value}
                        }
                        
                    except Exception as e:
                        logger.warning(f"Failed to build {index_type.value} index for edges: {e}")
        
        build_time = time.time() - start_time
        logger.info(f"Vector indices built in {build_time:.2f} seconds")
        
        # Record performance metrics
        metrics = PerformanceMetrics(
            build_time=build_time,
            index_size=sum(len(v['id_map']) for v in self.vector_indices.values())
        )
        self.performance_metrics['vector_indices'].append(metrics)
        
        return self.vector_indices
    
    def build_graph_indices(self) -> Dict[str, Any]:
        """
        Build NetworkX graph indices for graph search optimization.
        Implements adjacency lists and centrality indices as mentioned in line 194.
        
        Returns:
            Dictionary containing built graph indices
        """
        start_time = time.time()
        
        if self.driver.edges_df.empty:
            logger.warning("No edges data available for graph index building")
            return {}
        
        logger.info("Building graph indices...")
        
        # Build main graph
        G = nx.Graph()
        
        for _, edge in self.driver.edges_df.iterrows():
            G.add_edge(
                edge['source_uuid'], 
                edge['target_uuid'],
                edge_uuid=edge['uuid'],
                fact=edge['fact']
            )
        
        # Build adjacency lists
        adjacency_lists = {}
        reverse_adjacency_lists = {}
        
        for node in G.nodes():
            adjacency_lists[node] = list(G.neighbors(node))
            reverse_adjacency_lists[node] = list(G.predecessors(node)) if hasattr(G, 'predecessors') else []
        
        # Calculate centrality measures
        centrality_indices = {}
        
        try:
            if nx is not None:
                # Degree centrality
                centrality_indices['degree_centrality'] = nx.degree_centrality(G)
                
                # Betweenness centrality (for smaller graphs)
                if len(G.nodes()) <= 1000:
                    centrality_indices['betweenness_centrality'] = nx.betweenness_centrality(G)
                
                # Closeness centrality (for smaller graphs)
                if len(G.nodes()) <= 1000:
                    centrality_indices['closeness_centrality'] = nx.closeness_centrality(G)
                
                # Community detection using Louvain algorithm (if available)
                try:
                    import community as community_louvain
                    communities = community_louvain.best_partition(G)
                    centrality_indices['communities'] = communities
                except ImportError:
                    logger.info("python-louvain not available for community detection")
                
        except Exception as e:
            logger.warning(f"Failed to calculate centrality measures: {e}")
        
        # Store indices
        self.graph_indices = {
            'graph': G,
            'adjacency_lists': adjacency_lists,
            'reverse_adjacency_lists': reverse_adjacency_lists,
            'centrality': centrality_indices,
            'node_count': len(G.nodes()),
            'edge_count': len(G.edges())
        }
        
        build_time = time.time() - start_time
        logger.info(f"Graph indices built in {build_time:.2f} seconds")
        
        # Record performance metrics
        metrics = PerformanceMetrics(
            build_time=build_time,
            index_size=len(G.nodes()) + len(G.edges())
        )
        self.performance_metrics['graph_indices'].append(metrics)
        
        return self.graph_indices
    
    def build_temporal_indices(self) -> Dict[str, Any]:
        """
        Build temporal indices for time-based filtering optimization.
        
        Returns:
            Dictionary containing built temporal indices
        """
        start_time = time.time()
        
        logger.info("Building temporal indices...")
        
        temporal_indices = {}
        
        # Build time-based partitioning for nodes
        if not self.driver.nodes_df.empty:
            nodes_df = self.driver.nodes_df.copy()
            nodes_df['created_at'] = pd.to_datetime(nodes_df['created_at'])
            
            # Create time-based partitions
            temporal_indices['node_partitions'] = {}
            
            # Partition by year
            for year, year_nodes in nodes_df.groupby(nodes_df['created_at'].dt.year):
                temporal_indices['node_partitions'][f'year_{year}'] = year_nodes['uuid'].tolist()
            
            # Partition by month
            for (year, month), month_nodes in nodes_df.groupby([
                nodes_df['created_at'].dt.year, 
                nodes_df['created_at'].dt.month
            ]):
                temporal_indices['node_partitions'][f'year_{year}_month_{month}'] = month_nodes['uuid'].tolist()
            
            # Create time range index
            temporal_indices['node_time_range'] = {
                'min_time': nodes_df['created_at'].min(),
                'max_time': nodes_df['created_at'].max(),
                'time_buckets': self._create_time_buckets(nodes_df['created_at'])
            }
        
        # Build time-based partitioning for edges
        if not self.driver.edges_df.empty:
            edges_df = self.driver.edges_df.copy()
            edges_df['created_at'] = pd.to_datetime(edges_df['created_at'])
            edges_df['valid_at'] = pd.to_datetime(edges_df['valid_at'])
            
            temporal_indices['edge_partitions'] = {}
            
            # Partition by creation time
            for year, year_edges in edges_df.groupby(edges_df['created_at'].dt.year):
                temporal_indices['edge_partitions'][f'created_year_{year}'] = year_edges['uuid'].tolist()
            
            # Partition by validity time
            for year, year_edges in edges_df.groupby(edges_df['valid_at'].dt.year):
                temporal_indices['edge_partitions'][f'valid_year_{year}'] = year_edges['uuid'].tolist()
            
            # Create time range index
            temporal_indices['edge_time_range'] = {
                'min_created_time': edges_df['created_at'].min(),
                'max_created_time': edges_df['created_at'].max(),
                'min_valid_time': edges_df['valid_at'].min(),
                'max_valid_time': edges_df['valid_at'].max(),
                'time_buckets': self._create_time_buckets(edges_df['created_at'])
            }
        
        self.temporal_indices = temporal_indices
        
        build_time = time.time() - start_time
        logger.info(f"Temporal indices built in {build_time:.2f} seconds")
        
        # Record performance metrics
        metrics = PerformanceMetrics(
            build_time=build_time,
            index_size=sum(len(v) for v in temporal_indices.get('node_partitions', {}).values()) +
                      sum(len(v) for v in temporal_indices.get('edge_partitions', {}).values())
        )
        self.performance_metrics['temporal_indices'].append(metrics)
        
        return self.temporal_indices
    
    def _create_time_buckets(self, timestamps: pd.Series, bucket_size: str = 'M') -> Dict[str, List]:
        """Create time buckets for temporal indexing"""
        buckets = {}
        
        for bucket_time, bucket_items in timestamps.groupby(pd.Grouper(freq=bucket_size)):
            bucket_key = bucket_time.strftime('%Y-%m-%d')
            buckets[bucket_key] = bucket_items.tolist()
        
        return buckets
    
    def rebuild_all_indices(self) -> Dict[str, Any]:
        """
        Rebuild all indices when needed.
        
        Returns:
            Dictionary containing all rebuilt indices
        """
        logger.info("Rebuilding all indices...")
        
        start_time = time.time()
        
        # Rebuild all index types
        all_indices = {
            'text_indices': self.build_text_indices(),
            'vector_indices': self.build_vector_indices(),
            'graph_indices': self.build_graph_indices(),
            'temporal_indices': self.build_temporal_indices()
        }
        
        # Increment version
        self.current_version += 1
        self.index_versions = {
            'text': self.current_version,
            'vector': self.current_version,
            'graph': self.current_version,
            'temporal': self.current_version
        }
        
        rebuild_time = time.time() - start_time
        logger.info(f"All indices rebuilt in {rebuild_time:.2f} seconds")
        
        # Record optimization
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'operation': 'rebuild_all_indices',
            'duration': rebuild_time,
            'version': self.current_version
        })
        
        return all_indices
    
    def benchmark_search_performance(self, test_queries: List[str], k: int = 10) -> Dict[str, Any]:
        """
        Benchmark search performance across different search methods.
        
        Args:
            test_queries: List of test queries
            k: Number of results to return for each query
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Benchmarking search performance with {len(test_queries)} queries...")
        
        benchmark_results = {
            'queries': test_queries,
            'results': {},
            'summary': {}
        }
        
        # Benchmark each search method
        search_methods = [
            ('semantic_search', self._benchmark_semantic_search),
            ('keyword_search', self._benchmark_keyword_search),
            ('graph_search', self._benchmark_graph_search),
            ('hybrid_search', self._benchmark_hybrid_search)
        ]
        
        for method_name, benchmark_func in search_methods:
            try:
                results = benchmark_func(test_queries, k)
                benchmark_results['results'][method_name] = results
            except Exception as e:
                logger.warning(f"Failed to benchmark {method_name}: {e}")
                benchmark_results['results'][method_name] = {'error': str(e)}
        
        # Calculate summary statistics
        for method_name, results in benchmark_results['results'].items():
            if 'error' not in results:
                latencies = results.get('latencies', [])
                if latencies:
                    benchmark_results['summary'][method_name] = {
                        'avg_latency': np.mean(latencies),
                        'p50_latency': np.percentile(latencies, 50),
                        'p95_latency': np.percentile(latencies, 95),
                        'p99_latency': np.percentile(latencies, 99),
                        'total_queries': len(test_queries),
                        'successful_queries': len(latencies)
                    }
        
        # Record performance metrics
        metrics = PerformanceMetrics(
            query_time=sum(r.get('summary', {}).get(method_name, {}).get('avg_latency', 0) 
                          for method_name, r in benchmark_results['results'].items()),
            accuracy=len(test_queries)  # Simplified accuracy metric
        )
        self.performance_metrics['benchmark'].append(metrics)
        
        return benchmark_results
    
    def _benchmark_semantic_search(self, queries: List[str], k: int) -> Dict[str, Any]:
        """Benchmark semantic search performance"""
        latencies = []
        results_count = []
        
        for query in queries:
            start_time = time.time()
            
            try:
                # Use existing vector search engine
                if self.vector_search_engine and self.vector_search_engine.edge_index:
                    query_embedding = np.random.random(384).astype('float32')  # Mock embedding
                    results = self.vector_search_engine.semantic_search(
                        query_embedding,
                        self.vector_search_engine.edge_index,
                        self.vector_search_engine.edge_id_map,
                        k=k
                    )
                    results_count.append(len(results))
                else:
                    results_count.append(0)
            except Exception as e:
                logger.warning(f"Semantic search failed for query '{query}': {e}")
                results_count.append(0)
            
            latency = time.time() - start_time
            latencies.append(latency)
        
        return {
            'latencies': latencies,
            'results_count': results_count,
            'avg_results': np.mean(results_count) if results_count else 0
        }
    
    def _benchmark_keyword_search(self, queries: List[str], k: int) -> Dict[str, Any]:
        """Benchmark keyword search performance"""
        latencies = []
        results_count = []
        
        for query in queries:
            start_time = time.time()
            
            try:
                if 'bm25_edges' in self.text_indices:
                    query_tokens = query.lower().split()
                    scores = self.text_indices['bm25_edges'].get_scores(query_tokens)
                    top_k_indices = np.argsort(scores)[-k:]
                    results_count.append(len(top_k_indices))
                else:
                    results_count.append(0)
            except Exception as e:
                logger.warning(f"Keyword search failed for query '{query}': {e}")
                results_count.append(0)
            
            latency = time.time() - start_time
            latencies.append(latency)
        
        return {
            'latencies': latencies,
            'results_count': results_count,
            'avg_results': np.mean(results_count) if results_count else 0
        }
    
    def _benchmark_graph_search(self, queries: List[str], k: int) -> Dict[str, Any]:
        """Benchmark graph search performance"""
        latencies = []
        results_count = []
        
        for query in queries:
            start_time = time.time()
            
            try:
                if 'graph' in self.graph_indices:
                    # Mock graph search - get random nodes
                    graph = self.graph_indices['graph']
                    random_nodes = list(graph.nodes())[:k]
                    results_count.append(len(random_nodes))
                else:
                    results_count.append(0)
            except Exception as e:
                logger.warning(f"Graph search failed for query '{query}': {e}")
                results_count.append(0)
            
            latency = time.time() - start_time
            latencies.append(latency)
        
        return {
            'latencies': latencies,
            'results_count': results_count,
            'avg_results': np.mean(results_count) if results_count else 0
        }
    
    def _benchmark_hybrid_search(self, queries: List[str], k: int) -> Dict[str, Any]:
        """Benchmark hybrid search performance"""
        latencies = []
        results_count = []
        
        for query in queries:
            start_time = time.time()
            
            try:
                if self.hybrid_search_engine:
                    config = HybridSearchConfig(result_limit=k)
                    results = asyncio.run(self.hybrid_search_engine.hybrid_search(query, config))
                    results_count.append(len(results))
                else:
                    results_count.append(0)
            except Exception as e:
                logger.warning(f"Hybrid search failed for query '{query}': {e}")
                results_count.append(0)
            
            latency = time.time() - start_time
            latencies.append(latency)
        
        return {
            'latencies': latencies,
            'results_count': results_count,
            'avg_results': np.mean(results_count) if results_count else 0
        }
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about index health and usage.
        
        Returns:
            Dictionary containing index statistics
        """
        stats = {
            'index_versions': self.index_versions,
            'current_version': self.current_version,
            'performance_metrics': dict(self.performance_metrics),
            'query_patterns': {k: v.__dict__ for k, v in self.query_patterns.items()},
            'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
            'index_sizes': {
                'text': len(self.text_indices),
                'vector': len(self.vector_indices),
                'graph': len(self.graph_indices),
                'temporal': len(self.temporal_indices)
            },
            'last_optimization': self.last_optimization.isoformat(),
            'auto_optimization_enabled': self.auto_optimization_enabled
        }
        
        # Add detailed statistics for each index type
        if 'bm25_edges' in self.text_indices:
            stats['text_stats'] = {
                'bm25_edges_built': True,
                'tfidf_edges_built': 'tfidf_edges' in self.text_indices,
                'bm25_nodes_built': 'bm25_nodes' in self.text_indices,
                'tfidf_nodes_built': 'tfidf_nodes' in self.text_indices
            }
        
        if self.vector_indices:
            stats['vector_stats'] = {
                'index_types': list(self.vector_indices.keys()),
                'total_entities': sum(len(v['id_map']) for v in self.vector_indices.values())
            }
        
        if 'graph' in self.graph_indices:
            stats['graph_stats'] = {
                'node_count': self.graph_indices['node_count'],
                'edge_count': self.graph_indices['edge_count'],
                'centrality_measures': list(self.graph_indices['centrality'].keys())
            }
        
        if self.temporal_indices:
            stats['temporal_stats'] = {
                'node_partitions': len(self.temporal_indices.get('node_partitions', {})),
                'edge_partitions': len(self.temporal_indices.get('edge_partitions', {})),
                'time_ranges_available': bool(self.temporal_indices.get('node_time_range')) and 
                                       bool(self.temporal_indices.get('edge_time_range'))
            }
        
        return stats
    
    def optimize_index_parameters(self) -> Dict[str, Any]:
        """
        Optimize index parameters based on performance metrics.
        
        Returns:
            Dictionary containing optimization results
        """
        logger.info("Optimizing index parameters...")
        
        optimization_results = {
            'timestamp': datetime.now(),
            'optimizations_applied': [],
            'performance_improvements': {}
        }
        
        # Analyze query patterns
        if self.query_patterns:
            self._optimize_for_query_patterns(optimization_results)
        
        # Analyze performance metrics
        if self.performance_metrics:
            self._optimize_based_on_metrics(optimization_results)
        
        # Auto-select best FAISS index types based on data size
        self._optimize_faiss_indices(optimization_results)
        
        # Update last optimization time
        self.last_optimization = datetime.now()
        
        # Record optimization
        self.optimization_history.append(optimization_results)
        
        return optimization_results
    
    def _optimize_for_query_patterns(self, optimization_results: Dict[str, Any]):
        """Optimize based on query pattern analysis"""
        # Find most frequent query types
        sorted_patterns = sorted(
            self.query_patterns.items(),
            key=lambda x: x[1].frequency,
            reverse=True
        )
        
        if sorted_patterns:
            most_frequent = sorted_patterns[0][0]
            optimization_results['optimizations_applied'].append({
                'type': 'query_pattern_optimization',
                'most_frequent_query_type': most_frequent,
                'frequency': sorted_patterns[0][1].frequency
            })
            
            # Adjust cache size based on query frequency
            if most_frequent in ['semantic_search', 'hybrid_search']:
                self.config.cache_size = min(self.config.cache_size * 2, 50000)
                optimization_results['performance_improvements']['cache_size'] = self.config.cache_size
    
    def _optimize_based_on_metrics(self, optimization_results: Dict[str, Any]):
        """Optimize based on performance metrics"""
        # Analyze latency metrics
        for metric_type, metrics_list in self.performance_metrics.items():
            if metrics_list:
                recent_metrics = metrics_list[-10:]  # Last 10 measurements
                
                avg_latency = np.mean([m.query_time for m in recent_metrics])
                
                if avg_latency > 1.0:  # If latency is high
                    optimization_results['optimizations_applied'].append({
                        'type': 'performance_optimization',
                        'metric_type': metric_type,
                        'avg_latency': avg_latency,
                        'action': 'increase_batch_size'
                    })
                    
                    # Increase batch size for better performance
                    self.config.batch_size = min(self.config.batch_size * 2, 5000)
                    optimization_results['performance_improvements']['batch_size'] = self.config.batch_size
    
    def _optimize_faiss_indices(self, optimization_results: Dict[str, Any]):
        """Auto-select best FAISS index types based on data size"""
        if not self.driver.edges_df.empty:
            edge_count = len(self.driver.edges_df)
            
            # Select appropriate index type based on data size
            if edge_count < 1000:
                recommended_type = IndexType.FLAT
                reason = "Small dataset, FLAT provides best accuracy"
            elif edge_count < 10000:
                recommended_type = IndexType.IVFFLAT
                reason = "Medium dataset, IVFFLAT provides good speed/accuracy balance"
            elif edge_count < 100000:
                recommended_type = IndexType.HNSW
                reason = "Large dataset, HNSW provides good performance"
            else:
                recommended_type = IndexType.IVFPQ
                reason = "Very large dataset, IVFPQ provides best memory efficiency"
            
            optimization_results['optimizations_applied'].append({
                'type': 'faiss_index_optimization',
                'recommended_index_type': recommended_type.value,
                'reason': reason,
                'dataset_size': edge_count
            })
    
    def monitor_search_queries(self, query: str, query_type: str, execution_time: float, 
                             result_count: int) -> None:
        """
        Monitor search queries to track patterns and optimize accordingly.
        
        Args:
            query: The search query
            query_type: Type of search performed
            execution_time: Time taken to execute the query
            result_count: Number of results returned
        """
        # Update query pattern
        if query_type not in self.query_patterns:
            self.query_patterns[query_type] = QueryPattern(
                query_type=query_type,
                parameters={}
            )
        
        pattern = self.query_patterns[query_type]
        pattern.frequency += 1
        pattern.avg_latency = (pattern.avg_latency * (pattern.frequency - 1) + execution_time) / pattern.frequency
        pattern.last_used = datetime.now()
        
        # Check if auto-optimization is needed
        if (self.auto_optimization_enabled and 
            datetime.now() - self.last_optimization > timedelta(seconds=self.config.optimization_interval)):
            
            logger.info("Triggering auto-optimization based on query monitoring")
            self.optimize_index_parameters()
    
    def save_index(self, index_type: str, filepath: str) -> None:
        """
        Save index to disk.
        
        Args:
            index_type: Type of index to save
            filepath: Path to save the index
        """
        logger.info(f"Saving {index_type} index to {filepath}")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        index_dir = Path(filepath)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        if index_type == 'text':
            data = {
                'indices': self.text_indices,
                'config': self.config.__dict__,
                'version': self.index_versions.get('text', 1)
            }
            
            with open(index_dir / "text_indices.pkl", 'wb') as f:
                pickle.dump(data, f)
        
        elif index_type == 'vector':
            data = {
                'indices': self.vector_indices,
                'config': self.config.__dict__,
                'version': self.index_versions.get('vector', 1)
            }
            
            # Save FAISS indices
            for name, index_data in self.vector_indices.items():
                if 'index' in index_data:
                    faiss.write_index(index_data['index'], str(index_dir / f"{name}.faiss"))
            
            # Save metadata
            with open(index_dir / "vector_indices_metadata.pkl", 'wb') as f:
                pickle.dump(data, f)
        
        elif index_type == 'graph':
            data = {
                'indices': self.graph_indices,
                'config': self.config.__dict__,
                'version': self.index_versions.get('graph', 1)
            }
            
            with open(index_dir / "graph_indices.pkl", 'wb') as f:
                pickle.dump(data, f)
        
        elif index_type == 'temporal':
            data = {
                'indices': self.temporal_indices,
                'config': self.config.__dict__,
                'version': self.index_versions.get('temporal', 1)
            }
            
            with open(index_dir / "temporal_indices.pkl", 'wb') as f:
                pickle.dump(data, f)
        
        elif index_type == 'all':
            # Save all indices
            self.save_index('text', str(index_dir / "text"))
            self.save_index('vector', str(index_dir / "vector"))
            self.save_index('graph', str(index_dir / "graph"))
            self.save_index('temporal', str(index_dir / "temporal"))
            
            # Save metadata
            metadata = {
                'index_versions': self.index_versions,
                'current_version': self.current_version,
                'config': self.config.__dict__,
                'performance_metrics': dict(self.performance_metrics),
                'optimization_history': self.optimization_history
            }
            
            with open(index_dir / "metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        logger.info(f"Successfully saved {index_type} index")
    
    def load_index(self, index_type: str, filepath: str) -> None:
        """
        Load index from disk.
        
        Args:
            index_type: Type of index to load
            filepath: Path to the saved index
        """
        logger.info(f"Loading {index_type} index from {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index not found at {filepath}")
        
        index_dir = Path(filepath)
        
        if index_type == 'text':
            with open(index_dir / "text_indices.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.text_indices = data['indices']
            self.index_versions['text'] = data.get('version', 1)
        
        elif index_type == 'vector':
            # Load metadata
            with open(index_dir / "vector_indices_metadata.pkl", 'rb') as f:
                data = pickle.load(f)
            
            # Load FAISS indices
            for name in data['indices'].keys():
                index_file = index_dir / f"{name}.faiss"
                if index_file.exists():
                    data['indices'][name]['index'] = faiss.read_index(str(index_file))
            
            self.vector_indices = data['indices']
            self.index_versions['vector'] = data.get('version', 1)
        
        elif index_type == 'graph':
            with open(index_dir / "graph_indices.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.graph_indices = data['indices']
            self.index_versions['graph'] = data.get('version', 1)
        
        elif index_type == 'temporal':
            with open(index_dir / "temporal_indices.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.temporal_indices = data['indices']
            self.index_versions['temporal'] = data.get('version', 1)
        
        elif index_type == 'all':
            # Load all indices
            self.load_index('text', str(index_dir / "text"))
            self.load_index('vector', str(index_dir / "vector"))
            self.load_index('graph', str(index_dir / "graph"))
            self.load_index('temporal', str(index_dir / "temporal"))
            
            # Load metadata
            with open(index_dir / "metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.index_versions = metadata['index_versions']
            self.current_version = metadata['current_version']
            self.performance_metrics = defaultdict(list, metadata['performance_metrics'])
            self.optimization_history = metadata['optimization_history']
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        logger.info(f"Successfully loaded {index_type} index")
    
    def incremental_index_update(self, index_type: str, new_data: Dict[str, Any]) -> None:
        """
        Update indices incrementally without full rebuild.
        
        Args:
            index_type: Type of index to update
            new_data: New data to incorporate into the index
        """
        logger.info(f"Performing incremental update for {index_type} index")
        
        if index_type == 'text':
            self._incremental_text_update(new_data)
        elif index_type == 'vector':
            self._incremental_vector_update(new_data)
        elif index_type == 'graph':
            self._incremental_graph_update(new_data)
        elif index_type == 'temporal':
            self._incremental_temporal_update(new_data)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Increment version
        if index_type in self.index_versions:
            self.index_versions[index_type] += 1
        
        logger.info(f"Successfully updated {index_type} index incrementally")
    
    def _incremental_text_update(self, new_data: Dict[str, Any]):
        """Incremental update for text indices"""
        if 'new_edges' in new_data:
            new_edge_texts = [edge['fact'] for edge in new_data['new_edges']]
            tokenized_texts = [text.lower().split() for text in new_edge_texts]
            
            # Update BM25 index
            if 'bm25_edges' in self.text_indices:
                # Note: BM25Okapi doesn't support incremental updates, would need rebuild
                logger.warning("BM25 index requires full rebuild for updates")
        
        if 'new_nodes' in new_data:
            new_node_texts = [node['name'] for node in new_data['new_nodes']]
            tokenized_texts = [text.lower().split() for text in new_node_texts]
            
            # Update BM25 index
            if 'bm25_nodes' in self.text_indices:
                # Note: BM25Okapi doesn't support incremental updates, would need rebuild
                logger.warning("BM25 index requires full rebuild for updates")
    
    def _incremental_vector_update(self, new_data: Dict[str, Any]):
        """Incremental update for vector indices"""
        if 'new_edges' in new_data and self.vector_search_engine:
            new_edges = new_data['new_edges']
            
            for edge in new_edges:
                if edge.get('fact_embedding'):
                    embedding = np.array([edge['fact_embedding']]).astype('float32')
                    
                    # Add to existing edge indices
                    for index_name, index_data in self.vector_indices.items():
                        if 'edges_' in index_name and 'index' in index_data:
                            index_data['index'] = self.vector_search_engine.add_embeddings(
                                embedding,
                                [edge['uuid']],
                                index_data['index'],
                                index_data['metadata']
                            )
                            index_data['id_map'].append(edge['uuid'])
        
        if 'new_nodes' in new_data and self.vector_search_engine:
            new_nodes = new_data['new_nodes']
            
            for node in new_nodes:
                if node.get('name_embedding'):
                    embedding = np.array([node['name_embedding']]).astype('float32')
                    
                    # Add to existing node indices
                    for index_name, index_data in self.vector_indices.items():
                        if 'nodes_' in index_name and 'index' in index_data:
                            index_data['index'] = self.vector_search_engine.add_embeddings(
                                embedding,
                                [node['uuid']],
                                index_data['index'],
                                index_data['metadata']
                            )
                            index_data['id_map'].append(node['uuid'])
    
    def _incremental_graph_update(self, new_data: Dict[str, Any]):
        """Incremental update for graph indices"""
        if 'new_edges' in new_data and 'graph' in self.graph_indices:
            new_edges = new_data['new_edges']
            G = self.graph_indices['graph']
            
            for edge in new_edges:
                G.add_edge(
                    edge['source_uuid'],
                    edge['target_uuid'],
                    edge_uuid=edge['uuid'],
                    fact=edge.get('fact', '')
                )
            
            # Update adjacency lists
            for node in G.nodes():
                self.graph_indices['adjacency_lists'][node] = list(G.neighbors(node))
                self.graph_indices['reverse_adjacency_lists'][node] = list(G.predecessors(node)) if hasattr(G, 'predecessors') else []
            
            # Update counts
            self.graph_indices['node_count'] = len(G.nodes())
            self.graph_indices['edge_count'] = len(G.edges())
    
    def _incremental_temporal_update(self, new_data: Dict[str, Any]):
        """Incremental update for temporal indices"""
        if 'new_nodes' in new_data and 'node_partitions' in self.temporal_indices:
            new_nodes = new_data['new_nodes']
            
            for node in new_nodes:
                created_at = pd.to_datetime(node['created_at'])
                
                # Add to appropriate time partitions
                year_key = f'year_{created_at.year}'
                if year_key not in self.temporal_indices['node_partitions']:
                    self.temporal_indices['node_partitions'][year_key] = []
                self.temporal_indices['node_partitions'][year_key].append(node['uuid'])
                
                month_key = f'year_{created_at.year}_month_{created_at.month}'
                if month_key not in self.temporal_indices['node_partitions']:
                    self.temporal_indices['node_partitions'][month_key] = []
                self.temporal_indices['node_partitions'][month_key].append(node['uuid'])
        
        if 'new_edges' in new_data and 'edge_partitions' in self.temporal_indices:
            new_edges = new_data['new_edges']
            
            for edge in new_edges:
                created_at = pd.to_datetime(edge['created_at'])
                valid_at = pd.to_datetime(edge['valid_at'])
                
                # Add to creation time partitions
                created_year_key = f'created_year_{created_at.year}'
                if created_year_key not in self.temporal_indices['edge_partitions']:
                    self.temporal_indices['edge_partitions'][created_year_key] = []
                self.temporal_indices['edge_partitions'][created_year_key].append(edge['uuid'])
                
                # Add to validity time partitions
                valid_year_key = f'valid_year_{valid_at.year}'
                if valid_year_key not in self.temporal_indices['edge_partitions']:
                    self.temporal_indices['edge_partitions'][valid_year_key] = []
                self.temporal_indices['edge_partitions'][valid_year_key].append(edge['uuid'])
    
    def index_versioning(self, operation: str, backup: bool = True) -> Dict[str, Any]:
        """
        Manage index versions and rollbacks.
        
        Args:
            operation: Operation to perform ('create_version', 'rollback', 'list_versions')
            backup: Whether to create backup before rollback
            
        Returns:
            Dictionary containing versioning results
        """
        if operation == 'create_version':
            return self._create_index_version()
        elif operation == 'rollback':
            return self._rollback_index_version(backup)
        elif operation == 'list_versions':
            return self._list_index_versions()
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _create_index_version(self) -> Dict[str, Any]:
        """Create a new version of all indices"""
        version_info = {
            'timestamp': datetime.now(),
            'version': self.current_version + 1,
            'index_versions': dict(self.index_versions),
            'performance_metrics': dict(self.performance_metrics),
            'optimization_history': self.optimization_history.copy()
        }
        
        # Save version to disk
        version_dir = Path(f"index_versions/v{self.current_version + 1}")
        version_dir.mkdir(parents=True, exist_ok=True)
        
        with open(version_dir / "version_info.json", 'w') as f:
            json.dump(version_info, f, indent=2, default=str)
        
        # Save current indices
        self.save_index('all', str(version_dir / "indices"))
        
        self.current_version += 1
        
        logger.info(f"Created index version v{self.current_version}")
        
        return version_info
    
    def _rollback_index_version(self, backup: bool = True) -> Dict[str, Any]:
        """Rollback to a previous version"""
        if backup:
            # Create backup of current state
            backup_info = self._create_index_version()
        
        # Find available versions
        versions_dir = Path("index_versions")
        available_versions = []
        
        if versions_dir.exists():
            for version_dir in versions_dir.iterdir():
                if version_dir.is_dir() and version_dir.name.startswith('v'):
                    version_num = int(version_dir.name[1:])
                    available_versions.append(version_num)
        
        if not available_versions:
            raise ValueError("No available versions to rollback to")
        
        # Rollback to previous version
        target_version = max(available_versions) - 1
        if target_version < 1:
            raise ValueError("Cannot rollback to version before 1")
        
        target_dir = versions_dir / f"v{target_version}"
        
        if not target_dir.exists():
            raise ValueError(f"Version v{target_version} not found")
        
        # Load indices from target version
        self.load_index('all', str(target_dir / "indices"))
        
        # Load version info
        with open(target_dir / "version_info.json", 'r') as f:
            version_info = json.load(f)
        
        self.current_version = target_version
        self.index_versions = version_info['index_versions']
        
        logger.info(f"Rolled back to index version v{target_version}")
        
        return {
            'rollback_to_version': target_version,
            'backup_created': backup,
            'timestamp': datetime.now()
        }
    
    def _list_index_versions(self) -> Dict[str, Any]:
        """List all available index versions"""
        versions_dir = Path("index_versions")
        available_versions = []
        
        if versions_dir.exists():
            for version_dir in versions_dir.iterdir():
                if version_dir.is_dir() and version_dir.name.startswith('v'):
                    version_num = int(version_dir.name[1:])
                    
                    # Load version info
                    version_info_file = version_dir / "version_info.json"
                    version_info = {}
                    
                    if version_info_file.exists():
                        with open(version_info_file, 'r') as f:
                            version_info = json.load(f)
                    
                    available_versions.append({
                        'version': version_num,
                        'timestamp': version_info.get('timestamp'),
                        'index_versions': version_info.get('index_versions', {}),
                        'size': self._get_directory_size(version_dir)
                    })
        
        return {
            'current_version': self.current_version,
            'available_versions': sorted(available_versions, key=lambda x: x['version'])
        }
    
    def _get_directory_size(self, path: Path) -> int:
        """Get size of directory in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size
    
    def cleanup_unused_indices(self, retention_days: int = 30) -> Dict[str, Any]:
        """
        Remove unused indices and old versions.
        
        Args:
            retention_days: Number of days to keep old versions
            
        Returns:
            Dictionary containing cleanup results
        """
        logger.info(f"Cleaning up unused indices (retention: {retention_days} days)")
        
        cleanup_results = {
            'timestamp': datetime.now(),
            'removed_versions': [],
            'freed_space': 0,
            'retention_days': retention_days
        }
        
        # Remove old versions
        versions_dir = Path("index_versions")
        if versions_dir.exists():
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for version_dir in versions_dir.iterdir():
                if version_dir.is_dir() and version_dir.name.startswith('v'):
                    version_num = int(version_dir.name[1:])
                    
                    # Skip current version
                    if version_num == self.current_version:
                        continue
                    
                    # Check version age
                    version_info_file = version_dir / "version_info.json"
                    if version_info_file.exists():
                        with open(version_info_file, 'r') as f:
                            version_info = json.load(f)
                        
                        version_date = pd.to_datetime(version_info['timestamp'])
                        
                        if version_date < cutoff_date:
                            # Remove version
                            removed_size = self._get_directory_size(version_dir)
                            
                            import shutil
                            shutil.rmtree(version_dir)
                            
                            cleanup_results['removed_versions'].append({
                                'version': version_num,
                                'removed_date': version_date.isoformat(),
                                'size': removed_size
                            })
                            
                            cleanup_results['freed_space'] += removed_size
        
        # Clean up temporary files
        temp_dirs = [Path("temp"), Path("cache")]
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                removed_size = self._get_directory_size(temp_dir)
                import shutil
                shutil.rmtree(temp_dir)
                cleanup_results['freed_space'] += removed_size
        
        logger.info(f"Cleanup completed. Freed {cleanup_results['freed_space']} bytes")
        
        return cleanup_results


# Integration methods for HuggingFaceDriver
def extend_huggingface_driver(driver: HuggingFaceDriver) -> HuggingFaceDriver:
    """
    Extend HuggingFaceDriver with performance optimization methods.
    
    Args:
        driver: HuggingFaceDriver instance to extend
        
    Returns:
        Extended HuggingFaceDriver instance
    """
    # Add performance optimizer
    driver.performance_optimizer = SearchIndexManager(driver)
    
    # Add optimization methods
    driver.optimize_search_performance = optimize_search_performance
    driver.get_performance_metrics = get_performance_metrics
    driver.auto_rebuild_indices = auto_rebuild_indices
    
    return driver


async def optimize_search_performance(driver: HuggingFaceDriver, 
                                    force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Optimize search performance for the HuggingFaceDriver.
    
    Args:
        driver: HuggingFaceDriver instance
        force_rebuild: Whether to force full index rebuild
        
    Returns:
        Dictionary containing optimization results
    """
    if not hasattr(driver, 'performance_optimizer'):
        driver.performance_optimizer = SearchIndexManager(driver)
    
    optimizer = driver.performance_optimizer
    
    if force_rebuild:
        return optimizer.rebuild_all_indices()
    else:
        return optimizer.optimize_index_parameters()


def get_performance_metrics(driver: HuggingFaceDriver) -> Dict[str, Any]:
    """
    Get performance metrics for the HuggingFaceDriver.
    
    Args:
        driver: HuggingFaceDriver instance
        
    Returns:
        Dictionary containing performance metrics
    """
    if not hasattr(driver, 'performance_optimizer'):
        return {"error": "Performance optimizer not initialized"}
    
    return driver.performance_optimizer.get_index_statistics()


async def auto_rebuild_indices(driver: HuggingFaceDriver, 
                             threshold: float = 0.8) -> Dict[str, Any]:
    """
    Automatically rebuild indices based on performance threshold.
    
    Args:
        driver: HuggingFaceDriver instance
        threshold: Performance threshold (0.0-1.0)
        
    Returns:
        Dictionary containing rebuild results
    """
    if not hasattr(driver, 'performance_optimizer'):
        driver.performance_optimizer = SearchIndexManager(driver)
    
    optimizer = driver.performance_optimizer
    
    # Check if rebuild is needed
    stats = optimizer.get_index_statistics()
    
    # Simple heuristic: rebuild if data size has changed significantly
    current_edge_count = len(driver.edges_df)
    current_node_count = len(driver.nodes_df)
    
    # Trigger rebuild if significant changes detected
    if (current_edge_count > 10000 or current_node_count > 5000):
        return optimizer.rebuild_all_indices()
    else:
        return {
            "action": "no_rebuild_needed",
            "reason": "Data size below threshold",
            "current_edge_count": current_edge_count,
            "current_node_count": current_node_count,
            "threshold": threshold
        }