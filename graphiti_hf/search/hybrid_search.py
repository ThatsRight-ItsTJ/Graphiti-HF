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
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

# Import optional dependencies with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from rank_bm25 import BM25Okapi
    import networkx as nx
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

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search operations"""
    semantic_weight: float = 0.4
    keyword_weight: float = 0.3
    graph_weight: float = 0.3
    semantic_threshold: float = 0.0
    keyword_threshold: float = 0.0
    graph_distance_cutoff: int = 5
    result_limit: int = 10
    center_node_uuid: Optional[str] = None
    temporal_filter: Optional[datetime] = None
    edge_types: Optional[List[str]] = None
    batch_size: int = 100
    cache_enabled: bool = True
    max_cache_size: int = 10000


class HybridSearchEngine:
    """
    Hybrid search engine that combines semantic, keyword, and graph-based ranking
    for optimal knowledge graph retrieval.
    
    This implementation follows the design from graphiti_hf_plan.md lines 694-911,
    combining the strengths of multiple search methods:
    - Semantic search using sentence embeddings and FAISS
    - Keyword search using BM25 and TF-IDF
    - Graph-based ranking using NetworkX distance calculations
    """
    
    def __init__(self, driver: HuggingFaceDriver, embedder_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the hybrid search engine.
        
        Args:
            driver: HuggingFaceDriver instance
            embedder_model: Sentence transformer model for semantic search
        """
        self.driver = driver
        self.embedder = SentenceTransformer(embedder_model)
        self.bm25_index = None
        self.tfidf_vectorizer = None
        self._build_text_indices()
    
    def _build_text_indices(self):
        """Build BM25 and TF-IDF indices for keyword search"""
        if self.driver.edges_df.empty:
            return
            
        # Prepare text corpus from edge facts
        texts = self.driver.edges_df['fact'].fillna('').tolist()
        
        # Build BM25 index
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25_index = BM25Okapi(tokenized_texts)
        
        # Build TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.tfidf_vectorizer.fit(texts)
    
    async def hybrid_search(
        self, 
        query: str, 
        config: Optional[HybridSearchConfig] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic, keyword, and graph-based ranking.
        
        Args:
            query: Search query string
            config: Hybrid search configuration
            
        Returns:
            List of search results with combined scores and individual method scores
        """
        config = config or HybridSearchConfig()
        
        # 1. Semantic Search
        semantic_results = await self._semantic_search(query, config.result_limit * 3, config)
        
        # 2. Keyword Search  
        keyword_results = await self._keyword_search(query, config.result_limit * 3, config)
        
        # 3. Graph-based ranking (if center node provided)
        graph_scores = {}
        if config.center_node_uuid:
            graph_scores = await self._graph_distance_ranking(config.center_node_uuid, config)
        
        # 4. Combine and rank results
        combined_results = self._combine_rankings(
            semantic_results, keyword_results, graph_scores,
            config.semantic_weight, config.keyword_weight, config.graph_weight
        )
        
        return combined_results[:config.result_limit]
    
    async def _semantic_search(self, query: str, k: int, config: Optional[HybridSearchConfig] = None) -> List[Dict[str, Any]]:
        """
        Search using sentence embeddings and FAISS vector search.
        
        Args:
            query: Search query
            k: Number of results to return
            config: Search configuration
            
        Returns:
            List of semantic search results with similarity scores
        """
        if config is None:
            config = HybridSearchConfig()
        
        query_embedding = self.embedder.encode([query])[0]
        
        # Convert to numpy array if it's not already
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        
        # Search edges by fact embedding using existing vector search
        similar_edges = await self.driver.query_edges_by_embedding(
            query_embedding.tolist() if hasattr(query_embedding, 'tolist') else (query_embedding if isinstance(query_embedding, list) else [query_embedding]),
            k=k,
            similarity_threshold=config.semantic_threshold
        )
        
        results = []
        for edge in similar_edges:
            # Calculate similarity score
            if edge.fact_embedding:
                edge_embedding = np.array(edge.fact_embedding)
                similarity = np.dot(query_embedding, edge_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(edge_embedding)
                )
                
                results.append({
                    'edge': edge,
                    'semantic_score': float(similarity),
                    'type': 'semantic'
                })
        
        return sorted(results, key=lambda x: x['semantic_score'], reverse=True)
    
    async def _keyword_search(self, query: str, k: int, config: Optional[HybridSearchConfig] = None) -> List[Dict[str, Any]]:
        """
        Search using BM25 keyword matching and TF-IDF.
        
        Args:
            query: Search query
            k: Number of results to return
            config: Search configuration
            
        Returns:
            List of keyword search results with BM25 scores
        """
        if config is None:
            config = HybridSearchConfig()
        
        if self.bm25_index is None:
            return []
        
        query_tokens = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argpartition(bm25_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(bm25_scores[top_indices])][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.driver.edges_df):
                edge_row = self.driver.edges_df.iloc[idx]
                edge = self.driver._row_to_edge(edge_row)
                
                # Apply temporal filter if specified
                if config.temporal_filter:
                    if edge.valid_at and edge.valid_at <= config.temporal_filter:
                        if edge.invalidated_at and edge.invalidated_at <= config.temporal_filter:
                            continue
                
                # Apply edge type filter if specified
                if config.edge_types and edge.name not in config.edge_types:
                    continue
                
                results.append({
                    'edge': edge,
                    'keyword_score': float(bm25_scores[idx]),
                    'type': 'keyword'
                })
        
        return results
    
    async def _graph_distance_ranking(self, center_node_uuid: str, config: Optional[HybridSearchConfig] = None) -> Dict[str, float]:
        """
        Calculate graph distance-based scores from center node using NetworkX.
        
        Args:
            center_node_uuid: UUID of the center node for distance calculation
            config: Search configuration
            
        Returns:
            Dictionary mapping edge UUIDs to distance-based scores
        """
        if config is None:
            config = HybridSearchConfig()
        
        # Build NetworkX graph from edges
        if nx is None:
            return {}
            
        G = nx.Graph()
        
        for _, edge_row in self.driver.edges_df.iterrows():
            # Apply temporal filter if specified
            if config.temporal_filter:
                if edge_row['valid_at'] and edge_row['valid_at'] <= config.temporal_filter:
                    if edge_row['invalidated_at'] and edge_row['invalidated_at'] <= config.temporal_filter:
                        continue
            
            # Apply edge type filter if specified
            if config.edge_types and edge_row['name'] not in config.edge_types:
                continue
            
            G.add_edge(edge_row['source_uuid'], edge_row['target_uuid'],
                      edge_uuid=edge_row['uuid'])
        
        if center_node_uuid not in G:
            return {}
        
        # Calculate shortest path distances
        try:
            distances = nx.single_source_shortest_path_length(G, center_node_uuid, cutoff=config.graph_distance_cutoff)
        except Exception:
            # If center node is not in graph or other error, return empty scores
            return {}
        
        # Convert to scores (closer = higher score)
        max_distance = max(distances.values()) if distances else 1
        scores = {}
        
        for _, edge_row in self.driver.edges_df.iterrows():
            # Apply filters
            if config.temporal_filter:
                if edge_row['valid_at'] and edge_row['valid_at'] <= config.temporal_filter:
                    if edge_row['invalidated_at'] and edge_row['invalidated_at'] <= config.temporal_filter:
                        continue
            
            if config.edge_types and edge_row['name'] not in config.edge_types:
                continue
            
            source_dist = distances.get(edge_row['source_uuid'], max_distance + 1)
            target_dist = distances.get(edge_row['target_uuid'], max_distance + 1)
            min_dist = min(source_dist, target_dist)
            
            # Score is inversely proportional to distance
            score = 1.0 / (1.0 + min_dist) if min_dist <= max_distance else 0.0
            scores[edge_row['uuid']] = score
        
        return scores
    
    def _combine_rankings(
        self, 
        semantic_results: List[Dict], 
        keyword_results: List[Dict],
        graph_scores: Dict[str, float],
        semantic_weight: float,
        keyword_weight: float, 
        graph_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Combine different ranking methods using weighted scores.
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            graph_scores: Graph distance scores
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
            graph_weight: Weight for graph-based ranking
            
        Returns:
            Combined and ranked results
        """
        
        # Normalize scores
        semantic_scores = self._normalize_scores([r['semantic_score'] for r in semantic_results])
        keyword_scores = self._normalize_scores([r['keyword_score'] for r in keyword_results])
        
        # Create combined results
        all_edges = {}
        
        # Add semantic results
        for i, result in enumerate(semantic_results):
            edge_uuid = result['edge'].uuid
            all_edges[edge_uuid] = {
                'edge': result['edge'],
                'semantic_score': semantic_scores[i] if i < len(semantic_scores) else 0.0,
                'keyword_score': 0.0,
                'graph_score': graph_scores.get(edge_uuid, 0.0)
            }
        
        # Add keyword results
        for i, result in enumerate(keyword_results):
            edge_uuid = result['edge'].uuid
            if edge_uuid in all_edges:
                all_edges[edge_uuid]['keyword_score'] = keyword_scores[i] if i < len(keyword_scores) else 0.0
            else:
                all_edges[edge_uuid] = {
                    'edge': result['edge'],
                    'semantic_score': 0.0,
                    'keyword_score': keyword_scores[i] if i < len(keyword_scores) else 0.0,
                    'graph_score': graph_scores.get(edge_uuid, 0.0)
                }
        
        # Calculate combined scores
        final_results = []
        for edge_uuid, scores in all_edges.items():
            combined_score = (
                semantic_weight * scores['semantic_score'] +
                keyword_weight * scores['keyword_score'] +
                graph_weight * scores['graph_score']
            )
            
            final_results.append({
                'edge': scores['edge'],
                'combined_score': combined_score,
                'semantic_score': scores['semantic_score'],
                'keyword_score': scores['keyword_score'],
                'graph_score': scores['graph_score']
            })
        
        return sorted(final_results, key=lambda x: x['combined_score'], reverse=True)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to 0-1 range.
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
            
        scores = np.array(scores)
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return ((scores - min_score) / (max_score - min_score)).tolist()
    
    async def batch_hybrid_search(
        self, 
        queries: List[str], 
        config: Optional[HybridSearchConfig] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform batch hybrid search for multiple queries.
        
        Args:
            queries: List of search queries
            config: Hybrid search configuration
            
        Returns:
            List of search result lists, one per query
        """
        if config is None:
            config = HybridSearchConfig()
        
        # Process queries in parallel
        tasks = []
        for query in queries:
            task = self.hybrid_search(query, config)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the hybrid search engine.
        
        Returns:
            Dictionary containing search engine statistics
        """
        return {
            "bm25_index_built": self.bm25_index is not None,
            "tfidf_vectorizer_fitted": self.tfidf_vectorizer is not None,
            "embedder_model": self.embedder.__class__.__name__,
            "driver_edges_count": len(self.driver.edges_df),
            "driver_nodes_count": len(self.driver.nodes_df)
        }
    
    def rebuild_text_indices(self):
        """Rebuild text search indices"""
        self._build_text_indices()
        logger.info("Rebuilt text search indices")