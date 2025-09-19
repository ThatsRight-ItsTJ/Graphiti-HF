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
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle
import numpy as np
import pandas as pd
import faiss
from dataclasses import dataclass
from enum import Enum
import asyncio
from pathlib import Path

from graphiti_core.nodes import EntityNode, EpisodicNode, CommunityNode
from graphiti_core.edges import EntityEdge, EpisodicEdge, CommunityEdge

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Supported FAISS index types"""
    FLAT = "flat"
    IVFFLAT = "ivfflat"
    HNSW = "hnsw"
    PQ = "pq"
    IVFPQ = "ivfpq"


@dataclass
class SearchConfig:
    """Configuration for vector search operations"""
    index_type: IndexType = IndexType.FLAT
    n_lists: int = 100  # For IVF indices
    n_probes: int = 10  # For IVF indices
    ef_search: int = 64  # For HNSW indices
    m: int = 32  # For HNSW indices
    nbits: int = 8  # For PQ indices
    k: int = 10  # Number of results to return
    similarity_threshold: float = 0.0
    batch_size: int = 100
    use_gpu: bool = False
    normalize_embeddings: bool = True


class VectorSearchEngine:
    """
    Vector search engine using FAISS for fast nearest neighbor search.
    
    Supports searching on both node and edge embeddings with configurable
    index types and search parameters.
    """
    
    def __init__(self, embed_dim: int = 384, config: Optional[SearchConfig] = None):
        """
        Initialize the vector search engine.
        
        Args:
            embed_dim: Dimension of embeddings (default 384 for sentence-transformers)
            config: Search configuration
        """
        self.embed_dim = embed_dim
        self.config = config or SearchConfig()
        
        # FAISS indices for different entity types
        self.node_index = None
        self.edge_index = None
        self.community_index = None
        
        # Mapping from indices to original data
        self.node_id_map = []
        self.edge_id_map = []
        self.community_id_map = []
        
        # Index metadata
        self.node_index_metadata = {}
        self.edge_index_metadata = {}
        self.community_index_metadata = {}
        
        # Initialize FAISS based on config
        if self.config.use_gpu:
            self._init_gpu()
    
    def _init_gpu(self):
        """Initialize GPU support for FAISS"""
        try:
            if faiss.get_num_gpus() == 0:
                logger.warning("No GPUs available, falling back to CPU")
                return
            
            # Set default GPU memory
            res = faiss.StandardGpuResources()
            self.gpu_resources = res
            logger.info(f"Initialized FAISS with {faiss.get_num_gpus()} GPUs")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU support: {e}")
    
    def _create_index(self, index_type: IndexType) -> faiss.Index:
        """Create a FAISS index of the specified type"""
        if index_type == IndexType.FLAT:
            index = faiss.IndexFlatIP(self.embed_dim)  # Inner product for cosine similarity
        elif index_type == IndexType.IVFFLAT:
            quantizer = faiss.IndexFlatIP(self.embed_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embed_dim, self.config.n_lists, faiss.METRIC_INNER_PRODUCT)
            if not index.is_trained:
                index.train(np.random.random((1000, self.embed_dim)).astype('float32'))
        elif index_type == IndexType.HNSW:
            index = faiss.IndexHNSWFlat(self.embed_dim, self.config.m, faiss.METRIC_INNER_PRODUCT)
        elif index_type == IndexType.PQ:
            index = faiss.IndexPQ(self.embed_dim, self.config.nbits, 8, faiss.METRIC_INNER_PRODUCT)
        elif index_type == IndexType.IVFPQ:
            quantizer = faiss.IndexFlatIP(self.embed_dim)
            index = faiss.IndexIVFPQ(quantizer, self.embed_dim, self.config.n_lists, 
                                   self.config.nbits, 8, faiss.METRIC_INNER_PRODUCT)
            if not index.is_trained:
                index.train(np.random.random((1000, self.embed_dim)).astype('float32'))
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Wrap with GPU if available
        if self.config.use_gpu and hasattr(self, 'gpu_resources'):
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
        
        return index
    
    def build_index(self, 
                   embeddings: np.ndarray, 
                   id_map: List[str], 
                   index_type: Optional[IndexType] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> faiss.Index:
        """
        Build a FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings (n_samples, embed_dim)
            id_map: List of IDs corresponding to each embedding
            index_type: Type of index to create (uses config default if None)
            metadata: Additional metadata to store with the index
            
        Returns:
            Built FAISS index
        """
        if len(embeddings) == 0:
            logger.warning("No embeddings provided, returning empty index")
            return faiss.IndexFlatIP(self.embed_dim)
        
        # Validate input
        if len(embeddings) != len(id_map):
            raise ValueError("Number of embeddings must match number of IDs")
        
        if embeddings.shape[1] != self.embed_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embed_dim}, got {embeddings.shape[1]}")
        
        # Normalize embeddings if requested
        if self.config.normalize_embeddings:
            embeddings = self._normalize_embeddings(embeddings)
        
        # Create index
        index_type = index_type or self.config.index_type
        index = self._create_index(index_type)
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        # Store metadata
        if metadata:
            metadata.update({
                'index_type': index_type.value,
                'n_embeddings': len(embeddings),
                'embed_dim': self.embed_dim,
                'created_at': pd.Timestamp.now().isoformat()
            })
        
        return index
    
    def add_embeddings(self, 
                      embeddings: np.ndarray, 
                      id_map: List[str], 
                      index: faiss.Index,
                      metadata: Optional[Dict[str, Any]] = None) -> faiss.Index:
        """
        Add embeddings to an existing index.
        
        Args:
            embeddings: Numpy array of embeddings to add
            id_map: List of IDs corresponding to new embeddings
            index: Existing FAISS index to add to
            metadata: Metadata to update
            
        Returns:
            Updated FAISS index
        """
        if len(embeddings) == 0:
            return index
        
        # Validate input
        if embeddings.shape[1] != self.embed_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embed_dim}, got {embeddings.shape[1]}")
        
        # Normalize embeddings if requested
        if self.config.normalize_embeddings:
            embeddings = self._normalize_embeddings(embeddings)
        
        # Add to index
        index.add(embeddings.astype('float32'))
        
        # Update metadata
        if metadata:
            metadata['n_embeddings'] = index.ntotal
            metadata['last_updated'] = pd.Timestamp.now().isoformat()
        
        return index
    
    def semantic_search(self, 
                       query_embeddings: Union[np.ndarray, List[float]], 
                       index: faiss.Index,
                       id_map: List[str],
                       k: Optional[int] = None,
                       similarity_threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Perform semantic search using query embeddings.
        
        Args:
            query_embeddings: Query embedding(s) - single vector or batch
            index: FAISS index to search
            id_map: List of IDs corresponding to index entries
            k: Number of results to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of (id, similarity_score) tuples
        """
        if index is None or index.ntotal == 0:
            return []
        
        k = k or self.config.k
        similarity_threshold = similarity_threshold or self.config.similarity_threshold
        
        # Convert query to numpy array
        if isinstance(query_embeddings, list):
            query_embeddings = np.array([query_embeddings])
        elif isinstance(query_embeddings, np.ndarray) and query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Normalize query embeddings if requested
        if self.config.normalize_embeddings:
            query_embeddings = self._normalize_embeddings(query_embeddings)
        
        # Search index
        if isinstance(index, faiss.IndexIVF) and hasattr(index, 'nprobe'):
            index.nprobe = self.config.n_probes
        
        distances, indices = index.search(query_embeddings.astype('float32'), min(k, index.ntotal))
        
        # Process results
        results = []
        for batch_idx, (batch_distances, batch_indices) in enumerate(zip(distances, indices)):
            for dist, idx in zip(batch_distances, batch_indices):
                if idx == -1:  # FAISS returns -1 for no match
                    continue
                
                similarity = float(dist)
                if similarity >= similarity_threshold:
                    entity_id = id_map[idx]
                    results.append((entity_id, similarity))
        
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def batch_search(self, 
                    query_embeddings: np.ndarray,
                    index: faiss.Index,
                    id_map: List[str],
                    k: Optional[int] = None,
                    similarity_threshold: Optional[float] = None,
                    batch_size: Optional[int] = None) -> List[List[Tuple[str, float]]]:
        """
        Perform batch semantic search for multiple query embeddings.
        
        Args:
            query_embeddings: Array of query embeddings (n_queries, embed_dim)
            index: FAISS index to search
            id_map: List of IDs corresponding to index entries
            k: Number of results per query
            similarity_threshold: Minimum similarity score threshold
            batch_size: Batch size for processing
            
        Returns:
            List of result lists, one per query
        """
        if query_embeddings.shape[0] == 0:
            return []
        
        batch_size = batch_size or self.config.batch_size
        k = k or self.config.k
        similarity_threshold = similarity_threshold or self.config.similarity_threshold
        
        # Process in batches
        all_results = []
        for i in range(0, query_embeddings.shape[0], batch_size):
            batch_embeddings = query_embeddings[i:i + batch_size]
            batch_results = self.semantic_search(
                batch_embeddings, index, id_map, k, similarity_threshold
            )
            all_results.append(batch_results)
        
        return all_results
    
    def save_index(self, 
                  index: faiss.Index, 
                  id_map: List[str], 
                  metadata: Dict[str, Any],
                  filepath: str) -> None:
        """
        Save FAISS index and associated data to disk.
        
        Args:
            index: FAISS index to save
            id_map: List of IDs corresponding to index entries
            metadata: Metadata to save
            filepath: Path to save the index
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create index directory if it doesn't exist
        index_dir = Path(filepath)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, str(index_dir / "index.faiss"))
        
        # Save ID map and metadata
        data = {
            'id_map': id_map,
            'metadata': metadata,
            'embed_dim': self.embed_dim,
            'config': self.config.__dict__
        }
        
        with open(index_dir / "index_data.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved index to {filepath}")
    
    def load_index(self, filepath: str) -> Tuple[faiss.Index, List[str], Dict[str, Any]]:
        """
        Load FAISS index and associated data from disk.
        
        Args:
            filepath: Path to the saved index
            
        Returns:
            Tuple of (index, id_map, metadata)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index not found at {filepath}")
        
        # Load FAISS index
        index = faiss.read_index(str(Path(filepath) / "index.faiss"))
        
        # Load ID map and metadata
        with open(Path(filepath) / "index_data.pkl", 'rb') as f:
            data = pickle.load(f)
        
        # Update config if needed
        loaded_config = data.get('config', {})
        for key, value in loaded_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        logger.info(f"Loaded index from {filepath}")
        return index, data['id_map'], data['metadata']
    
    def resize_index(self, 
                    index: faiss.Index, 
                    target_size: int,
                    id_map: List[str],
                    embeddings: Optional[np.ndarray] = None) -> Tuple[faiss.Index, List[str]]:
        """
        Resize index to accommodate more embeddings.
        
        Args:
            index: Current FAISS index
            target_size: Target number of embeddings
            id_map: Current ID map
            embeddings: New embeddings to add (optional)
            
        Returns:
            Tuple of (resized_index, resized_id_map)
        """
        current_size = index.ntotal
        
        if current_size >= target_size:
            return index, id_map
        
        # Create new index
        new_index = self._create_index(self.config.index_type)
        
        # Get current embeddings
        current_embeddings = np.zeros((current_size, self.embed_dim)).astype('float32')
        index.reconstruct_n(0, current_size, current_embeddings)
        
        # Add to new index
        new_index.add(current_embeddings)
        
        # Add new embeddings if provided
        if embeddings is not None:
            new_index = self.add_embeddings(embeddings, id_map[current_size:], new_index)
        
        return new_index, id_map
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def get_index_stats(self, index: faiss.Index, id_map: List[str]) -> Dict[str, Any]:
        """Get statistics about the index"""
        if index is None:
            return {'status': 'empty'}
        
        return {
            'n_total': index.ntotal,
            'n_dimensions': self.embed_dim,
            'index_type': type(index).__name__,
            'id_map_size': len(id_map),
            'is_trained': getattr(index, 'is_trained', None),
            'n_lists': getattr(index, 'nlist', None),
            'n_probes': getattr(index, 'nprobe', None)
        }
    
    def clear_index(self, index_type: str = 'all') -> None:
        """Clear one or all indices"""
        if index_type in ['all', 'nodes']:
            self.node_index = None
            self.node_id_map = []
            self.node_index_metadata = {}
        
        if index_type in ['all', 'edges']:
            self.edge_index = None
            self.edge_id_map = []
            self.edge_index_metadata = {}
        
        if index_type in ['all', 'communities']:
            self.community_index = None
            self.community_id_map = []
            self.community_index_metadata = {}
        
        logger.info(f"Cleared {index_type} indices")