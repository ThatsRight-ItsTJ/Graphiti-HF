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
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, HfFolder
from pydantic import BaseModel

from graphiti_core.driver.driver import (
    COMMUNITY_INDEX_NAME,
    ENTITY_EDGE_INDEX_NAME,
    ENTITY_INDEX_NAME,
    EPISODE_INDEX_NAME,
    GraphDriver,
    GraphProvider,
    GraphDriverSession,
)
from graphiti_core.embedder import EMBEDDING_DIM
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError
from graphiti_core.nodes import (
    CommunityNode,
    EpisodicNode,
    EntityNode,
    EpisodeType,
    get_community_node_from_record,
    get_entity_node_from_record,
    get_episodic_node_from_record,
)
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    get_community_edge_from_record,
    get_entity_edge_from_record,
    get_episodic_edge_from_record,
)

# Import search components
from graphiti_hf.search.vector_search import VectorSearchEngine, SearchConfig, IndexType
from graphiti_hf.search.graph_traversal import GraphTraversalEngine, TraversalConfig, TraversalAlgorithm, EdgeFilterType
from graphiti_hf.search.hybrid_search import HybridSearchEngine, HybridSearchConfig

logger = logging.getLogger(__name__)


class DatasetSchema(BaseModel):
    """Schema definition for HuggingFace Datasets"""
    name: str
    features: Dict[str, Any]
    description: str


class HuggingFaceDriverSession(GraphDriverSession):
    """Session for HuggingFaceDriver operations"""
    
    def __init__(self, driver: 'HuggingFaceDriver'):
        self.driver = driver
    
    async def __aexit__(self, exc_type, exc, tb):
        pass
    
    async def run(self, query: str, **kwargs: Any) -> Any:
        """Execute a query (not implemented for HuggingFace datasets)"""
        raise NotImplementedError("HuggingFaceDriver does not support arbitrary queries")
    
    async def close(self):
        """Close the session"""
        pass
    
    async def execute_write(self, func, *args, **kwargs):
        """Execute a write operation"""
        return await func(*args, **kwargs)


class HuggingFaceDriver(GraphDriver):
    """
    HuggingFace Driver for Graphiti
    
    Uses HuggingFace Datasets as the storage backend for knowledge graphs.
    Data is stored as pandas DataFrames in HuggingFace datasets.
    Includes FAISS-based vector search capabilities.
    """
    
    provider = GraphProvider.NEO4J  # Using NEO4J as closest match
    fulltext_syntax = ""
    
    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        create_repo: bool = False,
        enable_vector_search: bool = True,
        vector_search_config: Optional[SearchConfig] = None,
        enable_performance_optimization: bool = True,
        performance_optimizer_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the HuggingFaceDriver
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "username/knowledge-graph")
            token: HuggingFace access token (if None, uses from environment)
            private: Whether the repository should be private
            create_repo: Whether to create the repository if it doesn't exist
            enable_vector_search: Whether to enable FAISS vector search
            vector_search_config: Configuration for vector search
            enable_performance_optimization: Whether to enable performance optimization
            performance_optimizer_config: Configuration for performance optimizer
            **kwargs: Additional arguments
        """
        self.repo_id = repo_id
        self.token = token or HfFolder.get_token()
        self.private = private
        self.create_repo = create_repo
        self._database = "default"
        self.enable_vector_search = enable_vector_search
        
        # Initialize datasets
        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        self.episodes_df = pd.DataFrame()
        self.communities_df = pd.DataFrame()
        
        # Vector search engine
        self.vector_search_engine = None
        if enable_vector_search:
            self.vector_search_engine = VectorSearchEngine(
                embed_dim=EMBEDDING_DIM,
                config=vector_search_config or SearchConfig()
            )
        
        # Load or create datasets
        self._load_or_create_datasets()
        
        # Build vector indices if enabled
        if enable_vector_search:
            self._build_vector_indices()
        
        # Initialize graph traversal engine
        self.traversal_engine = GraphTraversalEngine(self)
        
        # Initialize hybrid search engine
        self.hybrid_search_engine = HybridSearchEngine(self)
    
    def _load_or_create_datasets(self):
        """Load existing datasets from HuggingFace or create new ones"""
        try:
            dataset_dict = load_dataset(self.repo_id)
            
            # Load existing datasets
            if 'nodes' in dataset_dict:
                self.nodes_df = dataset_dict['nodes'].to_pandas()
            if 'edges' in dataset_dict:
                self.edges_df = dataset_dict['edges'].to_pandas()
            if 'episodes' in dataset_dict:
                self.episodes_df = dataset_dict['episodes'].to_pandas()
            if 'communities' in dataset_dict:
                self.communities_df = dataset_dict['communities'].to_pandas()
                
        except Exception as e:
            logger.warning(f"Could not load dataset from {self.repo_id}: {e}")
            logger.info("Creating new datasets...")
            
            # Create empty datasets
            self._create_empty_datasets()
    
    def _create_empty_datasets(self):
        """Create empty datasets with proper schemas"""
        # Node dataset schema
        nodes_schema = DatasetSchema(
            name="nodes",
            features={
                "uuid": "string",
                "name": "string",
                "group_id": "string",
                "labels": "string",  # JSON array as string
                "created_at": "timestamp[s]",
                "name_embedding": "list,float32",
                "summary": "string",
                "attributes": "string"  # JSON object as string
            },
            description="Entity nodes in the knowledge graph"
        )
        
        # Edge dataset schema
        edges_schema = DatasetSchema(
            name="edges",
            features={
                "uuid": "string",
                "source_uuid": "string",
                "target_uuid": "string",
                "name": "string",
                "fact": "string",
                "group_id": "string",
                "created_at": "timestamp[s]",
                "fact_embedding": "list,float32",
                "episodes": "string",  # JSON array as string
                "expired_at": "timestamp[s]",
                "valid_at": "timestamp[s]",
                "invalid_at": "timestamp[s]",
                "attributes": "string"  # JSON object as string
            },
            description="Entity edges in the knowledge graph"
        )
        
        # Episode dataset schema
        episodes_schema = DatasetSchema(
            name="episodes",
            features={
                "uuid": "string",
                "name": "string",
                "content": "string",
                "source": "string",
                "source_description": "string",
                "group_id": "string",
                "created_at": "timestamp[s]",
                "valid_at": "timestamp[s]",
                "entity_edges": "string"  # JSON array as string
            },
            description="Episodic nodes in the knowledge graph"
        )
        
        # Community dataset schema
        communities_schema = DatasetSchema(
            name="communities",
            features={
                "uuid": "string",
                "name": "string",
                "group_id": "string",
                "created_at": "timestamp[s]",
                "name_embedding": "list,float32",
                "summary": "string"
            },
            description="Community nodes in the knowledge graph"
        )
        
        # Create empty DataFrames
        self.nodes_df = pd.DataFrame(columns=list(nodes_schema.features.keys()))
        self.edges_df = pd.DataFrame(columns=list(edges_schema.features.keys()))
        self.episodes_df = pd.DataFrame(columns=list(episodes_schema.features.keys()))
        self.communities_df = pd.DataFrame(columns=list(communities_schema.features.keys()))
        
        # Push to hub
        self._push_to_hub("Initialized empty datasets")
    
    def _build_vector_indices(self):
        """Build FAISS indices for vector search"""
        if not self.vector_search_engine:
            return
        
        # Build node index
        if not self.nodes_df.empty and 'name_embedding' in self.nodes_df.columns:
            node_embeddings = self.nodes_df['name_embedding'].dropna()
            if not node_embeddings.empty:
                embedding_matrix = np.array(node_embeddings.tolist()).astype('float32')
                self.vector_search_engine.node_index = self.vector_search_engine.build_index(
                    embedding_matrix, 
                    self.nodes_df[node_embeddings.index]['uuid'].tolist(),
                    metadata={'entity_type': 'node'}
                )
                self.vector_search_engine.node_id_map = self.nodes_df[node_embeddings.index]['uuid'].tolist()
                self.vector_search_engine.node_index_metadata = {'entity_type': 'node'}
        
        # Build edge index
        if not self.edges_df.empty and 'fact_embedding' in self.edges_df.columns:
            edge_embeddings = self.edges_df['fact_embedding'].dropna()
            if not edge_embeddings.empty:
                embedding_matrix = np.array(edge_embeddings.tolist()).astype('float32')
                self.vector_search_engine.edge_index = self.vector_search_engine.build_index(
                    embedding_matrix,
                    self.edges_df[edge_embeddings.index]['uuid'].tolist(),
                    metadata={'entity_type': 'edge'}
                )
                self.vector_search_engine.edge_id_map = self.edges_df[edge_embeddings.index]['uuid'].tolist()
                self.vector_search_engine.edge_index_metadata = {'entity_type': 'edge'}
        
        # Build community index
        if not self.communities_df.empty and 'name_embedding' in self.communities_df.columns:
            community_embeddings = self.communities_df['name_embedding'].dropna()
            if not community_embeddings.empty:
                embedding_matrix = np.array(community_embeddings.tolist()).astype('float32')
                self.vector_search_engine.community_index = self.vector_search_engine.build_index(
                    embedding_matrix,
                    self.communities_df[community_embeddings.index]['uuid'].tolist(),
                    metadata={'entity_type': 'community'}
                )
                self.vector_search_engine.community_id_map = self.communities_df[community_embeddings.index]['uuid'].tolist()
                self.vector_search_engine.community_index_metadata = {'entity_type': 'community'}
        
        logger.info("Built vector search indices")
    
    def _update_vector_indices(self, entity_type: str, embeddings: np.ndarray, uuids: List[str]):
        """Update vector indices with new embeddings"""
        if not self.vector_search_engine:
            return
        
        if entity_type == 'node' and self.vector_search_engine.node_index:
            self.vector_search_engine.node_index = self.vector_search_engine.add_embeddings(
                embeddings, uuids, self.vector_search_engine.node_index,
                self.vector_search_engine.node_index_metadata
            )
            self.vector_search_engine.node_id_map.extend(uuids)
        elif entity_type == 'edge' and self.vector_search_engine.edge_index:
            self.vector_search_engine.edge_index = self.vector_search_engine.add_embeddings(
                embeddings, uuids, self.vector_search_engine.edge_index,
                self.vector_search_engine.edge_index_metadata
            )
            self.vector_search_engine.edge_id_map.extend(uuids)
        elif entity_type == 'community' and self.vector_search_engine.community_index:
            self.vector_search_engine.community_index = self.vector_search_engine.add_embeddings(
                embeddings, uuids, self.vector_search_engine.community_index,
                self.vector_search_engine.community_index_metadata
            )
            self.vector_search_engine.community_id_map.extend(uuids)
    
    def _push_to_hub(self, commit_message: str):
        """Push datasets to HuggingFace Hub"""
        try:
            # Create dataset dict
            dataset_dict = DatasetDict()
            
            if not self.nodes_df.empty:
                dataset_dict['nodes'] = Dataset.from_pandas(self.nodes_df)
            if not self.edges_df.empty:
                dataset_dict['edges'] = Dataset.from_pandas(self.edges_df)
            if not self.episodes_df.empty:
                dataset_dict['episodes'] = Dataset.from_pandas(self.episodes_df)
            if not self.communities_df.empty:
                dataset_dict['communities'] = Dataset.from_pandas(self.communities_df)
            
            # Push to hub
            dataset_dict.push_to_hub(
                repo_id=self.repo_id,
                token=self.token,
                private=self.private,
                commit_message=commit_message
            )
            
        except Exception as e:
            logger.error(f"Failed to push datasets to hub: {e}")
            raise
    
    def session(self, database: str | None = None) -> HuggingFaceDriverSession:
        """Create a session for the driver"""
        return HuggingFaceDriverSession(self)
    
    def close(self):
        """Close the driver"""
        pass
    
    async def execute_query(self, query: str, **kwargs: Any) -> Any:
        """Execute a query (not implemented for HuggingFace datasets)"""
        raise NotImplementedError("HuggingFaceDriver does not support arbitrary queries")
    
    async def delete_all_indexes(self) -> Any:
        """Delete all datasets"""
        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        self.episodes_df = pd.DataFrame()
        self.communities_df = pd.DataFrame()
        
        # Clear vector indices
        if self.vector_search_engine:
            self.vector_search_engine.clear_index()
        
        self._push_to_hub("Deleted all datasets")
    
    # Node operations
    async def save_node(self, node: Union[EntityNode, EpisodicNode, CommunityNode]) -> Any:
        """Save a node to the dataset"""
        if isinstance(node, EntityNode):
            df = self.nodes_df
            node_data = {
                'uuid': node.uuid,
                'name': node.name,
                'group_id': node.group_id,
                'labels': json.dumps(node.labels),
                'created_at': node.created_at,
                'name_embedding': node.name_embedding,
                'summary': node.summary,
                'attributes': json.dumps(node.attributes)
            }
        elif isinstance(node, EpisodicNode):
            df = self.episodes_df
            node_data = {
                'uuid': node.uuid,
                'name': node.name,
                'content': node.content,
                'source': node.source.value,
                'source_description': node.source_description,
                'group_id': node.group_id,
                'created_at': node.created_at,
                'valid_at': node.valid_at,
                'entity_edges': json.dumps(node.entity_edges)
            }
        elif isinstance(node, CommunityNode):
            df = self.communities_df
            node_data = {
                'uuid': node.uuid,
                'name': node.name,
                'group_id': node.group_id,
                'created_at': node.created_at,
                'name_embedding': node.name_embedding,
                'summary': node.summary
            }
        else:
            raise ValueError(f"Unknown node type: {type(node)}")
        
        # Add to DataFrame
        new_row = pd.DataFrame([node_data])
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Update the DataFrame
        if isinstance(node, EntityNode):
            self.nodes_df = df
            # Update vector index if embedding exists
            if node.name_embedding and self.vector_search_engine:
                embeddings = np.array([node.name_embedding]).astype('float32')
                self._update_vector_indices('node', embeddings, [node.uuid])
        elif isinstance(node, EpisodicNode):
            self.episodes_df = df
        elif isinstance(node, CommunityNode):
            self.communities_df = df
            # Update vector index if embedding exists
            if node.name_embedding and self.vector_search_engine:
                embeddings = np.array([node.name_embedding]).astype('float32')
                self._update_vector_indices('community', embeddings, [node.uuid])
        
        # Push to hub
        self._push_to_hub(f"Saved {type(node).__name__} node {node.uuid}")
        
        return node_data
    
    async def get_node_by_uuid(self, uuid: str, node_type: str = "Entity") -> Union[EntityNode, EpisodicNode, CommunityNode]:
        """Get a node by UUID"""
        if node_type == "Entity":
            df = self.nodes_df
            record = df[df['uuid'] == uuid]
            if record.empty:
                raise NodeNotFoundError(uuid)
            return get_entity_node_from_record(record.iloc[0], self.provider)
        elif node_type == "Episodic":
            df = self.episodes_df
            record = df[df['uuid'] == uuid]
            if record.empty:
                raise NodeNotFoundError(uuid)
            return get_episodic_node_from_record(record.iloc[0])
        elif node_type == "Community":
            df = self.communities_df
            record = df[df['uuid'] == uuid]
            if record.empty:
                raise NodeNotFoundError(uuid)
            return get_community_node_from_record(record.iloc[0])
        else:
            raise ValueError(f"Unknown node type: {node_type}")
    
    async def get_nodes_by_group_ids(self, group_ids: List[str], node_type: str = "Entity", limit: Optional[int] = None) -> List[Union[EntityNode, EpisodicNode, CommunityNode]]:
        """Get nodes by group IDs"""
        if node_type == "Entity":
            df = self.nodes_df[self.nodes_df['group_id'].isin(group_ids)]
        elif node_type == "Episodic":
            df = self.episodes_df[self.episodes_df['group_id'].isin(group_ids)]
        elif node_type == "Community":
            df = self.communities_df[self.communities_df['group_id'].isin(group_ids)]
        else:
            raise ValueError(f"Unknown node type: {node_type}")
        
        if limit:
            df = df.head(limit)
        
        if node_type == "Entity":
            return [get_entity_node_from_record(row, self.provider) for _, row in df.iterrows()]
        elif node_type == "Episodic":
            return [get_episodic_node_from_record(row) for _, row in df.iterrows()]
        elif node_type == "Community":
            return [get_community_node_from_record(row) for _, row in df.iterrows()]
    
    # Edge operations
    async def save_edge(self, edge: Union[EntityEdge, EpisodicEdge, CommunityEdge]) -> Any:
        """Save an edge to the dataset"""
        if isinstance(edge, EntityEdge):
            df = self.edges_df
            edge_data = {
                'uuid': edge.uuid,
                'source_uuid': edge.source_node_uuid,
                'target_uuid': edge.target_node_uuid,
                'name': edge.name,
                'fact': edge.fact,
                'group_id': edge.group_id,
                'created_at': edge.created_at,
                'fact_embedding': edge.fact_embedding,
                'episodes': json.dumps(edge.episodes),
                'expired_at': edge.expired_at,
                'valid_at': edge.valid_at,
                'invalid_at': edge.invalid_at,
                'attributes': json.dumps(edge.attributes)
            }
        elif isinstance(edge, EpisodicEdge):
            # Episodic edges are handled differently in the current implementation
            raise NotImplementedError("EpisodicEdge save not implemented yet")
        elif isinstance(edge, CommunityEdge):
            # Community edges are handled differently in the current implementation
            raise NotImplementedError("CommunityEdge save not implemented yet")
        else:
            raise ValueError(f"Unknown edge type: {type(edge)}")
        
        # Add to DataFrame
        new_row = pd.DataFrame([edge_data])
        df = pd.concat([df, new_row], ignore_index=True)
        self.edges_df = df
        
        # Update vector index if embedding exists
        if edge.fact_embedding and self.vector_search_engine:
            embeddings = np.array([edge.fact_embedding]).astype('float32')
            self._update_vector_indices('edge', embeddings, [edge.uuid])
        
        # Push to hub
        self._push_to_hub(f"Saved {type(edge).__name__} edge {edge.uuid}")
        
        return edge_data
    
    async def get_edge_by_uuid(self, uuid: str, edge_type: str = "Entity") -> Union[EntityEdge, EpisodicEdge, CommunityEdge]:
        """Get an edge by UUID"""
        if edge_type == "Entity":
            df = self.edges_df
            record = df[df['uuid'] == uuid]
            if record.empty:
                raise EdgeNotFoundError(uuid)
            return get_entity_edge_from_record(record.iloc[0], self.provider)
        elif edge_type == "Episodic":
            df = self.edges_df  # Episodic edges are stored differently
            record = df[df['uuid'] == uuid]
            if record.empty:
                raise EdgeNotFoundError(uuid)
            return get_episodic_edge_from_record(record.iloc[0])
        elif edge_type == "Community":
            df = self.edges_df  # Community edges are stored differently
            record = df[df['uuid'] == uuid]
            if record.empty:
                raise EdgeNotFoundError(uuid)
            return get_community_edge_from_record(record.iloc[0])
        else:
            raise ValueError(f"Unknown edge type: {edge_type}")
    
    async def get_edges_by_group_ids(self, group_ids: List[str], edge_type: str = "Entity", limit: Optional[int] = None) -> List[Union[EntityEdge, EpisodicEdge, CommunityEdge]]:
        """Get edges by group IDs"""
        if edge_type == "Entity":
            df = self.edges_df[self.edges_df['group_id'].isin(group_ids)]
        else:
            # For other edge types, we might need different logic
            df = self.edges_df[self.edges_df['group_id'].isin(group_ids)]
        
        if limit:
            df = df.head(limit)
        
        if edge_type == "Entity":
            return [get_entity_edge_from_record(row, self.provider) for _, row in df.iterrows()]
        elif edge_type == "Episodic":
            return [get_episodic_edge_from_record(row) for _, row in df.iterrows()]
        elif edge_type == "Community":
            return [get_community_edge_from_record(row) for _, row in df.iterrows()]
        else:
            return []
    
    # Vector search methods
    async def query_nodes_by_embedding(self, embedding: List[float], k: int = 10, 
                                    similarity_threshold: float = 0.0) -> List[EntityNode]:
        """Find similar nodes using FAISS vector search"""
        if not self.vector_search_engine or not self.vector_search_engine.node_index:
            return []
        
        # Perform vector search
        results = self.vector_search_engine.semantic_search(
            embedding, 
            self.vector_search_engine.node_index,
            self.vector_search_engine.node_id_map,
            k=k,
            similarity_threshold=similarity_threshold
        )
        
        # Convert UUIDs to EntityNode objects
        nodes = []
        for node_uuid, similarity in results:
            try:
                node = await self.get_node_by_uuid(node_uuid, "Entity")
                nodes.append(node)
            except NodeNotFoundError:
                continue
        
        return nodes
    
    async def query_edges_by_embedding(self, embedding: List[float], k: int = 10,
                                    similarity_threshold: float = 0.0) -> List[EntityEdge]:
        """Find similar edges using FAISS vector search"""
        if not self.vector_search_engine or not self.vector_search_engine.edge_index:
            return []
        
        # Perform vector search
        results = self.vector_search_engine.semantic_search(
            embedding,
            self.vector_search_engine.edge_index,
            self.vector_search_engine.edge_id_map,
            k=k,
            similarity_threshold=similarity_threshold
        )
        
        # Convert UUIDs to EntityEdge objects
        edges = []
        for edge_uuid, similarity in results:
            try:
                edge = await self.get_edge_by_uuid(edge_uuid, "Entity")
                edges.append(edge)
            except EdgeNotFoundError:
                continue
        
        return edges
    
    async def query_communities_by_embedding(self, embedding: List[float], k: int = 10,
                                          similarity_threshold: float = 0.0) -> List[CommunityNode]:
        """Find similar communities using FAISS vector search"""
        if not self.vector_search_engine or not self.vector_search_engine.community_index:
            return []
        
        # Perform vector search
        results = self.vector_search_engine.semantic_search(
            embedding,
            self.vector_search_engine.community_index,
            self.vector_search_engine.community_id_map,
            k=k,
            similarity_threshold=similarity_threshold
        )
        
        # Convert UUIDs to CommunityNode objects
        communities = []
        for community_uuid, similarity in results:
            try:
                community = await self.get_node_by_uuid(community_uuid, "Community")
                communities.append(community)
            except NodeNotFoundError:
                continue
        
        return communities
    
    async def batch_query_nodes_by_embedding(self, embeddings: List[List[float]], k: int = 10,
                                           similarity_threshold: float = 0.0) -> List[List[EntityNode]]:
        """Batch query nodes by multiple embeddings"""
        if not self.vector_search_engine or not self.vector_search_engine.node_index:
            return [[] for _ in embeddings]
        
        # Convert to numpy array
        embedding_array = np.array(embeddings).astype('float32')
        
        # Perform batch search
        results = self.vector_search_engine.batch_search(
            embedding_array,
            self.vector_search_engine.node_index,
            self.vector_search_engine.node_id_map,
            k=k,
            similarity_threshold=similarity_threshold
        )
        
        # Convert UUIDs to EntityNode objects
        all_nodes = []
        for result_group in results:
            nodes = []
            for node_uuid, similarity in result_group:
                try:
                    node = await self.get_node_by_uuid(node_uuid, "Entity")
                    nodes.append(node)
                except NodeNotFoundError:
                    continue
            all_nodes.append(nodes)
        
        return all_nodes
    
    async def batch_query_edges_by_embedding(self, embeddings: List[List[float]], k: int = 10,
                                          similarity_threshold: float = 0.0) -> List[List[EntityEdge]]:
        """Batch query edges by multiple embeddings"""
        if not self.vector_search_engine or not self.vector_search_engine.edge_index:
            return [[] for _ in embeddings]
        
        # Convert to numpy array
        embedding_array = np.array(embeddings).astype('float32')
        
        # Perform batch search
        results = self.vector_search_engine.batch_search(
            embedding_array,
            self.vector_search_engine.edge_index,
            self.vector_search_engine.edge_id_map,
            k=k,
            similarity_threshold=similarity_threshold
        )
        
        # Convert UUIDs to EntityEdge objects
        all_edges = []
        for result_group in results:
            edges = []
            for edge_uuid, similarity in result_group:
                try:
                    edge = await self.get_edge_by_uuid(edge_uuid, "Entity")
                    edges.append(edge)
                except EdgeNotFoundError:
                    continue
            all_edges.append(edges)
        
        return all_edges
    
    def get_vector_search_stats(self) -> Dict[str, Any]:
        """Get statistics about vector search indices"""
        if not self.vector_search_engine:
            return {"enabled": False}
        
        stats = {
            "enabled": True,
            "config": self.vector_search_engine.config.__dict__,
            "indices": {}
        }
        
        if self.vector_search_engine.node_index:
            stats["indices"]["nodes"] = self.vector_search_engine.get_index_stats(
                self.vector_search_engine.node_index,
                self.vector_search_engine.node_id_map
            )
        
        if self.vector_search_engine.edge_index:
            stats["indices"]["edges"] = self.vector_search_engine.get_index_stats(
                self.vector_search_engine.edge_index,
                self.vector_search_engine.edge_id_map
            )
        
        if self.vector_search_engine.community_index:
            stats["indices"]["communities"] = self.vector_search_engine.get_index_stats(
                self.vector_search_engine.community_index,
                self.vector_search_engine.community_id_map
            )
        
        return stats
    
    # Utility methods
    def get_nodes_df(self) -> pd.DataFrame:
        """Get nodes DataFrame"""
        return self.nodes_df
    
    def get_edges_df(self) -> pd.DataFrame:
        """Get edges DataFrame"""
        return self.edges_df
    
    def get_episodes_df(self) -> pd.DataFrame:
        """Get episodes DataFrame"""
        return self.episodes_df
    
    def get_communities_df(self) -> pd.DataFrame:
        """Get communities DataFrame"""
        return self.communities_df
    
    def _row_to_node(self, row) -> EntityNode:
        """Convert pandas row to EntityNode"""
        return EntityNode(
            uuid=row['uuid'],
            name=row['name'],
            labels=json.loads(row['labels']) if row['labels'] else [],
            group_id=row['group_id'],
            created_at=datetime.fromisoformat(row['created_at']),
            name_embedding=row['name_embedding'],
            summary=row['summary'],
            attributes=json.loads(row['attributes']) if row['attributes'] else {}
        )
    
    def _row_to_edge(self, row) -> EntityEdge:
        """Convert pandas row to EntityEdge"""
        return EntityEdge(
            uuid=row['uuid'],
            source_node_uuid=row['source_uuid'],
            target_node_uuid=row['target_uuid'],
            fact=row['fact'],
            fact_embedding=row['fact_embedding'],
            episodes=json.loads(row['episodes']) if row['episodes'] else [],
            created_at=datetime.fromisoformat(row['created_at']),
            valid_at=datetime.fromisoformat(row['valid_at']) if row['valid_at'] else None,
            invalidated_at=datetime.fromisoformat(row['invalidated_at']) if row['invalidated_at'] else None
        )
    
    # Graph traversal methods
    async def traverse_graph(
        self,
        start_nodes: List[str],
        algorithm: str = "bfs",
        max_depth: int = 5,
        edge_filter: str = "all",
        edge_types: Optional[List[str]] = None,
        temporal_filter: Optional[datetime] = None,
        early_termination_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Traverse the graph starting from specified nodes.
        
        Args:
            start_nodes: List of starting node UUIDs
            algorithm: Traversal algorithm ("bfs" or "dfs")
            max_depth: Maximum depth for traversal
            edge_filter: Edge filter type ("all", "incoming", "outgoing")
            edge_types: List of edge types to include
            temporal_filter: Filter edges by temporal validity
            early_termination_size: Stop traversal when reaching this many nodes
            
        Returns:
            Dictionary containing traversal results
        """
        # Create traversal configuration
        config = TraversalConfig(
            max_depth=max_depth,
            algorithm=TraversalAlgorithm(algorithm.lower()),
            edge_filter=EdgeFilterType(edge_filter.lower()),
            edge_types=edge_types,
            temporal_filter=temporal_filter,
            early_termination_size=early_termination_size
        )
        
        # Perform traversal
        if algorithm.lower() == "bfs":
            result = await self.traversal_engine.bfs_traversal(start_nodes, config)
        else:
            result = await self.traversal_engine.dfs_traversal(start_nodes, config)
        
        return {
            "nodes": [node.dict() for node in result.nodes],
            "edges": [edge.dict() for edge in result.edges],
            "paths": result.paths,
            "stats": result.traversal_stats
        }
    
    async def find_paths(
        self,
        start_nodes: List[str],
        target_nodes: List[str],
        max_depth: int = 10,
        edge_filter: str = "all",
        edge_types: Optional[List[str]] = None,
        temporal_filter: Optional[datetime] = None
    ) -> List[List[str]]:
        """
        Find paths between start and target nodes.
        
        Args:
            start_nodes: List of starting node UUIDs
            target_nodes: List of target node UUIDs
            max_depth: Maximum path length
            edge_filter: Edge filter type ("all", "incoming", "outgoing")
            edge_types: List of edge types to include
            temporal_filter: Filter edges by temporal validity
            
        Returns:
            List of paths, where each path is a list of node UUIDs
        """
        # Create traversal configuration
        config = TraversalConfig(
            max_depth=max_depth,
            edge_filter=EdgeFilterType(edge_filter.lower()),
            edge_types=edge_types,
            temporal_filter=temporal_filter
        )
        
        # Find paths
        paths = await self.traversal_engine.find_paths(start_nodes, target_nodes, config)
        return paths
    
    async def get_neighbors(
        self,
        node_uuids: List[str],
        depth: int = 1,
        edge_filter: str = "all",
        edge_types: Optional[List[str]] = None,
        temporal_filter: Optional[datetime] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get direct neighbors of nodes.
        
        Args:
            node_uuids: List of node UUIDs
            depth: Depth of neighbor search (1 = direct neighbors only)
            edge_filter: Edge filter type ("all", "incoming", "outgoing")
            edge_types: List of edge types to include
            temporal_filter: Filter edges by temporal validity
            
        Returns:
            Dictionary mapping node UUIDs to lists of neighboring edge information
        """
        # Create traversal configuration
        config = TraversalConfig(
            max_depth=depth,
            edge_filter=EdgeFilterType(edge_filter.lower()),
            edge_types=edge_types,
            temporal_filter=temporal_filter
        )
        
        # Get neighbors
        neighbor_edges = await self.traversal_engine.get_neighbors(node_uuids, depth, config)
        
        # Convert to serializable format
        result = {}
        for node_uuid, edges in neighbor_edges.items():
            result[node_uuid] = [
                {
                    "edge": edge.dict(),
                    "source_node": edge.source_node_uuid,
                    "target_node": edge.target_node_uuid
                }
                for edge in edges
            ]
        
        return result
    
    async def extract_subgraph(
        self,
        node_uuids: List[str],
        max_depth: int = 3,
        edge_filter: str = "all",
        edge_types: Optional[List[str]] = None,
        temporal_filter: Optional[datetime] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract connected subgraph containing specified nodes.
        
        Args:
            node_uuids: List of node UUIDs to include in subgraph
            max_depth: Maximum depth for subgraph extraction
            edge_filter: Edge filter type ("all", "incoming", "outgoing")
            edge_types: List of edge types to include
            temporal_filter: Filter edges by temporal validity
            
        Returns:
            Dictionary containing nodes and edges in the subgraph
        """
        # Create traversal configuration
        config = TraversalConfig(
            max_depth=max_depth,
            edge_filter=EdgeFilterType(edge_filter.lower()),
            edge_types=edge_types,
            temporal_filter=temporal_filter
        )
        
        # Extract subgraph
        nodes, edges = await self.traversal_engine.subgraph_extraction(node_uuids, config)
        
        return {
            "nodes": [node.dict() for node in nodes],
            "edges": [edge.dict() for edge in edges]
        }
    
    async def batch_traversal(
        self,
        start_node_groups: List[List[str]],
        algorithm: str = "bfs",
        max_depth: int = 5,
        edge_filter: str = "all",
        edge_types: Optional[List[str]] = None,
        temporal_filter: Optional[datetime] = None,
        early_termination_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform batch traversal operations for multiple groups of start nodes.
        
        Args:
            start_node_groups: List of start node groups
            algorithm: Traversal algorithm ("bfs" or "dfs")
            max_depth: Maximum depth for traversal
            edge_filter: Edge filter type ("all", "incoming", "outgoing")
            edge_types: List of edge types to include
            temporal_filter: Filter edges by temporal validity
            early_termination_size: Stop traversal when reaching this many nodes
            
        Returns:
            List of traversal result dictionaries
        """
        # Create traversal configuration
        config = TraversalConfig(
            max_depth=max_depth,
            algorithm=TraversalAlgorithm(algorithm.lower()),
            edge_filter=EdgeFilterType(edge_filter.lower()),
            edge_types=edge_types,
            temporal_filter=temporal_filter,
            early_termination_size=early_termination_size
        )
        
        # Perform batch traversal
        results = await self.traversal_engine.batch_traversal(start_node_groups, config)
        
        # Convert to serializable format
        output_results = []
        for result in results:
            output_results.append({
                "nodes": [node.dict() for node in result.nodes],
                "edges": [edge.dict() for edge in result.edges],
                "paths": result.paths,
                "stats": result.traversal_stats
            })
        
        return output_results
    
    def get_traversal_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the graph traversal engine.
        
        Returns:
            Dictionary containing traversal engine statistics
        """
        return self.traversal_engine.get_traversal_stats()
    
    def clear_traversal_cache(self):
        """Clear the traversal engine cache"""
        self.traversal_engine.clear_cache()
    
    # Hybrid search methods
    async def search_hybrid(
        self,
        query: str,
        limit: int = 10,
        semantic_weight: float = 0.4,
        keyword_weight: float = 0.3,
        graph_weight: float = 0.3,
        center_node_uuid: Optional[str] = None,
        temporal_filter: Optional[datetime] = None,
        edge_types: Optional[List[str]] = None,
        embedder_model: str = "all-MiniLM-L6-v2"
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic, keyword, and graph-based ranking.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            semantic_weight: Weight for semantic search (0.0-1.0)
            keyword_weight: Weight for keyword search (0.0-1.0)
            graph_weight: Weight for graph-based ranking (0.0-1.0)
            center_node_uuid: Optional center node for graph-based ranking
            temporal_filter: Filter results by temporal validity
            edge_types: List of edge types to include in search
            embedder_model: Sentence transformer model for semantic search
            
        Returns:
            List of search results with combined scores and individual method scores
        """
        # Create hybrid search configuration
        config = HybridSearchConfig(
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            graph_weight=graph_weight,
            result_limit=limit,
            center_node_uuid=center_node_uuid,
            temporal_filter=temporal_filter,
            edge_types=edge_types
        )
        
        # Perform hybrid search
        results = await self.hybrid_search_engine.hybrid_search(query, config)
        
        return results
    
    async def search_with_center(
        self,
        query: str,
        center_node_uuid: str,
        limit: int = 10,
        semantic_weight: float = 0.4,
        keyword_weight: float = 0.3,
        graph_weight: float = 0.3,
        temporal_filter: Optional[datetime] = None,
        edge_types: Optional[List[str]] = None,
        embedder_model: str = "all-MiniLM-L6-v2"
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with center-node based graph ranking.
        
        This method performs a hybrid search where graph-based ranking is centered
        around a specific node, providing results that are both semantically relevant
        and graphically close to the specified center node.
        
        Args:
            query: Search query string
            center_node_uuid: UUID of the center node for graph-based ranking
            limit: Maximum number of results to return
            semantic_weight: Weight for semantic search (0.0-1.0)
            keyword_weight: Weight for keyword search (0.0-1.0)
            graph_weight: Weight for graph-based ranking (0.0-1.0)
            temporal_filter: Filter results by temporal validity
            edge_types: List of edge types to include in search
            embedder_model: Sentence transformer model for semantic search
            
        Returns:
            List of search results with combined scores and individual method scores
        """
        # Validate center node exists
        try:
            await self.get_node_by_uuid(center_node_uuid, "Entity")
        except NodeNotFoundError:
            raise ValueError(f"Center node {center_node_uuid} not found in the graph")
        
        # Create hybrid search configuration with center node
        config = HybridSearchConfig(
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            graph_weight=graph_weight,
            result_limit=limit,
            center_node_uuid=center_node_uuid,
            temporal_filter=temporal_filter,
            edge_types=edge_types
        )
        
        # Perform hybrid search
        results = await self.hybrid_search_engine.hybrid_search(query, config)
        
        return results
    
    async def batch_search_hybrid(
        self,
        queries: List[str],
        limit: int = 10,
        semantic_weight: float = 0.4,
        keyword_weight: float = 0.3,
        graph_weight: float = 0.3,
        center_node_uuid: Optional[str] = None,
        temporal_filter: Optional[datetime] = None,
        edge_types: Optional[List[str]] = None,
        embedder_model: str = "all-MiniLM-L6-v2"
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform batch hybrid search for multiple queries.
        
        Args:
            queries: List of search query strings
            limit: Maximum number of results per query
            semantic_weight: Weight for semantic search (0.0-1.0)
            keyword_weight: Weight for keyword search (0.0-1.0)
            graph_weight: Weight for graph-based ranking (0.0-1.0)
            center_node_uuid: Optional center node for graph-based ranking
            temporal_filter: Filter results by temporal validity
            edge_types: List of edge types to include in search
            embedder_model: Sentence transformer model for semantic search
            
        Returns:
            List of search result lists, one per query
        """
        # Create hybrid search configuration
        config = HybridSearchConfig(
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            graph_weight=graph_weight,
            result_limit=limit,
            center_node_uuid=center_node_uuid,
            temporal_filter=temporal_filter,
            edge_types=edge_types
        )
        
        # Perform batch hybrid search
        results = await self.hybrid_search_engine.batch_hybrid_search(queries, config)
        
        return results
    
    def get_hybrid_search_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the hybrid search engine.
        
        Returns:
            Dictionary containing hybrid search engine statistics
        """
        return self.hybrid_search_engine.get_search_stats()
    
    def rebuild_hybrid_search_indices(self):
        """Rebuild hybrid search indices"""
        self.hybrid_search_engine.rebuild_text_indices()
        logger.info("Rebuilt hybrid search indices")
    
    # Performance optimization methods
    def optimize_search_performance(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Optimize search performance for the HuggingFaceDriver.
        
        Args:
            force_rebuild: Whether to force full index rebuild
            
        Returns:
            Dictionary containing optimization results
        """
        if not self.performance_optimizer:
            return {"error": "Performance optimization not enabled"}
        
        if force_rebuild:
            return self.performance_optimizer.rebuild_all_indices()
        else:
            return self.performance_optimizer.optimize_index_parameters()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the HuggingFaceDriver.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.performance_optimizer:
            return {"error": "Performance optimization not enabled"}
        
        return self.performance_optimizer.get_index_statistics()
    
    async def auto_rebuild_indices(self, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Automatically rebuild indices based on performance threshold.
        
        Args:
            threshold: Performance threshold (0.0-1.0)
            
        Returns:
            Dictionary containing rebuild results
        """
        if not self.performance_optimizer:
            return {"error": "Performance optimization not enabled"}
        
        # Check if rebuild is needed
        stats = self.performance_optimizer.get_index_statistics()
        
        # Simple heuristic: rebuild if data size has changed significantly
        current_edge_count = len(self.edges_df)
        current_node_count = len(self.nodes_df)
        
        # Trigger rebuild if significant changes detected
        if (current_edge_count > 10000 or current_node_count > 5000):
            return self.performance_optimizer.rebuild_all_indices()
        else:
            return {
                "action": "no_rebuild_needed",
                "reason": "Data size below threshold",
                "current_edge_count": current_edge_count,
                "current_node_count": current_node_count,
                "threshold": threshold
            }