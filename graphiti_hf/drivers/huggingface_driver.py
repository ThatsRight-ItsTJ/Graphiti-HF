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
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, HfFolder
from pydantic import BaseModel

# Import custom types
from graphiti_hf.models.custom_types import (
    CustomTypeManager,
    create_custom_entity,
    create_custom_edge,
    validate_entity_properties,
    validate_edge_properties,
    serialize_custom_type,
    deserialize_custom_type,
    get_type_manager,
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
)

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
from graphiti_hf.search.advanced_config import AdvancedSearchConfig, SearchMethod, RankingStrategy

# Import incremental updater, concurrency manager, temporal manager, and community detector
from graphiti_hf.processing.incremental_updater import IncrementalUpdater, Delta, DeltaOperation, DeltaOperationType, DeltaEntityType
from graphiti_hf.processing.concurrency_manager import ConcurrencyManager, BranchInfo, MergeResult, Transaction, TransactionStatus
from graphiti_hf.processing.temporal_manager import TemporalManager, TemporalQueryFilter, TemporalStats, TemporalConflict, TemporalResolutionStrategy
from graphiti_hf.analysis.community_detector import CommunityDetector, CommunityDetectionConfig, CommunityStats

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
        enable_advanced_search: bool = True,
        advanced_search_config: Optional[AdvancedSearchConfig] = None,
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
        
        # Advanced search configuration
        self.enable_advanced_search = enable_advanced_search
        self.advanced_search_config = advanced_search_config or AdvancedSearchConfig()
        
        # Load or create datasets
        self._load_or_create_datasets()
        
        # Build vector indices if enabled
        if enable_vector_search:
            self._build_vector_indices()
        
        # Initialize graph traversal engine
        self.traversal_engine = GraphTraversalEngine(self)
        
        # Initialize hybrid search engine
        self.hybrid_search_engine = HybridSearchEngine(self)
        
        # Initialize advanced search manager if enabled
        if enable_advanced_search:
            self.advanced_search_manager = AdvancedSearchManager(self)
        
        # Initialize incremental updater
        self.incremental_updater = IncrementalUpdater(self)
        
        # Initialize concurrency manager
        self.concurrency_manager = ConcurrencyManager(
            repo_id=self.repo_id,
            token=self.token,
            default_branch="main"
        )
        
        # Initialize temporal manager
        self.temporal_manager = TemporalManager(self)
        
        # Initialize community detector
        self.community_detector = CommunityDetector(self)
    
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
        if not hasattr(self, 'performance_optimizer') or not self.performance_optimizer:
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
        if not hasattr(self, 'performance_optimizer') or not self.performance_optimizer:
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
        if not hasattr(self, 'performance_optimizer') or not self.performance_optimizer:
            return {"error": "Performance optimization not enabled"}
        
        # Check if rebuild is needed
        stats = self.performance_optimizer.get_index_statistics()
        
        # Simple heuristic: rebuild if data size has changed significantly
        current_edge_count = len(self.edges_df)
        current_node_count = len(self.nodes_df)
        
        # Trigger rebuild if significant changes detected
        if (current_edge_count > 10000 or current_node_count > 5000):
            return self.performance_optimizer.rebuild_all_indices() if hasattr(self, 'performance_optimizer') else {"error": "Performance optimization not enabled"}
        else:
            return {
                "action": "no_rebuild_needed",
                "reason": "Data size below threshold",
                "current_edge_count": current_edge_count,
                "current_node_count": current_node_count,
                "threshold": threshold
            }
    
    # Incremental Update Methods
    async def add_entity_incremental(self, node: EntityNode, group_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Add a single entity incrementally.
        
        Args:
            node: EntityNode to add
            group_ids: Optional list of group IDs to filter by
            
        Returns:
            Dictionary containing update results
        """
        try:
            result = await self.incremental_updater.add_entities_incremental([node], group_ids)
            return result
        except Exception as e:
            logger.error(f"Error adding entity incrementally: {e}")
            return {"added": 0, "skipped": 0, "errors": [str(e)]}
    
    async def add_edge_incremental(self, edge: EntityEdge, group_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Add a single edge incrementally.
        
        Args:
            edge: EntityEdge to add
            group_ids: Optional list of group IDs to filter by
            
        Returns:
            Dictionary containing update results
        """
        try:
            result = await self.incremental_updater.add_edges_incremental([edge], group_ids)
            return result
        except Exception as e:
            logger.error(f"Error adding edge incrementally: {e}")
            return {"added": 0, "skipped": 0, "errors": [str(e)]}
    
    
    async def create_delta(self, operations: Optional[List[Dict[str, Any]]] = None) -> Delta:
        """
        Create a new delta with optional operations.
        
        Args:
            operations: Optional list of operations to include
            
        Returns:
            New Delta instance
        """
        try:
            # Convert operation dictionaries to DeltaOperation objects
            delta_operations = []
            if operations:
                for op_dict in operations:
                    operation = DeltaOperation(
                        operation_type=DeltaOperationType(op_dict.get("type", "add")),
                        entity_type=DeltaEntityType(op_dict.get("entity_type", "node")),
                        uuid=op_dict.get("uuid", str(uuid.uuid4())),
                        data=op_dict.get("data", {}),
                        metadata=op_dict.get("metadata", {})
                    )
                    delta_operations.append(operation)
            
            return self.incremental_updater.create_delta(delta_operations)
        except Exception as e:
            logger.error(f"Error creating delta: {e}")
            raise
    
    async def apply_delta(self, delta: Delta, validate: bool = True) -> Dict[str, Any]:
        """
        Apply a delta to the knowledge graph.
        
        Args:
            delta: Delta to apply
            validate: Whether to validate the delta before applying
            
        Returns:
            Dictionary containing application results
        """
        try:
            return await self.incremental_updater.apply_delta(delta, validate)
        except Exception as e:
            logger.error(f"Error applying delta: {e}")
            return {"applied": 0, "failed": 1, "errors": [str(e)]}
    
    async def rollback_delta(self, delta_id: str) -> Dict[str, Any]:
        """
        Rollback a previously applied delta.
        
        Args:
            delta_id: ID of the delta to rollback
            
        Returns:
            Dictionary containing rollback results
        """
        try:
            return await self.incremental_updater.rollback_delta(delta_id)
        except Exception as e:
            logger.error(f"Error rolling back delta: {e}")
            return {"rolled_back": 0, "failed": 1, "errors": [str(e)]}
    
    async def update_vector_indices_incremental(self, entity_type: str,
                                              embeddings: List[List[float]],
                                              uuids: List[str]) -> Dict[str, Any]:
        """
        Update vector indices incrementally with new embeddings.
        
        Args:
            entity_type: Type of entity ('node', 'edge', 'community')
            embeddings: List of embedding vectors
            uuids: List of corresponding UUIDs
            
        Returns:
            Dictionary containing update results
        """
        try:
            return await self.incremental_updater.update_vector_indices_incremental(
                entity_type, embeddings, uuids
            )
        except Exception as e:
            logger.error(f"Error updating vector indices incrementally: {e}")
            return {"updated": 0, "errors": [str(e)]}
    
    async def update_text_indices_incremental(self, entity_type: str,
                                            texts: List[str],
                                            uuids: List[str]) -> Dict[str, Any]:
        """
        Update text indices incrementally with new text content.
        
        Args:
            entity_type: Type of entity ('node', 'edge', 'community')
            texts: List of text content
            uuids: List of corresponding UUIDs
            
        Returns:
            Dictionary containing update results
        """
        try:
            return await self.incremental_updater.update_text_indices_incremental(
                entity_type, texts, uuids
            )
        except Exception as e:
            logger.error(f"Error updating text indices incrementally: {e}")
            return {"updated": 0, "errors": [str(e)]}
    
    async def update_graph_indices_incremental(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update graph structure indices incrementally.
        
        Args:
            operations: List of operations that affect graph structure
            
        Returns:
            Dictionary containing update results
        """
        try:
            # Convert operation dictionaries to DeltaOperation objects
            delta_operations = []
            for op_dict in operations:
                operation = DeltaOperation(
                    operation_type=DeltaOperationType(op_dict.get("type", "add")),
                    entity_type=DeltaEntityType(op_dict.get("entity_type", "edge")),
                    uuid=op_dict.get("uuid", str(uuid.uuid4())),
                    data=op_dict.get("data", {}),
                    metadata=op_dict.get("metadata", {})
                )
                delta_operations.append(operation)
            
            return await self.incremental_updater.update_graph_indices_incremental(delta_operations)
        except Exception as e:
            logger.error(f"Error updating graph indices incrementally: {e}")
            return {"updated": 0, "errors": [str(e)]}
    
    async def update_temporal_indices_incremental(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update temporal indices incrementally.
        
        Args:
            operations: List of operations with temporal data
            
        Returns:
            Dictionary containing update results
        """
        try:
            # Convert operation dictionaries to DeltaOperation objects
            delta_operations = []
            for op_dict in operations:
                operation = DeltaOperation(
                    operation_type=DeltaOperationType(op_dict.get("type", "add")),
                    entity_type=DeltaEntityType(op_dict.get("entity_type", "node")),
                    uuid=op_dict.get("uuid", str(uuid.uuid4())),
                    data=op_dict.get("data", {}),
                    metadata=op_dict.get("metadata", {})
                )
                delta_operations.append(operation)
            
            return await self.incremental_updater.update_temporal_indices_incremental(delta_operations)
        except Exception as e:
            logger.error(f"Error updating temporal indices incrementally: {e}")
            return {"updated": 0, "errors": [str(e)]}
    
    async def rebuild_indices_if_needed(self, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Rebuild indices if performance degradation is detected.
        
        Args:
            threshold: Performance threshold (0.0-1.0)
            
        Returns:
            Dictionary containing rebuild results
        """
        try:
            return await self.incremental_updater.rebuild_indices_if_needed(threshold)
        except Exception as e:
            logger.error(f"Error rebuilding indices: {e}")
            return {"rebuild_needed": False, "rebuilt": [], "errors": [str(e)]}
    
    async def batch_incremental_update(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply multiple incremental updates in a single batch operation.
        
        Args:
            updates: List of update dictionaries
            
        Returns:
            Dictionary containing batch results
        """
        try:
            return await self.incremental_updater.batch_incremental_update(updates)
        except Exception as e:
            logger.error(f"Error in batch incremental update: {e}")
            return {
                "total": len(updates),
                "successful": 0,
                "failed": len(updates),
                "details": [{"type": "batch", "success": False, "error": str(e)}]
            }
    
    async def process_large_delta(self, delta: Delta, chunk_size: int = 100) -> Dict[str, Any]:
        """
        Process a large delta by breaking it into smaller chunks.
        
        Args:
            delta: Delta to process
            chunk_size: Size of each chunk
            
        Returns:
            Dictionary containing processing results
        """
        try:
            return await self.incremental_updater.process_large_delta(delta, chunk_size)
        except Exception as e:
            logger.error(f"Error processing large delta: {e}")
            return {"total_chunks": 0, "successful_chunks": 0, "failed_chunks": 0, "errors": [str(e)]}
    
    async def parallel_delta_application(self, deltas: List[Delta], max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Apply multiple deltas in parallel.
        
        Args:
            deltas: List of deltas to apply
            max_concurrent: Maximum number of concurrent applications
            
        Returns:
            Dictionary containing parallel application results
        """
        try:
            return await self.incremental_updater.parallel_delta_application(deltas, max_concurrent)
        except Exception as e:
            logger.error(f"Error in parallel delta application: {e}")
            return {"total": len(deltas), "successful": 0, "failed": len(deltas), "errors": [str(e)]}
    
    async def monitor_delta_progress(self, delta_id: str, poll_interval: float = 1.0) -> Dict[str, Any]:
        """
        Monitor the progress of a delta application.
        
        Args:
            delta_id: ID of the delta to monitor
            poll_interval: Interval in seconds to check progress
            
        Returns:
            Dictionary containing progress information
        """
        try:
            return await self.incremental_updater.monitor_delta_progress(delta_id, poll_interval)
        except Exception as e:
            logger.error(f"Error monitoring delta progress: {e}")
            return {"delta_id": delta_id, "status": "error", "error": str(e)}
    
    def get_update_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about incremental updates.
        
        Returns:
            Dictionary containing update statistics
        """
        return self.incremental_updater.get_update_statistics()
    
    def clear_pending_deltas(self):
        """Clear all pending deltas"""
        self.incremental_updater.clear_pending_deltas()
    
    # Convenience methods for common operations
    async def upsert_entity(self, node: EntityNode, group_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Upsert an entity (add if doesn't exist, update if it does).
        
        Args:
            node: EntityNode to upsert
            group_ids: Optional list of group IDs to filter by
            
        Returns:
            Dictionary containing update results
        """
        try:
            # Check if entity exists
            existing = self.nodes_df[self.nodes_df['uuid'] == node.uuid]
            
            if existing.empty:
                # Add new entity
                return await self.add_entity_incremental(node, group_ids)
            else:
                # Update existing entity
                return await self.incremental_updater.update_entities_incremental([node], group_ids)
        except Exception as e:
            logger.error(f"Error upserting entity: {e}")
            return {"updated": 0, "added": 0, "errors": [str(e)]}
    
    async def upsert_edge(self, edge: EntityEdge, group_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Upsert an edge (add if doesn't exist, update if it does).
        
        Args:
            edge: EntityEdge to upsert
            group_ids: Optional list of group IDs to filter by
            
        Returns:
            Dictionary containing update results
        """
        try:
            # Check if edge exists
            existing = self.edges_df[self.edges_df['uuid'] == edge.uuid]
            
            if existing.empty:
                # Add new edge
                return await self.add_edge_incremental(edge, group_ids)
            else:
                # Update existing edge
                return await self.incremental_updater.update_edges_incremental([edge], group_ids)
        except Exception as e:
            logger.error(f"Error upserting edge: {e}")
            return {"updated": 0, "added": 0, "errors": [str(e)]}
    
    async def bulk_upsert_entities(self, nodes: List[EntityNode], group_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Bulk upsert entities (add if doesn't exist, update if it does).
        
        Args:
            nodes: List of EntityNode objects to upsert
            group_ids: Optional list of group IDs to filter by
            
        Returns:
            Dictionary containing update results
        """
        try:
            # Separate into adds and updates
            adds = []
            updates = []
            
            for node in nodes:
                existing = self.nodes_df[self.nodes_df['uuid'] == node.uuid]
                if existing.empty:
                    adds.append(node)
                else:
                    updates.append(node)
            
            # Process adds and updates
            results = {"added": 0, "updated": 0, "errors": []}
            
            if adds:
                add_result = await self.incremental_updater.add_entities_incremental(adds, group_ids)
                results["added"] = add_result["added"]
                results["errors"].extend(add_result["errors"])
            
            if updates:
                update_result = await self.incremental_updater.update_entities_incremental(updates, group_ids)
                results["updated"] = update_result["updated"]
                results["errors"].extend(update_result["errors"])
            
            return results
        except Exception as e:
            logger.error(f"Error bulk upserting entities: {e}")
            return {"added": 0, "updated": 0, "errors": [str(e)]}
    
    async def bulk_upsert_edges(self, edges: List[EntityEdge], group_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Bulk upsert edges (add if doesn't exist, update if it does).
        
        Args:
            edges: List of EntityEdge objects to upsert
            group_ids: Optional list of group IDs to filter by
            
        Returns:
            Dictionary containing update results
        """
        try:
            # Separate into adds and updates
            adds = []
            updates = []
            
            for edge in edges:
                existing = self.edges_df[self.edges_df['uuid'] == edge.uuid]
                if existing.empty:
                    adds.append(edge)
                else:
                    updates.append(edge)
            
            # Process adds and updates
            results = {"added": 0, "updated": 0, "errors": []}
            
            if adds:
                add_result = await self.incremental_updater.add_edges_incremental(adds, group_ids)
                results["added"] = add_result["added"]
                results["errors"].extend(add_result["errors"])
            
            if updates:
                update_result = await self.incremental_updater.update_edges_incremental(updates, group_ids)
                results["updated"] = update_result["updated"]
                results["errors"].extend(update_result["errors"])
            
            return results
        except Exception as e:
            logger.error(f"Error bulk upserting edges: {e}")
            return {"added": 0, "updated": 0, "errors": [str(e)]}

    # Concurrency Management Methods
    
    async def create_branch(self, branch_name: str, parent_branch: Optional[str] = None, description: Optional[str] = None) -> BranchInfo:
        """
        Create a new dataset branch for experimentation.
        
        Args:
            branch_name: Name of the new branch
            parent_branch: Parent branch (defaults to current branch)
            description: Description of the branch
            
        Returns:
            BranchInfo object for the created branch
        """
        return await self.concurrency_manager.create_branch(branch_name, parent_branch, description)
    
    async def merge_branch(self, source_branch: str, target_branch: str, strategy: str = "auto") -> MergeResult:
        """
        Merge dataset branches with conflict resolution.
        
        Args:
            source_branch: Source branch to merge from
            target_branch: Target branch to merge into
            strategy: Merge strategy for conflict resolution
            
        Returns:
            MergeResult containing merge information
        """
        from graphiti_hf.processing.concurrency_manager import MergeStrategy
        strategy_enum = MergeStrategy(strategy)
        return await self.concurrency_manager.merge_branch(source_branch, target_branch, strategy_enum)
    
    async def switch_branch(self, branch_name: str) -> BranchInfo:
        """
        Switch between branches.
        
        Args:
            branch_name: Name of the branch to switch to
            
        Returns:
            BranchInfo object for the switched branch
        """
        return await self.concurrency_manager.switch_branch(branch_name)
    
    async def list_branches(self, include_deleted: bool = False) -> List[BranchInfo]:
        """
        List available branches.
        
        Args:
            include_deleted: Whether to include deleted branches
            
        Returns:
            List of BranchInfo objects
        """
        return await self.concurrency_manager.list_branches(include_deleted)
    
    async def delete_branch(self, branch_name: str, force: bool = False) -> bool:
        """
        Delete a branch.
        
        Args:
            branch_name: Name of the branch to delete
            force: Whether to force delete even if it has unmerged changes
            
        Returns:
            True if deletion was successful
        """
        return await self.concurrency_manager.delete_branch(branch_name, force)
    
    async def get_version(self, entity_uuid: str, branch: Optional[str] = None) -> Optional[VersionInfo]:
        """
        Get current dataset version for an entity.
        
        Args:
            entity_uuid: UUID of the entity
            branch: Branch name (defaults to current branch)
            
        Returns:
            VersionInfo object or None if not found
        """
        return self.concurrency_manager.get_version(entity_uuid, branch)
    
    async def check_version_conflict(self, entity_uuid: str, expected_version: int, branch: Optional[str] = None) -> bool:
        """
        Check if there's a version conflict for an entity.
        
        Args:
            entity_uuid: UUID of the entity
            expected_version: Expected version number
            branch: Branch name (defaults to current branch)
            
        Returns:
            True if there's a conflict, False otherwise
        """
        return await self.concurrency_manager.check_version_conflict(entity_uuid, expected_version, branch)
    
    async def apply_with_lock(self, entity_uuid: str, entity_data: Dict[str, Any], branch: Optional[str] = None) -> VersionInfo:
        """
        Apply changes with optimistic locking.
        
        Args:
            entity_uuid: UUID of the entity
            entity_data: Entity data to apply
            branch: Branch name (defaults to current branch)
            
        Returns:
            VersionInfo object for the applied version
        """
        return await self.concurrency_manager.apply_with_lock(entity_uuid, entity_data, branch)
    
    async def begin_transaction(self, user_id: Optional[str] = None, branch: Optional[str] = None) -> str:
        """
        Start a new transaction.
        
        Args:
            user_id: User initiating the transaction
            branch: Branch name (defaults to current branch)
            
        Returns:
            Transaction ID
        """
        return await self.concurrency_manager.begin_transaction(user_id, branch)
    
    async def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit a transaction.
        
        Args:
            transaction_id: Transaction ID to commit
            
        Returns:
            True if commit was successful
        """
        return await self.concurrency_manager.commit_transaction(transaction_id)
    
    async def rollback_transaction(self, transaction_id: str) -> bool:
        """
        Rollback a transaction.
        
        Args:
            transaction_id: Transaction ID to rollback
            
        Returns:
            True if rollback was successful
        """
        return await self.concurrency_manager.rollback_transaction(transaction_id)
    
    async def get_transaction_status(self, transaction_id: str) -> Optional[Transaction]:
        """
        Check transaction progress.
        
        Args:
            transaction_id: Transaction ID to check
            
        Returns:
            Transaction object or None if not found
        """
        return await self.concurrency_manager.get_transaction_status(transaction_id)
    
    async def list_transactions(self, branch: Optional[str] = None, status: Optional[str] = None) -> List[Transaction]:
        """
        View active transactions.
        
        Args:
            branch: Filter by branch
            status: Filter by status
            
        Returns:
            List of Transaction objects
        """
        from graphiti_hf.processing.concurrency_manager import TransactionStatus
        status_enum = TransactionStatus(status) if status else None
        return await self.concurrency_manager.list_transactions(branch, status_enum)
    
    async def push_with_versioning(self, commit_message: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """
        Push changes to HuggingFace Hub with versioning information.
        
        Args:
            commit_message: Commit message for the push
            branch: Branch name (defaults to current branch)
            
        Returns:
            Dictionary with push information
        """
        branch = branch or self.concurrency_manager.current_branch
        
        # Get concurrency statistics
        stats = await self.get_concurrency_stats()
        
        # Push to hub
        self._push_to_hub(commit_message)
        
        return {
            "success": True,
            "branch": branch,
            "commit_message": commit_message,
            "concurrency_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_concurrency_stats(self) -> Dict[str, Any]:
        """
        Get statistics about concurrency management.
        
        Returns:
            Dictionary containing concurrency statistics
        """
        return await self.concurrency_manager.get_concurrency_stats()
    
    async def auto_merge_branches(self, source_branch: str, target_branch: str) -> MergeResult:
        """
        Auto-merge branches with automatic conflict resolution.
        
        Args:
            source_branch: Source branch to merge from
            target_branch: Target branch to merge into
            
        Returns:
            MergeResult containing merge information
        """
        return await self.merge_branch(source_branch, target_branch, "auto")
    
    async def manual_merge_branches(self, source_branch: str, target_branch: str) -> MergeResult:
        """
        Manual merge branches requiring user intervention.
        
        Args:
            source_branch: Source branch to merge from
            target_branch: Target branch to merge into
            
        Returns:
            MergeResult containing merge information
        """
        return await self.merge_branch(source_branch, target_branch, "manual")
    
    # Temporal Management Methods
    
    async def set_validity_period(
        self,
        entity_uuid: str,
        valid_from: datetime,
        valid_to: Optional[datetime] = None,
        entity_type: str = "edge"
    ) -> Dict[str, Any]:
        """
        Set temporal validity range for an entity
        
        Args:
            entity_uuid: UUID of the entity
            valid_from: Start of validity period
            valid_to: End of validity period (optional, means ongoing)
            entity_type: Type of entity ('node' or 'edge')
            
        Returns:
            Dictionary with update results
        """
        return await self.temporal_manager.set_validity_period(entity_uuid, valid_from, valid_to, entity_type)
    
    async def invalidate_edges(
        self,
        edge_uuids: List[str],
        invalidation_reason: str = "contradiction",
        invalidation_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Handle temporal edge invalidation for contradictions
        
        Args:
            edge_uuids: List of edge UUIDs to invalidate
            invalidation_reason: Reason for invalidation
            invalidation_time: When to invalidate (defaults to now)
            
        Returns:
            Dictionary with invalidation results
        """
        return await self.temporal_manager.invalidate_edges(edge_uuids, invalidation_reason, invalidation_time)
    
    async def get_valid_at(
        self,
        entity_uuid: str,
        query_time: datetime,
        entity_type: str = "edge"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve data valid at a specific time
        
        Args:
            entity_uuid: UUID of the entity
            query_time: Time to query for
            entity_type: Type of entity ('node' or 'edge')
            
        Returns:
            Entity data if valid at query time, None otherwise
        """
        return await self.temporal_manager.get_valid_at(entity_uuid, query_time, entity_type)
    
    async def get_historical_state(
        self,
        query_time: datetime,
        group_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Reconstruct historical graph state at a specific time
        
        Args:
            query_time: Time to reconstruct state for
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of entities to return
            
        Returns:
            Dictionary containing historical graph state
        """
        return await self.temporal_manager.get_historical_state(query_time, group_ids, limit)
    
    async def temporal_query(
        self,
        filter: TemporalQueryFilter,
        entity_type: str = "edge"
    ) -> List[Dict[str, Any]]:
        """
        Perform time-based queries on temporal data
        
        Args:
            filter: TemporalQueryFilter with query parameters
            entity_type: Type of entity to query ('node' or 'edge')
            
        Returns:
            List of matching temporal records
        """
        return await self.temporal_manager.temporal_query(filter, entity_type)
    
    async def temporal_search(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
        entity_type: str = "edge"
    ) -> List[Dict[str, Any]]:
        """
        Perform time-based search on temporal data
        
        Args:
            query: Search query string
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            limit: Maximum number of results
            entity_type: Type of entity to search
            
        Returns:
            List of matching temporal records
        """
        try:
            # Create temporal filter
            filter = TemporalQueryFilter(
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            # Perform temporal query
            results = await self.temporal_manager.temporal_query(filter, entity_type)
            
            # Apply text search if query provided
            if query:
                filtered_results = []
                query_lower = query.lower()
                
                for result in results:
                    # Check if query matches fact or name
                    if entity_type == "edge" and 'fact' in result:
                        if query_lower in result['fact'].lower():
                            filtered_results.append(result)
                    elif entity_type == "node" and 'name' in result:
                        if query_lower in result['name'].lower():
                            filtered_results.append(result)
                
                return filtered_results[:limit]
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in temporal search: {e}")
            return []
    
    async def get_historical_data(
        self,
        time_range: Tuple[datetime, datetime],
        entity_type: str = "edge",
        group_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get historical data within a time range
        
        Args:
            time_range: Tuple of (start_time, end_time)
            entity_type: Type of entity to retrieve
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results
            
        Returns:
            Dictionary with historical data
        """
        try:
            start_time, end_time = time_range
            
            # Get data within time range
            records = await self.temporal_manager.temporal_range_query(
                start_time, end_time, entity_type, group_ids, limit
            )
            
            return {
                "time_range": (start_time.isoformat(), end_time.isoformat()),
                "entity_type": entity_type,
                "total_records": len(records),
                "records": records,
                "retrieved_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return {
                "time_range": (time_range[0].isoformat(), time_range[1].isoformat()),
                "entity_type": entity_type,
                "error": str(e),
                "total_records": 0,
                "records": []
            }
    
    async def invalidate_temporal_edges(
        self,
        conflicting_edges: List[str],
        resolution_strategy: str = "invalidate"
    ) -> Dict[str, Any]:
        """
        Invalidate temporal edges with contradictions
        
        Args:
            conflicting_edges: List of edge UUIDs that conflict
            resolution_strategy: Strategy for resolving conflicts
            
        Returns:
            Dictionary with resolution results
        """
        try:
            strategy_enum = TemporalResolutionStrategy(resolution_strategy)
            return await self.temporal_manager.temporal_edge_invalidation(
                conflicting_edges, strategy_enum
            )
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid resolution strategy: {resolution_strategy}"
            }
    
    async def get_temporal_stats(self) -> TemporalStats:
        """
        Get comprehensive temporal statistics
        
        Returns:
            TemporalStats object with comprehensive statistics
        """
        return await self.temporal_manager.temporal_statistics()
    
    async def temporal_consistency_check(
        self,
        check_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate temporal integrity of the knowledge graph
        
        Args:
            check_types: List of check types to perform
            
        Returns:
            Dictionary with consistency check results
        """
        return await self.temporal_manager.temporal_consistency_check(check_types)
    
    async def temporal_deduplication(
        self,
        similarity_threshold: float = 0.95,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Remove duplicate temporal records
        
        Args:
            similarity_threshold: Threshold for considering records duplicates
            time_window_hours: Time window in hours to check for duplicates
            
        Returns:
            Dictionary with deduplication results
        """
        return await self.temporal_manager.temporal_deduplication(similarity_threshold, time_window_hours)
    
    async def resolve_temporal_conflicts(
        self,
        conflicts: List[Dict[str, Any]],
        strategy: str = "manual"
    ) -> Dict[str, Any]:
        """
        Resolve temporal conflicts in the knowledge graph
        
        Args:
            conflicts: List of conflict dictionaries
            strategy: Resolution strategy to apply
            
        Returns:
            Dictionary with resolution results
        """
        try:
            # Convert conflict dictionaries to TemporalConflict objects
            temporal_conflicts = []
            for conflict_data in conflicts:
                conflict = TemporalConflict(
                    conflict_id=conflict_data.get('conflict_id', str(uuid.uuid4())),
                    conflict_type=TemporalConflictType(conflict_data.get('conflict_type', 'overlapping_validity')),
                    affected_entities=conflict_data.get('affected_entities', []),
                    conflicting_records=[],  # Simplified for now
                    detected_at=datetime.now(),
                    resolution_strategy=TemporalResolutionStrategy(strategy)
                )
                temporal_conflicts.append(conflict)
            
            strategy_enum = TemporalResolutionStrategy(strategy)
            return await self.temporal_manager.resolve_temporal_conflicts(temporal_conflicts, strategy_enum)
            
        except Exception as e:
            logger.error(f"Error resolving temporal conflicts: {e}")
            return {"error": str(e)}
    
    async def temporal_range_query(
        self,
        start_time: datetime,
        end_time: datetime,
        entity_type: str = "edge",
        group_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query data within a specific time range
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            entity_type: Type of entity to query
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results
            
        Returns:
            List of records within the time range
        """
        return await self.temporal_manager.temporal_range_query(
            start_time, end_time, entity_type, group_ids, limit
        )
    
    async def temporal_point_query(
        self,
        query_time: datetime,
        entity_type: str = "edge",
        group_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query data valid at a specific time point
        
        Args:
            query_time: Time point to query
            entity_type: Type of entity to query
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results
            
        Returns:
            List of records valid at the time point
        """
        return await self.temporal_manager.temporal_point_query(
            query_time, entity_type, group_ids, limit
        )
    
    async def temporal_aggregation(
        self,
        aggregation_type: str,
        time_range: Tuple[datetime, datetime],
        group_by: Optional[str] = None,
        entity_type: str = "edge"
    ) -> Dict[str, Any]:
        """
        Aggregate data over time
        
        Args:
            aggregation_type: Type of aggregation ('count', 'sum', 'avg', 'max', 'min')
            time_range: Time range to aggregate over
            group_by: Field to group by (optional)
            entity_type: Type of entity to aggregate
            
        Returns:
            Aggregation results
        """
        return await self.temporal_manager.temporal_aggregation(
            aggregation_type, time_range, group_by, entity_type
        )
    
    async def detect_temporal_anomalies(
        self,
        anomaly_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect temporal anomalies in the knowledge graph
        
        Args:
            anomaly_types: List of anomaly types to detect
            
        Returns:
            List of detected anomalies
        """
        return await self.temporal_manager.detect_temporal_anomalies(anomaly_types)
    
    async def auto_temporal_cleanup(
        self,
        cleanup_strategy: str = "soft",
        older_than_days: int = 30
    ) -> Dict[str, Any]:
        """
        Clean up invalid temporal data
        
        Args:
            cleanup_strategy: Strategy for cleanup ('soft', 'hard', 'archive')
            older_than_days: Remove records older than this many days
            
        Returns:
            Dictionary with cleanup results
        """
        return await self.temporal_manager.auto_temporal_cleanup(cleanup_strategy, older_than_days)
    
    async def temporal_versioning(
        self,
        entity_uuid: str,
        version_action: str = "create"
    ) -> Dict[str, Any]:
        """
        Maintain temporal versions of entities
        
        Args:
            entity_uuid: UUID of the entity
            version_action: Action to perform ('create', 'list', 'restore')
            
        Returns:
            Dictionary with versioning results
        """
        return await self.temporal_manager.temporal_versioning(entity_uuid, version_action)
    
    async def build_temporal_indices(self) -> Dict[str, Any]:
        """
        Build efficient time-based query indices
        
        Returns:
            Dictionary with index building results
        """
        return await self.temporal_manager.build_temporal_indices()
    
    async def merge_temporal_records(
        self,
        entity_uuid: str,
        record_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Combine temporal data from multiple records
        
        Args:
            entity_uuid: UUID of the entity
            record_ids: List of record IDs to merge
            
        Returns:
            Dictionary with merge results
        """
        return await self.temporal_manager.merge_temporal_records(entity_uuid, record_ids)
    
    # Community Detection Methods
    
    async def detect_graph_communities(
        self,
        group_ids: Optional[List[str]] = None,
        algorithm: str = "louvain",
        resolution: float = 1.0,
        k_clusters: Optional[int] = None,
        similarity_threshold: float = 0.5,
        min_cluster_size: int = 2,
        max_iterations: int = 100,
        random_state: Optional[int] = 42,
        commit_message: Optional[str] = None
    ) -> Tuple[List[CommunityNode], List[CommunityEdge]]:
        """
        Detect communities in the knowledge graph using various algorithms
        
        Args:
            group_ids: List of group IDs to filter by
            algorithm: Community detection algorithm ('louvain', 'label_propagation', 'connected_components', 'clique_percolation', 'kmeans', 'hierarchical')
            resolution: Resolution parameter for Louvain algorithm
            k_clusters: Number of clusters for K-means and hierarchical clustering
            similarity_threshold: Similarity threshold for clique percolation
            min_cluster_size: Minimum cluster size
            max_iterations: Maximum iterations for iterative algorithms
            random_state: Random state for reproducibility
            commit_message: Optional commit message for storage
            
        Returns:
            Tuple of (community_nodes, community_edges)
        """
        config = CommunityDetectionConfig(
            algorithm=algorithm,
            resolution=resolution,
            k_clusters=k_clusters,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            max_iterations=max_iterations,
            random_state=random_state
        )
        
        communities = await self.community_detector.detect_communities(group_ids, config)
        
        # Store communities if commit message provided
        if commit_message:
            await self.community_detector.store_communities(communities[0], communities[1], commit_message)
        
        return communities
    
    async def get_community_info(
        self,
        community_uuid: Optional[str] = None,
        group_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get information about communities
        
        Args:
            community_uuid: Specific community UUID to get info for (optional)
            group_ids: List of group IDs to filter by
            
        Returns:
            Dictionary with community information
        """
        try:
            # Load communities
            community_nodes, community_edges = await self.community_detector.load_communities(group_ids)
            
            if community_uuid:
                # Get specific community info
                for node in community_nodes:
                    if node.uuid == community_uuid:
                        connected_entities = await self.community_detector._get_connected_entities(community_uuid)
                        return {
                            "community_uuid": node.uuid,
                            "community_name": node.name,
                            "group_id": node.group_id,
                            "created_at": node.created_at.isoformat(),
                            "summary": node.summary,
                            "member_count": len(connected_entities),
                            "members": connected_entities
                        }
                return {"error": f"Community {community_uuid} not found"}
            
            else:
                # Get general community statistics
                stats = await self.community_detector.get_community_stats(community_nodes, community_edges)
                return {
                    "total_communities": len(community_nodes),
                    "statistics": stats,
                    "communities": [
                        {
                            "uuid": node.uuid,
                            "name": node.name,
                            "group_id": node.group_id,
                            "created_at": node.created_at.isoformat(),
                            "summary": node.summary
                        }
                        for node in community_nodes
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error getting community info: {e}")
            return {"error": str(e)}
    
    async def analyze_community_structure(
        self,
        group_ids: Optional[List[str]] = None,
        algorithm: str = "louvain",
        include_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze community structure in the knowledge graph
        
        Args:
            group_ids: List of group IDs to filter by
            algorithm: Community detection algorithm to use
            include_analysis: Whether to include detailed analysis
            
        Returns:
            Dictionary with community analysis results
        """
        try:
            # Detect communities
            community_nodes, community_edges = await self.detect_graph_communities(
                group_ids=group_ids,
                algorithm=algorithm
            )
            
            if not community_nodes:
                return {"message": "No communities found", "total_communities": 0}
            
            # Build graph for analysis
            G = await self.community_detector._build_graph_from_dataset(group_ids)
            
            # Get basic statistics
            stats = await self.community_detector.get_community_stats(community_nodes, community_edges)
            
            result = {
                "algorithm": algorithm,
                "total_communities": len(community_nodes),
                "statistics": stats,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add detailed analysis if requested
            if include_analysis:
                # Analyze community structure
                communities = []
                for i, community_node in enumerate(community_nodes):
                    connected_entities = await self.community_detector._get_connected_entities(community_node.uuid)
                    communities.append(connected_entities)
                
                structure_analysis = await self.community_detector.analyze_community_structure(communities, G)
                result["structure_analysis"] = structure_analysis.dict()
                
                # Find core members
                core_members = await self.community_detector.find_core_members(community_nodes, G)
                result["core_members"] = core_members
                
                # Find bridges
                bridges = await self.community_detector.find_bridges(community_nodes, G)
                result["bridge_nodes"] = bridges
                
                # Calculate community similarities
                similarities = await self.community_detector.community_similarity(community_nodes)
                result["community_similarities"] = [sim.dict() for sim in similarities]
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing community structure: {e}")
            return {"error": str(e)}
    
    async def export_communities(
        self,
        format: str = "json",
        include_embeddings: bool = False,
        group_ids: Optional[List[str]] = None,
        commit_message: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Export community data
        
        Args:
            format: Export format ('json', 'csv', 'parquet')
            include_embeddings: Whether to include embeddings in export
            group_ids: List of group IDs to filter by
            commit_message: Optional commit message for storage
            
        Returns:
            Exported data as string or dictionary
        """
        try:
            # Load communities
            community_nodes, community_edges = await self.community_detector.load_communities(group_ids)
            
            # Export data
            export_data = await self.community_detector.export_communities(
                format=format,
                include_embeddings=include_embeddings
            )
            
            # Store export if commit message provided
            if commit_message:
                await self.community_detector.store_communities(community_nodes, community_edges, commit_message)
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting communities: {e}")
            return {"error": str(e)}
    
    async def batch_community_detection(
        self,
        group_id_batches: List[List[str]],
        algorithm: str = "louvain",
        resolution: float = 1.0,
        k_clusters: Optional[int] = None,
        similarity_threshold: float = 0.5,
        min_cluster_size: int = 2,
        max_iterations: int = 100,
        random_state: Optional[int] = 42,
        commit_message: Optional[str] = None
    ) -> List[Tuple[List[CommunityNode], List[CommunityEdge]]]:
        """
        Process large graphs in batches for community detection
        
        Args:
            group_id_batches: List of group ID batches
            algorithm: Community detection algorithm
            resolution: Resolution parameter for Louvain algorithm
            k_clusters: Number of clusters for K-means and hierarchical clustering
            similarity_threshold: Similarity threshold for clique percolation
            min_cluster_size: Minimum cluster size
            max_iterations: Maximum iterations for iterative algorithms
            random_state: Random state for reproducibility
            commit_message: Optional commit message for storage
            
        Returns:
            List of (community_nodes, community_edges) for each batch
        """
        config = CommunityDetectionConfig(
            algorithm=algorithm,
            resolution=resolution,
            k_clusters=k_clusters,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            max_iterations=max_iterations,
            random_state=random_state
        )
        
        results = await self.community_detector.batch_community_detection(group_id_batches, config)
        
        # Store all communities if commit message provided
        if commit_message:
            all_nodes = []
            all_edges = []
            for nodes, edges in results:
                all_nodes.extend(nodes)
                all_edges.extend(edges)
            await self.community_detector.store_communities(all_nodes, all_edges, commit_message)
        
        return results
    
    async def incremental_community_update(
        self,
        new_nodes: List[EntityNode],
        new_edges: List[EntityEdge],
        existing_communities: Optional[List[CommunityNode]] = None,
        commit_message: Optional[str] = None
    ) -> Tuple[List[CommunityNode], List[CommunityEdge]]:
        """
        Update communities incrementally with new nodes and edges
        
        Args:
            new_nodes: List of new entity nodes
            new_edges: List of new entity edges
            existing_communities: List of existing communities
            commit_message: Optional commit message for storage
            
        Returns:
            Updated (community_nodes, community_edges)
        """
        updated_communities = await self.community_detector.incremental_community_update(
            new_nodes, new_edges, existing_communities
        )
        
        # Store updated communities if commit message provided
        if commit_message:
            await self.community_detector.store_communities(
                updated_communities[0], updated_communities[1], commit_message
            )
        
        return updated_communities
    
    async def parallel_community_detection(
        self,
        group_ids: List[str],
        algorithm: str = "louvain",
        resolution: float = 1.0,
        k_clusters: Optional[int] = None,
        similarity_threshold: float = 0.5,
        min_cluster_size: int = 2,
        max_iterations: int = 100,
        random_state: Optional[int] = 42,
        max_workers: int = 4,
        commit_message: Optional[str] = None
    ) -> Tuple[List[CommunityNode], List[CommunityEdge]]:
        """
        Perform parallel community detection for multiple group IDs
        
        Args:
            group_ids: List of group IDs
            algorithm: Community detection algorithm
            resolution: Resolution parameter for Louvain algorithm
            k_clusters: Number of clusters for K-means and hierarchical clustering
            similarity_threshold: Similarity threshold for clique percolation
            min_cluster_size: Minimum cluster size
            max_iterations: Maximum iterations for iterative algorithms
            random_state: Random state for reproducibility
            max_workers: Maximum number of parallel workers
            commit_message: Optional commit message for storage
            
        Returns:
            Tuple of (community_nodes, community_edges)
        """
        config = CommunityDetectionConfig(
            algorithm=algorithm,
            resolution=resolution,
            k_clusters=k_clusters,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            max_iterations=max_iterations,
            random_state=random_state
        )
        
        communities = await self.community_detector.parallel_community_detection(
            group_ids, config, max_workers
        )
        
        # Store communities if commit message provided
        if commit_message:
            await self.community_detector.store_communities(communities[0], communities[1], commit_message)
        
        return communities
    
    async def community_versioning(
        self,
        action: str = "create",
        version_id: Optional[str] = None,
        commit_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track community evolution over time
        
        Args:
            action: Action to perform ('create', 'list', 'restore')
            version_id: Version ID for restore action
            commit_message: Optional commit message for storage
            
        Returns:
            Dictionary with versioning results
        """
        result = await self.community_detector.community_versioning(action, version_id)
        
        # Store version if created and commit message provided
        if action == "create" and commit_message and result.get("success"):
            # Get current communities and store them
            community_nodes, community_edges = await self.community_detector.load_communities()
            await self.community_detector.store_communities(community_nodes, community_edges, commit_message)
        
        return result
    
    async def community_caching(
        self,
        cache_key: Optional[str] = None,
        ttl: int = 3600
    ) -> Dict[str, Any]:
        """
        Cache community results for faster retrieval
        
        Args:
            cache_key: Optional cache key
            ttl: Time to live in seconds
            
        Returns:
            Dictionary with caching results
        """
        # Set TTL
        self.community_detector._cache_ttl = ttl
        
        # Get current communities
        community_nodes, community_edges = await self.community_detector.load_communities()
        
        # Cache communities
        result = await self.community_detector.community_caching(
            (community_nodes, community_edges), cache_key
        )
        
        return result
    
    async def get_cached_communities(
        self,
        cache_key: Optional[str] = None
    ) -> Optional[Tuple[List[CommunityNode], List[CommunityEdge]]]:
        """
        Retrieve cached community results
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            Cached communities or None if not found/expired
        """
        return await self.community_detector.get_cached_communities(cache_key)
    
    # Custom Type Management Methods
    
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
        custom_type_manager: Optional[CustomTypeManager] = None,
        **kwargs
    ):
        """
        Initialize the HuggingFaceDriver
        
        Args:
            repo_id: HuggingFace repository ID
            token: HuggingFace access token (optional)
            private: Whether the repository is private
            create_repo: Whether to create the repository if it doesn't exist
            enable_vector_search: Whether to enable vector search
            vector_search_config: Configuration for vector search
            enable_performance_optimization: Whether to enable performance optimization
            performance_optimizer_config: Configuration for performance optimizer
            custom_type_manager: Custom type manager for custom entity and edge types
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        self.repo_id = repo_id
        self.token = token or HfFolder.get_token()
        self.private = private
        self.create_repo = create_repo
        
        # Initialize custom type manager
        self.custom_type_manager = custom_type_manager or get_type_manager()
        
        # Initialize datasets
        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        self.episodes_df = pd.DataFrame()
        self.communities_df = pd.DataFrame()
        
        # Initialize vector search
        self.vector_search_engine = None
        if enable_vector_search:
            from graphiti_hf.search.vector_search import VectorSearchEngine
            self.vector_search_engine = VectorSearchEngine(vector_search_config or SearchConfig())
        
        # Initialize performance optimizer
        self.performance_optimizer = None
        if enable_performance_optimization:
            from graphiti_hf.search.performance_optimizer import PerformanceOptimizer
            self.performance_optimizer = PerformanceOptimizer(performance_optimizer_config or {})
        
        # Initialize traversal engine
        self.traversal_engine = None
        if self.vector_search_engine:
            from graphiti_hf.search.graph_traversal import GraphTraversalEngine
            self.traversal_engine = GraphTraversalEngine(self)
        
        # Initialize hybrid search engine
        self.hybrid_search_engine = None
        if self.vector_search_engine:
            from graphiti_hf.search.hybrid_search import HybridSearchEngine
            self.hybrid_search_engine = HybridSearchEngine(self)
        
        # Initialize incremental updater
        self.incremental_updater = None
        if self.vector_search_engine:
            from graphiti_hf.processing.incremental_updater import IncrementalUpdater
            self.incremental_updater = IncrementalUpdater(self)
        
        # Initialize concurrency manager
        self.concurrency_manager = None
        if self.vector_search_engine:
            from graphiti_hf.processing.concurrency_manager import ConcurrencyManager
            self.concurrency_manager = ConcurrencyManager(self)
        
        # Initialize temporal manager
        self.temporal_manager = None
        if self.vector_search_engine:
            from graphiti_hf.processing.temporal_manager import TemporalManager
            self.temporal_manager = TemporalManager(self)
        
        # Initialize community detector
        self.community_detector = None
        if self.vector_search_engine:
            from graphiti_hf.analysis.community_detector import CommunityDetector
            self.community_detector = CommunityDetector(self)
        
        # Initialize provider
        self.provider = "huggingface"
        
        # Load existing datasets or create new ones
        self._load_or_create_datasets()
    
    def get_custom_type_manager(self) -> CustomTypeManager:
        """Get the custom type manager"""
        return self.custom_type_manager
    
    def register_custom_entity_type(self, entity_type: str, schema: Dict[str, Any]) -> bool:
        """
        Register a new custom entity type
        
        Args:
            entity_type: Name of the entity type
            schema: Schema definition for the entity type
            
        Returns:
            True if registration was successful
        """
        return self.custom_type_manager.register_entity_type(entity_type, schema)
    
    def register_custom_edge_type(self, edge_type: str, schema: Dict[str, Any]) -> bool:
        """
        Register a new custom edge type
        
        Args:
            edge_type: Name of the edge type
            schema: Schema definition for the edge type
            
        Returns:
            True if registration was successful
        """
        return self.custom_type_manager.register_edge_type(edge_type, schema)
    
    def get_custom_entity_types(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered custom entity types"""
        return self.custom_type_manager.get_entity_types()
    
    def get_custom_edge_types(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered custom edge types"""
        return self.custom_type_manager.get_edge_types()
    
    def get_custom_entity_schema(self, entity_type: str) -> Optional[Dict[str, Any]]:
        """Get schema for a custom entity type"""
        return self.custom_type_manager.get_entity_schema(entity_type)
    
    def get_custom_edge_schema(self, edge_type: str) -> Optional[Dict[str, Any]]:
        """Get schema for a custom edge type"""
        return self.custom_type_manager.get_edge_schema(edge_type)
    
    def validate_custom_entity(self, entity_type: str, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate custom entity data against schema
        
        Args:
            entity_type: Type of entity to validate
            data: Entity data to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        return self.custom_type_manager.validate_entity(entity_type, data)
    
    def validate_custom_edge(self, edge_type: str, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate custom edge data against schema
        
        Args:
            edge_type: Type of edge to validate
            data: Edge data to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        return self.custom_type_manager.validate_edge(edge_type, data)
    
    async def save_custom_entity(self, entity_type: str, data: Dict[str, Any],
                               group_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Save a custom entity to the knowledge graph
        
        Args:
            entity_type: Type of entity to save
            data: Entity data
            group_ids: Optional list of group IDs
            
        Returns:
            Dictionary containing save results
        """
        # Validate entity data
        is_valid, errors = self.validate_custom_entity(entity_type, data)
        if not is_valid:
            return {"success": False, "errors": errors}
        
        # Create custom entity
        try:
            entity = create_custom_entity(entity_type, data)
        except ValueError as e:
            return {"success": False, "errors": [str(e)]}
        
        # Save entity
        try:
            result = await self.save_node(entity)
            return {"success": True, "entity": result}
        except Exception as e:
            return {"success": False, "errors": [str(e)]}
    
    async def save_custom_edge(self, edge_type: str, data: Dict[str, Any],
                             group_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Save a custom edge to the knowledge graph
        
        Args:
            edge_type: Type of edge to save
            data: Edge data
            group_ids: Optional list of group IDs
            
        Returns:
            Dictionary containing save results
        """
        # Validate edge data
        is_valid, errors = self.validate_custom_edge(edge_type, data)
        if not is_valid:
            return {"success": False, "errors": errors}
        
        # Create custom edge
        try:
            edge = create_custom_edge(edge_type, data)
        except ValueError as e:
            return {"success": False, "errors": [str(e)]}
        
        # Save edge
        try:
            result = await self.save_edge(edge)
            return {"success": True, "edge": result}
        except Exception as e:
            return {"success": False, "errors": [str(e)]}
    
    async def get_custom_entities(self, entity_type: str,
                                group_ids: Optional[List[str]] = None,
                                limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all entities of a custom type
        
        Args:
            entity_type: Type of entities to retrieve
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results
            
        Returns:
            List of entity dictionaries
        """
        # Get all nodes
        nodes = await self.get_nodes_by_group_ids(group_ids or [], "Entity", limit)
        
        # Filter by custom type
        custom_entities = []
        for node in nodes:
            if hasattr(node, 'custom_type') and node.custom_type == entity_type:
                custom_entities.append(node.dict())
        
        return custom_entities
    
    async def get_custom_edges(self, edge_type: str,
                             group_ids: Optional[List[str]] = None,
                             limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all edges of a custom type
        
        Args:
            edge_type: Type of edges to retrieve
            group_ids: Optional list of group IDs to filter by
            limit: Maximum number of results
            
        Returns:
            List of edge dictionaries
        """
        # Get all edges
        edges = await self.get_edges_by_group_ids(group_ids or [], "Entity", limit)
        
        # Filter by custom type
        custom_edges = []
        for edge in edges:
            if hasattr(edge, 'custom_type') and edge.custom_type == edge_type:
                custom_edges.append(edge.dict())
        
        return custom_edges
    
    async def search_custom_entities(self, entity_type: str, query: str,
                                   limit: int = 10,
                                   semantic_weight: float = 0.4,
                                   keyword_weight: float = 0.3,
                                   graph_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for custom entities using hybrid search
        
        Args:
            entity_type: Type of entities to search
            query: Search query
            limit: Maximum number of results
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
            graph_weight: Weight for graph-based ranking
            
        Returns:
            List of search results
        """
        # Perform hybrid search
        results = await self.search_hybrid(
            query=query,
            limit=limit,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            graph_weight=graph_weight
        )
        
        # Filter by custom type
        custom_results = []
        for result in results:
            if result.get('entity_type') == 'node' and result.get('custom_type') == entity_type:
                custom_results.append(result)
        
        return custom_results
    
    async def search_custom_edges(self, edge_type: str, query: str,
                                limit: int = 10,
                                semantic_weight: float = 0.4,
                                keyword_weight: float = 0.3,
                                graph_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for custom edges using hybrid search
        
        Args:
            edge_type: Type of edges to search
            query: Search query
            limit: Maximum number of results
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
            graph_weight: Weight for graph-based ranking
            
        Returns:
            List of search results
        """
        # Perform hybrid search
        results = await self.search_hybrid(
            query=query,
            limit=limit,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            graph_weight=graph_weight
        )
        
        # Filter by custom type
        custom_results = []
        for result in results:
            if result.get('entity_type') == 'edge' and result.get('custom_type') == edge_type:
                custom_results.append(result)
        
        return custom_results
    
    def get_custom_type_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about custom types
        
        Returns:
            Dictionary containing custom type statistics
        """
        entity_types = self.get_custom_entity_types()
        edge_types = self.get_custom_edge_types()
        
        return {
            "entity_types": {
                "count": len(entity_types),
                "types": list(entity_types.keys())
            },
            "edge_types": {
                "count": len(edge_types),
                "types": list(edge_types.keys())
            },
            "total_custom_types": len(entity_types) + len(edge_types)
        }
    
    def export_custom_types(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export custom type definitions
        
        Args:
            format: Export format ('json', 'yaml')
            
        Returns:
            Exported custom type definitions
        """
        entity_types = self.get_custom_entity_types()
        edge_types = self.get_custom_edge_types()
        
        data = {
            "entity_types": entity_types,
            "edge_types": edge_types,
            "exported_at": datetime.now().isoformat()
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2)
        elif format.lower() == "yaml":
            import yaml
            return yaml.dump(data, default_flow_style=False)
        else:
            return data
    
    def import_custom_types(self, data: Union[str, Dict[str, Any]],
                          format: str = "json") -> Dict[str, Any]:
        """
        Import custom type definitions
        
        Args:
            data: Custom type definitions to import
            format: Import format ('json', 'yaml')
            
        Returns:
            Dictionary containing import results
        """
        if isinstance(data, str):
            if format.lower() == "json":
                data = json.loads(data)
            elif format.lower() == "yaml":
                import yaml
                data = yaml.safe_load(data)
        
        results = {"entity_types": {}, "edge_types": {}}
        
        # Import entity types
        for entity_type, schema in data.get("entity_types", {}).items():
            success = self.register_custom_entity_type(entity_type, schema)
            results["entity_types"][entity_type] = success
        
        # Import edge types
        for edge_type, schema in data.get("edge_types", {}).items():
            success = self.register_custom_edge_type(edge_type, schema)
            results["edge_types"][edge_type] = success
        
        return results
    
    def clear_custom_types(self) -> Dict[str, Any]:
        """
        Clear all custom type definitions
        
        Returns:
            Dictionary containing clear results
        """
        results = {
            "cleared_entity_types": len(self.custom_type_manager.get_entity_types()),
            "cleared_edge_types": len(self.custom_type_manager.get_edge_types())
        }
        
        self.custom_type_manager.clear_types()
        return results