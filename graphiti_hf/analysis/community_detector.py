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
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import CommunityEdge, EntityEdge
from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import CommunityNode, EntityNode
from graphiti_core.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)


class CommunityDetectionConfig(BaseModel):
    """Configuration for community detection algorithms"""
    algorithm: str = "louvain"  # louvain, label_propagation, connected_components, clique_percolation, kmeans, hierarchical
    resolution: float = 1.0  # Resolution parameter for Louvain
    k_clusters: Optional[int] = None  # Number of clusters for K-means
    similarity_threshold: float = 0.5  # Similarity threshold for clique percolation
    min_cluster_size: int = 2  # Minimum cluster size
    max_iterations: int = 100  # Maximum iterations for iterative algorithms
    random_state: Optional[int] = 42  # Random state for reproducibility


class CommunityStats(BaseModel):
    """Statistics about detected communities"""
    total_communities: int
    average_size: float
    largest_community_size: int
    smallest_community_size: int
    modularity: float
    silhouette_score: Optional[float] = None
    communities: List[Dict[str, Any]]


class CommunityMember(BaseModel):
    """Information about a community member"""
    node_uuid: str
    node_name: str
    centrality: float
    role: str  # core, peripheral, bridge
    similarity_to_center: float


class CommunitySimilarity(BaseModel):
    """Similarity between two communities"""
    community1_uuid: str
    community2_uuid: str
    jaccard_similarity: float
    cosine_similarity: float
    common_members: int
    total_members: int


class CommunityDetector:
    """
    Community detection algorithms for dataset storage in Graphiti-HF
    
    Provides various community detection algorithms and analysis capabilities
    specifically designed to work with HuggingFace datasets.
    """
    
    def __init__(
        self,
        driver: GraphDriver,
        embedder: Optional[EmbedderClient] = None,
        llm_client: Optional[LLMClient] = None,
        embedder_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the CommunityDetector
        
        Args:
            driver: Graph driver instance
            embedder: Embedder client for generating embeddings
            llm_client: LLM client for community summarization
            embedder_model: Sentence transformer model for embeddings
        """
        self.driver = driver
        self.embedder = embedder
        self.llm_client = llm_client
        self.embedder_model = embedder_model
        self.sentence_transformer = SentenceTransformer(embedder_model) if embedder_model else None
        
        # Cache for community results
        self._community_cache: Dict[str, Any] = {}
        self._cache_ttl = 3600  # 1 hour cache TTL
        
    async def detect_communities(
        self,
        group_ids: Optional[List[str]] = None,
        config: Optional[CommunityDetectionConfig] = None
    ) -> Tuple[List[CommunityNode], List[CommunityEdge]]:
        """
        Detect communities using various algorithms
        
        Args:
            group_ids: List of group IDs to filter by
            config: Community detection configuration
            
        Returns:
            Tuple of (community_nodes, community_edges)
        """
        config = config or CommunityDetectionConfig()
        
        # Build graph from dataset
        G = await self._build_graph_from_dataset(group_ids)
        
        if len(G.nodes()) == 0:
            logger.warning("No nodes found for community detection")
            return [], []
        
        # Apply selected algorithm
        if config.algorithm == "louvain":
            communities = await self.louvain_detection(G, config)
        elif config.algorithm == "label_propagation":
            communities = await self.label_propagation(G, config)
        elif config.algorithm == "connected_components":
            communities = await self.connected_components(G, config)
        elif config.algorithm == "clique_percolation":
            communities = await self.clique_percolation(G, config)
        elif config.algorithm == "kmeans":
            communities = await self.kmeans_detection(G, config)
        elif config.algorithm == "hierarchical":
            communities = await self.hierarchical_detection(G, config)
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")
        
        # Create community nodes and edges
        community_nodes, community_edges = await self._create_community_entities(communities, group_ids)
        
        return community_nodes, community_edges
    
    async def louvain_detection(
        self,
        G: nx.Graph,
        config: Optional[CommunityDetectionConfig] = None
    ) -> List[List[str]]:
        """
        Detect communities using Louvain algorithm
        
        Args:
            G: NetworkX graph
            config: Detection configuration
            
        Returns:
            List of communities, each containing node UUIDs
        """
        try:
            import community as community_louvain
        except ImportError:
            logger.warning("python-louvain not installed, falling back to label propagation")
            return await self.label_propagation(G, config)
        
        config = config or CommunityDetectionConfig()
        
        # Apply Louvain algorithm
        partition = community_louvain.best_partition(
            G, 
            resolution=config.resolution,
            random_state=config.random_state
        )
        
        # Group nodes by community
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)
        
        # Filter small communities
        result = [
            community_nodes 
            for community_nodes in communities.values() 
            if len(community_nodes) >= config.min_cluster_size
        ]
        
        logger.info(f"Louvain detection found {len(result)} communities")
        return result
    
    async def label_propagation(
        self,
        G: nx.Graph,
        config: Optional[CommunityDetectionConfig] = None
    ) -> List[List[str]]:
        """
        Detect communities using label propagation algorithm
        
        Args:
            G: NetworkX graph
            config: Detection configuration
            
        Returns:
            List of communities, each containing node UUIDs
        """
        config = config or CommunityDetectionConfig()
        
        # Apply label propagation
        communities = nx.community.label_propagation_communities(G)
        
        # Convert to list of lists and filter small communities
        result = [
            list(community) 
            for community in communities 
            if len(community) >= config.min_cluster_size
        ]
        
        logger.info(f"Label propagation found {len(result)} communities")
        return result
    
    async def connected_components(
        self,
        G: nx.Graph,
        config: Optional[CommunityDetectionConfig] = None
    ) -> List[List[str]]:
        """
        Detect communities using connected components
        
        Args:
            G: NetworkX graph
            config: Detection configuration
            
        Returns:
            List of communities, each containing node UUIDs
        """
        config = config or CommunityDetectionConfig()
        
        # Find connected components
        components = nx.connected_components(G)
        
        # Filter small components
        result = [
            list(component) 
            for component in components 
            if len(component) >= config.min_cluster_size
        ]
        
        logger.info(f"Connected components found {len(result)} communities")
        return result
    
    async def clique_percolation(
        self,
        G: nx.Graph,
        config: Optional[CommunityDetectionConfig] = None
    ) -> List[List[str]]:
        """
        Detect communities using clique percolation method
        
        Args:
            G: NetworkX graph
            config: Detection configuration
            
        Returns:
            List of communities, each containing node UUIDs
        """
        config = config or CommunityDetectionConfig()
        
        # Apply clique percolation
        communities = nx.community.k_clique_communities(G, k=3)  # k=3 for triangles
        
        # Convert to list of lists and filter small communities
        result = [
            list(community) 
            for community in communities 
            if len(community) >= config.min_cluster_size
        ]
        
        logger.info(f"Clique percolation found {len(result)} communities")
        return result
    
    async def kmeans_detection(
        self,
        G: nx.Graph,
        config: Optional[CommunityDetectionConfig] = None
    ) -> List[List[str]]:
        """
        Detect communities using K-means clustering on node embeddings
        
        Args:
            G: NetworkX graph
            config: Detection configuration
            
        Returns:
            List of communities, each containing node UUIDs
        """
        config = config or CommunityDetectionConfig()
        
        # Generate node embeddings
        node_embeddings = await self._generate_node_embeddings(G)
        
        if not node_embeddings:
            logger.warning("No embeddings generated, falling back to connected components")
            return await self.connected_components(G, config)
        
        # Determine optimal k if not specified
        if config.k_clusters is None:
            config.k_clusters = self._determine_optimal_k(node_embeddings)
        
        # Apply K-means clustering
        kmeans = KMeans(
            n_clusters=config.k_clusters,
            random_state=config.random_state,
            n_init=10
        )
        
        cluster_labels = kmeans.fit_predict(node_embeddings)
        
        # Group nodes by cluster
        communities = defaultdict(list)
        for i, node in enumerate(G.nodes()):
            communities[cluster_labels[i]].append(node)
        
        # Filter small communities
        result = [
            community_nodes 
            for community_nodes in communities.values() 
            if len(community_nodes) >= config.min_cluster_size
        ]
        
        logger.info(f"K-means found {len(result)} communities")
        return result
    
    async def hierarchical_detection(
        self,
        G: nx.Graph,
        config: Optional[CommunityDetectionConfig] = None
    ) -> List[List[str]]:
        """
        Detect communities using hierarchical clustering
        
        Args:
            G: NetworkX graph
            config: Detection configuration
            
        Returns:
            List of communities, each containing node UUIDs
        """
        config = config or CommunityDetectionConfig()
        
        # Generate node embeddings
        node_embeddings = await self._generate_node_embeddings(G)
        
        if not node_embeddings:
            logger.warning("No embeddings generated, falling back to connected components")
            return await self.connected_components(G, config)
        
        # Apply hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=config.k_clusters,
            linkage='ward'
        )
        
        cluster_labels = clustering.fit_predict(node_embeddings)
        
        # Group nodes by cluster
        communities = defaultdict(list)
        for i, node in enumerate(G.nodes()):
            communities[cluster_labels[i]].append(node)
        
        # Filter small communities
        result = [
            community_nodes 
            for community_nodes in communities.values() 
            if len(community_nodes) >= config.min_cluster_size
        ]
        
        logger.info(f"Hierarchical clustering found {len(result)} communities")
        return result
    
    async def analyze_community_structure(
        self,
        communities: List[List[str]],
        G: nx.Graph
    ) -> CommunityStats:
        """
        Analyze community structure and compute statistics
        
        Args:
            communities: List of communities (node UUIDs)
            G: NetworkX graph
            
        Returns:
            CommunityStats object with analysis results
        """
        if not communities:
            return CommunityStats(
                total_communities=0,
                average_size=0.0,
                largest_community_size=0,
                smallest_community_size=0,
                modularity=0.0
            )
        
        # Basic statistics
        total_communities = len(communities)
        community_sizes = [len(community) for community in communities]
        average_size = np.mean(community_sizes)
        largest_community_size = max(community_sizes)
        smallest_community_size = min(community_sizes)
        
        # Calculate modularity
        modularity = self._calculate_modularity(communities, G)
        
        # Calculate silhouette score if embeddings available
        silhouette_score_val = None
        node_embeddings = await self._generate_node_embeddings(G)
        if node_embeddings and len(communities) > 1:
            try:
                # Create cluster labels
                cluster_labels = []
                for i, community in enumerate(communities):
                    cluster_labels.extend([i] * len(community))
                
                if len(set(cluster_labels)) > 1:
                    silhouette_score_val = silhouette_score(
                        node_embeddings[:len(cluster_labels)], 
                        cluster_labels
                    )
            except Exception as e:
                logger.warning(f"Could not calculate silhouette score: {e}")
        
        # Detailed community information
        community_details = []
        for i, community in enumerate(communities):
            community_details.append({
                "community_id": i,
                "size": len(community),
                "members": community,
                "density": nx.density(G.subgraph(community)) if len(community) > 1 else 0.0
            })
        
        return CommunityStats(
            total_communities=total_communities,
            average_size=average_size,
            largest_community_size=largest_community_size,
            smallest_community_size=smallest_community_size,
            modularity=modularity,
            silhouette_score=silhouette_score_val,
            communities=community_details
        )
    
    async def get_community_stats(
        self,
        community_nodes: List[CommunityNode],
        community_edges: List[CommunityEdge]
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics about communities
        
        Args:
            community_nodes: List of community nodes
            community_edges: List of community edges
            
        Returns:
            Dictionary with community statistics
        """
        if not community_nodes:
            return {
                "total_communities": 0,
                "total_members": 0,
                "average_members_per_community": 0.0,
                "largest_community_size": 0,
                "smallest_community_size": 0
            }
        
        # Count members for each community
        community_sizes = []
        for community_node in community_nodes:
            # Count connected entity nodes
            connected_entities = await self._get_connected_entities(community_node.uuid)
            community_sizes.append(len(connected_entities))
        
        return {
            "total_communities": len(community_nodes),
            "total_members": sum(community_sizes),
            "average_members_per_community": np.mean(community_sizes),
            "largest_community_size": max(community_sizes) if community_sizes else 0,
            "smallest_community_size": min(community_sizes) if community_sizes else 0,
            "community_sizes": community_sizes
        }
    
    async def find_core_members(
        self,
        community_nodes: List[CommunityNode],
        G: nx.Graph,
        top_k: int = 5
    ) -> Dict[str, List[CommunityMember]]:
        """
        Find core members of each community based on centrality
        
        Args:
            community_nodes: List of community nodes
            G: NetworkX graph
            top_k: Number of core members to return per community
            
        Returns:
            Dictionary mapping community UUID to list of core members
        """
        core_members = {}
        
        for community_node in community_nodes:
            # Get connected entities
            connected_entities = await self._get_connected_entities(community_node.uuid)
            
            if not connected_entities:
                continue
            
            # Create subgraph for the community
            community_subgraph = G.subgraph(connected_entities)
            
            # Calculate centrality measures
            centrality = nx.degree_centrality(community_subgraph)
            
            # Get top-k core members
            sorted_nodes = sorted(
                centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_k]
            
            # Create member information
            members = []
            for node_uuid, centrality_score in sorted_nodes:
                try:
                    entity_node = await self.driver.get_node_by_uuid(node_uuid, "Entity")
                    members.append(CommunityMember(
                        node_uuid=node_uuid,
                        node_name=entity_node.name,
                        centrality=centrality_score,
                        role="core",
                        similarity_to_center=1.0  # Placeholder
                    ))
                except NodeNotFoundError:
                    continue
            
            core_members[community_node.uuid] = members
        
        return core_members
    
    async def find_bridges(
        self,
        community_nodes: List[CommunityNode],
        G: nx.Graph
    ) -> List[Dict[str, Any]]:
        """
        Find bridge nodes between communities
        
        Args:
            community_nodes: List of community nodes
            G: NetworkX graph
            
        Returns:
            List of bridge node information
        """
        bridges = []
        
        # Create mapping from node to community
        node_to_community = {}
        for community_node in community_nodes:
            connected_entities = await self._get_connected_entities(community_node.uuid)
            for entity_uuid in connected_entities:
                node_to_community[entity_uuid] = community_node.uuid
        
        # Find edges between different communities
        for u, v, data in G.edges(data=True):
            if u in node_to_community and v in node_to_community:
                if node_to_community[u] != node_to_community[v]:
                    bridges.append({
                        "source_node": u,
                        "target_node": v,
                        "source_community": node_to_community[u],
                        "target_community": node_to_community[v],
                        "edge_data": data
                    })
        
        return bridges
    
    async def community_similarity(
        self,
        community_nodes: List[CommunityNode],
        similarity_threshold: float = 0.5
    ) -> List[CommunitySimilarity]:
        """
        Calculate similarity between communities
        
        Args:
            community_nodes: List of community nodes
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of community similarity measures
        """
        similarities = []
        
        # Get member sets for each community
        community_members = {}
        for community_node in community_nodes:
            connected_entities = await self._get_connected_entities(community_node.uuid)
            community_members[community_node.uuid] = set(connected_entities)
        
        # Calculate pairwise similarities
        community_list = list(community_nodes)
        for i in range(len(community_list)):
            for j in range(i + 1, len(community_list)):
                comm1_uuid = community_list[i].uuid
                comm2_uuid = community_list[j].uuid
                
                members1 = community_members[comm1_uuid]
                members2 = community_members[comm2_uuid]
                
                if not members1 or not members2:
                    continue
                
                # Calculate Jaccard similarity
                intersection = len(members1.intersection(members2))
                union = len(members1.union(members2))
                jaccard_similarity = intersection / union if union > 0 else 0.0
                
                # Calculate cosine similarity using embeddings
                cosine_similarity = await self._calculate_cosine_similarity(
                    community_list[i], community_list[j]
                )
                
                # Only include if above threshold
                if jaccard_similarity >= similarity_threshold or cosine_similarity >= similarity_threshold:
                    similarities.append(CommunitySimilarity(
                        community1_uuid=comm1_uuid,
                        community2_uuid=comm2_uuid,
                        jaccard_similarity=jaccard_similarity,
                        cosine_similarity=cosine_similarity,
                        common_members=intersection,
                        total_members=len(union)
                    ))
        
        return similarities
    
    async def store_communities(
        self,
        community_nodes: List[CommunityNode],
        community_edges: List[CommunityEdge],
        commit_message: str = "Store community detection results"
    ) -> Dict[str, Any]:
        """
        Store community information in datasets
        
        Args:
            community_nodes: List of community nodes
            community_edges: List of community edges
            commit_message: Commit message for storage
            
        Returns:
            Dictionary with storage results
        """
        results = {"nodes_stored": 0, "edges_stored": 0, "errors": []}
        
        try:
            # Store community nodes
            for node in community_nodes:
                try:
                    await self.driver.save_node(node)
                    results["nodes_stored"] += 1
                except Exception as e:
                    results["errors"].append(f"Error storing node {node.uuid}: {e}")
            
            # Store community edges
            for edge in community_edges:
                try:
                    await self.driver.save_edge(edge)
                    results["edges_stored"] += 1
                except Exception as e:
                    results["errors"].append(f"Error storing edge {edge.uuid}: {e}")
            
            # Push to hub if driver supports it
            if hasattr(self.driver, '_push_to_hub'):
                self.driver._push_to_hub(commit_message)
            
            logger.info(f"Stored {results['nodes_stored']} community nodes and {results['edges_stored']} community edges")
            
        except Exception as e:
            logger.error(f"Error storing communities: {e}")
            results["errors"].append(f"Storage error: {e}")
        
        return results
    
    async def load_communities(
        self,
        group_ids: Optional[List[str]] = None
    ) -> Tuple[List[CommunityNode], List[CommunityEdge]]:
        """
        Load community information from datasets
        
        Args:
            group_ids: List of group IDs to filter by
            
        Returns:
            Tuple of (community_nodes, community_edges)
        """
        try:
            # Load community nodes
            community_nodes = await self.driver.get_nodes_by_group_ids(
                group_ids or [], "Community"
            )
            
            # Load community edges
            community_edges = await self.driver.get_edges_by_group_ids(
                group_ids or [], "Community"
            )
            
            logger.info(f"Loaded {len(community_nodes)} community nodes and {len(community_edges)} community edges")
            
            return community_nodes, community_edges
            
        except Exception as e:
            logger.error(f"Error loading communities: {e}")
            return [], []
    
    async def update_communities(
        self,
        community_nodes: List[CommunityNode],
        community_edges: List[CommunityEdge]
    ) -> Dict[str, Any]:
        """
        Update community assignments
        
        Args:
            community_nodes: Updated community nodes
            community_edges: Updated community edges
            
        Returns:
            Dictionary with update results
        """
        results = {"updated": 0, "errors": []}
        
        try:
            # Update community nodes
            for node in community_nodes:
                try:
                    await self.driver.save_node(node)
                    results["updated"] += 1
                except Exception as e:
                    results["errors"].append(f"Error updating node {node.uuid}: {e}")
            
            # Update community edges
            for edge in community_edges:
                try:
                    await self.driver.save_edge(edge)
                    results["updated"] += 1
                except Exception as e:
                    results["errors"].append(f"Error updating edge {edge.uuid}: {e}")
            
            logger.info(f"Updated {results['updated']} community entities")
            
        except Exception as e:
            logger.error(f"Error updating communities: {e}")
            results["errors"].append(f"Update error: {e}")
        
        return results
    
    async def community_versioning(
        self,
        action: str = "create",
        version_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track community evolution over time
        
        Args:
            action: Action to perform ('create', 'list', 'restore')
            version_id: Version ID for restore action
            
        Returns:
            Dictionary with versioning results
        """
        if action == "create":
            # Create new version
            version_id = version_id or str(uuid.uuid4())
            timestamp = utc_now().isoformat()
            
            # Get current communities
            community_nodes, community_edges = await self.load_communities()
            
            # Store version information
            version_data = {
                "version_id": version_id,
                "timestamp": timestamp,
                "community_count": len(community_nodes),
                "community_nodes": [node.dict() for node in community_nodes],
                "community_edges": [edge.dict() for edge in community_edges]
            }
            
            # Store version (simplified - in real implementation would use proper versioning)
            self._community_cache[version_id] = version_data
            
            return {
                "success": True,
                "version_id": version_id,
                "timestamp": timestamp,
                "community_count": len(community_nodes)
            }
        
        elif action == "list":
            # List available versions
            versions = list(self._community_cache.keys())
            return {
                "success": True,
                "versions": versions,
                "count": len(versions)
            }
        
        elif action == "restore":
            if not version_id:
                return {"success": False, "error": "Version ID required for restore"}
            
            if version_id not in self._community_cache:
                return {"success": False, "error": f"Version {version_id} not found"}
            
            # Restore communities from version
            version_data = self._community_cache[version_id]
            
            # Recreate community nodes and edges
            community_nodes = []
            for node_data in version_data["community_nodes"]:
                community_nodes.append(CommunityNode(**node_data))
            
            community_edges = []
            for edge_data in version_data["community_edges"]:
                community_edges.append(CommunityEdge(**edge_data))
            
            # Store restored communities
            result = await self.store_communities(community_nodes, community_edges, f"Restore version {version_id}")
            
            return {
                "success": True,
                "version_id": version_id,
                "restored_communities": len(community_nodes),
                "result": result
            }
        
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
    
    async def export_communities(
        self,
        format: str = "json",
        include_embeddings: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Export community data
        
        Args:
            format: Export format ('json', 'csv', 'parquet')
            include_embeddings: Whether to include embeddings in export
            
        Returns:
            Exported data as string or dictionary
        """
        # Load communities
        community_nodes, community_edges = await self.load_communities()
        
        # Prepare export data
        export_data = {
            "export_timestamp": utc_now().isoformat(),
            "total_communities": len(community_nodes),
            "total_edges": len(community_edges),
            "communities": []
        }
        
        # Add community data
        for community_node in community_nodes:
            community_data = {
                "uuid": community_node.uuid,
                "name": community_node.name,
                "group_id": community_node.group_id,
                "created_at": community_node.created_at.isoformat(),
                "summary": community_node.summary,
                "name_embedding": community_node.name_embedding if include_embeddings else None
            }
            
            # Get connected entities
            connected_entities = await self._get_connected_entities(community_node.uuid)
            community_data["members"] = connected_entities
            
            export_data["communities"].append(community_data)
        
        # Export in requested format
        if format == "json":
            return json.dumps(export_data, indent=2)
        
        elif format == "csv":
            # Create CSV representation
            rows = []
            for community in export_data["communities"]:
                rows.append({
                    "community_uuid": community["uuid"],
                    "community_name": community["name"],
                    "group_id": community["group_id"],
                    "created_at": community["created_at"],
                    "summary": community["summary"],
                    "members": "|".join(community["members"])
                })
            
            df = pd.DataFrame(rows)
            return df.to_csv(index=False)
        
        elif format == "parquet":
            # Create Parquet representation
            rows = []
            for community in export_data["communities"]:
                rows.append({
                    "uuid": community["uuid"],
                    "name": community["name"],
                    "group_id": community["group_id"],
                    "created_at": community["created_at"],
                    "summary": community["summary"],
                    "members": "|".join(community["members"]),
                    "name_embedding": json.dumps(community["name_embedding"]) if community["name_embedding"] else None
                })
            
            df = pd.DataFrame(rows)
            return df.to_parquet(index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def batch_community_detection(
        self,
        group_id_batches: List[List[str]],
        config: Optional[CommunityDetectionConfig] = None
    ) -> List[Tuple[List[CommunityNode], List[CommunityEdge]]]:
        """
        Process large graphs in batches
        
        Args:
            group_id_batches: List of group ID batches
            config: Detection configuration
            
        Returns:
            List of (community_nodes, community_edges) for each batch
        """
        config = config or CommunityDetectionConfig()
        results = []
        
        for i, batch in enumerate(group_id_batches):
            logger.info(f"Processing batch {i+1}/{len(group_id_batches)} with {len(batch)} group IDs")
            
            try:
                communities = await self.detect_communities(batch, config)
                results.append(communities)
            except Exception as e:
                logger.error(f"Error processing batch {i+1}: {e}")
                results.append(([], []))
        
        return results
    
    async def incremental_community_update(
        self,
        new_nodes: List[EntityNode],
        new_edges: List[EntityEdge],
        existing_communities: Optional[List[CommunityNode]] = None
    ) -> Tuple[List[CommunityNode], List[CommunityEdge]]:
        """
        Update communities incrementally with new nodes and edges
        
        Args:
            new_nodes: List of new entity nodes
            new_edges: List of new entity edges
            existing_communities: List of existing communities
            
        Returns:
            Updated (community_nodes, community_edges)
        """
        # Build graph with new data
        G = await self._build_graph_from_dataset()
        
        # Add new nodes and edges to graph
        for node in new_nodes:
            G.add_node(node.uuid, name=node.name)
        
        for edge in new_edges:
            G.add_edge(edge.source_node_uuid, edge.target_node_uuid)
        
        # Detect communities on updated graph
        config = CommunityDetectionConfig()
        new_communities = await self.detect_communities(config=config)
        
        # Merge with existing communities if provided
        if existing_communities:
            # This is a simplified merge - in practice would need more sophisticated merging
            all_communities = existing_communities + new_communities[0]
            return all_communities, new_communities[1]
        
        return new_communities
    
    async def parallel_community_detection(
        self,
        group_ids: List[str],
        config: Optional[CommunityDetectionConfig] = None,
        max_workers: int = 4
    ) -> Tuple[List[CommunityNode], List[CommunityEdge]]:
        """
        Perform parallel community detection for multiple group IDs
        
        Args:
            group_ids: List of group IDs
            config: Detection configuration
            max_workers: Maximum number of parallel workers
            
        Returns:
            Tuple of (community_nodes, community_edges)
        """
        config = config or CommunityDetectionConfig()
        
        # Split group IDs into chunks for parallel processing
        chunk_size = max(1, len(group_ids) // max_workers)
        group_chunks = [
            group_ids[i:i + chunk_size] 
            for i in range(0, len(group_ids), chunk_size)
        ]
        
        # Process chunks in parallel
        tasks = []
        for chunk in group_chunks:
            task = self.detect_communities(chunk, config)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_community_nodes = []
        all_community_edges = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in parallel detection: {result}")
                continue
            
            community_nodes, community_edges = result
            all_community_nodes.extend(community_nodes)
            all_community_edges.extend(community_edges)
        
        return all_community_nodes, all_community_edges
    
    async def community_indexing(
        self,
        community_nodes: List[CommunityNode],
        strategy: str = "embedding"
    ) -> Dict[str, Any]:
        """
        Create efficient indices for community queries
        
        Args:
            community_nodes: List of community nodes
            strategy: Indexing strategy ('embedding', 'size', 'temporal')
            
        Returns:
            Dictionary with indexing results
        """
        results = {"indexed": 0, "errors": []}
        
        try:
            if strategy == "embedding":
                # Create embedding-based index
                if self.sentence_transformer:
                    embeddings = []
                    for node in community_nodes:
                        if node.name_embedding:
                            embeddings.append(node.name_embedding)
                    
                    if embeddings:
                        # Create FAISS-like index (simplified)
                        embedding_matrix = np.array(embeddings)
                        results["embedding_index"] = {
                            "shape": embedding_matrix.shape,
                            "method": "cosine_similarity"
                        }
                        results["indexed"] = len(embeddings)
            
            elif strategy == "size":
                # Create size-based index
                size_index = defaultdict(list)
                for node in community_nodes:
                    size = len(await self._get_connected_entities(node.uuid))
                    size_index[size].append(node.uuid)
                
                results["size_index"] = dict(size_index)
                results["indexed"] = len(community_nodes)
            
            elif strategy == "temporal":
                # Create temporal index
                temporal_index = defaultdict(list)
                for node in community_nodes:
                    created_date = node.created_at.date().isoformat()
                    temporal_index[created_date].append(node.uuid)
                
                results["temporal_index"] = dict(temporal_index)
                results["indexed"] = len(community_nodes)
            
            else:
                results["errors"].append(f"Unknown indexing strategy: {strategy}")
            
            logger.info(f"Created {strategy} index for {results['indexed']} communities")
            
        except Exception as e:
            logger.error(f"Error creating {strategy} index: {e}")
            results["errors"].append(f"Indexing error: {e}")
        
        return results
    
    async def community_caching(
        self,
        communities: Tuple[List[CommunityNode], List[CommunityEdge]],
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cache community results for faster retrieval
        
        Args:
            communities: Tuple of (community_nodes, community_edges)
            cache_key: Optional cache key
            
        Returns:
            Dictionary with caching results
        """
        cache_key = cache_key or f"communities_{utc_now().isoformat()}"
        
        try:
            # Prepare cache data
            cache_data = {
                "timestamp": utc_now().isoformat(),
                "community_nodes": [node.dict() for node in communities[0]],
                "community_edges": [edge.dict() for edge in communities[1]],
                "ttl": self._cache_ttl
            }
            
            # Store in cache
            self._community_cache[cache_key] = cache_data
            
            # Clean expired cache entries
            await self._clean_expired_cache()
            
            return {
                "success": True,
                "cache_key": cache_key,
                "cached_communities": len(communities[0]),
                "cache_size": len(self._community_cache)
            }
            
        except Exception as e:
            logger.error(f"Error caching communities: {e}")
            return {"success": False, "error": str(e)}
    
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
        if not cache_key:
            # Return most recent cache entry
            if not self._community_cache:
                return None
            
            cache_key = max(self._community_cache.keys())
        
        if cache_key not in self._community_cache:
            return None
        
        cache_data = self._community_cache[cache_key]
        
        # Check if cache is expired
        cache_time = datetime.fromisoformat(cache_data["timestamp"])
        if (utc_now() - cache_time).seconds > cache_data["ttl"]:
            del self._community_cache[cache_key]
            return None
        
        # Recreate community objects
        community_nodes = []
        for node_data in cache_data["community_nodes"]:
            community_nodes.append(CommunityNode(**node_data))
        
        community_edges = []
        for edge_data in cache_data["community_edges"]:
            community_edges.append(CommunityEdge(**edge_data))
        
        return community_nodes, community_edges
    
    # Helper methods
    
    async def _build_graph_from_dataset(
        self,
        group_ids: Optional[List[str]] = None
    ) -> nx.Graph:
        """Build NetworkX graph from dataset"""
        G = nx.Graph()
        
        # Get nodes
        nodes = await self.driver.get_nodes_by_group_ids(group_ids or [], "Entity")
        for node in nodes:
            G.add_node(node.uuid, name=node.name, group_id=node.group_id)
        
        # Get edges
        edges = await self.driver.get_edges_by_group_ids(group_ids or [], "Entity")
        for edge in edges:
            G.add_edge(edge.source_node_uuid, edge.target_node_uuid)
        
        logger.info(f"Built graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        return G
    
    async def _generate_node_embeddings(self, G: nx.Graph) -> Optional[np.ndarray]:
        """Generate embeddings for graph nodes"""
        if not self.sentence_transformer:
            return None
        
        try:
            # Generate node names for embedding
            node_names = [G.nodes[node].get('name', f'node_{node}') for node in G.nodes()]
            
            # Create embeddings
            embeddings = self.sentence_transformer.encode(node_names)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating node embeddings: {e}")
            return None
    
    def _determine_optimal_k(self, embeddings: np.ndarray) -> int:
        """Determine optimal number of clusters using elbow method"""
        if len(embeddings) < 2:
            return 1
        
        # Try different k values and calculate inertia
        inertias = []
        k_range = range(1, min(10, len(embeddings) // 2 + 1))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection
        if len(inertias) < 3:
            return 2
        
        # Find elbow point
        diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        max_diff_idx = diffs.index(max(diffs))
        
        return max_diff_idx + 2  # +2 because k starts from 1 and we want the point after max diff
    
    def _calculate_modularity(self, communities: List[List[str]], G: nx.Graph) -> float:
        """Calculate modularity of community partition"""
        try:
            # Create partition dictionary
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
            
            # Calculate modularity
            modularity = nx.community.modularity(G, partition)
            return modularity
            
        except Exception as e:
            logger.warning(f"Could not calculate modularity: {e}")
            return 0.0
    
    async def _get_connected_entities(self, community_uuid: str) -> List[str]:
        """Get entity nodes connected to a community"""
        try:
            # Get community edges
            edges = await self.driver.get_edges_by_group_ids([], "Community")
            
            # Find edges connected to this community
            connected_entities = set()
            for edge in edges:
                if edge.source_node_uuid == community_uuid:
                    connected_entities.add(edge.target_node_uuid)
                elif edge.target_node_uuid == community_uuid:
                    connected_entities.add(edge.source_node_uuid)
            
            return list(connected_entities)
            
        except Exception as e:
            logger.error(f"Error getting connected entities: {e}")
            return []
    
    async def _calculate_cosine_similarity(
        self,
        node1: CommunityNode,
        node2: CommunityNode
    ) -> float:
        """Calculate cosine similarity between two community nodes"""
        if not node1.name_embedding or not node2.name_embedding:
            return 0.0
        
        try:
            emb1 = np.array(node1.name_embedding)
            emb2 = np.array(node2.name_embedding)
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def _create_community_entities(
        self,
        communities: List[List[str]],
        group_ids: Optional[List[str]] = None
    ) -> Tuple[List[CommunityNode], List[CommunityEdge]]:
        """Create CommunityNode and CommunityEdge objects from detected communities"""
        community_nodes = []
        community_edges = []
        
        for i, community in enumerate(communities):
            if not community:
                continue
            
            # Create community node
            community_node = CommunityNode(
                uuid=str(uuid.uuid4()),
                name=f"Community_{i+1}",
                group_id=group_ids[0] if group_ids else "default",
                labels=["Community"],
                created_at=utc_now(),
                summary=f"Community with {len(community)} members"
            )
            
            community_nodes.append(community_node)
            
            # Create community edges
            for node_uuid in community:
                try:
                    # Get entity node
                    entity_node = await self.driver.get_node_by_uuid(node_uuid, "Entity")
                    
                    # Create community edge
                    community_edge = CommunityEdge(
                        uuid=str(uuid.uuid4()),
                        source_node_uuid=community_node.uuid,
                        target_node_uuid=entity_node.uuid,
                        name="HAS_MEMBER",
                        fact=f"{community_node.name} contains {entity_node.name}",
                        group_id=community_node.group_id,
                        created_at=utc_now()
                    )
                    
                    community_edges.append(community_edge)
                    
                except NodeNotFoundError:
                    logger.warning(f"Node {node_uuid} not found, skipping")
                    continue
        
        return community_nodes, community_edges
    
    async def _clean_expired_cache(self):
        """Clean expired cache entries"""
        current_time = utc_now()
        expired_keys = []
        
        for cache_key, cache_data in self._community_cache.items():
            cache_time = datetime.fromisoformat(cache_data["timestamp"])
            if (current_time - cache_time).seconds > cache_data["ttl"]:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self._community_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned {len(expired_keys)} expired cache entries")