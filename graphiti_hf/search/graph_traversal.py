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
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum

import pandas as pd
# import numpy as np  # Not used in this implementation

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver

logger = logging.getLogger(__name__)


class TraversalAlgorithm(Enum):
    """Supported traversal algorithms"""
    BFS = "bfs"
    DFS = "dfs"


class EdgeFilterType(Enum):
    """Types of edge filtering"""
    ALL = "all"
    INCOMING = "incoming"
    OUTGOING = "outgoing"


@dataclass
class TraversalConfig:
    """Configuration for graph traversal operations"""
    max_depth: int = 5
    max_path_length: int = 10
    algorithm: TraversalAlgorithm = TraversalAlgorithm.BFS
    weighted: bool = False
    edge_filter: EdgeFilterType = EdgeFilterType.ALL
    edge_types: Optional[List[str]] = None
    temporal_filter: Optional[datetime] = None
    early_termination_size: Optional[int] = None
    batch_size: int = 1000
    cache_enabled: bool = True
    max_cache_size: int = 10000


@dataclass
class TraversalResult:
    """Result of a graph traversal operation"""
    nodes: List[EntityNode]
    edges: List[EntityEdge]
    paths: List[List[str]]
    traversal_stats: Dict[str, Any]


class GraphTraversalEngine:
    """
    Graph traversal engine for HuggingFace-based knowledge graphs.
    
    Provides efficient BFS/DFS traversal, path finding, and subgraph extraction
    using pandas operations and pre-computed adjacency lists.
    """
    
    def __init__(self, driver: HuggingFaceDriver, config: Optional[TraversalConfig] = None):
        """
        Initialize the graph traversal engine.
        
        Args:
            driver: HuggingFaceDriver instance
            config: Traversal configuration
        """
        self.driver = driver
        self.config = config or TraversalConfig()
        
        # Cache for adjacency lists
        self._adjacency_cache: Dict[str, Dict[str, List[str]]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Performance optimization
        self._adjacency_lists: Dict[str, Dict[str, List[str]]] = {}
        self._reverse_adjacency_lists: Dict[str, Dict[str, List[str]]] = {}
        
        # Build adjacency lists on initialization
        self._build_adjacency_lists()
    
    def _build_adjacency_lists(self):
        """Build adjacency lists for efficient traversal"""
        if self.driver.edges_df.empty:
            return
        
        # Build forward and reverse adjacency lists
        self._adjacency_lists = defaultdict(lambda: defaultdict(list))
        self._reverse_adjacency_lists = defaultdict(lambda: defaultdict(list))
        
        for _, edge in self.driver.edges_df.iterrows():
            source = edge['source_uuid']
            target = edge['target_uuid']
            
            # Forward adjacency
            self._adjacency_lists[source][target].append(edge['uuid'])
            # Reverse adjacency
            self._reverse_adjacency_lists[target][source].append(edge['uuid'])
    
    def _get_neighbors(self, node_uuid: str, edge_filter: EdgeFilterType = EdgeFilterType.ALL) -> List[str]:
        """Get neighbors of a node based on edge filter"""
        if edge_filter == EdgeFilterType.ALL:
            return list(self._adjacency_lists.get(node_uuid, {}).keys())
        elif edge_filter == EdgeFilterType.OUTGOING:
            return list(self._adjacency_lists.get(node_uuid, {}).keys())
        elif edge_filter == EdgeFilterType.INCOMING:
            return list(self._reverse_adjacency_lists.get(node_uuid, {}).keys())
        else:
            return []
    
    def _get_edges_between_nodes(self, source_uuid: str, target_uuid: str) -> List[EntityEdge]:
        """Get edges between two nodes"""
        edge_uuids = self._adjacency_lists.get(source_uuid, {}).get(target_uuid, [])
        edges = []
        
        for edge_uuid in edge_uuids:
            try:
                edge = asyncio.run(self.driver.get_edge_by_uuid(edge_uuid, "Entity"))
                edges.append(edge)
            except Exception as e:
                logger.warning(f"Failed to get edge {edge_uuid}: {e}")
        
        return edges
    
    def _apply_edge_filters(self, edges: List[EntityEdge], edge_types: Optional[List[str]] = None) -> List[EntityEdge]:
        """Apply edge type filters"""
        if not edge_types:
            return edges
        
        filtered_edges = []
        for edge in edges:
            if edge.name in edge_types:
                filtered_edges.append(edge)
        
        return filtered_edges
    
    def _apply_temporal_filter(self, edges: List[EntityEdge], filter_time: Optional[datetime] = None) -> List[EntityEdge]:
        """Apply temporal filtering to edges"""
        if not filter_time:
            return edges
        
        filtered_edges = []
        for edge in edges:
            if edge.valid_at and edge.valid_at <= filter_time:
                if not edge.invalidated_at or edge.invalidated_at > filter_time:
                    filtered_edges.append(edge)
        
        return filtered_edges
    
    async def bfs_traversal(
        self,
        start_nodes: List[str],
        config: Optional[TraversalConfig] = None
    ) -> TraversalResult:
        """
        Perform Breadth-First Search traversal.
        
        Args:
            start_nodes: List of starting node UUIDs
            config: Traversal configuration
            
        Returns:
            TraversalResult containing nodes, edges, and paths
        """
        config = config or self.config
        
        if not start_nodes:
            return TraversalResult([], [], [], {"visited_nodes": 0, "visited_edges": 0})
        
        # Initialize traversal data structures
        visited_nodes = set(start_nodes)
        visited_edges = set()
        queue = deque(start_nodes)
        current_level = 0
        traversal_paths = defaultdict(list)
        
        # Track parent relationships for path reconstruction
        parent_map = {node: None for node in start_nodes}
        
        nodes_found = []
        edges_found = []
        
        while queue and current_level < config.max_depth:
            level_size = len(queue)
            
            # Process all nodes at current level
            for _ in range(level_size):
                current_node = queue.popleft()
                
                # Get neighbors based on edge filter
                neighbors = self._get_neighbors(current_node, config.edge_filter)
                
                for neighbor in neighbors:
                    if neighbor not in visited_nodes:
                        # Mark node as visited
                        visited_nodes.add(neighbor)
                        queue.append(neighbor)
                        parent_map[neighbor] = current_node  # type: ignore
                        
                        # Get edges between nodes
                        edges = self._get_edges_between_nodes(current_node, neighbor)
                        
                        # Apply filters
                        edges = self._apply_edge_filters(edges, config.edge_types)
                        edges = self._apply_temporal_filter(edges, config.temporal_filter)
                        
                        for edge in edges:
                            if edge.uuid not in visited_edges:
                                visited_edges.add(edge.uuid)
                                edges_found.append(edge)
                                
                                # Record path
                                path = self._reconstruct_path(current_node, neighbor, parent_map)  # type: ignore
                                traversal_paths[current_node].append({
                                    'target': neighbor,
                                    'path': path,
                                    'edge': edge
                                })
                        
                        # Get neighbor node details
                        try:
                            neighbor_node = await self.driver.get_node_by_uuid(neighbor, "Entity")
                            nodes_found.append(neighbor_node)
                        except Exception as e:
                            logger.warning(f"Failed to get node {neighbor}: {e}")
                
                # Check early termination
                if config.early_termination_size and len(visited_nodes) >= config.early_termination_size:
                    break
            
            current_level += 1
            
            # Check early termination
            if config.early_termination_size and len(visited_nodes) >= config.early_termination_size:
                break
        
        # Convert visited nodes to EntityNode objects
        start_node_objects = []
        for node_uuid in start_nodes:
            try:
                node = await self.driver.get_node_by_uuid(node_uuid, "Entity")
                start_node_objects.append(node)
            except Exception as e:
                logger.warning(f"Failed to get start node {node_uuid}: {e}")
        
        all_nodes = start_node_objects + nodes_found
        
        # Build path information
        paths_info = []
        for source, targets in traversal_paths.items():
            for target_info in targets:
                paths_info.append(target_info['path'])
        
        return TraversalResult(
            nodes=all_nodes,
            edges=edges_found,
            paths=paths_info,
            traversal_stats={
                "visited_nodes": len(visited_nodes),
                "visited_edges": len(visited_edges),
                "max_depth_reached": current_level,
                "algorithm": "BFS"
            }
        )
    
    async def dfs_traversal(
        self,
        start_nodes: List[str],
        config: Optional[TraversalConfig] = None
    ) -> TraversalResult:
        """
        Perform Depth-First Search traversal.
        
        Args:
            start_nodes: List of starting node UUIDs
            config: Traversal configuration
            
        Returns:
            TraversalResult containing nodes, edges, and paths
        """
        config = config or self.config
        
        if not start_nodes:
            return TraversalResult([], [], [], {"visited_nodes": 0, "visited_edges": 0})
        
        # Initialize traversal data structures
        visited_nodes = set(start_nodes)
        visited_edges = set()
        stack = list(start_nodes)
        current_depth = 0
        traversal_paths = defaultdict(list)
        
        # Track parent relationships for path reconstruction
        parent_map = {node: None for node in start_nodes}
        
        nodes_found = []
        edges_found = []
        
        while stack and current_depth < config.max_depth:
            current_node = stack.pop()
            
            # Get neighbors based on edge filter
            neighbors = self._get_neighbors(current_node, config.edge_filter)
            
            for neighbor in neighbors:
                if neighbor not in visited_nodes:
                    # Mark node as visited
                    visited_nodes.add(neighbor)
                    stack.append(neighbor)
                    parent_map[neighbor] = current_node  # type: ignore
                    
                    # Get edges between nodes
                    edges = self._get_edges_between_nodes(current_node, neighbor)
                    
                    # Apply filters
                    edges = self._apply_edge_filters(edges, config.edge_types)
                    edges = self._apply_temporal_filter(edges, config.temporal_filter)
                    
                    for edge in edges:
                        if edge.uuid not in visited_edges:
                            visited_edges.add(edge.uuid)
                            edges_found.append(edge)
                            
                            # Record path
                            path = self._reconstruct_path(current_node, neighbor, parent_map)  # type: ignore
                            traversal_paths[current_node].append({
                                'target': neighbor,
                                'path': path,
                                'edge': edge
                            })
                    
                    # Get neighbor node details
                    try:
                        neighbor_node = await self.driver.get_node_by_uuid(neighbor, "Entity")
                        nodes_found.append(neighbor_node)
                    except Exception as e:
                        logger.warning(f"Failed to get node {neighbor}: {e}")
            
            current_depth += 1
            
            # Check early termination
            if config.early_termination_size and len(visited_nodes) >= config.early_termination_size:
                break
        
        # Convert visited nodes to EntityNode objects
        start_node_objects = []
        for node_uuid in start_nodes:
            try:
                node = await self.driver.get_node_by_uuid(node_uuid, "Entity")
                start_node_objects.append(node)
            except Exception as e:
                logger.warning(f"Failed to get start node {node_uuid}: {e}")
        
        all_nodes = start_node_objects + nodes_found
        
        # Build path information
        paths_info = []
        for source, targets in traversal_paths.items():
            for target_info in targets:
                paths_info.append(target_info['path'])
        
        return TraversalResult(
            nodes=all_nodes,
            edges=edges_found,
            paths=paths_info,
            traversal_stats={
                "visited_nodes": len(visited_nodes),
                "visited_edges": len(visited_edges),
                "max_depth_reached": current_depth,
                "algorithm": "DFS"
            }
        )
    
    async def find_paths(
        self,
        start_nodes: List[str],
        target_nodes: List[str],
        config: Optional[TraversalConfig] = None
    ) -> List[List[str]]:
        """
        Find paths between start and target nodes.
        
        Args:
            start_nodes: List of starting node UUIDs
            target_nodes: List of target node UUIDs
            config: Traversal configuration
            
        Returns:
            List of paths, where each path is a list of node UUIDs
        """
        config = config or self.config
        
        if not start_nodes or not target_nodes:
            return []
        
        # Use BFS for shortest path finding
        result = await self.bfs_traversal(start_nodes, config)
        
        # Filter paths that end at target nodes
        target_set = set(target_nodes)
        valid_paths = []
        
        for path in result.paths:
            if path[-1] in target_set:
                valid_paths.append(path)
        
        return valid_paths
    
    async def get_neighbors(
        self,
        node_uuids: List[str],
        depth: int = 1,
        config: Optional[TraversalConfig] = None
    ) -> Dict[str, List[EntityEdge]]:
        """
        Get direct neighbors of nodes.
        
        Args:
            node_uuids: List of node UUIDs
            depth: Depth of neighbor search (1 = direct neighbors only)
            config: Traversal configuration
            
        Returns:
            Dictionary mapping node UUIDs to lists of neighboring edges
        """
        config = config or self.config
        config.max_depth = depth
        
        result = await self.bfs_traversal(node_uuids, config)
        
        # Group edges by source node
        neighbor_edges = defaultdict(list)
        for edge in result.edges:
            # Find which start node this edge connects to
            for start_node in node_uuids:
                if edge.source_node_uuid == start_node or edge.target_node_uuid == start_node:
                    neighbor_edges[start_node].append(edge)
                    break
        
        return dict(neighbor_edges)
    
    async def subgraph_extraction(
        self,
        node_uuids: List[str],
        config: Optional[TraversalConfig] = None
    ) -> Tuple[List[EntityNode], List[EntityEdge]]:
        """
        Extract connected subgraph containing specified nodes.
        
        Args:
            node_uuids: List of node UUIDs to include in subgraph
            config: Traversal configuration
            
        Returns:
            Tuple of (nodes, edges) in the subgraph
        """
        config = config or self.config
        
        # Perform traversal to get all connected nodes and edges
        result = await self.bfs_traversal(node_uuids, config)
        
        # Filter to only include nodes that are connected to the specified nodes
        connected_nodes = set()
        for path in result.paths:
            for node in path:
                connected_nodes.add(node)
        
        # Add the original specified nodes
        connected_nodes.update(node_uuids)
        
        # Filter nodes to only include connected ones
        filtered_nodes = [node for node in result.nodes if node.uuid in connected_nodes]
        
        # Filter edges to only include those between connected nodes
        filtered_edges = []
        for edge in result.edges:
            if edge.source_node_uuid in connected_nodes and edge.target_node_uuid in connected_nodes:
                filtered_edges.append(edge)
        
        return filtered_nodes, filtered_edges
    
    def _reconstruct_path(self, source: str, target: str, parent_map: Dict[str, Optional[str]]) -> List[str]:
        """Reconstruct path from source to target using parent map"""
        path = []
        current = target
        
        while current is not None:
            path.append(current)
            current = parent_map.get(current)
        
        # Reverse to get path from source to target
        path.reverse()
        
        # Only return path if it starts with source
        if path and path[0] == source:
            return path
        
        return []
    
    def get_traversal_stats(self) -> Dict[str, Any]:
        """Get statistics about the traversal engine"""
        return {
            "adjacency_lists_built": len(self._adjacency_lists),
            "reverse_adjacency_lists_built": len(self._reverse_adjacency_lists),
            "cache_size": len(self._adjacency_cache),
            "config": self.config.__dict__
        }
    
    def clear_cache(self):
        """Clear the adjacency list cache"""
        self._adjacency_cache.clear()
        self._cache_timestamps.clear()
    
    async def batch_traversal(
        self,
        start_node_groups: List[List[str]],
        config: Optional[TraversalConfig] = None
    ) -> List[TraversalResult]:
        """
        Perform batch traversal operations for multiple groups of start nodes.
        
        Args:
            start_node_groups: List of start node groups
            config: Traversal configuration
            
        Returns:
            List of TraversalResult objects
        """
        config = config or self.config
        
        # Process each group in sequence
        results = []
        for i, group in enumerate(start_node_groups):
            logger.info(f"Processing group {i+1}/{len(start_node_groups)} with {len(group)} start nodes")
            
            if config.algorithm == TraversalAlgorithm.BFS:
                result = await self.bfs_traversal(group, config)
            else:
                result = await self.dfs_traversal(group, config)
            
            results.append(result)
        
        return results