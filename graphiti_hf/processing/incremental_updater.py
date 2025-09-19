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
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from graphiti_core.nodes import EntityNode, EpisodicNode, CommunityNode
from graphiti_core.edges import EntityEdge, EpisodicEdge, CommunityEdge
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver

logger = logging.getLogger(__name__)


class DeltaOperationType(Enum):
    """Types of delta operations"""
    ADD = "add"
    UPDATE = "update"
    REMOVE = "remove"


class DeltaEntityType(Enum):
    """Types of entities that can be in a delta"""
    NODE = "node"
    EDGE = "edge"
    EPISODE = "episode"
    COMMUNITY = "community"


@dataclass
class DeltaOperation:
    """Represents a single operation in a delta"""
    operation_type: DeltaOperationType
    entity_type: DeltaEntityType
    uuid: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Delta:
    """Represents a collection of operations"""
    operations: List[DeltaOperation] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
    status: str = "pending"  # pending, applied, failed, rolled_back
    rollback_data: Optional[Dict[str, Any]] = None
    
    def add_operation(self, operation: DeltaOperation):
        """Add an operation to the delta"""
        self.operations.append(operation)
    
    def get_operations_by_type(self, operation_type: DeltaOperationType) -> List[DeltaOperation]:
        """Get operations filtered by type"""
        return [op for op in self.operations if op.operation_type == operation_type]
    
    def get_operations_by_entity_type(self, entity_type: DeltaEntityType) -> List[DeltaOperation]:
        """Get operations filtered by entity type"""
        return [op for op in self.operations if op.entity_type == entity_type]
    
    def is_empty(self) -> bool:
        """Check if delta is empty"""
        return len(self.operations) == 0
    
    def size(self) -> int:
        """Get the number of operations in the delta"""
        return len(self.operations)


class IncrementalUpdater:
    """
    Handles incremental updates to the HuggingFace knowledge graph without full rebuilds.
    
    This class provides methods for adding, updating, and removing entities and edges
    efficiently, maintaining data consistency and supporting batch operations.
    """
    
    def __init__(self, driver: HuggingFaceDriver, max_batch_size: int = 1000, 
                 max_workers: int = 4):
        """
        Initialize the IncrementalUpdater.
        
        Args:
            driver: HuggingFaceDriver instance
            max_batch_size: Maximum batch size for operations
            max_workers: Maximum number of workers for parallel processing
        """
        self.driver = driver
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Track pending deltas for rollback
        self.pending_deltas: Dict[str, Delta] = {}
        
        # Performance tracking
        self.update_stats = {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "last_update_time": None,
            "average_update_time": 0.0
        }
    
    # Entity Operations
    async def add_entities_incremental(self, nodes: List[EntityNode], 
                                     group_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Add new entities to the knowledge graph without full rebuild.
        
        Args:
            nodes: List of EntityNode objects to add
            group_ids: Optional list of group IDs to filter by
            
        Returns:
            Dictionary containing update statistics and results
        """
        start_time = datetime.now()
        results = {"added": 0, "skipped": 0, "errors": []}
        
        try:
            # Create delta for this operation
            delta = Delta()
            
            # Process nodes in batches
            for i in range(0, len(nodes), self.max_batch_size):
                batch = nodes[i:i + self.max_batch_size]
                
                batch_results = await self._process_node_batch(batch, "add", delta)
                results["added"] += batch_results["added"]
                results["skipped"] += batch_results["skipped"]
                results["errors"].extend(batch_results["errors"])
            
            # Apply the delta
            await self.apply_delta(delta)
            
            # Update statistics
            self._update_statistics(len(nodes), datetime.now() - start_time)
            
            logger.info(f"Added {results['added']} entities incrementally")
            
        except Exception as e:
            logger.error(f"Error adding entities incrementally: {e}")
            results["errors"].append(str(e))
            self.update_stats["failed_updates"] += 1
        
        return results
    
    async def update_entities_incremental(self, nodes: List[EntityNode], 
                                        group_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Update existing entities in the knowledge graph without full rebuild.
        
        Args:
            nodes: List of EntityNode objects to update
            group_ids: Optional list of group IDs to filter by
            
        Returns:
            Dictionary containing update statistics and results
        """
        start_time = datetime.now()
        results = {"updated": 0, "not_found": 0, "errors": []}
        
        try:
            # Create delta for this operation
            delta = Delta()
            
            # Process nodes in batches
            for i in range(0, len(nodes), self.max_batch_size):
                batch = nodes[i:i + self.max_batch_size]
                
                batch_results = await self._process_node_batch(batch, "update", delta)
                results["updated"] += batch_results["updated"]
                results["not_found"] += batch_results["not_found"]
                results["errors"].extend(batch_results["errors"])
            
            # Apply the delta
            await self.apply_delta(delta)
            
            # Update statistics
            self._update_statistics(len(nodes), datetime.now() - start_time)
            
            logger.info(f"Updated {results['updated']} entities incrementally")
            
        except Exception as e:
            logger.error(f"Error updating entities incrementally: {e}")
            results["errors"].append(str(e))
            self.update_stats["failed_updates"] += 1
        
        return results
    
    async def remove_entities_incremental(self, node_uuids: List[str], 
                                        cleanup_related_edges: bool = True) -> Dict[str, Any]:
        """
        Remove entities from the knowledge graph without full rebuild.
        
        Args:
            node_uuids: List of node UUIDs to remove
            cleanup_related_edges: Whether to remove related edges
            
        Returns:
            Dictionary containing update statistics and results
        """
        start_time = datetime.now()
        results = {"removed": 0, "not_found": 0, "errors": []}
        
        try:
            # Create delta for this operation
            delta = Delta()
            
            # Process node UUIDs in batches
            for i in range(0, len(node_uuids), self.max_batch_size):
                batch_uuids = node_uuids[i:i + self.max_batch_size]
                
                batch_results = await self._process_node_removal_batch(
                    batch_uuids, cleanup_related_edges, delta
                )
                results["removed"] += batch_results["removed"]
                results["not_found"] += batch_results["not_found"]
                results["errors"].extend(batch_results["errors"])
            
            # Apply the delta
            await self.apply_delta(delta)
            
            # Update statistics
            self._update_statistics(len(node_uuids), datetime.now() - start_time)
            
            logger.info(f"Removed {results['removed']} entities incrementally")
            
        except Exception as e:
            logger.error(f"Error removing entities incrementally: {e}")
            results["errors"].append(str(e))
            self.update_stats["failed_updates"] += 1
        
        return results
    
    # Edge Operations
    async def add_edges_incremental(self, edges: List[EntityEdge], 
                                   group_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Add new relationships to the knowledge graph without full rebuild.
        
        Args:
            edges: List of EntityEdge objects to add
            group_ids: Optional list of group IDs to filter by
            
        Returns:
            Dictionary containing update statistics and results
        """
        start_time = datetime.now()
        results = {"added": 0, "skipped": 0, "errors": []}
        
        try:
            # Create delta for this operation
            delta = Delta()
            
            # Process edges in batches
            for i in range(0, len(edges), self.max_batch_size):
                batch = edges[i:i + self.max_batch_size]
                
                batch_results = await self._process_edge_batch(batch, "add", delta)
                results["added"] += batch_results["added"]
                results["skipped"] += batch_results["skipped"]
                results["errors"].extend(batch_results["errors"])
            
            # Apply the delta
            await self.apply_delta(delta)
            
            # Update statistics
            self._update_statistics(len(edges), datetime.now() - start_time)
            
            logger.info(f"Added {results['added']} edges incrementally")
            
        except Exception as e:
            logger.error(f"Error adding edges incrementally: {e}")
            results["errors"].append(str(e))
            self.update_stats["failed_updates"] += 1
        
        return results
    
    async def update_edges_incremental(self, edges: List[EntityEdge], 
                                     group_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Update existing relationships in the knowledge graph without full rebuild.
        
        Args:
            edges: List of EntityEdge objects to update
            group_ids: Optional list of group IDs to filter by
            
        Returns:
            Dictionary containing update statistics and results
        """
        start_time = datetime.now()
        results = {"updated": 0, "not_found": 0, "errors": []}
        
        try:
            # Create delta for this operation
            delta = Delta()
            
            # Process edges in batches
            for i in range(0, len(edges), self.max_batch_size):
                batch = edges[i:i + self.max_batch_size]
                
                batch_results = await self._process_edge_batch(batch, "update", delta)
                results["updated"] += batch_results["updated"]
                results["not_found"] += batch_results["not_found"]
                results["errors"].extend(batch_results["errors"])
            
            # Apply the delta
            await self.apply_delta(delta)
            
            # Update statistics
            self._update_statistics(len(edges), datetime.now() - start_time)
            
            logger.info(f"Updated {results['updated']} edges incrementally")
            
        except Exception as e:
            logger.error(f"Error updating edges incrementally: {e}")
            results["errors"].append(str(e))
            self.update_stats["failed_updates"] += 1
        
        return results
    
    async def remove_edges_incremental(self, edge_uuids: List[str]) -> Dict[str, Any]:
        """
        Remove relationships from the knowledge graph without full rebuild.
        
        Args:
            edge_uuids: List of edge UUIDs to remove
            
        Returns:
            Dictionary containing update statistics and results
        """
        start_time = datetime.now()
        results = {"removed": 0, "not_found": 0, "errors": []}
        
        try:
            # Create delta for this operation
            delta = Delta()
            
            # Process edge UUIDs in batches
            for i in range(0, len(edge_uuids), self.max_batch_size):
                batch_uuids = edge_uuids[i:i + self.max_batch_size]
                
                batch_results = await self._process_edge_removal_batch(batch_uuids, delta)
                results["removed"] += batch_results["removed"]
                results["not_found"] += batch_results["not_found"]
                results["errors"].extend(batch_results["errors"])
            
            # Apply the delta
            await self.apply_delta(delta)
            
            # Update statistics
            self._update_statistics(len(edge_uuids), datetime.now() - start_time)
            
            logger.info(f"Removed {results['removed']} edges incrementally")
            
        except Exception as e:
            logger.error(f"Error removing edges incrementally: {e}")
            results["errors"].append(str(e))
            self.update_stats["failed_updates"] += 1
        
        return results
    
    # Delta Management
    def create_delta(self, operations: Optional[List[DeltaOperation]] = None) -> Delta:
        """
        Create a new delta with optional operations.
        
        Args:
            operations: Optional list of operations to include
            
        Returns:
            New Delta instance
        """
        delta = Delta()
        if operations:
            delta.operations = operations
        return delta
    
    async def apply_delta(self, delta: Delta, validate: bool = True) -> Dict[str, Any]:
        """
        Apply a delta to the knowledge graph.
        
        Args:
            delta: Delta to apply
            validate: Whether to validate the delta before applying
            
        Returns:
            Dictionary containing application results
        """
        start_time = datetime.now()
        results = {"applied": 0, "failed": 0, "errors": []}
        
        try:
            # Validate delta if requested
            if validate:
                validation_result = self.validate_delta(delta)
                if not validation_result["valid"]:
                    raise ValueError(f"Delta validation failed: {validation_result['errors']}")
            
            # Store rollback data
            rollback_data = self._create_rollback_data(delta)
            
            # Apply operations
            for operation in delta.operations:
                try:
                    await self._apply_operation(operation)
                    results["applied"] += 1
                except Exception as e:
                    logger.error(f"Failed to apply operation {operation.uuid}: {e}")
                    results["failed"] += 1
                    results["errors"].append(str(e))
                    
                    # Attempt rollback for this operation
                    try:
                        await self._rollback_operation(operation, rollback_data)
                    except Exception as rollback_error:
                        logger.error(f"Rollback failed for operation {operation.uuid}: {rollback_error}")
            
            # Update delta status
            delta.applied_at = datetime.now()
            delta.status = "applied" if results["failed"] == 0 else "partially_applied"
            delta.rollback_data = rollback_data
            
            # Store in pending deltas for potential rollback
            delta_id = str(uuid.uuid4())
            self.pending_deltas[delta_id] = delta
            
            # Update statistics
            self._update_statistics(len(delta.operations), datetime.now() - start_time)
            
            logger.info(f"Applied delta {delta_id}: {results['applied']} successful, {results['failed']} failed")
            
        except Exception as e:
            logger.error(f"Error applying delta: {e}")
            results["errors"].append(str(e))
            delta.status = "failed"
            self.update_stats["failed_updates"] += 1
        
        return results
    
    def validate_delta(self, delta: Delta) -> Dict[str, Any]:
        """
        Validate a delta before application.
        
        Args:
            delta: Delta to validate
            
        Returns:
            Dictionary containing validation results
        """
        results = {"valid": True, "errors": []}
        
        # Check for empty delta
        if delta.is_empty():
            results["valid"] = False
            results["errors"].append("Delta is empty")
            return results
        
        # Validate operations
        for i, operation in enumerate(delta.operations):
            try:
                # Check operation type
                if not isinstance(operation.operation_type, DeltaOperationType):
                    results["valid"] = False
                    results["errors"].append(f"Operation {i}: Invalid operation type")
                    continue
                
                # Check entity type
                if not isinstance(operation.entity_type, DeltaEntityType):
                    results["valid"] = False
                    results["errors"].append(f"Operation {i}: Invalid entity type")
                    continue
                
                # Check UUID format
                if not operation.uuid or len(operation.uuid) != 36:
                    results["valid"] = False
                    results["errors"].append(f"Operation {i}: Invalid UUID format")
                    continue
                
                # Validate based on operation type
                if operation.operation_type == DeltaOperationType.REMOVE:
                    # Check if entity exists for remove operations
                    exists = self._check_entity_exists_sync(operation)
                    if not exists:
                        results["valid"] = False
                        results["errors"].append(f"Operation {i}: Entity {operation.uuid} does not exist")
                
                elif operation.operation_type == DeltaOperationType.UPDATE:
                    # Check if entity exists for update operations
                    exists = self._check_entity_exists_sync(operation)
                    if not exists:
                        results["valid"] = False
                        results["errors"].append(f"Operation {i}: Entity {operation.uuid} does not exist")
                
            except Exception as e:
                results["valid"] = False
                results["errors"].append(f"Operation {i}: Validation error - {str(e)}")
        
        return results
    
    async def rollback_delta(self, delta_id: str) -> Dict[str, Any]:
        """
        Rollback a previously applied delta.
        
        Args:
            delta_id: ID of the delta to rollback
            
        Returns:
            Dictionary containing rollback results
        """
        results = {"rolled_back": 0, "failed": 0, "errors": []}
        
        try:
            if delta_id not in self.pending_deltas:
                raise ValueError(f"Delta {delta_id} not found")
            
            delta = self.pending_deltas[delta_id]
            
            if delta.rollback_data is None:
                raise ValueError(f"No rollback data available for delta {delta_id}")
            
            # Apply reverse operations
            for operation in reversed(delta.operations):
                try:
                    await self._rollback_operation(operation, delta.rollback_data)
                    results["rolled_back"] += 1
                except Exception as e:
                    logger.error(f"Failed to rollback operation {operation.uuid}: {e}")
                    results["failed"] += 1
                    results["errors"].append(str(e))
            
            # Update delta status
            delta.status = "rolled_back"
            
            logger.info(f"Rolled back delta {delta_id}: {results['rolled_back']} successful, {results['failed']} failed")
            
        except Exception as e:
            logger.error(f"Error rolling back delta {delta_id}: {e}")
            results["errors"].append(str(e))
        
        return results
    
    def merge_delta(self, delta1: Delta, delta2: Delta) -> Delta:
        """
        Merge two deltas into a single delta.
        
        Args:
            delta1: First delta
            delta2: Second delta
            
        Returns:
            Merged delta
        """
        merged_delta = Delta()
        merged_delta.operations = delta1.operations + delta2.operations
        merged_delta.created_at = min(delta1.created_at, delta2.created_at)
        
        return merged_delta
    
    # Index Updates
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
        results = {"updated": 0, "errors": []}
        
        try:
            if not self.driver.vector_search_engine:
                raise ValueError("Vector search engine not enabled")
            
            # Validate inputs
            if len(embeddings) != len(uuids):
                raise ValueError("Number of embeddings must match number of UUIDs")
            
            if len(embeddings) == 0:
                return results
            
            # Convert to numpy array
            embedding_matrix = np.array(embeddings).astype('float32')
            
            # Update indices
            if entity_type == 'node':
                if self.driver.vector_search_engine.node_index:
                    self.driver.vector_search_engine.node_index = self.driver.vector_search_engine.add_embeddings(
                        embedding_matrix, uuids, self.driver.vector_search_engine.node_index,
                        self.driver.vector_search_engine.node_index_metadata
                    )
                    self.driver.vector_search_engine.node_id_map.extend(uuids)
                    results["updated"] = len(uuids)
            
            elif entity_type == 'edge':
                if self.driver.vector_search_engine.edge_index:
                    self.driver.vector_search_engine.edge_index = self.driver.vector_search_engine.add_embeddings(
                        embedding_matrix, uuids, self.driver.vector_search_engine.edge_index,
                        self.driver.vector_search_engine.edge_index_metadata
                    )
                    self.driver.vector_search_engine.edge_id_map.extend(uuids)
                    results["updated"] = len(uuids)
            
            elif entity_type == 'community':
                if self.driver.vector_search_engine.community_index:
                    self.driver.vector_search_engine.community_index = self.driver.vector_search_engine.add_embeddings(
                        embedding_matrix, uuids, self.driver.vector_search_engine.community_index,
                        self.driver.vector_search_engine.community_index_metadata
                    )
                    self.driver.vector_search_engine.community_id_map.extend(uuids)
                    results["updated"] = len(uuids)
            
            else:
                raise ValueError(f"Unknown entity type: {entity_type}")
            
            logger.info(f"Updated {results['updated']} {entity_type} vector indices incrementally")
            
        except Exception as e:
            logger.error(f"Error updating vector indices incrementally: {e}")
            results["errors"].append(str(e))
        
        return results
    
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
        results = {"updated": 0, "errors": []}
        
        try:
            if not self.driver.hybrid_search_engine:
                raise ValueError("Hybrid search engine not enabled")
            
            # Validate inputs
            if len(texts) != len(uuids):
                raise ValueError("Number of texts must match number of UUIDs")
            
            if len(texts) == 0:
                return results
            
            # Update text indices
            if hasattr(self.driver.hybrid_search_engine, '_build_text_indices'):
                self.driver.hybrid_search_engine._build_text_indices()
                results["updated"] = len(uuids)
            
            logger.info(f"Updated {results['updated']} {entity_type} text indices incrementally")
            
        except Exception as e:
            logger.error(f"Error updating text indices incrementally: {e}")
            results["errors"].append(str(e))
        
        return results
    
    async def update_graph_indices_incremental(self, operations: List[DeltaOperation]) -> Dict[str, Any]:
        """
        Update graph structure indices incrementally.
        
        Args:
            operations: List of operations that affect graph structure
            
        Returns:
            Dictionary containing update results
        """
        results = {"updated": 0, "errors": []}
        
        try:
            # Extract graph structure changes
            add_edges = []
            remove_edges = []
            
            for op in operations:
                if op.entity_type == DeltaEntityType.EDGE:
                    if op.operation_type == DeltaOperationType.ADD:
                        add_edges.append(op)
                    elif op.operation_type == DeltaOperationType.REMOVE:
                        remove_edges.append(op)
            
            # Update adjacency lists or other graph structures
            if add_edges:
                await self._update_adjacency_lists_add(add_edges)
                results["updated"] += len(add_edges)
            
            if remove_edges:
                await self._update_adjacency_lists_remove(remove_edges)
                results["updated"] += len(remove_edges)
            
            logger.info(f"Updated graph indices for {results['updated']} operations")
            
        except Exception as e:
            logger.error(f"Error updating graph indices incrementally: {e}")
            results["errors"].append(str(e))
        
        return results
    
    async def update_temporal_indices_incremental(self, operations: List[DeltaOperation]) -> Dict[str, Any]:
        """
        Update temporal indices incrementally.
        
        Args:
            operations: List of operations with temporal data
            
        Returns:
            Dictionary containing update results
        """
        results = {"updated": 0, "errors": []}
        
        try:
            # Extract temporal operations
            temporal_ops = [op for op in operations if 'created_at' in op.data or 'valid_at' in op.data]
            
            if temporal_ops:
                # Update temporal indices
                await self._update_temporal_indices(temporal_ops)
                results["updated"] = len(temporal_ops)
            
            logger.info(f"Updated temporal indices for {results['updated']} operations")
            
        except Exception as e:
            logger.error(f"Error updating temporal indices incrementally: {e}")
            results["errors"].append(str(e))
        
        return results
    
    async def rebuild_indices_if_needed(self, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Rebuild indices if performance degradation is detected.
        
        Args:
            threshold: Performance threshold (0.0-1.0)
            
        Returns:
            Dictionary containing rebuild results
        """
        results = {"rebuild_needed": False, "rebuilt": [], "errors": []}
        
        try:
            # Check if rebuild is needed based on various metrics
            if await self._should_rebuild_indices(threshold):
                results["rebuild_needed"] = True
                
                # Rebuild different types of indices
                if self.driver.vector_search_engine:
                    self.driver._build_vector_indices()
                    results["rebuilt"].append("vector")
                
                if self.driver.hybrid_search_engine:
                    self.driver.rebuild_hybrid_search_indices()
                    results["rebuilt"].append("hybrid")
                
                if self.driver.traversal_engine:
                    self.driver.clear_traversal_cache()
                    results["rebuilt"].append("traversal")
                
                logger.info(f"Rebuilt indices: {', '.join(results['rebuilt'])}")
            
        except Exception as e:
            logger.error(f"Error rebuilding indices: {e}")
            results["errors"].append(str(e))
        
        return results
    
    # Batch Processing
    async def batch_incremental_update(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply multiple incremental updates in a single batch operation.
        
        Args:
            updates: List of update dictionaries, each containing:
                - 'type': 'add_entities', 'update_entities', 'remove_entities', 
                         'add_edges', 'update_edges', 'remove_edges'
                - 'data': The data for the operation
                - 'group_ids': Optional group IDs filter
                
        Returns:
            Dictionary containing batch results
        """
        start_time = datetime.now()
        results = {"total": len(updates), "successful": 0, "failed": 0, "details": []}
        
        try:
            # Create a single delta for all operations
            delta = Delta()
            
            # Process each update
            for update in updates:
                try:
                    update_result = await self._process_batch_update(update, delta)
                    results["details"].append(update_result)
                    
                    if update_result["success"]:
                        results["successful"] += 1
                    else:
                        results["failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing batch update: {e}")
                    results["failed"] += 1
                    results["details"].append({
                        "type": update.get("type", "unknown"),
                        "success": False,
                        "error": str(e)
                    })
            
            # Apply the combined delta
            if not delta.is_empty():
                await self.apply_delta(delta)
            
            # Update statistics
            self._update_statistics(len(updates), datetime.now() - start_time)
            
            logger.info(f"Batch update completed: {results['successful']}/{results['total']} successful")
            
        except Exception as e:
            logger.error(f"Error in batch incremental update: {e}")
            results["failed"] = len(updates) - results["successful"]
            results["details"].append({
                "type": "batch",
                "success": False,
                "error": str(e)
            })
        
        return results
    
    async def process_large_delta(self, delta: Delta, chunk_size: int = 100) -> Dict[str, Any]:
        """
        Process a large delta by breaking it into smaller chunks.
        
        Args:
            delta: Delta to process
            chunk_size: Size of each chunk
            
        Returns:
            Dictionary containing processing results
        """
        results = {"total_chunks": 0, "successful_chunks": 0, "failed_chunks": 0, "errors": []}
        
        try:
            # Break delta into chunks
            chunks = []
            for i in range(0, len(delta.operations), chunk_size):
                chunk_delta = Delta(operations=delta.operations[i:i + chunk_size])
                chunks.append(chunk_delta)
            
            results["total_chunks"] = len(chunks)
            
            # Process each chunk
            for chunk in chunks:
                try:
                    chunk_result = await self.apply_delta(chunk, validate=False)
                    
                    if chunk_result["applied"] > 0:
                        results["successful_chunks"] += 1
                    else:
                        results["failed_chunks"] += 1
                    
                    results["errors"].extend(chunk_result["errors"])
                    
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    results["failed_chunks"] += 1
                    results["errors"].append(str(e))
            
            logger.info(f"Processed large delta: {results['successful_chunks']}/{results['total_chunks']} chunks successful")
            
        except Exception as e:
            logger.error(f"Error processing large delta: {e}")
            results["errors"].append(str(e))
        
        return results
    
    async def parallel_delta_application(self, deltas: List[Delta], max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Apply multiple deltas in parallel.
        
        Args:
            deltas: List of deltas to apply
            max_concurrent: Maximum number of concurrent applications
            
        Returns:
            Dictionary containing parallel application results
        """
        results = {"total": len(deltas), "successful": 0, "failed": 0, "errors": []}
        
        try:
            # Process deltas in batches to limit concurrency
            for i in range(0, len(deltas), max_concurrent):
                batch_deltas = deltas[i:i + max_concurrent]
                
                # Apply deltas in parallel
                tasks = [self.apply_delta(delta) for delta in batch_deltas]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        results["failed"] += 1
                        results["errors"].append(f"Delta {i+j}: {str(result)}")
                    elif isinstance(result, dict) and result.get("applied", 0) > 0:
                        results["successful"] += 1
                    else:
                        results["failed"] += 1
                        if isinstance(result, dict) and "errors" in result:
                            results["errors"].extend(result["errors"])
            
            logger.info(f"Parallel delta application: {results['successful']}/{results['total']} successful")
            
        except Exception as e:
            logger.error(f"Error in parallel delta application: {e}")
            results["failed"] = len(deltas) - results["successful"]
            results["errors"].append(str(e))
        
        return results
    
    async def monitor_delta_progress(self, delta_id: str, poll_interval: float = 1.0) -> Dict[str, Any]:
        """
        Monitor the progress of a delta application.
        
        Args:
            delta_id: ID of the delta to monitor
            poll_interval: Interval in seconds to check progress
            
        Returns:
            Dictionary containing progress information
        """
        progress = {"delta_id": delta_id, "status": "not_found", "progress": 0.0}
        
        try:
            if delta_id not in self.pending_deltas:
                return progress
            
            delta = self.pending_deltas[delta_id]
            
            if delta.status == "pending":
                progress["status"] = "pending"
                progress["progress"] = 0.0
                progress["total_operations"] = len(delta.operations)
                progress["completed_operations"] = 0
            
            elif delta.status == "applied":
                progress["status"] = "completed"
                progress["progress"] = 1.0
                progress["total_operations"] = len(delta.operations)
                progress["completed_operations"] = len(delta.operations)
                progress["completed_at"] = delta.applied_at
            
            elif delta.status == "failed":
                progress["status"] = "failed"
                progress["progress"] = 0.0
                progress["error"] = "Delta application failed"
            
            elif delta.status == "rolled_back":
                progress["status"] = "rolled_back"
                progress["progress"] = 1.0
                progress["rolled_back_at"] = datetime.now()
            
            logger.info(f"Delta {delta_id} progress: {progress['status']} ({progress['progress']:.1%})")
            
        except Exception as e:
            logger.error(f"Error monitoring delta progress: {e}")
            progress["status"] = "error"
            progress["error"] = str(e)
        
        return progress
    
    # Utility Methods
    def get_update_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about incremental updates.
        
        Returns:
            Dictionary containing update statistics
        """
        return self.update_stats.copy()
    
    def clear_pending_deltas(self):
        """Clear all pending deltas"""
        self.pending_deltas.clear()
        logger.info("Cleared all pending deltas")
    
    async def _process_node_batch(self, nodes: List[EntityNode], operation: str, delta: Delta) -> Dict[str, Any]:
        """Process a batch of nodes for add/update operations"""
        results = {"added": 0, "updated": 0, "skipped": 0, "not_found": 0, "errors": []}
        
        for node in nodes:
            try:
                operation_obj = DeltaOperation(
                    operation_type=DeltaOperationType.ADD if operation == "add" else DeltaOperationType.UPDATE,
                    entity_type=DeltaEntityType.NODE,
                    uuid=node.uuid,
                    data=node.dict()
                )
                
                if operation == "add":
                    # Check if node already exists
                    existing = self.driver.nodes_df[self.driver.nodes_df['uuid'] == node.uuid]
                    if not existing.empty:
                        results["skipped"] += 1
                        continue
                    
                    await self.driver.save_node(node)
                    results["added"] += 1
                    delta.add_operation(operation_obj)
                
                elif operation == "update":
                    # Check if node exists
                    existing = self.driver.nodes_df[self.driver.nodes_df['uuid'] == node.uuid]
                    if existing.empty:
                        results["not_found"] += 1
                        continue
                    
                    await self.driver.save_node(node)
                    results["updated"] += 1
                    delta.add_operation(operation_obj)
                
            except Exception as e:
                logger.error(f"Error processing node {node.uuid}: {e}")
                results["errors"].append(f"Node {node.uuid}: {str(e)}")
        
        return results
    
    async def _process_edge_batch(self, edges: List[EntityEdge], operation: str, delta: Delta) -> Dict[str, Any]:
        """Process a batch of edges for add/update operations"""
        results = {"added": 0, "updated": 0, "skipped": 0, "not_found": 0, "errors": []}
        
        for edge in edges:
            try:
                operation_obj = DeltaOperation(
                    operation_type=DeltaOperationType.ADD if operation == "add" else DeltaOperationType.UPDATE,
                    entity_type=DeltaEntityType.EDGE,
                    uuid=edge.uuid,
                    data=edge.dict()
                )
                
                if operation == "add":
                    # Check if edge already exists
                    existing = self.driver.edges_df[self.driver.edges_df['uuid'] == edge.uuid]
                    if not existing.empty:
                        results["skipped"] += 1
                        continue
                    
                    await self.driver.save_edge(edge)
                    results["added"] += 1
                    delta.add_operation(operation_obj)
                
                elif operation == "update":
                    # Check if edge exists
                    existing = self.driver.edges_df[self.driver.edges_df['uuid'] == edge.uuid]
                    if existing.empty:
                        results["not_found"] += 1
                        continue
                    
                    await self.driver.save_edge(edge)
                    results["updated"] += 1
                    delta.add_operation(operation_obj)
                
            except Exception as e:
                logger.error(f"Error processing edge {edge.uuid}: {e}")
                results["errors"].append(f"Edge {edge.uuid}: {str(e)}")
        
        return results
    
    async def _process_node_removal_batch(self, node_uuids: List[str], 
                                        cleanup_related_edges: bool, delta: Delta) -> Dict[str, Any]:
        """Process a batch of node removals"""
        results = {"removed": 0, "not_found": 0, "errors": []}
        
        for node_uuid in node_uuids:
            try:
                # Check if node exists
                existing = self.driver.nodes_df[self.driver.nodes_df['uuid'] == node_uuid]
                if existing.empty:
                    results["not_found"] += 1
                    continue
                
                # Remove related edges if requested
                if cleanup_related_edges:
                    related_edges = self.driver.edges_df[
                        (self.driver.edges_df['source_uuid'] == node_uuid) |
                        (self.driver.edges_df['target_uuid'] == node_uuid)
                    ]
                    
                    for _, edge_row in related_edges.iterrows():
                        edge_uuid = edge_row['uuid']
                        await self._remove_edge_by_uuid(edge_uuid)
                
                # Remove the node
                self.driver.nodes_df = self.driver.nodes_df[self.driver.nodes_df['uuid'] != node_uuid]
                
                # Add to delta
                operation_obj = DeltaOperation(
                    operation_type=DeltaOperationType.REMOVE,
                    entity_type=DeltaEntityType.NODE,
                    uuid=node_uuid,
                    data={}
                )
                delta.add_operation(operation_obj)
                
                results["removed"] += 1
                
            except Exception as e:
                logger.error(f"Error removing node {node_uuid}: {e}")
                results["errors"].append(f"Node {node_uuid}: {str(e)}")
        
        return results
    
    async def _process_edge_removal_batch(self, edge_uuids: List[str], delta: Delta) -> Dict[str, Any]:
        """Process a batch of edge removals"""
        results = {"removed": 0, "not_found": 0, "errors": []}
        
        for edge_uuid in edge_uuids:
            try:
                result = await self._remove_edge_by_uuid(edge_uuid)
                if result["removed"]:
                    results["removed"] += 1
                    
                    # Add to delta
                    operation_obj = DeltaOperation(
                        operation_type=DeltaOperationType.REMOVE,
                        entity_type=DeltaEntityType.EDGE,
                        uuid=edge_uuid,
                        data={}
                    )
                    delta.add_operation(operation_obj)
                else:
                    results["not_found"] += 1
                    
            except Exception as e:
                logger.error(f"Error removing edge {edge_uuid}: {e}")
                results["errors"].append(f"Edge {edge_uuid}: {str(e)}")
        
        return results
    
    async def _remove_edge_by_uuid(self, edge_uuid: str) -> Dict[str, Any]:
        """Remove an edge by UUID"""
        results = {"removed": False, "error": None}
        
        try:
            # Check if edge exists
            existing = self.driver.edges_df[self.driver.edges_df['uuid'] == edge_uuid]
            if existing.empty:
                return results
            
            # Remove the edge
            self.driver.edges_df = self.driver.edges_df[self.driver.edges_df['uuid'] != edge_uuid]
            
            # Push to hub
            self.driver._push_to_hub(f"Removed edge {edge_uuid}")
            
            results["removed"] = True
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Error removing edge {edge_uuid}: {e}")
        
        return results
    
    async def _apply_operation(self, operation: DeltaOperation):
        """Apply a single delta operation"""
        if operation.entity_type == DeltaEntityType.NODE:
            if operation.operation_type == DeltaOperationType.ADD:
                node_data = operation.data
                node = EntityNode(**node_data)
                await self.driver.save_node(node)
            
            elif operation.operation_type == DeltaOperationType.UPDATE:
                node_data = operation.data
                node = EntityNode(**node_data)
                await self.driver.save_node(node)
            
            elif operation.operation_type == DeltaOperationType.REMOVE:
                self.driver.nodes_df = self.driver.nodes_df[self.driver.nodes_df['uuid'] != operation.uuid]
        
        elif operation.entity_type == DeltaEntityType.EDGE:
            if operation.operation_type == DeltaOperationType.ADD:
                edge_data = operation.data
                edge = EntityEdge(**edge_data)
                await self.driver.save_edge(edge)
            
            elif operation.operation_type == DeltaOperationType.UPDATE:
                edge_data = operation.data
                edge = EntityEdge(**edge_data)
                await self.driver.save_edge(edge)
            
            elif operation.operation_type == DeltaOperationType.REMOVE:
                self.driver.edges_df = self.driver.edges_df[self.driver.edges_df['uuid'] != operation.uuid]
        
        # Push changes to hub
        self.driver._push_to_hub(f"Applied delta operation: {operation.operation_type.value} {operation.entity_type.value}")
    
    async def _rollback_operation(self, operation: DeltaOperation, rollback_data: Dict[str, Any]):
        """Rollback a single delta operation"""
        if operation.operation_type == DeltaOperationType.ADD:
            # Remove the added entity
            if operation.entity_type == DeltaEntityType.NODE:
                self.driver.nodes_df = self.driver.nodes_df[self.driver.nodes_df['uuid'] != operation.uuid]
            elif operation.entity_type == DeltaEntityType.EDGE:
                self.driver.edges_df = self.driver.edges_df[self.driver.edges_df['uuid'] != operation.uuid]
        
        elif operation.operation_type == DeltaOperationType.UPDATE:
            # Restore the original entity
            original_data = rollback_data.get(operation.uuid)
            if original_data:
                if operation.entity_type == DeltaEntityType.NODE:
                    # Restore original node
                    self.driver.nodes_df = self.driver.nodes_df[self.driver.nodes_df['uuid'] != operation.uuid]
                    original_node = EntityNode(**original_data)
                    await self.driver.save_node(original_node)
                elif operation.entity_type == DeltaEntityType.EDGE:
                    # Restore original edge
                    self.driver.edges_df = self.driver.edges_df[self.driver.edges_df['uuid'] != operation.uuid]
                    original_edge = EntityEdge(**original_data)
                    await self.driver.save_edge(original_edge)
        
        elif operation.operation_type == DeltaOperationType.REMOVE:
            # Restore the removed entity
            original_data = rollback_data.get(operation.uuid)
            if original_data:
                if operation.entity_type == DeltaEntityType.NODE:
                    original_node = EntityNode(**original_data)
                    await self.driver.save_node(original_node)
                elif operation.entity_type == DeltaEntityType.EDGE:
                    original_edge = EntityEdge(**original_data)
                    await self.driver.save_edge(original_edge)
        
        # Push changes to hub
        self.driver._push_to_hub(f"Rollback delta operation: {operation.operation_type.value} {operation.entity_type.value}")
    
    def _create_rollback_data(self, delta: Delta) -> Dict[str, Any]:
        """Create rollback data for a delta"""
        rollback_data = {}
        
        for operation in delta.operations:
            if operation.operation_type == DeltaOperationType.UPDATE:
                # Store original data for update operations
                if operation.entity_type == DeltaEntityType.NODE:
                    existing = self.driver.nodes_df[self.driver.nodes_df['uuid'] == operation.uuid]
                    if not existing.empty:
                        rollback_data[operation.uuid] = existing.iloc[0].to_dict()
                
                elif operation.entity_type == DeltaEntityType.EDGE:
                    existing = self.driver.edges_df[self.driver.edges_df['uuid'] == operation.uuid]
                    if not existing.empty:
                        rollback_data[operation.uuid] = existing.iloc[0].to_dict()
        
        return rollback_data
    
    async def _check_entity_exists(self, operation: DeltaOperation) -> bool:
        """Check if an entity exists"""
        if operation.entity_type == DeltaEntityType.NODE:
            existing = self.driver.nodes_df[self.driver.nodes_df['uuid'] == operation.uuid]
            return not existing.empty
        
        elif operation.entity_type == DeltaEntityType.EDGE:
            existing = self.driver.edges_df[self.driver.edges_df['uuid'] == operation.uuid]
            return not existing.empty
        
        return False
    
    async def _update_adjacency_lists_add(self, operations: List[DeltaOperation]):
        """Update adjacency lists for added edges"""
        # This would update any adjacency list structures used for graph traversal
        # Implementation depends on specific graph representation
        pass
    
    async def _update_adjacency_lists_remove(self, operations: List[DeltaOperation]):
        """Update adjacency lists for removed edges"""
        # This would update any adjacency list structures used for graph traversal
        # Implementation depends on specific graph representation
        pass
    
    async def _update_temporal_indices(self, operations: List[DeltaOperation]):
        """Update temporal indices"""
        # This would update temporal indexing structures
        # Implementation depends on specific temporal indexing approach
        pass
    
    async def _should_rebuild_indices(self, threshold: float) -> bool:
        """Check if indices should be rebuilt based on performance metrics"""
        # Simple heuristic: rebuild if data size has changed significantly
        current_edge_count = len(self.driver.edges_df)
        current_node_count = len(self.driver.nodes_df)
        
        # Trigger rebuild if significant changes detected
        return current_edge_count > 10000 or current_node_count > 5000
    
    async def _process_batch_update(self, update: Dict[str, Any], delta: Delta) -> Dict[str, Any]:
        """Process a single batch update"""
        update_type = update.get("type")
        data = update.get("data", {})
        group_ids = update.get("group_ids")
        
        result = {"type": update_type, "success": False, "error": None}
        
        try:
            if update_type == "add_entities":
                nodes = [EntityNode(**node_data) for node_data in data]
                batch_result = await self._process_node_batch(nodes, "add", delta)
                result["success"] = batch_result["added"] > 0
                result["details"] = batch_result
            
            elif update_type == "update_entities":
                nodes = [EntityNode(**node_data) for node_data in data]
                batch_result = await self._process_node_batch(nodes, "update", delta)
                result["success"] = batch_result["updated"] > 0
                result["details"] = batch_result
            
            elif update_type == "remove_entities":
                batch_result = await self._process_node_removal_batch(data, True, delta)
                result["success"] = batch_result["removed"] > 0
                result["details"] = batch_result
            
            elif update_type == "add_edges":
                edges = [EntityEdge(**edge_data) for edge_data in data]
                batch_result = await self._process_edge_batch(edges, "add", delta)
                result["success"] = batch_result["added"] > 0
                result["details"] = batch_result
            
            elif update_type == "update_edges":
                edges = [EntityEdge(**edge_data) for edge_data in data]
                batch_result = await self._process_edge_batch(edges, "update", delta)
                result["success"] = batch_result["updated"] > 0
                result["details"] = batch_result
            
            elif update_type == "remove_edges":
                batch_result = await self._process_edge_removal_batch(data, delta)
                result["success"] = batch_result["removed"] > 0
                result["details"] = batch_result
            
            else:
                result["error"] = f"Unknown update type: {update_type}"
        
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error processing batch update {update_type}: {e}")
        
        return result
    
    def _update_statistics(self, item_count: int, duration: timedelta):
        """Update performance statistics"""
        self.update_stats["total_updates"] += item_count
        self.update_stats["successful_updates"] += item_count
        self.update_stats["last_update_time"] = datetime.now()
        
        # Calculate average update time
        if self.update_stats["total_updates"] > 0:
            total_time = self.update_stats.get("total_time", duration)
            self.update_stats["total_time"] = total_time + duration
            self.update_stats["average_update_time"] = self.update_stats["total_time"].total_seconds() / self.update_stats["total_updates"]
    
    def _check_entity_exists_sync(self, operation: DeltaOperation) -> bool:
        """Synchronous version of entity existence check"""
        if operation.entity_type == DeltaEntityType.NODE:
            existing = self.driver.nodes_df[self.driver.nodes_df['uuid'] == operation.uuid]
            return not existing.empty
        
        elif operation.entity_type == DeltaEntityType.EDGE:
            existing = self.driver.edges_df[self.driver.edges_df['uuid'] == operation.uuid]
            return not existing.empty
        
        return False