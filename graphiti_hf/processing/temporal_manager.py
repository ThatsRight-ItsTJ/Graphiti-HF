
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
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from collections import defaultdict

logger = logging.getLogger(__name__)


class TemporalConflictType(Enum):
    """Types of temporal conflicts that can occur"""
    OVERLAPPING_VALIDITY = "overlapping_validity"
    CONTRADICTING_FACTS = "contradicting_facts"
    MISSING_VALIDITY = "missing_validity"
    INVALID_TIMELINE = "invalid_timeline"


class TemporalResolutionStrategy(Enum):
    """Strategies for resolving temporal conflicts"""
    FIRST_WINS = "first_wins"  # Keep the earliest valid record
    LAST_WINS = "last_wins"    # Keep the latest valid record
    MERGE = "merge"           # Merge overlapping records
    INVALIDATE = "invalidate" # Invalidate conflicting records
    MANUAL = "manual"         # Require manual intervention


@dataclass
class TemporalRecord:
    """Base class for temporal records with bi-temporal tracking"""
    uuid: str
    event_occurrence_time: datetime  # When the fact actually happened
    data_ingestion_time: datetime    # When the data was added to the system
    valid_at: datetime               # When the record becomes valid
    invalid_at: Optional[datetime] = None  # When the record becomes invalid
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1


@dataclass
class TemporalConflict:
    """Represents a temporal conflict in the knowledge graph"""
    conflict_id: str
    conflict_type: TemporalConflictType
    affected_entities: List[str]
    conflicting_records: List[TemporalRecord]
    detected_at: datetime
    resolution_strategy: Optional[TemporalResolutionStrategy] = None
    resolved_at: Optional[datetime] = None
    resolution_metadata: Dict[str, Any] = field(default_factory=dict)


class TemporalQueryFilter(BaseModel):
    """Filter for temporal queries"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_occurrence_range: Optional[Tuple[datetime, datetime]] = None
    data_ingestion_range: Optional[Tuple[datetime, datetime]] = None
    valid_at_time: Optional[datetime] = None
    entity_types: Optional[List[str]] = None
    group_ids: Optional[List[str]] = None
    include_invalidated: bool = False
    limit: Optional[int] = None


class TemporalStats(BaseModel):
    """Statistics about temporal data"""
    total_records: int
    valid_records: int
    invalidated_records: int
    conflicts_detected: int
    conflicts_resolved: int
    time_span: Tuple[datetime, datetime]
    records_by_entity_type: Dict[str, int]
    temporal_gaps: List[Dict[str, Any]]
    data_ingestion_rate: float  # records per hour
    event_occurrence_distribution: Dict[str, int]


class TemporalManager:
    """
    Comprehensive temporal data management for Graphiti-HF
    
    Implements bi-temporal data model with event occurrence and ingestion times,
    handles temporal edge invalidation, provides temporal query capabilities,
    and maintains temporal consistency during concurrent updates.
    """
    
    def __init__(self, driver):
        """
        Initialize the TemporalManager
        
        Args:
            driver: HuggingFaceDriver instance
        """
        self.driver = driver
        self.temporal_index = defaultdict(list)  # UUID -> List[TemporalRecord]
        self.conflict_index = defaultdict(list)   # Entity UUID -> List[TemporalConflict]
        self.temporal_stats = TemporalStats(
            total_records=0,
            valid_records=0,
            invalidated_records=0,
            conflicts_detected=0,
            conflicts_resolved=0,
            time_span=(datetime.max, datetime.min),
            records_by_entity_type={},
            temporal_gaps=[],
            data_ingestion_rate=0.0,
            event_occurrence_distribution={}
        )
        self._temporal_indices_built = False
        
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
        try:
            # Get current records for the entity
            current_records = self.temporal_index.get(entity_uuid, [])
            
            # Create new temporal record
            temporal_record = TemporalRecord(
                uuid=entity_uuid,
                event_occurrence_time=valid_from,
                data_ingestion_time=datetime.now(),
                valid_at=valid_from,
                invalid_at=valid_to,
                metadata={"entity_type": entity_type}
            )
            
            # Add to temporal index
            self.temporal_index[entity_uuid].append(temporal_record)
            
            # Update the driver's data with temporal information
            if entity_type == "edge":
                # Update edge in driver
                edge_mask = self.driver.edges_df['uuid'] == entity_uuid
                if not self.driver.edges_df[edge_mask].empty:
                    self.driver.edges_df.loc[edge_mask, 'valid_at'] = valid_from
                    if valid_to:
                        self.driver.edges_df.loc[edge_mask, 'invalidated_at'] = valid_to
            else:
                # Update node in driver
                node_mask = self.driver.nodes_df['uuid'] == entity_uuid
                if not self.driver.nodes_df[node_mask].empty:
                    self.driver.nodes_df.loc[node_mask, 'valid_at'] = valid_from
                    if valid_to:
                        self.driver.nodes_df.loc[node_mask, 'invalidated_at'] = valid_to
            
            # Push changes to hub
            self.driver._push_to_hub(f"Updated validity period for {entity_type} {entity_uuid}")
            
            # Update statistics
            self._update_temporal_stats()
            
            return {
                "success": True,
                "entity_uuid": entity_uuid,
                "valid_from": valid_from.isoformat(),
                "valid_to": valid_to.isoformat() if valid_to else None,
                "records_updated": len(current_records) + 1,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error setting validity period for {entity_uuid}: {e}")
            return {
                "success": False,
                "error": str(e),
                "entity_uuid": entity_uuid
            }
    
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
        if not edge_uuids:
            return {"success": True, "invalidated": 0, "reason": "No edges provided"}
        
        invalidation_time = invalidation_time or datetime.now()
        results = {"success": True, "invalidated": 0, "failed": 0, "details": []}
        
        for edge_uuid in edge_uuids:
            try:
                # Update edge in driver
                edge_mask = self.driver.edges_df['uuid'] == edge_uuid
                if not self.driver.edges_df[edge_mask].empty:
                    self.driver.edges_df.loc[edge_mask, 'invalidated_at'] = invalidation_time
                    
                    # Add invalidation metadata
                    if 'metadata' not in self.driver.edges_df.columns:
                        self.driver.edges_df['metadata'] = '{}'
                    
                    current_metadata = json.loads(self.driver.edges_df.loc[edge_mask, 'metadata'].iloc[0])
                    current_metadata['invalidation_reason'] = invalidation_reason
                    current_metadata['invalidated_at'] = invalidation_time.isoformat()
                    self.driver.edges_df.loc[edge_mask, 'metadata'] = json.dumps(current_metadata)
                    
                    # Update temporal index
                    if edge_uuid in self.temporal_index:
                        for record in self.temporal_index[edge_uuid]:
                            record.invalid_at = invalidation_time
                            record.metadata['invalidation_reason'] = invalidation_reason
                    
                    results["invalidated"] += 1
                    results["details"].append({
                        "edge_uuid": edge_uuid,
                        "status": "success",
                        "invalidated_at": invalidation_time.isoformat()
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "edge_uuid": edge_uuid,
                        "status": "failed",
                        "error": "Edge not found"
                    })
                    
            except Exception as e:
                logger.error(f"Error invalidating edge {edge_uuid}: {e}")
                results["failed"] += 1
                results["details"].append({
                    "edge_uuid": edge_uuid,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Push changes to hub
        self.driver._push_to_hub(f"Invalidated {results['invalidated']} edges due to {invalidation_reason}")
        
        # Update statistics
        self._update_temporal_stats()
        
        return results
    
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
        try:
            # Get temporal records for the entity
            temporal_records = self.temporal_index.get(entity_uuid, [])
            
            # Find records valid at query time
            valid_records = [
                record for record in temporal_records
                if record.valid_at <= query_time and 
                   (record.invalid_at is None or record.invalid_at > query_time)
            ]
            
            if not valid_records:
                return None
            
            # Get the most recent valid record
            latest_record = max(valid_records, key=lambda r: r.valid_at)
            
            # Get entity data from driver
            if entity_type == "edge":
                entity_data = self.driver.edges_df[self.driver.edges_df['uuid'] == entity_uuid]
            else:
                entity_data = self.driver.nodes_df[self.driver.nodes_df['uuid'] == entity_uuid]
            
            if entity_data.empty:
                return None
            
            # Combine temporal and entity data
            result = entity_data.iloc[0].to_dict()
            result.update({
                "temporal_validity": {
                    "valid_from": latest_record.valid_at.isoformat(),
                    "valid_to": latest_record.invalid_at.isoformat() if latest_record.invalid_at else None,
                    "event_occurrence": latest_record.event_occurrence_time.isoformat(),
                    "data_ingestion": latest_record.data_ingestion_time.isoformat()
                },
                "is_valid_at_query_time": True
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting valid data for {entity_uuid} at {query_time}: {e}")
            return None
    
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
        try:
            # Get valid nodes at query time
            valid_nodes = []
            nodes_df = self.driver.nodes_df.copy()
            
            if group_ids:
                nodes_df = nodes_df[nodes_df['group_id'].isin(group_ids)]
            
            for _, node_row in nodes_df.iterrows():
                node_data = await self.get_valid_at(node_row['uuid'], query_time, "node")
                if node_data:
                    valid_nodes.append(node_data)
            
            # Get valid edges at query time
            valid_edges = []
            edges_df = self.driver.edges_df.copy()
            
            if group_ids:
                edges_df = edges_df[edges_df['group_id'].isin(group_ids)]
            
            for _, edge_row in edges_df.iterrows():
                edge_data = await self.get_valid_at(edge_row['uuid'], query_time, "edge")
                if edge_data:
                    valid_edges.append(edge_data)
            
            # Apply limit if specified
            if limit:
                valid_nodes = valid_nodes[:limit]
                valid_edges = valid_edges[:limit]
            
            return {
                "query_time": query_time.isoformat(),
                "valid_nodes": valid_nodes,
                "valid_edges": valid_edges,
                "node_count": len(valid_nodes),
                "edge_count": len(valid_edges),
                "reconstructed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error reconstructing historical state at {query_time}: {e}")
            return {
                "query_time": query_time.isoformat(),
                "error": str(e),
                "valid_nodes": [],
                "valid_edges": [],
                "node_count": 0,
                "edge_count": 0
            }
    
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
        try:
            # Get the appropriate DataFrame
            if entity_type == "edge":
                df = self.driver.edges_df.copy()
            else:
                df = self.driver.nodes_df.copy()
            
            # Apply group filter if specified
            if filter.group_ids:
                df = df[df['group_id'].isin(filter.group_ids)]
            
            # Apply temporal filters
            if filter.valid_at_time:
                df = df[df['valid_at'] <= filter.valid_at_time]
                if not filter.include_invalidated:
                    df = df[df['invalidated_at'] > filter.valid_at_time]
            
            if filter.start_time and filter.end_time:
                df = df[
                    (df['valid_at'] >= filter.start_time) & 
                    (df['valid_at'] <= filter.end_time)
                ]
            
            if filter.event_occurrence_range:
                start, end = filter.event_occurrence_range
                # This assumes event_occurrence_time is stored in a column
                # If not, we'll need to derive it from other temporal fields
                if 'event_occurrence_time' in df.columns:
                    df = df[
                        (df['event_occurrence_time'] >= start) & 
                        (df['event_occurrence_time'] <= end)
                    ]
            
            if filter.data_ingestion_range:
                start, end = filter.data_ingestion_range
                # This assumes data_ingestion_time is stored in a column
                # If not, we'll need to derive it from created_at
                if 'data_ingestion_time' in df.columns:
                    df = df[
                        (df['data_ingestion_time'] >= start) & 
                        (df['data_ingestion_time'] <= end)
                    ]
                else:
                    # Use created_at as proxy for data_ingestion_time
                    df = df[
                        (df['created_at'] >= start) & 
                        (df['created_at'] <= end)
                    ]
            
            # Apply entity type filter if specified
            if filter.entity_types:
                if entity_type == "edge":
                    df = df[df['name'].isin(filter.entity_types)]
                else:
                    df = df[df['labels'].apply(
                        lambda x: any(label in filter.entity_types for label in json.loads(x) if x)
                        if pd.notna(x) else False
                    )]
            
            # Apply limit if specified
            if filter.limit:
                df = df.head(filter.limit)
            
            # Convert to result format
            results = []
            for _, row in df.iterrows():
                result = row.to_dict()
                
                # Add temporal information
                temporal_info = {
                    "valid_at": row.get('valid_at', None),
                    "invalidated_at": row.get('invalidated_at', None),
                    "created_at": row.get('created_at', None),
                    "updated_at": row.get('updated_at', None)
                }
                
                # Estimate event_occurrence_time if not available
                if 'event_occurrence_time' not in row or pd.isna(row['event_occurrence_time']):
                    temporal_info['event_occurrence_time'] = temporal_info['valid_at']
                else:
                    temporal_info['event_occurrence_time'] = row['event_occurrence_time']
                
                # Estimate data_ingestion_time if not available
                if 'data_ingestion_time' not in row or pd.isna(row['data_ingestion_time']):
                    temporal_info['data_ingestion_time'] = temporal_info['created_at']
                else:
                    temporal_info['data_ingestion_time'] = row['data_ingestion_time']
                
                result['temporal_info'] = temporal_info
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing temporal query: {e}")
            return []
    
    # Bi-temporal Data Model Methods
    
    async def temporal_edge_invalidation(
        self, 
        conflicting_edges: List[str], 
        resolution_strategy: TemporalResolutionStrategy = TemporalResolutionStrategy.INVALIDATE
    ) -> Dict[str, Any]:
        """
        Handle contradictions through temporal edge invalidation
        
        Args:
            conflicting_edges: List of edge UUIDs that conflict
            resolution_strategy: Strategy for resolving conflicts
            
        Returns:
            Dictionary with resolution results
        """
        try:
            results = {
                "strategy": resolution_strategy.value,
                "processed": 0,
                "resolved": 0,
                "failed": 0,
                "conflicts": []
            }
            
            for edge_uuid in conflicting_edges:
                try:
                    # Get temporal records for the edge
                    temporal_records = self.temporal_index.get(edge_uuid, [])
                    
                    if len(temporal_records) <= 1:
                        # No conflict to resolve
                        continue
                    
                    # Sort by validity period
                    sorted_records = sorted(temporal_records, key=lambda r: r.valid_at)
                    
                    if resolution_strategy == TemporalResolutionStrategy.FIRST_WINS:
                        # Keep the earliest, invalidate others
                        primary_record = sorted_records[0]
                        records_to_invalidate = sorted_records[1:]
                        
                    elif resolution_strategy == TemporalResolutionStrategy.LAST_WINS:
                        # Keep the latest, invalidate others
                        primary_record = sorted_records[-1]
                        records_to_invalidate = sorted_records[:-1]
                        
                    elif resolution_strategy == TemporalResolutionStrategy.MERGE:
                        # Merge overlapping records
                        primary_record = self._merge_temporal_records(sorted_records)
                        records_to_invalidate = []
                        
                    else:  # INVALIDATE or MANUAL
                        # Invalidate all conflicting records
                        primary_record = None
                        records_to_invalidate = sorted_records
                    
                    # Apply invalidation
                    for record in records_to_invalidate:
                        await self._invalidate_single_record(edge_uuid, record)
                    
                    results["resolved"] += 1
                    results["conflicts"].append({
                        "edge_uuid": edge_uuid,
                        "strategy": resolution_strategy.value,
                        "records_processed": len(temporal_records),
                        "records_invalidated": len(records_to_invalidate)
                    })
                    
                except Exception as e:
                    logger.error(f"Error resolving conflict for edge {edge_uuid}: {e}")
                    results["failed"] += 1
                    results["conflicts"].append({
                        "edge_uuid": edge_uuid,
                        "error": str(e)
                    })
            
            results["processed"] = len(conflicting_edges)
            
            # Push changes to hub
            self.driver._push_to_hub(f"Resolved {results['resolved']} temporal conflicts using {resolution_strategy.value}")
            
            # Update statistics
            self._update_temporal_stats()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in temporal edge invalidation: {e}")
            return {"error": str(e), "processed": 0, "resolved": 0, "failed": len(conflicting_edges)}
    
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
        try:
            results = {
                "duplicates_found": 0,
                "duplicates_removed": 0,
                "groups_processed": 0,
                "processing_errors": []
            }
            
            # Get time window
            time_window = timedelta(hours=time_window_hours)
            
            # Process edges first
            edges_df = self.driver.edges_df.copy()
            if not edges_df.empty:
                # Group by similar facts within time window
                edge_groups = self._group_similar_edges(edges_df, similarity_threshold, time_window)
                
                for group_uuids, group_info in edge_groups.items():
                    if len(group_uuids) > 1:
                        results["duplicates_found"] += len(group_uuids) - 1
                        
                        # Keep the most recent record, invalidate others
                        latest_uuid = max(
                            group_uuids, 
                            key=lambda uuid: edges_df[edges_df['uuid'] == uuid]['created_at'].iloc[0]
                        )
                        
                        # Invalidate duplicates
                        duplicate_uuids = [uuid for uuid in group_uuids if uuid != latest_uuid]
                        invalidate_result = await self.invalidate_edges(
                            duplicate_uuids, 
                            invalidation_reason="temporal_deduplication"
                        )
                        
                        results["duplicates_removed"] += invalidate_result["invalidated"]
                        results["groups_processed"] += 1
            
            # Process nodes
            nodes_df = self.driver.nodes_df.copy()
            if not nodes_df.empty:
                # Group by similar names within time window
                node_groups = self._group_similar_nodes(nodes_df, similarity_threshold, time_window)
                
                for group_uuids, group_info in node_groups.items():
                    if len(group_uuids) > 1:
                        results["duplicates_found"] += len(group_uuids) - 1
                        
                        # Keep the most recent record, invalidate others
                        latest_uuid = max(
                            group_uuids, 
                            key=lambda uuid: nodes_df[nodes_df['uuid'] == uuid]['created_at'].iloc[0]
                        )
                        
                        # Mark duplicates as invalidated (nodes don't have direct invalidation)
                        for uuid in group_uuids:
                            if uuid != latest_uuid:
                                node_mask = nodes_df['uuid'] == uuid
                                nodes_df.loc[node_mask, 'invalidated_at'] = datetime.now()
                        
                        results["duplicates_removed"] += len(group_uuids) - 1
                        results["groups_processed"] += 1
            
            # Update driver DataFrames
            self.driver.nodes_df = nodes_df
            self.driver.edges_df = edges_df
            
            # Push changes to hub
            self.driver._push_to_hub(f"Temporal deduplication: removed {results['duplicates_removed']} duplicates")
            
            # Update statistics
            self._update_temporal_stats()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in temporal deduplication: {e}")
            return {"error": str(e), "duplicates_found": 0, "duplicates_removed": 0}
    
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
        if check_types is None:
            check_types = ["validity_periods", "overlapping_records", "missing_temporal_data"]
        
        results = {
            "checks_performed": [],
            "issues_found": 0,
            "issues_resolved": 0,
            "details": []
        }
        
        try:
            # Check 1: Validity periods
            if "validity_periods" in check_types:
                validity_result = await self._check_validity_periods()
                results["checks_performed"].append("validity_periods")
                results["issues_found"] += validity_result["issues_found"]
                results["details"].extend(validity_result["details"])
            
            # Check 2: Overlapping records
            if "overlapping_records" in check_types:
                overlap_result = await self._check_overlapping_records()
                results["checks_performed"].append("overlapping_records")
                results["issues_found"] += overlap_result["issues_found"]
                results["details"].extend(overlap_result["details"])
            
            # Check 3: Missing temporal data
            if "missing_temporal_data" in check_types:
                missing_result = await self._check_missing_temporal_data()
                results["checks_performed"].append("missing_temporal_data")
                results["issues_found"] += missing_result["issues_found"]
                results["details"].extend(missing_result["details"])
            
            # Check 4: Temporal gaps
            if "temporal_gaps" in check_types:
                gaps_result = await self._check_temporal_gaps()
                results["checks_performed"].append("temporal_gaps")
                results["issues_found"] += gaps_result["issues_found"]
                results["details"].extend(gaps_result["details"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in temporal consistency check: {e}")
            return {"error": str(e), "checks_performed": [], "issues_found": 0, "details": []}
    
    # Temporal Indexing Methods
    
    async def build_temporal_indices(self) -> Dict[str, Any]:
        """
        Build efficient time-based query indices
        
        Returns:
            Dictionary with index building results
        """
        try:
            results = {
                "indices_built": [],
                "records_indexed": 0,
                "build_errors": []
            }
            
            # Build time-based indices for nodes
            nodes_indexed = self._build_node_temporal_index()
            results["indices_built"].append("nodes")
            results["records_indexed"] += nodes_indexed
            
            # Build time-based indices for edges
            edges_indexed = self._build_edge_temporal_index()
            results["indices_built"].append("edges")
            results["records_indexed"] += edges_indexed
            
            # Build temporal conflict index
            conflicts_indexed = self._build_conflict_index()
            results["indices_built"].append("conflicts")
            results["records_indexed"] += conflicts_indexed
            
            self._temporal_indices_built = True
            
            logger.info(f"Built temporal indices for {results['records_indexed']} records")
            
            return results
            
        except Exception as e:
            logger.error(f"Error building temporal indices: {e}")
            return {"error": str(e), "indices_built": [], "records_indexed": 0}
    
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
        if not self._temporal_indices_built:
            await self.build_temporal_indices()
        
        try:
            filter = TemporalQueryFilter(
                start_time=start_time,
                end_time=end_time,
                group_ids=group_ids,
                limit=limit
            )
            
            return await self.temporal_query(filter, entity_type)
            
        except Exception as e:
            logger.error(f"Error in temporal range query: {e}")
            return []
    
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
        if not self._temporal_indices_built:
            await self.build_temporal_indices()
        
        try:
            filter = TemporalQueryFilter(
                valid_at_time=query_time,
                group_ids=group_ids,
                limit=limit
            )
            
            return await self.temporal_query(filter, entity_type)
            
        except Exception as e:
            logger.error(f"Error in temporal point query: {e}")
            return []
    
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
        try:
            start_time, end_time = time_range
            
            # Get data in time range
            records = await self.temporal_range_query(start_time, end_time, entity_type)
            
            if not records:
                return {
                    "aggregation_type": aggregation_type,
                    "time_range": (start_time.isoformat(), end_time.isoformat()),
                    "total_records": 0,
                    "aggregated_value": None,
                    "grouped_results": {}
                }
            
            # Perform aggregation
            if aggregation_type == "count":
                aggregated_value = len(records)
                
            elif aggregation_type == "sum":
                # This assumes there's a numeric field to sum
                # For now, we'll use a placeholder
                aggregated_value = len(records)  # Placeholder
                
            elif aggregation_type == "avg":
                # This assumes there's a numeric field to average
                # For now, we'll use a placeholder
                aggregated_value = len(records) / ((end_time - start_time).total_seconds() / 3600) if (end_time - start_time).total_seconds() > 0 else 0
                
            elif aggregation_type == "max":
                # This assumes there's a comparable field
                # For now, we'll use the latest record's timestamp
                latest_record = max(records, key=lambda r: r.get('created_at', datetime.min))
                aggregated_value = latest_record.get('created_at', None)
                
            elif aggregation_type == "min":
                # This assumes there's a comparable field
                # For now, we'll use the earliest record's timestamp
                earliest_record = min(records, key=lambda r: r.get('created_at', datetime.min))
                aggregated_value = earliest_record.get('created_at', None)
            
            else:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
            
            # Group by field if specified
            grouped_results = {}
            if group_by:
                for record in records:
                    group_key = record.get(group_by, "unknown")
                    if group_key not in grouped_results:
                        grouped_results[group_key] = []
                    grouped_results[group_key].append(record)
                
                # Apply aggregation to each group
                for group_key, group_records in grouped_results.items():
                    if aggregation_type == "count":
                        grouped_results[group_key] = len(group_records)
                    else:
                        grouped_results[group_key] = aggregated_value  # Placeholder for other aggregations
            
            return {
                "aggregation_type": aggregation_type,
                "time_range": (start_time.isoformat(), end_time.isoformat()),
                "total_records": len(records),
                "aggregated_value": aggregated_value,
                "grouped_results": grouped_results,
                "entity_type": entity_type
            }
            
        except Exception as e:
            logger.error(f"Error in temporal aggregation: {e}")
            return {"error": str(e)}
    
    async def temporal_statistics(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> TemporalStats:
        """
        Get comprehensive temporal data analysis
        
        Args:
            time_range: Optional time range to analyze (defaults to all time)
            
        Returns:
            TemporalStats object with comprehensive statistics
        """
        try:
            # Update current statistics
            self._update_temporal_stats()
            
            # Filter by time range if specified
            if time_range:
                start_time, end_time = time_range
                
                # Filter records by time range
                filtered_stats = TemporalStats(
                    total_records=0,
                    valid_records=0,
                    invalidated_records=0,
                    conflicts_detected=0,
                    conflicts_resolved=0,
                    time_span=time_range,
                    records_by_entity_type={},
                    temporal_gaps=[],
                    data_ingestion_rate=0.0,
                    event_occurrence_distribution={}
                )
                
                # This is a simplified version - in practice, you'd want to
                # query the actual data within the time range
                return filtered_stats
            
            return self.temporal_stats
            
        except Exception as e:
            logger.error(f"Error getting temporal statistics: {e}")
            return TemporalStats(
                total_records=0,
                valid_records=0,
                invalidated_records=0,
                conflicts_detected=0,
                conflicts_resolved=0,
                time_span=(datetime.min, datetime.max),
                records_by_entity_type={},
                temporal_gaps=[],
                data_ingestion_rate=0.0,
                event_occurrence_distribution={}
            )
    
    # Temporal Conflict Resolution Methods
    
    async def resolve_temporal_conflicts(
        self, 
        conflicts: List[TemporalConflict],
        strategy: TemporalResolutionStrategy = TemporalResolutionStrategy.MANUAL
    ) -> Dict[str, Any]:
        """
        Handle overlapping time periods and conflicts
        
        Args:
            conflicts: List of conflicts to resolve
            strategy: Resolution strategy to apply
            
        Returns:
            Dictionary with resolution results
        """
        try:
            results = {
                "total_conflicts": len(conflicts),
                "resolved_conflicts": 0,
                "failed_resolutions": 0,
                "resolution_details": []
            }
            
            for conflict in conflicts:
                try:
                    # Apply resolution strategy
                    if strategy == TemporalResolutionStrategy.FIRST_WINS:
                        resolution_result = await self._resolve_first_wins(conflict)
                    elif strategy == TemporalResolutionStrategy.LAST_WINS:
                        resolution_result = await self._resolve_last_wins(conflict)
                    elif strategy == TemporalResolutionStrategy.MERGE:
                        resolution_result = await self._resolve_merge(conflict)
                    elif strategy == TemporalResolutionStrategy.INVALIDATE:
                        resolution_result = await self._resolve_invalidate(conflict)
                    else:  # MANUAL
                        resolution_result = await self._resolve_manual(conflict)
                    
                    # Update conflict status
                    conflict.resolution_strategy = strategy
                    conflict.resolved_at = datetime.now()
                    conflict.resolution_metadata = resolution_result
                    
                    results["resolved_conflicts"] += 1
                    results["resolution_details"].append({
                        "conflict_id": conflict.conflict_id,
                        "status": "resolved",
                        "strategy": strategy.value,
                        "affected_entities": conflict.affected_entities
                    })
                    
                except Exception as e:
                    logger.error(f"Error resolving conflict {conflict.conflict_id}: {e}")
                    results["failed_resolutions"] += 1
                    results["resolution_details"].append({
                        "conflict_id": conflict.conflict_id,
                        "status": "failed",
                        "error": str(e)
                    })
            
            # Update conflict index
            for conflict in conflicts:
                if conflict.resolved_at:
                    # Remove from active conflicts
                    for entity_uuid in conflict.affected_entities:
                        if entity_uuid in self.conflict_index:
                            self.conflict_index[entity_uuid] = [
                                c for c in self.conflict_index[entity_uuid] 
                                if c.conflict_id != conflict.conflict_id
                            ]
            
            # Push changes to hub
            self.driver._push_to_hub(f"Resolved {results['resolved_conflicts']} temporal conflicts")
            
            # Update statistics
            self._update_temporal_stats()
            
            return results
            
        except Exception as e:
            logger.error(f"Error resolving temporal conflicts: {e}")
            return {"error": str(e), "total_conflicts": len(conflicts), "resolved_conflicts": 0}
    
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
        try:
            # Get temporal records for the entity
            temporal_records = self.temporal_index.get(entity_uuid, [])
            
            # Filter records to merge
            records_to_merge = [r for r in temporal_records if r.uuid in record_ids]
            
            if len(records_to_merge) < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 records to merge",
                    "records_found": len(records_to_merge)
                }
            
            # Sort by validity period
            sorted_records = sorted(records_to_merge, key=lambda r: r.valid_at)
            
            # Create merged record
            merged_record = self._merge_temporal_records(sorted_records)
            
            # Invalidate original records
            for record in records_to_merge:
                await self._invalidate_single_record(entity_uuid, record)
            
            # Add merged record
            self.temporal_index[entity_uuid].append(merged_record)
            
            # Update driver data
            if entity_uuid in self.driver.edges_df['uuid'].values:
                edge_mask = self.driver.edges_df['uuid'] == entity_uuid
                self.driver.edges_df.loc[edge_mask, 'valid_at'] = merged_record.valid_at
                if merged_record.invalid_at:
                    self.driver.edges_df.loc[edge_mask, 'invalidated_at'] = merged_record.invalid_at
            
            # Push changes to hub
            self.driver._push_to_hub(f"Merged {len(records_to_merge)} temporal records for {entity_uuid}")
            
            return {
                "success": True,
                "entity_uuid": entity_uuid,
                "records_merged": len(records_to_merge),
                "merged_record": {
                    "valid_from": merged_record.valid_at.isoformat(),
                    "valid_to": merged_record.invalid_at.isoformat() if merged_record.invalid_at else None,
                    "event_occurrence": merged_record.event_occurrence_time.isoformat(),
                    "data_ingestion": merged_record.data_ingestion_time.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error merging temporal records for {entity_uuid}: {e}")
            return {"success": False, "error": str(e)}
    
    async def detect_temporal_anomalies(
        self,
        anomaly_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify inconsistencies in temporal data
        
        Args:
            anomaly_types: List of anomaly types to detect
            
        Returns:
            List of detected anomalies
        """
        if anomaly_types is None:
            anomaly_types = ["gaps", "overlaps", "inconsistencies", "outliers"]
        
        anomalies = []
        
        try:
            # Detect temporal gaps
            if "gaps" in anomaly_types:
                gap_anomalies = await self._detect_temporal_gaps()
                anomalies.extend(gap_anomalies)
            
            # Detect overlapping records
            if "overlaps" in anomaly_types:
                overlap_anomalies = await self._detect_overlapping_records()
                anomalies.extend(overlap_anomalies)
            
            # Detect inconsistencies
            if "inconsistencies" in anomaly_types:
                inconsistency_anomalies = await self._detect_temporal_inconsistencies()
                anomalies.extend(inconsistency_anomalies)
            
            # Detect outliers
            if "outliers" in anomaly_types:
                outlier_anomalies = await self._detect_temporal_outliers()
                anomalies.extend(outlier_anomalies)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting temporal anomalies: {e}")
            return [{"error": str(e), "detection_failed": True}]
    
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
        try:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            results = {
                "cleanup_strategy": cleanup_strategy,
                "cutoff_time": cutoff_time.isoformat(),
                "records_processed": 0,
                "records_cleaned": 0,
                "errors": []
            }
            
            # Process edges
            edges_to_clean = []
            for _, edge_row in self.driver.edges_df.iterrows():
                if (edge_row.get('invalidated_at') and 
                    edge_row['invalidated_at'] < cutoff_time):
                    edges_to_clean.append(edge_row['uuid'])
            
            # Process nodes
            nodes_to_clean = []
            for _, node_row in self.driver.nodes_df.iterrows():
                if (node_row.get('invalidated_at') and 
                    node_row['invalidated_at'] < cutoff_time):
                    nodes_to_clean.append(node_row['uuid'])
            
            results["records_processed"] = len(edges_to_clean) + len(nodes_to_clean)
            
            if cleanup_strategy == "soft":
                # Mark as archived but keep in dataset
                for edge_uuid in edges_to_clean:
                    edge_mask = self.driver.edges_df['uuid'] == edge_uuid
                    self.driver.edges_df.loc[edge_mask, 'metadata'] = self.driver.edges_df.loc[edge_mask, 'metadata'].apply(
                        lambda x: json.dumps({**json.loads(x), "archived": True}) if x else '{"archived": True}'
                    )
                
                for node_uuid in nodes_to_clean:
                    node_mask = self.driver.nodes_df['uuid'] == node_uuid
                    self.driver.nodes_df.loc[node_mask, 'metadata'] = self.driver.nodes_df.loc[node_mask, 'metadata'].apply(
                        lambda x: json.dumps({**json.loads(x), "archived": True}) if x else '{"archived": True}'
                    )
                
                results["records_cleaned"] = len(edges_to_clean) + len(nodes_to_clean)
                
            elif cleanup_strategy == "hard":
                # Remove from dataset
                if edges_to_clean:
                    self.driver.edges_df = self.driver.edges_df[
                        ~self.driver.edges_df['uuid'].isin(edges_to_clean)
                    ]
                
                if nodes_to_clean:
                    self.driver.nodes_df = self.driver.nodes_df[
                        ~self.driver.nodes_df['uuid'].isin(nodes_to_clean)
                    ]
                
                results["records_cleaned"] = len(edges_to_clean) + len(nodes_to_clean)
                
                # Update temporal index
                for edge_uuid in edges_to_clean:
                    if edge_uuid in self.temporal_index:
                        del self.temporal_index[edge_uuid]
                
                for node_uuid in nodes_to_clean:
                    if node_uuid in self.temporal_index:
                        del self.temporal_index[node_uuid]
            
            elif cleanup_strategy == "archive":
                # Move to archive dataset (not implemented in this version)
                results["records_cleaned"] = 0
                results["errors"].append("Archive cleanup strategy not implemented")
            
            # Push changes to hub
            self.driver._push_to_hub(f"Temporal cleanup: removed {results['records_cleaned']} old records")
            
            # Update statistics
            self._update_temporal_stats()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in auto temporal cleanup: {e}")
            return {"error": str(e), "records_processed": 0, "records_cleaned": 0}
    
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
        try:
            if version_action == "create":
                # Create a new version of the entity
                temporal_records = self.temporal_index.get(entity_uuid, [])
                
                if temporal_records:
                    latest_record = max(temporal_records, key=lambda r: r.valid_at)
                    new_version = TemporalRecord(
                        uuid=f"{entity_uuid}_v{latest_record.version + 1}",
                        event_occurrence_time=latest_record.event_occurrence_time,
                        data_ingestion_time=datetime.now(),
                        valid_at=datetime.now(),
                        invalid_at=None,
                        metadata={"parent_entity": entity_uuid, "version": latest_record.version + 1}
                    )
                    
                    self.temporal_index[entity_uuid].append(new_version)
                    
                    return {
                        "success": True,
                        "version_created": new_version.version,
                        "version_uuid": new_version.uuid,
                        "parent_entity": entity_uuid
                    }
                else:
                    return {
                        "success": False,
                        "error": "No existing records to version"
                    }
            
            elif version_action == "list":
                # List all versions of the entity
                temporal_records = self.temporal_index.get(entity_uuid, [])
                
                versions = []
                for record in temporal_records:
                    versions.append({
                        "version": record.version,
                        "uuid": record.uuid,
                        "valid_from": record.valid_at.isoformat(),
                        "valid_to": record.invalid_at.isoformat() if record.invalid_at else None,
                        "created_at": record.created_at.isoformat()
                    })
                
                return {
                    "success": True,
                    "entity_uuid": entity_uuid,
                    "versions": versions,
                    "total_versions": len(versions)
                }
            
            elif version_action == "restore":
                # Restore a specific version (not fully implemented)
                return {
                    "success": False,
                    "error": "Version restore not implemented yet"
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown version action: {version_action}"
                }
            
        except Exception as e:
            logger.error(f"Error in temporal versioning for {entity_uuid}: {e}")
            return {"success": False, "error": str(e)}
    
    # Helper Methods
    
    def _update_temporal_stats(self):
        """Update temporal statistics"""
        try:
            # Count total records
            total_nodes = len(self.driver.nodes_df)
            total_edges = len(self.driver.edges_df)
            self.temporal_stats.total_records = total_nodes + total_edges
            
            # Count valid and invalidated records
            valid_nodes = len(self.driver.nodes_df[self.driver.nodes_df['invalidated_at'].isna()])
            valid_edges = len(self.driver.edges_df[self.driver.edges_df['invalidated_at'].isna()])
            self.temporal_stats.valid_records = valid_nodes + valid_edges
            
            self.temporal_stats.invalidated_records = self.temporal_stats.total_records - self.temporal_stats.valid_records
            
            # Update time span
            all_times = []
            if not self.driver.nodes_df.empty:
                all_times.extend(self.driver.nodes_df['created_at'].dropna().tolist())
            if not self.driver.edges_df.empty:
                all_times.extend(self.driver.edges_df['created_at'].dropna().tolist())
            
            if all_times:
                self.temporal_stats.time_span = (min(all_times), max(all_times))
            
            # Count conflicts
            total_conflicts = sum(len(conflicts) for conflicts in self.conflict_index.values())
            self.temporal_stats.conflicts_detected = total_conflicts
            
            # Calculate data ingestion rate (simplified)
            if self.temporal_stats.time_span[0] != datetime.min:
                time_span_hours = (self.temporal_stats.time_span[1] - self.temporal_stats.time_span[0]).total_seconds() / 3600
                if time_span_hours > 0:
                    self.temporal_stats.data_ingestion_rate = self.temporal_stats.total_records / time_span_hours
            
        except Exception as e:
            logger.error(f"Error updating temporal stats: {e}")
    
    def _merge_temporal_records(self, records: List[TemporalRecord]) -> TemporalRecord:
        """Merge multiple temporal records into one"""
        if not records:
            raise ValueError("Cannot merge empty list of records")
        
        # Sort by validity period
        sorted_records = sorted(records, key=lambda r: r.valid_at)
        
        # Create merged record with earliest start and latest end
        merged = TemporalRecord(
            uuid=f"{sorted_records[0].uuid}_merged",
            event_occurrence_time=min(r.event_occurrence_time for r in sorted_records),
            data_ingestion_time=datetime.now(),
            valid_at=sorted_records[0].valid_at,
            invalid_at=max((r.invalid_at for r in sorted_records if r.invalid_at), default=None),
            metadata={"merged_from": [r.uuid for r in sorted_records]}
        )
        
        return merged
    
    async def _invalidate_single_record(self, entity_uuid: str, record: TemporalRecord):
        """Invalidate a single temporal record"""
        try:
            # Update in driver
            if entity_uuid in self.driver.edges_df['uuid'].values:
                edge_mask = self.driver.edges_df['uuid'] == entity_uuid
                self.driver.edges_df.loc[edge_mask, 'invalidated_at'] = datetime.now()
            
            # Update in temporal index
            if entity_uuid in self.temporal_index:
                for i, existing_record in enumerate(self.temporal_index[entity_uuid]):
                    if existing_record.uuid == record.uuid:
                        self.temporal_index[entity_uuid][i].invalid_at = datetime.now()
                        break
            
        except Exception as e:
            logger.error(f"Error invalidating record {record.uuid}: {e}")
    
    def _build_node_temporal_index(self) -> int:
        """Build temporal index for nodes"""
        indexed_count = 0
        
        try:
            for _, node_row in self.driver.nodes_df.iterrows():
                node_uuid = node_row['uuid']
                
                # Create temporal record
                temporal_record = TemporalRecord(
                    uuid=node_uuid,
                    event_occurrence_time=node_row.get('created_at', datetime.now()),
                    data_ingestion_time=node_row.get('created_at', datetime.now()),
                    valid_at=node_row.get('valid_at', node_row.get('created_at', datetime.now())),
                    invalid_at=node_row.get('invalidated_at'),
                    metadata={"entity_type": "node"}
                )
                
                self.temporal_index[node_uuid].append(temporal_record)
                indexed_count += 1
                
        except Exception as e:
            logger.error(f"Error building node temporal index: {e}")
        
        return indexed_count
    
    def _build_edge_temporal_index(self) -> int:
        """Build temporal index for edges"""
        indexed_count = 0
        
        try:
            for _, edge_row in self.driver.edges_df.iterrows():
                edge_uuid = edge_row['uuid']
                
                # Create temporal record
                temporal_record = TemporalRecord(
                    uuid=edge_uuid,
                    event_occurrence_time=edge_row.get('created_at', datetime.now()),
                    data_ingestion_time=edge_row.get('created_at', datetime.now()),
                    valid_at=edge_row.get('valid_at', edge_row.get('created_at', datetime.now())),
                    invalid_at=edge_row.get('invalidated_at'),
                    metadata={"entity_type": "edge"}
                )
                
                self.temporal_index[edge_uuid].append(temporal_record)
                indexed_count += 1
                
        except Exception as e:
            logger.error(f"Error building edge temporal index: {e}")
        
        return indexed_count
    
    def _build_conflict_index(self) -> int:
        """Build conflict index"""
        # This would typically scan for existing conflicts
        # For now, return 0 as conflicts are detected dynamically
        return 0
    
    def _group_similar_edges(self, edges_df, similarity_threshold, time_window):
        """Group similar edges within time window for deduplication"""
        groups = {}
        
        try:
            # Simple grouping by exact fact match within time window
            # In a real implementation, you'd use more sophisticated similarity
            for _, edge1 in edges_df.iterrows():
                for _, edge2 in edges_df.iterrows():
                    if (edge1['uuid'] != edge2['uuid'] and
                        edge1['fact'] == edge2['fact'] and
                        abs((edge1['created_at'] - edge2['created_at']).total_seconds()) <= time_window.total_seconds()):
                        
                        group_key = edge1['fact']
                        if group_key not in groups:
                            groups[group_key] = []
                        groups[group_key].extend([edge1['uuid'], edge2['uuid']])
            
            # Remove duplicates and create proper groups
            final_groups = {}
            for group_key, uuids in groups.items():
                unique_uuids = list(set(uuids))
                final_groups[group_key] = unique_uuids
            
            return final_groups
            
        except Exception as e:
            logger.error(f"Error grouping similar edges: {e}")
            return {}
    
    def _group_similar_nodes(self, nodes_df, similarity_threshold, time_window):
        """Group similar nodes within time window for deduplication"""
        groups = {}
        
        try:
            # Simple grouping by exact name match within time window
            for _, node1 in nodes_df.iterrows():
                for _, node2 in nodes_df.iterrows():
                    if (node1['uuid'] != node2['uuid'] and
                        node1['name'] == node2['name'] and
                        abs((node1['created_at'] - node2['created_at']).total_seconds()) <= time_window.total_seconds()):
                        
                        group_key = node1['name']
                        if group_key not in groups:
                            groups[group_key] = []
                        groups[group_key].extend([node1['uuid'], node2['uuid']])
            
            # Remove duplicates and create proper groups
            final_groups = {}
            for group_key, uuids in groups.items():
                unique_uuids = list(set(uuids))
                final_groups[group_key] = unique_uuids
            
            return final_groups
            
        except Exception as e:
            logger.error(f"Error grouping similar nodes: {e}")
            return {}
    
    async def _check_validity_periods(self) -> Dict[str, Any]:
        """Check validity periods for consistency"""
        issues = []
        
        try:
            # Check edges for validity period issues
            for _, edge_row in self.driver.edges_df.iterrows():
                valid_at = edge_row.get('valid_at')
                invalidated_at = edge_row.get('invalidated_at')
                created_at = edge_row.get('created_at')
                
                if valid_at and created_at and valid_at < created_at:
                    issues.append({
                        "type": "invalid_validity_period",
                        "entity_uuid": edge_row['uuid'],
                        "entity_type": "edge",
                        "issue": f"valid_at ({valid_at}) is before created_at ({created_at})"
                    })
                
                if valid_at and invalidated_at and invalidated_at < valid_at:
                    issues.append({
                        "type": "invalid_invalidation",
                        "entity_uuid": edge_row['uuid'],
                        "entity_type": "edge",
                        "issue": f"invalidated_at ({invalidated_at}) is before valid_at ({valid_at})"
                    })
            
            # Check nodes similarly
            for _, node_row in self.driver.nodes_df.iterrows():
                valid_at = node_row.get('valid_at')
                invalidated_at = node_row.get('invalidated_at')
                created_at = node_row.get('created_at')
                
                if valid_at and created_at and valid_at < created_at:
                    issues.append({
                        "type": "invalid_validity_period",
                        "entity_uuid": node_row['uuid'],
                        "entity_type": "node",
                        "issue": f"valid_at ({valid_at}) is before created_at ({created_at})"
                    })
                
                if valid_at and invalidated_at and invalidated_at < valid_at:
                    issues.append({
                        "type": "invalid_invalidation",
                        "entity_uuid": node_row['uuid'],
                        "entity_type": "node",
                        "issue": f"invalidated_at ({invalidated_at}) is before valid_at ({valid_at})"
                    })
            
            return {
                "issues_found": len(issues),
                "details": issues
            }
            
        except Exception as e:
            logger.error(f"Error checking validity periods: {e}")
            return {"issues_found": 0, "details": []}
    
    async def _check_overlapping_records(self) -> Dict[str, Any]:
        """Check for overlapping temporal records"""
        issues = []
        
        try:
            # Check for overlapping edges by source-target pairs
            edge_groups = self.driver.edges_df.groupby(['source_uuid', 'target_uuid'])
            
            for (source_uuid, target_uuid), group in edge_groups:
                sorted_group = group.sort_values('valid_at')
                
                for i in range(len(sorted_group) - 1):
                    current = sorted_group.iloc[i]
                    next_record = sorted_group.iloc[i + 1]
                    
                    current_end = current.get('invalidated_at') or datetime.max
                    next_start = next_record.get('valid_at')
                    
                    if next_start < current_end:
                        issues.append({
                            "type": "overlapping_records",
                            "entity_uuids": [current['uuid'], next_record['uuid']],
                            "entity_type": "edge",
                            "source_target": f"{source_uuid}->{target_uuid}",
                            "overlap_period": {
                                "start": next_start.isoformat(),
                                "end": current_end.isoformat()
                            }
                        })
            
            return {
                "issues_found": len(issues),
                "details": issues
            }
            
        except Exception as e:
            logger.error(f"Error checking overlapping records: {e}")
            return {"issues_found": 0, "details": []}
    
    async def _check_missing_temporal_data(self) -> Dict[str, Any]:
        """Check for missing temporal data"""
        issues = []
        
        try:
            # Check edges for missing temporal fields
            for _, edge_row in self.driver.edges_df.iterrows():
                if pd.isna(edge_row.get('valid_at')):
                    issues.append({
                        "type": "missing_validity",
                        "entity_uuid": edge_row['uuid'],
                        "entity_type": "edge",
                        "missing_field": "valid_at"
                    })
                
                if pd.isna(edge_row.get('created_at')):
                    issues.append({
                        "type": "missing_timestamp",
                        "entity_uuid": edge_row['uuid'],
                        "entity_type": "edge",
                        "missing_field": "created_at"
                    })
            
            # Check nodes similarly
            for _, node_row in self.driver.nodes_df.iterrows():
                if pd.isna(node_row.get('valid_at')):
                    issues.append({
                        "type": "missing_validity",
                        "entity_uuid": node_row['uuid'],
                        "entity_type": "node",
                        "missing_field": "valid_at"
                    })
                
                if pd.isna(node_row.get('created_at')):
                    issues.append({
                        "type": "missing_timestamp",
                        "entity_uuid": node_row['uuid'],
                        "entity_type": "node",
                        "missing_field": "created_at"
                    })
            
            return {
                "issues_found": len(issues),
                "details": issues
            }
            
        except Exception as e:
            logger.error(f"Error checking missing temporal data: {e}")
            return {"issues_found": 0, "details": []}
    
    async def _check_temporal_gaps(self) -> Dict[str, Any]:
        """Check for temporal gaps in the data"""
        issues = []
        
        try:
            # Check for gaps in edge creation times
            if not self.driver.edges_df.empty:
                sorted_edges = self.driver.edges_df.sort_values('created_at')
                
                for i in range(len(sorted_edges) - 1):
                    current = sorted_edges.iloc[i]
                    next_record = sorted_edges.iloc[i + 1]
                    
                    time_diff = (next_record['created_at'] - current['created_at']).total_seconds()
                    
                    # If gap is more than 7 days, flag as potential issue
                    if time_diff > 7 * 24 * 3600:  # 7 days in seconds
                        issues.append({
                            "type": "temporal_gap",
                            "entity_uuids": [current['uuid'], next_record['uuid']],
                            "entity_type": "edge",
                            "gap_duration": time_diff / 3600,  # hours
                            "gap_period": {
                                "start": current['created_at'].isoformat(),
                                "end": next_record['created_at'].isoformat()
                            }
                        })
            
            return {
                "issues_found": len(issues),
                "details": issues
            }
            
        except Exception as e:
            logger.error(f"Error checking temporal gaps: {e}")
            return {"issues_found": 0, "details": []}
    
    async def _resolve_first_wins(self, conflict: TemporalConflict) -> Dict[str, Any]:
        """Resolve conflict using first-wins strategy"""
        try:
            # Sort by validity period, keep the earliest
            sorted_records = sorted(conflict.conflicting_records, key=lambda r: r.valid_at)
            primary_record = sorted_records[0]
            records_to_invalidate = sorted_records[1:]
            
            # Invalidate other records
            for record in records_to_invalidate:
                await self._invalidate_single_record(conflict.affected_entities[0], record)
            
            return {
                "strategy": "first_wins",
                "primary_record": primary_record.uuid,
                "invalidated_records": [r.uuid for r in records_to_invalidate]
            }
            
        except Exception as e:
            logger.error(f"Error in first-wins resolution: {e}")
            return {"error": str(e)}
    
    async def _resolve_last_wins(self, conflict: TemporalConflict) -> Dict[str, Any]:
        """Resolve conflict using last-wins strategy"""
        try:
            # Sort by validity period, keep the latest
            sorted_records = sorted(conflict.conflicting_records, key=lambda r: r.valid_at)
            primary_record = sorted_records[-1]
            records_to_invalidate = sorted_records[:-1]
            
            # Invalidate other records
            for record in records_to_invalidate:
                await self._invalidate_single_record(conflict.affected_entities[0], record)
            
            return {
                "strategy": "last_wins",
                "primary_record": primary_record.uuid,
                "invalidated_records": [r.uuid for r in records_to_invalidate]
            }
            
        except Exception as e:
            logger.error(f"Error in last-wins resolution: {e}")
            return {"error": str(e)}
    
    async def _resolve_merge(self, conflict: TemporalConflict) -> Dict[str, Any]:
        """Resolve conflict using merge strategy"""
        try:
            # Merge all conflicting records
            merged_record = self._merge_temporal_records(conflict.conflicting_records)
            
            # Invalidate original records
            for record in conflict.conflicting_records:
                await self._invalidate_single_record(conflict.affected_entities[0], record)
            
            # Add merged record
            self.temporal_index[conflict.affected_entities[0]].append(merged_record)
            
            return {
                "strategy": "merge",
                "merged_record": merged_record.uuid,
                "original_records": [r.uuid for r in conflict.conflicting_records]
            }
            
        except Exception as e:
            logger.error(f"Error in merge resolution: {e}")
            return {"error": str(e)}
    
    async def _resolve_invalidate(self, conflict: TemporalConflict) -> Dict[str, Any]:
        """Resolve conflict using invalidate strategy"""
        try:
            # Invalidate all conflicting records
            invalidated_records = []
            for record in conflict.conflicting_records:
                await self._invalidate_single_record(conflict.affected_entities[0], record)
                invalidated_records.append(record.uuid)
            
            return {
                "strategy": "invalidate",
                "invalidated_records": invalidated_records
            }
            
        except Exception as e:
            logger.error(f"Error in invalidate resolution: {e}")
            return {"error": str(e)}
    
    async def _resolve_manual(self, conflict: TemporalConflict) -> Dict[str, Any]:
        """Resolve conflict using manual strategy"""
        try:
            # For manual resolution, we just mark the conflict as requiring attention
            return {
                "strategy": "manual",
                "requires_attention": True,
                "conflict_id": conflict.conflict_id,
                "affected_entities": conflict.affected_entities
            }
            
        except Exception as e:
            logger.error(f"Error in manual resolution: {e}")
            return {"error": str(e)}
    
    async def _detect_temporal_gaps(self) -> List[Dict[str, Any]]:
        """Detect temporal gaps in the data"""
        gaps = []
        
        try:
            # Similar to _check_temporal_gaps but returns anomalies
            if not self.driver.edges_df.empty:
                sorted_edges = self.driver.edges_df.sort_values('created_at')
                
                for i in range(len(sorted_edges) - 1):
                    current = sorted_edges.iloc[i]
                    next_record = sorted_edges.iloc[i + 1]
                    
                    time_diff = (next_record['created_at'] - current['created_at']).total_seconds()
                    
                    if time_diff > 7 * 24 * 3600:  # 7 days in seconds
                        gaps.append({
                            "type": "temporal_gap",
                            "severity": "medium" if time_diff < 30 * 24 * 3600 else "high",
                            "entity_uuids": [current['uuid'], next_record['uuid']],
                            "gap_duration_hours": time_diff / 3600,
                            "gap_period": {
                                "start": current['created_at'].isoformat(),
                                "end": next_record['created_at'].isoformat()
                            }
                        })
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error detecting temporal gaps: {e}")
            return []
    
    async def _detect_overlapping_records(self) -> List[Dict[str, Any]]:
        """Detect overlapping temporal records"""
        overlaps = []
        
        try:
            # Similar to _check_overlapping_records but returns anomalies
            edge_groups = self.driver.edges_df.groupby(['source_uuid', 'target_uuid'])
            
            for (source_uuid, target_uuid), group in edge_groups:
                sorted_group = group.sort_values('valid_at')
                
                for i in range(len(sorted_group) - 1):
                    current = sorted_group.iloc[i]
                    next_record = sorted_group.iloc[i + 1]
                    
                    current_end = current.get('invalidated_at') or datetime.max
                    next_start = next_record.get('valid_at')
                    
                    if next_start < current_end:
                        overlaps.append({
                            "type": "overlapping_records",
                            "severity": "high",
                            "entity_uuids": [current['uuid'], next_record['uuid']],
                            "overlap_period": {
                                "start": next_start.isoformat(),
                                "end": current_end.isoformat()
                            }
                        })
            
            return overlaps
            
        except Exception as e:
            logger.error(f"Error detecting overlapping records: {e}")
            return []
    
    async def _detect_temporal_inconsistencies(self) -> List[Dict[str, Any]]:
        """Detect temporal inconsistencies"""
        inconsistencies = []
        
        try:
            # Check for various inconsistencies
            for _, edge_row in self.driver.edges_df.iterrows():
                # Check for future dates
                if edge_row.get('valid_at') and edge_row['valid_at'] > datetime.now():
                    inconsistencies.append({
                        "type": "future_date",
                        "severity": "low",
                        "entity_uuid": edge_row['uuid'],
                        "entity_type": "edge",
                        "field": "valid_at",
                        "value": edge_row['valid_at'].isoformat()
                    })
                
                # Check for very old dates (before 2000)
                if edge_row.get('created_at') and edge_row['created_at'].year < 2000:
                    inconsistencies.append({
                        "type": "unrealistic_date",
                        "severity": "medium",
                        "entity_uuid": edge_row['uuid'],
                        "entity_type": "edge",
                        "field": "created_at",
                        "value": edge_row['created_at'].isoformat()
                    })
            
            return inconsistencies
            
        except Exception as e:
            logger.error(f"Error detecting temporal inconsistencies: {e}")
            return []
    
    async def _detect_temporal_outliers(self) -> List[Dict[str, Any]]:
        """Detect temporal outliers"""
        outliers = []
        
        try:
            # Check for creation time outliers
            if not self.driver.edges_df.empty:
                creation_times = self.driver.edges_df['created_at'].dropna()
                if len(creation_times) > 2:
                    mean_time = creation_times.mean()
                    std_time = creation_times.std()
                    
                    # Find outliers (more than 3 standard deviations from mean)
                    for _, edge_row in self.driver.edges_df.iterrows():
                        if not pd.isna(edge_row['created_at']):
                            time_diff = abs((edge_row['created_at'] - mean_time).total_seconds())
                            if time_diff > 3 * std_time.total_seconds():
                                outliers.append({
                                    "type": "creation_time_outlier",
                                    "severity": "medium",
                                    "entity_uuid": edge_row['uuid'],
                                    "entity_type": "edge",
                                    "deviation_std": time_diff / std_time.total_seconds(),
                                    "creation_time": edge_row['created_at'].isoformat()
                                })
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting temporal outliers: {e}")
            return []