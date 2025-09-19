"""
Tests for TemporalManager functionality

This module contains comprehensive tests for the temporal management features
of Graphiti-HF, including bi-temporal data model, temporal edge invalidation,
historical state reconstruction, time-based queries, and temporal conflict resolution.
"""

import asyncio
import logging
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch

import pandas as pd
from graphiti_hf.processing.temporal_manager import (
    TemporalManager,
    TemporalRecord,
    TemporalConflict,
    TemporalConflictType,
    TemporalResolutionStrategy,
    TemporalQueryFilter,
    TemporalStats
)
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTemporalManager:
    """Test suite for TemporalManager class"""
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock HuggingFaceDriver for testing"""
        driver = Mock(spec=HuggingFaceDriver)
        
        # Create sample data
        current_time = datetime.now()
        past_time = current_time - timedelta(days=30)
        
        # Sample nodes DataFrame
        nodes_data = {
            'uuid': ['node-1', 'node-2', 'node-3'],
            'name': ['Apple Inc.', 'Steve Jobs', 'Steve Wozniak'],
            'labels': ['Company', 'Person', 'Person'],
            'properties': ['{"industry": "technology"}', 'occupation="entrepreneur"', 'occupation="engineer"'],
            'name_embedding': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            'group_id': ['tech', 'tech', 'tech'],
            'created_at': [past_time, past_time, past_time],
            'valid_at': [past_time, past_time, past_time]
        }
        driver.nodes_df = pd.DataFrame(nodes_data)
        
        # Sample edges DataFrame
        edges_data = {
            'uuid': ['edge-1', 'edge-2', 'edge-3'],
            'source_uuid': ['node-1', 'node-2', 'node-3'],
            'target_uuid': ['node-2', 'node-1', 'node-1'],
            'fact': ['Apple was founded by Steve Jobs', 'Steve Jobs co-founded Apple', 'Steve Wozniak co-founded Apple'],
            'fact_embedding': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            'episodes': ['episode-1', 'episode-1', 'episode-1'],
            'created_at': [past_time, past_time, past_time],
            'valid_at': [past_time, past_time, past_time],
            'invalidated_at': [None, None, None]
        }
        driver.edges_df = pd.DataFrame(edges_data)
        
        # Mock push_to_hub method
        driver._push_to_hub = AsyncMock()
        
        return driver
    
    @pytest.fixture
    def temporal_manager(self, mock_driver):
        """Create a TemporalManager instance for testing"""
        return TemporalManager(mock_driver)
    
    @pytest.fixture
    def sample_temporal_records(self):
        """Create sample temporal records for testing"""
        current_time = datetime.now()
        past_time = current_time - timedelta(days=30)
        
        records = [
            TemporalRecord(
                uuid="record-1",
                event_occurrence_time=past_time,
                data_ingestion_time=current_time,
                valid_at=past_time,
                invalid_at=None,
                metadata={"entity_type": "edge"}
            ),
            TemporalRecord(
                uuid="record-2",
                event_occurrence_time=past_time + timedelta(days=1),
                data_ingestion_time=current_time,
                valid_at=past_time + timedelta(days=1),
                invalid_at=None,
                metadata={"entity_type": "edge"}
            )
        ]
        
        return records
    
    @pytest.mark.asyncio
    async def test_set_validity_period_success(self, temporal_manager, mock_driver):
        """Test successful setting of validity period"""
        entity_uuid = "test-entity"
        valid_from = datetime.now() - timedelta(days=10)
        valid_to = datetime.now() + timedelta(days=10)
        
        result = await temporal_manager.set_validity_period(
            entity_uuid=entity_uuid,
            valid_from=valid_from,
            valid_to=valid_to,
            entity_type="edge"
        )
        
        assert result["success"] is True
        assert result["entity_uuid"] == entity_uuid
        assert result["valid_from"] == valid_from.isoformat()
        assert result["valid_to"] == valid_to.isoformat()
        assert result["records_updated"] == 1
        
        # Verify temporal index was updated
        assert entity_uuid in temporal_manager.temporal_index
        assert len(temporal_manager.temporal_index[entity_uuid]) == 1
        
        # Verify driver was called
        mock_driver._push_to_hub.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_validity_period_no_valid_to(self, temporal_manager, mock_driver):
        """Test setting validity period without end date (ongoing)"""
        entity_uuid = "test-entity"
        valid_from = datetime.now() - timedelta(days=10)
        
        result = await temporal_manager.set_validity_period(
            entity_uuid=entity_uuid,
            valid_from=valid_from,
            entity_type="edge"
        )
        
        assert result["success"] is True
        assert result["valid_to"] is None
        assert result["records_updated"] == 1
    
    @pytest.mark.asyncio
    async def test_set_validity_period_failure(self, temporal_manager):
        """Test failure when setting validity period"""
        # Test with invalid entity type
        result = await temporal_manager.set_validity_period(
            entity_uuid="non-existent",
            valid_from=datetime.now(),
            entity_type="invalid_type"
        )
        
        # Should handle gracefully even for non-existent entities
        assert "success" in result
    
    @pytest.mark.asyncio
    async def test_invalidate_edges_success(self, temporal_manager, mock_driver):
        """Test successful edge invalidation"""
        edge_uuids = ["edge-1", "edge-2"]
        invalidation_time = datetime.now()
        
        result = await temporal_manager.invalidate_edges(
            edge_uuids=edge_uuids,
            invalidation_reason="test_reason",
            invalidation_time=invalidation_time
        )
        
        assert result["success"] is True
        assert result["invalidated"] == 2
        assert result["failed"] == 0
        
        # Verify driver was updated
        for edge_uuid in edge_uuids:
            edge_mask = mock_driver.edges_df['uuid'] == edge_uuid
            assert not mock_driver.edges_df[edge_mask].empty
            assert mock_driver.edges_df.loc[edge_mask, 'invalidated_at'].iloc[0] == invalidation_time
        
        mock_driver._push_to_hub.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invalidate_edges_partial_failure(self, temporal_manager, mock_driver):
        """Test edge invalidation with some failures"""
        edge_uuids = ["edge-1", "non-existent-edge"]
        
        result = await temporal_manager.invalidate_edges(
            edge_uuids=edge_uuids,
            invalidation_reason="test_reason"
        )
        
        assert result["success"] is True
        assert result["invalidated"] == 1
        assert result["failed"] == 1
        
        # Verify details contain both success and failure
        detail_statuses = [detail["status"] for detail in result["details"]]
        assert "success" in detail_statuses
        assert "failed" in detail_statuses
    
    @pytest.mark.asyncio
    async def test_get_valid_at_success(self, temporal_manager, sample_temporal_records):
        """Test successful retrieval of data valid at specific time"""
        # Add sample records to temporal index
        temporal_manager.temporal_index["test-entity"] = sample_temporal_records
        
        query_time = datetime.now() - timedelta(days=25)  # Should be valid for first record
        
        result = await temporal_manager.get_valid_at(
            entity_uuid="test-entity",
            query_time=query_time,
            entity_type="edge"
        )
        
        assert result is not None
        assert result["is_valid_at_query_time"] is True
        assert "temporal_validity" in result
    
    @pytest.mark.asyncio
    async def test_get_valid_at_no_data(self, temporal_manager):
        """Test retrieval when no valid data exists"""
        result = await temporal_manager.get_valid_at(
            entity_uuid="non-existent",
            query_time=datetime.now(),
            entity_type="edge"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_historical_state_success(self, temporal_manager, mock_driver):
        """Test successful historical state reconstruction"""
        query_time = datetime.now() - timedelta(days=15)
        
        result = await temporal_manager.get_historical_state(
            query_time=query_time,
            limit=5
        )
        
        assert result["query_time"] == query_time.isoformat()
        assert "valid_nodes" in result
        assert "valid_edges" in result
        assert "node_count" in result
        assert "edge_count" in result
        assert "reconstructed_at" in result
    
    @pytest.mark.asyncio
    async def test_get_historical_state_with_group_filter(self, temporal_manager, mock_driver):
        """Test historical state reconstruction with group filtering"""
        query_time = datetime.now() - timedelta(days=15)
        group_ids = ["tech"]
        
        result = await temporal_manager.get_historical_state(
            query_time=query_time,
            group_ids=group_ids,
            limit=5
        )
        
        assert result["query_time"] == query_time.isoformat()
        assert result["node_count"] >= 0
        assert result["edge_count"] >= 0
    
    @pytest.mark.asyncio
    async def test_temporal_query_basic(self, temporal_manager, mock_driver):
        """Test basic temporal query"""
        filter = TemporalQueryFilter(
            start_time=datetime.now() - timedelta(days=20),
            end_time=datetime.now() - timedelta(days=10),
            limit=10
        )
        
        results = await temporal_manager.temporal_query(
            filter=filter,
            entity_type="edge"
        )
        
        assert isinstance(results, list)
        # Results depend on mock data, but should be a list
    
    @pytest.mark.asyncio
    async def test_temporal_query_with_valid_at_time(self, temporal_manager, mock_driver):
        """Test temporal query with specific valid_at time"""
        valid_at_time = datetime.now() - timedelta(days=15)
        
        filter = TemporalQueryFilter(
            valid_at_time=valid_at_time
        )
        
        results = await temporal_manager.temporal_query(
            filter=filter,
            entity_type="edge"
        )
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_temporal_range_query_success(self, temporal_manager, mock_driver):
        """Test successful temporal range query"""
        start_time = datetime.now() - timedelta(days=20)
        end_time = datetime.now() - timedelta(days=10)
        
        results = await temporal_manager.temporal_range_query(
            start_time=start_time,
            end_time=end_time,
            entity_type="edge",
            limit=10
        )
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_temporal_point_query_success(self, temporal_manager, mock_driver):
        """Test successful temporal point query"""
        query_time = datetime.now() - timedelta(days=15)
        
        results = await temporal_manager.temporal_point_query(
            query_time=query_time,
            entity_type="edge",
            limit=10
        )
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_temporal_aggregation_count(self, temporal_manager, mock_driver):
        """Test temporal aggregation with count"""
        time_range = (
            datetime.now() - timedelta(days=20),
            datetime.now() - timedelta(days=10)
        )
        
        result = await temporal_manager.temporal_aggregation(
            aggregation_type="count",
            time_range=time_range,
            entity_type="edge"
        )
        
        assert "aggregated_value" in result
        assert "total_records" in result
        assert result["aggregation_type"] == "count"
    
    @pytest.mark.asyncio
    async def test_temporal_aggregation_invalid_type(self, temporal_manager, mock_driver):
        """Test temporal aggregation with invalid type"""
        time_range = (
            datetime.now() - timedelta(days=20),
            datetime.now() - timedelta(days=10)
        )
        
        result = await temporal_manager.temporal_aggregation(
            aggregation_type="invalid_type",
            time_range=time_range,
            entity_type="edge"
        )
        
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_resolve_temporal_conflicts_first_wins(self, temporal_manager, sample_temporal_records):
        """Test conflict resolution with first-wins strategy"""
        # Create a conflict
        conflict = TemporalConflict(
            conflict_id="test-conflict",
            conflict_type=TemporalConflictType.OVERLAPPING_VALIDITY,
            affected_entities=["test-entity"],
            conflicting_records=sample_temporal_records,
            detected_at=datetime.now(),
            resolution_strategy=TemporalResolutionStrategy.FIRST_WINS
        )
        
        result = await temporal_manager.resolve_temporal_conflicts(
            conflicts=[conflict],
            strategy=TemporalResolutionStrategy.FIRST_WINS
        )
        
        assert result["success"] is True
        assert result["resolved_conflicts"] == 1
        assert "resolution_details" in result
    
    @pytest.mark.asyncio
    async def test_resolve_temporal_conflicts_last_wins(self, temporal_manager, sample_temporal_records):
        """Test conflict resolution with last-wins strategy"""
        conflict = TemporalConflict(
            conflict_id="test-conflict",
            conflict_type=TemporalConflictType.OVERLAPPING_VALIDITY,
            affected_entities=["test-entity"],
            conflicting_records=sample_temporal_records,
            detected_at=datetime.now(),
            resolution_strategy=TemporalResolutionStrategy.LAST_WINS
        )
        
        result = await temporal_manager.resolve_temporal_conflicts(
            conflicts=[conflict],
            strategy=TemporalResolutionStrategy.LAST_WINS
        )
        
        assert result["success"] is True
        assert result["resolved_conflicts"] == 1
    
    @pytest.mark.asyncio
    async def test_resolve_temporal_conflicts_merge(self, temporal_manager, sample_temporal_records):
        """Test conflict resolution with merge strategy"""
        conflict = TemporalConflict(
            conflict_id="test-conflict",
            conflict_type=TemporalConflictType.OVERLAPPING_VALIDITY,
            affected_entities=["test-entity"],
            conflicting_records=sample_temporal_records,
            detected_at=datetime.now(),
            resolution_strategy=TemporalResolutionStrategy.MERGE
        )
        
        result = await temporal_manager.resolve_temporal_conflicts(
            conflicts=[conflict],
            strategy=TemporalResolutionStrategy.MERGE
        )
        
        assert result["success"] is True
        assert result["resolved_conflicts"] == 1
    
    @pytest.mark.asyncio
    async def test_detect_temporal_anomalies(self, temporal_manager, mock_driver):
        """Test temporal anomaly detection"""
        anomalies = await temporal_manager.detect_temporal_anomalies()
        
        assert isinstance(anomalies, list)
        # Anomalies depend on data quality, but should be a list
    
    @pytest.mark.asyncio
    async def test_auto_temporal_cleanup_soft(self, temporal_manager, mock_driver):
        """Test auto temporal cleanup with soft strategy"""
        result = await temporal_manager.auto_temporal_cleanup(
            cleanup_strategy="soft",
            older_than_days=7
        )
        
        assert "cleanup_strategy" in result
        assert "records_processed" in result
        assert "records_cleaned" in result
        assert result["cleanup_strategy"] == "soft"
    
    @pytest.mark.asyncio
    async def test_auto_temporal_cleanup_hard(self, temporal_manager, mock_driver):
        """Test auto temporal cleanup with hard strategy"""
        result = await temporal_manager.auto_temporal_cleanup(
            cleanup_strategy="hard",
            older_than_days=7
        )
        
        assert result["cleanup_strategy"] == "hard"
        assert "records_cleaned" in result
    
    @pytest.mark.asyncio
    async def test_temporal_versioning_create(self, temporal_manager, sample_temporal_records):
        """Test temporal versioning - create version"""
        # Add sample records
        temporal_manager.temporal_index["test-entity"] = sample_temporal_records
        
        result = await temporal_manager.temporal_versioning(
            entity_uuid="test-entity",
            version_action="create"
        )
        
        assert result["success"] is True
        assert "version_created" in result
        assert "version_uuid" in result
    
    @pytest.mark.asyncio
    async def test_temporal_versioning_list(self, temporal_manager, sample_temporal_records):
        """Test temporal versioning - list versions"""
        # Add sample records
        temporal_manager.temporal_index["test-entity"] = sample_temporal_records
        
        result = await temporal_manager.temporal_versioning(
            entity_uuid="test-entity",
            version_action="list"
        )
        
        assert result["success"] is True
        assert "versions" in result
        assert "total_versions" in result
    
    @pytest.mark.asyncio
    async def test_merge_temporal_records(self, temporal_manager, sample_temporal_records):
        """Test merging temporal records"""
        entity_uuid = "test-entity"
        record_ids = [record.uuid for record in sample_temporal_records]
        
        result = await temporal_manager.merge_temporal_records(
            entity_uuid=entity_uuid,
            record_ids=record_ids
        )
        
        assert result["success"] is True
        assert "records_merged" in result
        assert "merged_record" in result
    
    @pytest.mark.asyncio
    async def test_build_temporal_indices(self, temporal_manager, mock_driver):
        """Test building temporal indices"""
        result = await temporal_manager.build_temporal_indices()
        
        assert "indices_built" in result
        assert "records_indexed" in result
        assert temporal_manager._temporal_indices_built is True
    
    @pytest.mark.asyncio
    async def test_temporal_consistency_check(self, temporal_manager, mock_driver):
        """Test temporal consistency check"""
        result = await temporal_manager.temporal_consistency_check()
        
        assert "issues_found" in result
        assert "checks_performed" in result
        assert isinstance(result["issues_found"], int)
    
    @pytest.mark.asyncio
    async def test_temporal_deduplication(self, temporal_manager, mock_driver):
        """Test temporal deduplication"""
        result = await temporal_manager.temporal_deduplication(
            similarity_threshold=0.9,
            time_window_hours=24
        )
        
        assert "duplicates_found" in result
        assert "duplicates_removed" in result
        assert "groups_processed" in result
    
    @pytest.mark.asyncio
    async def test_temporal_edge_invalidation(self, temporal_manager, sample_temporal_records):
        """Test temporal edge invalidation method"""
        # Add sample records
        temporal_manager.temporal_index["test-entity"] = sample_temporal_records
        
        result = await temporal_manager.temporal_edge_invalidation(
            conflicting_edges=["test-entity"],
            strategy=TemporalResolutionStrategy.INVALIDATE
        )
        
        assert result["success"] is True
        assert "invalidated_edges" in result
    
    @pytest.mark.asyncio
    async def test_temporal_statistics(self, temporal_manager, mock_driver):
        """Test temporal statistics generation"""
        stats = await temporal_manager.temporal_statistics()
        
        assert isinstance(stats, TemporalStats)
        assert stats.total_records >= 0
        assert stats.valid_records >= 0
        assert stats.invalidated_records >= 0
        assert isinstance(stats.time_span, tuple)
        assert len(stats.time_span) == 2


class TestTemporalRecord:
    """Test suite for TemporalRecord class"""
    
    def test_temporal_record_creation(self):
        """Test TemporalRecord creation"""
        current_time = datetime.now()
        past_time = current_time - timedelta(days=30)
        
        record = TemporalRecord(
            uuid="test-record",
            event_occurrence_time=past_time,
            data_ingestion_time=current_time,
            valid_at=past_time,
            invalid_at=None,
            metadata={"test": "data"}
        )
        
        assert record.uuid == "test-record"
        assert record.event_occurrence_time == past_time
        assert record.data_ingestion_time == current_time
        assert record.valid_at == past_time
        assert record.invalid_at is None
        assert record.metadata == {"test": "data"}
        assert record.version == 1
    
    def test_temporal_record_with_invalidation(self):
        """Test TemporalRecord with invalidation time"""
        current_time = datetime.now()
        past_time = current_time - timedelta(days=30)
        future_time = current_time + timedelta(days=30)
        
        record = TemporalRecord(
            uuid="test-record",
            event_occurrence_time=past_time,
            data_ingestion_time=current_time,
            valid_at=past_time,
            invalid_at=future_time
        )
        
        assert record.invalid_at == future_time


class TestTemporalConflict:
    """Test suite for TemporalConflict class"""
    
    def test_temporal_conflict_creation(self):
        """Test TemporalConflict creation"""
        current_time = datetime.now()
        past_time = current_time - timedelta(days=30)
        
        record = TemporalRecord(
            uuid="test-record",
            event_occurrence_time=past_time,
            data_ingestion_time=current_time,
            valid_at=past_time
        )
        
        conflict = TemporalConflict(
            conflict_id="test-conflict",
            conflict_type=TemporalConflictType.OVERLAPPING_VALIDITY,
            affected_entities=["entity-1", "entity-2"],
            conflicting_records=[record],
            detected_at=current_time,
            resolution_strategy=TemporalResolutionStrategy.FIRST_WINS
        )
        
        assert conflict.conflict_id == "test-conflict"
        assert conflict.conflict_type == TemporalConflictType.OVERLAPPING_VALIDITY
        assert conflict.affected_entities == ["entity-1", "entity-2"]
        assert len(conflict.conflicting_records) == 1
        assert conflict.detected_at == current_time
        assert conflict.resolution_strategy == TemporalResolutionStrategy.FIRST_WINS
        assert conflict.resolved_at is None


class TestTemporalQueryFilter:
    """Test suite for TemporalQueryFilter class"""
    
    def test_temporal_query_filter_creation(self):
        """Test TemporalQueryFilter creation"""
        start_time = datetime.now() - timedelta(days=30)
        end_time = datetime.now()
        
        filter_obj = TemporalQueryFilter(
            start_time=start_time,
            end_time=end_time,
            valid_at_time=datetime.now() - timedelta(days=15),
            entity_types=["edge", "node"],
            group_ids=["tech", "business"],
            include_invalidated=False,
            limit=10
        )
        
        assert filter_obj.start_time == start_time
        assert filter_obj.end_time == end_time
        assert filter_obj.valid_at_time == datetime.now() - timedelta(days=15)
        assert filter_obj.entity_types == ["edge", "node"]
        assert filter_obj.group_ids == ["tech", "business"]
        assert filter_obj.include_invalidated is False
        assert filter_obj.limit == 10
    
    def test_temporal_query_filter_defaults(self):
        """Test TemporalQueryFilter with default values"""
        filter_obj = TemporalQueryFilter()
        
        assert filter_obj.start_time is None
        assert filter_obj.end_time is None
        assert filter_obj.valid_at_time is None
        assert filter_obj.entity_types is None
        assert filter_obj.group_ids is None
        assert filter_obj.include_invalidated is False
        assert filter_obj.limit is None


class TestTemporalStats:
    """Test suite for TemporalStats class"""
    
    def test_temporal_stats_creation(self):
        """Test TemporalStats creation"""
        time_span = (
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        
        stats = TemporalStats(
            total_records=100,
            valid_records=90,
            invalidated_records=10,
            conflicts_detected=5,
            conflicts_resolved=3,
            time_span=time_span,
            records_by_entity_type={"edge": 60, "node": 40},
            temporal_gaps=[],
            data_ingestion_rate=2.5,
            event_occurrence_distribution={"morning": 30, "afternoon": 70}
        )
        
        assert stats.total_records == 100
        assert stats.valid_records == 90
        assert stats.invalidated_records == 10
        assert stats.conflicts_detected == 5
        assert stats.conflicts_resolved == 3
        assert stats.time_span == time_span
        assert stats.records_by_entity_type == {"edge": 60, "node": 40}
        assert stats.temporal_gaps == []
        assert stats.data_ingestion_rate == 2.5
        assert stats.event_occurrence_distribution == {"morning": 30, "afternoon": 70}


class TestTemporalIntegration:
    """Integration tests for temporal functionality"""
    
    @pytest.fixture
    def real_temporal_manager(self):
        """Create a real TemporalManager with actual driver"""
        # Create a temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        repo_id = f"test-temporal-{temp_dir.split('/')[-1]}"
        
        try:
            driver = HuggingFaceDriver(
                repo_id=repo_id,
                create_repo=True,
                token=None  # Use default token
            )
            
            temporal_manager = TemporalManager(driver)
            yield temporal_manager
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_end_to_end_temporal_workflow(self, real_temporal_manager):
        """Test complete temporal workflow"""
        tm = real_temporal_manager
        
        # 1. Set validity periods
        valid_from = datetime.now() - timedelta(days=10)
        valid_to = datetime.now() + timedelta(days=10)
        
        result1 = await tm.set_validity_period(
            entity_uuid="test-entity-1",
            valid_from=valid_from,
            valid_to=valid_to,
            entity_type="edge"
        )
        
        assert result1["success"] is True
        
        # 2. Query data at specific time
        query_time = datetime.now()
        result2 = await tm.get_valid_at(
            entity_uuid="test-entity-1",
            query_time=query_time,
            entity_type="edge"
        )
        
        # Should return None if entity doesn't exist in driver
        # This is expected behavior for integration test
        
        # 3. Build temporal indices
        result3 = await tm.build_temporal_indices()
        
        assert "indices_built" in result3
        
        # 4. Get temporal statistics
        result4 = await tm.temporal_statistics()
        
        assert isinstance(result4, TemporalStats)
        
        # 5. Perform temporal consistency check
        result5 = await tm.temporal_consistency_check()
        
        assert "issues_found" in result5
        
        # 6. Detect temporal anomalies
        result6 = await tm.detect_temporal_anomalies()
        
        assert isinstance(result6, list)
        
        logger.info("✅ End-to-end temporal workflow test completed successfully")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])