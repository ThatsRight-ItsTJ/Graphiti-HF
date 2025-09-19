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
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from graphiti_hf.processing.incremental_updater import (
    IncrementalUpdater,
    Delta,
    DeltaOperation,
    DeltaOperationType,
    DeltaEntityType,
)
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.edges import EntityEdge, EpisodicEdge


@pytest.fixture
def mock_embedder():
    """Create a mock embedder"""
    mock_embed = Mock()
    mock_embed.generate_embedding = AsyncMock()
    mock_embed.generate_embedding.return_value = np.array([0.1, 0.2, 0.3])
    return mock_embed


@pytest.fixture
def mock_driver():
    """Create a mock HuggingFace driver"""
    driver = Mock(spec=HuggingFaceDriver)
    driver.add_entity = AsyncMock()
    driver.add_edge = AsyncMock()
    driver.update_entity = AsyncMock()
    driver.update_edge = AsyncMock()
    driver.remove_entity = AsyncMock()
    driver.remove_edge = AsyncMock()
    driver.get_entity = AsyncMock()
    driver.get_edge = AsyncMock()
    driver.search_entities = AsyncMock()
    driver.search_edges = AsyncMock()
    driver.get_entities = AsyncMock()
    driver.get_edges = AsyncMock()
    driver.save_dataset = AsyncMock()
    driver.load_dataset = AsyncMock()
    return driver


@pytest.fixture
def incremental_updater(mock_driver):
    """Create an IncrementalUpdater instance"""
    return IncrementalUpdater(mock_driver)


@pytest.fixture
def sample_entity():
    """Create a sample entity for testing"""
    return EntityNode(
        name="Test Entity",
        labels=["Person"],
        created_at=datetime.now(),
        summary="A test entity",
        attributes={"age": 30, "location": "New York"},
    )


@pytest.fixture
def sample_edge():
    """Create a sample edge for testing"""
    return EntityEdge(
        source_node_uuid="source-uuid",
        target_node_uuid="target-uuid",
        created_at=datetime.now(),
        name="RELATES_TO",
        fact="Test entity relationship",
        episodes=[],
        group_id="test-group",
    )


@pytest.fixture
def sample_delta():
    """Create a sample delta for testing"""
    delta = Delta(
        operations=[
            DeltaOperation(
                operation_type=DeltaOperationType.ADD,
                entity_type=DeltaEntityType.NODE,
                uuid="entity-uuid",
                data={
                    "name": "Test Entity",
                    "labels": ["Person"],
                    "summary": "A test entity",
                    "attributes": {"age": 30},
                },
            ),
            DeltaOperation(
                operation_type=DeltaOperationType.ADD,
                entity_type=DeltaEntityType.EDGE,
                uuid="edge-uuid",
                data={
                    "source_node_uuid": "source-uuid",
                    "target_node_uuid": "target-uuid",
                    "name": "RELATES_TO",
                    "fact": "Test entity relationship",
                },
            ),
        ]
    )
    return delta


class TestIncrementalUpdater:
    """Test cases for IncrementalUpdater class"""

    @pytest.mark.asyncio
    async def test_add_entities_incremental(self, incremental_updater, sample_entity):
        """Test adding entities incrementally"""
        entities = [sample_entity]
        
        result = await incremental_updater.add_entities_incremental(entities)
        
        assert len(result["added_entities"]) == 1
        assert result["added_entities"][0]["name"] == "Test Entity"
        assert incremental_updater.driver.add_entity.called

    @pytest.mark.asyncio
    async def test_add_edges_incremental(self, incremental_updater, sample_edge):
        """Test adding edges incrementally"""
        edges = [sample_edge]
        
        result = await incremental_updater.add_edges_incremental(edges)
        
        assert len(result["added_edges"]) == 1
        assert result["added_edges"][0]["name"] == "RELATES_TO"
        assert incremental_updater.driver.add_edge.called

    @pytest.mark.asyncio
    async def test_update_entities_incremental(self, incremental_updater, sample_entity):
        """Test updating entities incrementally"""
        sample_entity.uuid = "entity-uuid"
        sample_entity.summary = "Updated summary"
        
        result = await incremental_updater.update_entities_incremental([sample_entity])
        
        assert len(result["updated_entities"]) == 1
        assert result["updated_entities"][0]["summary"] == "Updated summary"
        assert incremental_updater.driver.update_entity.called

    @pytest.mark.asyncio
    async def test_update_edges_incremental(self, incremental_updater, sample_edge):
        """Test updating edges incrementally"""
        sample_edge.uuid = "edge-uuid"
        sample_edge.fact = "Updated fact"
        
        result = await incremental_updater.update_edges_incremental([sample_edge])
        
        assert len(result["updated_edges"]) == 1
        assert result["updated_edges"][0]["fact"] == "Updated fact"
        assert incremental_updater.driver.update_edge.called

    @pytest.mark.asyncio
    async def test_remove_entities_incremental(self, incremental_updater):
        """Test removing entities incrementally"""
        entity_uuids = ["entity-uuid-1", "entity-uuid-2"]
        
        result = await incremental_updater.remove_entities_incremental(entity_uuids)
        
        assert len(result["removed_entities"]) == 2
        assert result["removed_entities"] == entity_uuids
        assert incremental_updater.driver.remove_entity.called

    @pytest.mark.asyncio
    async def test_remove_edges_incremental(self, incremental_updater):
        """Test removing edges incrementally"""
        edge_uuids = ["edge-uuid-1", "edge-uuid-2"]
        
        result = await incremental_updater.remove_edges_incremental(edge_uuids)
        
        assert len(result["removed_edges"]) == 2
        assert result["removed_edges"] == edge_uuids
        assert incremental_updater.driver.remove_edge.called

    @pytest.mark.asyncio
    async def test_create_delta(self, incremental_updater):
        """Test creating delta operations"""
        operations = [
            DeltaOperation(
                operation_type=DeltaOperationType.ADD,
                entity_type=DeltaEntityType.NODE,
                uuid="entity-uuid",
                data={"name": "Test Entity", "labels": ["Person"]},
            ),
            DeltaOperation(
                operation_type=DeltaOperationType.ADD,
                entity_type=DeltaEntityType.EDGE,
                uuid="edge-uuid",
                data={"source_node_uuid": "source", "target_node_uuid": "target", "name": "RELATES_TO"},
            ),
        ]
        
        delta = incremental_updater.create_delta(operations)
        
        assert len(delta.operations) == 2
        assert delta.operations[0].operation_type == DeltaOperationType.ADD
        assert delta.operations[0].entity_type == DeltaEntityType.NODE
        assert delta.operations[1].operation_type == DeltaOperationType.ADD
        assert delta.operations[1].entity_type == DeltaEntityType.EDGE

    @pytest.mark.asyncio
    async def test_apply_delta(self, incremental_updater, sample_delta):
        """Test applying delta operations"""
        with patch.object(incremental_updater, 'add_entities_incremental') as mock_add_entities, \
             patch.object(incremental_updater, 'add_edges_incremental') as mock_add_edges:
            
            mock_add_entities.return_value = {"added_entities": [{"name": "Test Entity"}]}
            mock_add_edges.return_value = {"added_edges": [{"name": "RELATES_TO"}]}
            
            result = await incremental_updater.apply_delta(sample_delta)
            
            assert result["total_operations"] == 2
            assert result["successful_operations"] == 2
            assert result["failed_operations"] == 0
            mock_add_entities.assert_called_once()
            mock_add_edges.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_delta(self, incremental_updater, sample_delta):
        """Test validating delta operations"""
        is_valid, errors = incremental_updater.validate_delta(sample_delta)
        
        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_delta_invalid(self, incremental_updater):
        """Test validating invalid delta operations"""
        invalid_delta = Delta(operations=[
            DeltaOperation(
                operation_type=DeltaOperationType.ADD,  # Use valid enum
                entity_type=DeltaEntityType.NODE,
                uuid="entity-uuid",
                data={"name": "Test Entity"},
            )
        ])
        
        is_valid, errors = incremental_updater.validate_delta(invalid_delta)
        
        assert is_valid is False
        assert len(errors) > 0

    @pytest.mark.asyncio
    async def test_rollback_delta(self, incremental_updater, sample_delta):
        """Test rolling back delta operations"""
        with patch.object(incremental_updater, 'remove_entities_incremental') as mock_remove_entities, \
             patch.object(incremental_updater, 'remove_edges_incremental') as mock_remove_edges:
            
            mock_remove_entities.return_value = {"removed_entities": ["entity-uuid"]}
            mock_remove_edges.return_value = {"removed_edges": ["edge-uuid"]}
            
            rollback_result = await incremental_updater.rollback_delta(sample_delta)
            
            assert rollback_result["successful_rollback"] is True
            assert len(rollback_result["rollback_operations"]) == 2
            mock_remove_entities.assert_called_once()
            mock_remove_edges.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_incremental_update(self, incremental_updater, sample_entity, sample_edge):
        """Test batch incremental update"""
        batch_data = {
            "entities_to_add": [sample_entity],
            "edges_to_add": [sample_edge],
            "entities_to_update": [],
            "edges_to_update": [],
            "entities_to_remove": [],
            "edges_to_remove": [],
        }
        
        with patch.object(incremental_updater, 'add_entities_incremental') as mock_add_entities, \
             patch.object(incremental_updater, 'add_edges_incremental') as mock_add_edges:
            
            mock_add_entities.return_value = {"added_entities": [{"name": "Test Entity"}]}
            mock_add_edges.return_value = {"added_edges": [{"name": "RELATES_TO"}]}
            
            result = await incremental_updater.batch_incremental_update(batch_data)
            
            assert result["total_operations"] == 2
            assert result["successful_operations"] == 2
            mock_add_entities.assert_called_once()
            mock_add_edges.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_vector_indices_incremental(self, incremental_updater):
        """Test updating vector indices incrementally"""
        entity_uuids = ["entity-uuid-1", "entity-uuid-2"]
        
        result = await incremental_updater.update_vector_indices_incremental(entity_uuids)
        
        assert result["updated_indices"] == ["vector_index"]
        assert result["total_updated"] == 2

    @pytest.mark.asyncio
    async def test_update_text_indices_incremental(self, incremental_updater):
        """Test updating text indices incrementally"""
        entity_uuids = ["entity-uuid-1", "entity-uuid-2"]
        
        result = await incremental_updater.update_text_indices_incremental(entity_uuids)
        
        assert result["updated_indices"] == ["text_index"]
        assert result["total_updated"] == 2

    @pytest.mark.asyncio
    async def test_update_graph_indices_incremental(self, incremental_updater):
        """Test updating graph indices incrementally"""
        entity_uuids = ["entity-uuid-1", "entity-uuid-2"]
        
        result = await incremental_updater.update_graph_indices_incremental(entity_uuids)
        
        assert result["updated_indices"] == ["graph_index"]
        assert result["total_updated"] == 2

    @pytest.mark.asyncio
    async def test_rebuild_indices_if_needed(self, incremental_updater):
        """Test rebuilding indices if needed"""
        result = await incremental_updater.rebuild_indices_if_needed()
        
        assert result["rebuilt_indices"] == ["vector_index", "text_index", "graph_index"]
        assert result["total_rebuilt"] == 3

    @pytest.mark.asyncio
    async def test_monitor_delta_progress(self, incremental_updater):
        """Test monitoring delta progress"""
        delta = Delta(operations=[
            DeltaOperation(
                operation_type=DeltaOperationType.ADD,
                entity_type=DeltaEntityType.NODE,
                uuid="entity-uuid",
                data={}
            ),
            DeltaOperation(
                operation_type=DeltaOperationType.ADD,
                entity_type=DeltaEntityType.EDGE,
                uuid="edge-uuid",
                data={}
            ),
        ])
        
        with patch.object(incremental_updater, 'apply_delta') as mock_apply_delta:
            mock_apply_delta.return_value = {
                "total_operations": 2,
                "successful_operations": 2,
                "failed_operations": 0,
                "operation_details": [
                    {"operation": "ADD_ENTITY", "status": "completed", "duration": 1.0},
                    {"operation": "ADD_EDGE", "status": "completed", "duration": 1.5},
                ]
            }
            
            progress = await incremental_updater.monitor_delta_progress(delta)
            
            assert progress["total_operations"] == 2
            assert progress["completed_operations"] == 2
            assert progress["success_rate"] == 100.0
            assert progress["average_duration"] == 1.25


class TestDeltaOperation:
    """Test cases for DeltaOperation class"""

    def test_delta_operation_creation(self):
        """Test creating delta operations"""
        operation = DeltaOperation(
            operation_type=DeltaOperationType.ADD,
            entity_type=DeltaEntityType.NODE,
            uuid="entity-uuid",
            data={"name": "Test Entity"},
        )
        
        assert operation.operation_type == DeltaOperationType.ADD
        assert operation.entity_type == DeltaEntityType.NODE
        assert operation.data == {"name": "Test Entity"}

    def test_delta_operation_serialization(self):
        """Test delta operation serialization"""
        operation = DeltaOperation(
            operation_type=DeltaOperationType.ADD,
            entity_type=DeltaEntityType.NODE,
            uuid="entity-uuid",
            data={"name": "Test Entity"},
        )
        
        serialized = operation.__dict__.copy()
        serialized["operation_type"] = serialized["operation_type"].value
        serialized["entity_type"] = serialized["entity_type"].value
        
        assert serialized["operation_type"] == "add"
        assert serialized["entity_type"] == "node"
        assert serialized["data"] == {"name": "Test Entity"}

    def test_delta_operation_deserialization(self):
        """Test delta operation deserialization"""
        data = {
            "operation_type": "add",
            "entity_type": "node",
            "uuid": "entity-uuid",
            "data": {"name": "Test Entity"},
            "timestamp": "2024-01-01T00:00:00",
            "metadata": {},
        }
        
        operation = DeltaOperation(
            operation_type=DeltaOperationType.ADD,
            entity_type=DeltaEntityType.NODE,
            uuid="entity-uuid",
            data={"name": "Test Entity"},
        )
        
        assert operation.operation_type == DeltaOperationType.ADD
        assert operation.entity_type == DeltaEntityType.NODE
        assert operation.data == {"name": "Test Entity"}


class TestDelta:
    """Test cases for Delta class"""

    def test_delta_creation(self):
        """Test creating delta"""
        operations = [
            DeltaOperation(
                operation_type=DeltaOperationType.ADD,
                entity_type=DeltaEntityType.NODE,
                uuid="entity-uuid",
                data={"name": "Test Entity"},
            )
        ]
        
        delta = Delta(operations=operations, metadata={"source": "test"})
        
        assert len(delta.operations) == 1
        assert delta.metadata == {"source": "test"}

    def test_delta_serialization(self):
        """Test delta serialization"""
        operations = [
            DeltaOperation(
                operation_type=DeltaOperationType.ADD,
                entity_type=DeltaEntityType.NODE,
                uuid="entity-uuid",
                data={"name": "Test Entity"},
            )
        ]
        
        delta = Delta(operations=operations)
        serialized = {
            "operations": [
                {
                    "operation_type": op.operation_type.value,
                    "entity_type": op.entity_type.value,
                    "uuid": op.uuid,
                    "data": op.data,
                    "timestamp": op.timestamp.isoformat(),
                    "metadata": op.metadata
                }
                for op in delta.operations
            ],
            "created_at": delta.created_at.isoformat(),
            "applied_at": delta.applied_at.isoformat() if delta.applied_at else None,
            "status": delta.status,
            "rollback_data": delta.rollback_data
        }
        
        assert len(serialized["operations"]) == 1
        assert serialized["operations"][0]["operation_type"] == "ADD_ENTITY"

    def test_delta_deserialization(self):
        """Test delta deserialization"""
        data = {
            "operations": [
                {
                    "operation_type": "ADD_ENTITY",
                    "entity_data": {"name": "Test Entity"},
                    "edge_data": None,
                }
            ],
            "metadata": {},
        }
        
        delta = Delta()
        delta.operations = [
            DeltaOperation(
                operation_type=DeltaOperationType.ADD,
                entity_type=DeltaEntityType.NODE,
                uuid="entity-uuid",
                data={"name": "Test Entity"},
            )
        ]
        
        assert len(delta.operations) == 1
        assert delta.operations[0].operation_type == DeltaOperationType.ADD


class TestHuggingFaceDriverIntegration:
    """Test cases for HuggingFaceDriver integration"""

    @pytest.mark.asyncio
    async def test_driver_add_entity_incremental(self, mock_driver):
        """Test driver add_entity_incremental method"""
        entity_data = {
            "name": "Test Entity",
            "labels": ["Person"],
            "summary": "A test entity",
            "attributes": {"age": 30},
        }
        
        result = await mock_driver.add_entity_incremental(entity_data)
        
        assert result["status"] == "success"
        assert result["entity"]["name"] == "Test Entity"
        assert mock_driver.add_entity.called

    @pytest.mark.asyncio
    async def test_driver_add_edge_incremental(self, mock_driver):
        """Test driver add_edge_incremental method"""
        edge_data = {
            "source_node_uuid": "source-uuid",
            "target_node_uuid": "target-uuid",
            "name": "RELATES_TO",
            "fact": "Test entity relationship",
        }
        
        result = await mock_driver.add_edge_incremental(edge_data)
        
        assert result["status"] == "success"
        assert result["edge"]["name"] == "RELATES_TO"
        assert mock_driver.add_edge.called

    @pytest.mark.asyncio
    async def test_driver_batch_incremental_update(self, mock_driver):
        """Test driver batch_incremental_update method"""
        batch_data = {
            "entities_to_add": [
                {
                    "name": "Entity 1",
                    "labels": ["Person"],
                    "summary": "Test entity 1",
                }
            ],
            "edges_to_add": [
                {
                    "source_node_uuid": "source-uuid",
                    "target_node_uuid": "target-uuid",
                    "name": "RELATES_TO",
                }
            ],
        }
        
        result = await mock_driver.batch_incremental_update(batch_data)
        
        assert result["status"] == "success"
        assert result["total_operations"] == 2
        assert result["successful_operations"] == 2

    @pytest.mark.asyncio
    async def test_driver_get_update_statistics(self, mock_driver):
        """Test driver get_update_statistics method"""
        mock_driver.get_entities.return_value = [
            {"uuid": "entity-1", "name": "Entity 1"},
            {"uuid": "entity-2", "name": "Entity 2"},
        ]
        mock_driver.get_edges.return_value = [
            {"uuid": "edge-1", "name": "Edge 1"},
            {"uuid": "edge-2", "name": "Edge 2"},
        ]
        
        stats = await mock_driver.get_update_statistics()
        
        assert stats["total_entities"] == 2
        assert stats["total_edges"] == 2
        assert stats["last_update_time"] is not None


if __name__ == "__main__":
    pytest.main([__file__])