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

"""
Tests for ConcurrencyManager functionality
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from graphiti_hf.processing.concurrency_manager import (
    ConcurrencyManager,
    BranchInfo,
    BranchStatus,
    MergeResult,
    MergeStrategy,
    Transaction,
    TransactionStatus,
    ConflictType,
    VersionInfo
)
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge


class TestConcurrencyManager:
    """Test cases for ConcurrencyManager"""
    
    @pytest.fixture
    def concurrency_manager(self):
        """Create a ConcurrencyManager instance for testing"""
        with patch('graphiti_hf.processing.concurrency_manager.HfApi'):
            manager = ConcurrencyManager(
                repo_id="test/repo",
                token="test-token",
                default_branch="main"
            )
            return manager
    
    @pytest.fixture
    def sample_entity(self):
        """Create a sample entity for testing"""
        return EntityNode(
            uuid="test-entity-1",
            name="Test Entity",
            labels=["test"],
            group_id="test-group",
            created_at=datetime.now(),
            valid_at=datetime.now()
        )
    
    @pytest.fixture
    def sample_edge(self):
        """Create a sample edge for testing"""
        return EntityEdge(
            uuid="test-edge-1",
            source_node_uuid="source-entity",
            target_node_uuid="target-entity",
            fact="Test relationship",
            group_id="test-group",
            created_at=datetime.now(),
            valid_at=datetime.now()
        )
    
    # Branch Management Tests
    
    @pytest.mark.asyncio
    async def test_create_branch(self, concurrency_manager):
        """Test creating a new branch"""
        branch_info = await concurrency_manager.create_branch(
            "test-branch",
            description="Test branch"
        )
        
        assert branch_info.name == "test-branch"
        assert branch_info.status == BranchStatus.ACTIVE
        assert branch_info.parent_branch == "main"
        assert branch_info.description == "Test branch"
        assert "test-branch" in concurrency_manager.branches
    
    @pytest.mark.asyncio
    async def test_create_branch_already_exists(self, concurrency_manager):
        """Test creating a branch that already exists"""
        await concurrency_manager.create_branch("existing-branch")
        
        with pytest.raises(ValueError, match="Branch 'existing-branch' already exists"):
            await concurrency_manager.create_branch("existing-branch")
    
    @pytest.mark.asyncio
    async def test_list_branches(self, concurrency_manager):
        """Test listing branches"""
        # Create some branches
        await concurrency_manager.create_branch("branch1")
        await concurrency_manager.create_branch("branch2")
        
        branches = await concurrency_manager.list_branches()
        
        assert len(branches) >= 2
        branch_names = [b.name for b in branches]
        assert "main" in branch_names
        assert "branch1" in branch_names
        assert "branch2" in branch_names
    
    @pytest.mark.asyncio
    async def test_switch_branch(self, concurrency_manager):
        """Test switching between branches"""
        # Create a new branch
        await concurrency_manager.create_branch("test-branch")
        
        # Switch to it
        branch_info = await concurrency_manager.switch_branch("test-branch")
        
        assert branch_info.name == "test-branch"
        assert concurrency_manager.current_branch == "test-branch"
    
    @pytest.mark.asyncio
    async def test_switch_to_nonexistent_branch(self, concurrency_manager):
        """Test switching to a non-existent branch"""
        with pytest.raises(ValueError, match="Branch 'nonexistent' does not exist"):
            await concurrency_manager.switch_branch("nonexistent")
    
    @pytest.mark.asyncio
    async def test_delete_branch(self, concurrency_manager):
        """Test deleting a branch"""
        await concurrency_manager.create_branch("to-delete")
        
        result = await concurrency_manager.delete_branch("to-delete")
        
        assert result is True
        assert concurrency_manager.branches["to-delete"].status == BranchStatus.DELETED
    
    @pytest.mark.asyncio
    async def test_delete_default_branch(self, concurrency_manager):
        """Test deleting the default branch (should fail)"""
        with pytest.raises(ValueError, match="Cannot delete the default branch"):
            await concurrency_manager.delete_branch("main")
    
    # Version Management Tests
    
    def test_get_version(self, concurrency_manager):
        """Test getting version information"""
        # Create a version
        version = concurrency_manager._create_version("main", "test-user", {"test": "data"})
        
        # Get it back
        retrieved = concurrency_manager.get_version("test-entity", "main")
        
        assert retrieved is not None
        assert retrieved.version == version.version
        assert retrieved.branch == "main"
    
    def test_get_version_not_found(self, concurrency_manager):
        """Test getting version for non-existent entity"""
        version = concurrency_manager.get_version("nonexistent-entity")
        
        assert version is None
    
    # Transaction Management Tests
    
    @pytest.mark.asyncio
    async def test_begin_transaction(self, concurrency_manager):
        """Test starting a transaction"""
        tx_id = await concurrency_manager.begin_transaction("test-user", "main")
        
        assert tx_id is not None
        assert tx_id in concurrency_manager.transactions
        assert tx_id in concurrency_manager.active_transactions
        assert concurrency_manager.transactions[tx_id].status == TransactionStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_commit_transaction(self, concurrency_manager):
        """Test committing a transaction"""
        tx_id = await concurrency_manager.begin_transaction("test-user")
        
        success = await concurrency_manager.commit_transaction(tx_id)
        
        assert success is True
        assert concurrency_manager.transactions[tx_id].status == TransactionStatus.COMMITTED
        assert tx_id not in concurrency_manager.active_transactions
    
    @pytest.mark.asyncio
    async def test_rollback_transaction(self, concurrency_manager):
        """Test rolling back a transaction"""
        tx_id = await concurrency_manager.begin_transaction("test-user")
        
        success = await concurrency_manager.rollback_transaction(tx_id)
        
        assert success is True
        assert concurrency_manager.transactions[tx_id].status == TransactionStatus.ROLLED_BACK
        assert tx_id not in concurrency_manager.active_transactions
    
    @pytest.mark.asyncio
    async def test_get_transaction_status(self, concurrency_manager):
        """Test getting transaction status"""
        tx_id = await concurrency_manager.begin_transaction("test-user")
        
        transaction = await concurrency_manager.get_transaction_status(tx_id)
        
        assert transaction is not None
        assert transaction.id == tx_id
        assert transaction.status == TransactionStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_list_transactions(self, concurrency_manager):
        """Test listing transactions"""
        # Create some transactions
        tx1 = await concurrency_manager.begin_transaction("user1")
        tx2 = await concurrency_manager.begin_transaction("user2")
        
        transactions = await concurrency_manager.list_transactions()
        
        assert len(transactions) >= 2
        tx_ids = [t.id for t in transactions]
        assert tx1 in tx_ids
        assert tx2 in tx_ids
    
    # Merge Strategy Tests
    
    @pytest.mark.asyncio
    async def test_auto_merge_strategy(self, concurrency_manager):
        """Test auto merge strategy"""
        conflict = Mock()
        conflict.conflict_type = ConflictType.NODE_CONFLICT
        conflict.conflict_data = {
            "current_data": {"name": "Current"},
            "incoming_data": {"name": "Incoming"}
        }
        
        result = await concurrency_manager._auto_merge(conflict)
        
        assert result["name"] == "Incoming"  # Auto merge should prefer incoming
    
    @pytest.mark.asyncio
    async def test_manual_merge_strategy(self, concurrency_manager):
        """Test manual merge strategy"""
        conflict = Mock()
        
        result = await concurrency_manager._manual_merge(conflict)
        
        assert result == {}  # Manual merge returns empty dict for user intervention
    
    @pytest.mark.asyncio
    async def test_timestamp_based_merge(self, concurrency_manager):
        """Test timestamp-based merge strategy"""
        conflict = Mock()
        conflict.conflict_data = {
            "current_data": {"updated_at": datetime(2024, 1, 1)},
            "incoming_data": {"updated_at": datetime(2024, 1, 2)}
        }
        
        result = await concurrency_manager._timestamp_based_merge(conflict)
        
        # Should return the incoming data (newer timestamp)
        assert result["updated_at"] == datetime(2024, 1, 2)
    
    # Conflict Detection Tests
    
    @pytest.mark.asyncio
    async def test_detect_conflicts(self, concurrency_manager):
        """Test conflict detection"""
        source_data = {
            "nodes": [
                {"uuid": "entity1", "name": "Source Entity", "entity_type": "node"}
            ],
            "edges": []
        }
        
        target_data = {
            "nodes": [
                {"uuid": "entity1", "name": "Target Entity", "entity_type": "node"}
            ],
            "edges": []
        }
        
        conflicts = await concurrency_manager._detect_conflicts(source_data, target_data)
        
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.NODE_CONFLICT
        assert conflicts[0].entity_uuid == "entity1"
    
    @pytest.mark.asyncio
    async def test_no_conflicts(self, concurrency_manager):
        """Test scenario with no conflicts"""
        source_data = {
            "nodes": [
                {"uuid": "entity1", "name": "Entity 1", "entity_type": "node"}
            ],
            "edges": []
        }
        
        target_data = {
            "nodes": [
                {"uuid": "entity2", "name": "Entity 2", "entity_type": "node"}
            ],
            "edges": []
        }
        
        conflicts = await concurrency_manager._detect_conflicts(source_data, target_data)
        
        assert len(conflicts) == 0
    
    # Statistics Tests
    
    @pytest.mark.asyncio
    async def test_get_concurrency_stats(self, concurrency_manager):
        """Test getting concurrency statistics"""
        # Create some test data
        await concurrency_manager.create_branch("test-branch")
        await concurrency_manager.begin_transaction("test-user")
        
        stats = await concurrency_manager.get_concurrency_stats()
        
        assert stats["total_branches"] >= 2
        assert stats["active_branches"] >= 1
        assert stats["current_branch"] == "main"
        assert stats["total_versions"] >= 1
        assert stats["active_transactions"] >= 1
        assert stats["total_transactions"] >= 1


class TestConcurrencyIntegration:
    """Integration tests for ConcurrencyManager with HuggingFaceDriver"""
    
    @pytest.mark.asyncio
    async def test_driver_concurrency_integration(self):
        """Test that HuggingFaceDriver properly integrates with ConcurrencyManager"""
        with patch('graphiti_hf.drivers.huggingface_driver.HfApi'):
            from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver
            
            driver = HuggingFaceDriver(
                repo_id="test/integration",
                create_repo=True,
                enable_vector_search=False
            )
            
            # Test that concurrency manager is initialized
            assert driver.concurrency_manager is not None
            assert driver.concurrency_manager.repo_id == "test/integration"
            
            # Test branch management through driver
            branch_info = await driver.create_branch("test-branch")
            assert branch_info.name == "test-branch"
            
            # Test transaction management through driver
            tx_id = await driver.begin_transaction("test-user")
            assert tx_id is not None
            
            # Test transaction status through driver
            transaction = await driver.get_transaction_status(tx_id)
            assert transaction is not None
            
            driver.close()


class TestMergeStrategies:
    """Test specific merge strategy implementations"""
    
    @pytest.mark.asyncio
    async def test_resolve_node_conflict(self, concurrency_manager):
        """Test node conflict resolution"""
        conflict = Mock()
        conflict.conflict_data = {
            "current_data": {
                "uuid": "entity1",
                "name": "Current Entity",
                "properties": {"color": "red"},
                "created_at": datetime(2024, 1, 1)
            },
            "incoming_data": {
                "uuid": "entity1", 
                "name": "Updated Entity",
                "properties": {"size": "large"}
            }
        }
        
        result = await concurrency_manager._resolve_node_conflict(conflict)
        
        assert result["name"] == "Updated Entity"  # From incoming
        assert result["properties"]["color"] == "red"  # From current (preserved)
        assert result["properties"]["size"] == "large"  # From incoming (added)
        assert result["created_at"] == datetime(2024, 1, 1)  # From current (preserved)
    
    @pytest.mark.asyncio
    async def test_resolve_edge_conflict(self, concurrency_manager):
        """Test edge conflict resolution"""
        conflict = Mock()
        conflict.conflict_data = {
            "current_data": {
                "uuid": "edge1",
                "fact": "Old relationship"
            },
            "incoming_data": {
                "uuid": "edge1",
                "fact": "New relationship"
            }
        }
        
        result = await concurrency_manager._resolve_edge_conflict(conflict)
        
        assert result["fact"] == "New relationship"  # Incoming wins for edges
    
    @pytest.mark.asyncio
    async def test_resolve_timestamp_conflict(self, concurrency_manager):
        """Test timestamp-based conflict resolution"""
        conflict = Mock()
        conflict.conflict_data = {
            "current_data": {
                "updated_at": datetime(2024, 1, 1, 10, 0, 0)
            },
            "incoming_data": {
                "updated_at": datetime(2024, 1, 1, 10, 0, 1)  # 1 second newer
            }
        }
        
        result = await concurrency_manager._resolve_timestamp_conflict(conflict)
        
        # Should return the incoming data (newer timestamp)
        assert result["updated_at"] == datetime(2024, 1, 1, 10, 0, 1)


class TestErrorHandling:
    """Test error handling in ConcurrencyManager"""
    
    @pytest.mark.asyncio
    async def test_merge_nonexistent_branches(self, concurrency_manager):
        """Test merging non-existent branches"""
        with pytest.raises(ValueError, match="Source branch 'nonexistent' does not exist"):
            await concurrency_manager.merge_branch("nonexistent", "main")
    
    @pytest.mark.asyncio
    async def test_merge_same_branch(self, concurrency_manager):
        """Test merging a branch into itself"""
        with pytest.raises(ValueError, match="Cannot merge branch into itself"):
            await concurrency_manager.merge_branch("main", "main")
    
    @pytest.mark.asyncio
    async def test_commit_nonexistent_transaction(self, concurrency_manager):
        """Test committing a non-existent transaction"""
        with pytest.raises(ValueError, match="Transaction nonexistent not found"):
            await concurrency_manager.commit_transaction("nonexistent")
    
    @pytest.mark.asyncio
    async def test_rollback_nonexistent_transaction(self, concurrency_manager):
        """Test rolling back a non-existent transaction"""
        with pytest.raises(ValueError, match="Transaction nonexistent not found"):
            await concurrency_manager.rollback_transaction("nonexistent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])