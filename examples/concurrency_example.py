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
Concurrent Writes through Dataset Versioning - Example

This example demonstrates how to use the ConcurrencyManager to handle
multiple simultaneous updates to HuggingFace datasets with proper
branch management, optimistic locking, and conflict resolution.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver
from graphiti_hf.processing.concurrency_manager import (
    ConcurrencyManager, MergeStrategy, TransactionStatus
)
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_branch_management_example():
    """Demonstrate basic branch management operations"""
    logger.info("=== Basic Branch Management Example ===")
    
    # Initialize driver with concurrency management
    driver = HuggingFaceDriver(
        repo_id="test/concurrency-branch-example",
        create_repo=True,
        enable_vector_search=False  # Disable for simplicity
    )
    
    try:
        # Create development branch
        logger.info("Creating development branch...")
        dev_branch = await driver.create_branch(
            "development",
            description="Development branch for testing"
        )
        logger.info(f"Created branch: {dev_branch.name} (status: {dev_branch.status.value})")
        
        # Create feature branch
        logger.info("Creating feature branch...")
        feature_branch = await driver.create_branch(
            "feature-experiment",
            parent_branch="development",
            description="Branch for experimental features"
        )
        logger.info(f"Created branch: {feature_branch.name}")
        
        # List all branches
        logger.info("Listing all branches...")
        branches = await driver.list_branches()
        for branch in branches:
            logger.info(f"  - {branch.name}: {branch.status.value} (created: {branch.created_at})")
        
        # Switch to development branch
        logger.info("Switching to development branch...")
        switched_branch = await driver.switch_branch("development")
        logger.info(f"Switched to: {switched_branch.name}")
        
        # Add some data to development branch
        logger.info("Adding data to development branch...")
        entity1 = EntityNode(
            uuid="dev-entity-1",
            name="Development Entity 1",
            labels=["development", "test"],
            group_id="dev-group",
            created_at=datetime.now(),
            valid_at=datetime.now()
        )
        await driver.add_entity_incremental(entity1)
        
        # Switch to feature branch
        logger.info("Switching to feature branch...")
        await driver.switch_branch("feature-experiment")
        
        # Add different data to feature branch
        logger.info("Adding data to feature branch...")
        entity2 = EntityNode(
            uuid="feature-entity-1",
            name="Feature Entity 1",
            labels=["feature", "experimental"],
            group_id="feature-group",
            created_at=datetime.now(),
            valid_at=datetime.now()
        )
        await driver.add_entity_incremental(entity2)
        
        # Merge feature branch back to development
        logger.info("Merging feature branch to development...")
        merge_result = await driver.merge_branch(
            "feature-experiment",
            "development",
            "auto"
        )
        logger.info(f"Merge result: {merge_result.success}")
        logger.info(f"Conflicts resolved: {merge_result.resolved_conflicts}")
        
        # List branches again
        logger.info("Branches after merge:")
        branches = await driver.list_branches()
        for branch in branches:
            logger.info(f"  - {branch.name}: {branch.status.value}")
        
    finally:
        driver.close()


async def optimistic_locking_example():
    """Demonstrate optimistic locking with version checking"""
    logger.info("=== Optimistic Locking Example ===")
    
    driver = HuggingFaceDriver(
        repo_id="test/concurrency-locking-example",
        create_repo=True,
        enable_vector_search=False
    )
    
    try:
        # Add an entity
        entity = EntityNode(
            uuid="locked-entity-1",
            name="Optimistically Locked Entity",
            labels=["locked"],
            group_id="lock-group",
            created_at=datetime.now(),
            valid_at=datetime.now()
        )
        await driver.add_entity_incremental(entity)
        
        # Get current version
        version = await driver.get_version("locked-entity-1")
        logger.info(f"Current version: {version.version if version else 'None'}")
        
        # Simulate concurrent update with version checking
        logger.info("Simulating concurrent update...")
        try:
            # This should succeed (no conflict)
            result = await driver.apply_with_lock(
                "locked-entity-1",
                {"name": "Updated Entity Name", "updated_by": "user1"},
                "main"
            )
            logger.info(f"Update successful, new version: {result.version}")
            
            # Now try to update with old version (should fail)
            logger.info("Attempting update with old version...")
            conflict = await driver.check_version_conflict(
                "locked-entity-1",
                version.version if version else 1  # Old version
            )
            if conflict:
                logger.warning("Version conflict detected as expected!")
            else:
                logger.error("Expected conflict not detected")
                
        except Exception as e:
            logger.error(f"Optimistic locking error: {e}")
        
    finally:
        driver.close()


async def transaction_management_example():
    """Demonstrate transaction management for atomic operations"""
    logger.info("=== Transaction Management Example ===")
    
    driver = HuggingFaceDriver(
        repo_id="test/concurrency-transaction-example",
        create_repo=True,
        enable_vector_search=False
    )
    
    try:
        # Start a transaction
        logger.info("Starting transaction...")
        transaction_id = await driver.begin_transaction(
            user_id="test-user",
            branch="main"
        )
        logger.info(f"Transaction started: {transaction_id}")
        
        # Add multiple entities in the transaction
        logger.info("Adding entities in transaction...")
        entities = []
        for i in range(3):
            entity = EntityNode(
                uuid=f"tx-entity-{i}",
                name=f"Transaction Entity {i}",
                labels=["transaction"],
                group_id="tx-group",
                created_at=datetime.now(),
                valid_at=datetime.now()
            )
            entities.append(entity)
            await driver.add_entity_incremental(entity)
        
        # Check transaction status
        transaction = await driver.get_transaction_status(transaction_id)
        if transaction:
            logger.info(f"Transaction status: {transaction.status.value}")
            logger.info(f"Operations in transaction: {len(transaction.operations)}")
        else:
            logger.error("Transaction not found")
        
        # Commit the transaction
        logger.info("Committing transaction...")
        success = await driver.commit_transaction(transaction_id)
        logger.info(f"Transaction committed: {success}")
        
        # List active transactions
        logger.info("Active transactions:")
        active_transactions = await driver.list_transactions(status="active")
        for tx in active_transactions:
            logger.info(f"  - {tx.id}: {tx.status.value}")
        
        # Start another transaction and rollback
        logger.info("Starting transaction for rollback...")
        tx_id = await driver.begin_transaction(user_id="test-user")
        
        # Add an entity
        rollback_entity = EntityNode(
            uuid="rollback-entity",
            name="Will Be Rolled Back",
            labels=["rollback"],
            group_id="rollback-group",
            created_at=datetime.now(),
            valid_at=datetime.now()
        )
        await driver.add_entity_incremental(rollback_entity)
        
        # Rollback the transaction
        logger.info("Rolling back transaction...")
        rollback_success = await driver.rollback_transaction(tx_id)
        logger.info(f"Transaction rolled back: {rollback_success}")
        
    finally:
        driver.close()


async def merge_strategies_example():
    """Demonstrate different merge strategies"""
    logger.info("=== Merge Strategies Example ===")
    
    driver = HuggingFaceDriver(
        repo_id="test/concurrency-merge-example",
        create_repo=True,
        enable_vector_search=False
    )
    
    try:
        # Create source and target branches
        logger.info("Creating source and target branches...")
        await driver.create_branch("source", description="Source branch with changes")
        await driver.create_branch("target", description="Target branch to merge into")
        
        # Add different data to each branch
        # Source branch
        await driver.switch_branch("source")
        source_entity = EntityNode(
            uuid="shared-entity",
            name="Source Version",
            labels=["source"],
            group_id="shared-group",
            created_at=datetime.now(),
            valid_at=datetime.now()
        )
        await driver.add_entity_incremental(source_entity)
        
        # Target branch
        await driver.switch_branch("target")
        target_entity = EntityNode(
            uuid="shared-entity",
            name="Target Version",
            labels=["target"],
            group_id="shared-group",
            created_at=datetime.now(),
            valid_at=datetime.now()
        )
        await driver.add_entity_incremental(target_entity)
        
        # Test different merge strategies
        strategies = ["auto", "manual", "timestamp", "priority"]
        
        for strategy in strategies:
            logger.info(f"\nTesting {strategy} merge strategy...")
            
            # Create fresh branches for each test
            await driver.create_branch(f"source-{strategy}")
            await driver.create_branch(f"target-{strategy}")
            
            # Add conflicting data
            await driver.switch_branch(f"source-{strategy}")
            await driver.add_entity_incremental(source_entity)
            
            await driver.switch_branch(f"target-{strategy}")
            await driver.add_entity_incremental(target_entity)
            
            # Attempt merge
            try:
                merge_result = await driver.merge_branch(
                    f"source-{strategy}",
                    f"target-{strategy}",
                    strategy
                )
                logger.info(f"  {strategy} merge: {merge_result.success}")
                logger.info(f"  Conflicts: {len(merge_result.conflicts)}")
                logger.info(f"  Resolved: {merge_result.resolved_conflicts}")
                
            except Exception as e:
                logger.error(f"  {strategy} merge failed: {e}")
        
    finally:
        driver.close()


async def concurrent_users_simulation():
    """Simulate multiple users working concurrently"""
    logger.info("=== Concurrent Users Simulation ===")
    
    driver = HuggingFaceDriver(
        repo_id="test/concurrency-users-example",
        create_repo=True,
        enable_vector_search=False
    )
    
    try:
        # Create a shared entity
        shared_entity = EntityNode(
            uuid="shared-entity",
            name="Shared Entity",
            labels=["shared"],
            group_id="shared-group",
            created_at=datetime.now(),
            valid_at=datetime.now()
        )
        await driver.add_entity_incremental(shared_entity)
        
        # Simulate multiple users updating the same entity
        async def user_update(user_id: str, update_data: Dict[str, Any]):
            """Simulate a user updating the entity"""
            logger.info(f"User {user_id} attempting update...")
            
            try:
                # Get current version
                version = await driver.get_version("shared-entity")
                current_version = version.version if version else 1
                
                # Apply update with optimistic locking
                result = await driver.apply_with_lock(
                    "shared-entity",
                    update_data,
                    "main"
                )
                
                logger.info(f"User {user_id} updated successfully to version {result.version}")
                return True
                
            except Exception as e:
                logger.error(f"User {user_id} failed: {e}")
                return False
        
        # Simulate concurrent updates
        updates = [
            {"user_id": "alice", "data": {"name": "Entity Updated by Alice"}},
            {"user_id": "bob", "data": {"name": "Entity Updated by Bob"}},
            {"user_id": "charlie", "data": {"name": "Entity Updated by Charlie"}},
        ]
        
        # Run updates concurrently
        logger.info("Running concurrent updates...")
        tasks = [user_update(update["user_id"], update["data"]) for update in updates]
        results = await asyncio.gather(*tasks)
        
        # Check results
        successful = sum(results)
        logger.info(f"Successful updates: {successful}/{len(results)}")
        
        # Get final entity state
        final_entity = None
        for _, row in driver.nodes_df.iterrows():
            if row["uuid"] == "shared-entity":
                final_entity = row
                break
        
        if final_entity is not None:
            logger.info(f"Final entity name: {final_entity['name']}")
        
    finally:
        driver.close()


async def performance_monitoring_example():
    """Demonstrate concurrency performance monitoring"""
    logger.info("=== Performance Monitoring Example ===")
    
    driver = HuggingFaceDriver(
        repo_id="test/concurrency-performance-example",
        create_repo=True,
        enable_vector_search=False
    )
    
    try:
        # Perform various operations to generate statistics
        logger.info("Generating concurrency statistics...")
        
        # Create multiple branches
        for i in range(5):
            await driver.create_branch(f"branch-{i}")
        
        # Add entities to different branches
        for i in range(10):
            entity = EntityNode(
                uuid=f"perf-entity-{i}",
                name=f"Performance Entity {i}",
                labels=["performance"],
                group_id="perf-group",
                created_at=datetime.now(),
                valid_at=datetime.now()
            )
            await driver.add_entity_incremental(entity)
        
        # Start multiple transactions
        transaction_ids = []
        for i in range(3):
            tx_id = await driver.begin_transaction(user_id=f"user-{i}")
            transaction_ids.append(tx_id)
        
        # Get concurrency statistics
        stats = await driver.get_concurrency_stats()
        logger.info("=== Concurrency Statistics ===")
        logger.info(f"Total branches: {stats['total_branches']}")
        logger.info(f"Active branches: {stats['active_branches']}")
        logger.info(f"Current branch: {stats['current_branch']}")
        logger.info(f"Total versions: {stats['total_versions']}")
        logger.info(f"Active transactions: {stats['active_transactions']}")
        logger.info(f"Total transactions: {stats['total_transactions']}")
        
        # List branches
        logger.info("Active branches:")
        for branch in stats['branches']:
            logger.info(f"  - {branch['name']}: {branch['status']} (v{branch['version']})")
        
        # Commit transactions
        for tx_id in transaction_ids:
            await driver.commit_transaction(tx_id)
        
        # Get final statistics
        final_stats = await driver.get_concurrency_stats()
        logger.info("=== Final Statistics ===")
        logger.info(f"Active transactions: {final_stats['active_transactions']}")
        
    finally:
        driver.close()


async def main():
    """Run all concurrency examples"""
    logger.info("Starting Graphiti-HF Concurrency Examples")
    
    try:
        await basic_branch_management_example()
        await asyncio.sleep(1)
        
        await optimistic_locking_example()
        await asyncio.sleep(1)
        
        await transaction_management_example()
        await asyncio.sleep(1)
        
        await merge_strategies_example()
        await asyncio.sleep(1)
        
        await concurrent_users_simulation()
        await asyncio.sleep(1)
        
        await performance_monitoring_example()
        
        logger.info("All concurrency examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())