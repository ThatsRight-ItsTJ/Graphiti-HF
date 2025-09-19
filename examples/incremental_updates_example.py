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
Example: Incremental Updates with Graphiti-HF

This example demonstrates how to use the incremental update capabilities
of the Graphiti-HF system to efficiently maintain and evolve knowledge graphs
without full rebuilds.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_incremental_updates():
    """Demonstrate basic incremental update operations"""
    logger.info("=== Basic Incremental Updates Example ===")
    
    # Initialize driver (using a local test repo)
    driver = HuggingFaceDriver(
        repo_id="test/incremental-updates-example",
        create_repo=True,
        enable_vector_search=True
    )
    
    try:
        # Create some test entities
        entity1 = EntityNode(
            uuid="entity-1",
            name="Machine Learning",
            labels=["topic", "technology"],
            group_id="test-group",
            created_at=datetime.now(),
            valid_at=datetime.now(),
            summary="Field of study focused on algorithms and statistical models"
        )
        
        entity2 = EntityNode(
            uuid="entity-2", 
            name="Deep Learning",
            labels=["topic", "technology", "subfield"],
            group_id="test-group",
            created_at=datetime.now(),
            valid_at=datetime.now(),
            summary="Subset of machine learning using neural networks"
        )
        
        # Add entities incrementally
        logger.info("Adding entities incrementally...")
        result1 = await driver.add_entity_incremental(entity1)
        logger.info(f"Add entity 1 result: {result1}")
        
        result2 = await driver.add_entity_incremental(entity2)
        logger.info(f"Add entity 2 result: {result2}")
        
        # Create a relationship between entities
        edge1 = EntityEdge(
            uuid="edge-1",
            source_node_uuid="entity-1",
            target_node_uuid="entity-2",
            fact="Deep learning is a subfield of machine learning",
            group_id="test-group",
            created_at=datetime.now(),
            valid_at=datetime.now()
        )
        
        # Add edge incrementally
        logger.info("Adding edge incrementally...")
        edge_result = await driver.add_edge_incremental(edge1)
        logger.info(f"Add edge result: {edge_result}")
        
        # Update an entity
        entity1_updated = EntityNode(
            uuid="entity-1",
            name="Machine Learning & AI",
            labels=["topic", "technology", "field"],
            group_id="test-group",
            created_at=datetime.now(),
            valid_at=datetime.now(),
            summary="Field of study focused on algorithms and statistical models for artificial intelligence"
        )
        
        logger.info("Updating entity incrementally...")
        update_result = await driver.upsert_entity(entity1_updated)
        logger.info(f"Update entity result: {update_result}")
        
        # Get update statistics
        stats = driver.get_update_statistics()
        logger.info(f"Update statistics: {stats}")
        
    finally:
        # Clean up
        driver.close()


async def delta_management_example():
    """Demonstrate delta management operations"""
    logger.info("=== Delta Management Example ===")
    
    driver = HuggingFaceDriver(
        repo_id="test/delta-management-example",
        create_repo=True,
        enable_vector_search=True
    )
    
    try:
        # Create a delta with multiple operations
        operations = [
            {
                "type": "add",
                "entity_type": "node",
                "data": {
                    "name": "Natural Language Processing",
                    "labels": ["topic", "technology", "subfield"],
                    "group_id": "test-group"
                }
            },
            {
                "type": "add", 
                "entity_type": "node",
                "data": {
                    "name": "Computer Vision",
                    "labels": ["topic", "technology", "subfield"],
                    "group_id": "test-group"
                }
            },
            {
                "type": "add",
                "entity_type": "edge",
                "data": {
                    "source_uuid": "entity-1",  # Machine Learning
                    "target_uuid": "entity-3",  # NLP
                    "fact": "Natural Language Processing is a subfield of Machine Learning",
                    "group_id": "test-group"
                }
            }
        ]
        
        # Create delta
        logger.info("Creating delta...")
        delta = await driver.create_delta(operations)
        # The delta ID is generated when applying the delta
        logger.info(f"Created delta with {len(delta.operations)} operations")
        logger.info(f"Delta contains {len(delta.operations)} operations")
        
        # Apply delta
        logger.info("Applying delta...")
        apply_result = await driver.apply_delta(delta, validate=True)
        logger.info(f"Delta application result: {apply_result}")
        
        # Monitor delta progress (if supported)
        if hasattr(driver.incremental_updater, 'monitor_delta_progress'):
            # For demonstration, we'll use the first operation's UUID as a placeholder
            # In practice, you'd get the actual delta_id from the apply_delta result
            placeholder_id = f"delta-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            progress = await driver.monitor_delta_progress(placeholder_id)
            logger.info(f"Delta progress: {progress}")
        
        # Create another delta for rollback demonstration
        rollback_operations = [
            {
                "type": "add",
                "entity_type": "node",
                "data": {
                    "name": "Test Entity for Rollback",
                    "labels": ["test"],
                    "group_id": "test-group"
                }
            }
        ]
        
        rollback_delta = await driver.create_delta(rollback_operations)
        await driver.apply_delta(rollback_delta)
        
        # Rollback the delta
        logger.info("Rolling back delta...")
        # For demonstration, we'll use a placeholder ID
        # In practice, you'd get the actual delta_id from the apply_delta result
        placeholder_id = f"rollback-delta-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        rollback_result = await driver.rollback_delta(placeholder_id)
        logger.info(f"Rollback result: {rollback_result}")
        
    finally:
        driver.close()


async def batch_processing_example():
    """Demonstrate batch processing capabilities"""
    logger.info("=== Batch Processing Example ===")
    
    driver = HuggingFaceDriver(
        repo_id="test/batch-processing-example", 
        create_repo=True,
        enable_vector_search=True
    )
    
    try:
        # Create multiple entities for batch processing
        entities = []
        for i in range(10):
            entity = EntityNode(
                uuid=f"batch-entity-{i}",
                name=f"Technology {i}",
                labels=["topic", "technology"],
                group_id="batch-group",
                created_at=datetime.now(),
                valid_at=datetime.now(),
                summary=f"Description of technology {i}"
            )
            entities.append(entity)
        
        # Bulk upsert entities
        logger.info("Bulk upserting entities...")
        bulk_result = await driver.bulk_upsert_entities(entities)
        logger.info(f"Bulk upsert result: {bulk_result}")
        
        # Create batch updates
        batch_updates = [
            {
                "type": "add",
                "entity_type": "node", 
                "data": {
                    "name": "Quantum Computing",
                    "labels": ["topic", "technology", "emerging"],
                    "group_id": "batch-group"
                }
            },
            {
                "type": "add",
                "entity_type": "node",
                "data": {
                    "name": "Blockchain",
                    "labels": ["topic", "technology", "emerging"],
                    "group_id": "batch-group"
                }
            },
            {
                "type": "add",
                "entity_type": "edge",
                "data": {
                    "source_uuid": "batch-entity-0",
                    "target_uuid": "batch-entity-1", 
                    "fact": "Technology 0 is related to Technology 1",
                    "group_id": "batch-group"
                }
            }
        ]
        
        # Apply batch updates
        logger.info("Applying batch updates...")
        batch_result = await driver.batch_incremental_update(batch_updates)
        logger.info(f"Batch update result: {batch_result}")
        
        # Process large delta (chunking)
        large_operations = []
        for i in range(50):
            large_operations.append({
                "type": "add",
                "entity_type": "node",
                "data": {
                    "name": f"Large Entity {i}",
                    "labels": ["large", "batch"],
                    "group_id": "large-batch-group"
                }
            })
        
        large_delta = await driver.create_delta(large_operations)
        
        # Process large delta with chunking
        logger.info("Processing large delta with chunking...")
        large_result = await driver.process_large_delta(large_delta, chunk_size=10)
        logger.info(f"Large delta processing result: {large_result}")
        
        # Parallel delta application
        delta1 = await driver.create_delta([{
            "type": "add",
            "entity_type": "node",
            "data": {"name": "Parallel 1", "labels": ["parallel"], "group_id": "parallel-group"}
        }])
        
        delta2 = await driver.create_delta([{
            "type": "add", 
            "entity_type": "node",
            "data": {"name": "Parallel 2", "labels": ["parallel"], "group_id": "parallel-group"}
        }])
        
        logger.info("Applying deltas in parallel...")
        parallel_result = await driver.parallel_delta_application([delta1, delta2], max_concurrent=2)
        logger.info(f"Parallel delta result: {parallel_result}")
        
    finally:
        driver.close()


async def index_update_example():
    """Demonstrate incremental index updates"""
    logger.info("=== Index Update Example ===")
    
    driver = HuggingFaceDriver(
        repo_id="test/index-update-example",
        create_repo=True,
        enable_vector_search=True
    )
    
    try:
        # Add some entities with embeddings
        entities_with_embeddings = []
        for i in range(5):
            entity = EntityNode(
                uuid=f"embedding-entity-{i}",
                name=f"Concept {i}",
                labels=["concept"],
                group_id="embedding-group",
                created_at=datetime.now(),
                valid_at=datetime.now(),
                summary=f"Description of concept {i}",
                name_embedding=[0.1 * i, 0.2 * i, 0.3 * i]  # Mock embeddings
            )
            entities_with_embeddings.append(entity)
        
        # Add entities
        for entity in entities_with_embeddings:
            await driver.add_entity_incremental(entity)
        
        # Update vector indices incrementally
        logger.info("Updating vector indices incrementally...")
        new_embeddings = [
            [0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0],
            [1.1, 1.2, 1.3]
        ]
        new_uuids = ["embedding-entity-5", "embedding-entity-6", "embedding-entity-7"]
        
        vector_result = await driver.update_vector_indices_incremental(
            "node", new_embeddings, new_uuids
        )
        logger.info(f"Vector index update result: {vector_result}")
        
        # Update text indices incrementally
        logger.info("Updating text indices incrementally...")
        new_texts = [
            "This is a new text document about machine learning",
            "Another document about deep learning and neural networks",
            "Document about natural language processing and AI"
        ]
        text_uuids = ["text-entity-1", "text-entity-2", "text-entity-3"]
        
        text_result = await driver.update_text_indices_incremental(
            "edge", new_texts, text_uuids
        )
        logger.info(f"Text index update result: {text_result}")
        
        # Update graph indices
        graph_operations = [
            {
                "type": "add",
                "entity_type": "edge",
                "data": {
                    "source_uuid": "embedding-entity-0",
                    "target_uuid": "embedding-entity-1",
                    "fact": "Concept 0 is related to Concept 1",
                    "group_id": "graph-group"
                }
            }
        ]
        
        graph_result = await driver.update_graph_indices_incremental(graph_operations)
        logger.info(f"Graph index update result: {graph_result}")
        
        # Update temporal indices
        temporal_operations = [
            {
                "type": "add",
                "entity_type": "node", 
                "data": {
                    "name": "Temporal Entity",
                    "labels": ["temporal"],
                    "group_id": "temporal-group",
                    "created_at": datetime.now(),
                    "valid_at": datetime.now()
                }
            }
        ]
        
        temporal_result = await driver.update_temporal_indices_incremental(temporal_operations)
        logger.info(f"Temporal index update result: {temporal_result}")
        
        # Rebuild indices if needed
        rebuild_result = await driver.rebuild_indices_if_needed(threshold=0.8)
        logger.info(f"Index rebuild result: {rebuild_result}")
        
    finally:
        driver.close()


async def performance_monitoring_example():
    """Demonstrate performance monitoring and statistics"""
    logger.info("=== Performance Monitoring Example ===")
    
    driver = HuggingFaceDriver(
        repo_id="test/performance-monitoring-example",
        create_repo=True,
        enable_vector_search=True
    )
    
    try:
        # Perform various operations to generate statistics
        logger.info("Performing operations to generate statistics...")
        
        # Add multiple entities
        for i in range(20):
            entity = EntityNode(
                uuid=f"perf-entity-{i}",
                name=f"Performance Test Entity {i}",
                labels=["performance"],
                group_id="perf-group",
                created_at=datetime.now(),
                valid_at=datetime.now()
            )
            await driver.add_entity_incremental(entity)
        
        # Add multiple edges
        for i in range(15):
            edge = EntityEdge(
                uuid=f"perf-edge-{i}",
                source_node_uuid=f"perf-entity-{i}",
                target_node_uuid=f"perf-entity-{(i + 1) % 20}",
                fact=f"Relationship {i}",
                group_id="perf-group",
                created_at=datetime.now(),
                valid_at=datetime.now()
            )
            await driver.add_edge_incremental(edge)
        
        # Get comprehensive update statistics
        stats = driver.get_update_statistics()
        logger.info("=== Update Statistics ===")
        logger.info(f"Total updates: {stats.get('total_updates', 0)}")
        logger.info(f"Successful updates: {stats.get('successful_updates', 0)}")
        logger.info(f"Failed updates: {stats.get('failed_updates', 0)}")
        logger.info(f"Average update time: {stats.get('average_update_time', 0):.4f}s")
        logger.info(f"Last update time: {stats.get('last_update_time', 'N/A')}")
        
        # Get vector search statistics
        vector_stats = driver.get_vector_search_stats()
        logger.info("=== Vector Search Statistics ===")
        logger.info(f"Vector search enabled: {vector_stats.get('enabled', False)}")
        if vector_stats.get('indices'):
            for index_type, index_stats in vector_stats['indices'].items():
                logger.info(f"{index_type} index: {index_stats}")
        
        # Get hybrid search statistics
        if hasattr(driver, 'get_hybrid_search_stats'):
            hybrid_stats = driver.get_hybrid_search_stats()
            logger.info("=== Hybrid Search Statistics ===")
            logger.info(f"Hybrid search stats: {hybrid_stats}")
        
        # Get traversal statistics
        traversal_stats = driver.get_traversal_stats()
        logger.info("=== Traversal Statistics ===")
        logger.info(f"Traversal stats: {traversal_stats}")
        
    finally:
        driver.close()


async def main():
    """Run all examples"""
    logger.info("Starting Graphiti-HF Incremental Updates Examples")
    
    try:
        await basic_incremental_updates()
        await asyncio.sleep(1)  # Small delay between examples
        
        await delta_management_example()
        await asyncio.sleep(1)
        
        await batch_processing_example()
        await asyncio.sleep(1)
        
        await index_update_example()
        await asyncio.sleep(1)
        
        await performance_monitoring_example()
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())