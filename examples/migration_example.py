"""
Example usage of Graphiti-HF migration utilities

This script demonstrates how to use the migration utilities to migrate
from a traditional Graphiti instance to Graphiti-HF.
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime

# Import migration utilities
from graphiti_hf.migration import (
    GraphitiMigrator, 
    BatchProcessor, 
    setup_migration_logging,
    create_migration_summary,
    validate_triplet_data,
    transform_episode_format
)

# Import Graphiti components
from graphiti_core import Graphiti
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver

# Configure logging
setup_migration_logging(log_level="INFO", log_file="migration.log")


async def example_migrate_from_neo4j():
    """
    Example: Migrate from Neo4j Graphiti to HuggingFace datasets
    """
    print("🚀 Example: Migrating from Neo4j to HuggingFace...")
    
    # Initialize source Graphiti (this would be your existing instance)
    # Note: In a real scenario, you would initialize with your actual Neo4j connection
    source_graphiti = Graphiti()
    
    # Initialize migrator with progress callback
    def progress_callback(progress: float, batch_stats: Dict[str, Any]):
        """Progress callback function"""
        print(f"📊 Progress: {progress:.1f}% - {batch_stats}")
    
    migrator = GraphitiMigrator(progress_callback=progress_callback)
    
    try:
        # Perform migration
        migration_stats = await migrator.migrate_from_neo4j(
            source_graphiti=source_graphiti,
            target_repo_id="your-org/knowledge-graph-migration",
            batch_size=500,
            group_ids=["research", "engineering"],  # Optional: specific groups
            include_embeddings=True,
            skip_existing=True
        )
        
        # Print migration summary
        print(create_migration_summary(migration_stats))
        
        # Validate migration integrity
        target_driver = HuggingFaceDriver("your-org/knowledge-graph-migration")
        validation_result = await migrator.validate_migration_integrity(
            source_graphiti, 
            target_driver
        )
        
        print("🔍 Migration Validation Results:")
        print(f"  Source Stats: {validation_result['source_stats']}")
        print(f"  Target Stats: {validation_result['target_stats']}")
        print(f"  Integrity Check: {validation_result['integrity_check']}")
        
        if validation_result['discrepancies']:
            print("⚠️ Discrepancies found:")
            for discrepancy in validation_result['discrepancies']:
                print(f"  - {discrepancy}")
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")


async def example_batch_process_triplets():
    """
    Example: Process triplets in batches
    """
    print("\n🔄 Example: Processing triplets in batches...")
    
    # Create sample triplets
    sample_triplets = [
        {"source": "Alice", "target": "Bob", "relation": "works_with"},
        {"source": "Bob", "target": "Company", "relation": "employed_at"},
        {"source": "Alice", "target": "Project", "relation": "leads"},
        {"source": "Project", "target": "Technology", "relation": "uses"},
        {"source": "Company", "target": "Technology", "relation": "develops"},
        {"source": "Alice", "target": "Bob", "relation": "manages"},
        {"source": "Bob", "target": "Project", "relation": "contributes_to"},
        {"source": "Alice", "target": "Technology", "relation": "expert_in"},
    ]
    
    # Initialize target driver
    target_driver = HuggingFaceDriver("your-org/triplet-processing", create_repo=True)
    
    # Initialize batch processor
    processor = BatchProcessor(target_driver)
    
    try:
        # Process triplets in batches
        processing_stats = await processor.batch_add_triplets(
            triplets=sample_triplets,
            batch_size=3,  # Small batch for demonstration
            group_id="example_group",
            include_embeddings=True
        )
        
        print(f"✅ Processing completed: {processing_stats}")
        
        # Rebuild indices
        await processor.rebuild_indices()
        print("🔧 Indices rebuilt successfully")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")


async def example_data_transformation():
    """
    Example: Data transformation utilities
    """
    print("\n🔄 Example: Data transformation utilities...")
    
    # Sample episode data from different sources
    sample_episodes = [
        {
            "uuid": "ep1",
            "name": "Team Meeting",
            "content": "Alice and Bob discussed the project.",
            "source_description": "Meeting notes",
            "source": "notion",
            "valid_at": "2024-01-15T10:00:00",
            "created_at": "2024-01-15T10:00:00",
            "group_id": "meetings"
        },
        {
            "name": "Design Review",  # Missing UUID
            "content": "The design was reviewed by the team.",
            "source_description": "Design document",
            "source": "figma",
            "valid_at": "2024-01-16T14:00:00",
            "group_id": "design"
        }
    ]
    
    # Transform episode formats
    for i, episode in enumerate(sample_episodes):
        print(f"\n📝 Transforming episode {i+1}...")
        print(f"  Original: {episode}")
        
        transformed = transform_episode_format(episode)
        print(f"  Transformed: {transformed}")
    
    # Validate triplet data
    sample_triplets = [
        {"source": "Alice", "target": "Bob", "relation": "collaborates_with"},
        {"source": "Bob", "target": "", "relation": "works_on"},  # Invalid: empty target
        {"source": "", "target": "Project", "relation": "leads"},  # Invalid: empty source
        {"source": "Alice", "target": "Project"},  # Invalid: missing relation
    ]
    
    print(f"\n🔍 Validating {len(sample_triplets)} triplets...")
    for i, triplet in enumerate(sample_triplets):
        is_valid = validate_triplet_data(triplet)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  Triplet {i+1}: {status} - {triplet}")


async def example_migration_scenarios():
    """
    Example: Different migration scenarios
    """
    print("\n🎯 Example: Different migration scenarios...")
    
    # Scenario 1: Small knowledge graph
    print("\n1️⃣ Small Knowledge Graph Migration")
    print("   - ~1000 nodes")
    print("   - ~5000 edges")
    print("   - Single group")
    print("   - Full migration with embeddings")
    
    # Scenario 2: Large enterprise knowledge graph
    print("\n2️⃣ Large Enterprise Knowledge Graph Migration")
    print("   - ~1M nodes")
    print("   - ~5M edges")
    print("   - Multiple groups (HR, Engineering, Marketing)")
    print("   - Incremental migration")
    print("   - Skip existing data")
    
    # Scenario 3: Research paper knowledge graph
    print("\n3️⃣ Research Paper Knowledge Graph Migration")
    print("   - ~10,000 nodes (papers, authors, concepts)")
    print("   - ~50,000 edges (citations, collaborations)")
    print("   - Academic group")
    print("   - Include paper embeddings")
    
    # Scenario 4: Personal knowledge management
    print("\n4️⃣ Personal Knowledge Management Migration")
    print("   - ~500 nodes (notes, projects, people)")
    print("   - ~2000 edges (connections, references)")
    print("   - Personal group")
    print("   - Include note embeddings")


async def example_error_handling():
    """
    Example: Error handling during migration
    """
    print("\n⚠️ Example: Error handling during migration...")
    
    # Simulate different error scenarios
    error_scenarios = [
        {
            "scenario": "Network connectivity issues",
            "error_type": "ConnectionError",
            "recommendation": "Check network connection and retry"
        },
        {
            "scenario": "HuggingFace rate limits",
            "error_type": "RateLimitError", 
            "recommendation": "Implement exponential backoff and retry"
        },
        {
            "scenario": "Invalid episode data",
            "error_type": "ValidationError",
            "recommendation": "Validate data before processing"
        },
        {
            "scenario": "Insufficient disk space",
            "error_type": "StorageError",
            "recommendation": "Free up disk space or use larger instance"
        }
    ]
    
    print("Common migration errors and handling strategies:")
    for i, scenario in enumerate(error_scenarios, 1):
        print(f"\n{i}. {scenario['scenario']}")
        print(f"   Error Type: {scenario['error_type']}")
        print(f"   Recommendation: {scenario['recommendation']}")


async def example_performance_optimization():
    """
    Example: Performance optimization strategies
    """
    print("\n⚡ Example: Performance optimization strategies...")
    
    optimization_strategies = [
        {
            "strategy": "Batch Processing",
            "description": "Process episodes in batches to reduce API calls",
            "implementation": "Use batch_size parameter (1000-5000 episodes per batch)"
        },
        {
            "strategy": "Parallel Processing",
            "description": "Process multiple batches simultaneously",
            "implementation": "Use asyncio.gather() for concurrent batch processing"
        },
        {
            "strategy": "Incremental Updates",
            "description": "Only process new or changed episodes",
            "implementation": "Use skip_existing=True and track processed episodes"
        },
        {
            "strategy": "Memory Management",
            "description": "Avoid loading entire datasets in memory",
            "implementation": "Use chunked processing and streaming"
        },
        {
            "strategy": "Index Optimization",
            "description": "Build search indices after migration",
            "implementation": "Call rebuild_indices() when migration is complete"
        }
    ]
    
    print("Performance optimization strategies:")
    for i, strategy in enumerate(optimization_strategies, 1):
        print(f"\n{i}. {strategy['strategy']}")
        print(f"   Description: {strategy['description']}")
        print(f"   Implementation: {strategy['implementation']}")


async def main():
    """
    Main function to run all examples
    """
    print("🧪 Graphiti-HF Migration Examples")
    print("=" * 50)
    
    # Run all examples
    await example_migrate_from_neo4j()
    await example_batch_process_triplets()
    await example_data_transformation()
    await example_migration_scenarios()
    await example_error_handling()
    await example_performance_optimization()
    
    print("\n✅ All examples completed!")
    print("\n📚 Next steps:")
    print("1. Set up your source Graphiti instance")
    print("2. Configure HuggingFace authentication")
    print("3. Run migration with appropriate parameters")
    print("4. Validate migration results")
    print("5. Deploy your migrated knowledge graph")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())