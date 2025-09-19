"""
Temporal Consistency Example for Graphiti-HF

This example demonstrates how to use the temporal management features
of Graphiti-HF to maintain bi-temporal data consistency, handle
temporal edge invalidation, and perform time-based queries.

Key features demonstrated:
1. Bi-temporal data model with event occurrence and ingestion times
2. Temporal edge invalidation for contradictions
3. Historical state reconstruction
4. Time-based queries and temporal indexing
5. Temporal conflict resolution
6. Temporal data analysis and statistics
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def temporal_consistency_demo():
    """
    Demonstrate temporal consistency features of Graphiti-HF
    """
    try:
        # Import Graphiti-HF
        from graphiti_hf import GraphitiHF
        
        # Initialize Graphiti-HF with a temporary repository for demo
        repo_id = "temporal-consistency-demo"
        graphiti = GraphitiHF(repo_id)
        
        logger.info("🚀 Starting Temporal Consistency Demo")
        logger.info("=" * 50)
        
        # 1. Demonstrate bi-temporal data model
        await demo_bi_temporal_data_model(graphiti)
        
        # 2. Demonstrate temporal edge invalidation
        await demo_temporal_edge_invalidation(graphiti)
        
        # 3. Demonstrate historical state reconstruction
        await demo_historical_state_reconstruction(graphiti)
        
        # 4. Demonstrate time-based queries
        await demo_time_based_queries(graphiti)
        
        # 5. Demonstrate temporal conflict resolution
        await demo_temporal_conflict_resolution(graphiti)
        
        # 6. Demonstrate temporal data analysis
        await demo_temporal_data_analysis(graphiti)
        
        # 7. Demonstrate temporal indexing
        await demo_temporal_indexing(graphiti)
        
        logger.info("✅ Temporal Consistency Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in temporal consistency demo: {e}")
        raise


async def demo_bi_temporal_data_model(graphiti):
    """
    Demonstrate bi-temporal data model with event occurrence and ingestion times
    """
    logger.info("\n📅 1. Bi-temporal Data Model Demo")
    logger.info("-" * 30)
    
    try:
        # Create some sample data with temporal information
        current_time = datetime.now()
        past_time = current_time - timedelta(days=30)
        future_time = current_time + timedelta(days=7)
        
        # Add an episode with temporal context
        episode_result = await graphiti.add_episode(
            name="Company History",
            episode_body="""
            Apple Inc. was founded on April 1, 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.
            The company was incorporated on January 3, 1977. In 1984, Apple introduced the Macintosh,
            which was the first commercially successful personal computer to feature a mouse.
            """,
            source_description="Company historical records",
            reference_time=past_time  # This represents when the events actually occurred
        )
        
        logger.info(f"📝 Added episode: {episode_result.episode.name}")
        logger.info(f"   Reference time: {past_time}")
        logger.info(f"   Ingestion time: {current_time}")
        logger.info(f"   Nodes created: {len(episode_result.nodes)}")
        logger.info(f"   Edges created: {len(episode_result.edges)}")
        
        # Set validity periods for some entities
        if episode_result.edges:
            edge_uuid = episode_result.edges[0].uuid
            validity_result = await graphiti.driver.set_validity_period(
                entity_uuid=edge_uuid,
                valid_from=past_time,
                valid_to=future_time,
                entity_type="edge"
            )
            
            logger.info(f"⏰ Set validity period for edge {edge_uuid}")
            logger.info(f"   Valid from: {past_time}")
            logger.info(f"   Valid to: {future_time}")
            logger.info(f"   Result: {validity_result['success']}")
        
        # Get temporal statistics
        temporal_stats = await graphiti.driver.get_temporal_stats()
        logger.info(f"📊 Temporal Statistics:")
        logger.info(f"   Total records: {temporal_stats.total_records}")
        logger.info(f"   Valid records: {temporal_stats.valid_records}")
        logger.info(f"   Invalidated records: {temporal_stats.invalidated_records}")
        logger.info(f"   Data ingestion rate: {temporal_stats.data_ingestion_rate:.2f} records/hour")
        
    except Exception as e:
        logger.error(f"❌ Error in bi-temporal data model demo: {e}")


async def demo_temporal_edge_invalidation(graphiti):
    """
    Demonstrate temporal edge invalidation for contradictions
    """
    logger.info("\n🔄 2. Temporal Edge Invalidation Demo")
    logger.info("-" * 35)
    
    try:
        # Add some contradictory information
        contradiction_time = datetime.now() - timedelta(days=15)
        
        # First statement (true)
        await graphiti.add_episode(
            name="Company Location - Initial",
            episode_body="Apple Inc. is headquartered in Cupertino, California.",
            source_description="Official company website",
            reference_time=contradiction_time
        )
        
        # Contradictory statement (false)
        await graphiti.add_episode(
            name="Company Location - Contradiction",
            episode_body="Apple Inc. is headquartered in New York City.",
            source_description="Incorrect news article",
            reference_time=contradiction_time + timedelta(hours=1)
        )
        
        # Get all edges to find the contradictory ones
        all_edges = await graphiti.driver.query_edges({})
        logger.info(f"📋 Total edges in graph: {len(all_edges)}")
        
        # Find edges with similar facts (contradictions)
        contradictory_edges = []
        for i, edge1 in enumerate(all_edges):
            for edge2 in all_edges[i+1:]:
                if (edge1.source_node_uuid == edge2.source_node_uuid and 
                    edge1.target_node_uuid == edge2.target_node_uuid and
                    "headquartered" in edge1.fact.lower() and 
                    "headquartered" in edge2.fact.lower()):
                    contradictory_edges.extend([edge1.uuid, edge2.uuid])
        
        if contradictory_edges:
            logger.info(f"🔍 Found {len(contradictory_edges)} potentially contradictory edges")
            
            # Invalidate the contradictory edges
            invalidation_result = await graphiti.driver.invalidate_edges(
                contradictory_edges,
                invalidation_reason="contradictory_information",
                invalidation_time=datetime.now()
            )
            
            logger.info(f"🗑️  Invalidation result:")
            logger.info(f"   Invalidated: {invalidation_result['invalidated']} edges")
            logger.info(f"   Failed: {invalidation_result['failed']} edges")
            
            # Show details of invalidation
            for detail in invalidation_result['details']:
                logger.info(f"   {detail['edge_uuid']}: {detail['status']}")
        else:
            logger.info("🔍 No contradictory edges found to invalidate")
        
    except Exception as e:
        logger.error(f"❌ Error in temporal edge invalidation demo: {e}")


async def demo_historical_state_reconstruction(graphiti):
    """
    Demonstrate historical state reconstruction
    """
    logger.info("\n🕰️  3. Historical State Reconstruction Demo")
    logger.info("-" * 40)
    
    try:
        # Define different time points to reconstruct
        current_time = datetime.now()
        time_points = [
            current_time - timedelta(days=60),  # 60 days ago
            current_time - timedelta(days=30),  # 30 days ago
            current_time - timedelta(days=7),   # 7 days ago
            current_time                        # Now
        ]
        
        for i, query_time in enumerate(time_points):
            logger.info(f"\n📅 Reconstructing state for {query_time.strftime('%Y-%m-%d %H:%M')}")
            
            # Get historical state
            historical_state = await graphiti.driver.get_historical_state(
                query_time=query_time,
                limit=10  # Limit for demo purposes
            )
            
            logger.info(f"   📊 Historical State:")
            logger.info(f"      Valid nodes: {historical_state['node_count']}")
            logger.info(f"      Valid edges: {historical_state['edge_count']}")
            logger.info(f"      Query time: {historical_state['query_time']}")
            
            # Show some sample nodes and edges
            if historical_state['valid_nodes']:
                logger.info(f"   📝 Sample nodes:")
                for node in historical_state['valid_nodes'][:3]:
                    logger.info(f"      - {node.get('name', 'Unknown')} (valid: {node.get('valid_at', 'N/A')})")
            
            if historical_state['valid_edges']:
                logger.info(f"   🔗 Sample edges:")
                for edge in historical_state['valid_edges'][:3]:
                    logger.info(f"      - {edge.get('fact', 'Unknown fact')[:50]}... (valid: {edge.get('valid_at', 'N/A')})")
        
    except Exception as e:
        logger.error(f"❌ Error in historical state reconstruction demo: {e}")


async def demo_time_based_queries(graphiti):
    """
    Demonstrate time-based queries
    """
    logger.info("\n🔍 4. Time-based Queries Demo")
    logger.info("-" * 30)
    
    try:
        current_time = datetime.now()
        
        # Define time ranges for queries
        time_ranges = [
            ("Last 7 days", current_time - timedelta(days=7), current_time),
            ("Last 30 days", current_time - timedelta(days=30), current_time),
            ("Last 90 days", current_time - timedelta(days=90), current_time)
        ]
        
        for range_name, start_time, end_time in time_ranges:
            logger.info(f"\n📅 Querying: {range_name}")
            logger.info(f"   Time range: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
            
            # Perform temporal range query
            results = await graphiti.driver.temporal_range_query(
                start_time=start_time,
                end_time=end_time,
                entity_type="edge",
                limit=20
            )
            
            logger.info(f"   📊 Results found: {len(results)} edges")
            
            # Show temporal information for results
            for result in results[:5]:  # Show first 5 results
                temporal_info = result.get('temporal_info', {})
                logger.info(f"   🔗 {result.get('fact', 'Unknown fact')[:50]}...")
                logger.info(f"      Valid: {temporal_info.get('valid_at', 'N/A')}")
                logger.info(f"      Created: {temporal_info.get('created_at', 'N/A')}")
        
        # Demonstrate temporal point query
        logger.info(f"\n🕐 Point query at specific time")
        point_time = current_time - timedelta(days=15)
        
        point_results = await graphiti.driver.temporal_point_query(
            query_time=point_time,
            entity_type="edge",
            limit=10
        )
        
        logger.info(f"   📊 Results valid at {point_time.strftime('%Y-%m-%d')}: {len(point_results)}")
        
        # Demonstrate temporal search
        logger.info(f"\n🔍 Temporal search with text query")
        search_results = await graphiti.driver.temporal_search(
            query="Apple",
            start_time=current_time - timedelta(days=30),
            end_time=current_time,
            limit=10
        )
        
        logger.info(f"   📊 Search results: {len(search_results)}")
        
    except Exception as e:
        logger.error(f"❌ Error in time-based queries demo: {e}")


async def demo_temporal_conflict_resolution(graphiti):
    """
    Demonstrate temporal conflict resolution
    """
    logger.info("\n⚖️  5. Temporal Conflict Resolution Demo")
    logger.info("-" * 40)
    
    try:
        # Create some conflicting data
        conflict_time = datetime.now() - timedelta(days=20)
        
        # Add conflicting information about company founding
        await graphiti.add_episode(
            name="Company Founding - Version 1",
            episode_body="Apple Inc. was founded on April 1, 1976.",
            source_description="Early company records",
            reference_time=conflict_time
        )
        
        await graphiti.add_episode(
            name="Company Founding - Version 2",
            episode_body="Apple Inc. was founded on January 3, 1977.",
            source_description="Incorporation records",
            reference_time=conflict_time + timedelta(hours=1)
        )
        
        # Detect temporal conflicts
        anomalies = await graphiti.driver.detect_temporal_anomalies()
        logger.info(f"🔍 Detected {len(anomalies)} temporal anomalies")
        
        for anomaly in anomalies[:3]:  # Show first 3 anomalies
            logger.info(f"   - {anomaly.get('type', 'Unknown')}: {anomaly.get('severity', 'Unknown')}")
        
        # Perform temporal consistency check
        consistency_result = await graphiti.driver.temporal_consistency_check()
        logger.info(f"📊 Consistency check:")
        logger.info(f"   Issues found: {consistency_result['issues_found']}")
        logger.info(f"   Checks performed: {len(consistency_result['checks_performed'])}")
        
        # Resolve conflicts using different strategies
        if consistency_result['issues_found'] > 0:
            logger.info(f"\n🛠️  Resolving conflicts with different strategies:")
            
            strategies = ["first_wins", "last_wins", "merge"]
            
            for strategy in strategies:
                logger.info(f"\n   Using strategy: {strategy}")
                
                # Create mock conflict data for demo
                conflict_data = [{
                    "conflict_id": f"conflict_{strategy}",
                    "conflict_type": "overlapping_validity",
                    "affected_entities": ["demo_entity"],
                    "details": "Demo conflict for testing"
                }]
                
                resolution_result = await graphiti.driver.resolve_temporal_conflicts(
                    conflicts=conflict_data,
                    strategy=strategy
                )
                
                logger.info(f"      Resolved: {resolution_result.get('resolved_conflicts', 0)} conflicts")
                logger.info(f"      Failed: {resolution_result.get('failed_resolutions', 0)} conflicts")
        
    except Exception as e:
        logger.error(f"❌ Error in temporal conflict resolution demo: {e}")


async def demo_temporal_data_analysis(graphiti):
    """
    Demonstrate temporal data analysis
    """
    logger.info("\n📈 6. Temporal Data Analysis Demo")
    logger.info("-" * 35)
    
    try:
        current_time = datetime.now()
        
        # Perform temporal aggregation
        logger.info(f"\n📊 Temporal Aggregation:")
        
        aggregation_types = ["count", "avg", "max", "min"]
        time_range = (current_time - timedelta(days=30), current_time)
        
        for agg_type in aggregation_types:
            try:
                result = await graphiti.driver.temporal_aggregation(
                    aggregation_type=agg_type,
                    time_range=time_range,
                    entity_type="edge"
                )
                
                logger.info(f"   {agg_type.upper()}: {result.get('aggregated_value', 'N/A')}")
                logger.info(f"      Total records: {result.get('total_records', 0)}")
                
            except Exception as e:
                logger.info(f"   {agg_type.upper()}: Not applicable for this data")
        
        # Get detailed temporal statistics
        logger.info(f"\n📈 Detailed Temporal Statistics:")
        stats = await graphiti.driver.get_temporal_stats()
        
        logger.info(f"   Total records: {stats.total_records}")
        logger.info(f"   Valid records: {stats.valid_records}")
        logger.info(f"   Invalidated records: {stats.invalidated_records}")
        logger.info(f"   Conflicts detected: {stats.conflicts_detected}")
        logger.info(f"   Time span: {stats.time_span[0].strftime('%Y-%m-%d')} to {stats.time_span[1].strftime('%Y-%m-%d')}")
        logger.info(f"   Data ingestion rate: {stats.data_ingestion_rate:.2f} records/hour")
        
        # Show records by entity type
        if stats.records_by_entity_type:
            logger.info(f"   Records by entity type:")
            for entity_type, count in stats.records_by_entity_type.items():
                logger.info(f"      {entity_type}: {count}")
        
        # Demonstrate temporal deduplication
        logger.info(f"\n🔍 Temporal Deduplication:")
        dedup_result = await graphiti.driver.temporal_deduplication(
            similarity_threshold=0.9,
            time_window_hours=48
        )
        
        logger.info(f"   Duplicates found: {dedup_result['duplicates_found']}")
        logger.info(f"   Duplicates removed: {dedup_result['duplicates_removed']}")
        logger.info(f"   Groups processed: {dedup_result['groups_processed']}")
        
    except Exception as e:
        logger.error(f"❌ Error in temporal data analysis demo: {e}")


async def demo_temporal_indexing(graphiti):
    """
    Demonstrate temporal indexing
    """
    logger.info("\n🗂️  7. Temporal Indexing Demo")
    logger.info("-" * 30)
    
    try:
        # Build temporal indices
        logger.info(f"🔨 Building temporal indices...")
        index_result = await graphiti.driver.build_temporal_indices()
        
        logger.info(f"   Indices built: {len(index_result['indices_built'])}")
        logger.info(f"   Records indexed: {index_result['records_indexed']}")
        
        for index_name in index_result['indices_built']:
            logger.info(f"   - {index_name}")
        
        # Perform efficient temporal queries using indices
        current_time = datetime.now()
        
        logger.info(f"\n⚡ Efficient temporal queries with indices:")
        
        # Range query
        start_time = current_time - timedelta(days=14)
        end_time = current_time - timedelta(days=7)
        
        range_start = datetime.now()
        range_results = await graphiti.driver.temporal_range_query(
            start_time=start_time,
            end_time=end_time,
            entity_type="edge",
            limit=50
        )
        range_end = datetime.now()
        
        logger.info(f"   Range query ({start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}):")
        logger.info(f"      Results: {len(range_results)}")
        logger.info(f"      Duration: {(range_end - range_start).total_seconds():.3f}s")
        
        # Point query
        point_time = current_time - timedelta(days=10)
        
        point_start = datetime.now()
        point_results = await graphiti.driver.temporal_point_query(
            query_time=point_time,
            entity_type="edge",
            limit=50
        )
        point_end = datetime.now()
        
        logger.info(f"   Point query ({point_time.strftime('%Y-%m-%d')}):")
        logger.info(f"      Results: {len(point_results)}")
        logger.info(f"      Duration: {(point_end - point_start).total_seconds():.3f}s")
        
        # Auto temporal cleanup
        logger.info(f"\n🧹 Auto temporal cleanup:")
        cleanup_result = await graphiti.driver.auto_temporal_cleanup(
            cleanup_strategy="soft",
            older_than_days=7
        )
        
        logger.info(f"   Strategy: {cleanup_result['cleanup_strategy']}")
        logger.info(f"   Records processed: {cleanup_result['records_processed']}")
        logger.info(f"   Records cleaned: {cleanup_result['records_cleaned']}")
        
    except Exception as e:
        logger.error(f"❌ Error in temporal indexing demo: {e}")


async def main():
    """
    Main function to run the temporal consistency demo
    """
    try:
        await temporal_consistency_demo()
        
        logger.info("\n🎉 Temporal Consistency Demo completed!")
        logger.info("📚 Key features demonstrated:")
        logger.info("   ✅ Bi-temporal data model with event occurrence and ingestion times")
        logger.info("   ✅ Temporal edge invalidation for contradictions")
        logger.info("   ✅ Historical state reconstruction")
        logger.info("   ✅ Time-based queries and temporal indexing")
        logger.info("   ✅ Temporal conflict resolution")
        logger.info("   ✅ Temporal data analysis and statistics")
        logger.info("   ✅ Efficient temporal indexing and cleanup")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())