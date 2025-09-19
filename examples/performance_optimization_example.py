"""
Performance Optimization Example for Graphiti-HF

This example demonstrates how to use the performance optimization features
of Graphiti-HF to optimize search performance with indexing strategies.

Key features demonstrated:
1. Building different types of indices (BM25, TF-IDF, FAISS, NetworkX, temporal)
2. Performance monitoring and benchmarking
3. Automatic index optimization
4. Index management and versioning
5. Integration with HuggingFaceDriver
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Graphiti-HF components
from graphiti_hf import HuggingFaceDriver
from graphiti_hf.search.performance_optimizer import (
    SearchIndexManager, 
    IndexConfig, 
    IndexType,
    PerformanceMetrics
)
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge


async def create_sample_data(driver: HuggingFaceDriver) -> None:
    """Create sample data for performance optimization demonstration"""
    logger.info("Creating sample data for performance optimization...")
    
    # Sample nodes
    nodes = [
        EntityNode(
            name="Machine Learning",
            labels=["Topic", "Technology"],
            group_id="tech",
            summary="Field of study focused on algorithms and statistical models"
        ),
        EntityNode(
            name="Python",
            labels=["Programming Language", "Technology"],
            group_id="tech",
            summary="High-level programming language"
        ),
        EntityNode(
            name="Neural Networks",
            labels=["Technology", "AI"],
            group_id="ai",
            summary="Computing systems inspired by biological neural networks"
        ),
        EntityNode(
            name="Natural Language Processing",
            labels=["AI", "Technology"],
            group_id="ai",
            summary="Field of AI focused on understanding human language"
        ),
        EntityNode(
            name="Computer Vision",
            labels=["AI", "Technology"],
            group_id="ai",
            summary="Field of AI focused on understanding visual information"
        ),
        EntityNode(
            name="Data Science",
            labels=["Field", "Technology"],
            group_id="tech",
            summary="Interdisciplinary field using scientific methods"
        ),
        EntityNode(
            name="Deep Learning",
            labels=["AI", "Technology"],
            group_id="ai",
            summary="Subset of machine learning using neural networks"
        ),
        EntityNode(
            name="TensorFlow",
            labels=["Framework", "Technology"],
            group_id="tech",
            summary="Open-source machine learning framework"
        ),
        EntityNode(
            name="PyTorch",
            labels=["Framework", "Technology"],
            group_id="tech",
            summary="Open-source machine learning framework"
        ),
        EntityNode(
            name="Transformers",
            labels=["Architecture", "AI"],
            group_id="ai",
            summary="Neural network architecture for sequence modeling"
        )
    ]
    
    # Sample edges
    edges = [
        EntityEdge(
            source_node_uuid=nodes[0].uuid,  # Machine Learning
            target_node_uuid=nodes[2].uuid,  # Neural Networks
            fact="Machine Learning includes Neural Networks as a subfield",
            group_id="tech"
        ),
        EntityEdge(
            source_node_uuid=nodes[0].uuid,  # Machine Learning
            target_node_uuid=nodes[5].uuid,  # Data Science
            fact="Machine Learning is a key component of Data Science",
            group_id="tech"
        ),
        EntityEdge(
            source_node_uuid=nodes[1].uuid,  # Python
            target_node_uuid=nodes[0].uuid,  # Machine Learning
            fact="Python is commonly used for Machine Learning",
            group_id="tech"
        ),
        EntityEdge(
            source_node_uuid=nodes[2].uuid,  # Neural Networks
            target_node_uuid=nodes[6].uuid,  # Deep Learning
            fact="Deep Learning is based on Neural Networks",
            group_id="ai"
        ),
        EntityEdge(
            source_node_uuid=nodes[2].uuid,  # Neural Networks
            target_node_uuid=nodes[3].uuid,  # NLP
            fact="Neural Networks are used in Natural Language Processing",
            group_id="ai"
        ),
        EntityEdge(
            source_node_uuid=nodes[2].uuid,  # Neural Networks
            target_node_uuid=nodes[4].uuid,  # Computer Vision
            fact="Neural Networks are used in Computer Vision",
            group_id="ai"
        ),
        EntityEdge(
            source_node_uuid=nodes[6].uuid,  # Deep Learning
            target_node_uuid=nodes[3].uuid,  # NLP
            fact="Deep Learning powers modern Natural Language Processing",
            group_id="ai"
        ),
        EntityEdge(
            source_node_uuid=nodes[6].uuid,  # Deep Learning
            target_node_uuid=nodes[4].uuid,  # Computer Vision
            fact="Deep Learning powers modern Computer Vision",
            group_id="ai"
        ),
        EntityEdge(
            source_node_uuid=nodes[6].uuid,  # Deep Learning
            target_node_uuid=nodes[9].uuid,  # Transformers
            fact="Deep Learning uses Transformers architecture",
            group_id="ai"
        ),
        EntityEdge(
            source_node_uuid=nodes[7].uuid,  # TensorFlow
            target_node_uuid=nodes[0].uuid,  # Machine Learning
            fact="TensorFlow is a framework for Machine Learning",
            group_id="tech"
        ),
        EntityEdge(
            source_node_uuid=nodes[8].uuid,  # PyTorch
            target_node_uuid=nodes[0].uuid,  # Machine Learning
            fact="PyTorch is a framework for Machine Learning",
            group_id="tech"
        ),
        EntityEdge(
            source_node_uuid=nodes[9].uuid,  # Transformers
            target_node_uuid=nodes[3].uuid,  # NLP
            fact="Transformers architecture revolutionized Natural Language Processing",
            group_id="ai"
        )
    ]
    
    # Save nodes and edges
    await driver.save_nodes(nodes)
    await driver.save_edges(edges)
    
    logger.info(f"Created {len(nodes)} nodes and {len(edges)} edges")


async def demonstrate_text_indices(driver: HuggingFaceDriver) -> None:
    """Demonstrate text index building and optimization"""
    logger.info("=== Demonstrating Text Index Optimization ===")
    
    if not driver.performance_optimizer:
        logger.warning("Performance optimization not enabled")
        return
    
    optimizer = driver.performance_optimizer
    
    # Build text indices
    logger.info("Building BM25 and TF-IDF indices...")
    text_indices = optimizer.build_text_indices()
    
    logger.info(f"Built text indices: {list(text_indices.keys())}")
    
    # Test text search
    test_queries = [
        "machine learning algorithms",
        "neural networks deep learning",
        "natural language processing",
        "python programming"
    ]
    
    for query in test_queries:
        logger.info(f"Testing text search for: '{query}'")
        
        # Test BM25 search
        if 'bm25_edges' in text_indices:
            bm25_index = text_indices['bm25_edges']
            query_tokens = query.lower().split()
            bm25_scores = bm25_index.get_scores(query_tokens)
            
            # Get top results
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:3]
            logger.info(f"BM25 top scores: {[bm25_scores[i] for i in top_indices]}")
        
        # Test TF-IDF search
        if 'tfidf_edges' in text_indices:
            tfidf_index = text_indices['tfidf_edges']
            query_vector = tfidf_index.transform([query])
            # Note: TF-IDF scoring would require additional implementation
            logger.info("TF-IDF search completed")


async def demonstrate_vector_indices(driver: HuggingFaceDriver) -> None:
    """Demonstrate vector index building and optimization"""
    logger.info("=== Demonstrating Vector Index Optimization ===")
    
    if not driver.performance_optimizer:
        logger.warning("Performance optimization not enabled")
        return
    
    optimizer = driver.performance_optimizer
    
    # Build vector indices with different types
    logger.info("Building FAISS indices with different types...")
    
    index_types = [IndexType.FLAT, IndexType.IVFFLAT, IndexType.HNSW]
    vector_indices = {}
    
    for index_type in index_types:
        logger.info(f"Building {index_type.value} index...")
        indices = optimizer.build_vector_indices([index_type])
        vector_indices.update(indices)
    
    logger.info(f"Built vector indices: {list(vector_indices.keys())}")
    
    # Test vector search
    if driver.vector_search_engine and driver.vector_search_engine.node_index:
        logger.info("Testing vector search...")
        
        # Get a sample node embedding for testing
        sample_nodes = await driver.get_nodes_by_group_ids(["tech"], limit=1)
        if sample_nodes:
            # Create a dummy query embedding (in practice, this would be from an embedder)
            query_embedding = [0.1] * 384  # 384-dimensional embedding
            
            # Perform vector search
            results = await driver.query_nodes_by_embedding(query_embedding, k=5)
            logger.info(f"Vector search found {len(results)} nodes")


async def demonstrate_graph_indices(driver: HuggingFaceDriver) -> None:
    """Demonstrate graph index building and optimization"""
    logger.info("=== Demonstrating Graph Index Optimization ===")
    
    if not driver.performance_optimizer:
        logger.warning("Performance optimization not enabled")
        return
    
    optimizer = driver.performance_optimizer
    
    # Build graph indices
    logger.info("Building NetworkX graph indices...")
    graph_indices = optimizer.build_graph_indices()
    
    logger.info(f"Graph indices built:")
    logger.info(f"  - Nodes: {graph_indices['node_count']}")
    logger.info(f"  - Edges: {graph_indices['edge_count']}")
    logger.info(f"  - Adjacency lists: {len(graph_indices['adjacency_lists'])}")
    
    # Test graph traversal
    logger.info("Testing graph traversal...")
    
    # Get some sample nodes
    tech_nodes = await driver.get_nodes_by_group_ids(["tech"], limit=3)
    if tech_nodes:
        start_nodes = [node.uuid for node in tech_nodes]
        
        # Perform BFS traversal
        traversal_result = await driver.traverse_graph(
            start_nodes=start_nodes,
            algorithm="bfs",
            max_depth=2
        )
        
        logger.info(f"BFS traversal found:")
        logger.info(f"  - Nodes: {len(traversal_result['nodes'])}")
        logger.info(f"  - Edges: {len(traversal_result['edges'])}")
        logger.info(f"  - Paths: {len(traversal_result['paths'])}")


async def demonstrate_temporal_indices(driver: HuggingFaceDriver) -> None:
    """Demonstrate temporal index building and optimization"""
    logger.info("=== Demonstrating Temporal Index Optimization ===")
    
    if not driver.performance_optimizer:
        logger.warning("Performance optimization not enabled")
        return
    
    optimizer = driver.performance_optimizer
    
    # Build temporal indices
    logger.info("Building temporal indices...")
    temporal_indices = optimizer.build_temporal_indices()
    
    logger.info("Temporal indices built successfully")
    
    # Test temporal filtering
    logger.info("Testing temporal filtering...")
    
    # Get current time for filtering
    current_time = datetime.now()
    
    # Test temporal-based node filtering
    if 'node_partitions' in temporal_indices:
        # Get nodes from current year
        current_year = current_time.year
        year_key = f'year_{current_year}'
        
        if year_key in temporal_indices['node_partitions']:
            node_uuids = temporal_indices['node_partitions'][year_key]
            logger.info(f"Found {len(node_uuids)} nodes from {current_year}")
    
    # Test temporal-based edge filtering
    if 'edge_partitions' in temporal_indices:
        # Get edges from current year
        current_year = current_time.year
        year_key = f'created_year_{current_year}'
        
        if year_key in temporal_indices['edge_partitions']:
            edge_uuids = temporal_indices['edge_partitions'][year_key]
            logger.info(f"Found {len(edge_uuids)} edges created in {current_year}")


async def demonstrate_performance_monitoring(driver: HuggingFaceDriver) -> None:
    """Demonstrate performance monitoring and benchmarking"""
    logger.info("=== Demonstrating Performance Monitoring ===")
    
    if not driver.performance_optimizer:
        logger.warning("Performance optimization not enabled")
        return
    
    optimizer = driver.performance_optimizer
    
    # Get performance statistics
    logger.info("Getting performance statistics...")
    stats = optimizer.get_index_statistics()
    
    logger.info("Performance Statistics:")
    logger.info(json.dumps(stats, indent=2, default=str))
    
    # Benchmark search performance
    logger.info("Benchmarking search performance...")
    
    test_queries = [
        "machine learning",
        "neural networks",
        "python programming",
        "data science"
    ]
    
    benchmark_results = []
    
    for query in test_queries:
        start_time = time.time()
        
        # Perform hybrid search
        results = await driver.search_hybrid(
            query=query,
            limit=10,
            semantic_weight=0.4,
            keyword_weight=0.3,
            graph_weight=0.3
        )
        
        end_time = time.time()
        query_time = end_time - start_time
        
        benchmark_results.append({
            "query": query,
            "results_count": len(results),
            "query_time": query_time,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Query '{query}': {len(results)} results in {query_time:.3f}s")
    
    # Store benchmark results
    optimizer.performance_metrics['search_benchmark'].extend([
        PerformanceMetrics(
            query_time=result['query_time'],
            accuracy=len(result['results']) / 10.0,  # Assuming max 10 results
            index_size=len(driver.edges_df)
        )
        for result in benchmark_results
    ])
    
    logger.info("Benchmarking completed")


async def demonstrate_index_management(driver: HuggingFaceDriver) -> None:
    """Demonstrate index management and versioning"""
    logger.info("=== Demonstrating Index Management ===")
    
    if not driver.performance_optimizer:
        logger.warning("Performance optimization not enabled")
        return
    
    optimizer = driver.performance_optimizer
    
    # Create index version
    logger.info("Creating index version...")
    version_info = optimizer.index_versioning('create_version')
    logger.info(f"Created version: v{version_info['version']}")
    
    # List available versions
    logger.info("Listing available versions...")
    versions = optimizer.index_versioning('list_versions')
    logger.info(f"Available versions: {len(versions['available_versions'])}")
    
    for version in versions['available_versions']:
        logger.info(f"  - Version {version['version']}: {version.get('timestamp', 'N/A')}")
    
    # Test index saving and loading
    logger.info("Testing index persistence...")
    
    # Save indices
    save_path = "temp_test_indices"
    optimizer.save_index('all', save_path)
    logger.info(f"Indices saved to {save_path}")
    
    # Load indices
    optimizer.load_index('all', save_path)
    logger.info("Indices loaded successfully")
    
    # Clean up
    import shutil
    import os
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        logger.info("Cleaned up temporary files")


async def demonstrate_auto_optimization(driver: HuggingFaceDriver) -> None:
    """Demonstrate automatic optimization features"""
    logger.info("=== Demonstrating Automatic Optimization ===")
    
    if not driver.performance_optimizer:
        logger.warning("Performance optimization not enabled")
        return
    
    optimizer = driver.performance_optimizer
    
    # Test automatic index optimization
    logger.info("Testing automatic index optimization...")
    optimization_results = optimizer.optimize_index_parameters()
    
    logger.info("Optimization Results:")
    logger.info(json.dumps(optimization_results, indent=2, default=str))
    
    # Test automatic index rebuilding
    logger.info("Testing automatic index rebuilding...")
    rebuild_results = await driver.auto_rebuild_indices(threshold=0.8)
    
    logger.info("Auto Rebuild Results:")
    logger.info(json.dumps(rebuild_results, indent=2, default=str))
    
    # Test cleanup
    logger.info("Testing index cleanup...")
    cleanup_results = optimizer.cleanup_unused_indices(retention_days=7)
    
    logger.info("Cleanup Results:")
    logger.info(f"  - Removed versions: {len(cleanup_results['removed_versions'])}")
    logger.info(f"  - Freed space: {cleanup_results['freed_space']} bytes")


async def demonstrate_comprehensive_workflow(driver: HuggingFaceDriver) -> None:
    """Demonstrate a comprehensive performance optimization workflow"""
    logger.info("=== Demonstrating Comprehensive Performance Optimization Workflow ===")
    
    if not driver.performance_optimizer:
        logger.warning("Performance optimization not enabled")
        return
    
    optimizer = driver.performance_optimizer
    
    # Step 1: Build all indices
    logger.info("Step 1: Building all indices...")
    start_time = time.time()
    
    all_indices = optimizer.rebuild_all_indices()
    build_time = time.time() - start_time
    
    logger.info(f"All indices built in {build_time:.2f} seconds")
    logger.info(f"Index types built: {list(all_indices.keys())}")
    
    # Step 2: Get performance baseline
    logger.info("Step 2: Getting performance baseline...")
    baseline_stats = optimizer.get_index_statistics()
    
    logger.info("Baseline Statistics:")
    logger.info(f"  - Total indices: {len(all_indices)}")
    logger.info(f"  - Node count: {len(driver.nodes_df)}")
    logger.info(f"  - Edge count: {len(driver.edges_df)}")
    
    # Step 3: Perform benchmark searches
    logger.info("Step 3: Performing benchmark searches...")
    benchmark_queries = [
        "machine learning neural networks",
        "python data science",
        "ai deep learning",
        "natural language processing"
    ]
    
    benchmark_results = []
    
    for query in benchmark_queries:
        start_time = time.time()
        
        # Perform hybrid search
        results = await driver.search_hybrid(
            query=query,
            limit=10,
            semantic_weight=0.4,
            keyword_weight=0.3,
            graph_weight=0.3
        )
        
        end_time = time.time()
        query_time = end_time - start_time
        
        benchmark_results.append({
            "query": query,
            "results_count": len(results),
            "query_time": query_time,
            "avg_score": sum(r.get('combined_score', 0) for r in results) / len(results) if results else 0
        })
        
        logger.info(f"  '{query}': {len(results)} results in {query_time:.3f}s")
    
    # Step 4: Optimize indices based on usage
    logger.info("Step 4: Optimizing indices based on usage patterns...")
    optimization_results = optimizer.optimize_index_parameters()
    
    logger.info("Optimization Results:")
    logger.info(f"  - Recommended actions: {len(optimization_results.get('recommendations', []))}")
    logger.info(f"  - Performance improvement: {optimization_results.get('estimated_improvement', 0):.1%}")
    
    # Step 5: Create version snapshot
    logger.info("Step 5: Creating version snapshot...")
    version_info = optimizer.index_versioning('create_version')
    logger.info(f"Created version snapshot: v{version_info['version']}")
    
    # Step 6: Generate performance report
    logger.info("Step 6: Generating performance report...")
    
    performance_report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_stats": baseline_stats,
        "benchmark_results": benchmark_results,
        "optimization_results": optimization_results,
        "version_info": version_info,
        "recommendations": [
            "Consider using HNSW indices for large-scale vector search",
            "Enable auto-optimization for better performance over time",
            "Regular cleanup of old versions to save storage space",
            "Monitor query patterns to further optimize index selection"
        ]
    }
    
    # Save performance report
    report_path = "performance_optimization_report.json"
    with open(report_path, 'w') as f:
        json.dump(performance_report, f, indent=2, default=str)
    
    logger.info(f"Performance report saved to {report_path}")
    
    logger.info("=== Comprehensive Performance Optimization Workflow Completed ===")


async def main():
    """Main function demonstrating performance optimization features"""
    logger.info("Starting Graphiti-HF Performance Optimization Example")
    
    # Initialize HuggingFaceDriver with performance optimization enabled
    repo_id = "test-performance-optimization"
    
    try:
        # Create driver with performance optimization
        driver = HuggingFaceDriver(
            repo_id=repo_id,
            enable_vector_search=True,
            enable_performance_optimization=True,
            performance_optimizer_config={
                "auto_optimize": True,
                "optimization_interval": 3600,
                "performance_threshold": 0.8,
                "cache_size": 10000
            }
        )
        
        logger.info(f"Initialized HuggingFaceDriver for repository: {repo_id}")
        
        # Create sample data
        await create_sample_data(driver)
        
        # Demonstrate different optimization features
        await demonstrate_text_indices(driver)
        await demonstrate_vector_indices(driver)
        await demonstrate_graph_indices(driver)
        await demonstrate_temporal_indices(driver)
        await demonstrate_performance_monitoring(driver)
        await demonstrate_index_management(driver)
        await demonstrate_auto_optimization(driver)
        
        # Demonstrate comprehensive workflow
        await demonstrate_comprehensive_workflow(driver)
        
        logger.info("Performance optimization example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in performance optimization example: {e}")
        raise
    
    finally:
        # Clean up test repository
        try:
            logger.info("Cleaning up test repository...")
            # Note: In production, you might want to keep the repository for inspection
            # For this example, we'll clean it up
            await driver.delete_all_indexes()
            logger.info("Test repository cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up test repository: {e}")


if __name__ == "__main__":
    asyncio.run(main())