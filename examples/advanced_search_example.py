"""
Advanced Search Configuration Example for Graphiti-HF

This example demonstrates how to use the AdvancedSearchConfig class
to configure and optimize search operations in the Graphiti-HF system.

The example covers:
1. Creating advanced search configurations
2. Applying configurations to search engines
3. Using domain-specific optimizations
4. Performance tuning
5. Hybrid search with custom weights
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Import Graphiti-HF components
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver
from graphiti_hf.search.advanced_config import (
    AdvancedSearchConfig, 
    SearchMethod, 
    RankingStrategy,
    SemanticSearchConfig,
    GraphSearchConfig,
    HybridSearchConfig,
    PerformanceConfig,
    DomainConfig,
    TemporalConfig
)
from graphiti_hf.search.integration import (
    SearchEngineIntegrator,
    create_semantic_search_config,
    create_graph_search_config,
    create_hybrid_search_config,
    create_domain_specific_config
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_sample_data(driver: HuggingFaceDriver):
    """Create sample data for demonstration."""
    logger.info("Creating sample data...")
    
    # Create sample nodes
    nodes_data = [
        {
            "name": "Machine Learning",
            "labels": ["topic", "technical"],
            "summary": "Field of study that gives computers the ability to learn without being explicitly programmed.",
            "attributes": {"field": "AI", "difficulty": "advanced"}
        },
        {
            "name": "Natural Language Processing",
            "labels": ["topic", "technical"],
            "summary": "Subfield of linguistics, computer science, and AI concerned with interactions between computers and human language.",
            "attributes": {"field": "AI", "difficulty": "intermediate"}
        },
        {
            "name": "Company A",
            "labels": ["organization", "business"],
            "summary": "Technology company focused on AI solutions.",
            "attributes": {"industry": "technology", "size": "large"}
        },
        {
            "name": "John Doe",
            "labels": ["person", "personal"],
            "summary": "Software engineer with expertise in machine learning.",
            "attributes": {"role": "engineer", "experience": "5 years"}
        }
    ]
    
    # Create sample edges
    edges_data = [
        {
            "source_name": "Machine Learning",
            "target_name": "Natural Language Processing",
            "name": "related_to",
            "fact": "Machine Learning and Natural Language Processing are closely related fields in AI.",
            "attributes": {"strength": "strong", "type": "conceptual"}
        },
        {
            "source_name": "John Doe",
            "target_name": "Machine Learning",
            "name": "expert_in",
            "fact": "John Doe is an expert in Machine Learning.",
            "attributes": {"proficiency": "expert", "years": "3"}
        },
        {
            "source_name": "Company A",
            "target_name": "John Doe",
            "name": "employs",
            "fact": "Company A employs John Doe as a software engineer.",
            "attributes": {"position": "Senior Engineer", "start_date": "2022-01-15"}
        },
        {
            "source_name": "Company A",
            "target_name": "Machine Learning",
            "name": "uses",
            "fact": "Company A uses Machine Learning in their products.",
            "attributes": {"application": "product development", "impact": "high"}
        }
    ]
    
    # Add nodes
    for node_data in nodes_data:
        node = await driver.create_node(
            name=node_data["name"],
            labels=node_data["labels"],
            summary=node_data["summary"],
            attributes=node_data["attributes"]
        )
        logger.info(f"Created node: {node_data['name']}")
    
    # Add edges
    for edge_data in edges_data:
        # Get source and target nodes
        source_nodes = await driver.search_nodes(edge_data["source_name"], limit=1)
        target_nodes = await driver.search_nodes(edge_data["target_name"], limit=1)
        
        if source_nodes and target_nodes:
            source_uuid = source_nodes[0]["uuid"]
            target_uuid = target_nodes[0]["uuid"]
            
            edge = await driver.create_edge(
                source_node_uuid=source_uuid,
                target_node_uuid=target_uuid,
                name=edge_data["name"],
                fact=edge_data["fact"],
                attributes=edge_data["attributes"]
            )
            logger.info(f"Created edge: {edge_data['source_name']} -> {edge_data['target_name']}")


async def example_basic_configuration():
    """Demonstrate basic search configuration."""
    logger.info("=== Basic Configuration Example ===")
    
    # Initialize driver with advanced search enabled
    driver = HuggingFaceDriver(
        repo_id="test-advanced-search",
        create_repo=True,
        enable_advanced_search=True
    )
    
    # Create sample data
    await create_sample_data(driver)
    
    # Create an advanced search configuration
    config = AdvancedSearchConfig(name="basic_config")
    
    # Configure semantic search
    config.configure_semantic_search(
        index_type="hnsw",
        k=15,
        similarity_threshold=0.7,
        use_gpu=False
    )
    
    # Configure hybrid search
    config.configure_hybrid_search(
        semantic_weight=0.6,
        keyword_weight=0.3,
        graph_weight=0.1,
        result_limit=10
    )
    
    # Set search weights
    config.set_search_weights(
        semantic_weight=0.6,
        keyword_weight=0.3,
        graph_weight=0.1
    )
    
    # Apply configuration
    integrator = SearchEngineIntegrator(driver)
    results = integrator.apply_all_configs(config)
    
    logger.info("Configuration application results:")
    for engine, result in results.items():
        logger.info(f"  {engine}: {result['message']}")
    
    # Test search with the configuration
    query = "machine learning"
    search_results = await driver.search_hybrid(
        query=query,
        limit=5,
        semantic_weight=0.6,
        keyword_weight=0.3,
        graph_weight=0.1
    )
    
    logger.info(f"Search results for '{query}':")
    for i, result in enumerate(search_results, 1):
        logger.info(f"  {i}. {result.get('name', result.get('fact', 'Unknown'))} "
                   f"(score: {result.get('combined_score', 0):.3f})")
    
    return driver


async def example_domain_specific_configuration():
    """Demonstrate domain-specific search configuration."""
    logger.info("=== Domain-Specific Configuration Example ===")
    
    # Initialize driver
    driver = HuggingFaceDriver(
        repo_id="test-domain-search",
        create_repo=True,
        enable_advanced_search=True
    )
    
    # Create sample data
    await create_sample_data(driver)
    
    # Create domain-specific configurations
    technical_config = create_domain_specific_config("technical")
    business_config = create_domain_specific_config("business")
    personal_config = create_domain_specific_config("personal")
    
    logger.info("Domain-specific configurations created:")
    logger.info(f"  Technical: semantic_weight={technical_config.hybrid_config.semantic_weight:.2f}")
    logger.info(f"  Business: keyword_weight={business_config.hybrid_config.keyword_weight:.2f}")
    logger.info(f"  Personal: graph_weight={personal_config.hybrid_config.graph_weight:.2f}")
    
    # Apply technical configuration
    integrator = SearchEngineIntegrator(driver)
    integrator.apply_all_configs(technical_config)
    
    # Test search with technical domain focus
    query = "AI algorithms"
    search_results = await driver.search_hybrid(
        query=query,
        limit=5,
        semantic_weight=technical_config.hybrid_config.semantic_weight,
        keyword_weight=technical_config.hybrid_config.keyword_weight,
        graph_weight=technical_config.hybrid_config.graph_weight
    )
    
    logger.info(f"Technical domain search for '{query}':")
    for i, result in enumerate(search_results, 1):
        logger.info(f"  {i}. {result.get('name', result.get('fact', 'Unknown'))} "
                   f"(score: {result.get('combined_score', 0):.3f})")
    
    return driver


async def example_performance_optimization():
    """Demonstrate performance optimization configuration."""
    logger.info("=== Performance Optimization Example ===")
    
    # Initialize driver
    driver = HuggingFaceDriver(
        repo_id="test-performance-search",
        create_repo=True,
        enable_advanced_search=True
    )
    
    # Create sample data
    await create_sample_data(driver)
    
    # Create performance-optimized configuration
    config = AdvancedSearchConfig(name="performance_optimized")
    
    # Configure batch processing
    config.set_batch_sizes(
        vector_batch_size=50,
        traversal_batch_size=200,
        hybrid_batch_size=100
    )
    
    # Configure caching
    config.set_cache_settings(
        enabled=True,
        max_size=5000,
        ttl_seconds=3600
    )
    
    # Configure parallel processing
    config.set_parallel_settings(
        max_workers=4,
        enable_parallel=True
    )
    
    # Configure memory limits
    config.set_memory_limits(
        max_memory_mb=1024,
        enable_memory_monitoring=True
    )
    
    # Apply configuration
    integrator = SearchEngineIntegrator(driver)
    results = integrator.apply_all_configs(config)
    
    logger.info("Performance optimization applied:")
    for engine, result in results.items():
        logger.info(f"  {engine}: {result['message']}")
    
    # Test batch search
    queries = [
        "machine learning",
        "natural language processing",
        "AI algorithms",
        "deep learning"
    ]
    
    start_time = datetime.now()
    batch_results = await driver.batch_search_hybrid(queries, limit=3)
    end_time = datetime.now()
    
    logger.info(f"Batch search completed in {(end_time - start_time).total_seconds():.2f} seconds")
    logger.info(f"Processed {len(queries)} queries")
    
    for i, (query, results) in enumerate(zip(queries, batch_results)):
        logger.info(f"  Query '{query}': {len(results)} results")
    
    return driver


async def example_temporal_search():
    """Demonstrate temporal search configuration."""
    logger.info("=== Temporal Search Example ===")
    
    # Initialize driver
    driver = HuggingFaceDriver(
        repo_id="test-temporal-search",
        create_repo=True,
        enable_advanced_search=True
    )
    
    # Create sample data with temporal information
    logger.info("Creating temporal sample data...")
    
    # Create nodes with different timestamps
    nodes_data = [
        {
            "name": "Old Technology",
            "labels": ["topic"],
            "summary": "Technology from previous decade.",
            "created_at": datetime.now() - timedelta(days=365)
        },
        {
            "name": "Modern AI",
            "labels": ["topic"],
            "summary": "Current state-of-the-art AI technology.",
            "created_at": datetime.now() - timedelta(days=30)
        },
        {
            "name": "Future Tech",
            "labels": ["topic"],
            "summary": "Emerging technology trends.",
            "created_at": datetime.now() - timedelta(days=7)
        }
    ]
    
    for node_data in nodes_data:
        node = await driver.create_node(
            name=node_data["name"],
            labels=node_data["labels"],
            summary=node_data["summary"],
            created_at=node_data["created_at"]
        )
        logger.info(f"Created temporal node: {node_data['name']}")
    
    # Create temporal configuration
    config = AdvancedSearchConfig(name="temporal_config")
    
    # Configure temporal search
    config.configure_temporal_search(
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now(),
        temporal_weight=0.8,
        recency_weight=0.6
    )
    
    # Configure hybrid search with temporal focus
    config.configure_hybrid_search(
        semantic_weight=0.4,
        keyword_weight=0.3,
        graph_weight=0.3,
        temporal_filter=config.temporal_config.start_date
    )
    
    # Apply configuration
    integrator = SearchEngineIntegrator(driver)
    integrator.apply_all_configs(config)
    
    # Test temporal search
    query = "technology"
    search_results = await driver.search_hybrid(
        query=query,
        limit=5,
        semantic_weight=0.4,
        keyword_weight=0.3,
        graph_weight=0.3
    )
    
    logger.info(f"Temporal search for '{query}' (last 90 days):")
    for i, result in enumerate(search_results, 1):
        created_date = result.get('created_at', 'Unknown')
        logger.info(f"  {i}. {result.get('name', 'Unknown')} "
                   f"(created: {created_date}, score: {result.get('combined_score', 0):.3f})")
    
    return driver


async def example_custom_ranking():
    """Demonstrate custom ranking configuration."""
    logger.info("=== Custom Ranking Example ===")
    
    # Initialize driver
    driver = HuggingFaceDriver(
        repo_id="test-custom-ranking",
        create_repo=True,
        enable_advanced_search=True
    )
    
    # Create sample data
    await create_sample_data(driver)
    
    # Create custom ranking configuration
    config = AdvancedSearchConfig(name="custom_ranking")
    
    # Configure custom scoring
    config.set_custom_scoring(
        scoring_function="weighted_sum",
        weights={"semantic": 0.5, "keyword": 0.3, "graph": 0.2, "temporal": 0.0}
    )
    
    # Configure result ranking
    config.set_result_ranking(
        strategy=RankingStrategy.WEIGHTED_SUM,
        normalize_scores=True,
        apply_post_processing=True
    )
    
    # Configure similarity thresholds
    config.set_similarity_thresholds(
        semantic_threshold=0.6,
        keyword_threshold=0.4,
        graph_threshold=0.3
    )
    
    # Apply configuration
    integrator = SearchEngineIntegrator(driver)
    results = integrator.apply_all_configs(config)
    
    logger.info("Custom ranking configuration applied:")
    for engine, result in results.items():
        logger.info(f"  {engine}: {result['message']}")
    
    # Test search with custom ranking
    query = "AI expert"
    search_results = await driver.search_hybrid(
        query=query,
        limit=5,
        semantic_weight=0.5,
        keyword_weight=0.3,
        graph_weight=0.2
    )
    
    logger.info(f"Custom ranking search for '{query}':")
    for i, result in enumerate(search_results, 1):
        logger.info(f"  {i}. {result.get('name', result.get('fact', 'Unknown'))} "
                   f"(semantic: {result.get('semantic_score', 0):.3f}, "
                   f"keyword: {result.get('keyword_score', 0):.3f}, "
                   f"graph: {result.get('graph_score', 0):.3f})")
    
    return driver


async def example_configuration_presets():
    """Demonstrate using configuration presets."""
    logger.info("=== Configuration Presets Example ===")
    
    # Initialize driver
    driver = HuggingFaceDriver(
        repo_id="test-presets",
        create_repo=True,
        enable_advanced_search=True
    )
    
    # Create sample data
    await create_sample_data(driver)
    
    # Create preset configurations
    semantic_config = create_semantic_search_config()
    graph_config = create_graph_search_config()
    hybrid_config = create_hybrid_search_config()
    
    logger.info("Configuration presets created:")
    logger.info(f"  Semantic: {semantic_config.name}")
    logger.info(f"  Graph: {graph_config.name}")
    logger.info(f"  Hybrid: {hybrid_config.name}")
    
    # Test each preset
    presets = [
        ("Semantic Search", semantic_config),
        ("Graph Search", graph_config),
        ("Hybrid Search", hybrid_config)
    ]
    
    for preset_name, preset_config in presets:
        logger.info(f"\n--- Testing {preset_name} ---")
        
        # Apply preset configuration
        integrator = SearchEngineIntegrator(driver)
        integrator.apply_all_configs(preset_config)
        
        # Test search
        query = "learning"
        if "Semantic" in preset_name:
            results = await driver.search_nodes(query, limit=3)
        elif "Graph" in preset_name:
            # For graph search, we'll use a node-based search
            results = await driver.search_nodes(query, limit=3)
        else:
            results = await driver.search_hybrid(
                query=query,
                limit=3,
                semantic_weight=preset_config.hybrid_config.semantic_weight,
                keyword_weight=preset_config.hybrid_config.keyword_weight,
                graph_weight=preset_config.hybrid_config.graph_weight
            )
        
        logger.info(f"Results for '{query}' using {preset_name}:")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result.get('name', result.get('fact', 'Unknown'))}")
    
    return driver


async def example_configuration_management():
    """Demonstrate configuration management features."""
    logger.info("=== Configuration Management Example ===")
    
    # Initialize driver
    driver = HuggingFaceDriver(
        repo_id="test-config-management",
        create_repo=True,
        enable_advanced_search=True
    )
    
    # Create a comprehensive configuration
    config = AdvancedSearchConfig(name="comprehensive_config")
    
    # Configure all search methods
    config.configure_semantic_search(
        index_type="hnsw",
        k=20,
        similarity_threshold=0.7
    )
    
    config.configure_graph_search(
        max_depth=7,
        algorithm="bfs",
        weighted=True
    )
    
    config.configure_hybrid_search(
        semantic_weight=0.5,
        keyword_weight=0.3,
        graph_weight=0.2
    )
    
    # Configure performance
    config.set_batch_sizes(
        vector_batch_size=75,
        traversal_batch_size=150,
        hybrid_batch_size=100
    )
    
    config.set_cache_settings(
        enabled=True,
        max_size=10000,
        ttl_seconds=7200
    )
    
    # Configure domain settings
    config.set_domain_weights(
        technical=0.6,
        business=0.3,
        personal=0.1
    )
    
    # Save configuration
    save_result = driver.advanced_search_manager.save_search_config(config)
    logger.info(f"Configuration saved: {save_result}")
    
    # Load configuration
    loaded_config = driver.advanced_search_manager.load_search_config("comprehensive_config")
    if loaded_config:
        logger.info(f"Configuration loaded: {loaded_config.name}")
        logger.info(f"Semantic weight: {loaded_config.hybrid_config.semantic_weight}")
    
    # Optimize configuration
    optimization_result = driver.advanced_search_manager.optimize_search_config("comprehensive_config")
    logger.info(f"Configuration optimized: {optimization_result}")
    
    # List all saved configurations
    saved_configs = driver.advanced_search_manager.list_saved_configs()
    logger.info(f"Saved configurations: {saved_configs}")
    
    return driver


async def main():
    """Run all examples."""
    logger.info("Starting Advanced Search Configuration Examples")
    
    try:
        # Run all examples
        await example_basic_configuration()
        await example_domain_specific_configuration()
        await example_performance_optimization()
        await example_temporal_search()
        await example_custom_ranking()
        await example_configuration_presets()
        await example_configuration_management()
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        raise
    
    finally:
        logger.info("Examples completed")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())