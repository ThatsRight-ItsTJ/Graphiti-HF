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
Hybrid Search Example for Graphiti-HF

This example demonstrates how to use the hybrid search functionality in Graphiti-HF,
which combines semantic, keyword, and graph-based ranking for optimal knowledge graph retrieval.

The example shows:
1. Basic hybrid search with configurable weights
2. Center-node based hybrid search
3. Batch hybrid search operations
4. Temporal filtering and edge type filtering
5. Search result analysis and visualization
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import Graphiti-HF components
from graphiti_hf import HuggingFaceDriver
from graphiti_hf.search.hybrid_search import HybridSearchConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_sample_knowledge_graph(driver: HuggingFaceDriver):
    """
    Create a sample knowledge graph for demonstration purposes.
    
    This creates a simple graph about technology companies, products, and relationships.
    """
    logger.info("Creating sample knowledge graph...")
    
    # Sample nodes (entities)
    nodes_data = [
        {
            "name": "Apple Inc.",
            "labels": ["Company", "Technology"],
            "group_id": "tech_companies",
            "attributes": {"industry": "Consumer Electronics", "founded": 1976}
        },
        {
            "name": "Microsoft",
            "labels": ["Company", "Technology"],
            "group_id": "tech_companies", 
            "attributes": {"industry": "Software", "founded": 1975}
        },
        {
            "name": "Google",
            "labels": ["Company", "Technology"],
            "group_id": "tech_companies",
            "attributes": {"industry": "Internet", "founded": 1998}
        },
        {
            "name": "iPhone",
            "labels": ["Product", "Device"],
            "group_id": "apple_products",
            "attributes": {"category": "Smartphone", "launch_year": 2007}
        },
        {
            "name": "Windows",
            "labels": ["Product", "Operating System"],
            "group_id": "microsoft_products",
            "attributes": {"category": "OS", "launch_year": 1985}
        },
        {
            "name": "Android",
            "labels": ["Product", "Operating System"],
            "group_id": "google_products",
            "attributes": {"category": "OS", "launch_year": 2008}
        },
        {
            "name": "Tim Cook",
            "labels": ["Person", "CEO"],
            "group_id": "executives",
            "attributes": {"company": "Apple Inc.", "position": "CEO"}
        },
        {
            "name": "Satya Nadella",
            "labels": ["Person", "CEO"],
            "group_id": "executives",
            "attributes": {"company": "Microsoft", "position": "CEO"}
        },
        {
            "name": "Sundar Pichai",
            "labels": ["Person", "CEO"],
            "group_id": "executives",
            "attributes": {"company": "Google", "position": "CEO"}
        }
    ]
    
    # Sample edges (relationships)
    edges_data = [
        {
            "source": "Apple Inc.",
            "target": "iPhone",
            "name": "PRODUCES",
            "fact": "Apple Inc. produces iPhone smartphones",
            "group_id": "relationships",
            "episodes": ["company_announcement_2007"]
        },
        {
            "source": "Microsoft",
            "target": "Windows",
            "name": "PRODUCES",
            "fact": "Microsoft develops and sells Windows operating systems",
            "group_id": "relationships",
            "episodes": ["product_launch_1985"]
        },
        {
            "source": "Google",
            "target": "Android",
            "name": "PRODUCES",
            "fact": "Google created the Android mobile operating system",
            "group_id": "relationships",
            "episodes": ["acquisition_2005"]
        },
        {
            "source": "Tim Cook",
            "target": "Apple Inc.",
            "name": "WORKS_FOR",
            "fact": "Tim Cook is the CEO of Apple Inc.",
            "group_id": "relationships",
            "episodes": ["ceo_appointment_2011"]
        },
        {
            "source": "Satya Nadella",
            "target": "Microsoft",
            "name": "WORKS_FOR",
            "fact": "Satya Nadella is the CEO of Microsoft",
            "group_id": "relationships",
            "episodes": ["ceo_appointment_2014"]
        },
        {
            "source": "Sundar Pichai",
            "target": "Google",
            "name": "WORKS_FOR",
            "fact": "Sundar Pichai is the CEO of Google",
            "group_id": "relationships",
            "episodes": ["ceo_appointment_2015"]
        },
        {
            "source": "Apple Inc.",
            "target": "Microsoft",
            "name": "COMPETES_WITH",
            "fact": "Apple Inc. competes with Microsoft in the technology market",
            "group_id": "relationships",
            "episodes": ["market_analysis_2020"]
        },
        {
            "source": "Google",
            "target": "Apple Inc.",
            "name": "COMPETES_WITH",
            "fact": "Google competes with Apple Inc. in mobile and cloud services",
            "group_id": "relationships",
            "episodes": ["market_analysis_2020"]
        },
        {
            "source": "Microsoft",
            "target": "Google",
            "name": "COMPETES_WITH",
            "fact": "Microsoft competes with Google in cloud services and search",
            "group_id": "relationships",
            "episodes": ["market_analysis_2020"]
        }
    ]
    
    # Create nodes
    from graphiti_core.nodes import EntityNode
    from graphiti_core.edges import EntityEdge
    
    created_nodes = []
    for node_data in nodes_data:
        node = EntityNode(
            name=node_data["name"],
            labels=node_data["labels"],
            group_id=node_data["group_id"],
            attributes=node_data["attributes"]
        )
        await driver.save_node(node)
        created_nodes.append(node)
    
    # Create edges
    created_edges = []
    for edge_data in edges_data:
        # Find source and target nodes
        source_node = next((n for n in created_nodes if n.name == edge_data["source"]), None)
        target_node = next((n for n in created_nodes if n.name == edge_data["target"]), None)
        
        if source_node and target_node:
            edge = EntityEdge(
                source_node_uuid=source_node.uuid,
                target_node_uuid=target_node.uuid,
                name=edge_data["name"],
                fact=edge_data["fact"],
                group_id=edge_data["group_id"],
                episodes=edge_data["episodes"]
            )
            await driver.save_edge(edge)
            created_edges.append(edge)
    
    logger.info(f"Created {len(created_nodes)} nodes and {len(created_edges)} edges")
    return created_nodes, created_edges


async def demonstrate_basic_hybrid_search(driver: HuggingFaceDriver):
    """
    Demonstrate basic hybrid search functionality.
    """
    logger.info("\n=== Basic Hybrid Search Demo ===")
    
    # Test queries
    queries = [
        "smartphone company",
        "operating system",
        "tech company CEO",
        "mobile device"
    ]
    
    for query in queries:
        logger.info(f"\nSearching for: '{query}'")
        
        # Perform hybrid search with default weights
        results = await driver.search_hybrid(
            query=query,
            limit=5,
            semantic_weight=0.4,
            keyword_weight=0.3,
            graph_weight=0.3
        )
        
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result['edge'].fact}")
            logger.info(f"     Combined Score: {result['combined_score']:.3f}")
            logger.info(f"     Semantic: {result['semantic_score']:.3f}, "
                       f"Keyword: {result['keyword_score']:.3f}, "
                       f"Graph: {result['graph_score']:.3f}")
            logger.info("")


async def demonstrate_center_node_search(driver: HuggingFaceDriver):
    """
    Demonstrate center-node based hybrid search.
    """
    logger.info("\n=== Center-Node Hybrid Search Demo ===")
    
    # First, get a node to use as center
    nodes = await driver.get_nodes_by_group_ids(["tech_companies"], limit=1)
    if not nodes:
        logger.warning("No tech company nodes found for center node demo")
        return
    
    center_node = nodes[0]
    logger.info(f"Using center node: {center_node.name}")
    
    # Test queries with center node
    queries = [
        "smartphone",
        "operating system",
        "CEO"
    ]
    
    for query in queries:
        logger.info(f"\nSearching for '{query}' centered around {center_node.name}")
        
        # Perform hybrid search with center node
        results = await driver.search_with_center(
            query=query,
            center_node_uuid=center_node.uuid,
            limit=5,
            semantic_weight=0.3,
            keyword_weight=0.3,
            graph_weight=0.4  # Higher weight for graph proximity
        )
        
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result['edge'].fact}")
            logger.info(f"     Combined Score: {result['combined_score']:.3f}")
            logger.info(f"     Semantic: {result['semantic_score']:.3f}, "
                       f"Keyword: {result['keyword_score']:.3f}, "
                       f"Graph: {result['graph_score']:.3f}")
            logger.info("")


async def demonstrate_batch_search(driver: HuggingFaceDriver):
    """
    Demonstrate batch hybrid search functionality.
    """
    logger.info("\n=== Batch Hybrid Search Demo ===")
    
    # Multiple queries to search in batch
    queries = [
        "smartphone",
        "operating system", 
        "CEO",
        "company"
    ]
    
    logger.info(f"Performing batch search for {len(queries)} queries...")
    
    # Perform batch hybrid search
    all_results = await driver.batch_search_hybrid(
        queries=queries,
        limit=3,
        semantic_weight=0.4,
        keyword_weight=0.3,
        graph_weight=0.3
    )
    
    # Display results
    for i, (query, results) in enumerate(zip(queries, all_results)):
        logger.info(f"\nQuery {i+1}: '{query}'")
        logger.info(f"Found {len(results)} results:")
        for j, result in enumerate(results, 1):
            logger.info(f"  {j}. {result['edge'].fact}")
            logger.info(f"     Score: {result['combined_score']:.3f}")
        logger.info("")


async def demonstrate_filtering(driver: HuggingFaceDriver):
    """
    Demonstrate temporal and edge type filtering in hybrid search.
    """
    logger.info("\n=== Filtering Demo ===")
    
    # Temporal filtering - search for recent relationships
    recent_date = datetime.now() - timedelta(days=30)
    
    logger.info("Searching for recent relationships (last 30 days)...")
    results = await driver.search_hybrid(
        query="recent",
        limit=5,
        temporal_filter=recent_date,
        semantic_weight=0.5,
        keyword_weight=0.3,
        graph_weight=0.2
    )
    
    logger.info(f"Found {len(results)} recent results:")
    for result in results:
        logger.info(f"  - {result['edge'].fact}")
    
    # Edge type filtering - search only for specific relationship types
    logger.info("\nSearching only for PRODUCES relationships...")
    results = await driver.search_hybrid(
        query="product",
        limit=5,
        edge_types=["PRODUCES"],
        semantic_weight=0.4,
        keyword_weight=0.4,
        graph_weight=0.2
    )
    
    logger.info(f"Found {len(results)} PRODUCES relationships:")
    for result in results:
        logger.info(f"  - {result['edge'].fact}")


async def demonstrate_weight_optimization(driver: HuggingFaceDriver):
    """
    Demonstrate different weight configurations for hybrid search.
    """
    logger.info("\n=== Weight Optimization Demo ===")
    
    query = "smartphone"
    
    # Different weight configurations
    weight_configs = [
        {"name": "Semantic Heavy", "semantic": 0.7, "keyword": 0.2, "graph": 0.1},
        {"name": "Balanced", "semantic": 0.4, "keyword": 0.3, "graph": 0.3},
        {"name": "Keyword Heavy", "semantic": 0.2, "keyword": 0.7, "graph": 0.1},
        {"name": "Graph Heavy", "semantic": 0.2, "keyword": 0.2, "graph": 0.6}
    ]
    
    for config in weight_configs:
        logger.info(f"\n{config['name']} weights (S:{config['semantic']}, K:{config['keyword']}, G:{config['graph']}):")
        
        results = await driver.search_hybrid(
            query=query,
            limit=3,
            semantic_weight=config['semantic'],
            keyword_weight=config['keyword'],
            graph_weight=config['graph']
        )
        
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result['edge'].fact} (score: {result['combined_score']:.3f})")


async def demonstrate_search_stats(driver: HuggingFaceDriver):
    """
    Demonstrate getting search engine statistics.
    """
    logger.info("\n=== Search Statistics Demo ===")
    
    # Get hybrid search stats
    stats = driver.get_hybrid_search_stats()
    logger.info("Hybrid Search Engine Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Get vector search stats
    vector_stats = driver.get_vector_search_stats()
    logger.info("\nVector Search Engine Statistics:")
    for key, value in vector_stats.items():
        logger.info(f"  {key}: {value}")
    
    # Get traversal stats
    traversal_stats = driver.get_traversal_stats()
    logger.info("\nGraph Traversal Statistics:")
    for key, value in traversal_stats.items():
        logger.info(f"  {key}: {value}")


async def main():
    """
    Main function to run the hybrid search example.
    """
    logger.info("Starting Graphiti-HF Hybrid Search Example")
    
    # Initialize driver (using a temporary repo for demo)
    repo_id = "temp-hybrid-search-demo"
    
    try:
        # Create driver
        driver = HuggingFaceDriver(
            repo_id=repo_id,
            create_repo=True,
            enable_vector_search=True
        )
        
        # Create sample knowledge graph
        await create_sample_knowledge_graph(driver)
        
        # Demonstrate different search functionalities
        await demonstrate_basic_hybrid_search(driver)
        await demonstrate_center_node_search(driver)
        await demonstrate_batch_search(driver)
        await demonstrate_filtering(driver)
        await demonstrate_weight_optimization(driver)
        await demonstrate_search_stats(driver)
        
        logger.info("\n✅ Hybrid search example completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in hybrid search example: {e}")
        raise
    
    finally:
        # Clean up - delete the temporary repo
        try:
            logger.info("Cleaning up temporary repository...")
            # Note: In a real scenario, you might want to keep the data
            # For demo purposes, we'll clean up
            await driver.delete_all_indexes()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


if __name__ == "__main__":
    asyncio.run(main())