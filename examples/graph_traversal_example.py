"""
Graph Traversal Example for Graphiti-HF

This example demonstrates how to use the graph traversal algorithms (BFS/DFS)
implemented in Graphiti-HF for exploring knowledge graphs stored in HuggingFace datasets.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

# Import Graphiti-HF components
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver
from graphiti_hf.search.graph_traversal import TraversalConfig, TraversalAlgorithm, EdgeFilterType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_sample_graph(driver: HuggingFaceDriver):
    """Create a sample knowledge graph for demonstration"""
    logger.info("Creating sample knowledge graph...")
    
    # Create sample nodes
    nodes = [
        {
            "uuid": "node1",
            "name": "Alice",
            "labels": ["Person"],
            "group_id": "demo",
            "created_at": datetime.now(),
            "summary": "Software engineer",
            "attributes": {"age": 30, "occupation": "Software Engineer"}
        },
        {
            "uuid": "node2", 
            "name": "Bob",
            "labels": ["Person"],
            "group_id": "demo",
            "created_at": datetime.now(),
            "summary": "Data scientist",
            "attributes": {"age": 28, "occupation": "Data Scientist"}
        },
        {
            "uuid": "node3",
            "name": "Charlie",
            "labels": ["Person"], 
            "group_id": "demo",
            "created_at": datetime.now(),
            "summary": "Product manager",
            "attributes": {"age": 35, "occupation": "Product Manager"}
        },
        {
            "uuid": "node4",
            "name": "TechCorp",
            "labels": ["Company"],
            "group_id": "demo",
            "created_at": datetime.now(),
            "summary": "Technology company",
            "attributes": {"industry": "Technology", "size": "Large"}
        },
        {
            "uuid": "node5",
            "name": "AI Project",
            "labels": ["Project"],
            "group_id": "demo", 
            "created_at": datetime.now(),
            "summary": "Machine learning project",
            "attributes": {"status": "Active", "priority": "High"}
        }
    ]
    
    # Create sample edges
    edges = [
        {
            "uuid": "edge1",
            "source_uuid": "node1",
            "target_uuid": "node2", 
            "name": "COLLABORATES_WITH",
            "fact": "Alice collaborates with Bob on machine learning projects",
            "group_id": "demo",
            "created_at": datetime.now(),
            "fact_embedding": [0.1, 0.2, 0.3, 0.4, 0.5],  # Mock embedding
            "episodes": ["episode1"],
            "valid_at": datetime.now()
        },
        {
            "uuid": "edge2",
            "source_uuid": "node2",
            "target_uuid": "node3",
            "name": "REPORTS_TO", 
            "fact": "Bob reports to Charlie",
            "group_id": "demo",
            "created_at": datetime.now(),
            "fact_embedding": [0.2, 0.3, 0.4, 0.5, 0.6],  # Mock embedding
            "episodes": ["episode2"],
            "valid_at": datetime.now()
        },
        {
            "uuid": "edge3",
            "source_uuid": "node1",
            "target_uuid": "node4",
            "name": "WORKS_AT",
            "fact": "Alice works at TechCorp",
            "group_id": "demo",
            "created_at": datetime.now(),
            "fact_embedding": [0.3, 0.4, 0.5, 0.6, 0.7],  # Mock embedding
            "episodes": ["episode3"],
            "valid_at": datetime.now()
        },
        {
            "uuid": "edge4",
            "source_uuid": "node2",
            "target_uuid": "node4",
            "name": "WORKS_AT",
            "fact": "Bob works at TechCorp",
            "group_id": "demo",
            "created_at": datetime.now(),
            "fact_embedding": [0.4, 0.5, 0.6, 0.7, 0.8],  # Mock embedding
            "episodes": ["episode4"],
            "valid_at": datetime.now()
        },
        {
            "uuid": "edge5",
            "source_uuid": "node1",
            "target_uuid": "node5",
            "name": "LEADS",
            "fact": "Alice leads the AI Project",
            "group_id": "demo",
            "created_at": datetime.now(),
            "fact_embedding": [0.5, 0.6, 0.7, 0.8, 0.9],  # Mock embedding
            "episodes": ["episode5"],
            "valid_at": datetime.now()
        },
        {
            "uuid": "edge6",
            "source_uuid": "node2",
            "target_uuid": "node5",
            "name": "CONTRIBUTES_TO",
            "fact": "Bob contributes to the AI Project",
            "group_id": "demo",
            "created_at": datetime.now(),
            "fact_embedding": [0.6, 0.7, 0.8, 0.9, 1.0],  # Mock embedding
            "episodes": ["episode6"],
            "valid_at": datetime.now()
        }
    ]
    
    # Save nodes and edges
    from graphiti_core.nodes import EntityNode
    from graphiti_core.edges import EntityEdge
    
    entity_nodes = [EntityNode(**node) for node in nodes]
    entity_edges = [EntityEdge(**edge) for edge in edges]
    
    await driver.save_nodes(entity_nodes)
    await driver.save_edges(entity_edges)
    
    logger.info(f"Created {len(nodes)} nodes and {len(edges)} edges")
    return entity_nodes, entity_edges


async def example_basic_traversal():
    """Example of basic graph traversal using BFS and DFS"""
    logger.info("=== Basic Traversal Example ===")
    
    # Initialize driver
    driver = HuggingFaceDriver("demo/graph-traversal-basic", create_repo=True)
    
    # Create sample graph
    await create_sample_graph(driver)
    
    # Start from Alice (node1)
    start_nodes = ["node1"]
    
    # BFS Traversal
    logger.info("Performing BFS traversal...")
    bfs_result = await driver.traverse_graph(
        start_nodes=start_nodes,
        algorithm="bfs",
        max_depth=3
    )
    
    print(f"BFS Results:")
    print(f"  Nodes found: {len(bfs_result['nodes'])}")
    print(f"  Edges found: {len(bfs_result['edges'])}")
    print(f"  Paths: {len(bfs_result['paths'])}")
    print(f"  Stats: {bfs_result['stats']}")
    
    # DFS Traversal
    logger.info("Performing DFS traversal...")
    dfs_result = await driver.traverse_graph(
        start_nodes=start_nodes,
        algorithm="dfs",
        max_depth=3
    )
    
    print(f"\nDFS Results:")
    print(f"  Nodes found: {len(dfs_result['nodes'])}")
    print(f"  Edges found: {len(dfs_result['edges'])}")
    print(f"  Paths: {len(dfs_result['paths'])}")
    print(f"  Stats: {dfs_result['stats']}")


async def example_path_finding():
    """Example of finding paths between nodes"""
    logger.info("=== Path Finding Example ===")
    
    # Initialize driver
    driver = HuggingFaceDriver("demo/graph-traversal-paths", create_repo=True)
    
    # Create sample graph
    await create_sample_graph(driver)
    
    # Find paths from Alice to Charlie
    start_nodes = ["node1"]  # Alice
    target_nodes = ["node3"]  # Charlie
    
    paths = await driver.find_paths(
        start_nodes=start_nodes,
        target_nodes=target_nodes,
        max_depth=5
    )
    
    print(f"Paths from Alice to Charlie: {len(paths)}")
    for i, path in enumerate(paths):
        print(f"  Path {i+1}: {' -> '.join(path)}")


async def example_neighbor_retrieval():
    """Example of retrieving neighbors of nodes"""
    logger.info("=== Neighbor Retrieval Example ===")
    
    # Initialize driver
    driver = HuggingFaceDriver("demo/graph-traversal-neighbors", create_repo=True)
    
    # Create sample graph
    await create_sample_graph(driver)
    
    # Get neighbors of Alice
    node_uuids = ["node1"]  # Alice
    
    neighbors = await driver.get_neighbors(
        node_uuids=node_uuids,
        depth=2,
        edge_filter="all"
    )
    
    print(f"Neighbors of Alice:")
    for node_uuid, edge_info in neighbors.items():
        print(f"  Node {node_uuid}:")
        for edge_data in edge_info:
            print(f"    - Edge: {edge_data['edge']['name']}")
            print(f"      Fact: {edge_data['edge']['fact']}")
            print(f"      Connected to: {edge_data['target_node']}")


async def example_subgraph_extraction():
    """Example of extracting subgraphs"""
    logger.info("=== Subgraph Extraction Example ===")
    
    # Initialize driver
    driver = HuggingFaceDriver("demo/graph-traversal-subgraph", create_repo=True)
    
    # Create sample graph
    await create_sample_graph(driver)
    
    # Extract subgraph containing Alice and the AI Project
    node_uuids = ["node1", "node5"]  # Alice and AI Project
    
    subgraph = await driver.extract_subgraph(
        node_uuids=node_uuids,
        max_depth=2
    )
    
    print(f"Subgraph containing Alice and AI Project:")
    print(f"  Nodes: {len(subgraph['nodes'])}")
    for node in subgraph['nodes']:
        print(f"    - {node['name']} ({node['labels']})")
    
    print(f"  Edges: {len(subgraph['edges'])}")
    for edge in subgraph['edges']:
        print(f"    - {edge['name']}: {edge['fact']}")


async def example_advanced_traversal():
    """Example of advanced traversal with filtering and configuration"""
    logger.info("=== Advanced Traversal Example ===")
    
    # Initialize driver
    driver = HuggingFaceDriver("demo/graph-traversal-advanced", create_repo=True)
    
    # Create sample graph
    await create_sample_graph(driver)
    
    # Advanced BFS with edge type filtering
    start_nodes = ["node1"]  # Alice
    
    advanced_result = await driver.traverse_graph(
        start_nodes=start_nodes,
        algorithm="bfs",
        max_depth=3,
        edge_filter="all",
        edge_types=["WORKS_AT", "LEADS"],  # Only include these edge types
        early_termination_size=10  # Stop after finding 10 nodes
    )
    
    print(f"Advanced BFS Results:")
    print(f"  Nodes found: {len(advanced_result['nodes'])}")
    print(f"  Edges found: {len(advanced_result['edges'])}")
    
    # Filter results to show only WORKS_AT edges
    works_at_edges = [edge for edge in advanced_result['edges'] if edge['name'] == 'WORKS_AT']
    print(f"  WORKS_AT edges: {len(works_at_edges)}")
    for edge in works_at_edges:
        print(f"    - {edge['fact']}")


async def example_batch_traversal():
    """Example of batch traversal operations"""
    logger.info("=== Batch Traversal Example ===")
    
    # Initialize driver
    driver = HuggingFaceDriver("demo/graph-traversal-batch", create_repo=True)
    
    # Create sample graph
    await create_sample_graph(driver)
    
    # Define multiple start node groups
    start_node_groups = [
        ["node1"],  # Alice
        ["node2"],  # Bob
        ["node3"]   # Charlie
    ]
    
    # Perform batch traversal
    batch_results = await driver.batch_traversal(
        start_node_groups=start_node_groups,
        algorithm="bfs",
        max_depth=2,
        edge_filter="all"
    )
    
    print(f"Batch Traversal Results:")
    for i, result in enumerate(batch_results):
        print(f"  Group {i+1} (starting with {start_node_groups[i][0]}):")
        print(f"    Nodes: {len(result['nodes'])}")
        print(f"    Edges: {len(result['edges'])}")
        print(f"    Paths: {len(result['paths'])}")


async def example_traversal_stats():
    """Example of getting traversal statistics"""
    logger.info("=== Traversal Statistics Example ===")
    
    # Initialize driver
    driver = HuggingFaceDriver("demo/graph-traversal-stats", create_repo=True)
    
    # Create sample graph
    await create_sample_graph(driver)
    
    # Get traversal statistics
    stats = driver.get_traversal_stats()
    
    print(f"Traversal Engine Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


async def main():
    """Main function to run all examples"""
    logger.info("Starting Graph Traversal Examples...")
    
    try:
        await example_basic_traversal()
        await example_path_finding()
        await example_neighbor_retrieval()
        await example_subgraph_extraction()
        await example_advanced_traversal()
        await example_batch_traversal()
        await example_traversal_stats()
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())