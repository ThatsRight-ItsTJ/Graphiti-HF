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
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

from graphiti_hf import GraphitiHF
from graphiti_hf.search.vector_search import SearchConfig, IndexType
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.edges import EntityEdge

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_basic_vector_search():
    """Demonstrate basic vector search functionality"""
    print("🔍 Basic Vector Search Demo")
    print("=" * 50)
    
    # Initialize GraphitiHF with vector search enabled
    graphiti = GraphitiHF(
        repo_id="demo/vector-search-basic",
        enable_vector_search=True,
        vector_search_config=SearchConfig(
            index_type=IndexType.FLAT,
            k=5,
            similarity_threshold=0.7,
            use_gpu=False
        )
    )
    
    # Create sample nodes with embeddings
    nodes = [
        EntityNode(
            name="Machine Learning",
            labels=["Topic", "Technology"],
            group_id="demo",
            name_embedding=[0.1, 0.2, 0.3] * 128  # Simulated 384-dim embedding
        ),
        EntityNode(
            name="Deep Learning",
            labels=["Topic", "Technology", "AI"],
            group_id="demo",
            name_embedding=[0.2, 0.3, 0.4] * 128  # Simulated 384-dim embedding
        ),
        EntityNode(
            name="Natural Language Processing",
            labels=["Topic", "AI", "Language"],
            group_id="demo",
            name_embedding=[0.3, 0.4, 0.5] * 128  # Simulated 384-dim embedding
        ),
        EntityNode(
            name="Computer Vision",
            labels=["Topic", "AI", "Image"],
            group_id="demo",
            name_embedding=[0.4, 0.5, 0.6] * 128  # Simulated 384-dim embedding
        )
    ]
    
    # Create sample edges with embeddings
    edges = [
        EntityEdge(
            source_node_uuid=nodes[0].uuid,
            target_node_uuid=nodes[1].uuid,
            fact="Machine Learning is a subset of Deep Learning",
            fact_embedding=[0.15, 0.25, 0.35] * 128,  # Simulated embedding
            group_id="demo"
        ),
        EntityEdge(
            source_node_uuid=nodes[1].uuid,
            target_node_uuid=nodes[2].uuid,
            fact="Deep Learning includes Natural Language Processing",
            fact_embedding=[0.25, 0.35, 0.45] * 128,  # Simulated embedding
            group_id="demo"
        ),
        EntityEdge(
            source_node_uuid=nodes[1].uuid,
            target_node_uuid=nodes[3].uuid,
            fact="Deep Learning includes Computer Vision",
            fact_embedding=[0.3, 0.4, 0.5] * 128,  # Simulated embedding
            group_id="demo"
        )
    ]
    
    # Save nodes and edges
    print("📚 Saving nodes and edges...")
    for node in nodes:
        await graphiti.driver.save_node(node)
    
    for edge in edges:
        await graphiti.driver.save_edge(edge)
    
    # Get vector search stats
    stats = graphiti.driver.get_vector_search_stats()
    print(f"📊 Vector Search Stats: {json.dumps(stats, indent=2)}")
    
    # Perform node similarity search
    print("\n🔍 Node Similarity Search:")
    query_embedding = [0.15, 0.25, 0.35] * 128  # Similar to ML node
    similar_nodes = await graphiti.driver.query_nodes_by_embedding(
        query_embedding, k=3, similarity_threshold=0.5
    )
    
    for i, node in enumerate(similar_nodes):
        print(f"  {i+1}. {node.name} (similarity: N/A)")
    
    # Perform edge similarity search
    print("\n🔍 Edge Similarity Search:")
    query_embedding = [0.2, 0.3, 0.4] * 128  # Similar to DL-NLP edge
    similar_edges = await graphiti.driver.query_edges_by_embedding(
        query_embedding, k=3, similarity_threshold=0.5
    )
    
    for i, edge in enumerate(similar_edges):
        print(f"  {i+1}. {edge.fact[:50]}... (similarity: N/A)")
    
    # Perform batch search
    print("\n🔍 Batch Node Search:")
    query_embeddings = [
        [0.1, 0.2, 0.3] * 128,  # Similar to ML
        [0.4, 0.5, 0.6] * 128   # Similar to CV
    ]
    
    batch_results = await graphiti.driver.batch_query_nodes_by_embedding(
        query_embeddings, k=2, similarity_threshold=0.5
    )
    
    for i, results in enumerate(batch_results):
        print(f"  Query {i+1}:")
        for j, node in enumerate(results):
            print(f"    {j+1}. {node.name}")


async def demo_advanced_vector_search():
    """Demonstrate advanced vector search with different index types"""
    print("\n\n🚀 Advanced Vector Search Demo")
    print("=" * 50)
    
    # Test different index types
    index_types = [IndexType.FLAT, IndexType.IVFFLAT, IndexType.HNSW]
    
    for index_type in index_types:
        print(f"\n📊 Testing {index_type.value.upper()} Index:")
        
        # Initialize with specific index type
        graphiti = GraphitiHF(
            repo_id=f"demo/vector-search-{index_type.value}",
            enable_vector_search=True,
            vector_search_config=SearchConfig(
                index_type=index_type,
                k=3,
                similarity_threshold=0.6,
                use_gpu=False
            )
        )
        
        # Create sample data
        nodes = [
            EntityNode(
                name=f"Topic_{i}",
                labels=["Topic"],
                group_id="advanced_demo",
                name_embedding=[i/10, (i+1)/10, (i+2)/10] * 128
            )
            for i in range(10)
        ]
        
        # Save nodes
        for node in nodes:
            await graphiti.driver.save_node(node)
        
        # Query with similar embedding
        query_embedding = [0.05, 0.15, 0.25] * 128
        similar_nodes = await graphiti.driver.query_nodes_by_embedding(
            query_embedding, k=3, similarity_threshold=0.5
        )
        
        print(f"  Found {len(similar_nodes)} similar nodes:")
        for node in similar_nodes:
            print(f"    - {node.name}")


async def demo_episode_processing():
    """Demonstrate episode processing with vector search"""
    print("\n\n📖 Episode Processing with Vector Search Demo")
    print("=" * 50)
    
    graphiti = GraphitiHF(
        repo_id="demo/episode-vector-search",
        enable_vector_search=True,
        vector_search_config=SearchConfig(
            index_type=IndexType.FLAT,
            k=5,
            similarity_threshold=0.7
        )
    )
    
    # Add an episode
    episode_content = """
    Machine Learning is a subset of Artificial Intelligence that focuses on 
    algorithms that can learn from data. Deep Learning is a specialized form 
    of Machine Learning that uses neural networks with multiple layers. 
    Natural Language Processing applies ML techniques to understand and 
    generate human language.
    """
    
    print("📝 Adding episode...")
    result = await graphiti.add_episode(
        name="Introduction to ML Concepts",
        episode_body=episode_content,
        source_description="Educational material",
        reference_time=datetime.now()
    )
    
    print(f"✅ Added {len(result.nodes)} nodes and {len(result.edges)} edges")
    
    # Get all nodes and edges
    all_nodes = await graphiti.driver.get_nodes_by_group_ids(["default"])
    all_edges = await graphiti.driver.get_edges_by_group_ids(["default"])
    
    print(f"\n📊 Knowledge Graph contains:")
    print(f"  - {len(all_nodes)} nodes")
    print(f"  - {len(all_edges)} edges")
    
    # Show extracted entities
    print("\n🔍 Extracted Entities:")
    for node in all_nodes:
        if node.name_embedding:  # Only show nodes with embeddings
            print(f"  - {node.name} ({', '.join(node.labels)})")
    
    # Show relationships
    print("\n🔗 Extracted Relationships:")
    for edge in all_edges:
        if edge.fact_embedding:  # Only show edges with embeddings
            print(f"  - {edge.fact}")


async def demo_performance_comparison():
    """Compare performance of different search configurations"""
    print("\n\n⚡ Performance Comparison Demo")
    print("=" * 50)
    
    # Create a larger dataset
    print("📊 Creating test dataset with 100 nodes...")
    graphiti = GraphitiHF(
        repo_id="demo/performance-test",
        enable_vector_search=True,
        vector_search_config=SearchConfig(
            index_type=IndexType.FLAT,
            k=10,
            similarity_threshold=0.5
        )
    )
    
    # Generate 100 nodes with random embeddings
    import random
    nodes = []
    for i in range(100):
        embedding = [random.random() for _ in range(384)]
        node = EntityNode(
            name=f"Entity_{i}",
            labels=["Entity"],
            group_id="performance_test",
            name_embedding=embedding
        )
        nodes.append(node)
    
    # Save nodes in batches
    batch_size = 20
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]
        for node in batch:
            await graphiti.driver.save_node(node)
        print(f"  Saved batch {i//batch_size + 1}/{(len(nodes)-1)//batch_size + 1}")
    
    # Test different search configurations
    configs = [
        ("FLAT", IndexType.FLAT, 10),
        ("IVF", IndexType.IVFFLAT, 10),
        ("HNSW", IndexType.HNSW, 64)
    ]
    
    query_embedding = [0.5] * 384  # Fixed query
    
    for config_name, index_type, m_value in configs:
        print(f"\n🔍 Testing {config_name} configuration...")
        
        # Reinitialize with different config
        test_graphiti = GraphitiHF(
            repo_id="demo/performance-test",
            enable_vector_search=True,
            vector_search_config=SearchConfig(
                index_type=index_type,
                k=10,
                similarity_threshold=0.5,
                m=m_value if index_type == IndexType.HNSW else 32
            )
        )
        
        # Measure search time
        import time
        start_time = time.time()
        results = await test_graphiti.driver.query_nodes_by_embedding(
            query_embedding, k=10, similarity_threshold=0.5
        )
        search_time = time.time() - start_time
        
        print(f"  - Found {len(results)} results in {search_time:.4f} seconds")
        print(f"  - Index type: {test_graphiti.driver.vector_search_engine.config.index_type.value}")


async def main():
    """Run all demos"""
    print("🚀 Graphiti-HF Vector Search Demo")
    print("=" * 60)
    
    try:
        await demo_basic_vector_search()
        await demo_advanced_vector_search()
        await demo_episode_processing()
        await demo_performance_comparison()
        
        print("\n✅ All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())