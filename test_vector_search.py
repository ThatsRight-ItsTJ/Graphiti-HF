#!/usr/bin/env python3
"""
Test script for vector search functionality in Graphiti-HF
"""

import asyncio
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphiti_hf.search.vector_search import VectorSearchEngine, SearchConfig, IndexType
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge


def test_vector_search_engine():
    """Test the VectorSearchEngine class"""
    print("🧪 Testing VectorSearchEngine...")
    
    # Create test embeddings
    embed_dim = 384
    n_samples = 100
    
    # Generate random embeddings
    import numpy as np
    embeddings = np.random.random((n_samples, embed_dim)).astype('float32')
    id_map = [f"entity_{i}" for i in range(n_samples)]
    
    # Initialize search engine
    config = SearchConfig(
        index_type=IndexType.FLAT,
        k=5,
        similarity_threshold=0.5
    )
    
    search_engine = VectorSearchEngine(embed_dim=embed_dim, config=config)
    
    # Build index
    print("  🔨 Building index...")
    index = search_engine.build_index(embeddings, id_map)
    
    # Test search
    print("  🔍 Testing search...")
    query_embedding = np.random.random(embed_dim).astype('float32')
    results = search_engine.semantic_search(query_embedding, index, id_map)
    
    print(f"  ✅ Found {len(results)} results")
    assert len(results) <= 5, "Should return at most k results"
    
    # Test batch search
    print("  📦 Testing batch search...")
    query_embeddings = np.random.random((5, embed_dim)).astype('float32')
    batch_results = search_engine.batch_search(query_embeddings, index, id_map)
    
    print(f"  ✅ Batch search returned {len(batch_results)} result groups")
    assert len(batch_results) == 5, "Should return 5 result groups"
    
    # Test index save/load
    print("  💾 Testing save/load...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        search_engine.save_index(index, id_map, {"test": "data"}, tmpdir)
        
        loaded_index, loaded_id_map, loaded_metadata = search_engine.load_index(tmpdir)
        
        # Test that loaded index works
        loaded_results = search_engine.semantic_search(query_embedding, loaded_index, loaded_id_map)
        assert len(loaded_results) == len(results), "Loaded index should return same results"
    
    print("  ✅ VectorSearchEngine tests passed!")


async def test_huggingface_driver():
    """Test the HuggingFaceDriver with vector search"""
    print("\n🧪 Testing HuggingFaceDriver with vector search...")
    
    # Use a temporary repo for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_id = f"test-driver-{os.path.basename(tmpdir)}"
        
        # Initialize driver
        print("  🚀 Initializing driver...")
        config = SearchConfig(
            index_type=IndexType.FLAT,
            k=3,
            similarity_threshold=0.5
        )
        
        driver = HuggingFaceDriver(
            repo_id=repo_id,
            enable_vector_search=True,
            vector_search_config=config,
            create_repo=True
        )
        
        # Create test nodes
        print("  📝 Creating test nodes...")
        nodes = []
        for i in range(5):
            node = EntityNode(
                name=f"Test Node {i}",
                labels=["Test"],
                group_id="test_group",
                name_embedding=[i/10, (i+1)/10, (i+2)/10] * 128  # 384 dim
            )
            nodes.append(node)
        
        # Save nodes
        for node in nodes:
            await driver.save_node(node)
        
        # Test vector search
        print("  🔍 Testing vector search...")
        query_embedding = [0.05, 0.15, 0.25] * 128
        similar_nodes = await driver.query_nodes_by_embedding(query_embedding, k=3)
        
        print(f"  ✅ Found {len(similar_nodes)} similar nodes")
        assert len(similar_nodes) <= 3, "Should return at most k results"
        
        # Test edge creation and search
        print("  🔗 Testing edge creation...")
        if len(nodes) >= 2:
            edge = EntityEdge(
                source_node_uuid=nodes[0].uuid,
                target_node_uuid=nodes[1].uuid,
                fact="Test relationship",
                fact_embedding=[0.1, 0.2, 0.3] * 128,
                group_id="test_group"
            )
            await driver.save_edge(edge)
            
            # Test edge search
            query_embedding = [0.1, 0.2, 0.3] * 128
            similar_edges = await driver.query_edges_by_embedding(query_embedding, k=3)
            
            print(f"  ✅ Found {len(similar_edges)} similar edges")
            assert len(similar_edges) <= 3, "Should return at most k results"
        
        # Test vector search stats
        print("  📊 Getting vector search stats...")
        stats = driver.get_vector_search_stats()
        print(f"  ✅ Stats: {stats}")
        assert stats["enabled"], "Vector search should be enabled"
        
        print("  ✅ HuggingFaceDriver tests passed!")


async def test_integration():
    """Test integration between components"""
    print("\n🧪 Testing Integration...")
    
    # Test that all components can work together
    from graphiti_hf.search.vector_search import VectorSearchEngine
    
    # Create a mock driver-like object
    class MockDriver:
        def __init__(self):
            self.nodes_df = pd.DataFrame()
            self.edges_df = pd.DataFrame()
            self.vector_search_engine = VectorSearchEngine()
    
    # Test instantiation
    driver = MockDriver()
    assert driver.vector_search_engine is not None
    
    print("  ✅ Integration tests passed!")


async def main():
    """Run all tests"""
    print("🧪 Starting Graphiti-HF Vector Search Tests")
    print("=" * 50)
    
    try:
        # Test VectorSearchEngine
        test_vector_search_engine()
        
        # Test HuggingFaceDriver
        await test_huggingface_driver()
        
        # Test integration
        await test_integration()
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues in test_vector_search_engine
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)