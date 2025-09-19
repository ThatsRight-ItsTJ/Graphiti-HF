"""
Community Detection Example for Graphiti-HF

This example demonstrates how to use the community detection capabilities
in Graphiti-HF to identify and analyze communities within knowledge graphs.
"""

import asyncio
import logging
from datetime import datetime
from typing import List

from graphiti_hf import GraphitiHF
from graphiti_hf.analysis.community_detector import CommunityDetectionConfig
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_sample_knowledge_graph() -> GraphitiHF:
    """
    Create a sample knowledge graph with some interconnected entities
    for community detection demonstration.
    """
    # Initialize Graphiti-HF
    graphiti = GraphitiHF("test-community-detection", create_repo=True)
    
    # Sample data representing different domains
    sample_episodes = [
        {
            "name": "Technology Team",
            "content": """
            Alice Johnson is a Senior Machine Learning Engineer at Google. 
            She works on the recommendation system project with Bob Smith, 
            who is the Product Manager. Charlie Brown is the Data Scientist 
            who provides the analytics for the project.
            """,
            "source_description": "HR system update"
        },
        {
            "name": "Marketing Team", 
            "content": """
            David Wilson is the Marketing Director at Google. He works with 
            Emma Davis on social media campaigns and Frank Miller on content 
            strategy. Grace Lee is the Brand Manager who coordinates their efforts.
            """,
            "source_description": "Marketing department update"
        },
        {
            "name": "Cross-Team Collaboration",
            "content": """
            The recommendation system project (led by Alice Johnson and Bob Smith) 
            requires input from the marketing team. David Wilson's team provides 
            user feedback data, while Emma Davis helps with campaign integration.
            Frank Miller coordinates with Charlie Brown on data analysis.
            """,
            "source_description": "Project collaboration update"
        },
        {
            "name": "Engineering Team",
            "content": """
            Henry Garcia is a Senior Software Engineer working on the backend 
            infrastructure. He collaborates with Ivan Rodriguez on API design 
            and Julia Kim on database optimization. Kevin Lee is the DevOps 
            Engineer who manages the deployment pipeline.
            """,
            "source_description": "Engineering department update"
        },
        {
            "name": "Research Team",
            "content": """
            Linda Martinez is a Research Scientist working on AI algorithms. 
            She publishes papers with Michael Thompson and Nancy White. 
            Oliver Brown is the Research Manager who oversees their publications.
            """,
            "source_description": "Research department update"
        }
    ]
    
    # Add episodes to build the knowledge graph
    logger.info("Building knowledge graph with sample data...")
    for episode in sample_episodes:
        result = await graphiti.add_episode(
            name=episode["name"],
            episode_body=episode["content"],
            source_description=episode["source_description"],
            reference_time=datetime.now()
        )
        logger.info(f"Added {len(result.nodes)} nodes and {len(result.edges)} edges")
    
    return graphiti


async def demonstrate_basic_community_detection(graphiti: GraphitiHF):
    """
    Demonstrate basic community detection functionality.
    """
    logger.info("\n=== Basic Community Detection ===")
    
    # Detect communities using Louvain algorithm
    communities = await graphiti.detect_graph_communities(
        algorithm="louvain",
        commit_message="Initial community detection"
    )
    
    logger.info(f"Detected {len(communities[0])} communities")
    
    # Display community information
    for i, community_node in enumerate(communities[0]):
        logger.info(f"Community {i+1}: {community_node.name}")
        logger.info(f"  UUID: {community_node.uuid}")
        logger.info(f"  Summary: {community_node.summary}")
        
        # Get community members
        community_info = await graphiti.get_community_info(community_node.uuid)
        if "member_count" in community_info:
            logger.info(f"  Members: {community_info['member_count']}")
            logger.info(f"  Member UUIDs: {community_info.get('members', [])}")


async def demonstrate_algorithms_comparison(graphiti: GraphitiHF):
    """
    Compare different community detection algorithms.
    """
    logger.info("\n=== Algorithm Comparison ===")
    
    algorithms = ["louvain", "label_propagation", "connected_components", "kmeans"]
    
    for algorithm in algorithms:
        logger.info(f"\n--- {algorithm.upper()} Algorithm ---")
        
        try:
            communities = await graphiti.detect_graph_communities(
                algorithm=algorithm,
                commit_message=f"Community detection with {algorithm}"
            )
            
            logger.info(f"Found {len(communities[0])} communities")
            
            # Get basic statistics
            stats = await graphiti.get_community_info()
            if "statistics" in stats:
                logger.info(f"Average members per community: {stats['statistics'].get('average_members_per_community', 0):.2f}")
                logger.info(f"Largest community: {stats['statistics'].get('largest_community_size', 0)} members")
                
        except Exception as e:
            logger.error(f"Error with {algorithm} algorithm: {e}")


async def demonstrate_community_analysis(graphiti: GraphitiHF):
    """
    Demonstrate detailed community analysis capabilities.
    """
    logger.info("\n=== Community Analysis ===")
    
    # Perform comprehensive community analysis
    analysis = await graphiti.analyze_community_structure(
        algorithm="louvain",
        include_analysis=True
    )
    
    logger.info(f"Analysis performed with {analysis.get('algorithm', 'unknown')} algorithm")
    logger.info(f"Total communities: {analysis.get('total_communities', 0)}")
    
    # Display statistics
    if "statistics" in analysis:
        stats = analysis["statistics"]
        logger.info(f"Total members: {stats.get('total_members', 0)}")
        logger.info(f"Average members per community: {stats.get('average_members_per_community', 0):.2f}")
    
    # Display structure analysis
    if "structure_analysis" in analysis:
        structure = analysis["structure_analysis"]
        logger.info(f"Modularity: {structure.get('modularity', 0):.4f}")
        logger.info(f"Silhouette score: {structure.get('silhouette_score', 'N/A')}")
        
        # Display community sizes
        if "communities" in structure:
            logger.info("Community sizes:")
            for comm in structure["communities"]:
                logger.info(f"  Community {comm.get('community_id', '?')}: {comm.get('size', 0)} members")
    
    # Display core members
    if "core_members" in analysis:
        logger.info("\nCore Members:")
        for community_uuid, members in analysis["core_members"].items():
            logger.info(f"  Community {community_uuid[:8]}...:")
            for member in members[:3]:  # Show top 3 core members
                logger.info(f"    - {member.node_name} (centrality: {member.centrality:.3f})")
    
    # Display bridge nodes
    if "bridge_nodes" in analysis and analysis["bridge_nodes"]:
        logger.info(f"\nBridge nodes between communities: {len(analysis['bridge_nodes'])}")
        for bridge in analysis["bridge_nodes"][:3]:  # Show first 3 bridges
            logger.info(f"  {bridge['source_node'][:8]}... <-> {bridge['target_node'][:8]}...")
            logger.info(f"    Communities: {bridge['source_community'][:8]}... -> {bridge['target_community'][:8]}...")


async def demonstrate_community_similarity(graphiti: GraphitiHF):
    """
    Demonstrate community similarity analysis.
    """
    logger.info("\n=== Community Similarity ===")
    
    # Get community information
    community_info = await graphiti.get_community_info()
    
    if "communities" not in community_info or len(community_info["communities"]) < 2:
        logger.info("Need at least 2 communities for similarity analysis")
        return
    
    # Load communities to calculate similarity
    from graphiti_core.nodes import CommunityNode
    from graphiti_hf.analysis.community_detector import CommunityDetector
    
    detector = CommunityDetector(graphiti.driver)
    community_nodes = []
    
    for comm_info in community_info["communities"]:
        # Create CommunityNode objects (simplified)
        community_node = CommunityNode(
            uuid=comm_info["uuid"],
            name=comm_info["name"],
            group_id="default",
            labels=["Community"],
            created_at=datetime.now(),
            summary=comm_info.get("summary", "")
        )
        community_nodes.append(community_node)
    
    # Calculate similarities
    similarities = await detector.community_similarity(community_nodes)
    
    logger.info(f"Found {len(similarities)} similar community pairs")
    
    for similarity in similarities:
        logger.info(f"Communities {similarity.community1_uuid[:8]}... and {similarity.community2_uuid[:8]}...")
        logger.info(f"  Jaccard similarity: {similarity.jaccard_similarity:.3f}")
        logger.info(f"  Cosine similarity: {similarity.cosine_similarity:.3f}")
        logger.info(f"  Common members: {similarity.common_members}/{similarity.total_members}")


async def demonstrate_batch_processing(graphiti: GraphitiHF):
    """
    Demonstrate batch community detection for large graphs.
    """
    logger.info("\n=== Batch Processing ===")
    
    # Create group ID batches for demonstration
    group_batches = [
        ["default"],  # Single batch for this example
        # In real scenarios, you might have:
        # ["tech_team", "marketing_team", "research_team"],
        # ["engineering_team", "product_team"],
        # ["executive_team", "hr_team"]
    ]
    
    try:
        batch_results = await graphiti.batch_community_detection(
            group_id_batches=group_batches,
            algorithm="louvain",
            commit_message="Batch community detection"
        )
        
        logger.info(f"Processed {len(batch_results)} batches")
        
        for i, (nodes, edges) in enumerate(batch_results):
            logger.info(f"Batch {i+1}: {len(nodes)} communities, {len(edges)} edges")
            
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")


async def demonstrate_incremental_updates(graphiti: GraphitiHF):
    """
    Demonstrate incremental community updates.
    """
    logger.info("\n=== Incremental Updates ===")
    
    # Get current communities
    current_communities = await graphiti.get_community_info()
    initial_count = current_communities.get("total_communities", 0)
    
    logger.info(f"Initial community count: {initial_count}")
    
    # Add new nodes and edges
    new_nodes = [
        EntityNode(
            name="Rachel Green",
            labels=["Person"],
            group_id="default"
        ),
        EntityNode(
            name="Ross Geller", 
            labels=["Person"],
            group_id="default"
        )
    ]
    
    new_edges = [
        EntityEdge(
            source_node_uuid=new_nodes[0].uuid,
            target_node_uuid=new_nodes[1].uuid,
            fact="Rachel and Ross are colleagues",
            group_id="default"
        )
    ]
    
    # Perform incremental update
    updated_communities = await graphiti.incremental_community_update(
        new_nodes=new_nodes,
        new_edges=new_edges,
        commit_message="Incremental community update"
    )
    
    logger.info(f"Updated community count: {len(updated_communities[0])}")
    
    # Show the change
    final_communities = await graphiti.get_community_info()
    final_count = final_communities.get("total_communities", 0)
    logger.info(f"Change in community count: {final_count - initial_count}")


async def demonstrate_export_functionality(graphiti: GraphitiHF):
    """
    Demonstrate community export functionality.
    """
    logger.info("\n=== Export Functionality ===")
    
    # Export in different formats
    formats = ["json", "csv"]
    
    for format_type in formats:
        logger.info(f"\n--- Exporting as {format_type.upper()} ---")
        
        try:
            exported_data = await graphiti.export_communities(
                format=format_type,
                include_embeddings=False,
                commit_message=f"Export communities as {format_type}"
            )
            
            if format_type == "json":
                # Parse and show summary
                import json
                data = json.loads(exported_data)
                logger.info(f"Exported {data.get('total_communities', 0)} communities")
                logger.info(f"Export timestamp: {data.get('export_timestamp', 'N/A')}")
                
            elif format_type == "csv":
                # Show first few lines
                lines = exported_data.split('\n')
                logger.info(f"CSV export has {len(lines)} lines")
                logger.info("First few lines:")
                for line in lines[:5]:
                    logger.info(f"  {line}")
                    
        except Exception as e:
            logger.error(f"Error exporting as {format_type}: {e}")


async def demonstrate_caching(graphiti: GraphitiHF):
    """
    Demonstrate community caching functionality.
    """
    logger.info("\n=== Caching ===")
    
    # Cache communities
    cache_result = await graphiti.community_caching(
        cache_key="demo_cache",
        ttl=300  # 5 minutes
    )
    
    logger.info(f"Caching result: {cache_result}")
    
    # Retrieve cached communities
    cached_communities = await graphiti.get_cached_communities("demo_cache")
    
    if cached_communities:
        logger.info(f"Retrieved {len(cached_communities[0])} cached communities")
    else:
        logger.info("No cached communities found")


async def demonstrate_versioning(graphiti: GraphitiHF):
    """
    Demonstrate community versioning functionality.
    """
    logger.info("\n=== Versioning ===")
    
    # Create a version
    version_result = await graphiti.community_versioning(
        action="create",
        version_id="demo_v1"
    )
    
    logger.info(f"Version creation result: {version_result}")
    
    # List versions
    list_result = await graphiti.community_versioning(action="list")
    logger.info(f"Available versions: {list_result.get('versions', [])}")
    
    # Restore version (if available)
    if list_result.get("versions"):
        restore_result = await graphiti.community_versioning(
            action="restore",
            version_id=list_result["versions"][0]
        )
        logger.info(f"Restore result: {restore_result}")


async def main():
    """
    Main function to run all community detection examples.
    """
    logger.info("Starting Community Detection Examples for Graphiti-HF")
    
    try:
        # Create sample knowledge graph
        graphiti = await create_sample_knowledge_graph()
        
        # Run all demonstrations
        await demonstrate_basic_community_detection(graphiti)
        await demonstrate_algorithms_comparison(graphiti)
        await demonstrate_community_analysis(graphiti)
        await demonstrate_community_similarity(graphiti)
        await demonstrate_batch_processing(graphiti)
        await demonstrate_incremental_updates(graphiti)
        await demonstrate_export_functionality(graphiti)
        await demonstrate_caching(graphiti)
        await demonstrate_versioning(graphiti)
        
        logger.info("\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise
    
    finally:
        # Clean up (optional - you might want to keep the dataset for testing)
        logger.info("Examples completed. Dataset remains available for testing.")


if __name__ == "__main__":
    asyncio.run(main())