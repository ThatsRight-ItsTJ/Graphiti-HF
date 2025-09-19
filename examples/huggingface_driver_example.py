"""
Example usage of the HuggingFaceDriver

This script demonstrates how to use the HuggingFaceDriver to create
and manage a knowledge graph using HuggingFace datasets as storage.
"""

import asyncio
from datetime import datetime
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver
from graphiti_core.nodes import EntityNode, EpisodicNode, CommunityNode, EpisodeType
from graphiti_core.edges import EntityEdge


async def main():
    """
    Main example function demonstrating HuggingFaceDriver usage
    """
    # Initialize the HuggingFace driver
    # This will create or load a dataset repository on HuggingFace Hub
    driver = HuggingFaceDriver(
        repo_id="your-username/knowledge-graph-example",  # Replace with your username
        token="your-huggingface-token",  # Replace with your HuggingFace token
        private=False,
        create_repo=True
    )
    
    print("=== HuggingFaceDriver Example ===\n")
    
    # Create some entity nodes
    print("1. Creating entity nodes...")
    
    alice = EntityNode(
        name="Alice",
        group_id="example-group",
        labels=["Person"],
        attributes={"age": 30, "occupation": "Engineer"}
    )
    
    bob = EntityNode(
        name="Bob",
        group_id="example-group",
        labels=["Person"],
        attributes={"age": 25, "occupation": "Designer"}
    )
    
    company = EntityNode(
        name="TechCorp",
        group_id="example-group",
        labels=["Organization"],
        attributes={"industry": "Technology", "size": "Large"}
    )
    
    # Save the nodes
    await driver.save_node(alice)
    await driver.save_node(bob)
    await driver.save_node(company)
    
    print(f"   - Created entity: {alice.name} (UUID: {alice.uuid})")
    print(f"   - Created entity: {bob.name} (UUID: {bob.uuid})")
    print(f"   - Created entity: {company.name} (UUID: {company.uuid})")
    
    # Create some episodic nodes
    print("\n2. Creating episodic nodes...")
    
    episode1 = EpisodicNode(
        name="Team Meeting",
        content="Alice and Bob had a team meeting with TechCorp representatives.",
        source=EpisodeType.text,
        source_description="Meeting notes",
        group_id="example-group",
        valid_at=datetime.now()
    )
    
    episode2 = EpisodicNode(
        name="Project Kickoff",
        content="The project was officially kicked off with TechCorp.",
        source=EpisodeType.text,
        source_description="Project documentation",
        group_id="example-group",
        valid_at=datetime.now()
    )
    
    # Save the episodes
    await driver.save_node(episode1)
    await driver.save_node(episode2)
    
    print(f"   - Created episode: {episode1.name} (UUID: {episode1.uuid})")
    print(f"   - Created episode: {episode2.name} (UUID: {episode2.uuid})")
    
    # Create some entity edges
    print("\n3. Creating entity edges...")
    
    edge1 = EntityEdge(
        source_node_uuid=alice.uuid,
        target_node_uuid=bob.uuid,
        name="KNOWS",
        fact="Alice knows Bob",
        group_id="example-group"
    )
    
    edge2 = EntityEdge(
        source_node_uuid=alice.uuid,
        target_node_uuid=company.uuid,
        name="WORKS_FOR",
        fact="Alice works for TechCorp",
        group_id="example-group"
    )
    
    edge3 = EntityEdge(
        source_node_uuid=bob.uuid,
        target_node_uuid=company.uuid,
        name="WORKS_FOR",
        fact="Bob works for TechCorp",
        group_id="example-group"
    )
    
    # Save the edges
    await driver.save_edge(edge1)
    await driver.save_edge(edge2)
    await driver.save_edge(edge3)
    
    print(f"   - Created edge: {edge1.name} (UUID: {edge1.uuid})")
    print(f"   - Created edge: {edge2.name} (UUID: {edge2.uuid})")
    print(f"   - Created edge: {edge3.name} (UUID: {edge3.uuid})")
    
    # Create a community node
    print("\n4. Creating community node...")
    
    team_community = CommunityNode(
        name="Project Team",
        group_id="example-group",
        summary="The project team working with TechCorp"
    )
    
    await driver.save_node(team_community)
    print(f"   - Created community: {team_community.name} (UUID: {team_community.uuid})")
    
    # Retrieve nodes and edges
    print("\n5. Retrieving data from the knowledge graph...")
    
    # Get all nodes by group ID
    all_entities = await driver.get_nodes_by_group_ids(["example-group"], "Entity")
    print(f"   - Retrieved {len(all_entities)} entities:")
    for entity in all_entities:
        print(f"     * {entity.name} ({entity.labels})")
    
    all_episodes = await driver.get_nodes_by_group_ids(["example-group"], "Episodic")
    print(f"   - Retrieved {len(all_episodes)} episodes:")
    for episode in all_episodes:
        print(f"     * {episode.name}")
    
    all_edges = await driver.get_edges_by_group_ids(["example-group"], "Entity")
    print(f"   - Retrieved {len(all_edges)} edges:")
    for edge in all_edges:
        print(f"     * {edge.name}: {edge.fact}")
    
    # Get specific nodes by UUID
    print("\n6. Retrieving specific nodes...")
    
    retrieved_alice = await driver.get_node_by_uuid(alice.uuid, "Entity")
    print(f"   - Retrieved Alice: {retrieved_alice.name}, Age: {retrieved_alice.attributes.get('age')}")
    
    retrieved_edge = await driver.get_edge_by_uuid(edge1.uuid, "Entity")
    print(f"   - Retrieved edge: {retrieved_edge.fact}")
    
    # Show the current state of the datasets
    print("\n7. Dataset Statistics:")
    print(f"   - Nodes: {len(driver.get_nodes_df())}")
    print(f"   - Edges: {len(driver.get_edges_df())}")
    print(f"   - Episodes: {len(driver.get_episodes_df())}")
    print(f"   - Communities: {len(driver.get_communities_df())}")
    
    print("\n=== Example Complete ===")
    print(f"Your knowledge graph has been saved to: https://huggingface.co/{driver.repo_id}")
    print("You can view and manage your datasets on the HuggingFace Hub!")


if __name__ == "__main__":
    asyncio.run(main())
