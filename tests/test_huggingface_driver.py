"""
Tests for HuggingFaceDriver
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from graphiti_core.nodes import EntityNode, EpisodicNode, CommunityNode, EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver


@pytest.fixture
def mock_hf_driver():
    """Create a mock HuggingFaceDriver for testing"""
    with patch('graphiti_hf.drivers.huggingface_driver.HfFolder.get_token', return_value='mock_token'), \
         patch('graphiti_hf.drivers.huggingface_driver.load_dataset') as mock_load_dataset, \
         patch('graphiti_hf.drivers.huggingface_driver.DatasetDict.push_to_hub') as mock_push:
        
        # Mock the load_dataset to raise an exception (so we create empty datasets)
        mock_load_dataset.side_effect = Exception("Dataset not found")
        
        driver = HuggingFaceDriver(repo_id="test/repo")
        yield driver, mock_push


@pytest.mark.asyncio
async def test_save_and_get_entity_node(mock_hf_driver):
    """Test saving and retrieving an entity node"""
    driver, mock_push = mock_hf_driver
    
    # Create a test entity node
    node = EntityNode(
        name="Test Entity",
        group_id="test-group",
        labels=["Person"],
        attributes={"age": 30}
    )
    
    # Save the node
    await driver.save_node(node)
    
    # Verify the node was added to the DataFrame
    assert len(driver.nodes_df) == 1
    assert driver.nodes_df.iloc[0]['uuid'] == node.uuid
    assert driver.nodes_df.iloc[0]['name'] == "Test Entity"
    assert driver.nodes_df.iloc[0]['group_id'] == "test-group"
    assert "Person" in driver.nodes_df.iloc[0]['labels']
    
    # Verify push_to_hub was called
    mock_push.assert_called()
    
    # Get the node
    retrieved_node = await driver.get_node_by_uuid(node.uuid, "Entity")
    assert retrieved_node.uuid == node.uuid
    assert retrieved_node.name == "Test Entity"
    assert "Person" in retrieved_node.labels


@pytest.mark.asyncio
async def test_save_and_get_episodic_node(mock_hf_driver):
    """Test saving and retrieving an episodic node"""
    driver, mock_push = mock_hf_driver
    
    # Create a test episodic node
    node = EpisodicNode(
        name="Test Episode",
        content="This is a test episode",
        source=EpisodeType.text,
        source_description="Test source",
        group_id="test-group",
        valid_at=datetime.now()
    )
    
    # Save the node
    await driver.save_node(node)
    
    # Verify the node was added to the DataFrame
    assert len(driver.episodes_df) == 1
    assert driver.episodes_df.iloc[0]['uuid'] == node.uuid
    assert driver.episodes_df.iloc[0]['name'] == "Test Episode"
    assert driver.episodes_df.iloc[0]['content'] == "This is a test episode"
    
    # Verify push_to_hub was called
    mock_push.assert_called()
    
    # Get the node
    retrieved_node = await driver.get_node_by_uuid(node.uuid, "Episodic")
    assert retrieved_node.uuid == node.uuid
    assert retrieved_node.name == "Test Episode"
    assert retrieved_node.content == "This is a test episode"


@pytest.mark.asyncio
async def test_save_and_get_community_node(mock_hf_driver):
    """Test saving and retrieving a community node"""
    driver, mock_push = mock_hf_driver
    
    # Create a test community node
    node = CommunityNode(
        name="Test Community",
        group_id="test-group",
        summary="A test community"
    )
    
    # Save the node
    await driver.save_node(node)
    
    # Verify the node was added to the DataFrame
    assert len(driver.communities_df) == 1
    assert driver.communities_df.iloc[0]['uuid'] == node.uuid
    assert driver.communities_df.iloc[0]['name'] == "Test Community"
    assert driver.communities_df.iloc[0]['summary'] == "A test community"
    
    # Verify push_to_hub was called
    mock_push.assert_called()
    
    # Get the node
    retrieved_node = await driver.get_node_by_uuid(node.uuid, "Community")
    assert retrieved_node.uuid == node.uuid
    assert retrieved_node.name == "Test Community"
    assert retrieved_node.summary == "A test community"


@pytest.mark.asyncio
async def test_save_and_get_entity_edge(mock_hf_driver):
    """Test saving and retrieving an entity edge"""
    driver, mock_push = mock_hf_driver
    
    # Create a test entity edge
    edge = EntityEdge(
        source_node_uuid="source-uuid",
        target_node_uuid="target-uuid",
        name="KNOWS",
        fact="Source knows Target",
        group_id="test-group"
    )
    
    # Save the edge
    await driver.save_edge(edge)
    
    # Verify the edge was added to the DataFrame
    assert len(driver.edges_df) == 1
    assert driver.edges_df.iloc[0]['uuid'] == edge.uuid
    assert driver.edges_df.iloc[0]['source_uuid'] == "source-uuid"
    assert driver.edges_df.iloc[0]['target_uuid'] == "target-uuid"
    assert driver.edges_df.iloc[0]['name'] == "KNOWS"
    assert driver.edges_df.iloc[0]['fact'] == "Source knows Target"
    
    # Verify push_to_hub was called
    mock_push.assert_called()
    
    # Get the edge
    retrieved_edge = await driver.get_edge_by_uuid(edge.uuid, "Entity")
    assert retrieved_edge.uuid == edge.uuid
    assert retrieved_edge.source_node_uuid == "source-uuid"
    assert retrieved_edge.target_node_uuid == "target-uuid"
    assert retrieved_edge.name == "KNOWS"
    assert retrieved_edge.fact == "Source knows Target"


@pytest.mark.asyncio
async def test_get_nodes_by_group_ids(mock_hf_driver):
    """Test getting nodes by group IDs"""
    driver, mock_push = mock_hf_driver
    
    # Create test nodes
    node1 = EntityNode(name="Node 1", group_id="group1")
    node2 = EntityNode(name="Node 2", group_id="group2")
    node3 = EntityNode(name="Node 3", group_id="group1")
    
    # Save nodes
    await driver.save_node(node1)
    await driver.save_node(node2)
    await driver.save_node(node3)
    
    # Get nodes by group IDs
    nodes = await driver.get_nodes_by_group_ids(["group1"], "Entity")
    assert len(nodes) == 2
    assert nodes[0].name in ["Node 1", "Node 3"]
    assert nodes[1].name in ["Node 1", "Node 3"]
    
    # Get nodes with limit
    nodes = await driver.get_nodes_by_group_ids(["group1"], "Entity", limit=1)
    assert len(nodes) == 1


@pytest.mark.asyncio
async def test_get_edges_by_group_ids(mock_hf_driver):
    """Test getting edges by group IDs"""
    driver, mock_push = mock_hf_driver
    
    # Create test edges
    edge1 = EntityEdge(source_node_uuid="1", target_node_uuid="2", name="RELATES", group_id="group1")
    edge2 = EntityEdge(source_node_uuid="2", target_node_uuid="3", name="RELATES", group_id="group2")
    edge3 = EntityEdge(source_node_uuid="3", target_node_uuid="4", name="RELATES", group_id="group1")
    
    # Save edges
    await driver.save_edge(edge1)
    await driver.save_edge(edge2)
    await driver.save_edge(edge3)
    
    # Get edges by group IDs
    edges = await driver.get_edges_by_group_ids(["group1"], "Entity")
    assert len(edges) == 2
    assert edges[0].source_node_uuid in ["1", "3"]
    assert edges[1].source_node_uuid in ["1", "3"]
    
    # Get edges with limit
    edges = await driver.get_edges_by_group_ids(["group1"], "Entity", limit=1)
    assert len(edges) == 1


@pytest.mark.asyncio
async def test_delete_all_indexes(mock_hf_driver):
    """Test deleting all indexes/datasets"""
    driver, mock_push = mock_hf_driver
    
    # Add some data
    node = EntityNode(name="Test Node", group_id="test-group")
    edge = EntityEdge(source_node_uuid="1", target_node_uuid="2", name="RELATES", group_id="test-group")
    
    await driver.save_node(node)
    await driver.save_edge(edge)
    
    # Verify data exists
    assert len(driver.nodes_df) == 1
    assert len(driver.edges_df) == 1
    
    # Delete all indexes
    await driver.delete_all_indexes()
    
    # Verify data is cleared
    assert len(driver.nodes_df) == 0
    assert len(driver.edges_df) == 0
    assert len(driver.episodes_df) == 0
    assert len(driver.communities_df) == 0
    
    # Verify push_to_hub was called
    mock_push.assert_called()


@pytest.mark.asyncio
async def test_node_not_found_error(mock_hf_driver):
    """Test that NodeNotFoundError is raised when node doesn't exist"""
    driver, _ = mock_hf_driver
    
    # Try to get a non-existent node
    with pytest.raises(Exception):  # Should raise NodeNotFoundError
        await driver.get_node_by_uuid("non-existent-uuid", "Entity")


@pytest.mark.asyncio
async def test_edge_not_found_error(mock_hf_driver):
    """Test that EdgeNotFoundError is raised when edge doesn't exist"""
    driver, _ = mock_hf_driver
    
    # Try to get a non-existent edge
    with pytest.raises(Exception):  # Should raise EdgeNotFoundError
        await driver.get_edge_by_uuid("non-existent-uuid", "Entity")


def test_driver_initialization(mock_hf_driver):
    """Test driver initialization"""
    driver, _ = mock_hf_driver
    
    # Verify driver properties
    assert driver.repo_id == "test/repo"
    assert driver.token == "mock_token"
    assert driver.provider.name == "neo4j"  # Using NEO4J as closest match
    assert driver._database == "default"
    
    # Verify DataFrames are empty initially
    assert len(driver.nodes_df) == 0
    assert len(driver.edges_df) == 0
    assert len(driver.episodes_df) == 0
    assert len(driver.communities_df) == 0


def test_driver_session(mock_hf_driver):
    """Test driver session creation"""
    driver, _ = mock_hf_driver
    
    session = driver.session()
    assert session is not None
    assert hasattr(session, 'driver')
    assert session.driver == driver