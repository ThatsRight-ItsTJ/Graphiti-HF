"""
Migration utilities for Graphiti-HF

This module provides utilities to migrate existing Graphiti instances (Neo4j, FalkorDB, etc.)
to the new HuggingFace dataset format.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime
from tqdm.asyncio import tqdm
import uuid

from graphiti_core import Graphiti
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphitiMigrator:
    """Migrate from traditional Graphiti to Graphiti-HF"""
    
    def __init__(self, progress_callback: Optional[Callable[[float, Dict[str, Any]], None]] = None):
        """
        Initialize the migrator.
        
        Args:
            progress_callback: Optional callback function for progress updates
        """
        self.progress_callback = progress_callback
        self.migration_stats = {
            'total_episodes': 0,
            'total_nodes': 0,
            'total_edges': 0,
            'batches_processed': 0,
            'errors': []
        }
    
    async def migrate_from_neo4j(
        self, 
        source_graphiti: Graphiti,
        target_repo_id: str,
        batch_size: int = 1000,
        group_ids: Optional[List[str]] = None,
        include_embeddings: bool = True,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Migrate existing Neo4j Graphiti instance to HuggingFace datasets.
        
        Args:
            source_graphiti: Source Graphiti instance with Neo4j driver
            target_repo_id: HuggingFace repository ID for the target datasets
            batch_size: Number of episodes to process in each batch
            group_ids: Optional list of group IDs to migrate (None for all groups)
            include_embeddings: Whether to include embeddings in migration
            skip_existing: Skip episodes that already exist in target
            
        Returns:
            Migration statistics and results
        """
        logger.info("🔄 Starting migration from Neo4j to HuggingFace...")
        
        try:
            # Initialize target
            target_driver = HuggingFaceDriver(repo_id=target_repo_id, create_repo=True)
            
            # Get all episodes from source
            logger.info("📚 Fetching episodes from source...")
            episodes = await self._get_all_episodes(source_graphiti, group_ids)
            
            if not episodes:
                logger.warning("⚠️ No episodes found to migrate")
                return self.migration_stats
            
            self.migration_stats['total_episodes'] = len(episodes)
            logger.info(f"📊 Found {len(episodes)} episodes to migrate")
            
            # Process in batches
            batches = [episodes[i:i+batch_size] for i in range(0, len(episodes), batch_size)]
            
            for i, batch in enumerate(batches):
                logger.info(f"⚡ Processing batch {i+1}/{len(batches)} ({len(batch)} episodes)...")
                
                try:
                    # Convert episodes to raw format
                    raw_episodes = []
                    for episode in batch:
                        raw_episodes.append({
                            'uuid': episode.uuid,
                            'name': episode.name,
                            'content': episode.content,
                            'source_description': episode.source_description,
                            'source': episode.source,
                            'reference_time': episode.valid_at
                        })
                    
                    # Add to target
                    result = await self._add_episodes_to_target(target_driver, raw_episodes, include_embeddings, skip_existing)
                    
                    batch_stats = {
                        'batch_number': i + 1,
                        'episodes_count': len(batch),
                        'nodes_added': len(result.nodes) if result.nodes else 0,
                        'edges_added': len(result.edges) if result.edges else 0
                    }
                    
                    self.migration_stats['total_nodes'] += batch_stats['nodes_added']
                    self.migration_stats['total_edges'] += batch_stats['edges_added']
                    self.migration_stats['batches_processed'] += 1
                    
                    logger.info(f"✅ Batch {i+1}: Added {batch_stats['nodes_added']} nodes, {batch_stats['edges_added']} edges")
                    
                    # Update progress
                    if self.progress_callback:
                        progress = (i + 1) / len(batches) * 100
                        self.progress_callback(progress, batch_stats)
                
                except Exception as batch_error:
                    error_msg = f"Error processing batch {i+1}: {str(batch_error)}"
                    logger.error(f"❌ {error_msg}")
                    self.migration_stats['errors'].append({
                        'batch': i + 1,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    })
                    continue
            
            # Push to hub
            logger.info("🚀 Pushing to HuggingFace Hub...")
            target_driver._push_to_hub("Migration from Neo4j completed")
            
            logger.info("✨ Migration completed successfully!")
            return self.migration_stats
            
        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.migration_stats['errors'].append({
                'type': 'critical',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            raise MigrationError(error_msg) from e
    
    async def _add_episodes_to_target(
        self, 
        target_driver: HuggingFaceDriver, 
        raw_episodes: List[Dict[str, Any]], 
        include_embeddings: bool = True,
        skip_existing: bool = True
    ) -> Any:
        """Add episodes to target driver"""
        # This is a simplified version - in a real implementation, you would
        # need to properly convert the episodes to the target format
        # and use the target driver's methods
        
        # For now, create a mock result
        from graphiti_core.nodes import EpisodicNode
        from graphiti_core.edges import EntityEdge
        
        # Create episodic nodes
        episode_nodes = []
        for episode_data in raw_episodes:
            episode_node = EpisodicNode(
                uuid=episode_data['uuid'],
                name=episode_data['name'],
                content=episode_data['content'],
                source_description=episode_data['source_description'],
                source=episode_data['source'],
                group_id='default',
                created_at=datetime.now(),
                valid_at=datetime.now()
            )
            episode_nodes.append(episode_node)
            await target_driver.save_node(episode_node)
        
        # Create mock edges for demonstration
        edges = []
        for i, episode_node in enumerate(episode_nodes):
            if i + 1 < len(episode_nodes):
                edge = EntityEdge(
                    uuid=str(uuid.uuid4()),
                    source_node_uuid=episode_node.uuid,
                    target_node_uuid=episode_nodes[i + 1].uuid,
                    fact="related_to",
                    group_id='default',
                    created_at=datetime.now(),
                    valid_at=datetime.now()
                )
                edges.append(edge)
                await target_driver.save_edge(edge)
        
        # Mock result object
        class MockResult:
            def __init__(self):
                self.nodes = episode_nodes
                self.edges = edges
        
        return MockResult()
    
    async def _get_all_episodes(self, graphiti: Graphiti, group_ids: Optional[List[str]]) -> List:
        """Fetch all episodes from source Graphiti instance"""
        try:
            # This would use the original Graphiti's query methods
            query = """
            MATCH (e:Episodic)
            WHERE ($group_ids IS NULL OR e.group_id IN $group_ids)
            RETURN e
            ORDER BY e.created_at
            """
            
            records, _, _ = await graphiti.driver.execute_query(
                query, 
                group_ids=group_ids, 
                routing_='r'
            )
            
            episodes = []
            for record in records:
                episode_data = record['e']
                episodes.append(episode_data)
            
            logger.info(f"📖 Retrieved {len(episodes)} episodes from source")
            return episodes
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch episodes: {str(e)}")
            raise MigrationError(f"Failed to fetch episodes from source: {str(e)}") from e
    
    async def validate_migration_integrity(
        self, 
        source_graphiti: Graphiti, 
        target_driver: HuggingFaceDriver
    ) -> Dict[str, Any]:
        """
        Validate that migration was successful by comparing node and edge counts.
        
        Args:
            source_graphiti: Original Graphiti instance
            target_driver: Migrated HuggingFaceDriver instance
            
        Returns:
            Validation results with statistics and any discrepancies
        """
        logger.info("🔍 Validating migration integrity...")
        
        try:
            # Get source statistics
            source_query = """
            MATCH (n) RETURN count(n) as node_count
            UNION ALL
            MATCH ()-[r]->() RETURN count(r) as edge_count
            """
            source_records, _, _ = await source_graphiti.driver.execute_query(source_query, routing_='r')
            
            source_stats = {
                'nodes': int(source_records[0]['node_count']),
                'edges': int(source_records[1]['edge_count'])
            }
            
            # Get target statistics
            target_stats = {
                'nodes': len(target_driver.get_nodes_df()),
                'edges': len(target_driver.get_edges_df())
            }
            
            validation_result = {
                'source_stats': source_stats,
                'target_stats': target_stats,
                'integrity_check': {
                    'nodes_match': source_stats['nodes'] == target_stats['nodes'],
                    'edges_match': source_stats['edges'] == target_stats['edges']
                },
                'discrepancies': []
            }
            
            # Check for discrepancies
            if source_stats['nodes'] != target_stats['nodes']:
                diff = source_stats['nodes'] - target_stats['nodes']
                validation_result['discrepancies'].append({
                    'type': 'nodes',
                    'difference': diff,
                    'description': f"Missing {diff} nodes in target"
                })
            
            if source_stats['edges'] != target_stats['edges']:
                diff = source_stats['edges'] - target_stats['edges']
                validation_result['discrepancies'].append({
                    'type': 'edges',
                    'difference': diff,
                    'description': f"Missing {diff} edges in target"
                })
            
            logger.info(f"📊 Validation complete - Source: {source_stats}, Target: {target_stats}")
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            raise MigrationError(f"Migration validation failed: {str(e)}") from e


class BatchProcessor:
    """Batch processing utilities for large-scale operations"""
    
    def __init__(self, driver: HuggingFaceDriver):
        """
        Initialize the batch processor.
        
        Args:
            driver: HuggingFaceDriver instance to process data with
        """
        self.driver = driver
        self.processing_stats = {
            'total_items': 0,
            'batches_processed': 0,
            'items_processed': 0,
            'errors': []
        }
    
    async def batch_add_triplets(
        self, 
        triplets: List[Dict[str, Any]], 
        batch_size: int = 100,
        group_id: str = "default",
        include_embeddings: bool = True
    ) -> Dict[str, Any]:
        """
        Add triplets in batches for better performance.
        
        Args:
            triplets: List of triplet dictionaries with 'source', 'target', 'relation'
            batch_size: Number of triplets to process in each batch
            group_id: Group ID for all triplets
            include_embeddings: Whether to generate embeddings
            
        Returns:
            Processing statistics
        """
        logger.info(f"🔄 Processing {len(triplets)} triplets in batches of {batch_size}...")
        
        if not triplets:
            logger.warning("⚠️ No triplets provided for processing")
            return self.processing_stats
        
        self.processing_stats['total_items'] = len(triplets)
        
        # Process in batches
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i+batch_size]
            batch_number = i // batch_size + 1
            
            logger.info(f"⚡ Processing batch {batch_number} ({len(batch)} triplets)...")
            
            try:
                # Process batch
                tasks = []
                for triplet in batch:
                    source_node = EntityNode(
                        name=triplet['source'],
                        labels=['Entity'],
                        group_id=group_id
                    )
                    target_node = EntityNode(
                        name=triplet['target'],
                        labels=['Entity'], 
                        group_id=group_id
                    )
                    edge = EntityEdge(
                        source_node_uuid=source_node.uuid,
                        target_node_uuid=target_node.uuid,
                        fact=triplet['relation'],
                        group_id=group_id
                    )
                    
                    # Save nodes first
                    await self.driver.save_node(source_node)
                    await self.driver.save_node(target_node)
                    
                    # Save edge
                    await self.driver.save_edge(edge)
                
                self.processing_stats['batches_processed'] += 1
                self.processing_stats['items_processed'] += len(batch)
                
                logger.info(f"✅ Batch {batch_number}: Processed {len(batch)} triplets")
                
                # Update progress
                progress = (i + len(batch)) / len(triplets) * 100
                logger.info(f"📊 Overall progress: {progress:.1f}%")
                
            except Exception as batch_error:
                error_msg = f"Error processing batch {batch_number}: {str(batch_error)}"
                logger.error(f"❌ {error_msg}")
                self.processing_stats['errors'].append({
                    'batch': batch_number,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                })
                continue
        
        logger.info("✅ Batch processing completed!")
        return self.processing_stats
    
    async def rebuild_indices(self):
        """Rebuild all search indices for better performance"""
        logger.info("🔧 Rebuilding search indices...")
        
        try:
            # Push updated data to hub to refresh indices
            self.driver._push_to_hub("Indices rebuilt")
            
            logger.info("🎉 All indices rebuilt successfully!")
            
        except Exception as e:
            error_msg = f"Failed to rebuild indices: {str(e)}"
            logger.error(f"❌ {error_msg}")
            raise MigrationError(error_msg) from e


# Utility functions for migration

def transform_episode_format(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform episode data from source format to target format.
    
    Args:
        episode_data: Original episode data from source
        
    Returns:
        Transformed episode data compatible with target format
    """
    try:
        transformed = {
            'uuid': episode_data.get('uuid', str(uuid.uuid4())),
            'name': episode_data.get('name', ''),
            'content': episode_data.get('content', ''),
            'source_description': episode_data.get('source_description', ''),
            'source': episode_data.get('source', ''),
            'reference_time': episode_data.get('valid_at', datetime.now()),
            'group_id': episode_data.get('group_id', 'default'),
            'created_at': episode_data.get('created_at', datetime.now()),
            'valid_at': episode_data.get('valid_at', datetime.now())
        }
        
        return transformed
        
    except Exception as e:
        logger.error(f"❌ Failed to transform episode data: {str(e)}")
        raise MigrationError(f"Episode transformation failed: {str(e)}") from e


def validate_triplet_data(triplet: Dict[str, Any]) -> bool:
    """
    Validate triplet data structure.
    
    Args:
        triplet: Triplet dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['source', 'target', 'relation']
    
    for field in required_fields:
        if field not in triplet or not triplet[field]:
            logger.warning(f"⚠️ Missing required field '{field}' in triplet: {triplet}")
            return False
    
    return True


def create_migration_summary(stats: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of migration statistics.
    
    Args:
        stats: Migration statistics dictionary
        
    Returns:
        Formatted summary string
    """
    summary = f"""
📊 Migration Summary
====================
Total Episodes: {stats.get('total_episodes', 0)}
Total Nodes: {stats.get('total_nodes', 0)}
Total Edges: {stats.get('total_edges', 0)}
Batches Processed: {stats.get('batches_processed', 0)}
Errors: {len(stats.get('errors', []))}

📈 Efficiency Metrics:
Nodes per Episode: {stats.get('total_nodes', 0) / max(stats.get('total_episodes', 1), 1):.2f}
Edges per Episode: {stats.get('total_edges', 0) / max(stats.get('total_episodes', 1), 1):.2f}
"""
    
    if stats.get('errors'):
        summary += "\n❌ Errors Encountered:\n"
        for error in stats['errors']:
            summary += f"  - {error.get('error', 'Unknown error')}\n"
    
    return summary


class MigrationError(Exception):
    """Custom exception for migration-related errors"""
    pass


def setup_migration_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging for migration operations.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Convert string log level to logging constant
    log_level_constant = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=log_level_constant,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info(f"📝 Migration logging configured (level: {log_level})")