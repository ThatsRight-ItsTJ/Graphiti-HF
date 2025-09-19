
"""
Enhanced Deduplicator for Graphiti-HF

Provides advanced deduplication capabilities for entities and edges using
similarity algorithms, embeddings, and conflict resolution.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication"""
    similarity_threshold: float = 0.8
    name_weight: float = 0.6
    embedding_weight: float = 0.4
    attribute_weight: float = 0.2
    temporal_consistency: bool = True
    conflict_resolution: str = "merge"  # "merge", "keep_newer", "keep_older", "keep_better"
    max_workers: int = 4
    batch_size: int = 100


class Deduplicator:
    """
    Enhanced deduplicator for entities and edges with multiple similarity strategies
    
    Handles deduplication using name similarity, embeddings, attributes,
    and temporal consistency with configurable conflict resolution.
    """
    
    def __init__(self, config: Optional[DeduplicationConfig] = None):
        """
        Initialize the deduplicator
        
        Args:
            config: Deduplication configuration
        """
        self.config = config or DeduplicationConfig()
        
        # Initialize text vectorizer for similarity calculations
        self.text_vectorizer = TfidfVectorizer(
            max_features=10000, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Deduplication statistics
        self.stats = {
            'total_duplicates_found': 0,
            'entities_deduplicated': 0,
            'edges_deduplicated': 0,
            'conflicts_resolved': 0,
            'temporal_conflicts': 0
        }
    
    async def deduplicate_entities(
        self,
        new_entities: List[EntityNode],
        existing_entities: Optional[List[EntityNode]] = None
    ) -> Tuple[List[EntityNode], List[str]]:
        """
        Deduplicate entities using multiple similarity strategies
        
        Args:
            new_entities: Newly extracted entities to deduplicate
            existing_entities: Existing entities in the knowledge graph
            
        Returns:
            Tuple of (deduplicated_entities, duplicate_entity_uuids)
        """
        if not new_entities:
            return [], []
        
        # Combine all entities for comparison
        all_entities = (existing_entities or []) + new_entities
        
        # Calculate similarity matrix
        similarity_matrix = await self._calculate_entity_similarity_matrix(all_entities)
        
        # Find duplicates
        duplicates = self._find_entity_duplicates(similarity_matrix, all_entities)
        
        # Resolve conflicts
        resolved_entities = self._resolve_entity_conflicts(duplicates, all_entities)
        
        # Filter out duplicates from new entities
        final_entities = [
            entity for entity in new_entities 
            if entity.uuid not in [d['duplicate_uuid'] for d in duplicates]
        ]
        
        # Update statistics
        self.stats['total_duplicates_found'] += len(duplicates)
        self.stats['entities_deduplicated'] += len(duplicates)
        
        return final_entities, [d['duplicate_uuid'] for d in duplicates]
    
    async def deduplicate_edges(
        self,
        new_edges: List[EntityEdge],
        existing_edges: Optional[List[EntityEdge]] = None
    ) -> Tuple[List[EntityEdge], List[str]]:
        """
        Deduplicate edges using multiple similarity strategies
        
        Args:
            new_edges: Newly extracted edges to deduplicate
            existing_edges: Existing edges in the knowledge graph
            
        Returns:
            Tuple of (deduplicated_edges, duplicate_edge_uuids)
        """
        if not new_edges:
            return [], []
        
        # Combine all edges for comparison
        all_edges = (existing_edges or []) + new_edges
        
        # Calculate similarity matrix
        similarity_matrix = await self._calculate_edge_similarity_matrix(all_edges)
        
        # Find duplicates
        duplicates = self._find_edge_duplicates(similarity_matrix, all_edges)
        
        # Resolve conflicts
        resolved_edges = self._resolve_edge_conflicts(duplicates, all_edges)
        
        # Filter out duplicates from new edges
        final_edges = [
            edge for edge in new_edges 
            if edge.uuid not in [d['duplicate_uuid'] for d in duplicates]
        ]
        
        # Update statistics
        self.stats['total_duplicates_found'] += len(duplicates)
        self.stats['edges_deduplicated'] += len(duplicates)
        
        return final_edges, [d['duplicate_uuid'] for d in duplicates]
    
    async def deduplicate_incremental(
        self,
        new_entities: List[EntityNode],
        new_edges: List[EntityEdge],
        existing_entities: Optional[List[EntityNode]] = None,
        existing_edges: Optional[List[EntityEdge]] = None
    ) -> Dict[str, Any]:
        """
        Perform incremental deduplication for both entities and edges
        
        Args:
            new_entities: Newly extracted entities
            new_edges: Newly extracted edges
            existing_entities: Existing entities
            existing_edges: Existing edges
            
        Returns:
            Dictionary with deduplication results
        """
        # Process entities and edges in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            loop = asyncio.get_event_loop()
            
            entity_task = loop.run_in_executor(
                executor,
                lambda: asyncio.run(self.deduplicate_entities(new_entities, existing_entities))
            )
            edge_task = loop.run_in_executor(
                executor,
                lambda: asyncio.run(self.deduplicate_edges(new_edges, existing_edges))
            )
            
            deduped_entities, duplicate_entities = await entity_task
            deduped_edges, duplicate_edges = await edge_task
        
        return {
            'entities': {
                'deduplicated': deduped_entities,
                'duplicates': duplicate_entities
            },
            'edges': {
                'deduplicated': deduped_edges,
                'duplicates': duplicate_edges
            },
            'stats': {
                'entities_deduplicated': len(duplicate_entities),
                'edges_deduplicated': len(duplicate_edges),
                'total_duplicates': len(duplicate_entities) + len(duplicate_edges)
            }
        }
    
    def get_deduplication_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset deduplication statistics"""
        self.stats = {
            'total_duplicates_found': 0,
            'entities_deduplicated': 0,
            'edges_deduplicated': 0,
            'conflicts_resolved': 0,
            'temporal_conflicts': 0
        }
    
    # Helper methods
    async def _calculate_entity_similarity_matrix(
        self,
        entities: List[EntityNode]
    ) -> np.ndarray:
        """Calculate similarity matrix for entities"""
        n = len(entities)
        similarity_matrix = np.zeros((n, n))
        
        # Extract names for text similarity
        names = [entity.name for entity in entities if entity.name]
        
        if len(names) < 2:
            return similarity_matrix
        
        # Calculate name similarity
        try:
            name_vectors = self.text_vectorizer.fit_transform(names)
            name_similarities = cosine_similarity(name_vectors)
            
            # Apply weights
            for i in range(n):
                for j in range(n):
                    if i != j and entities[i].name and entities[j].name:
                        similarity_matrix[i][j] = name_similarities[i][j] * self.config.name_weight
        except Exception as e:
            logger.warning(f"Error calculating name similarity: {e}")
        
        # Add embedding similarity
        await self._add_embedding_similarity(
            similarity_matrix, entities, 'name_embedding'
        )
        
        # Add attribute similarity
        self._add_attribute_similarity(similarity_matrix, entities)
        
        return similarity_matrix
    
    async def _calculate_edge_similarity_matrix(
        self,
        edges: List[EntityEdge]
    ) -> np.ndarray:
        """Calculate similarity matrix for edges"""
        n = len(edges)
        similarity_matrix = np.zeros((n, n))
        
        # Extract facts for text similarity
        facts = [edge.fact for edge in edges if edge.fact]
        
        if len(facts) < 2:
            return similarity_matrix
        
        # Calculate fact similarity
        try:
            fact_vectors = self.text_vectorizer.fit_transform(facts)
            fact_similarities = cosine_similarity(fact_vectors)
            
            # Apply weights
            for i in range(n):
                for j in range(n):
                    if i != j and edges[i].fact and edges[j].fact:
                        similarity_matrix[i][j] = fact_similarities[i][j] * self.config.name_weight
        except Exception as e:
            logger.warning(f"Error calculating fact similarity: {e}")
        
        # Add embedding similarity
        await self._add_embedding_similarity(
            similarity_matrix, edges, 'fact_embedding'
        )
        
        # Add structural similarity (source-target pairs)
        self._add_structural_similarity(similarity_matrix, edges)
        
        return similarity_matrix
    
    async def _add_embedding_similarity(
        self,
        similarity_matrix: np.ndarray,
        items: List[Any],
        embedding_attr: str
    ):
        """Add embedding similarity to the similarity matrix"""
        n = len(items)
        
        # Collect embeddings
        embeddings = []
        valid_indices = []
        
        for i, item in enumerate(items):
            embedding = getattr(item, embedding_attr, None)
            if embedding is not None:
                try:
                    emb_array = np.array(embedding).reshape(1, -1)
                    embeddings.append(emb_array)
                    valid_indices.append(i)
                except (ValueError, TypeError):
                    continue
        
        if len(embeddings) < 2:
            return
        
        # Calculate embedding similarities
        try:
            emb_matrix = np.vstack(embeddings)
            emb_similarities = cosine_similarity(emb_matrix)
            
            # Map back to original indices
            for idx_i, i in enumerate(valid_indices):
                for idx_j, j in enumerate(valid_indices):
                    if idx_i != idx_j:
                        similarity = emb_similarities[idx_i][idx_j] * self.config.embedding_weight
                        similarity_matrix[i][j] += similarity
        except Exception as e:
            logger.warning(f"Error calculating embedding similarity: {e}")
    
    def _add_attribute_similarity(
        self,
        similarity_matrix: np.ndarray,
        entities: List[EntityNode]
    ):
        """Add attribute similarity to the similarity matrix"""
        n = len(entities)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    attr_sim = self._calculate_attribute_similarity(
                        entities[i].attributes, entities[j].attributes
                    )
                    similarity_matrix[i][j] += attr_sim * self.config.attribute_weight
    
    def _add_structural_similarity(
        self,
        similarity_matrix: np.ndarray,
        edges: List[EntityEdge]
    ):
        """Add structural similarity (source-target pairs) to the similarity matrix"""
        n = len(edges)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if edges have same source-target pair
                    if (edges[i].source_node_uuid == edges[j].source_node_uuid and
                        edges[i].target_node_uuid == edges[j].target_node_uuid):
                        similarity_matrix[i][j] += 0.3  # Boost for same structure
    
    def _calculate_attribute_similarity(
        self,
        attrs1: Dict[str, Any],
        attrs2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two attribute dictionaries"""
        if not attrs1 or not attrs2:
            return 0.0
        
        # Find common attributes
        common_keys = set(attrs1.keys()) & set(attrs2.keys())
        
        if not common_keys:
            return 0.0
        
        # Calculate similarity for common attributes
        total_similarity = 0.0
        valid_attrs = 0
        
        for key in common_keys:
            val1 = attrs1[key]
            val2 = attrs2[key]
            
            if val1 == val2:
                total_similarity += 1.0
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity
                try:
                    vec1 = self.text_vectorizer.transform([val1])
                    vec2 = self.text_vectorizer.transform([val2])
                    sim = cosine_similarity(vec1, vec2)[0][0]
                    total_similarity += sim
                except:
                    total_similarity += 0.0
            else:
                # Type mismatch or different values
                total_similarity += 0.0
            
            valid_attrs += 1
        
        return total_similarity / valid_attrs if valid_attrs > 0 else 0.0
    
    def _find_entity_duplicates(
        self,
        similarity_matrix: np.ndarray,
        entities: List[EntityNode]
    ) -> List[Dict[str, Any]]:
        """Find duplicate entities based on similarity matrix"""
        duplicates = []
        n = len(entities)
        processed = set()
        
        for i in range(n):
            if i in processed:
                continue
            
            # Find similar entities
            similar_indices = np.where(similarity_matrix[i] >= self.config.similarity_threshold)[0]
            
            # Skip if no duplicates found
            if len(similar_indices) <= 1:
                processed.add(i)
                continue
            
            # Mark all similar entities as processed
            processed.update(similar_indices)
            
            # Create duplicate records
            for j in similar_indices:
                if j != i:
                    duplicates.append({
                        'original_index': i,
                        'duplicate_index': j,
                        'similarity': similarity_matrix[i][j],
                        'original_uuid': entities[i].uuid,
                        'duplicate_uuid': entities[j].uuid,
                        'type': 'entity'
                    })
        
        return duplicates
    
    def _find_edge_duplicates(
        self,
        similarity_matrix: np.ndarray,
        edges: List[EntityEdge]
    ) -> List[Dict[str, Any]]:
        """Find duplicate edges based on similarity matrix"""
        duplicates = []
        n = len(edges)
        processed = set()
        
        for i in range(n):
            if i in processed:
                continue
            
            # Find similar edges
            similar_indices = np.where(similarity_matrix[i] >= self.config.similarity_threshold)[0]
            
            # Skip if no duplicates found
            if len(similar_indices) <= 1:
                processed.add(i)
                continue
            
            # Mark all similar edges as processed
            processed.update(similar_indices)
            
            # Create duplicate records
            for j in similar_indices:
                if j != i:
                    duplicates.append({
                        'original_index': i,
                        'duplicate_index': j,
                        'similarity': similarity_matrix[i][j],
                        'original_uuid': edges[i].uuid,
                        'duplicate_uuid': edges[j].uuid,
                        'type': 'edge'
                    })
        
        return duplicates
    
    def _resolve_entity_conflicts(
        self,
        duplicates: List[Dict[str, Any]],
        entities: List[EntityNode]
    ) -> List[EntityNode]:
        """Resolve conflicts between duplicate entities"""
        resolved_entities = []
        processed_indices = set()
        
        for duplicate in duplicates:
            original_idx = duplicate['original_index']
            duplicate_idx = duplicate['duplicate_index']
            
            if original_idx in processed_indices or duplicate_idx in processed_indices:
                continue
            
            # Get entities
            original_entity = entities[original_idx]
            duplicate_entity = entities[duplicate_idx]
            
            # Resolve conflict based on strategy
            resolved_entity = self._resolve_entity_conflict(
                entities[original_idx], entities[duplicate_idx]
            )
            
            resolved_entities.append(resolved_entity)
            processed_indices.add(original_idx)
            processed_indices.add(duplicate_idx)
            
            self.stats['conflicts_resolved'] += 1
        
        # Add unprocessed entities
        for i, entity in enumerate(entities):
            if i not in processed_indices:
                resolved_entities.append(entity)
        
        return resolved_entities
    
    def _resolve_edge_conflicts(
        self,
        duplicates: List[Dict[str, Any]],
        edges: List[EntityEdge]
    ) -> List[EntityEdge]:
        """Resolve conflicts between duplicate edges"""
        resolved_edges = []
        processed_indices = set()
        
        for duplicate in duplicates:
            original_idx = duplicate['original_index']
            duplicate_idx = duplicate['duplicate_index']
            
            if original_idx in processed_indices or duplicate_idx in processed_indices:
                continue
            
            # Get edges
            original_edge = edges[original_idx]
            duplicate_edge = edges[duplicate_idx]
            
            # Resolve conflict based on strategy
            resolved_edge = self._resolve_edge_conflict(
                edges[original_idx], edges[duplicate_idx]
            )
            
            resolved_edges.append(resolved_edge)
            processed_indices.add(original_idx)
            processed_indices.add(duplicate_idx)
            
            self.stats['conflicts_resolved'] += 1
        
        # Add unprocessed edges
        for i, edge in enumerate(edges):
            if i not in processed_indices:
                resolved_edges.append(edge)
        
        return resolved_edges
    
    def _resolve_entity_conflict(
        self,
        entity1: EntityNode,
        entity2: EntityNode
    ) -> EntityNode:
        """Resolve conflict between two duplicate entities"""
        if self.config.conflict_resolution == "merge":
            return self._merge_entities(entity1, entity2)
        elif self.config.conflict_resolution == "keep_newer":
            return entity1 if entity1.created_at >= entity2.created_at else entity2
        elif self.config.conflict_resolution == "keep_older":
            return entity1 if entity1.created_at <= entity2.created_at else entity2
        elif self.config.conflict_resolution == "keep_better":
            return self._select_better_entity(entity1, entity2)
        else:
            # Default: keep the first one
            return entity1
    
    def _resolve_edge_conflict(
        self,
        edge1: EntityEdge,
        edge2: EntityEdge
    ) -> EntityEdge:
        """Resolve conflict between two duplicate edges"""
        if self.config.conflict_resolution == "merge":
            return self._merge_edges(edge1, edge2)
        elif self.config.conflict_resolution == "keep_newer":
            return edge1 if edge1.created_at >= edge2.created_at else edge2
        elif self.config.conflict_resolution == "keep_older":
            return edge1 if edge1.created_at <= edge2.created_at else edge2
        elif self.config.conflict_resolution == "keep_better":
            return self._select_better_edge(edge1, edge2)
        else:
            # Default: keep the first one
            return edge1
    
    def _merge_entities(self, entity1: EntityNode, entity2: EntityNode) -> EntityNode:
        """Merge two entities into one"""
        # Combine labels
        merged_labels = list(set(entity1.labels + entity2.labels))
        
        # Combine attributes
        merged_attributes = entity1.attributes.copy()
        for key, value in entity2.attributes.items():
            if key in merged_attributes:
                # If both have the attribute, combine them
                if isinstance(merged_attributes[key], list) and isinstance(value, list):
                    merged_attributes[key] = list(set(merged_attributes[key] + value))
                elif merged_attributes[key] != value:
                    # Different values, create a list
                    merged_attributes[key] = [merged_attributes[key], value]
            else:
                merged_attributes[key] = value
        
        # Use the earlier creation time
        merged_created_at = min(entity1.created_at, entity2.created_at)
        
        # Create merged entity
        merged_entity = EntityNode(
            name=entity1.name,  # Keep the first name
            labels=merged_labels,
            group_id=entity1.group_id,
            attributes=merged_attributes,
            created_at=merged_created_at
        )
        
        return merged_entity
    
    def _merge_edges(self, edge1: EntityEdge, edge2: EntityEdge) -> EntityEdge:
        """Merge two edges into one"""
        # Combine facts
        merged_fact = f"{edge1.fact} | {edge2.fact}"
        
        # Combine attributes
        merged_attributes = edge1.attributes.copy()
        for key, value in edge2.attributes.items():
            if key in merged_attributes:
                # If both have the attribute, combine them
                if isinstance(merged_attributes[key], list) and isinstance(value, list):
                    merged_attributes[key] = list(set(merged_attributes[key] + value))
                elif merged_attributes[key] != value:
                    # Different values, create a list
                    merged_attributes[key] = [merged_attributes[key], value]
            else:
                merged_attributes[key] = value
        
        # Combine episodes
        merged_episodes = list(set((edge1.episodes or []) + (edge2.episodes or [])))
        
        # Use the earlier creation time
        merged_created_at = min(edge1.created_at, edge2.created_at)
        
        # Create merged edge
        merged_edge = EntityEdge(
            source_node_uuid=edge1.source_node_uuid,
            target_node_uuid=edge1.target_node_uuid,
            name=edge1.name,  # Keep the first name
            fact=merged_fact,
            group_id=edge1.group_id,
            attributes=merged_attributes,
            episodes=merged_episodes,
            created_at=merged_created_at
        )
        
        return merged_edge
    
    def _select_better_entity(self, entity1: EntityNode, entity2: EntityNode) -> EntityNode:
        """Select the better quality entity"""
        score1 = self._calculate_entity_quality(entity1)
        score2 = self._calculate_entity_quality(entity2)
        
        return entity1 if score1 >= score2 else entity2
    
    def _select_better_edge(self, edge1: EntityEdge, edge2: EntityEdge) -> EntityEdge:
        """Select the better quality edge"""
        score1 = self._calculate_edge_quality_score(edge1)
        score2 = self._calculate_edge_quality_score(edge2)
        
        return edge1 if score1 >= score2 else edge2
    
    def _calculate_entity_quality(self, entity: EntityNode) -> float:
        """Calculate quality score for an entity"""
        score = 0.0
        
        # Name length (optimal around 3-10 words)
        name_words = len(entity.name.split()) if entity.name else 0
        if 3 <= name_words <= 10:
            score += 0.3
        elif name_words > 0:
            score += 0.1
        
        # Number of labels (optimal around 2-5)
        label_count = len(entity.labels)
        if 2 <= label_count <= 5:
            score += 0.3
        elif label_count > 0:
            score += 0.1
        
        # Has embedding
        if entity.name_embedding:
            score += 0.2
        
        # Has attributes
        if entity.attributes and len(entity.attributes) > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_edge_quality_score(self, edge: EntityEdge) -> float:
        """Calculate quality score for an edge"""
        score = 0.0
        
        # Fact length (optimal around 10-50 characters)
        fact_length = len(edge.fact) if edge.fact else 0
        if 10 <= fact_length <= 50:
            score += 0.3
        elif fact_length > 0:
            score += 0.1
        
        # Has embedding
        if edge.fact_embedding:
            score += 0.3
        
        # Has episodes
        if edge.episodes and len(edge.episodes) > 0:
            score += 0.2
        
        # Valid temporal information
        if edge.valid_at:
            score += 0.2
        return min(score, 1.0)
       