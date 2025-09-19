"""
Episode Processor for Graphiti-HF

Enhanced episode processing capabilities for HuggingFace datasets with entity and edge extraction,
deduplication, and validation.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.prompts.extract_nodes import extract_nodes_prompt
from graphiti_core.prompts.extract_edges import extract_edges_prompt
from graphiti_core.prompts.dedupe_nodes import dedupe_nodes_prompt
from graphiti_core.prompts.dedupe_edges import dedupe_edges_prompt

logger = logging.getLogger(__name__)


@dataclass
class EpisodeProcessingConfig:
    """Configuration for episode processing"""
    entity_extraction_model: str = "gpt-4"
    edge_extraction_model: str = "gpt-4"
    embedder_model: str = "all-MiniLM-L6-v2"
    entity_types: Optional[List[str]] = None
    edge_types: Optional[List[str]] = None
    similarity_threshold: float = 0.8
    batch_size: int = 10
    max_workers: int = 4
    enable_validation: bool = True
    enable_deduplication: bool = True
    temporal_consistency: bool = True
    conflict_resolution: str = "merge"  # "merge", "keep_newer", "keep_older"


@dataclass
class EpisodeProcessingResult:
    """Result of episode processing"""
    episode_uuid: str
    processed_nodes: List[EntityNode]
    processed_edges: List[EntityEdge]
    duplicate_entities: List[str]
    duplicate_edges: List[str]
    validation_errors: List[str]
    processing_time: float
    quality_score: float


class EpisodeProcessor:
    """
    Enhanced episode processor for HuggingFace datasets
    
    Handles entity extraction, edge extraction, deduplication, and validation
    for episodic data in knowledge graphs.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        embedder_client: EmbedderClient,
        config: Optional[EpisodeProcessingConfig] = None
    ):
        """
        Initialize the episode processor
        
        Args:
            llm_client: LLM client for entity and edge extraction
            embedder_client: Embedder client for semantic similarity
            config: Processing configuration
        """
        self.llm_client = llm_client
        self.embedder_client = embedder_client
        self.config = config or EpisodeProcessingConfig()
        
        # Initialize text vectorizer for similarity calculations
        self.text_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        
        # Processing statistics
        self.stats = {
            'total_episodes_processed': 0,
            'total_entities_extracted': 0,
            'total_edges_extracted': 0,
            'duplicates_removed': 0,
            'validation_errors': 0
        }
    
    async def process_episode(
        self,
        episode_uuid: str,
        content: str,
        source_description: str = "",
        reference_time: Optional[datetime] = None,
        existing_nodes: Optional[List[EntityNode]] = None,
        existing_edges: Optional[List[EntityEdge]] = None
    ) -> EpisodeProcessingResult:
        """
        Process a single episode with entity and edge extraction
        
        Args:
            episode_uuid: UUID of the episode
            content: Text content to process
            source_description: Description of the source
            reference_time: Time reference for temporal consistency
            existing_nodes: Existing nodes for deduplication
            existing_edges: Existing edges for deduplication
            
        Returns:
            EpisodeProcessingResult with processing results
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Extract entities
            extracted_entities = await self.extract_entities(
                content, 
                existing_nodes or []
            )
            
            # Step 2: Extract edges
            extracted_edges = await self.extract_edges(
                content,
                extracted_entities,
                existing_edges or []
            )
            
            # Step 3: Deduplicate entities
            deduped_entities, duplicate_entities = await self.deduplicate_entities(
                extracted_entities,
                existing_nodes or []
            )
            
            # Step 4: Deduplicate edges
            deduped_edges, duplicate_edges = await self.deduplicate_edges(
                extracted_edges,
                existing_edges or []
            )
            
            # Step 5: Validate results
            validation_errors = []
            if self.config.enable_validation:
                validation_errors = await self.validate_episode(
                    deduped_entities,
                    deduped_edges,
                    content
                )
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                deduped_entities,
                deduped_edges,
                validation_errors
            )
            
            # Update statistics
            self.stats['total_episodes_processed'] += 1
            self.stats['total_entities_extracted'] += len(deduped_entities)
            self.stats['total_edges_extracted'] += len(deduped_edges)
            self.stats['duplicates_removed'] += len(duplicate_entities) + len(duplicate_edges)
            self.stats['validation_errors'] += len(validation_errors)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EpisodeProcessingResult(
                episode_uuid=episode_uuid,
                processed_nodes=deduped_entities,
                processed_edges=deduped_edges,
                duplicate_entities=duplicate_entities,
                duplicate_edges=duplicate_edges,
                validation_errors=validation_errors,
                processing_time=processing_time,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Error processing episode {episode_uuid}: {e}")
            raise
    
    async def process_episode_batch(
        self,
        episodes: List[Dict[str, Any]],
        existing_nodes: Optional[List[EntityNode]] = None,
        existing_edges: Optional[List[EntityEdge]] = None
    ) -> List[EpisodeProcessingResult]:
        """
        Process multiple episodes in batch for better performance
        
        Args:
            episodes: List of episode dictionaries with uuid, content, source_description, reference_time
            existing_nodes: Existing nodes for deduplication
            existing_edges: Existing edges for deduplication
            
        Returns:
            List of EpisodeProcessingResult objects
        """
        if not episodes:
            return []
        
        logger.info(f"Processing batch of {len(episodes)} episodes")
        
        # Process episodes in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            loop = asyncio.get_event_loop()
            
            tasks = []
            for episode in episodes:
                task = loop.run_in_executor(
                    executor,
                    lambda ep: asyncio.run(self._process_episode_sync(
                        ep['uuid'],
                        ep['content'],
                        ep.get('source_description', ''),
                        ep.get('reference_time'),
                        existing_nodes,
                        existing_edges
                    )),
                    episode
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch processing: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def extract_entities(
        self,
        content: str,
        existing_entities: Optional[List[EntityNode]] = None
    ) -> List[EntityNode]:
        """
        Extract entities from content using LLM with enhanced capabilities
        
        Args:
            content: Text content to extract entities from
            existing_entities: Existing entities for context
            
        Returns:
            List of extracted EntityNode objects
        """
        try:
            # Prepare prompt with existing entities for context
            context = ""
            if existing_entities:
                context = f"Existing entities in the knowledge graph:\n"
                for entity in existing_entities[:10]:  # Limit context length
                    context += f"- {entity.name} ({', '.join(entity.labels)})\n"
            
            # Build extraction prompt
            prompt = extract_nodes_prompt(
                text=content,
                entity_types=self.config.entity_types,
                context=context
            )
            
            # Get LLM response
            response = await self.llm_client.generate(prompt)
            
            # Parse response to extract entities
            entities = self._parse_entities_from_response(response)
            
            # Generate embeddings for entities
            for entity in entities:
                if entity.name:
                    entity.name_embedding = await self.embedder_client.embed(entity.name)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def extract_edges(
        self,
        content: str,
        extracted_entities: List[EntityNode],
        existing_edges: Optional[List[EntityEdge]] = None
    ) -> List[EntityEdge]:
        """
        Extract edges from content using LLM with enhanced capabilities
        
        Args:
            content: Text content to extract edges from
            extracted_entities: Entities that were extracted from the content
            existing_edges: Existing edges for context
            
        Returns:
            List of extracted EntityEdge objects
        """
        try:
            # Prepare entity context
            entity_context = ""
            for entity in extracted_entities:
                entity_context += f"{entity.name} ({', '.join(entity.labels)})\n"
            
            # Prepare edge context
            edge_context = ""
            if existing_edges:
                edge_context = f"Existing relationships in the knowledge graph:\n"
                for edge in existing_edges[:10]:  # Limit context length
                    edge_context += f"- {edge.source_node_uuid} -> {edge.target_node_uuid}: {edge.fact}\n"
            
            # Build extraction prompt
            prompt = extract_edges_prompt(
                text=content,
                entities=entity_context,
                edge_types=self.config.edge_types,
                context=edge_context
            )
            
            # Get LLM response
            response = await self.llm_client.generate(prompt)
            
            # Parse response to extract edges
            edges = self._parse_edges_from_response(response, extracted_entities)
            
            # Generate embeddings for edges
            for edge in edges:
                if edge.fact:
                    edge.fact_embedding = await self.embedder_client.embed(edge.fact)
            
            return edges
            
        except Exception as e:
            logger.error(f"Error extracting edges: {e}")
            return []
    
    async def deduplicate_entities(
        self,
        extracted_entities: List[EntityNode],
        existing_entities: Optional[List[EntityNode]] = None
    ) -> Tuple[List[EntityNode], List[str]]:
        """
        Deduplicate entities using name similarity and embeddings
        
        Args:
            extracted_entities: Newly extracted entities
            existing_entities: Existing entities in the knowledge graph
            
        Returns:
            Tuple of (deduplicated_entities, duplicate_entity_uuids)
        """
        if not self.config.enable_deduplication:
            return extracted_entities, []
        
        deduplicated_entities = []
        duplicate_uuids = []
        
        # Combine existing and extracted entities
        all_entities = (existing_entities or []) + extracted_entities
        
        # Create similarity matrix
        entity_names = [entity.name for entity in all_entities if entity.name]
        if len(entity_names) < 2:
            return extracted_entities, []
        
        # Calculate name similarities
        name_vectors = self.text_vectorizer.fit_transform(entity_names)
        name_similarities = cosine_similarity(name_vectors)
        
        # Calculate embedding similarities
        embedding_similarities = np.zeros((len(all_entities), len(all_entities)))
        for i, entity1 in enumerate(all_entities):
            for j, entity2 in enumerate(all_entities):
                if (entity1.name_embedding and entity2.name_embedding and 
                    i != j):
                    emb1 = np.array(entity1.name_embedding).reshape(1, -1)
                    emb2 = np.array(entity2.name_embedding).reshape(1, -1)
                    embedding_similarities[i][j] = cosine_similarity(emb1, emb2)[0][0]
        
        # Combine similarity scores
        combined_similarities = (name_similarities + embedding_similarities) / 2
        
        # Find duplicates
        processed_indices = set()
        for i, entity in enumerate(all_entities):
            if i in processed_indices:
                continue
                
            # Find similar entities
            similar_indices = np.where(combined_similarities[i] >= self.config.similarity_threshold)[0]
            
            # Skip if no duplicates found
            if len(similar_indices) <= 1:
                deduplicated_entities.append(entity)
                continue
            
            # Mark all similar entities as processed
            processed_indices.update(similar_indices)
            
            # Keep the entity with highest quality score
            best_entity = entity
            best_score = self._calculate_entity_quality(entity)
            
            for j in similar_indices:
                if j != i:
                    score = self._calculate_entity_quality(all_entities[j])
                    if score > best_score:
                        best_entity = all_entities[j]
                        best_score = score
            
            deduplicated_entities.append(best_entity)
            
            # Mark other entities as duplicates
            for j in similar_indices:
                if j != i and all_entities[j].uuid not in [e.uuid for e in deduplicated_entities]:
                    duplicate_uuids.append(all_entities[j].uuid)
        
        # Filter out duplicates from extracted entities
        final_entities = [
            entity for entity in extracted_entities 
            if entity.uuid not in duplicate_uuids
        ]
        
        return final_entities, duplicate_uuids
    
    async def deduplicate_edges(
        self,
        extracted_edges: List[EntityEdge],
        existing_edges: Optional[List[EntityEdge]] = None
    ) -> Tuple[List[EntityEdge], List[str]]:
        """
        Deduplicate edges using fact similarity and embeddings
        
        Args:
            extracted_edges: Newly extracted edges
            existing_edges: Existing edges in the knowledge graph
            
        Returns:
            Tuple of (deduplicated_edges, duplicate_edge_uuids)
        """
        if not self.config.enable_deduplication:
            return extracted_edges, []
        
        deduplicated_edges = []
        duplicate_uuids = []
        
        # Combine existing and extracted edges
        all_edges = (existing_edges or []) + extracted_edges
        
        # Create similarity matrix
        edge_facts = [edge.fact for edge in all_edges if edge.fact]
        if len(edge_facts) < 2:
            return extracted_edges, []
        
        # Calculate fact similarities
        fact_vectors = self.text_vectorizer.fit_transform(edge_facts)
        fact_similarities = cosine_similarity(fact_vectors)
        
        # Calculate embedding similarities
        embedding_similarities = np.zeros((len(all_edges), len(all_edges)))
        for i, edge1 in enumerate(all_edges):
            for j, edge2 in enumerate(all_edges):
                if (edge1.fact_embedding and edge2.fact_embedding and 
                    i != j):
                    emb1 = np.array(edge1.fact_embedding).reshape(1, -1)
                    emb2 = np.array(edge2.fact_embedding).reshape(1, -1)
                    embedding_similarities[i][j] = cosine_similarity(emb1, emb2)[0][0]
        
        # Combine similarity scores
        combined_similarities = (fact_similarities + embedding_similarities) / 2
        
        # Find duplicates
        processed_indices = set()
        for i, edge in enumerate(all_edges):
            if i in processed_indices:
                continue
                
            # Find similar edges
            similar_indices = np.where(combined_similarities[i] >= self.config.similarity_threshold)[0]
            
            # Skip if no duplicates found
            if len(similar_indices) <= 1:
                deduplicated_edges.append(edge)
                continue
            
            # Mark all similar edges as processed
            processed_indices.update(similar_indices)
            
            # Keep the edge with highest quality score
            best_edge = edge
            best_score = self._calculate_edge_quality(edge)
            
            for j in similar_indices:
                if j != i:
                    score = self._calculate_edge_quality(all_edges[j])
                    if score > best_score:
                        best_edge = all_edges[j]
                        best_score = score
            
            deduplicated_edges.append(best_edge)
            
            # Mark other edges as duplicates
            for j in similar_indices:
                if j != i and all_edges[j].uuid not in [e.uuid for e in deduplicated_edges]:
                    duplicate_uuids.append(all_edges[j].uuid)
        
        # Filter out duplicates from extracted edges
        final_edges = [
            edge for edge in extracted_edges 
            if edge.uuid not in duplicate_uuids
        ]
        
        return final_edges, duplicate_uuids
    
    async def validate_episode(
        self,
        entities: List[EntityNode],
        edges: List[EntityEdge],
        content: str
    ) -> List[str]:
        """
        Validate extracted entities and edges against content and existing knowledge
        
        Args:
            entities: Extracted entities to validate
            edges: Extracted edges to validate
            content: Original content for validation
            
        Returns:
            List of validation error messages
        """
        validation_errors = []
        
        # Validate entity names are not empty
        for entity in entities:
            if not entity.name or not entity.name.strip():
                validation_errors.append(f"Entity has empty name: {entity.uuid}")
        
        # Validate edge facts are not empty
        for edge in edges:
            if not edge.fact or not edge.fact.strip():
                validation_errors.append(f"Edge has empty fact: {edge.uuid}")
        
        # Validate edge connectivity
        entity_uuids = {entity.uuid for entity in entities}
        for edge in edges:
            if edge.source_node_uuid not in entity_uuids:
                validation_errors.append(
                    f"Edge source node {edge.source_node_uuid} not found in extracted entities"
                )
            if edge.target_node_uuid not in entity_uuids:
                validation_errors.append(
                    f"Edge target node {edge.target_node_uuid} not found in extracted entities"
                )
        
        # Validate temporal consistency if enabled
        if self.config.temporal_consistency:
            temporal_errors = self._validate_temporal_consistency(entities, edges)
            validation_errors.extend(temporal_errors)
        
        # Validate content coverage
        coverage_errors = self._validate_content_coverage(entities, edges, content)
        validation_errors.extend(coverage_errors)
        
        return validation_errors
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'total_episodes_processed': 0,
            'total_entities_extracted': 0,
            'total_edges_extracted': 0,
            'duplicates_removed': 0,
            'validation_errors': 0
        }
    
    # Helper methods
    async def _process_episode_sync(
        self,
        episode_uuid: str,
        content: str,
        source_description: str,
        reference_time: Optional[datetime],
        existing_nodes: Optional[List[EntityNode]],
        existing_edges: Optional[List[EntityEdge]]
    ) -> EpisodeProcessingResult:
        """Synchronous wrapper for episode processing"""
        return await self.process_episode(
            episode_uuid,
            content,
            source_description,
            reference_time,
            existing_nodes,
            existing_edges
        )
    
    def _parse_entities_from_response(self, response: str) -> List[EntityNode]:
        """Parse entities from LLM response"""
        entities = []
        
        try:
            # Try to parse as JSON first
            data = json.loads(response)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        entity = EntityNode(
                            name=item.get('name', ''),
                            labels=item.get('labels', []),
                            group_id=item.get('group_id', 'default'),
                            attributes=item.get('attributes', {})
                        )
                        entities.append(entity)
        except json.JSONDecodeError:
            # Fallback to regex parsing
            entity_pattern = r'Entity:\s*(.*?)(?:\n|$)'
            name_pattern = r'Name:\s*(.*?)(?:\n|$)'
            labels_pattern = r'Labels:\s*(.*?)(?:\n|$)'
            
            entities_text = re.findall(entity_pattern, response, re.IGNORECASE)
            for entity_text in entities_text:
                name_match = re.search(name_pattern, entity_text, re.IGNORECASE)
                labels_match = re.search(labels_pattern, entity_text, re.IGNORECASE)
                
                if name_match:
                    entity = EntityNode(
                        name=name_match.group(1).strip(),
                        labels=labels_match.group(1).split(',') if labels_match else [],
                        group_id='default'
                    )
                    entities.append(entity)
        
        return entities
    
    def _parse_edges_from_response(self, response: str, entities: List[EntityNode]) -> List[EntityEdge]:
        """Parse edges from LLM response"""
        edges = []
        entity_map = {entity.name: entity.uuid for entity in entities}
        
        try:
            # Try to parse as JSON first
            data = json.loads(response)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        source_name = item.get('source', '')
                        target_name = item.get('target', '')
                        fact = item.get('fact', '')
                        
                        if source_name in entity_map and target_name in entity_map:
                            edge = EntityEdge(
                                source_node_uuid=entity_map[source_name],
                                target_node_uuid=entity_map[target_name],
                                fact=fact,
                                group_id=item.get('group_id', 'default')
                            )
                            edges.append(edge)
        except json.JSONDecodeError:
            # Fallback to regex parsing
            source_pattern = r'Source:\s*(.*?)(?:\n|$)'
            target_pattern = r'Target:\s*(.*?)(?:\n|$)'
            fact_pattern = r'Fact:\s*(.*?)(?:\n|$)'
            
            edges_text = re.findall(r'Relationship:\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
            for edge_text in edges_text:
                source_match = re.search(source_pattern, edge_text, re.IGNORECASE)
                target_match = re.search(target_pattern, edge_text, re.IGNORECASE)
                fact_match = re.search(fact_pattern, edge_text, re.IGNORECASE)
                
                if (source_match and target_match and fact_match and
                    source_match.group(1) in entity_map and 
                    target_match.group(1) in entity_map):
                    
                    edge = EntityEdge(
                        source_node_uuid=entity_map[source_match.group(1)],
                        target_node_uuid=entity_map[target_match.group(1)],
                        fact=fact_match.group(1).strip(),
                        group_id='default'
                    )
                    edges.append(edge)
        
        return edges
    
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
    
    def _calculate_edge_quality(self, edge: EntityEdge) -> float:
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
    
    def _calculate_quality_score(
        self,
        entities: List[EntityNode],
        edges: List[EntityEdge],
        validation_errors: List[str]
    ) -> float:
        """Calculate overall quality score for episode processing"""
        if not entities and not edges:
            return 0.0
        
        # Entity quality average
        entity_scores = [self._calculate_entity_quality(entity) for entity in entities]
        avg_entity_score = sum(entity_scores) / len(entity_scores) if entity_scores else 0.0
        
        # Edge quality average
        edge_scores = [self._calculate_edge_quality(edge) for edge in edges]
        avg_edge_score = sum(edge_scores) / len(edge_scores) if edge_scores else 0.0
        
        # Validation penalty
        validation_penalty = min(len(validation_errors) * 0.1, 0.5)
        
        # Combined score
        combined_score = (avg_entity_score + avg_edge_score) / 2 - validation_penalty
        return max(0.0, min(1.0, combined_score))
    
    def _validate_temporal_consistency(
        self,
        entities: List[EntityNode],
        edges: List[EntityEdge]
    ) -> List[str]:
        """Validate temporal consistency of entities and edges"""
        errors = []
        
        # Check for temporal conflicts in edges
        for edge in edges:
            if edge.valid_at and edge.invalidated_at:
                if edge.valid_at >= edge.invalidated_at:
                    errors.append(
                        f"Edge {edge.uuid} has valid_at >= invalidated_at"
                    )
        
        return errors
    
    def _validate_content_coverage(
        self,
        entities: List[EntityNode],
        edges: List[EntityEdge],
        content: str
    ) -> List[str]:
        """Validate that extracted content covers the original content well"""
        errors = []
        
        # Simple heuristic: check if important keywords are covered
        content_words = set(content.lower().split())
        
        # Collect entity and edge keywords
        extracted_keywords = set()
        for entity in entities:
            if entity.name:
                extracted_keywords.update(entity.name.lower().split())
        
        for edge in edges:
            if edge.fact:
                extracted_keywords.update(edge.fact.lower().split())
        
        # Calculate coverage
        if content_words:
            coverage = len(content_words.intersection(extracted_keywords)) / len(content_words)
            if coverage < 0.1:  # Less than 10% coverage
                errors.append(
                    f"Low content coverage: {coverage:.2%} of content words covered by extracted entities/edges"
                )
        
        return errors