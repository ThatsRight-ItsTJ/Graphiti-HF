
"""
Enhanced Edge Extractor for Graphiti-HF

Provides advanced edge extraction capabilities with custom schemas,
temporal support, and relationship validation.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.embedder.client import EmbedderClient

logger = logging.getLogger(__name__)


@dataclass
class EdgeExtractionConfig:
    """Configuration for edge extraction"""
    model: str = "gpt-4"
    custom_types: Optional[Dict[str, BaseModel]] = None
    enable_validation: bool = True
    enable_embedding: bool = True
    enable_temporal: bool = True
    batch_size: int = 10
    similarity_threshold: float = 0.8
    context_window: int = 4000


class EdgeSchema(BaseModel):
    """Custom edge schema definition"""
    name: str
    description: str
    source_types: List[str]
    target_types: List[str]
    attributes: Dict[str, Any] = Field(default_factory=dict)
    required_attributes: List[str] = Field(default_factory=list)
    validation_rules: List[str] = Field(default_factory=list)
    temporal_attributes: List[str] = Field(default_factory=list)


class EdgeExtractor:
    """
    Enhanced edge extractor with custom schema support and temporal validation
    
    Handles relationship extraction between entities with configurable schemas,
    temporal support, and validation.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        embedder_client: EmbedderClient,
        config: Optional[EdgeExtractionConfig] = None
    ):
        """
        Initialize the edge extractor
        
        Args:
            llm_client: LLM client for edge extraction
            embedder_client: Embedder client for semantic embeddings
            config: Edge extraction configuration
        """
        self.llm_client = llm_client
        self.embedder_client = embedder_client
        self.config = config or EdgeExtractionConfig()
        
        # Default edge types
        self.default_edge_types = {
            "WORKS_AT": EdgeSchema(
                name="WORKS_AT",
                description="Employment relationship",
                source_types=["Person"],
                target_types=["Organization"],
                attributes={"role": str, "start_date": datetime, "end_date": datetime},
                required_attributes=[],
                validation_rules=[],
                temporal_attributes=["start_date", "end_date"]
            ),
            "LIVES_IN": EdgeSchema(
                name="LIVES_IN",
                description="Residence relationship",
                source_types=["Person"],
                target_types=["Location"],
                attributes={"since": datetime, "type": str},
                required_attributes=[],
                validation_rules=[],
                temporal_attributes=["since"]
            ),
            "LOCATED_IN": EdgeSchema(
                name="LOCATED_IN",
                description="Location containment relationship",
                source_types=["Location"],
                target_types=["Location"],
                attributes={"distance": float, "type": str},
                required_attributes=[],
                validation_rules=[],
                temporal_attributes=[]
            ),
            "PARTICIPATED_IN": EdgeSchema(
                name="PARTICIPATED_IN",
                description="Event participation relationship",
                source_types=["Person"],
                target_types=["Event"],
                attributes={"role": str, "contribution": str},
                required_attributes=[],
                validation_rules=[],
                temporal_attributes=[]
            ),
            "RELATED_TO": EdgeSchema(
                name="RELATED_TO",
                description="General relationship",
                source_types=["*"],
                target_types=["*"],
                attributes={"relationship_type": str, "strength": float},
                required_attributes=[],
                validation_rules=[],
                temporal_attributes=[]
            )
        }
        
        # Merge custom edge types
        if self.config.custom_types:
            self.default_edge_types.update(self.config.custom_types)
        
        # Entity mapping for quick lookup
        self.entity_map: Dict[str, EntityNode] = {}
        
        # Extraction statistics
        self.stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'validation_failures': 0,
            'embedding_failures': 0,
            'temporal_conflicts': 0
        }
    
    def set_entities(self, entities: List[EntityNode]):
        """Set the available entities for edge extraction"""
        self.entity_map = {entity.name: entity for entity in entities}
    
    async def extract_edges(
        self,
        content: str,
        entities: List[EntityNode],
        context: Optional[str] = None,
        edge_types: Optional[List[str]] = None,
        existing_edges: Optional[List[EntityEdge]] = None
    ) -> List[EntityEdge]:
        """
        Extract edges from content with enhanced capabilities
        
        Args:
            content: Text content to extract edges from
            entities: Available entities for relationship extraction
            context: Additional context for extraction
            edge_types: Specific edge types to extract (None for all)
            existing_edges: Existing edges for context and deduplication
            
        Returns:
            List of extracted EntityEdge objects
        """
        try:
            # Set entity mapping
            self.set_entities(entities)
            
            # Filter edge types if specified
            target_types = edge_types or list(self.default_edge_types.keys())
            schemas = {k: v for k, v in self.default_edge_types.items() if k in target_types}
            
            # Prepare extraction prompt
            prompt = self._build_extraction_prompt(content, entities, schemas, context, existing_edges)
            
            # Generate edges using LLM
            response = await self.llm_client.generate(prompt)
            
            # Parse response to extract edges
            edges = self._parse_edges_from_response(response, schemas)
            
            # Validate extracted edges
            if self.config.enable_validation:
                valid_edges = []
                for edge in edges:
                    if self._validate_edge(edge, schemas.get(edge.name, "")):
                        valid_edges.append(edge)
                        self.stats['successful_extractions'] += 1
                    else:
                        self.stats['validation_failures'] += 1
                edges = valid_edges
            
            # Generate embeddings
            if self.config.enable_embedding:
                await self._generate_embeddings(edges)
            
            # Update statistics
            self.stats['total_extractions'] += len(edges)
            
            return edges
            
        except Exception as e:
            logger.error(f"Error extracting edges: {e}")
            return []
    
    async def extract_edges_batch(
        self,
        contents: List[str],
        entities_list: List[List[EntityNode]],
        contexts: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None,
        existing_edges: Optional[List[EntityEdge]] = None
    ) -> List[List[EntityEdge]]:
        """
        Extract edges from multiple content pieces in batch
        
        Args:
            contents: List of text contents to extract edges from
            entities_list: List of entity lists, one per content piece
            contexts: Optional list of context strings
            edge_types: Specific edge types to extract
            existing_edges: Existing edges for context
            
        Returns:
            List of edge lists, one per content piece
        """
        if not contents or not entities_list:
            return []
        
        # Prepare contexts
        if contexts is None:
            contexts = [""] * len(contents)
        
        # Process in batches
        all_results = []
        for i in range(0, len(contents), self.config.batch_size):
            batch_contents = contents[i:i + self.config.batch_size]
            batch_entities = entities_list[i:i + self.config.batch_size]
            batch_contexts = contexts[i:i + self.config.batch_size]
            
            batch_results = []
            for content, entities, context in zip(batch_contents, batch_entities, batch_contexts):
                edges = await self.extract_edges(
                    content, entities, context, edge_types, existing_edges
                )
                batch_results.append(edges)
            
            all_results.extend(batch_results)
        
        return all_results
    
    async def add_custom_edge_type(
        self,
        type_name: str,
        schema: EdgeSchema
    ) -> bool:
        """
        Add a custom edge type to the extractor
        
        Args:
            type_name: Name of the edge type
            schema: Schema definition for the edge type
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            self.default_edge_types[type_name] = schema
            logger.info(f"Added custom edge type: {type_name}")
            return True
        except Exception as e:
            logger.error(f"Error adding custom edge type {type_name}: {e}")
            return False
    
    def get_edge_types(self) -> List[str]:
        """Get list of available edge types"""
        return list(self.default_edge_types.keys())
    
    def get_edge_schema(self, edge_type: str) -> Optional[EdgeSchema]:
        """Get schema for a specific edge type"""
        return self.default_edge_types.get(edge_type)
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset extraction statistics"""
        self.stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'validation_failures': 0,
            'embedding_failures': 0,
            'temporal_conflicts': 0
        }
    
    # Helper methods
    def _build_extraction_prompt(
        self,
        content: str,
        entities: List[EntityNode],
        schemas: Dict[str, EdgeSchema],
        context: Optional[str],
        existing_edges: Optional[List[EntityEdge]]
    ) -> str:
        """Build extraction prompt for LLM"""
        
        # Build entity context
        entity_context = "Available entities:\n"
        for entity in entities:
            entity_context += f"- {entity.name} ({', '.join(entity.labels)})\n"
        
        # Build edge type descriptions
        type_descriptions = []
        for type_name, schema in schemas.items():
            type_desc = f"{type_name}: {schema.description}"
            type_desc += f" (Source: {', '.join(schema.source_types)}, Target: {', '.join(schema.target_types)})"
            if schema.attributes:
                attrs = ", ".join(f"{k}: {v.__name__}" for k, v in schema.attributes.items())
                type_desc += f" (Attributes: {attrs})"
            type_descriptions.append(type_desc)
        
        # Build existing edges context
        existing_context = ""
        if existing_edges:
            existing_context = "Existing relationships in the knowledge graph:\n"
            for edge in existing_edges[:5]:  # Limit to avoid context overflow
                existing_context += f"- {edge.source_node_uuid} -> {edge.target_node_uuid}: {edge.fact}\n"
        
        # Build context
        full_context = ""
        if context:
            full_context = f"Additional context: {context}\n"
        if existing_context:
            full_context += existing_context
        
        # Build final prompt
        prompt = f"""
Extract relationships between entities from the following text. Focus on identifying relationships of these types:

{chr(10).join(type_descriptions)}

Available entities:
{entity_context}

Text content:
{content}

{full_context}

For each relationship found, provide:
1. Source entity name (from the available entities list)
2. Target entity name (from the available entities list)
3. Relationship type from the list above
4. Relevant attributes based on the relationship type schema
5. Confidence score (0.1-1.0)

Format your response as a JSON array of objects with these fields:
- source: string (entity name)
- target: string (entity name)
- name: string (relationship type)
- fact: string (natural language description)
- attributes: object with key-value pairs
- confidence: number (0.1-1.0)

Example:
[
  {{
    "source": "John Doe",
    "target": "Tech Corp",
    "name": "WORKS_AT",
    "fact": "John Doe works at Tech Corp as an engineer",
    "attributes": {{"role": "Engineer", "start_date": "2023-01-01"}},
    "confidence": 0.9
  }}
]
"""
        return prompt
    
    def _parse_edges_from_response(
        self,
        response: str,
        schemas: Dict[str, EdgeSchema]
    ) -> List[EntityEdge]:
        """Parse edges from LLM response"""
        edges = []
        
        try:
            # Try to parse as JSON first
            data = json.loads(response)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        edge = self._create_edge_from_dict(item, schemas)
                        if edge:
                            edges.append(edge)
        except json.JSONDecodeError:
            # Fallback to regex parsing
            logger.warning("Failed to parse JSON response, falling back to regex")
            edges = self._parse_edges_with_regex(response, schemas)
        
        return edges
    
    def _create_edge_from_dict(
        self,
        item: Dict[str, Any],
        schemas: Dict[str, EdgeSchema]
    ) -> Optional[EntityEdge]:
        """Create EntityEdge from dictionary item"""
        try:
            source_name = item.get('source', '').strip()
            target_name = item.get('target', '').strip()
            edge_name = item.get('name', '').strip()
            
            if not all([source_name, target_name, edge_name]):
                return None
            
            # Check if entities exist
            if source_name not in self.entity_map or target_name not in self.entity_map:
                return None
            
            # Get entity UUIDs
            source_uuid = self.entity_map[source_name].uuid
            target_uuid = self.entity_map[target_name].uuid
            
            # Validate edge type
            if edge_name not in schemas:
                return None
            
            # Get schema for validation
            schema = schemas[edge_name]
            
            # Process attributes
            attributes = {}
            if item.get('attributes'):
                for attr_name, attr_value in item.get('attributes', {}).items():
                    if attr_name in schema.attributes:
                        # Type conversion if needed
                        expected_type = schema.attributes[attr_name]
                        try:
                            if expected_type == datetime and isinstance(attr_value, str):
                                attributes[attr_name] = datetime.fromisoformat(attr_value)
                            elif expected_type == list and isinstance(attr_value, str):
                                attributes[attr_name] = attr_value.split(',')
                            else:
                                attributes[attr_name] = attr_value
                        except (ValueError, TypeError):
                            # Skip invalid attribute values
                            continue
            
            # Create edge
            edge = EntityEdge(
                source_node_uuid=source_uuid,
                target_node_uuid=target_uuid,
                name=edge_name,
                fact=item.get('fact', f"{source_name} {edge_name} {target_name}"),
                group_id='default',
                attributes=attributes
            )
            
            return edge
            
        except Exception as e:
            logger.error(f"Error creating edge from dict: {e}")
            return None
    
    def _parse_edges_with_regex(
        self,
        response: str,
        schemas: Dict[str, EdgeSchema]
    ) -> List[EntityEdge]:
        """Parse edges using regex as fallback"""
        edges = []
        
        # Simple regex patterns for edge extraction
        patterns = [
            r'Source:\s*(.*?)(?:\n|$)',
            r'Target:\s*(.*?)(?:\n|$)',
            r'Relationship:\s*(.*?)(?:\n|$)',
            r'Fact:\s*(.*?)(?:\n|$)'
        ]
        
        # This is a simplified fallback - in production, you'd want more sophisticated parsing
        lines = response.split('\n')
        current_edge = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match edge patterns
            if line.startswith('Source:'):
                current_edge['source'] = line.split(':', 1)[1].strip()
            elif line.startswith('Target:'):
                current_edge['target'] = line.split(':', 1)[1].strip()
            elif line.startswith('Relationship:') or line.startswith('Type:'):
                current_edge['name'] = line.split(':', 1)[1].strip()
            elif line.startswith('Fact:'):
                current_edge['fact'] = line.split(':', 1)[1].strip()
            
            # If we have a complete edge, create it
            if all(key in current_edge for key in ['source', 'target', 'name']):
                edge = self._create_edge_from_dict(current_edge, schemas)
                if edge:
                    edges.append(edge)
                current_edge = {}
        
        return edges
    
    def _validate_edge(
        self,
        edge: EntityEdge,
        schema: Optional[EdgeSchema] = None
    ) -> bool:
        """Validate an edge against its schema"""
        if not schema:
            return True  # No schema validation
        
        # Check if source and target types are compatible
        source_entity = next((e for e in self.entity_map.values() if e.uuid == edge.source_node_uuid), None)
        target_entity = next((e for e in self.entity_map.values() if e.uuid == edge.target_node_uuid), None)
        
        if not source_entity or not target_entity:
            return False
        
        # Check source type compatibility
        if "*" not in schema.source_types:
            source_label = source_entity.labels[0] if source_entity.labels else ""
            if source_label not in schema.source_types:
                return False
        
        # Check target type compatibility
        if "*" not in schema.target_types:
            target_label = target_entity.labels[0] if target_entity.labels else ""
            if target_label not in schema.target_types:
                return False
        
        # Check required attributes
        for req_attr in schema.required_attributes:
            if req_attr not in edge.attributes:
                return False
        
        # Apply validation rules
        for rule in schema.validation_rules:
            try:
                if not eval(rule, {}, {'edge': edge, 'attributes': edge.attributes}):
                    return False
            except Exception:
                # Skip invalid validation rules
                continue
        
        # Check temporal consistency if enabled
        if self.config.enable_temporal:
            temporal_errors = self._validate_temporal_consistency(edge, schema)
            if temporal_errors:
                self.stats['temporal_conflicts'] += len(temporal_errors)
                return False
        
        return True
    
    def _validate_temporal_consistency(
        self,
        edge: EntityEdge,
        schema: EdgeSchema
    ) -> List[str]:
        """Validate temporal consistency of edge attributes"""
        errors = []
        
        # Check temporal attributes
        for attr in schema.temporal_attributes:
            if attr in edge.attributes:
                attr_value = edge.attributes[attr]
                if isinstance(attr_value, datetime):
                    # Check if date is in the future
                    if attr_value > datetime.now():
                        errors.append(f"Temporal attribute {attr} is in the future")
        
        return errors
    
    async def _generate_embeddings(self, edges: List[EntityEdge]):
        """Generate embeddings for edges"""
        try:
            # Generate embeddings in batch for better performance
            edge_facts = [edge.fact for edge in edges if edge.fact]
            
            if edge_facts:
                embeddings = await self.embedder_client.embed_batch(edge_facts)
                
                # Assign embeddings back to edges
                for i, edge in enumerate(edges):
                    if i < len(embeddings):
                        edge.fact_embedding = embeddings[i]
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            self.stats['embedding_failures'] += len(edges)