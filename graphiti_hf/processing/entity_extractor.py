"""
Enhanced Entity Extractor for Graphiti-HF

Provides advanced entity extraction capabilities with custom schemas,
validation, and embedding generation.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np

from graphiti_core.nodes import EntityNode
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.embedder.client import EmbedderClient

logger = logging.getLogger(__name__)


@dataclass
class EntityExtractionConfig:
    """Configuration for entity extraction"""
    model: str = "gpt-4"
    custom_types: Optional[Dict[str, BaseModel]] = None
    enable_validation: bool = True
    enable_embedding: bool = True
    batch_size: int = 10
    similarity_threshold: float = 0.8
    context_window: int = 4000


class EntitySchema(BaseModel):
    """Custom entity schema definition"""
    name: str
    description: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    required_attributes: List[str] = Field(default_factory=list)
    validation_rules: List[str] = Field(default_factory=list)


class EntityExtractor:
    """
    Enhanced entity extractor with custom schema support and validation
    
    Handles entity extraction from text with configurable schemas,
    validation rules, and embedding generation.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        embedder_client: EmbedderClient,
        config: Optional[EntityExtractionConfig] = None
    ):
        """
        Initialize the entity extractor
        
        Args:
            llm_client: LLM client for entity extraction
            embedder_client: Embedder client for semantic embeddings
            config: Entity extraction configuration
        """
        self.llm_client = llm_client
        self.embedder_client = embedder_client
        self.config = config or EntityExtractionConfig()
        
        # Default entity types
        self.default_entity_types = {
            "Person": EntitySchema(
                name="Person",
                description="A person or individual",
                attributes={"age": int, "occupation": str, "location": str},
                required_attributes=[],
                validation_rules=[]
            ),
            "Organization": EntitySchema(
                name="Organization", 
                description="A company, institution, or organization",
                attributes={"industry": str, "founded_year": int, "size": str},
                required_attributes=[],
                validation_rules=[]
            ),
            "Location": EntitySchema(
                name="Location",
                description="A place or geographical location", 
                attributes={"country": str, "city": str, "coordinates": List[float]},
                required_attributes=[],
                validation_rules=[]
            ),
            "Event": EntitySchema(
                name="Event",
                description="An occurrence or happening",
                attributes={"date": datetime, "participants": List[str], "outcome": str},
                required_attributes=[],
                validation_rules=[]
            )
        }
        
        # Merge custom entity types
        if self.config.custom_types:
            self.default_entity_types.update(self.config.custom_types)
        
        # Extraction statistics
        self.stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'validation_failures': 0,
            'embedding_failures': 0
        }
    
    async def extract_entities(
        self,
        content: str,
        context: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        existing_entities: Optional[List[EntityNode]] = None
    ) -> List[EntityNode]:
        """
        Extract entities from content with enhanced capabilities
        
        Args:
            content: Text content to extract entities from
            context: Additional context for extraction
            entity_types: Specific entity types to extract (None for all)
            existing_entities: Existing entities for context and deduplication
            
        Returns:
            List of extracted EntityNode objects
        """
        try:
            # Filter entity types if specified
            target_types = entity_types or list(self.default_entity_types.keys())
            schemas = {k: v for k, v in self.default_entity_types.items() if k in target_types}
            
            # Prepare extraction prompt
            prompt = self._build_extraction_prompt(content, schemas, context, existing_entities)
            
            # Generate entities using LLM
            response = await self.llm_client.generate(prompt)
            
            # Parse response to extract entities
            entities = self._parse_entities_from_response(response, schemas)
            
            # Validate extracted entities
            if self.config.enable_validation:
                valid_entities = []
                for entity in entities:
                    if self._validate_entity(entity, schemas.get(entity.labels[0] if entity.labels else "")):
                        valid_entities.append(entity)
                        self.stats['successful_extractions'] += 1
                    else:
                        self.stats['validation_failures'] += 1
                entities = valid_entities
            
            # Generate embeddings
            if self.config.enable_embedding:
                await self._generate_embeddings(entities)
            
            # Update statistics
            self.stats['total_extractions'] += len(entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def extract_entities_batch(
        self,
        contents: List[str],
        contexts: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
        existing_entities: Optional[List[EntityNode]] = None
    ) -> List[List[EntityNode]]:
        """
        Extract entities from multiple content pieces in batch
        
        Args:
            contents: List of text contents to extract entities from
            contexts: Optional list of context strings
            entity_types: Specific entity types to extract
            existing_entities: Existing entities for context
            
        Returns:
            List of entity lists, one per content piece
        """
        if not contents:
            return []
        
        # Prepare contexts
        if contexts is None:
            contexts = [""] * len(contents)
        
        # Process in batches
        all_results = []
        for i in range(0, len(contents), self.config.batch_size):
            batch_contents = contents[i:i + self.config.batch_size]
            batch_contexts = contexts[i:i + self.config.batch_size]
            
            batch_results = []
            for content, context in zip(batch_contents, batch_contexts):
                entities = await self.extract_entities(
                    content, context, entity_types, existing_entities
                )
                batch_results.append(entities)
            
            all_results.extend(batch_results)
        
        return all_results
    
    async def add_custom_entity_type(
        self,
        type_name: str,
        schema: EntitySchema
    ) -> bool:
        """
        Add a custom entity type to the extractor
        
        Args:
            type_name: Name of the entity type
            schema: Schema definition for the entity type
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            self.default_entity_types[type_name] = schema
            logger.info(f"Added custom entity type: {type_name}")
            return True
        except Exception as e:
            logger.error(f"Error adding custom entity type {type_name}: {e}")
            return False
    
    def get_entity_types(self) -> List[str]:
        """Get list of available entity types"""
        return list(self.default_entity_types.keys())
    
    def get_entity_schema(self, entity_type: str) -> Optional[EntitySchema]:
        """Get schema for a specific entity type"""
        return self.default_entity_types.get(entity_type)
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset extraction statistics"""
        self.stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'validation_failures': 0,
            'embedding_failures': 0
        }
    
    # Helper methods
    def _build_extraction_prompt(
        self,
        content: str,
        schemas: Dict[str, EntitySchema],
        context: Optional[str],
        existing_entities: Optional[List[EntityNode]]
    ) -> str:
        """Build extraction prompt for LLM"""
        
        # Build entity type descriptions
        type_descriptions = []
        for type_name, schema in schemas.items():
            type_desc = f"{type_name}: {schema.description}"
            if schema.attributes:
                attrs = ", ".join(f"{k}: {v.__name__}" for k, v in schema.attributes.items())
                type_desc += f" (Attributes: {attrs})"
            type_descriptions.append(type_desc)
        
        # Build existing entities context
        existing_context = ""
        if existing_entities:
            existing_context = "Existing entities in the knowledge graph:\n"
            for entity in existing_entities[:5]:  # Limit to avoid context overflow
                existing_context += f"- {entity.name} ({', '.join(entity.labels)})\n"
        
        # Build context
        full_context = ""
        if context:
            full_context = f"Additional context: {context}\n"
        if existing_context:
            full_context += existing_context
        
        # Build final prompt
        prompt = f"""
Extract entities from the following text. Focus on identifying entities of these types:

{chr(10).join(type_descriptions)}

Text content:
{content}

{full_context}

For each entity found, provide:
1. Entity name (the specific name/identifier)
2. Entity type(s) from the list above
3. Relevant attributes based on the entity type schema
4. Confidence score (0.1-1.0)

Format your response as a JSON array of objects with these fields:
- name: string
- labels: array of strings (entity types)
- attributes: object with key-value pairs
- confidence: number (0.1-1.0)

Example:
[
  {{
    "name": "John Doe",
    "labels": ["Person"],
    "attributes": {{"age": 35, "occupation": "Engineer"}},
    "confidence": 0.9
  }}
]
"""
        return prompt
    
    def _parse_entities_from_response(
        self,
        response: str,
        schemas: Dict[str, EntitySchema]
    ) -> List[EntityNode]:
        """Parse entities from LLM response"""
        entities = []
        
        try:
            # Try to parse as JSON first
            data = json.loads(response)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        entity = self._create_entity_from_dict(item, schemas)
                        if entity:
                            entities.append(entity)
        except json.JSONDecodeError:
            # Fallback to regex parsing
            logger.warning("Failed to parse JSON response, falling back to regex")
            entities = self._parse_entities_with_regex(response, schemas)
        
        return entities
    
    def _create_entity_from_dict(
        self,
        item: Dict[str, Any],
        schemas: Dict[str, EntitySchema]
    ) -> Optional[EntityNode]:
        """Create EntityNode from dictionary item"""
        try:
            name = item.get('name', '').strip()
            if not name:
                return None
            
            labels = item.get('labels', [])
            if not labels:
                return None
            
            # Validate entity type
            valid_labels = []
            for label in labels:
                if label in schemas:
                    valid_labels.append(label)
            
            if not valid_labels:
                return None
            
            # Get schema for validation
            primary_schema = schemas.get(valid_labels[0])
            
            # Process attributes
            attributes = {}
            if primary_schema and item.get('attributes'):
                for attr_name, attr_value in item.get('attributes', {}).items():
                    if attr_name in primary_schema.attributes:
                        # Type conversion if needed
                        expected_type = primary_schema.attributes[attr_name]
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
            
            # Create entity
            entity = EntityNode(
                name=name,
                labels=valid_labels,
                group_id='default',
                attributes=attributes
            )
            
            return entity
            
        except Exception as e:
            logger.error(f"Error creating entity from dict: {e}")
            return None
    
    def _parse_entities_with_regex(
        self,
        response: str,
        schemas: Dict[str, EntitySchema]
    ) -> List[EntityNode]:
        """Parse entities using regex as fallback"""
        entities = []
        
        # Simple regex patterns for entity extraction
        entity_patterns = [
            r'Entity:\s*(.*?)(?:\n|$)',
            r'Name:\s*(.*?)(?:\n|$)',
            r'Type[s]?:\s*(.*?)(?:\n|$)',
            r'Attributes?:\s*(.*?)(?:\n|$)'
        ]
        
        # This is a simplified fallback - in production, you'd want more sophisticated parsing
        lines = response.split('\n')
        current_entity = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match entity patterns
            if line.startswith('Entity:') or line.startswith('Name:'):
                current_entity['name'] = line.split(':', 1)[1].strip()
            elif line.startswith('Type:') or line.startswith('Types:'):
                types = line.split(':', 1)[1].strip()
                current_entity['labels'] = [t.strip() for t in types.split(',')]
            elif line.startswith('Attributes:') or line.startswith('Attribute:'):
                attr_text = line.split(':', 1)[1].strip()
                # Simple attribute parsing
                current_entity['attributes'] = {'description': attr_text}
            
            # If we have a complete entity, create it
            if 'name' in current_entity and 'labels' in current_entity:
                entity = self._create_entity_from_dict(current_entity, schemas)
                if entity:
                    entities.append(entity)
                current_entity = {}
        
        return entities
    
    def _validate_entity(
        self,
        entity: EntityNode,
        schema: Optional[EntitySchema] = None
    ) -> bool:
        """Validate an entity against its schema"""
        if not schema:
            return True  # No schema validation
        
        # Check required attributes
        for req_attr in schema.required_attributes:
            if req_attr not in entity.attributes:
                return False
        
        # Apply validation rules
        for rule in schema.validation_rules:
            try:
                if not eval(rule, {}, {'entity': entity, 'attributes': entity.attributes}):
                    return False
            except Exception:
                # Skip invalid validation rules
                continue
        
        return True
    
    async def _generate_embeddings(self, entities: List[EntityNode]):
        """Generate embeddings for entities"""
        try:
            # Generate embeddings in batch for better performance
            entity_names = [entity.name for entity in entities if entity.name]
            
            if entity_names:
                embeddings = await self.embedder_client.embed_batch(entity_names)
                
                # Assign embeddings back to entities
                for i, entity in enumerate(entities):
                    if i < len(embeddings):
                        entity.name_embedding = embeddings[i]
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            self.stats['embedding_failures'] += len(entities)