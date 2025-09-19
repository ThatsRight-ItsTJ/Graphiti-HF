"""
Custom Entity and Edge Types Support for Graphiti-HF

This module provides comprehensive support for custom entity and edge types using Pydantic models,
allowing users to define domain-specific knowledge graph structures while maintaining compatibility
with the existing Graphiti ecosystem.
"""

from typing import Dict, List, Optional, Any, Type, Union, TypeVar
from datetime import datetime
import json
import uuid
from pydantic import BaseModel, Field, validator
from abc import ABC, abstractmethod

# Import core Graphiti types
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge

# Type variables for generic type hints
EntityType = TypeVar('EntityType', bound=BaseModel)
EdgeType = TypeVar('EdgeType', bound=BaseModel)


class CustomTypeManager:
    """
    Manager class for registering, validating, and managing custom entity and edge types.
    
    This class provides a centralized way to define custom types, validate instances,
    and handle serialization/deserialization for the Graphiti-HF system.
    """
    
    def __init__(self):
        self.entity_types: Dict[str, Type[BaseModel]] = {}
        self.edge_types: Dict[str, Type[BaseModel]] = {}
        self.type_schemas: Dict[str, Dict[str, Any]] = {}
        self.edge_type_map: Dict[tuple, List[str]] = {}
        
        # Register default custom types
        self._register_default_types()
    
    def _register_default_types(self):
        """Register the default custom entity and edge types"""
        # Entity types
        self.register_entity_type('Person', PersonEntity)
        self.register_entity_type('Company', CompanyEntity)
        self.register_entity_type('Project', ProjectEntity)
        self.register_entity_type('Document', DocumentEntity)
        self.register_entity_type('Event', EventEntity)
        
        # Edge types
        self.register_edge_type('WorksAt', WorksAtEdge)
        self.register_edge_type('CollaboratesOn', CollaboratesOnEdge)
        self.register_edge_type('AuthoredBy', AuthoredByEdge)
        self.register_edge_type('ParticipatesIn', ParticipatesInEdge)
        self.register_edge_type('RelatedTo', RelatedToEdge)
        
        # Define default edge type mappings
        self.edge_type_map = {
            ('Person', 'Company'): ['WorksAt'],
            ('Person', 'Project'): ['CollaboratesOn'],
            ('Person', 'Document'): ['AuthoredBy'],
            ('Person', 'Event'): ['ParticipatesIn'],
            ('Company', 'Project'): ['Sponsors', 'Owns'],
            ('Company', 'Event'): ['Sponsors'],
            ('Project', 'Document'): ['Contains'],
            ('Project', 'Event'): ['Milestone'],
            ('Document', 'Event'): ['References'],
        }
    
    def register_entity_type(self, type_name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a custom entity type.
        
        Args:
            type_name: Name of the entity type
            model_class: Pydantic model class for the entity type
        """
        self.entity_types[type_name] = model_class
        self.type_schemas[type_name] = {
            'type': 'entity',
            'schema': model_class.schema(),
            'fields': list(model_class.__fields__.keys())
        }
    
    def register_edge_type(self, type_name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a custom edge type.
        
        Args:
            type_name: Name of the edge type
            model_class: Pydantic model class for the edge type
        """
        self.edge_types[type_name] = model_class
        self.type_schemas[type_name] = {
            'type': 'edge',
            'schema': model_class.schema(),
            'fields': list(model_class.__fields__.keys())
        }
    
    def validate_entity(self, entity_type: str, entity_data: Dict[str, Any]) -> BaseModel:
        """
        Validate entity data against the registered entity type.
        
        Args:
            entity_type: Name of the entity type
            entity_data: Dictionary containing entity data
            
        Returns:
            Validated entity instance
            
        Raises:
            ValueError: If entity type is not registered or validation fails
        """
        if entity_type not in self.entity_types:
            raise ValueError(f"Entity type '{entity_type}' is not registered")
        
        try:
            return self.entity_types[entity_type](**entity_data)
        except Exception as e:
            raise ValueError(f"Validation failed for entity type '{entity_type}': {str(e)}")
    
    def validate_edge(self, edge_type: str, edge_data: Dict[str, Any]) -> BaseModel:
        """
        Validate edge data against the registered edge type.
        
        Args:
            edge_type: Name of the edge type
            edge_data: Dictionary containing edge data
            
        Returns:
            Validated edge instance
            
        Raises:
            ValueError: If edge type is not registered or validation fails
        """
        if edge_type not in self.edge_types:
            raise ValueError(f"Edge type '{edge_type}' is not registered")
        
        try:
            return self.edge_types[edge_type](**edge_data)
        except Exception as e:
            raise ValueError(f"Validation failed for edge type '{edge_type}': {str(e)}")
    
    def get_type_schema(self, type_name: str) -> Dict[str, Any]:
        """
        Retrieve the schema for a registered type.
        
        Args:
            type_name: Name of the type
            
        Returns:
            Schema dictionary for the type
            
        Raises:
            ValueError: If type is not registered
        """
        if type_name not in self.type_schemas:
            raise ValueError(f"Type '{type_name}' is not registered")
        
        return self.type_schemas[type_name]
    
    def get_valid_edge_types(self, source_type: str, target_type: str) -> List[str]:
        """
        Get valid edge types between two entity types.
        
        Args:
            source_type: Source entity type
            target_type: Target entity type
            
        Returns:
            List of valid edge type names
        """
        return self.edge_type_map.get((source_type, target_type), [])
    
    def is_valid_edge_type(self, source_type: str, target_type: str, edge_type: str) -> bool:
        """
        Check if an edge type is valid between two entity types.
        
        Args:
            source_type: Source entity type
            target_type: Target entity type
            edge_type: Edge type to validate
            
        Returns:
            True if edge type is valid, False otherwise
        """
        valid_edge_types = self.get_valid_edge_types(source_type, target_type)
        return edge_type in valid_edge_types
    
    def add_edge_type_mapping(self, source_type: str, target_type: str, edge_types: List[str]) -> None:
        """
        Add a new edge type mapping.
        
        Args:
            source_type: Source entity type
            target_type: Target entity type
            edge_types: List of valid edge type names
        """
        self.edge_type_map[(source_type, target_type)] = edge_types
    
    def list_entity_types(self) -> List[str]:
        """Get list of registered entity type names."""
        return list(self.entity_types.keys())
    
    def list_edge_types(self) -> List[str]:
        """Get list of registered edge type names."""
        return list(self.edge_types.keys())


# Custom Entity Types
class PersonEntity(BaseModel):
    """Custom entity type representing a person."""
    
    name: str = Field(..., description="Full name of the person")
    age: Optional[int] = Field(None, description="Age of the person")
    occupation: Optional[str] = Field(None, description="Occupation or job title")
    location: Optional[str] = Field(None, description="Location or address")
    skills: List[str] = Field(default_factory=list, description="List of skills")
    
    @validator('age')
    def validate_age(cls, v):
        if v is not None and v < 0:
            raise ValueError('Age cannot be negative')
        return v
    
    class Config:
        extra = "allow"


class CompanyEntity(BaseModel):
    """Custom entity type representing a company."""
    
    name: str = Field(..., description="Name of the company")
    industry: Optional[str] = Field(None, description="Industry sector")
    founded_year: Optional[int] = Field(None, description="Year the company was founded")
    headquarters: Optional[str] = Field(None, description="Headquarters location")
    employee_count: Optional[int] = Field(None, description="Number of employees")
    
    @validator('founded_year')
    def validate_founded_year(cls, v):
        if v is not None and v < 1800:
            raise ValueError('Founded year must be after 1800')
        return v
    
    @validator('employee_count')
    def validate_employee_count(cls, v):
        if v is not None and v < 0:
            raise ValueError('Employee count cannot be negative')
        return v
    
    class Config:
        extra = "allow"


class ProjectEntity(BaseModel):
    """Custom entity type representing a project."""
    
    name: str = Field(..., description="Name of the project")
    description: Optional[str] = Field(None, description="Project description")
    start_date: Optional[datetime] = Field(None, description="Project start date")
    end_date: Optional[datetime] = Field(None, description="Project end date")
    status: Optional[str] = Field(None, description="Project status")
    tags: List[str] = Field(default_factory=list, description="Project tags")
    
    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values and v is not None and values['start_date'] is not None:
            if v < values['start_date']:
                raise ValueError('End date cannot be before start date')
        return v
    
    class Config:
        extra = "allow"


class DocumentEntity(BaseModel):
    """Custom entity type representing a document."""
    
    title: str = Field(..., description="Title of the document")
    content: Optional[str] = Field(None, description="Document content")
    author: Optional[str] = Field(None, description="Author of the document")
    publication_date: Optional[datetime] = Field(None, description="Publication date")
    keywords: List[str] = Field(default_factory=list, description="Document keywords")
    
    class Config:
        extra = "allow"


class EventEntity(BaseModel):
    """Custom entity type representing an event."""
    
    name: str = Field(..., description="Name of the event")
    date: Optional[datetime] = Field(None, description="Event date and time")
    location: Optional[str] = Field(None, description="Event location")
    participants: List[str] = Field(default_factory=list, description="Event participants")
    description: Optional[str] = Field(None, description="Event description")
    
    class Config:
        extra = "allow"


# Custom Edge Types
class WorksAtEdge(BaseModel):
    """Custom edge type representing employment relationship."""
    
    role: Optional[str] = Field(None, description="Job role or position")
    start_date: Optional[datetime] = Field(None, description="Start date of employment")
    end_date: Optional[datetime] = Field(None, description="End date of employment")
    department: Optional[str] = Field(None, description="Department or division")
    
    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values and v is not None and values['start_date'] is not None:
            if v < values['start_date']:
                raise ValueError('End date cannot be before start date')
        return v
    
    class Config:
        extra = "allow"


class CollaboratesOnEdge(BaseModel):
    """Custom edge type representing collaboration relationship."""
    
    role: Optional[str] = Field(None, description="Role in collaboration")
    contribution: Optional[str] = Field(None, description="Type of contribution")
    hours_spent: Optional[int] = Field(None, description="Hours spent on collaboration")
    
    @validator('hours_spent')
    def validate_hours_spent(cls, v):
        if v is not None and v < 0:
            raise ValueError('Hours spent cannot be negative')
        return v
    
    class Config:
        extra = "allow"


class AuthoredByEdge(BaseModel):
    """Custom edge type representing authorship relationship."""
    
    contribution_type: Optional[str] = Field(None, description="Type of contribution (e.g., author, editor)")
    publication_date: Optional[datetime] = Field(None, description="Publication date")
    
    class Config:
        extra = "allow"


class ParticipatesInEdge(BaseModel):
    """Custom edge type representing participation relationship."""
    
    role: Optional[str] = Field(None, description="Role in participation")
    attendance_status: Optional[str] = Field(None, description="Attendance status")
    
    class Config:
        extra = "allow"


class RelatedToEdge(BaseModel):
    """Custom edge type representing general relationship."""
    
    relationship_type: Optional[str] = Field(None, description="Type of relationship")
    strength: Optional[float] = Field(None, description="Relationship strength (0.0 to 1.0)")
    
    @validator('strength')
    def validate_strength(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Strength must be between 0.0 and 1.0')
        return v
    
    class Config:
        extra = "allow"


# Type Validation and Serialization Functions
def validate_entity_properties(entity_type: str, properties: Dict[str, Any], type_manager: CustomTypeManager) -> Dict[str, Any]:
    """
    Validate entity properties against the registered entity type.
    
    Args:
        entity_type: Name of the entity type
        properties: Dictionary of entity properties
        type_manager: CustomTypeManager instance
        
    Returns:
        Validated properties dictionary
        
    Raises:
        ValueError: If validation fails
    """
    try:
        validated_entity = type_manager.validate_entity(entity_type, properties)
        return validated_entity.dict()
    except ValueError as e:
        raise ValueError(f"Entity validation failed: {str(e)}")


def validate_edge_properties(edge_type: str, properties: Dict[str, Any], type_manager: CustomTypeManager) -> Dict[str, Any]:
    """
    Validate edge properties against the registered edge type.
    
    Args:
        edge_type: Name of the edge type
        properties: Dictionary of edge properties
        type_manager: CustomTypeManager instance
        
    Returns:
        Validated properties dictionary
        
    Raises:
        ValueError: If validation fails
    """
    try:
        validated_edge = type_manager.validate_edge(edge_type, properties)
        return validated_edge.dict()
    except ValueError as e:
        raise ValueError(f"Edge validation failed: {str(e)}")


def serialize_custom_type(instance: Union[BaseModel, EntityNode, EntityEdge]) -> Dict[str, Any]:
    """
    Serialize a custom type instance to a dictionary suitable for dataset storage.
    
    Args:
        instance: Custom type instance to serialize
        
    Returns:
        Serialized dictionary
    """
    if isinstance(instance, (EntityNode, EntityEdge)):
        # Handle core Graphiti types
        return instance.dict()
    else:
        # Handle custom Pydantic types
        data = instance.dict()
        
        # Convert datetime objects to ISO format strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, list):
                data[key] = [
                    item.isoformat() if isinstance(item, datetime) else item
                    for item in value
                ]
        
        return data


def deserialize_custom_type(data: Dict[str, Any], target_type: Type[BaseModel]) -> BaseModel:
    """
    Deserialize a dictionary to a custom type instance.
    
    Args:
        data: Dictionary to deserialize
        target_type: Target type class
        
    Returns:
        Deserialized instance
    """
    # Convert ISO format strings back to datetime objects
    for key, value in data.items():
        if isinstance(value, str) and key.endswith('_date'):
            try:
                data[key] = datetime.fromisoformat(value)
            except ValueError:
                pass  # Keep as string if not a valid datetime
        elif isinstance(value, list):
            data[key] = [
                datetime.fromisoformat(item) if isinstance(item, str) and item.endswith('T') else item
                for item in value
            ]
    
    return target_type(**data)


def type_converter(
    instance: Union[BaseModel, EntityNode, EntityEdge], 
    target_type: Type[Union[BaseModel, EntityNode, EntityEdge]]
) -> Union[BaseModel, EntityNode, EntityEdge]:
    """
    Convert between different type systems while preserving data.
    
    Args:
        instance: Instance to convert
        target_type: Target type class
        
    Returns:
        Converted instance
    """
    if isinstance(instance, target_type):
        return instance
    
    # Convert to dictionary and then to target type
    data = serialize_custom_type(instance)
    return target_type(**data)


# Global type manager instance
default_type_manager = CustomTypeManager()


def get_type_manager() -> CustomTypeManager:
    """Get the default type manager instance."""
    return default_type_manager


def create_custom_entity(
    entity_type: str,
    name: str,
    type_manager: Optional[CustomTypeManager] = None,
    **kwargs
) -> EntityNode:
    """
    Create a custom entity node with the specified type.
    
    Args:
        entity_type: Name of the entity type
        name: Name of the entity
        type_manager: CustomTypeManager instance (uses default if None)
        **kwargs: Additional properties for the entity
        
    Returns:
        EntityNode instance with custom properties
        
    Raises:
        ValueError: If entity type is not registered
    """
    if type_manager is None:
        type_manager = get_type_manager()
    
    # Validate the entity properties
    validated_properties = validate_entity_properties(entity_type, kwargs, type_manager)
    
    # Create EntityNode with custom properties
    entity_data = {
        'name': name,
        'labels': [entity_type],
        'properties': validated_properties,
        'group_id': kwargs.get('group_id', 'default')
    }
    
    return EntityNode(**entity_data)


def create_custom_edge(
    edge_type: str,
    source_node_uuid: str,
    target_node_uuid: str,
    fact: str,
    type_manager: Optional[CustomTypeManager] = None,
    **kwargs
) -> EntityEdge:
    """
    Create a custom edge with the specified type.
    
    Args:
        edge_type: Name of the edge type
        source_node_uuid: UUID of the source node
        target_node_uuid: UUID of the target node
        fact: Description of the edge
        type_manager: CustomTypeManager instance (uses default if None)
        **kwargs: Additional properties for the edge
        
    Returns:
        EntityEdge instance with custom properties
        
    Raises:
        ValueError: If edge type is not registered
    """
    if type_manager is None:
        type_manager = get_type_manager()
    
    # Validate the edge properties
    validated_properties = validate_edge_properties(edge_type, kwargs, type_manager)
    
    # Create EntityEdge with custom properties
    edge_data = {
        'source_node_uuid': source_node_uuid,
        'target_node_uuid': target_node_uuid,
        'fact': fact,
        'properties': validated_properties,
        'group_id': kwargs.get('group_id', 'default')
    }
    
    return EntityEdge(**edge_data)