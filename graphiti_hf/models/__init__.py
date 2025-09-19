"""
Graphiti-HF Models Package

This package contains custom entity and edge types for Graphiti-HF.
"""

from .custom_types import (
    # Custom Type Manager
    CustomTypeManager,
    
    # Custom Entity Types
    PersonEntity,
    CompanyEntity,
    ProjectEntity,
    DocumentEntity,
    EventEntity,
    
    # Custom Edge Types
    WorksAtEdge,
    CollaboratesOnEdge,
    AuthoredByEdge,
    ParticipatesInEdge,
    RelatedToEdge,
    
    # Type Validation and Serialization
    validate_entity_properties,
    validate_edge_properties,
    serialize_custom_type,
    deserialize_custom_type,
    type_converter,
    
    # Utility Functions
    create_custom_entity,
    create_custom_edge,
    get_type_manager,
)

__all__ = [
    # Custom Type Manager
    'CustomTypeManager',
    
    # Custom Entity Types
    'PersonEntity',
    'CompanyEntity',
    'ProjectEntity',
    'DocumentEntity',
    'EventEntity',
    
    # Custom Edge Types
    'WorksAtEdge',
    'CollaboratesOnEdge',
    'AuthoredByEdge',
    'ParticipatesInEdge',
    'RelatedToEdge',
    
    # Type Validation and Serialization
    'validate_entity_properties',
    'validate_edge_properties',
    'serialize_custom_type',
    'deserialize_custom_type',
    'type_converter',
    
    # Utility Functions
    'create_custom_entity',
    'create_custom_edge',
    'get_type_manager',
]