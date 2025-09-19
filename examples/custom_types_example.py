"""
Example demonstrating custom entity and edge types in Graphiti-HF

This example shows how to:
1. Register custom entity types (Person, Company, Project, Document, Event)
2. Register custom edge types (WorksAt, CollaboratesOn, AuthoredBy, ParticipatesIn, RelatedTo)
3. Create and save custom entities and edges
4. Query and search custom types
5. Validate custom type data
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List

from graphiti_hf.drivers.huggingface_driver import HuggingFaceDriver
from graphiti_hf.models.custom_types import (
    PersonEntity,
    CompanyEntity,
    ProjectEntity,
    DocumentEntity,
    EventEntity,
    WorksAtEdge,
    CollaboratesOnEdge,
    AuthoredByEdge,
    ParticipatesInEdge,
    RelatedToEdge,
)


async def demonstrate_custom_types():
    """Demonstrate custom types functionality"""
    
    # Initialize the driver
    driver = HuggingFaceDriver(
        repo_id="your-username/your-custom-types-repo",
        create_repo=True,
        private=True
    )
    
    print("=== Custom Types Example ===\n")
    
    # 1. Register custom entity types
    print("1. Registering custom entity types...")
    
    # Person entity schema
    person_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "email": {"type": "string", "format": "email"},
            "skills": {"type": "array", "items": {"type": "string"}},
            "experience_years": {"type": "integer", "minimum": 0}
        },
        "required": ["name", "email"]
    }
    
    # Company entity schema
    company_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "industry": {"type": "string"},
            "size": {"type": "string", "enum": ["startup", "small", "medium", "large", "enterprise"]},
            "founded_year": {"type": "integer", "minimum": 1800, "maximum": 2024},
            "revenue": {"type": "number", "minimum": 0}
        },
        "required": ["name", "industry"]
    }
    
    # Project entity schema
    project_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "status": {"type": "string", "enum": ["planning", "active", "completed", "on-hold"]},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "budget": {"type": "number", "minimum": 0},
            "start_date": {"type": "string", "format": "date"},
            "end_date": {"type": "string", "format": "date"}
        },
        "required": ["name", "description"]
    }
    
    # Register entity types
    driver.register_custom_entity_type("Person", person_schema)
    driver.register_custom_entity_type("Company", company_schema)
    driver.register_custom_entity_type("Project", project_schema)
    
    print("✓ Registered entity types: Person, Company, Project")
    
    # 2. Register custom edge types
    print("\n2. Registering custom edge types...")
    
    # WorksAt edge schema
    works_at_schema = {
        "type": "object",
        "properties": {
            "position": {"type": "string"},
            "start_date": {"type": "string", "format": "date"},
            "end_date": {"type": "string", "format": "date"},
            "salary": {"type": "number", "minimum": 0},
            "is_current": {"type": "boolean"}
        },
        "required": ["position"]
    }
    
    # CollaboratesOn edge schema
    collaborates_on_schema = {
        "type": "object",
        "properties": {
            "role": {"type": "string"},
            "contribution": {"type": "string"},
            "start_date": {"type": "string", "format": "date"},
            "end_date": {"type": "string", "format": "date"}
        },
        "required": ["role"]
    }
    
    # AuthoredBy edge schema
    authored_by_schema = {
        "type": "object",
        "properties": {
            "authorship_type": {"type": "string", "enum": ["primary", "secondary", "contributor"]},
            "contribution_percentage": {"type": "number", "minimum": 0, "maximum": 100},
            "publication_date": {"type": "string", "format": "date"}
        },
        "required": ["authorship_type"]
    }
    
    # Register edge types
    driver.register_custom_edge_type("WorksAt", works_at_schema)
    driver.register_custom_edge_type("CollaboratesOn", collaborates_on_schema)
    driver.register_custom_edge_type("AuthoredBy", authored_by_schema)
    
    print("✓ Registered edge types: WorksAt, CollaboratesOn, AuthoredBy")
    
    # 3. Create and save custom entities
    print("\n3. Creating and saving custom entities...")
    
    # Create a Person entity
    person_data = {
        "name": "Alice Johnson",
        "age": 30,
        "email": "alice.johnson@example.com",
        "skills": ["Python", "Machine Learning", "Data Analysis"],
        "experience_years": 8
    }
    
    person_result = await driver.save_custom_entity("Person", person_data)
    print(f"✓ Saved Person: {person_result}")
    
    # Create a Company entity
    company_data = {
        "name": "TechCorp Inc.",
        "industry": "Technology",
        "size": "medium",
        "founded_year": 2015,
        "revenue": 50000000
    }
    
    company_result = await driver.save_custom_entity("Company", company_data)
    print(f"✓ Saved Company: {company_result}")
    
    # Create a Project entity
    project_data = {
        "name": "AI Development Platform",
        "description": "Building an AI-powered development platform for enterprise customers",
        "status": "active",
        "priority": "high",
        "budget": 1000000,
        "start_date": "2024-01-15",
        "end_date": "2024-12-31"
    }
    
    project_result = await driver.save_custom_entity("Project", project_data)
    print(f"✓ Saved Project: {project_result}")
    
    # 4. Create and save custom edges
    print("\n4. Creating and saving custom edges...")
    
    # Get entity UUIDs
    person_uuid = person_result["entity"]["uuid"]
    company_uuid = company_result["entity"]["uuid"]
    project_uuid = project_result["entity"]["uuid"]
    
    # Create a WorksAt edge
    works_at_data = {
        "source_node_uuid": person_uuid,
        "target_node_uuid": company_uuid,
        "position": "Senior Data Scientist",
        "start_date": "2022-03-01",
        "is_current": True,
        "salary": 120000
    }
    
    works_at_result = await driver.save_custom_edge("WorksAt", works_at_data)
    print(f"✓ Saved WorksAt edge: {works_at_result}")
    
    # Create a CollaboratesOn edge
    collaborates_on_data = {
        "source_node_uuid": person_uuid,
        "target_node_uuid": project_uuid,
        "role": "Lead Developer",
        "contribution": "AI model development and deployment",
        "start_date": "2024-01-15"
    }
    
    collaborates_on_result = await driver.save_custom_edge("CollaboratesOn", collaborates_on_data)
    print(f"✓ Saved CollaboratesOn edge: {collaborates_on_result}")
    
    # 5. Query custom entities and edges
    print("\n5. Querying custom entities and edges...")
    
    # Get all Person entities
    persons = await driver.get_custom_entities("Person")
    print(f"✓ Found {len(persons)} Person entities")
    for person in persons:
        print(f"  - {person['name']} ({person['email']})")
    
    # Get all Company entities
    companies = await driver.get_custom_entities("Company")
    print(f"✓ Found {len(companies)} Company entities")
    for company in companies:
        print(f"  - {company['name']} ({company['industry']})")
    
    # Get all WorksAt edges
    works_at_edges = await driver.get_custom_edges("WorksAt")
    print(f"✓ Found {len(works_at_edges)} WorksAt edges")
    for edge in works_at_edges:
        print(f"  - {edge['source_node_name']} works at {edge['target_node_name']} as {edge['position']}")
    
    # 6. Search custom entities
    print("\n6. Searching custom entities...")
    
    # Search for Person entities
    search_results = await driver.search_custom_entities(
        "Person", 
        "Alice Johnson", 
        limit=5
    )
    print(f"✓ Found {len(search_results)} Person entities matching 'Alice Johnson'")
    for result in search_results:
        print(f"  - {result['name']} (similarity: {result['similarity']:.3f})")
    
    # Search for Company entities
    search_results = await driver.search_custom_entities(
        "Company", 
        "TechCorp", 
        limit=5
    )
    print(f"✓ Found {len(search_results)} Company entities matching 'TechCorp'")
    for result in search_results:
        print(f"  - {result['name']} (similarity: {result['similarity']:.3f})")
    
    # 7. Validate custom type data
    print("\n7. Validating custom type data...")
    
    # Valid Person data
    valid_person = {
        "name": "Bob Smith",
        "email": "bob.smith@example.com",
        "age": 35,
        "skills": ["Java", "Spring", "PostgreSQL"],
        "experience_years": 12
    }
    
    is_valid, errors = driver.validate_custom_entity("Person", valid_person)
    print(f"✓ Valid Person data: {is_valid} (errors: {errors})")
    
    # Invalid Person data (missing required field)
    invalid_person = {
        "name": "Invalid Person",
        "age": 25,
        "skills": ["Python"]
        # Missing required 'email' field
    }
    
    is_valid, errors = driver.validate_custom_entity("Person", invalid_person)
    print(f"✗ Invalid Person data: {is_valid} (errors: {errors})")
    
    # 8. Get custom type statistics
    print("\n8. Custom type statistics...")
    
    stats = driver.get_custom_type_statistics()
    print(f"✓ Entity types: {stats['entity_types']['count']}")
    print(f"✓ Edge types: {stats['edge_types']['count']}")
    print(f"✓ Total custom types: {stats['total_custom_types']}")
    
    # 9. Export and import custom types
    print("\n9. Exporting and importing custom types...")
    
    # Export custom types
    exported_data = driver.export_custom_types("json")
    print(f"✓ Exported custom types: {len(exported_data)} characters")
    
    # Import custom types (clear first, then import)
    driver.clear_custom_types()
    print("✓ Cleared custom types")
    
    import_result = driver.import_custom_types(exported_data, "json")
    print(f"✓ Imported custom types: {import_result}")
    
    # 10. Demonstrate built-in custom types
    print("\n10. Using built-in custom types...")
    
    # Create a Person using the built-in type
    alice = PersonEntity(
        name="Alice Johnson",
        email="alice@example.com",
        age=30,
        skills=["Python", "Machine Learning"],
        experience_years=8
    )
    
    alice_result = await driver.save_node(alice)
    print(f"✓ Saved built-in Person: {alice_result['name']}")
    
    # Create a Company using the built-in type
    techcorp = CompanyEntity(
        name="TechCorp Inc.",
        industry="Technology",
        size="medium",
        founded_year=2015,
        revenue=50000000
    )
    
    techcorp_result = await driver.save_node(techcorp)
    print(f"✓ Saved built-in Company: {techcorp_result['name']}")
    
    # Create a WorksAt edge using the built-in type
    works_at_edge = WorksAtEdge(
        source_node_uuid=alice_result['uuid'],
        target_node_uuid=techcorp_result['uuid'],
        position="Senior Data Scientist",
        start_date=datetime(2022, 3, 1),
        is_current=True,
        salary=120000
    )
    
    works_at_edge_result = await driver.save_edge(works_at_edge)
    print(f"✓ Saved built-in WorksAt edge: {works_at_edge_result['position']}")
    
    print("\n=== Custom Types Example Complete ===")


async def main():
    """Main function to run the custom types example"""
    try:
        await demonstrate_custom_types()
    except Exception as e:
        print(f"Error running custom types example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())