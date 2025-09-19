"""
Enhanced Validator for Graphiti-HF

Provides advanced validation capabilities for entities, edges, and the
overall knowledge graph with custom validation rules and monitoring.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
import numpy as np

from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation"""
    enable_entity_validation: bool = True
    enable_edge_validation: bool = True
    enable_graph_validation: bool = True
    enable_temporal_validation: bool = True
    max_workers: int = 4
    batch_size: int = 100
    conflict_resolution: str = "merge"  # "merge", "keep_newer", "keep_older", "keep_better"


class ValidationRule(BaseModel):
    """Custom validation rule definition"""
    name: str
    description: str
    validator: Callable[[Any], bool]
    severity: str = "error"  # "error", "warning", "info"
    auto_fix: Optional[Callable[[Any], Any]] = None
    enabled: bool = True


class ValidationReport:
    """Validation report containing results and statistics"""
    
    def __init__(self):
        self.total_validations = 0
        self.passed_validations = 0
        self.failed_validations = 0
        self.warnings = 0
        self.errors = 0
        self.infos = 0
        self.validation_results: List[Dict[str, Any]] = []
        self.auto_fixes_applied = 0
        self.conflicts_resolved = 0
    
    def add_result(
        self,
        item_type: str,
        item_id: str,
        rule_name: str,
        passed: bool,
        message: str,
        severity: str = "info",
        auto_fix: bool = False
    ):
        """Add a validation result to the report"""
        result = {
            'item_type': item_type,
            'item_id': item_id,
            'rule_name': rule_name,
            'passed': passed,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'auto_fix_applied': auto_fix
        }
        
        self.validation_results.append(result)
        self.total_validations += 1
        
        if passed:
            self.passed_validations += 1
        else:
            self.failed_validations += 1
        
        if severity == "error":
            self.errors += 1
        elif severity == "warning":
            self.warnings += 1
        else:
            self.infos += 1
        
        if auto_fix:
            self.auto_fixes_applied += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary statistics"""
        return {
            'total_validations': self.total_validations,
            'passed_validations': self.passed_validations,
            'failed_validations': self.failed_validations,
            'success_rate': self.passed_validations / self.total_validations if self.total_validations > 0 else 0,
            'warnings': self.warnings,
            'errors': self.errors,
            'infos': self.infos,
            'auto_fixes_applied': self.auto_fixes_applied,
            'conflicts_resolved': self.conflicts_resolved
        }
    
    def get_failures(self) -> List[Dict[str, Any]]:
        """Get list of validation failures"""
        return [result for result in self.validation_results if not result['passed']]
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get list of validation errors"""
        return [result for result in self.validation_results if result['severity'] == 'error']
    
    def get_warnings(self) -> List[Dict[str, Any]]:
        """Get list of validation warnings"""
        return [result for result in self.validation_results if result['severity'] == 'warning']


class Validator:
    """
    Enhanced validator for entities, edges, and knowledge graphs
    
    Handles validation with custom rules, temporal consistency,
    and auto-fix capabilities.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the validator
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        
        # Initialize validation rules
        self.entity_rules: List[ValidationRule] = []
        self.edge_rules: List[ValidationRule] = []
        self.graph_rules: List[ValidationRule] = []
        
        # Load default rules
        self._load_default_rules()
        
        # Validation statistics
        self.stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'auto_fixes_applied': 0,
            'conflicts_resolved': 0
        }
    
    def _load_default_rules(self):
        """Load default validation rules"""
        # Entity validation rules
        self.entity_rules.extend([
            ValidationRule(
                name="Entity Name Validation",
                description="Validate entity names are not empty",
                validator=lambda entity: bool(entity.name and entity.name.strip()),
                severity="error"
            ),
            ValidationRule(
                name="Entity Labels Validation",
                description="Validate entities have at least one label",
                validator=lambda entity: len(entity.labels) > 0,
                severity="error"
            ),
            ValidationRule(
                name="Entity Attribute Validation",
                description="Validate entity attributes are properly formatted",
                validator=lambda entity: self._validate_entity_attributes(entity),
                severity="warning"
            ),
            ValidationRule(
                name="Entity Temporal Validation",
                description="Validate entity temporal consistency",
                validator=lambda entity: self._validate_entity_temporal(entity),
                severity="warning"
            )
        ])
        
        # Edge validation rules
        self.edge_rules.extend([
            ValidationRule(
                name="Edge Source Validation",
                description="Validate edge source nodes exist",
                validator=lambda edge: bool(edge.source_node_uuid),
                severity="error"
            ),
            ValidationRule(
                name="Edge Target Validation",
                description="Validate edge target nodes exist",
                validator=lambda edge: bool(edge.target_node_uuid),
                severity="error"
            ),
            ValidationRule(
                name="Edge Fact Validation",
                description="Validate edge facts are not empty",
                validator=lambda edge: bool(edge.fact and edge.fact.strip()),
                severity="error"
            ),
            ValidationRule(
                name="Edge Temporal Validation",
                description="Validate edge temporal consistency",
                validator=lambda edge: self._validate_edge_temporal(edge),
                severity="warning"
            ),
            ValidationRule(
                name="Edge Structural Validation",
                description="Validate edge structural consistency",
                validator=lambda edge: self._validate_edge_structural(edge),
                severity="warning"
            )
        ])
        
        # Graph validation rules
        self.graph_rules.extend([
            ValidationRule(
                name="Graph Connectivity Validation",
                description="Validate graph connectivity",
                validator=lambda graph: self._validate_graph_connectivity([], [], graph),
                severity="warning"
            ),
            ValidationRule(
                name="Graph Consistency Validation",
                description="Validate graph consistency",
                validator=lambda graph: self._validate_graph_consistency([], [], graph),
                severity="error"
            ),
            ValidationRule(
                name="Graph Temporal Validation",
                description="Validate graph temporal consistency",
                validator=lambda graph: self._validate_graph_temporal([], [], graph),
                severity="warning"
            )
        ])
    
    async def validate_entities(
        self,
        entities: List[EntityNode],
        entity_map: Optional[Dict[str, EntityNode]] = None
    ) -> ValidationReport:
        """
        Validate entities using configured rules
        
        Args:
            entities: List of entities to validate
            entity_map: Optional mapping of entity names to entities
            
        Returns:
            ValidationReport containing results
        """
        report = ValidationReport()
        
        if not self.config.enable_entity_validation or not entities:
            return report
        
        # Validate each entity
        for entity in entities:
            await self._validate_entity(entity, report, entity_map)
        
        # Update statistics
        self.stats['total_validations'] += report.total_validations
        self.stats['passed_validations'] += report.passed_validations
        self.stats['failed_validations'] += report.failed_validations
        self.stats['auto_fixes_applied'] += report.auto_fixes_applied
        
        return report
    
    async def validate_edges(
        self,
        edges: List[EntityEdge],
        entity_map: Optional[Dict[str, EntityNode]] = None
    ) -> ValidationReport:
        """
        Validate edges using configured rules
        
        Args:
            edges: List of edges to validate
            entity_map: Optional mapping of entity names to entities
            
        Returns:
            ValidationReport containing results
        """
        report = ValidationReport()
        
        if not self.config.enable_edge_validation or not edges:
            return report
        
        # Validate each edge
        for edge in edges:
            await self._validate_edge(edge, report, entity_map)
        
        # Update statistics
        self.stats['total_validations'] += report.total_validations
        self.stats['passed_validations'] += report.passed_validations
        self.stats['failed_validations'] += report.failed_validations
        self.stats['auto_fixes_applied'] += report.auto_fixes_applied
        
        return report
    
    async def validate_graph(
        self,
        entities: List[EntityNode],
        edges: List[EntityEdge]
    ) -> ValidationReport:
        """
        Validate the entire knowledge graph
        
        Args:
            entities: List of entities in the graph
            edges: List of edges in the graph
            
        Returns:
            ValidationReport containing results
        """
        report = ValidationReport()
        
        if not self.config.enable_graph_validation:
            return report
        
        # Build entity map for validation
        entity_map = {entity.uuid: entity for entity in entities}
        
        # Validate graph connectivity
        await self._validate_graph_connectivity(entities, edges, report)
        
        # Validate graph consistency
        await self._validate_graph_consistency(entities, edges, report)
        
        # Validate graph temporal consistency
        if self.config.enable_temporal_validation:
            await self._validate_graph_temporal(entities, edges, report)
        
        # Update statistics
        self.stats['total_validations'] += report.total_validations
        self.stats['passed_validations'] += report.passed_validations
        self.stats['failed_validations'] += report.failed_validations
        
        return report
    
    async def validate_incremental(
        self,
        new_entities: List[EntityNode],
        new_edges: List[EntityEdge],
        existing_entities: Optional[List[EntityNode]] = None,
        existing_edges: Optional[List[EntityEdge]] = None
    ) -> Dict[str, Any]:
        """
        Perform incremental validation of new entities and edges
        
        Args:
            new_entities: Newly extracted entities
            new_edges: Newly extracted edges
            existing_entities: Existing entities in the graph
            existing_edges: Existing edges in the graph
            
        Returns:
            Dictionary with validation results for entities and edges
        """
        # Build complete entity map
        all_entities = (existing_entities or []) + new_entities
        entity_map = {entity.uuid: entity for entity in all_entities}
        
        # Validate in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            loop = asyncio.get_event_loop()
            
            entity_task = loop.run_in_executor(
                executor,
                lambda: asyncio.run(self.validate_entities(new_entities, entity_map))
            )
            edge_task = loop.run_in_executor(
                executor,
                lambda: asyncio.run(self.validate_edges(new_edges, entity_map))
            )
            
            entity_report = await entity_task
            edge_report = await edge_task
        
        # Validate graph if we have existing data
        graph_report = ValidationReport()
        if existing_entities and existing_edges:
            graph_report = await self.validate_graph(all_entities, (existing_edges or []) + new_edges)
        
        return {
            'entities': entity_report,
            'edges': edge_report,
            'graph': graph_report,
            'summary': {
                'total_validations': (
                    entity_report.total_validations +
                    edge_report.total_validations +
                    graph_report.total_validations
                ),
                'passed_validations': (
                    entity_report.passed_validations +
                    edge_report.passed_validations +
                    graph_report.passed_validations
                ),
                'failed_validations': (
                    entity_report.failed_validations +
                    edge_report.failed_validations +
                    graph_report.failed_validations
                ),
                'auto_fixes_applied': (
                    entity_report.auto_fixes_applied +
                    edge_report.auto_fixes_applied
                )
            }
        }
    
    def add_entity_rule(self, rule: ValidationRule):
        """Add a custom entity validation rule"""
        self.entity_rules.append(rule)
        logger.info(f"Added entity validation rule: {rule.name}")
    
    def add_edge_rule(self, rule: ValidationRule):
        """Add a custom edge validation rule"""
        self.edge_rules.append(rule)
        logger.info(f"Added edge validation rule: {rule.name}")
    
    def add_graph_rule(self, rule: ValidationRule):
        """Add a custom graph validation rule"""
        self.graph_rules.append(rule)
        logger.info(f"Added graph validation rule: {rule.name}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset validation statistics"""
        self.stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'auto_fixes_applied': 0,
            'conflicts_resolved': 0
        }
    
    # Helper methods
    async def _validate_entity(
        self,
        entity: EntityNode,
        report: ValidationReport,
        entity_map: Optional[Dict[str, EntityNode]] = None
    ):
        """Validate a single entity"""
        for rule in self.entity_rules:
            if not rule.enabled:
                continue
            
            try:
                passed = rule.validator(entity)
                
                if not passed:
                    # Try auto-fix if available
                    auto_fix_applied = False
                    if rule.auto_fix:
                        try:
                            fixed_entity = rule.auto_fix(entity)
                            entity = fixed_entity
                            auto_fix_applied = True
                            report.auto_fixes_applied += 1
                        except Exception as e:
                            logger.warning(f"Auto-fix failed for rule {rule.name}: {e}")
                    
                    report.add_result(
                        item_type="entity",
                        item_id=str(entity.uuid),
                        rule_name=rule.name,
                        passed=False,
                        message=f"{rule.description}: {getattr(entity, 'name', 'Unknown entity')}",
                        severity=rule.severity,
                        auto_fix=auto_fix_applied
                    )
                else:
                    report.add_result(
                        item_type="entity",
                        item_id=str(entity.uuid),
                        rule_name=rule.name,
                        passed=True,
                        message=f"{rule.description}: {getattr(entity, 'name', 'Unknown entity')}",
                        severity=rule.severity
                    )
            except Exception as e:
                logger.error(f"Error validating entity with rule {rule.name}: {e}")
                report.add_result(
                    item_type="entity",
                    item_id=str(entity.uuid),
                    rule_name=rule.name,
                    passed=False,
                    message=f"Validation error: {e}",
                    severity="error"
                )
    
    async def _validate_edge(
        self,
        edge: EntityEdge,
        report: ValidationReport,
        entity_map: Optional[Dict[str, EntityNode]] = None
    ):
        """Validate a single edge"""
        for rule in self.edge_rules:
            if not rule.enabled:
                continue
            
            try:
                passed = rule.validator(edge)
                
                if not passed:
                    # Try auto-fix if available
                    auto_fix_applied = False
                    if rule.auto_fix:
                        try:
                            fixed_edge = rule.auto_fix(edge)
                            edge = fixed_edge
                            auto_fix_applied = True
                            report.auto_fixes_applied += 1
                        except Exception as e:
                            logger.warning(f"Auto-fix failed for rule {rule.name}: {e}")
                    
                    report.add_result(
                        item_type="edge",
                        item_id=str(edge.uuid),
                        rule_name=rule.name,
                        passed=False,
                        message=f"{rule.description}: {edge.fact}",
                        severity=rule.severity,
                        auto_fix=auto_fix_applied
                    )
                else:
                    report.add_result(
                        item_type="edge",
                        item_id=str(edge.uuid),
                        rule_name=rule.name,
                        passed=True,
                        message=f"{rule.description}: {edge.fact}",
                        severity=rule.severity
                    )
            except Exception as e:
                logger.error(f"Error validating edge with rule {rule.name}: {e}")
                report.add_result(
                    item_type="edge",
                    item_id=str(edge.uuid),
                    rule_name=rule.name,
                    passed=False,
                    message=f"Validation error: {e}",
                    severity="error"
                )
    
    async def _validate_graph_connectivity(
        self,
        entities: List[EntityNode],
        edges: List[EntityEdge],
        report: ValidationReport
    ):
        """Validate graph connectivity"""
        # Check for isolated nodes
        connected_nodes = set()
        
        for edge in edges:
            connected_nodes.add(edge.source_node_uuid)
            connected_nodes.add(edge.target_node_uuid)
        
        isolated_entities = [
            entity for entity in entities
            if entity.uuid not in connected_nodes
        ]
        
        if isolated_entities:
            report.add_result(
                item_type="graph",
                item_id="connectivity",
                rule_name="Graph Connectivity Validation",
                passed=False,
                message=f"Found {len(isolated_entities)} isolated entities",
                severity="warning"
            )
        else:
            report.add_result(
                item_type="graph",
                item_id="connectivity",
                rule_name="Graph Connectivity Validation",
                passed=True,
                message="Graph is fully connected",
                severity="info"
            )
    
    async def _validate_graph_consistency(
        self,
        entities: List[EntityNode],
        edges: List[EntityEdge],
        report: ValidationReport
    ):
        """Validate graph consistency"""
        # Check for dangling references
        entity_uuids = {entity.uuid for entity in entities}
        
        dangling_edges = []
        for edge in edges:
            if (edge.source_node_uuid not in entity_uuids or
                edge.target_node_uuid not in entity_uuids):
                dangling_edges.append(edge)
        
        if dangling_edges:
            report.add_result(
                item_type="graph",
                item_id="consistency",
                rule_name="Graph Consistency Validation",
                passed=False,
                message=f"Found {len(dangling_edges)} dangling edges",
                severity="error"
            )
        else:
            report.add_result(
                item_type="graph",
                item_id="consistency",
                rule_name="Graph Consistency Validation",
                passed=True,
                message="Graph is consistent",
                severity="info"
            )
    
    async def _validate_graph_temporal(
        self,
        entities: List[EntityNode],
        edges: List[EntityEdge],
        report: ValidationReport
    ):
        """Validate graph temporal consistency"""
        # Check for temporal inconsistencies
        temporal_issues = []
        
        # Check entity temporal consistency
        for entity in entities:
            if hasattr(entity, 'created_at') and hasattr(entity, 'updated_at'):
                if entity.created_at > entity.updated_at:
                    temporal_issues.append(f"Entity {entity.name} has created_at > updated_at")
        
        # Check edge temporal consistency
        for edge in edges:
            if hasattr(edge, 'created_at') and hasattr(edge, 'updated_at'):
                if edge.created_at > edge.updated_at:
                    temporal_issues.append(f"Edge {edge.fact} has created_at > updated_at")
        
        if temporal_issues:
            report.add_result(
                item_type="graph",
                item_id="temporal",
                rule_name="Graph Temporal Validation",
                passed=False,
                message=f"Found {len(temporal_issues)} temporal issues",
                severity="warning"
            )
        else:
            report.add_result(
                item_type="graph",
                item_id="temporal",
                rule_name="Graph Temporal Validation",
                passed=True,
                message="Graph temporal consistency is valid",
                severity="info"
            )
    
    # Validation helper methods
    def _validate_entity_attributes(self, entity: EntityNode) -> bool:
        """Validate entity attributes"""
        if not entity.attributes:
            return True
        
        for key, value in entity.attributes.items():
            if not isinstance(key, str) or not key.strip():
                return False
            
            if value is None:
                continue
            
            # Basic type validation
            if isinstance(value, str):
                if not value.strip():
                    continue
            elif isinstance(value, (int, float)):
                if value < 0:
                    return False
            elif isinstance(value, list):
                if not all(isinstance(item, str) for item in value):
                    return False
            elif isinstance(value, dict):
                if not all(isinstance(k, str) for k in value.keys()):
                    return False
        
        return True
    
    def _validate_entity_temporal(self, entity: EntityNode) -> bool:
        """Validate entity temporal consistency"""
        if not hasattr(entity, 'created_at'):
            return True
        
        current_time = datetime.now()
        
        if entity.created_at > current_time:
            return False
        
        if hasattr(entity, 'updated_at'):
            if entity.updated_at > current_time:
                return False
            if entity.created_at > entity.updated_at:
                return False
        
        return True
    
    def _validate_edge_temporal(self, edge: EntityEdge) -> bool:
        """Validate edge temporal consistency"""
        if not hasattr(edge, 'created_at'):
            return True
        
        current_time = datetime.now()
        
        if edge.created_at > current_time:
            return False
        
        if hasattr(edge, 'updated_at'):
            if edge.updated_at > current_time:
                return False
            if edge.created_at > edge.updated_at:
                return False
        
        return True
    
    def _validate_edge_structural(self, edge: EntityEdge) -> bool:
        """Validate edge structural consistency"""
        # Check for self-loops
        if edge.source_node_uuid == edge.target_node_uuid:
            return False
        
        # Check for duplicate edges (same source-target pair with same name)
        # This is a simplified check - in production, you'd want more sophisticated logic
        if not edge.name or not edge.name.strip():
            return False
        
        return True
    