# src/services/types_core.py
"""
Core type definitions for the entire system.
This file should have ZERO imports from other services to avoid circular dependencies.
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod

# === CORE ENUMS ===
class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AgentCapability(Enum):
    DATABASE_EXPLORATION = "database_exploration"
    SCHEMA_ANALYSIS = "schema_analysis"
    DOMAIN_RESEARCH = "domain_research"
    LITERATURE_SEARCH = "literature_search"
    SQL_GENERATION = "sql_generation"
    RESULT_VALIDATION = "result_validation"

class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    UNAVAILABLE = "unavailable"

class ProcessingPath(Enum):
    LIGHTNING_RAG = "lightning_rag"
    SEMANTIC_BRIDGE = "semantic_bridge"
    AGENTIC_FALLBACK = "agentic_fallback"
    ERROR_RECOVERY = "error_recovery"

class ComplexityLevel(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class QueryIntent(Enum):
    PROFILE_ANALYSIS = "profile_analysis"
    SCHEMA_EXPLORATION = "schema_exploration"
    DATA_VISUALIZATION = "data_visualization"
    COMPARATIVE_ANALYSIS = "comparative_analysis"

class QueryClassification(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    RESEARCH = "research"
    ANALYTICAL = "analytical"

# === CORE DATA CLASSES ===
@dataclass
class TaskRequest:
    task_id: str
    description: str
    required_capabilities: List[AgentCapability]
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout_seconds: int = 60
    context: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3

@dataclass
class TaskResult:
    task_id: str
    success: bool
    result_data: Any
    execution_time: float
    agent_id: str
    confidence_score: float = 0.8
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resources_used: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentMetrics:
    agent_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_execution_time: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    capabilities: List[AgentCapability] = field(default_factory=list)
    health_score: float = 1.0

@dataclass
class RoutingDecision:
    path: ProcessingPath
    confidence: float
    reasoning: List[str]
    performance_budget: int
    fallback_path: ProcessingPath
    enrichments_needed: List[str]
    unknown_terms: List[str]
    complexity_factors: Dict[str, Any]
    estimated_cost: str

@dataclass
class ResponseIntelligenceConfig:
    response_format: str = "structured"
    target_audience: str = "researcher"
    complexity_level: str = "intermediate"
    include_visualization: bool = True
    include_metadata: bool = True

@dataclass
class ResponseConfig:
    format_type: str = "structured"
    target_audience: str = "researcher"
    complexity_level: str = "intermediate"

# === CORE INTERFACES ===
class BaseAgent(ABC):
    """Base interface for all agents"""
    
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        pass
    
    @abstractmethod
    def process_task(self, task_request: TaskRequest) -> TaskResult:
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        pass

class BaseRouter(ABC):
    """Base interface for query routing"""
    
    @abstractmethod
    def route_query(self, query: str, context: Dict[str, Any] = None) -> RoutingDecision:
        pass
    
    @abstractmethod
    def record_query_result(self, query: str, routing_decision: RoutingDecision, 
                          execution_time: float, success: bool, error_type: str = None):
        pass

class BaseIntelligenceEngine(ABC):
    """Base interface for intelligence engines"""
    
    @abstractmethod
    def classify_query(self, query: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def extract_entities(self, query: str) -> List[Dict[str, Any]]:
        pass

# === CONFIGURATION CLASSES ===
@dataclass
class SystemConfiguration:
    """System-wide configuration"""
    max_concurrent_agents: int = 6
    default_timeout_seconds: int = 300
    circuit_breaker_threshold: int = 5
    memory_threshold_mb: int = 2048
    enable_monitoring: bool = True
    enable_caching: bool = True
    log_level: str = "INFO"

@dataclass
class DatabaseConfiguration:
    """Database configuration"""
    connection_string: str
    pool_size: int = 15
    max_overflow: int = 25
    pool_pre_ping: bool = True
    connect_args: Dict[str, Any] = field(default_factory=dict)

# === ERROR CLASSES ===
class SystemError(Exception):
    """Base system error"""
    pass

class AgentError(SystemError):
    """Agent-specific error"""
    pass

class RoutingError(SystemError):
    """Routing-specific error"""
    pass

class CircuitBreakerError(SystemError):
    """Circuit breaker error"""
    pass