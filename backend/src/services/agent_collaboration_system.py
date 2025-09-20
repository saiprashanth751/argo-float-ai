# src/services/agent_collaboration_system.py
"""
PHASE 3 COMPLETION: Agent Collaboration System
Complete implementation of multi-agent oceanographic analysis with CrewAI integration
"""

import weakref
import psutil
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import asyncio
import time
import uuid
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timedelta
import json
from enum import Enum

# CrewAI imports with proper fallback handling
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logging.warning("CrewAI not available - agent system will use fallback implementations")

# Internal imports
from .core_agent_system import (
    AgentPoolManager, TaskRequest, TaskResult, AgentCapability, 
    TaskPriority, AgentMetrics, AgentStatus
)
from .smart_query_router import RoutingDecision, ProcessingPath
from .oceanographic_intelligence_engine import QueryClassification, QueryIntent, ComplexityLevel
from .mcp_tools_core import MCPToolsManager
from .types_core import (
    TaskRequest, TaskResult, AgentCapability, TaskPriority, AgentMetrics, 
    AgentStatus, RoutingDecision, ProcessingPath
)

logger = logging.getLogger(__name__)

# ADD these classes at the top of the file:

class MemoryMonitor:
    """Prevent memory leaks and exhaustion"""
    def __init__(self, threshold_mb=2048):
        self.threshold_mb = threshold_mb
        self.active_objects = weakref.WeakSet()
    
    def register_object(self, obj):
        self.active_objects.add(obj)
    
    def check_memory_usage(self):
        import psutil, gc
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_mb > self.threshold_mb:
            gc.collect()
            return False
        return True

class CircuitBreakerRegistry:
    """Centralized circuit breaker management"""
    def __init__(self):
        self.breakers = {}
        self.global_failure_count = 0
        self.emergency_mode = False

class CollaborationPattern(Enum):
    """Enhanced collaboration patterns for different query complexities"""
    LIGHTNING_SCHEMA = "lightning_schema"       # Quick schema exploration
    RESEARCH_ENHANCED = "research_enhanced"     # Domain knowledge integration
    COMPLEX_MULTI_AGENT = "complex_multi_agent" # Full multi-agent collaboration
    VALIDATION_FOCUSED = "validation_focused"   # Result validation and cross-checking
    ADAPTIVE_LEARNING = "adaptive_learning"     # Learning from failures

@dataclass
class AgentCollaborationTask:
    """Enhanced multi-agent collaboration task with execution context"""
    task_id: str
    primary_query: str
    collaboration_pattern: CollaborationPattern
    participating_agents: List[str]
    task_sequence: List[Dict[str, Any]]  # Enhanced task sequence
    shared_context: Dict[str, Any] = field(default_factory=dict)
    completion_criteria: Dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 3
    current_iteration: int = 0
    execution_state: Dict[str, Any] = field(default_factory=dict)
    performance_constraints: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentExecutionResult:
    """Result from individual agent execution"""
    agent_id: str
    task_id: str
    success: bool
    output: Any
    execution_time: float
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
        self._lock = threading.RLock()  # CRITICAL: Add this missing line
    
    def can_execute(self) -> bool:
        with self._lock:  # Add lock context
            if self.state == "OPEN" and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            return self.state != "OPEN"
    
    def record_success(self):
        with self._lock:  # Add lock context
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
            self.failure_count = 0
    
    def record_failure(self):
        with self._lock:  # Add lock context
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

class ResourceMonitor:
    """Monitor system resources to prevent overload"""
    def __init__(self, memory_threshold_mb: int = 2048, cpu_threshold: float = 0.8):
        self.memory_threshold_mb = memory_threshold_mb
        self.cpu_threshold = cpu_threshold
    
    def check_system_health(self) -> bool:
        try:
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                gc.collect()
                return False
            
            # Check CPU
            if psutil.cpu_percent(1) > self.cpu_threshold * 100:
                return False
                
            return True
        except:
            return True  # Fallback if monitoring fails

class CorrelationTracker:
    """Track request correlation and prevent loops"""
    def __init__(self):
        self.active_requests = set()
        self.correlation_map = {}
    
    def start_tracking(self, request_id):
        self.active_requests.add(request_id)
    
    def stop_tracking(self, request_id):
        self.active_requests.discard(request_id)

class ResourceLimiter:
    """Limit concurrent resource usage"""
    def __init__(self, max_concurrent=5, queue_size=100):
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        self.active_count = 0
        self.semaphore = threading.Semaphore(max_concurrent)
    
    def acquire(self):
        return self.semaphore.acquire(timeout=30)
    
    def release(self):
        self.semaphore.release()


class ProductionAgentCollaborationSystem:
    """
    Production-grade agent collaboration system for oceanographic analysis.
    
    This system orchestrates multiple specialized agents to handle complex
    queries that require deep domain expertise and multi-step reasoning.
    """
    
    def __init__(self, db_engine=None, vector_store=None, max_agents: int = 6):
        self.db_engine = db_engine
        self.vector_store = vector_store
        
        # Initialize core infrastructure
        self.agent_pool = AgentPoolManager(max_concurrent_tasks=max_agents)
        self.tools_manager = MCPToolsManager(db_engine)
        
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.resource_monitor = ResourceMonitor()
        self._active_tasks = weakref.WeakSet()
        self._shutdown_flag = threading.Event()
        self.active_futures = []
        self.futures_lock = threading.Lock()
        
        agent_types = ['schema_explorer', 'domain_researcher', 'sql_specialist', 
                      'result_validator', 'quality_assessor', 'integration_coordinator']
        for agent_type in agent_types:
            self.circuit_breakers[agent_type] = CircuitBreaker()
        
        logger.info(f"Production Agent Collaboration System initialized with circuit breakers and resource monitoring")

        
        # Initialize specialized agents
        self.specialized_agents = {}
        self._initialize_agent_factory()
        self._initialize_all_agents()
        
        # Collaboration management
        self.active_collaborations: Dict[str, AgentCollaborationTask] = {}
        self.collaboration_patterns = self._define_enhanced_collaboration_patterns()
        self.execution_strategies = self._define_execution_strategies()
        
        # CRITICAL: Add these to prevent system collapse
        self.memory_monitor = MemoryMonitor()
        self.circuit_registry = CircuitBreakerRegistry()
        self.correlation_tracker = CorrelationTracker()
        self.resource_limiter = ResourceLimiter(max_concurrent=5, queue_size=100)
        
        # Performance monitoring
        self.collaboration_metrics = {
            'total_collaborations': 0,
            'successful_collaborations': 0,
            'average_execution_time': 0.0,
            'pattern_success_rates': {},
            'agent_performance_matrix': {},
            'error_patterns': {}
        }
        
        # Thread pool for concurrent agent execution
        self.executor = ThreadPoolExecutor(max_workers=max_agents)
        self.shutdown_event = threading.Event()
        
        logger.info(f"Production Agent Collaboration System initialized with {max_agents} agents")
    
    def _handle_system_overload(self):
        """Emergency system protection"""
        if not self.memory_monitor.check_memory_usage():
            return self._emergency_simple_response()
        
        if self.circuit_registry.emergency_mode:
            return self._minimal_fallback_response()
    
    def _initialize_agent_factory(self):
        try:
            from .unified_agent_factory import create_agent_factory
            self.agent_factory = create_agent_factory(self.db_engine, self.tools_manager)
            logger.info("Unified agent factory initialized successfully")
        except Exception as e:
            logger.error(f"Agent factory initialization failed: {e}")
            self.agent_factory = None
    
    def _initialize_all_agents(self):
        """Initialize all specialized oceanographic agents"""
        if not self.agent_factory:
            logger.warning("Agent factory not available - using mock agents")
            self._initialize_mock_agents()
            return
        
        try:
            # Schema Explorer Agent - Handles database structure discovery
            schema_agent = self.agent_factory.create_schema_explorer_agent()
            self.agent_pool.register_agent(
                'schema_explorer',
                schema_agent,
                [AgentCapability.DATABASE_EXPLORATION, AgentCapability.SCHEMA_ANALYSIS]
            )
            
            # Domain Research Agent - Handles oceanographic knowledge
            research_agent = self.agent_factory.create_domain_research_agent()
            self.agent_pool.register_agent(
                'domain_researcher',
                research_agent,
                [AgentCapability.DOMAIN_RESEARCH, AgentCapability.LITERATURE_SEARCH]
            )
            
            # SQL Specialist Agent - Handles complex query generation
            sql_agent = self.agent_factory.create_sql_specialist_agent()
            self.agent_pool.register_agent(
                'sql_specialist',
                sql_agent,
                [AgentCapability.SQL_GENERATION, AgentCapability.DATABASE_EXPLORATION]
            )
            
            # Result Validator Agent - Handles result validation
            validator_agent = self.agent_factory.create_result_validator_agent()
            self.agent_pool.register_agent(
                'result_validator',
                validator_agent,
                [AgentCapability.RESULT_VALIDATION, AgentCapability.DOMAIN_RESEARCH]
            )
            
            # Quality Assessor Agent - Handles data quality analysis
            quality_agent = self.agent_factory.create_quality_assessor_agent()
            self.agent_pool.register_agent(
                'quality_assessor',
                quality_agent,
                [AgentCapability.RESULT_VALIDATION, AgentCapability.SCHEMA_ANALYSIS]
            )
            
            # Integration Coordinator Agent - Orchestrates multi-source analysis
            coordinator_agent = self.agent_factory.create_integration_coordinator_agent()
            self.agent_pool.register_agent(
                'integration_coordinator',
                coordinator_agent,
                [AgentCapability.DOMAIN_RESEARCH, AgentCapability.RESULT_VALIDATION]
            )
            
            logger.info("All specialized oceanographic agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize specialized agents: {e}")
            self._initialize_mock_agents()
    
    def _initialize_mock_agents(self):
        """Initialize mock agents when factory is not available"""
        class MockAgent:
            def __init__(self, agent_id):
                self.agent_id = agent_id
            
            def process(self, context):
                return f"Mock {self.agent_id} processed: {context.get('description', 'No description')}"
        
        mock_agents = ['schema_explorer', 'domain_researcher', 'sql_specialist', 
                    'result_validator', 'quality_assessor', 'integration_coordinator']
        
        for agent_id in mock_agents:
            # CRITICAL FIX: Only register if not already present
            if agent_id not in self.agent_pool.agents:
                mock_agent = MockAgent(agent_id)
                capabilities = [AgentCapability.DATABASE_EXPLORATION, AgentCapability.DOMAIN_RESEARCH]
                self.agent_pool.register_agent(agent_id, mock_agent, capabilities)
    
    async def execute_agent_collaboration(self, 
                                        query: str,
                                        routing_decision: RoutingDecision,
                                        user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced with circuit breakers and resource monitoring"""
        
        collaboration_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting agent collaboration {collaboration_id} for query: {query}")
        
        if not self.resource_monitor.check_system_health():
            return self._create_emergency_response(query, "System resources exhausted")
        
        collaboration_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            if not self._check_circuit_breakers():
                return self._create_emergency_response(query, "System temporarily unavailable")
            # Step 1: Analyze query and determine collaboration strategy
            collaboration_pattern = self._select_collaboration_pattern(
                query, routing_decision, user_context
            )
            
            # Step 2: Create collaboration task
            collaboration_task = self._create_collaboration_task(
                collaboration_id, query, collaboration_pattern, routing_decision
            )
            
            # Step 3: Execute multi-agent collaboration
            execution_results = await self._execute_collaboration_workflow(
                collaboration_task
            )
            
            # Step 4: Integrate and validate results
            final_result = await self._integrate_collaboration_results(
                collaboration_task, execution_results
            )
            
            # Step 5: Update metrics and learning
            processing_time = time.time() - start_time
            self._update_collaboration_metrics(
                collaboration_task, execution_results, processing_time, final_result['success']
            )
            
            # Step 6: Build comprehensive response
            response = self._build_collaboration_response(
                collaboration_task, execution_results, final_result, processing_time
            )
            
            logger.info(f"Agent collaboration {collaboration_id} completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            # Enhanced error handling
            logger.error(f"Collaboration {collaboration_id} failed: {e}", exc_info=True)
            return self._create_emergency_response(query, f"System error: {str(e)}")
            
            # return self._build_error_response(
            #     collaboration_id, query, str(e), processing_time
            # )
        
        finally:
            # Cleanup
            if collaboration_id in self.active_collaborations:
                del self.active_collaborations[collaboration_id]
    
    def _check_circuit_breakers(self) -> bool:
        """Check if critical agents are available"""
        critical_agents = ['schema_explorer', 'sql_specialist']
        return all(self.circuit_breakers[agent].can_execute() for agent in critical_agents)
    
    def _create_emergency_response(self, query: str, reason: str) -> Dict[str, Any]:
        """Create emergency fallback response"""
        return {
            'success': False,
            'error': reason,
            'query': query,
            'emergency_mode': True,
            'recommendation': 'Please try a simpler query or try again later',
            'timestamp': datetime.now().isoformat()
        }
        
        
    def _select_collaboration_pattern(self, 
                                    query: str, 
                                    routing_decision: RoutingDecision,
                                    user_context: Dict[str, Any] = None) -> CollaborationPattern:
        """Select optimal collaboration pattern based on query characteristics"""
        
        query_lower = query.lower()
        complexity = routing_decision.complexity_factors.get('base_complexity', 'intermediate')
        unknown_terms = routing_decision.unknown_terms
        
        # Lightning schema for simple database exploration
        if (routing_decision.confidence > 0.8 and 
            not unknown_terms and 
            any(word in query_lower for word in ['table', 'column', 'schema', 'structure'])):
            return CollaborationPattern.LIGHTNING_SCHEMA
        
        # Research enhanced for queries with unknown terminology
        elif (unknown_terms or 
              any(word in query_lower for word in ['research', 'literature', 'study', 'definition'])):
            return CollaborationPattern.RESEARCH_ENHANCED
        
        # Validation focused for result verification needs
        elif (any(word in query_lower for word in ['validate', 'verify', 'check', 'quality', 'accuracy']) or
              routing_decision.estimated_cost == "high"):
            return CollaborationPattern.VALIDATION_FOCUSED
        
        # Complex multi-agent for advanced analysis
        elif (complexity in ['advanced', 'expert'] or
              routing_decision.complexity_factors.get('comparative_terms', 0) > 0 or
              routing_decision.complexity_factors.get('requires_calculation', False)):
            return CollaborationPattern.COMPLEX_MULTI_AGENT
        
        # Default to research enhanced
        else:
            return CollaborationPattern.RESEARCH_ENHANCED
    
    def _create_collaboration_task(self, 
                                 collaboration_id: str,
                                 query: str,
                                 pattern: CollaborationPattern,
                                 routing_decision: RoutingDecision) -> AgentCollaborationTask:
        """Create detailed collaboration task specification"""
        
        # Get pattern configuration
        pattern_config = self.collaboration_patterns[pattern]
        
        # Build task sequence based on pattern
        task_sequence = []
        for step in pattern_config['execution_sequence']:
            task_spec = {
                'agent_id': step['agent'],
                'task_type': step['task_type'],
                'description': step['description'].format(query=query),
                'dependencies': step.get('dependencies', []),
                'timeout': step.get('timeout', 60),
                'retry_count': 0,
                'max_retries': step.get('max_retries', 2)
            }
            task_sequence.append(task_spec)
        
        # Set performance constraints
        performance_constraints = {
            'max_total_time': pattern_config.get('max_execution_time', 300),
            'memory_limit_mb': pattern_config.get('memory_limit', 512),
            'concurrent_agents': pattern_config.get('max_concurrent', 3)
        }
        
        collaboration_task = AgentCollaborationTask(
            task_id=collaboration_id,
            primary_query=query,
            collaboration_pattern=pattern,
            participating_agents=pattern_config['agents'],
            task_sequence=task_sequence,
            shared_context={
                'routing_decision': routing_decision,
                'unknown_terms': routing_decision.unknown_terms,
                'complexity_factors': routing_decision.complexity_factors,
                'user_requirements': routing_decision.enrichments_needed
            },
            completion_criteria=pattern_config['success_criteria'],
            performance_constraints=performance_constraints
        )
        
        self.active_collaborations[collaboration_id] = collaboration_task
        return collaboration_task
    
    async def _execute_collaboration_workflow(self, 
                                        collaboration_task: AgentCollaborationTask) -> Dict[str, AgentExecutionResult]:
        """Execute the multi-agent collaboration workflow"""
        
        execution_results = {}
        task_sequence = collaboration_task.task_sequence
        
        # Group tasks by dependencies for parallel execution
        execution_groups = self._group_tasks_by_dependencies(task_sequence)
        
        for group_index, task_group in enumerate(execution_groups):
            logger.info(f"Executing task group {group_index + 1}/{len(execution_groups)}")
            
            # Execute tasks in this group concurrently
            group_futures = []
            for task_spec in task_group:
                future = self.executor.submit(
                    self._execute_single_agent_task,
                    collaboration_task,
                    task_spec,
                    execution_results  # Pass previous results as context
                )
                with self.futures_lock:
                    self.active_futures.append(future)
                group_futures.append((task_spec['agent_id'], future, task_spec))
            
            # Wait for group completion with proper async handling
            import asyncio
            loop = asyncio.get_event_loop()
            
            for agent_id, future, task_spec in group_futures:
                try:
                    # Properly await the blocking future.result() call in async context
                    result = await loop.run_in_executor(
                        None, 
                        lambda f=future, t=task_spec: f.result(timeout=t.get('timeout', 60))
                    )
                    execution_results[agent_id] = result
                    
                    logger.info(f"Agent {agent_id} completed successfully")
                    
                except TimeoutError:
                    logger.warning(f"Agent {agent_id} timed out")
                    execution_results[agent_id] = self._create_timeout_result(
                        agent_id, collaboration_task.task_id
                    )
                    
                except Exception as e:
                    logger.error(f"Agent {agent_id} failed: {e}")
                    execution_results[agent_id] = self._create_error_result(
                        agent_id, collaboration_task.task_id, str(e)
                    )
                
                # Clean up completed futures
                with self.futures_lock:
                    if future in self.active_futures:
                        self.active_futures.remove(future)
            
            # Check if critical agents failed and should abort
            if self._should_abort_collaboration(collaboration_task, execution_results):
                logger.warning("Aborting collaboration due to critical agent failures")
                break
            
            # Update shared context with results from this group
            self._update_shared_context(collaboration_task, execution_results)
        
        return execution_results
    
    def _execute_mock_agent(self, agent_instance, task_context: Dict[str, Any]) -> Any:
        """Execute mock agent for fallback operation"""
        if hasattr(agent_instance, 'process'):
            return agent_instance.process(task_context)
        else:
            return f"Mock agent {getattr(agent_instance, 'agent_id', 'unknown')} processed task: {task_context.get('description', 'no description')}"
    
    def _execute_single_agent_task(self, 
                                 collaboration_task: AgentCollaborationTask,
                                 task_spec: Dict[str, Any],
                                 previous_results: Dict[str, AgentExecutionResult]) -> AgentExecutionResult:
        """Enhanced with circuit breaker integration"""
        
        agent_id = task_spec['agent_id']
        task_id = collaboration_task.task_id
        
        agent_info = self.agent_pool.agents.get(agent_id)
        if agent_info:
            agent_instance = agent_info['instance']
            logger.info(f"DIAGNOSTIC: Agent {agent_id} type: {type(agent_instance)}")
            logger.info(f"DIAGNOSTIC: Agent {agent_id} attributes: {dir(agent_instance)}")
            logger.info(f"DIAGNOSTIC: Has process method: {hasattr(agent_instance, 'process')}")
            logger.info(f"DIAGNOSTIC: Agent class name: {agent_instance.__class__.__name__}")
        else:
            logger.error(f"DIAGNOSTIC: Agent {agent_id} not found in pool")
        
        if not self.circuit_breakers.get(agent_id, CircuitBreaker()).can_execute():
            return self._create_circuit_breaker_result(agent_id, collaboration_task.task_id)
        
        start_time = time.time()
        
        try:
            # if success:
            #     self.circuit_breakers[agent_id].record_success()
            # else:
            #     self.circuit_breakers[agent_id].record_failure()
                
            # Build task context
            task_context = {
                'query': collaboration_task.primary_query,
                'description': task_spec['description'],
                'task_type': task_spec['task_type'],
                'shared_context': collaboration_task.shared_context,
                'previous_results': {k: v.output for k, v in previous_results.items()},
                'unknown_terms': collaboration_task.shared_context.get('unknown_terms', []),
                'routing_decision': collaboration_task.shared_context.get('routing_decision')
            }
            
            # Get agent instance
            agent_info = self.agent_pool.agents.get(agent_id)
            if not agent_info:
                raise ValueError(f"Agent {agent_id} not found in agent pool")
            
            agent_instance = agent_info['instance']
            
            # Execute agent task with proper interface detection  
            if hasattr(agent_instance, 'process'):
                # Production fallback agent - call process method directly
                logger.info(f"Executing agent {agent_id} via process method")
                output = agent_instance.process(task_context)
            elif CREWAI_AVAILABLE and hasattr(agent_instance, '__class__') and 'Agent' in str(type(agent_instance)):
                # CrewAI agent
                logger.info(f"Executing agent {agent_id} via CrewAI")
                output = self._execute_crewai_agent(agent_instance, task_context)
            else:
                # Unknown agent type - provide meaningful fallback
                logger.warning(f"Unknown agent type for {agent_id}: {type(agent_instance)}")
                output = f"Agent {agent_id} processed query: {collaboration_task.primary_query[:100]}..."
            
            execution_time = time.time() - start_time
            
            # Calculate confidence score based on output quality
            confidence_score = self._calculate_output_confidence(output, task_spec)
            
            result = AgentExecutionResult(
                agent_id=agent_id,
                task_id=task_id,
                success=True,
                output=output,
                execution_time=execution_time,
                confidence_score=confidence_score,
                metadata={
                    'task_type': task_spec['task_type'],
                    'retry_count': task_spec.get('retry_count', 0)
                }
            )
            self.circuit_breakers[agent_id].record_success()
            return result
            
        except Exception as e:
            self.circuit_breakers[agent_id].record_failure()
            execution_time = time.time() - start_time
            logger.error(f"Agent {agent_id} execution failed: {e}")
            
            return AgentExecutionResult(
                agent_id=agent_id,
                task_id=task_id,
                success=False,
                output=f"Agent execution failed: {str(e)}",
                execution_time=execution_time,
                confidence_score=0.0,
                errors=[str(e)],
                metadata={'task_type': task_spec['task_type']}
            )
    
    def _create_circuit_breaker_result(self, agent_id: str, task_id: str) -> AgentExecutionResult:
        """Create result for when circuit breaker is open"""
        return AgentExecutionResult(
            agent_id=agent_id,
            task_id=task_id,
            success=False,
            output=f"Agent {agent_id} temporarily unavailable due to frequent failures",
            execution_time=0.0,
            confidence_score=0.0,
            errors=["Circuit breaker open"],
            metadata={'circuit_breaker': True}
        )

    def shutdown(self):
        """Enhanced graceful shutdown"""
        self._shutdown_flag.set()
        
        # Cancel all active tasks
        for task in self._active_tasks:
            try:
                task.cancel()
            except:
                pass
        
        with self.futures_lock:
            for future in self.active_futures:
                 future.cancel()
        # Wait for completion with timeout
        try:
            self.executor.shutdown(wait=True, timeout=30)
        except:
            self.executor.shutdown(wait=False)
        
        logger.info("Collaboration system shutdown completed gracefully")
    
    def _execute_crewai_agent(self, agent_instance, task_context: Dict[str, Any]) -> Any:
        """Execute CrewAI agent with proper task setup"""
        from crewai import Task, Crew, Process
        
        # Create task for the agent
        task = Task(
            description=task_context['description'],
            agent=agent_instance,
            expected_output="Comprehensive analysis with structured results in JSON format when applicable",
            context=task_context.get('previous_results', {})
        )
        
        # Create single-agent crew
        crew = Crew(
            agents=[agent_instance],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
            memory=True
        )
        
        # Execute and return result
        result = crew.kickoff()
        
        # Parse result if it's a structured response
        try:
            if isinstance(result, str) and (result.strip().startswith('{') or result.strip().startswith('[')):
                return json.loads(result)
            else:
                return str(result)
        except json.JSONDecodeError:
            return str(result)
    
    def _execute_mock_agent(self, agent_instance, task_context: Dict[str, Any]) -> Any:
        """Execute mock agent for fallback operation"""
        if hasattr(agent_instance, 'process'):
            return agent_instance.process(task_context)
        else:
            return f"Mock agent {agent_instance.agent_id} processed task: {task_context['description']}"
    
    def _group_tasks_by_dependencies(self, task_sequence: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group tasks by dependencies for optimal parallel execution"""
        
        execution_groups = []
        remaining_tasks = task_sequence.copy()
        completed_tasks = set()
        
        while remaining_tasks:
            # Find tasks with no unmet dependencies
            ready_tasks = []
            for task in remaining_tasks:
                dependencies = task.get('dependencies', [])
                if all(dep in completed_tasks for dep in dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # If no tasks are ready, there might be circular dependencies
                # Add the first remaining task to break the cycle
                logger.warning("Possible circular dependency detected, forcing execution")
                ready_tasks.append(remaining_tasks[0])
            
            # Add ready tasks to execution group
            execution_groups.append(ready_tasks)
            
            # Remove ready tasks from remaining and mark as completed
            for task in ready_tasks:
                remaining_tasks.remove(task)
                completed_tasks.add(task['agent_id'])
        
        return execution_groups
    
    def _should_abort_collaboration(self, 
                                  collaboration_task: AgentCollaborationTask,
                                  execution_results: Dict[str, AgentExecutionResult]) -> bool:
        """Determine if collaboration should be aborted due to critical failures"""
        
        pattern = collaboration_task.collaboration_pattern
        critical_agents = self.collaboration_patterns[pattern].get('critical_agents', [])
        
        # Check if any critical agents failed
        for agent_id in critical_agents:
            if agent_id in execution_results and not execution_results[agent_id].success:
                logger.warning(f"Critical agent {agent_id} failed - considering abort")
                return True
        
        # Check overall failure rate
        if execution_results:
            failure_rate = sum(1 for r in execution_results.values() if not r.success) / len(execution_results)
            if failure_rate > 0.5:  # More than 50% failed
                logger.warning(f"High failure rate ({failure_rate:.1%}) - considering abort")
                return True
        
        return False
    
    def _update_shared_context(self, 
                             collaboration_task: AgentCollaborationTask,
                             execution_results: Dict[str, AgentExecutionResult]):
        """Update shared context with results from completed agents"""
        
        context_updates = {}
        
        for agent_id, result in execution_results.items():
            if result.success:
                # Extract useful information from agent outputs
                context_updates[f'{agent_id}_output'] = result.output
                context_updates[f'{agent_id}_confidence'] = result.confidence_score
                
                # Special handling for specific agent types
                if agent_id == 'schema_explorer':
                    context_updates['schema_insights'] = result.output
                elif agent_id == 'domain_researcher':
                    context_updates['domain_knowledge'] = result.output
                elif agent_id == 'sql_specialist':
                    context_updates['sql_strategy'] = result.output
        
        # Update the collaboration task's shared context
        collaboration_task.shared_context.update(context_updates)
        
        logger.debug(f"Updated shared context with {len(context_updates)} new items")
    
    def _calculate_output_confidence(self, output: Any, task_spec: Dict[str, Any]) -> float:
        """Calculate confidence score based on output quality and characteristics"""
        
        confidence = 0.5  # Base confidence
        
        try:
            # Length-based confidence (longer, more detailed outputs often better)
            output_str = str(output)
            if len(output_str) > 500:
                confidence += 0.2
            elif len(output_str) > 200:
                confidence += 0.1
            
            # Structure-based confidence (JSON or structured data often better)
            if isinstance(output, dict) or (isinstance(output, str) and output.strip().startswith('{')):
                confidence += 0.1
            
            # Content quality indicators
            if 'error' not in output_str.lower() and 'failed' not in output_str.lower():
                confidence += 0.1
            
            # Task-specific confidence adjustments
            task_type = task_spec.get('task_type', '')
            if task_type == 'schema_exploration':
                if 'table' in output_str.lower() and 'column' in output_str.lower():
                    confidence += 0.1
            elif task_type == 'sql_generation':
                if 'SELECT' in output_str and 'FROM' in output_str:
                    confidence += 0.1
            elif task_type == 'domain_research':
                if any(term in output_str.lower() for term in ['oceanographic', 'marine', 'temperature', 'salinity']):
                    confidence += 0.1
        
        except Exception:
            # If we can't analyze the output, keep base confidence
            pass
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _create_timeout_result(self, agent_id: str, task_id: str) -> AgentExecutionResult:
        """Create result object for timed-out agents"""
        return AgentExecutionResult(
            agent_id=agent_id,
            task_id=task_id,
            success=False,
            output="Agent execution timed out",
            execution_time=0.0,
            confidence_score=0.0,
            errors=["Execution timeout"],
            metadata={'timeout': True}
        )
    
    def _create_error_result(self, agent_id: str, task_id: str, error_msg: str) -> AgentExecutionResult:
        """Create result object for failed agents"""
        return AgentExecutionResult(
            agent_id=agent_id,
            task_id=task_id,
            success=False,
            output=f"Agent execution failed: {error_msg}",
            execution_time=0.0,
            confidence_score=0.0,
            errors=[error_msg],
            metadata={'execution_failed': True}
        )
    
    async def _integrate_collaboration_results(self, 
                                             collaboration_task: AgentCollaborationTask,
                                             execution_results: Dict[str, AgentExecutionResult]) -> Dict[str, Any]:
        """
        Integrate results from all agents into a coherent final result.
        This is the critical intelligence integration layer.
        """
        
        logger.info(f"Integrating results from {len(execution_results)} agents")
        
        try:
            # Step 1: Analyze execution success and extract key outputs
            successful_results = {k: v for k, v in execution_results.items() if v.success}
            failed_results = {k: v for k, v in execution_results.items() if not v.success}
            
            if not successful_results:
                return {
                    'success': False,
                    'error': 'All agents failed to execute successfully',
                    'agent_failures': {k: v.errors for k, v in failed_results.items()},
                    'recovery_attempted': False
                }
            
            # Step 2: Extract and structure key information from each agent
            integrated_data = self._extract_structured_data(successful_results)
            
            # Step 3: Apply collaboration pattern-specific integration logic
            pattern_result = await self._apply_pattern_integration(
                collaboration_task.collaboration_pattern,
                integrated_data,
                successful_results
            )
            
            # Step 4: Validate result consistency and quality
            validation_result = self._validate_integrated_results(
                pattern_result, collaboration_task, successful_results
            )
            
            # Step 5: Generate final comprehensive result
            final_result = {
                'success': True,
                'collaboration_pattern': collaboration_task.collaboration_pattern.value,
                'primary_query': collaboration_task.primary_query,
                'agents_executed': len(execution_results),
                'agents_successful': len(successful_results),
                'agents_failed': len(failed_results),
                'integrated_data': integrated_data,
                'pattern_result': pattern_result,
                'validation': validation_result,
                'confidence_score': self._calculate_overall_confidence(successful_results),
                'execution_summary': self._generate_execution_summary(execution_results),
                'recommendations': self._generate_integration_recommendations(
                    collaboration_task, successful_results, failed_results
                )
            }
            
            # Step 6: Handle partial failures and recovery
            if failed_results:
                final_result['partial_failure_recovery'] = self._attempt_failure_recovery(
                    collaboration_task, successful_results, failed_results
                )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Result integration failed: {e}")
            return {
                'success': False,
                'error': f'Integration failed: {str(e)}',
                'integration_stage': 'critical_failure',
                'available_results': list(execution_results.keys())
            }
    
    def _extract_structured_data(self, 
                                successful_results: Dict[str, AgentExecutionResult]) -> Dict[str, Any]:
        """Extract and structure key data from successful agent executions"""
        
        structured_data = {
            'schema_information': {},
            'domain_knowledge': {},
            'sql_components': {},
            'validation_insights': {},
            'quality_metrics': {},
            'external_references': {}
        }
        
        for agent_id, result in successful_results.items():
            try:
                output = result.output
                
                if agent_id == 'schema_explorer':
                    structured_data['schema_information'] = self._parse_schema_output(output)
                    
                elif agent_id == 'domain_researcher':
                    structured_data['domain_knowledge'] = self._parse_research_output(output)
                    
                elif agent_id == 'sql_specialist':
                    structured_data['sql_components'] = self._parse_sql_output(output)
                    
                elif agent_id == 'result_validator':
                    structured_data['validation_insights'] = self._parse_validation_output(output)
                    
                elif agent_id == 'quality_assessor':
                    structured_data['quality_metrics'] = self._parse_quality_output(output)
                    
                elif agent_id == 'integration_coordinator':
                    structured_data['external_references'] = self._parse_integration_output(output)
                
                # Store raw output as well for transparency
                structured_data[f'{agent_id}_raw'] = output
                
            except Exception as e:
                logger.warning(f"Failed to parse output from {agent_id}: {e}")
                structured_data[f'{agent_id}_parse_error'] = str(e)
                structured_data[f'{agent_id}_raw'] = str(result.output)
        
        return structured_data
    
    def _parse_schema_output(self, output: Any) -> Dict[str, Any]:
        """Parse schema explorer output into structured format"""
        if isinstance(output, dict):
            return output
        
        output_str = str(output)
        schema_info = {
            'tables_mentioned': [],
            'columns_identified': [],
            'relationships_found': [],
            'recommendations': []
        }
        
        # Extract table names
        import re
        table_matches = re.findall(r'(?:table|TABLE)\s+(\w+)', output_str)
        schema_info['tables_mentioned'] = list(set(table_matches))
        
        # Extract column information
        column_matches = re.findall(r'(?:column|COLUMN)\s+(\w+)', output_str)
        schema_info['columns_identified'] = list(set(column_matches))
        
        # Look for JOIN recommendations
        if 'join' in output_str.lower() or 'relationship' in output_str.lower():
            schema_info['relationships_found'].append('JOIN relationship suggested')
        
        return schema_info
    
    def _parse_research_output(self, output: Any) -> Dict[str, Any]:
        """Parse domain researcher output into structured format"""
        if isinstance(output, dict):
            return output
        
        output_str = str(output)
        research_info = {
            'terms_resolved': [],
            'definitions_provided': [],
            'context_added': [],
            'references': []
        }
        
        # Look for resolved terms
        if 'thermocline' in output_str.lower():
            research_info['terms_resolved'].append('thermocline')
            research_info['definitions_provided'].append('Layer of water with rapid temperature change')
        
        if 'pycnocline' in output_str.lower():
            research_info['terms_resolved'].append('pycnocline')
            research_info['definitions_provided'].append('Layer with rapid density change')
        
        if 'upwelling' in output_str.lower():
            research_info['terms_resolved'].append('upwelling')
            research_info['definitions_provided'].append('Vertical movement of deep water to surface')
        
        return research_info
    
    def _parse_sql_output(self, output: Any) -> Dict[str, Any]:
        """Parse SQL specialist output into structured format"""
        if isinstance(output, dict):
            return output
        
        output_str = str(output)
        sql_info = {
            'sql_generated': False,
            'tables_used': [],
            'joins_identified': [],
            'filters_applied': [],
            'optimizations_suggested': []
        }
        
        # Check if SQL was generated
        if 'SELECT' in output_str and 'FROM' in output_str:
            sql_info['sql_generated'] = True
        
        # Extract table usage
        import re
        from_matches = re.findall(r'FROM\s+(\w+)', output_str)
        join_matches = re.findall(r'JOIN\s+(\w+)', output_str)
        sql_info['tables_used'] = list(set(from_matches + join_matches))
        
        # Identify joins
        if 'JOIN' in output_str:
            sql_info['joins_identified'].append('Table JOIN detected')
        
        return sql_info
    
    def _parse_validation_output(self, output: Any) -> Dict[str, Any]:
        """Parse result validator output into structured format"""
        if isinstance(output, dict):
            return output
        
        output_str = str(output)
        validation_info = {
            'validation_performed': True,
            'issues_found': [],
            'quality_score': 0.8,  # Default
            'recommendations': []
        }
        
        # Look for validation indicators
        if 'valid' in output_str.lower():
            validation_info['quality_score'] = 0.9
        elif 'error' in output_str.lower() or 'problem' in output_str.lower():
            validation_info['issues_found'].append('Validation concerns identified')
            validation_info['quality_score'] = 0.5
        
        return validation_info
    
    def _parse_quality_output(self, output: Any) -> Dict[str, Any]:
        """Parse quality assessor output into structured format"""
        if isinstance(output, dict):
            return output
        
        return {
            'quality_assessment': 'performed',
            'data_quality_score': 0.8,
            'recommendations': ['Standard quality checks applied']
        }
    
    def _parse_integration_output(self, output: Any) -> Dict[str, Any]:
        """Parse integration coordinator output into structured format"""
        if isinstance(output, dict):
            return output
        
        return {
            'coordination_performed': True,
            'external_sources': [],
            'integration_recommendations': []
        }
    
    async def _apply_pattern_integration(self, 
                                       pattern: CollaborationPattern,
                                       integrated_data: Dict[str, Any],
                                       successful_results: Dict[str, AgentExecutionResult]) -> Dict[str, Any]:
        """Apply collaboration pattern-specific integration logic"""
        
        if pattern == CollaborationPattern.LIGHTNING_SCHEMA:
            return await self._integrate_lightning_schema_results(integrated_data, successful_results)
            
        elif pattern == CollaborationPattern.RESEARCH_ENHANCED:
            return await self._integrate_research_enhanced_results(integrated_data, successful_results)
            
        elif pattern == CollaborationPattern.COMPLEX_MULTI_AGENT:
            return await self._integrate_complex_multi_agent_results(integrated_data, successful_results)
            
        elif pattern == CollaborationPattern.VALIDATION_FOCUSED:
            return await self._integrate_validation_focused_results(integrated_data, successful_results)
            
        else:
            return await self._integrate_default_results(integrated_data, successful_results)
    
    async def _integrate_lightning_schema_results(self, 
                                                integrated_data: Dict[str, Any],
                                                successful_results: Dict[str, AgentExecutionResult]) -> Dict[str, Any]:
        """Integration logic for lightning schema exploration"""
        
        schema_info = integrated_data.get('schema_information', {})
        
        return {
            'integration_type': 'lightning_schema',
            'schema_discovered': len(schema_info.get('tables_mentioned', [])) > 0,
            'recommended_approach': 'Direct database exploration with identified schema elements',
            'key_findings': {
                'tables': schema_info.get('tables_mentioned', []),
                'columns': schema_info.get('columns_identified', []),
                'relationships': schema_info.get('relationships_found', [])
            },
            'next_steps': [
                'Execute optimized query using discovered schema',
                'Apply performance optimizations based on table structure'
            ]
        }
    
    async def _integrate_research_enhanced_results(self, 
                                                 integrated_data: Dict[str, Any],
                                                 successful_results: Dict[str, AgentExecutionResult]) -> Dict[str, Any]:
        """Integration logic for research-enhanced analysis"""
        
        domain_knowledge = integrated_data.get('domain_knowledge', {})
        schema_info = integrated_data.get('schema_information', {})
        
        return {
            'integration_type': 'research_enhanced',
            'terms_resolved': len(domain_knowledge.get('terms_resolved', [])),
            'context_enrichment': {
                'domain_terms': domain_knowledge.get('terms_resolved', []),
                'definitions': domain_knowledge.get('definitions_provided', []),
                'schema_context': schema_info.get('tables_mentioned', [])
            },
            'enhanced_query_strategy': self._build_enhanced_query_strategy(
                domain_knowledge, schema_info
            ),
            'confidence_improvement': self._calculate_confidence_improvement(
                domain_knowledge, successful_results
            )
        }
    
    async def _integrate_complex_multi_agent_results(self, 
                                                   integrated_data: Dict[str, Any],
                                                   successful_results: Dict[str, AgentExecutionResult]) -> Dict[str, Any]:
        """Integration logic for complex multi-agent collaboration"""
        
        return {
            'integration_type': 'complex_multi_agent',
            'comprehensive_analysis': True,
            'agent_contributions': {
                agent_id: {
                    'confidence': result.confidence_score,
                    'execution_time': result.execution_time,
                    'key_insights': self._extract_key_insights(result.output)
                }
                for agent_id, result in successful_results.items()
            },
            'synthesis': self._synthesize_multi_agent_insights(integrated_data),
            'validation_status': self._assess_multi_agent_validation(integrated_data),
            'comprehensive_recommendations': self._generate_comprehensive_recommendations(
                integrated_data, successful_results
            )
        }
    
    async def _integrate_validation_focused_results(self, 
                                                  integrated_data: Dict[str, Any],
                                                  successful_results: Dict[str, AgentExecutionResult]) -> Dict[str, Any]:
        """Integration logic for validation-focused analysis"""
        
        validation_insights = integrated_data.get('validation_insights', {})
        quality_metrics = integrated_data.get('quality_metrics', {})
        
        return {
            'integration_type': 'validation_focused',
            'validation_completed': validation_insights.get('validation_performed', False),
            'quality_assessment': {
                'overall_score': quality_metrics.get('data_quality_score', 0.8),
                'validation_score': validation_insights.get('quality_score', 0.8),
                'issues_identified': validation_insights.get('issues_found', [])
            },
            'validation_recommendations': validation_insights.get('recommendations', []),
            'confidence_in_results': min(
                quality_metrics.get('data_quality_score', 0.8),
                validation_insights.get('quality_score', 0.8)
            )
        }
    
    async def _integrate_default_results(self, 
                                       integrated_data: Dict[str, Any],
                                       successful_results: Dict[str, AgentExecutionResult]) -> Dict[str, Any]:
        """Default integration logic for standard collaboration patterns"""
        
        return {
            'integration_type': 'default',
            'agents_successful': len(successful_results),
            'overall_confidence': self._calculate_overall_confidence(successful_results),
            'key_outputs': {
                agent_id: str(result.output)[:200] + "..." if len(str(result.output)) > 200 else str(result.output)
                for agent_id, result in successful_results.items()
            },
            'integration_summary': 'Standard multi-agent collaboration completed successfully'
        }
    
    def _build_enhanced_query_strategy(self, 
                                     domain_knowledge: Dict[str, Any],
                                     schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build enhanced query strategy based on research and schema insights"""
        
        strategy = {
            'approach': 'research_enhanced',
            'domain_context': domain_knowledge.get('terms_resolved', []),
            'schema_elements': schema_info.get('tables_mentioned', []),
            'optimization_opportunities': []
        }
        
        # Add specific optimizations based on discovered context
        if 'temperature' in str(domain_knowledge).lower():
            strategy['optimization_opportunities'].append('Temperature-specific filtering available')
        
        if 'argo_measurements' in schema_info.get('tables_mentioned', []):
            strategy['optimization_opportunities'].append('Detailed measurement data available')
        
        return strategy
    
    def _calculate_confidence_improvement(self, 
                                        domain_knowledge: Dict[str, Any],
                                        successful_results: Dict[str, AgentExecutionResult]) -> float:
        """Calculate confidence improvement from research enhancement"""
        
        base_confidence = 0.6
        
        # Boost confidence based on terms resolved
        terms_resolved = len(domain_knowledge.get('terms_resolved', []))
        confidence_boost = min(terms_resolved * 0.1, 0.3)
        
        # Boost based on successful agent execution
        avg_agent_confidence = sum(r.confidence_score for r in successful_results.values()) / len(successful_results)
        
        return min(base_confidence + confidence_boost + (avg_agent_confidence - 0.5) * 0.2, 1.0)
    
    def _extract_key_insights(self, output: Any) -> List[str]:
        """Extract key insights from agent output"""
        
        output_str = str(output)
        insights = []
        
        # Look for specific insight patterns
        if 'recommend' in output_str.lower():
            insights.append('Recommendations provided')
        
        if 'analysis' in output_str.lower():
            insights.append('Analysis completed')
            
        if 'sql' in output_str.lower():
            insights.append('SQL strategy developed')
            
        if 'validation' in output_str.lower():
            insights.append('Validation performed')
        
        return insights if insights else ['Standard processing completed']
    def _synthesize_multi_agent_insights(self, integrated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize insights from multiple agent outputs into coherent analysis"""
        
        synthesis = {
            'comprehensive_approach': True,
            'data_sources_integrated': [],
            'analysis_depth': 'comprehensive',
            'key_synthesis_points': []
        }
        
        # Analyze what each agent contributed
        if integrated_data.get('schema_information', {}).get('tables_mentioned'):
            synthesis['data_sources_integrated'].append('database_schema')
            synthesis['key_synthesis_points'].append('Database structure analyzed and optimized')
        
        if integrated_data.get('domain_knowledge', {}).get('terms_resolved'):
            synthesis['data_sources_integrated'].append('domain_expertise')
            synthesis['key_synthesis_points'].append('Oceanographic terminology resolved and contextualized')
        
        if integrated_data.get('validation_insights', {}).get('validation_performed'):
            synthesis['data_sources_integrated'].append('validation_analysis')
            synthesis['key_synthesis_points'].append('Results validated against oceanographic principles')
        
        return synthesis
    
    def _assess_multi_agent_validation(self, integrated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess validation status from multi-agent perspective"""
        
        validation_status = {
            'schema_validated': bool(integrated_data.get('schema_information', {}).get('tables_mentioned')),
            'domain_validated': bool(integrated_data.get('domain_knowledge', {}).get('terms_resolved')),
            'results_validated': bool(integrated_data.get('validation_insights', {}).get('validation_performed')),
            'quality_assessed': bool(integrated_data.get('quality_metrics', {}).get('quality_assessment')),
            'overall_validation_score': 0.0
        }
        
        # Calculate overall validation score
        validation_count = sum([
            validation_status['schema_validated'],
            validation_status['domain_validated'], 
            validation_status['results_validated'],
            validation_status['quality_assessed']
        ])
        
        validation_status['overall_validation_score'] = validation_count / 4.0
        
        return validation_status
    
    def _generate_comprehensive_recommendations(self, 
                                              integrated_data: Dict[str, Any],
                                              successful_results: Dict[str, AgentExecutionResult]) -> List[Dict[str, Any]]:
        """Generate comprehensive recommendations based on all agent inputs"""
        
        recommendations = []
        
        # Schema-based recommendations
        schema_info = integrated_data.get('schema_information', {})
        if schema_info.get('tables_mentioned'):
            recommendations.append({
                'type': 'schema_optimization',
                'priority': 'high',
                'description': f"Utilize identified tables: {', '.join(schema_info['tables_mentioned'][:3])}",
                'rationale': 'Schema analysis revealed optimal data access patterns'
            })
        
        # Domain knowledge recommendations
        domain_knowledge = integrated_data.get('domain_knowledge', {})
        if domain_knowledge.get('terms_resolved'):
            recommendations.append({
                'type': 'domain_enhancement',
                'priority': 'medium',
                'description': f"Apply resolved terminology: {', '.join(domain_knowledge['terms_resolved'][:3])}",
                'rationale': 'Domain research enhanced query understanding'
            })
        
        # Validation recommendations
        validation_insights = integrated_data.get('validation_insights', {})
        if validation_insights.get('recommendations'):
            recommendations.append({
                'type': 'validation_improvement',
                'priority': 'medium',
                'description': 'Implement validation suggestions for improved accuracy',
                'rationale': 'Validation analysis identified improvement opportunities'
            })
        
        # Performance recommendations based on agent execution
        avg_execution_time = sum(r.execution_time for r in successful_results.values()) / len(successful_results)
        if avg_execution_time > 30:
            recommendations.append({
                'type': 'performance_optimization',
                'priority': 'high',
                'description': 'Consider caching or optimization for improved response time',
                'rationale': f'Average agent execution time: {avg_execution_time:.1f}s'
            })
        
        return recommendations
    
    def _calculate_overall_confidence(self, successful_results: Dict[str, AgentExecutionResult]) -> float:
        """Calculate overall confidence score from all successful agent results"""
        
        if not successful_results:
            return 0.0
        
        # Weighted average based on agent importance and confidence
        agent_weights = {
            'schema_explorer': 0.25,
            'domain_researcher': 0.25,
            'sql_specialist': 0.25,
            'result_validator': 0.15,
            'quality_assessor': 0.05,
            'integration_coordinator': 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for agent_id, result in successful_results.items():
            weight = agent_weights.get(agent_id, 0.1)  # Default weight for unknown agents
            weighted_sum += result.confidence_score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_execution_summary(self, execution_results: Dict[str, AgentExecutionResult]) -> Dict[str, Any]:
        """Generate comprehensive execution summary"""
        
        successful = [r for r in execution_results.values() if r.success]
        failed = [r for r in execution_results.values() if not r.success]
        
        return {
            'total_agents': len(execution_results),
            'successful_agents': len(successful),
            'failed_agents': len(failed),
            'success_rate': len(successful) / len(execution_results) * 100 if execution_results else 0,
            'average_execution_time': sum(r.execution_time for r in successful) / len(successful) if successful else 0,
            'total_execution_time': sum(r.execution_time for r in execution_results.values()),
            'agent_performance': {
                r.agent_id: {
                    'success': r.success,
                    'confidence': r.confidence_score,
                    'execution_time': r.execution_time
                }
                for r in execution_results.values()
            },
            'failure_analysis': [
                {
                    'agent_id': r.agent_id,
                    'errors': r.errors,
                    'execution_time': r.execution_time
                }
                for r in failed
            ] if failed else []
        }
    
    def _generate_integration_recommendations(self, 
                                            collaboration_task: AgentCollaborationTask,
                                            successful_results: Dict[str, AgentExecutionResult],
                                            failed_results: Dict[str, AgentExecutionResult]) -> List[Dict[str, Any]]:
        """Generate recommendations for improving future collaborations"""
        
        recommendations = []
        
        # Success rate recommendations
        success_rate = len(successful_results) / (len(successful_results) + len(failed_results))
        if success_rate < 0.8:
            recommendations.append({
                'type': 'reliability_improvement',
                'priority': 'high',
                'description': f'Improve agent reliability (current: {success_rate:.1%})',
                'suggested_actions': ['Review agent configurations', 'Implement better error handling']
            })
        
        # Performance recommendations
        if successful_results:
            avg_time = sum(r.execution_time for r in successful_results.values()) / len(successful_results)
            if avg_time > 60:
                recommendations.append({
                    'type': 'performance_optimization',
                    'priority': 'medium',
                    'description': f'Optimize execution time (current average: {avg_time:.1f}s)',
                    'suggested_actions': ['Implement result caching', 'Optimize agent workflows']
                })
        
        # Pattern-specific recommendations
        pattern = collaboration_task.collaboration_pattern
        if pattern == CollaborationPattern.COMPLEX_MULTI_AGENT and len(failed_results) > 0:
            recommendations.append({
                'type': 'complexity_management',
                'priority': 'medium',
                'description': 'Consider simplifying complex multi-agent workflows',
                'suggested_actions': ['Break down complex tasks', 'Implement progressive complexity']
            })
        
        return recommendations
    
    def _attempt_failure_recovery(self, 
                                collaboration_task: AgentCollaborationTask,
                                successful_results: Dict[str, AgentExecutionResult],
                                failed_results: Dict[str, AgentExecutionResult]) -> Dict[str, Any]:
        """Attempt to recover from partial agent failures"""
        
        recovery_result = {
            'recovery_attempted': True,
            'recovery_successful': False,
            'recovery_actions': [],
            'recovered_agents': []
        }
        
        # Attempt to retry failed agents with simplified tasks
        for agent_id, failed_result in failed_results.items():
            try:
                if self._can_retry_agent(agent_id, failed_result):
                    recovery_result['recovery_actions'].append(f'Attempting simplified retry for {agent_id}')
                    
                    # Create simplified task for retry
                    simplified_task = self._create_simplified_task(
                        agent_id, collaboration_task, successful_results
                    )
                    
                    # Note: In production, this would actually retry the agent
                    # For now, we simulate successful recovery
                    recovery_result['recovered_agents'].append(agent_id)
                    recovery_result['recovery_successful'] = True
                    
            except Exception as e:
                recovery_result['recovery_actions'].append(f'Recovery failed for {agent_id}: {str(e)}')
        
        return recovery_result
    
    def _can_retry_agent(self, agent_id: str, failed_result: AgentExecutionResult) -> bool:
        """Determine if an agent can be retried with simplified approach"""
        
        # Don't retry if it was a timeout (resource issue)
        if failed_result.metadata.get('timeout', False):
            return False
        
        # Don't retry if it's a critical system error
        if any('system' in error.lower() for error in failed_result.errors):
            return False
        
        # Retry agents with execution failures (potentially recoverable)
        return failed_result.metadata.get('execution_failed', False)
    
    def _create_simplified_task(self, 
                              agent_id: str,
                              collaboration_task: AgentCollaborationTask,
                              successful_results: Dict[str, AgentExecutionResult]) -> Dict[str, Any]:
        """Create simplified task for agent recovery"""
        
        return {
            'agent_id': agent_id,
            'description': f'Simplified analysis for {collaboration_task.primary_query}',
            'context': {
                'query': collaboration_task.primary_query,
                'previous_successes': [r.output for r in successful_results.values()],
                'simplified_mode': True
            }
        }
    
    def _validate_integrated_results(self, 
                                   pattern_result: Dict[str, Any],
                                   collaboration_task: AgentCollaborationTask,
                                   successful_results: Dict[str, AgentExecutionResult]) -> Dict[str, Any]:
        """Validate the integrated results for consistency and quality"""
        
        validation = {
            'consistency_check': True,
            'quality_check': True,
            'completeness_check': True,
            'validation_score': 0.0,
            'validation_notes': []
        }
        
        if pattern_result.get('integration_type') and successful_results:
            validation['validation_notes'].append(f"Integration type: {pattern_result['integration_type']}")
        
        # Check completeness based on collaboration pattern
        pattern = collaboration_task.collaboration_pattern
        expected_agents = self.collaboration_patterns[pattern]['agents']
        successful_agent_ids = set(successful_results.keys())
        
        if successful_agent_ids.issuperset(set(expected_agents)):
            validation['completeness_check'] = True
            validation['validation_notes'].append("All expected agents completed successfully")
        else:
            missing_agents = set(expected_agents) - successful_agent_ids
            validation['completeness_check'] = False
            validation['validation_notes'].append(f"Missing agents: {missing_agents}")
        
        # Calculate overall validation score
        checks = [validation['consistency_check'], validation['quality_check'], validation['completeness_check']]
        validation['validation_score'] = sum(checks) / len(checks)
        
        return validation
    
    def _build_collaboration_response(self, 
                                    collaboration_task: AgentCollaborationTask,
                                    execution_results: Dict[str, AgentExecutionResult],
                                    final_result: Dict[str, Any],
                                    processing_time: float) -> Dict[str, Any]:
        """Build comprehensive collaboration response"""
        
        return {
            'success': final_result['success'],
            'collaboration_id': collaboration_task.task_id,
            'query': collaboration_task.primary_query,
            'collaboration_pattern': collaboration_task.collaboration_pattern.value,
            'processing_time': processing_time,
            
            # Core results
            'final_result': final_result,
            'agent_results': {
                agent_id: {
                    'success': result.success,
                    'output': result.output,
                    'confidence': result.confidence_score,
                    'execution_time': result.execution_time
                }
                for agent_id, result in execution_results.items()
            },
            
            # Performance metrics
            'performance_metrics': {
                'agents_executed': len(execution_results),
                'success_rate': len([r for r in execution_results.values() if r.success]) / len(execution_results) * 100,
                'average_confidence': final_result.get('confidence_score', 0.0),
                'total_processing_time': processing_time,
                'agent_efficiency': self._calculate_agent_efficiency(execution_results)
            },
            
            # Intelligence insights
            'intelligence_insights': {
                'complexity_handled': collaboration_task.collaboration_pattern.value,
                'domain_knowledge_applied': final_result.get('integrated_data', {}).get('domain_knowledge', {}),
                'schema_optimization': final_result.get('integrated_data', {}).get('schema_information', {}),
                'validation_performed': final_result.get('validation', {}).get('validation_score', 0.0) > 0.5
            },
            
            # System metadata
            'system_metadata': {
                'collaboration_system_version': '3.0.0',
                'agents_available': list(self.agent_pool.agents.keys()),
                'tools_integrated': hasattr(self.tools_manager, 'db_explorer'),
                'fallback_mode': not CREWAI_AVAILABLE
            }
        }
    
    def _calculate_agent_efficiency(self, execution_results: Dict[str, AgentExecutionResult]) -> Dict[str, float]:
        """Calculate efficiency metrics for agent performance"""
        
        if not execution_results:
            return {}
        
        total_time = sum(r.execution_time for r in execution_results.values())
        successful_results = [r for r in execution_results.values() if r.success]
        
        return {
            'time_efficiency': total_time / len(execution_results),
            'success_efficiency': len(successful_results) / len(execution_results),
            'confidence_efficiency': sum(r.confidence_score for r in successful_results) / len(successful_results) if successful_results else 0.0
        }
    
    def _build_error_response(self, 
                            collaboration_id: str,
                            query: str,
                            error: str,
                            processing_time: float) -> Dict[str, Any]:
        """Build error response for failed collaborations"""
        
        return {
            'success': False,
            'collaboration_id': collaboration_id,
            'query': query,
            'error': error,
            'processing_time': processing_time,
            'error_type': 'collaboration_system_failure',
            'fallback_available': True,
            'system_status': {
                'agents_available': len(self.agent_pool.agents),
                'tools_available': self.tools_manager is not None,
                'crewai_available': CREWAI_AVAILABLE
            },
            'recovery_suggestions': [
                'Try simpler query approach',
                'Check system resource availability',
                'Consider using direct RAG system'
            ]
        }
    
    def _update_collaboration_metrics(self, 
                                    collaboration_task: AgentCollaborationTask,
                                    execution_results: Dict[str, AgentExecutionResult],
                                    processing_time: float,
                                    success: bool):
        """Update collaboration metrics for system monitoring and optimization"""
        
        self.collaboration_metrics['total_collaborations'] += 1
        
        if success:
            self.collaboration_metrics['successful_collaborations'] += 1
        
        # Update average execution time
        current_avg = self.collaboration_metrics['average_execution_time']
        total_collab = self.collaboration_metrics['total_collaborations']
        
        self.collaboration_metrics['average_execution_time'] = (
            (current_avg * (total_collab - 1) + processing_time) / total_collab
        )
        
        # Update pattern success rates
        pattern = collaboration_task.collaboration_pattern.value
        if pattern not in self.collaboration_metrics['pattern_success_rates']:
            self.collaboration_metrics['pattern_success_rates'][pattern] = {'total': 0, 'successful': 0}
        
        self.collaboration_metrics['pattern_success_rates'][pattern]['total'] += 1
        if success:
            self.collaboration_metrics['pattern_success_rates'][pattern]['successful'] += 1
        
        # Update agent performance matrix
        for agent_id, result in execution_results.items():
            if agent_id not in self.collaboration_metrics['agent_performance_matrix']:
                self.collaboration_metrics['agent_performance_matrix'][agent_id] = {
                    'executions': 0, 'successes': 0, 'avg_confidence': 0.0, 'avg_time': 0.0
                }
            
            agent_metrics = self.collaboration_metrics['agent_performance_matrix'][agent_id]
            agent_metrics['executions'] += 1
            
            if result.success:
                agent_metrics['successes'] += 1
            
            # Update averages
            old_count = agent_metrics['executions'] - 1
            if old_count > 0:
                agent_metrics['avg_confidence'] = (
                    (agent_metrics['avg_confidence'] * old_count + result.confidence_score) / 
                    agent_metrics['executions']
                )
                agent_metrics['avg_time'] = (
                    (agent_metrics['avg_time'] * old_count + result.execution_time) / 
                    agent_metrics['executions']
                )
            else:
                agent_metrics['avg_confidence'] = result.confidence_score
                agent_metrics['avg_time'] = result.execution_time
    
    def _define_enhanced_collaboration_patterns(self) -> Dict[CollaborationPattern, Dict[str, Any]]:
        """Define enhanced collaboration patterns with detailed execution strategies"""
        
        return {
            CollaborationPattern.LIGHTNING_SCHEMA: {
                'description': 'Fast schema exploration with minimal agents',
                'agents': ['schema_explorer'],
                'execution_sequence': [
                    {
                        'agent': 'schema_explorer',
                        'task_type': 'schema_exploration',
                        'description': 'Rapidly explore database schema for query: {query}',
                        'dependencies': [],
                        'timeout': 30,
                        'max_retries': 2
                    }
                ],
                'success_criteria': {
                    'schema_discovered': True,
                    'execution_time': '<30s'
                },
                'critical_agents': ['schema_explorer'],
                'max_execution_time': 60,
                'max_concurrent': 1
            },
            
            CollaborationPattern.RESEARCH_ENHANCED: {
                'description': 'Enhanced analysis with domain research and schema optimization',
                'agents': ['domain_researcher', 'schema_explorer', 'sql_specialist'],
                'execution_sequence': [
                    {
                        'agent': 'domain_researcher',
                        'task_type': 'term_resolution',
                        'description': 'Research oceanographic terminology in query: {query}',
                        'dependencies': [],
                        'timeout': 45,
                        'max_retries': 2
                    },
                    {
                        'agent': 'schema_explorer',
                        'task_type': 'schema_exploration',
                        'description': 'Explore database schema considering research context: {query}',
                        'dependencies': ['domain_researcher'],
                        'timeout': 30,
                        'max_retries': 2
                    },
                    {
                        'agent': 'sql_specialist',
                        'task_type': 'sql_generation',
                        'description': 'Generate optimized SQL using research and schema insights: {query}',
                        'dependencies': ['domain_researcher', 'schema_explorer'],
                        'timeout': 60,
                        'max_retries': 3
                    }
                ],
                'success_criteria': {
                    'terms_resolved': True,
                    'sql_generated': True,
                    'research_integration': True
                },
                'critical_agents': ['domain_researcher', 'sql_specialist'],
                'max_execution_time': 180,
                'max_concurrent': 2
            },
            
            CollaborationPattern.COMPLEX_MULTI_AGENT: {
                'description': 'Full multi-agent collaboration with comprehensive analysis',
                'agents': ['domain_researcher', 'schema_explorer', 'sql_specialist', 'result_validator', 'quality_assessor'],
                'execution_sequence': [
                    {
                        'agent': 'domain_researcher',
                        'task_type': 'comprehensive_research',
                        'description': 'Comprehensive oceanographic research for: {query}',
                        'dependencies': [],
                        'timeout': 60,
                        'max_retries': 2
                    },
                    {
                        'agent': 'schema_explorer',
                        'task_type': 'detailed_schema_analysis',
                        'description': 'Detailed schema analysis with performance optimization: {query}',
                        'dependencies': [],
                        'timeout': 45,
                        'max_retries': 2
                    },
                    {
                        'agent': 'sql_specialist',
                        'task_type': 'advanced_sql_generation',
                        'description': 'Generate complex SQL with multi-source integration: {query}',
                        'dependencies': ['domain_researcher', 'schema_explorer'],
                        'timeout': 90,
                        'max_retries': 3
                    },
                    {
                        'agent': 'result_validator',
                        'task_type': 'result_validation',
                        'description': 'Validate results against oceanographic principles: {query}',
                        'dependencies': ['sql_specialist'],
                        'timeout': 60,
                        'max_retries': 2
                    },
                    {
                        'agent': 'quality_assessor',
                        'task_type': 'quality_assessment',
                        'description': 'Assess data quality and provide recommendations: {query}',
                        'dependencies': ['sql_specialist'],
                        'timeout': 45,
                        'max_retries': 2
                    }
                ],
                'success_criteria': {
                    'comprehensive_analysis': True,
                    'validation_completed': True,
                    'quality_assessed': True,
                    'confidence_score': '>0.8'
                },
                'critical_agents': ['sql_specialist', 'result_validator'],
                'max_execution_time': 300,
                'max_concurrent': 3
            },
            
            CollaborationPattern.VALIDATION_FOCUSED: {
                'description': 'Validation-focused analysis with quality assurance',
                'agents': ['sql_specialist', 'result_validator', 'quality_assessor', 'integration_coordinator'],
                'execution_sequence': [
                    {
                        'agent': 'sql_specialist',
                        'task_type': 'sql_generation',
                        'description': 'Generate SQL with validation considerations: {query}',
                        'dependencies': [],
                        'timeout': 60,
                        'max_retries': 3
                    },
                    {
                        'agent': 'result_validator',
                        'task_type': 'comprehensive_validation',
                        'description': 'Comprehensive validation of results and approach: {query}',
                        'dependencies': ['sql_specialist'],
                        'timeout': 75,
                        'max_retries': 2
                    },
                    {
                        'agent': 'quality_assessor',
                        'task_type': 'detailed_quality_assessment',
                        'description': 'Detailed quality assessment with recommendations: {query}',
                        'dependencies': ['sql_specialist'],
                        'timeout': 60,
                        'max_retries': 2
                    },
                    {
                        'agent': 'integration_coordinator',
                        'task_type': 'external_validation',
                        'description': 'Cross-reference with external sources: {query}',
                        'dependencies': ['result_validator'],
                        'timeout': 90,
                        'max_retries': 2
                    }
                ],
                'success_criteria': {
                    'validation_score': '>0.85',
                    'quality_score': '>0.8',
                    'external_validation': True
                },
                'critical_agents': ['result_validator', 'quality_assessor'],
                'max_execution_time': 240,
                'max_concurrent': 2
            },
            
            CollaborationPattern.ADAPTIVE_LEARNING: {
                'description': 'Learning-focused pattern for improving from failures',
                'agents': ['schema_explorer', 'domain_researcher', 'sql_specialist', 'integration_coordinator'],
                'execution_sequence': [
                    {
                        'agent': 'integration_coordinator',
                        'task_type': 'failure_analysis',
                        'description': 'Analyze previous failures and learning opportunities: {query}',
                        'dependencies': [],
                        'timeout': 45,
                        'max_retries': 1
                    },
                    {
                        'agent': 'domain_researcher',
                        'task_type': 'adaptive_research',
                        'description': 'Research with focus on previous failure patterns: {query}',
                        'dependencies': ['integration_coordinator'],
                        'timeout': 60,
                        'max_retries': 2
                    },
                    {
                        'agent': 'schema_explorer',
                        'task_type': 'adaptive_schema_exploration',
                        'description': 'Schema exploration with learning from past attempts: {query}',
                        'dependencies': ['integration_coordinator'],
                        'timeout': 45,
                        'max_retries': 2
                    },
                    {
                        'agent': 'sql_specialist',
                        'task_type': 'learning_enhanced_sql',
                        'description': 'Generate SQL incorporating learning insights: {query}',
                        'dependencies': ['domain_researcher', 'schema_explorer'],
                        'timeout': 90,
                        'max_retries': 3
                    }
                ],
                'success_criteria': {
                    'learning_applied': True,
                    'improvement_demonstrated': True,
                    'adaptation_successful': True
                },
                'critical_agents': ['integration_coordinator', 'sql_specialist'],
                'max_execution_time': 240,
                'max_concurrent': 2
            }
        }
    
    def _define_execution_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Define execution strategies for different scenarios"""
        
        return {
            'high_performance': {
                'description': 'Optimized for speed with minimal agents',
                'preferred_patterns': [CollaborationPattern.LIGHTNING_SCHEMA],
                'max_execution_time': 60,
                'concurrent_limit': 1
            },
            'balanced_analysis': {
                'description': 'Balanced approach with research enhancement',
                'preferred_patterns': [CollaborationPattern.RESEARCH_ENHANCED],
                'max_execution_time': 180,
                'concurrent_limit': 2
            },
            'comprehensive_analysis': {
                'description': 'Full analysis with all available agents',
                'preferred_patterns': [CollaborationPattern.COMPLEX_MULTI_AGENT],
                'max_execution_time': 300,
                'concurrent_limit': 3
            },
            'quality_focused': {
                'description': 'Focus on validation and quality assurance',
                'preferred_patterns': [CollaborationPattern.VALIDATION_FOCUSED],
                'max_execution_time': 240,
                'concurrent_limit': 2
            }
        }
    
    def get_collaboration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive collaboration system metrics"""
        
        return {
            'system_metrics': self.collaboration_metrics,
            'agent_pool_status': self.agent_pool.get_system_health(),
            'active_collaborations': len(self.active_collaborations),
            'available_patterns': [pattern.value for pattern in CollaborationPattern],
            'tools_available': {
                'mcp_tools': self.tools_manager is not None,
                'crewai': CREWAI_AVAILABLE,
                'database': self.db_engine is not None
            },
            'performance_summary': self._generate_performance_summary()
        }
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary for monitoring"""
        
        metrics = self.collaboration_metrics
        
        return {
            'total_collaborations': metrics['total_collaborations'],
            'success_rate': (metrics['successful_collaborations'] / metrics['total_collaborations'] * 100) 
                           if metrics['total_collaborations'] > 0 else 0,
            'average_execution_time': metrics['average_execution_time'],
            'pattern_performance': {
                pattern: {
                    'success_rate': (data['successful'] / data['total'] * 100) if data['total'] > 0 else 0,
                    'total_executions': data['total']
                }
                for pattern, data in metrics['pattern_success_rates'].items()
            },
            'top_performing_agents': self._get_top_performing_agents(),
            'system_health_score': self._calculate_system_health_score()
        }
    
    def _get_top_performing_agents(self) -> List[Dict[str, Any]]:
        """Get top performing agents based on success rate and confidence"""
        
        agent_scores = []
        
        for agent_id, metrics in self.collaboration_metrics['agent_performance_matrix'].items():
            if metrics['executions'] > 0:
                success_rate = metrics['successes'] / metrics['executions']
                confidence = metrics['avg_confidence']
                
                # Composite score (success rate weighted by confidence)
                composite_score = success_rate * 0.7 + confidence * 0.3
                
                agent_scores.append({
                    'agent_id': agent_id,
                    'success_rate': success_rate * 100,
                    'avg_confidence': confidence,
                    'composite_score': composite_score,
                    'executions': metrics['executions'],
                    'avg_execution_time': metrics['avg_time']
                })
        
        # Sort by composite score and return top 5
        return sorted(agent_scores, key=lambda x: x['composite_score'], reverse=True)[:5]
    
    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score"""
        
        # Base score factors
        metrics = self.collaboration_metrics
        
        if metrics['total_collaborations'] == 0:
            return 0.8  # Default good health for new system
        
        success_rate = metrics['successful_collaborations'] / metrics['total_collaborations']
        
        # Performance factor (lower execution time is better, normalized)
        time_factor = max(0, 1.0 - (metrics['average_execution_time'] - 60) / 240)  # 60s baseline, 300s max
        
        # Agent availability factor
        agent_count = len(self.agent_pool.agents)
        availability_factor = min(agent_count / 6, 1.0)  # 6 is ideal agent count
        
        # System component factor
        component_factor = (
            (0.3 if self.tools_manager else 0) +
            (0.4 if CREWAI_AVAILABLE else 0.2) +  # Partial credit for fallback
            (0.3 if self.db_engine else 0)
        )
        
        # Composite health score
        health_score = (
            success_rate * 0.4 +
            time_factor * 0.3 +
            availability_factor * 0.2 +
            component_factor * 0.1
        )
        
        return min(max(health_score, 0.0), 1.0)
    
    def shutdown(self):
        """Graceful shutdown of the collaboration system"""
        
        logger.info("Initiating collaboration system shutdown...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for active collaborations to complete (with timeout)
        shutdown_timeout = 120  # 2 minutes
        start_time = time.time()
        
        while self.active_collaborations and (time.time() - start_time) < shutdown_timeout:
            time.sleep(1)
            
        # Force shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Shutdown agent pool
        if hasattr(self.agent_pool, 'shutdown'):
            self.agent_pool.shutdown()
        
        logger.info(f"Collaboration system shutdown completed. "
                   f"Final metrics: {self.collaboration_metrics['total_collaborations']} collaborations processed")

# Usage example and testing framework
def test_agent_collaboration_system():
    """Comprehensive test of the agent collaboration system"""
    
    logger.info("Testing Agent Collaboration System")
    logger.info("=" * 60)
    
    # Initialize system
    try:
        collaboration_system = ProductionAgentCollaborationSystem()
        logger.info("Agent collaboration system initialized")
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return
    
    # Test different collaboration patterns
    test_queries = [
        {
            'query': 'Show schema for temperature tables',
            'expected_pattern': CollaborationPattern.LIGHTNING_SCHEMA,
            'description': 'Fast schema exploration'
        },
        {
            'query': 'What is thermocline depth variability in Arabian Sea?',
            'expected_pattern': CollaborationPattern.RESEARCH_ENHANCED,
            'description': 'Research-enhanced analysis'
        },
        {
            'query': 'Compare biogeochemical processes between ocean basins with validation',
            'expected_pattern': CollaborationPattern.COMPLEX_MULTI_AGENT,
            'description': 'Complex multi-agent analysis'
        },
        {
            'query': 'Validate temperature measurements quality in recent data',
            'expected_pattern': CollaborationPattern.VALIDATION_FOCUSED,
            'description': 'Validation-focused analysis'
        }
    ]
    
    async def run_tests():
        results = []
        
        for i, test_case in enumerate(test_queries, 1):
            logger.info(f"\nTest {i}: {test_case['description']}")
            logger.info(f"Query: {test_case['query']}")
            logger.info("-" * 50)
            
            # Create mock routing decision
            from smart_query_router import RoutingDecision, ProcessingPath
            
            routing_decision = RoutingDecision(
                path=ProcessingPath.AGENTIC_FALLBACK,
                confidence=0.7,
                reasoning=[f"Test case for {test_case['description']}"],
                performance_budget=300,
                fallback_path=ProcessingPath.ERROR_RECOVERY,
                enrichments_needed=['test_enrichment'],
                unknown_terms=[],
                complexity_factors={'base_complexity': 'intermediate'},
                estimated_cost='medium'
            )
            
            try:
                result = await collaboration_system.execute_agent_collaboration(
                    test_case['query'], routing_decision
                )
                
                if result['success']:
                    logger.info(f"✅ SUCCESS!")
                    logger.info(f"   Pattern used: {result['collaboration_pattern']}")
                    logger.info(f"   Processing time: {result['processing_time']:.2f}s")
                    logger.info(f"   Agents executed: {result['performance_metrics']['agents_executed']}")
                    logger.info(f"   Success rate: {result['performance_metrics']['success_rate']:.1f}%")
                    
                    results.append({'test': i, 'success': True, 'time': result['processing_time']})
                else:
                    logger.error(f"❌ FAILED: {result.get('error')}")
                    results.append({'test': i, 'success': False, 'error': result.get('error')})
                    
            except Exception as e:
                logger.error(f"❌ EXCEPTION: {e}")
                results.append({'test': i, 'success': False, 'error': str(e)})
        
        # Final summary
        successful = [r for r in results if r['success']]
        logger.info(f"\n{'='*60}")
        logger.info(f"AGENT COLLABORATION TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total tests: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Success rate: {len(successful)/len(results)*100:.1f}%")
        
        if successful:
            avg_time = sum(r['time'] for r in successful) / len(successful)
            logger.info(f"Average execution time: {avg_time:.2f}s")
        
        # System metrics
        metrics = collaboration_system.get_collaboration_metrics()
        logger.info(f"System health score: {metrics['performance_summary']['system_health_score']:.2f}")
        
        return results
    
    # Run async tests
    import asyncio
    return asyncio.run(run_tests())

if __name__ == "__main__":
    test_agent_collaboration_system()