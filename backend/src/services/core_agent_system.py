# src/services/core_agent_system.py
"""
Phase 3: Core CrewAI Agent System Architecture

This is the foundation layer that provides the robust infrastructure for 
all specialized oceanographic agents. It handles:

1. Agent lifecycle management
2. Error recovery and fallback mechanisms
3. Performance monitoring and optimization
4. Inter-agent communication protocols
5. Resource allocation and timeout management

DESIGN PRINCIPLES:
- Fail-safe: Every operation has fallback mechanisms
- Observable: All actions are logged and monitored
- Scalable: Can handle concurrent agent operations
- Robust: Handles network failures, timeouts, and resource constraints
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import threading
from contextlib import asynccontextmanager
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import queue
import uuid

# CrewAI and MCP imports with fallbacks
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logging.warning("CrewAI not available - agent system will run in simulation mode")

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status tracking"""
    IDLE = "idle"
    WORKING = "working"
    FAILED = "failed"
    TIMEOUT = "timeout"
    COMPLETED = "completed"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AgentCapability(Enum):
    """Agent capability categories"""
    DATABASE_EXPLORATION = "database_exploration"
    DOMAIN_RESEARCH = "domain_research"
    SQL_GENERATION = "sql_generation"
    RESULT_VALIDATION = "result_validation"
    LITERATURE_SEARCH = "literature_search"
    SCHEMA_ANALYSIS = "schema_analysis"

@dataclass
class AgentMetrics:
    """Performance metrics for individual agents"""
    agent_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_execution_time: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    error_patterns: List[str] = field(default_factory=list)
    capabilities_used: Dict[str, int] = field(default_factory=dict)

@dataclass
class TaskRequest:
    """Task request with full context and requirements"""
    task_id: str
    description: str
    required_capabilities: List[AgentCapability]
    input_data: Dict[str, Any]
    priority: TaskPriority
    timeout_seconds: int
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskResult:
    """Comprehensive task result with metadata"""
    task_id: str
    success: bool
    result_data: Dict[str, Any]
    execution_time: float
    agent_id: str
    error_message: Optional[str] = None
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    resources_used: Dict[str, float] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)

class AgentPoolManager:
    """
    Manages a pool of specialized agents with load balancing and health monitoring
    """
    
    def __init__(self, max_concurrent_tasks: int = 5):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.agents: Dict[str, Any] = {}  # Agent instances
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.task_queue = queue.PriorityQueue()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # System monitoring
        self.system_health = {
            'total_tasks_processed': 0,
            'current_load': 0,
            'error_rate': 0.0,
            'average_response_time': 0.0,
            'last_health_check': datetime.now()
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'max_execution_time': 300,  # 5 minutes
            'max_error_rate': 0.15,     # 15%
            'max_concurrent_load': 0.8  # 80% of capacity
        }
        
        self._shutdown_event = threading.Event()
        self._monitoring_thread = None
        
        logger.info(f"Agent Pool Manager initialized with {max_concurrent_tasks} max concurrent tasks")
    
    def register_agent(self, agent_id: str, agent_instance: Any, 
                      capabilities: List[AgentCapability]) -> bool:
        """
        Register a specialized agent with the pool
        
        Args:
            agent_id: Unique identifier for the agent
            agent_instance: The actual agent object (CrewAI Agent or fallback)
            capabilities: List of capabilities this agent provides
            
        Returns:
            bool: True if registration successful
        """
        try:
            self.agents[agent_id] = {
                'instance': agent_instance,
                'capabilities': capabilities,
                'status': AgentStatus.IDLE,
                'registered_at': datetime.now(),
                'last_health_check': datetime.now()
            }
            
            self.agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)
            
            logger.info(f"Agent {agent_id} registered with capabilities: {[cap.value for cap in capabilities]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False
    
    def submit_task(self, task_request: TaskRequest) -> str:
        """
        Submit a task for execution by appropriate agent
        
        Args:
            task_request: Complete task specification
            
        Returns:
            str: Task ID for tracking
        """
        try:
            # Validate task request
            if not self._validate_task_request(task_request):
                raise ValueError(f"Invalid task request: {task_request.task_id}")
            
            # Find suitable agent
            suitable_agents = self._find_suitable_agents(task_request.required_capabilities)
            if not suitable_agents:
                raise RuntimeError(f"No agents available for capabilities: {task_request.required_capabilities}")
            
            # Add to queue with priority
            priority_score = self._calculate_priority_score(task_request)
            self.task_queue.put((priority_score, task_request))
            
            logger.info(f"Task {task_request.task_id} queued with priority {priority_score}")
            return task_request.task_id
            
        except Exception as e:
            logger.error(f"Failed to submit task {task_request.task_id}: {e}")
            raise
    
    async def execute_task_async(self, task_request: TaskRequest) -> TaskResult:
        """
        Execute task asynchronously with full error handling and monitoring
        
        Args:
            task_request: Task to execute
            
        Returns:
            TaskResult: Complete execution result
        """
        start_time = time.time()
        task_id = task_request.task_id
        
        try:
            # Select best agent for this task
            selected_agent_id = self._select_best_agent(task_request.required_capabilities)
            if not selected_agent_id:
                raise RuntimeError(f"No suitable agent available for task {task_id}")
            
            # Update agent status
            self.agents[selected_agent_id]['status'] = AgentStatus.WORKING
            self.active_tasks[task_id] = task_request
            
            # Execute with timeout protection
            result_data = await asyncio.wait_for(
                self._execute_agent_task(selected_agent_id, task_request),
                timeout=task_request.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            # Create successful result
            result = TaskResult(
                task_id=task_id,
                success=True,
                result_data=result_data,
                execution_time=execution_time,
                agent_id=selected_agent_id,
                confidence_score=result_data.get('confidence', 0.8),
                metadata={
                    'capabilities_used': [cap.value for cap in task_request.required_capabilities],
                    'retry_count': task_request.retry_count,
                    'priority': task_request.priority.value
                },
                resources_used={
                    'cpu_time': execution_time,
                    'memory_mb': result_data.get('memory_usage', 0)
                }
            )
            
            # Update metrics
            self._update_agent_metrics(selected_agent_id, True, execution_time)
            
            logger.info(f"Task {task_id} completed successfully in {execution_time:.2f}s by agent {selected_agent_id}")
            return result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"Task {task_id} timed out after {execution_time:.2f}s")
            
            return self._create_timeout_result(task_request, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task_id} failed after {execution_time:.2f}s: {e}")
            
            return self._create_error_result(task_request, execution_time, str(e))
            
        finally:
            # Cleanup
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            # Reset agent status
            for agent_id, agent_info in self.agents.items():
                if agent_info['status'] == AgentStatus.WORKING:
                    agent_info['status'] = AgentStatus.IDLE
    
    async def _execute_agent_task(self, agent_id: str, task_request: TaskRequest) -> Dict[str, Any]:
        """
        Execute task using specified agent with proper error handling
        
        Args:
            agent_id: ID of agent to execute task
            task_request: Task specification
            
        Returns:
            Dict containing execution results
        """
        agent_info = self.agents[agent_id]
        agent_instance = agent_info['instance']
        
        try:
            if CREWAI_AVAILABLE and hasattr(agent_instance, 'execute'):
                # CrewAI agent execution
                result = await self._execute_crewai_task(agent_instance, task_request)
            else:
                # Fallback execution for non-CrewAI agents
                result = await self._execute_fallback_task(agent_instance, task_request)
            
            return result
            
        except Exception as e:
            logger.error(f"Agent {agent_id} execution failed: {e}")
            raise
    
    async def _execute_crewai_task(self, agent_instance: Any, task_request: TaskRequest) -> Dict[str, Any]:
        """Execute task using CrewAI agent"""
        try:
            # Create CrewAI task
            task = Task(
                description=task_request.description,
                agent=agent_instance,
                expected_output="Structured JSON response with analysis results"
            )
            
            # Create single-agent crew for execution
            crew = Crew(
                agents=[agent_instance],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )
            
            # Execute crew
            result = crew.kickoff()
            
            return {
                'output': str(result),
                'success': True,
                'agent_type': 'crewai',
                'confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"CrewAI execution failed: {e}")
            raise
    
    async def _execute_fallback_task(self, agent_instance: Any, task_request: TaskRequest) -> Dict[str, Any]:
        """Execute task using fallback agent implementation"""
        try:
            # Call agent's process method directly
            if hasattr(agent_instance, 'process'):
                result = agent_instance.process(task_request.input_data)
            elif callable(agent_instance):
                result = agent_instance(task_request.input_data)
            else:
                raise ValueError(f"Agent instance not executable: {type(agent_instance)}")
            
            return {
                'output': result,
                'success': True,
                'agent_type': 'fallback',
                'confidence': 0.6
            }
            
        except Exception as e:
            logger.error(f"Fallback execution failed: {e}")
            raise
    
    def _validate_task_request(self, task_request: TaskRequest) -> bool:
        """Validate task request completeness and feasibility"""
        try:
            # Check required fields
            if not all([
                task_request.task_id,
                task_request.description,
                task_request.required_capabilities,
                task_request.timeout_seconds > 0
            ]):
                return False
            
            # Check capability availability
            available_capabilities = set()
            for agent_info in self.agents.values():
                available_capabilities.update(agent_info['capabilities'])
            
            required_caps = set(task_request.required_capabilities)
            if not required_caps.issubset(available_capabilities):
                logger.warning(f"Missing capabilities for task {task_request.task_id}: {required_caps - available_capabilities}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Task validation error: {e}")
            return False
    
    def _find_suitable_agents(self, required_capabilities: List[AgentCapability]) -> List[str]:
        """Find agents that can handle the required capabilities"""
        suitable_agents = []
        
        for agent_id, agent_info in self.agents.items():
            agent_capabilities = set(agent_info['capabilities'])
            required_caps = set(required_capabilities)
            
            # Agent must have ALL required capabilities
            if required_caps.issubset(agent_capabilities):
                suitable_agents.append(agent_id)
        
        return suitable_agents
    
    def _select_best_agent(self, required_capabilities: List[AgentCapability]) -> Optional[str]:
        """Select the best available agent for the task based on performance metrics"""
        suitable_agents = self._find_suitable_agents(required_capabilities)
        if not suitable_agents:
            return None
        
        # Score agents based on performance metrics
        best_agent = None
        best_score = -1
        
        for agent_id in suitable_agents:
            if self.agents[agent_id]['status'] != AgentStatus.IDLE:
                continue
            
            metrics = self.agent_metrics[agent_id]
            
            # Calculate performance score
            success_rate = (metrics.successful_tasks / max(metrics.total_tasks, 1))
            speed_score = 1.0 / max(metrics.average_execution_time, 1.0)
            recency_score = 1.0 / max((datetime.now() - metrics.last_activity).total_seconds() / 3600, 1.0)
            
            composite_score = (success_rate * 0.5) + (speed_score * 0.3) + (recency_score * 0.2)
            
            if composite_score > best_score:
                best_score = composite_score
                best_agent = agent_id
        
        return best_agent
    
    def _calculate_priority_score(self, task_request: TaskRequest) -> int:
        """Calculate priority score for queue ordering (lower score = higher priority)"""
        base_priority = task_request.priority.value
        
        # Adjust based on retry count (higher retries get higher priority)
        retry_bonus = task_request.retry_count * 0.5
        
        # Adjust based on age (older tasks get higher priority)
        age_minutes = (datetime.now() - task_request.created_at).total_seconds() / 60
        age_bonus = min(age_minutes / 10, 2.0)  # Cap at 2.0
        
        final_score = base_priority - retry_bonus - age_bonus
        return int(final_score * 10)  # Scale for priority queue
    
    def _update_agent_metrics(self, agent_id: str, success: bool, execution_time: float):
        """Update performance metrics for an agent"""
        try:
            metrics = self.agent_metrics[agent_id]
            
            metrics.total_tasks += 1
            metrics.last_activity = datetime.now()
            
            if success:
                metrics.successful_tasks += 1
            else:
                metrics.failed_tasks += 1
            
            # Update average execution time
            if metrics.total_tasks == 1:
                metrics.average_execution_time = execution_time
            else:
                # Weighted average favoring recent performance
                weight = 0.7
                metrics.average_execution_time = (
                    (1 - weight) * metrics.average_execution_time + 
                    weight * execution_time
                )
            
            # Update system health metrics
            self._update_system_health_metrics()
            
        except Exception as e:
            logger.error(f"Failed to update metrics for agent {agent_id}: {e}")
    
    def _update_system_health_metrics(self):
        """Update overall system health metrics"""
        try:
            total_tasks = sum(m.total_tasks for m in self.agent_metrics.values())
            successful_tasks = sum(m.successful_tasks for m in self.agent_metrics.values())
            
            self.system_health.update({
                'total_tasks_processed': total_tasks,
                'current_load': len(self.active_tasks) / self.max_concurrent_tasks,
                'error_rate': 1.0 - (successful_tasks / max(total_tasks, 1)),
                'last_health_check': datetime.now()
            })
            
            if total_tasks > 0:
                avg_time = sum(m.average_execution_time for m in self.agent_metrics.values()) / len(self.agent_metrics)
                self.system_health['average_response_time'] = avg_time
            
        except Exception as e:
            logger.error(f"Failed to update system health metrics: {e}")
    
    def _create_timeout_result(self, task_request: TaskRequest, execution_time: float) -> TaskResult:
        """Create result object for timed-out tasks"""
        return TaskResult(
            task_id=task_request.task_id,
            success=False,
            result_data={'error': 'Task execution timed out'},
            execution_time=execution_time,
            agent_id='timeout',
            error_message=f"Task timed out after {task_request.timeout_seconds} seconds",
            confidence_score=0.0,
            metadata={'timeout_seconds': task_request.timeout_seconds}
        )
    
    def _create_error_result(self, task_request: TaskRequest, execution_time: float, error_message: str) -> TaskResult:
        """Create result object for failed tasks"""
        return TaskResult(
            task_id=task_request.task_id,
            success=False,
            result_data={'error': error_message},
            execution_time=execution_time,
            agent_id='error',
            error_message=error_message,
            confidence_score=0.0,
            metadata={
                'retry_count': task_request.retry_count,
                'max_retries': task_request.max_retries
            }
        )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        return {
            'system_metrics': self.system_health.copy(),
            'agent_status': {
                agent_id: {
                    'status': info['status'].value,
                    'capabilities': [cap.value for cap in info['capabilities']],
                    'last_health_check': info['last_health_check'].isoformat()
                }
                for agent_id, info in self.agents.items()
            },
            'performance_metrics': {
                agent_id: {
                    'total_tasks': metrics.total_tasks,
                    'success_rate': metrics.successful_tasks / max(metrics.total_tasks, 1),
                    'average_execution_time': metrics.average_execution_time,
                    'last_activity': metrics.last_activity.isoformat()
                }
                for agent_id, metrics in self.agent_metrics.items()
            },
            'active_tasks': len(self.active_tasks),
            'queue_length': self.task_queue.qsize(),
            'system_health_score': self._calculate_system_health_score()
        }
    
    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score (0-1)"""
        try:
            # Base score factors
            error_rate = self.system_health.get('error_rate', 0)
            load_factor = self.system_health.get('current_load', 0)
            
            # Health score calculation
            error_score = max(0, 1.0 - (error_rate * 2))  # Penalize errors heavily
            load_score = max(0, 1.0 - max(0, load_factor - 0.7) * 3)  # Penalize high load
            
            # Agent availability score
            idle_agents = sum(1 for info in self.agents.values() if info['status'] == AgentStatus.IDLE)
            availability_score = idle_agents / max(len(self.agents), 1)
            
            # Composite health score
            health_score = (error_score * 0.4) + (load_score * 0.4) + (availability_score * 0.2)
            
            return min(max(health_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 0.5  # Default moderate health
    
    def shutdown(self):
        """Graceful shutdown of the agent pool"""
        logger.info("Initiating agent pool shutdown...")
        
        self._shutdown_event.set()
        
        # Wait for active tasks to complete (with timeout)
        shutdown_timeout = 60  # seconds
        start_time = time.time()
        
        while self.active_tasks and (time.time() - start_time) < shutdown_timeout:
            time.sleep(1)
        
        # Force shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Agent pool shutdown completed")

# Usage Example and Testing Framework
class AgentSystemTester:
    """Testing framework for the agent system"""
    
    def __init__(self):
        self.pool_manager = AgentPoolManager(max_concurrent_tasks=3)
        self.test_results = []
    
    def setup_test_agents(self):
        """Set up mock agents for testing"""
        
        # Mock Database Explorer Agent
        class MockDatabaseAgent:
            def process(self, data):
                return f"Database exploration completed for query: {data.get('query', 'unknown')}"
        
        # Mock Domain Research Agent  
        class MockResearchAgent:
            def process(self, data):
                return f"Research completed for terms: {data.get('unknown_terms', [])}"
        
        # Register test agents
        self.pool_manager.register_agent(
            'test_db_agent',
            MockDatabaseAgent(),
            [AgentCapability.DATABASE_EXPLORATION, AgentCapability.SCHEMA_ANALYSIS]
        )
        
        self.pool_manager.register_agent(
            'test_research_agent', 
            MockResearchAgent(),
            [AgentCapability.DOMAIN_RESEARCH, AgentCapability.LITERATURE_SEARCH]
        )
    
    async def run_basic_tests(self):
        """Run basic functionality tests"""
        
        self.setup_test_agents()
        
        # Test 1: Basic task execution
        task1 = TaskRequest(
            task_id='test_task_1',
            description='Test database exploration',
            required_capabilities=[AgentCapability.DATABASE_EXPLORATION],
            input_data={'query': 'SHOW TABLES'},
            priority=TaskPriority.MEDIUM,
            timeout_seconds=30
        )
        
        result1 = await self.pool_manager.execute_task_async(task1)
        self.test_results.append(('basic_execution', result1.success))
        
        # Test 2: Multi-capability task
        task2 = TaskRequest(
            task_id='test_task_2',
            description='Test research capabilities',
            required_capabilities=[AgentCapability.DOMAIN_RESEARCH],
            input_data={'unknown_terms': ['thermocline', 'pycnocline']},
            priority=TaskPriority.HIGH,
            timeout_seconds=30
        )
        
        result2 = await self.pool_manager.execute_task_async(task2)
        self.test_results.append(('research_execution', result2.success))
        
        # Test 3: System health check
        health = self.pool_manager.get_system_health()
        self.test_results.append(('health_check', health['system_health_score'] > 0.5))
        
        return self.test_results

# Example usage
async def demo_agent_system():
    """Demonstrate the agent system capabilities"""
    
    tester = AgentSystemTester()
    results = await tester.run_basic_tests()
    
    print("Agent System Test Results:")
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
    
    # Show system health
    health = tester.pool_manager.get_system_health()
    print(f"\nSystem Health Score: {health['system_health_score']:.2f}")
    print(f"Active Agents: {len(health['agent_status'])}")
    print(f"Total Tasks Processed: {health['system_metrics']['total_tasks_processed']}")

if __name__ == "__main__":
    asyncio.run(demo_agent_system())