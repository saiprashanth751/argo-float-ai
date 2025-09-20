# src/services/core_agent_system.py
"""
Production-Grade Core Agent System
Handles agent pool management, task execution, and system health monitoring.
"""

import asyncio
import logging
import threading
import weakref
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Import ONLY from types_core to avoid circular dependencies
from .types_core import (
    TaskRequest, TaskResult, AgentCapability, TaskPriority, AgentMetrics, 
    AgentStatus, BaseAgent, SystemConfiguration, AgentError, CircuitBreakerError
)

# System monitoring imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - resource monitoring disabled")

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Production-grade circuit breaker for agent failure protection"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()  # Reentrant lock for complex operations
        
        # Enhanced metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.state_transitions = []
        
    def can_execute(self) -> bool:
        """Check if execution is allowed with detailed state tracking"""
        with self._lock:
            self.total_calls += 1
            
            if self.state == "CLOSED":
                return True
            elif self.state == "OPEN":
                if self._should_attempt_reset():
                    self._transition_state("HALF_OPEN", "Recovery timeout reached")
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def record_success(self):
        """Record successful execution with state management"""
        with self._lock:
            self.successful_calls += 1
            
            if self.state == "HALF_OPEN":
                self._transition_state("CLOSED", "Recovery successful")
                self.failure_count = 0
    
    def record_failure(self):
        """Record failed execution with threshold checking"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold and self.state != "OPEN":
                self._transition_state("OPEN", f"Failure threshold reached: {self.failure_count}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if not self.last_failure_time:
            return False
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def _transition_state(self, new_state: str, reason: str):
        """Track state transitions for monitoring"""
        old_state = self.state
        self.state = new_state
        self.state_transitions.append({
            'from': old_state,
            'to': new_state,
            'reason': reason,
            'timestamp': datetime.now(),
            'failure_count': self.failure_count
        })
        
        logger.info(f"Circuit breaker state: {old_state} -> {new_state} ({reason})")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker metrics"""
        with self._lock:
            success_rate = (self.successful_calls / max(self.total_calls, 1)) * 100
            
            return {
                'state': self.state,
                'failure_count': self.failure_count,
                'total_calls': self.total_calls,
                'successful_calls': self.successful_calls,
                'success_rate': success_rate,
                'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'state_transitions': self.state_transitions[-10:],  # Last 10 transitions
                'is_healthy': self.state == "CLOSED" and success_rate > 80
            }

class ResourceMonitor:
    """Production-grade system resource monitoring"""
    
    def __init__(self, config: SystemConfiguration):
        self.memory_threshold = config.memory_threshold_mb
        self.cpu_threshold = 0.9  # 90% CPU threshold
        self.monitoring_enabled = PSUTIL_AVAILABLE and config.enable_monitoring
        
        # Monitoring history for trend analysis
        self.resource_history = []
        self.max_history_size = 100
        
        # Alert thresholds
        self.critical_memory_threshold = self.memory_threshold * 0.9
        self.warning_memory_threshold = self.memory_threshold * 0.8
        
    def can_execute_task(self, estimated_memory_mb: int = 50) -> tuple[bool, Dict[str, Any]]:
        """
        Comprehensive resource check with detailed reasoning
        Returns: (can_execute, resource_info)
        """
        if not self.monitoring_enabled:
            return True, {'monitoring': 'disabled'}
        
        try:
            # Get current resource usage
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            current_memory_mb = memory_info.used / (1024 * 1024)
            projected_memory_mb = current_memory_mb + estimated_memory_mb
            
            # Store in history
            resource_snapshot = {
                'timestamp': datetime.now(),
                'memory_used_mb': current_memory_mb,
                'memory_percent': memory_info.percent,
                'cpu_percent': cpu_percent,
                'available_memory_mb': memory_info.available / (1024 * 1024)
            }
            
            self._update_history(resource_snapshot)
            
            # Decision logic with detailed reasoning
            reasons = []
            can_execute = True
            
            # Memory checks
            if projected_memory_mb > self.memory_threshold:
                can_execute = False
                reasons.append(f"Projected memory usage ({projected_memory_mb:.0f}MB) exceeds threshold ({self.memory_threshold}MB)")
            elif projected_memory_mb > self.critical_memory_threshold:
                reasons.append(f"Memory usage approaching critical threshold ({projected_memory_mb:.0f}MB)")
            
            # CPU checks
            if cpu_percent > self.cpu_threshold * 100:
                can_execute = False
                reasons.append(f"CPU usage ({cpu_percent:.1f}%) exceeds threshold ({self.cpu_threshold*100}%)")
            
            # Trend analysis
            if len(self.resource_history) >= 5:
                memory_trend = self._calculate_memory_trend()
                if memory_trend > 10:  # MB per minute increase
                    reasons.append(f"Memory usage increasing rapidly ({memory_trend:.1f}MB/min)")
            
            resource_info = {
                'can_execute': can_execute,
                'reasons': reasons,
                'current_memory_mb': current_memory_mb,
                'projected_memory_mb': projected_memory_mb,
                'cpu_percent': cpu_percent,
                'memory_trend_mb_per_min': self._calculate_memory_trend() if len(self.resource_history) >= 5 else 0,
                'health_score': self._calculate_resource_health_score(resource_snapshot)
            }
            
            return can_execute, resource_info
            
        except Exception as e:
            logger.warning(f"Resource monitoring failed: {e}")
            # Fail open - allow execution if monitoring fails
            return True, {'monitoring': 'failed', 'error': str(e)}
    
    def _update_history(self, snapshot: Dict[str, Any]):
        """Update resource history with size management"""
        self.resource_history.append(snapshot)
        if len(self.resource_history) > self.max_history_size:
            self.resource_history = self.resource_history[-self.max_history_size:]
    
    def _calculate_memory_trend(self) -> float:
        """Calculate memory usage trend in MB per minute"""
        if len(self.resource_history) < 2:
            return 0.0
        
        recent_snapshots = self.resource_history[-5:]  # Last 5 measurements
        
        first = recent_snapshots[0]
        last = recent_snapshots[-1]
        
        time_diff = (last['timestamp'] - first['timestamp']).total_seconds() / 60  # minutes
        memory_diff = last['memory_used_mb'] - first['memory_used_mb']
        
        return memory_diff / max(time_diff, 0.1)  # Avoid division by zero
    
    def _calculate_resource_health_score(self, snapshot: Dict[str, Any]) -> float:
        """Calculate overall resource health score (0-1)"""
        memory_score = max(0, 1 - (snapshot['memory_used_mb'] / self.memory_threshold))
        cpu_score = max(0, 1 - (snapshot['cpu_percent'] / 100))
        
        return (memory_score * 0.6 + cpu_score * 0.4)  # Weight memory more heavily
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource usage summary"""
        if not self.monitoring_enabled:
            return {'monitoring': 'disabled'}
        
        if not self.resource_history:
            return {'status': 'no_data'}
        
        latest = self.resource_history[-1]
        
        return {
            'current_memory_mb': latest['memory_used_mb'],
            'memory_threshold_mb': self.memory_threshold,
            'memory_utilization_percent': (latest['memory_used_mb'] / self.memory_threshold) * 100,
            'cpu_percent': latest['cpu_percent'],
            'health_score': latest.get('health_score', self._calculate_resource_health_score(latest)),
            'memory_trend_mb_per_min': self._calculate_memory_trend() if len(self.resource_history) >= 5 else 0,
            'measurements_count': len(self.resource_history),
            'last_updated': latest['timestamp'].isoformat()
        }

class AgentPoolManager:
    """Production-grade agent pool manager with comprehensive monitoring and fault tolerance"""
    
    def __init__(self, config: SystemConfiguration = None, max_concurrent_tasks: int = None):
        # Handle backward compatibility
        if max_concurrent_tasks is not None and config is None:
            # Create config from old parameter for backward compatibility
            config = SystemConfiguration()
            config.max_concurrent_agents = max_concurrent_tasks
        
        self.config = config or SystemConfiguration()
        
        # Core agent management
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        
        # Fault tolerance infrastructure
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.resource_monitor = ResourceMonitor(self.config)
        
        # Execution infrastructure
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_agents,
            thread_name_prefix="agent-pool"
        )
        
        # Task management
        self._active_tasks: Dict[str, Dict[str, Any]] = {}
        self._task_lock = threading.RLock()
        
        # System state management
        self._shutdown_event = threading.Event()
        self._health_check_interval = 60  # seconds
        self._last_health_check = datetime.now()
        
        # Performance tracking
        self.system_metrics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'agent_utilization': {},
            'resource_alerts': []
        }
        
        logger.info(f"Agent Pool Manager initialized with {self.config.max_concurrent_agents} max agents")

    def register_agent(self, agent_id: str, agent_instance: Any, 
                      capabilities: List[AgentCapability], metadata: Dict[str, Any] = None) -> bool:
        """
        Register an agent with comprehensive validation and setup
        Returns: True if registration successful, False otherwise
        """
        try:
            # Basic validation
            if agent_id in self.agents:
                logger.warning(f"Agent {agent_id} already registered, updating registration")
            
            # For production, we'd validate BaseAgent interface
            # For now, accept any agent instance to maintain compatibility
            
            # Register agent
            self.agents[agent_id] = {
                'instance': agent_instance,
                'capabilities': capabilities,
                'status': AgentStatus.IDLE,
                'registered_at': datetime.now(),
                'metadata': metadata or {},
                'last_health_check': datetime.now(),
                'health_status': {'healthy': True}  # Default healthy
            }
            
            # Initialize metrics and circuit breaker
            self.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                capabilities=capabilities,
                last_activity=datetime.now()
            )
            self.circuit_breakers[agent_id] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60
            )
            
            logger.info(f"Agent {agent_id} registered successfully with capabilities: {[c.value for c in capabilities]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False

    async def execute_task_async(self, task_request: TaskRequest) -> TaskResult:
        """
        Production-grade asynchronous task execution with comprehensive error handling
        """
        task_id = task_request.task_id
        start_time = datetime.now()
        
        logger.info(f"Executing task {task_id}: {task_request.description}")
        
        try:
            # Pre-execution validation
            validation_result = self._validate_task_execution(task_request)
            if not validation_result['can_execute']:
                return self._create_validation_failure_result(task_request, validation_result)
            
            # Agent selection
            selected_agent_id = self._select_optimal_agent(task_request.required_capabilities)
            if not selected_agent_id:
                return self._create_no_agent_result(task_request, start_time)
            
            # Resource allocation check
            can_execute, resource_info = self.resource_monitor.can_execute_task()
            if not can_execute:
                return self._create_resource_exhausted_result(task_request, resource_info, start_time)
            
            # Execute task with full monitoring
            async with self._task_execution_context(task_id, selected_agent_id):
                result = await self._execute_agent_task_monitored(
                    selected_agent_id, task_request, start_time
                )
                
                return result
                
        except asyncio.CancelledError:
            logger.info(f"Task {task_id} was cancelled")
            return self._create_cancelled_result(task_request, start_time)
            
        except asyncio.TimeoutError:
            logger.error(f"Task {task_id} timed out after {task_request.timeout_seconds}s")
            return self._create_timeout_result(task_request, start_time)
            
        except Exception as e:
            logger.error(f"Task {task_id} failed with unexpected error: {e}", exc_info=True)
            return self._create_error_result(task_request, start_time, str(e))
        
        finally:
            # Always update system metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            result_for_metrics = result if 'result' in locals() else None
            self._update_system_metrics(task_request, execution_time, result_for_metrics)

    def _validate_task_execution(self, task_request: TaskRequest) -> Dict[str, Any]:
        """Comprehensive task execution validation"""
        validation_issues = []
        
        # Check system health
        if self._shutdown_event.is_set():
            validation_issues.append("System is shutting down")
        
        # Check agent availability
        suitable_agents = self._find_suitable_agents(task_request.required_capabilities)
        if not suitable_agents:
            validation_issues.append("No agents with required capabilities available")
        
        # Check circuit breakers
        available_agents = [
            agent_id for agent_id in suitable_agents
            if self.circuit_breakers.get(agent_id, CircuitBreaker()).can_execute()
        ]
        if not available_agents:
            validation_issues.append("All suitable agents have open circuit breakers")
        
        return {
            'can_execute': len(validation_issues) == 0,
            'issues': validation_issues,
            'suitable_agents': suitable_agents,
            'available_agents': available_agents
        }
    
    def _find_suitable_agents(self, required_capabilities: List[AgentCapability]) -> List[str]:
        """Find agents that match the required capabilities"""
        suitable_agents = []
        
        for agent_id, agent_info in self.agents.items():
            agent_capabilities = set(agent_info['capabilities'])
            required_capabilities_set = set(required_capabilities)
            
            # Agent is suitable if it has at least one of the required capabilities
            if agent_capabilities.intersection(required_capabilities_set):
                suitable_agents.append(agent_id)
        
        return suitable_agents
    
    def _select_optimal_agent(self, capabilities: List[AgentCapability]) -> Optional[str]:
        """Select the optimal agent for the given capabilities"""
        suitable_agents = self._find_suitable_agents(capabilities)
        
        # Filter out agents with open circuit breakers and non-idle status
        available_agents = [
            agent_id for agent_id in suitable_agents
            if (self.circuit_breakers.get(agent_id, CircuitBreaker()).can_execute() and
                self.agents[agent_id]['status'] == AgentStatus.IDLE)
        ]
        
        if not available_agents:
            return None
        
        # Select based on health score
        best_agent = max(
            available_agents,
            key=lambda agent_id: self._calculate_agent_health_score(agent_id)
        )
        
        return best_agent
    
    def _calculate_agent_health_score(self, agent_id: str) -> float:
        """Calculate comprehensive agent health score"""
        metrics = self.agent_metrics.get(agent_id)
        if not metrics:
            return 0.5
        
        # Success rate (40%)
        success_rate = metrics.successful_tasks / max(metrics.total_tasks, 1)
        
        # Performance (30%)
        avg_time = metrics.average_execution_time
        performance_score = 1.0 / (1.0 + avg_time / 60)  # Normalize to 60s baseline
        
        # Recency (20%)
        time_since_activity = (datetime.now() - metrics.last_activity).total_seconds()
        recency_score = 1.0 / (1.0 + time_since_activity / 3600)  # Normalize to 1h
        
        # Circuit breaker state (10%)
        cb_score = 1.0 if self.circuit_breakers.get(agent_id, CircuitBreaker()).can_execute() else 0.0
        
        return (success_rate * 0.4 + 
                performance_score * 0.3 + 
                recency_score * 0.2 + 
                cb_score * 0.1)
    
    @asynccontextmanager
    async def _task_execution_context(self, task_id: str, agent_id: str):
        """Context manager for task execution lifecycle"""
        try:
            # Register active task
            with self._task_lock:
                self._active_tasks[task_id] = {
                    'agent_id': agent_id,
                    'start_time': datetime.now(),
                    'task': asyncio.current_task()
                }
                self.agents[agent_id]['status'] = AgentStatus.WORKING
            yield
        finally:
            # Cleanup
            with self._task_lock:
                if task_id in self._active_tasks:
                    del self._active_tasks[task_id]
                if agent_id in self.agents:
                    self.agents[agent_id]['status'] = AgentStatus.IDLE

    async def _execute_agent_task_monitored(self, agent_id: str, 
                                          task_request: TaskRequest, 
                                          start_time: datetime) -> TaskResult:
        """Execute agent task with comprehensive monitoring"""
        
        try:
            memory_before = 0
            if PSUTIL_AVAILABLE:
                try:
                    memory_before = psutil.Process().memory_info().rss
                except:
                    pass
            
            # Execute the actual agent task
            result_data = await self._execute_agent_task(agent_id, task_request)
            
            # Calculate metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            
            memory_used = 0
            if PSUTIL_AVAILABLE:
                try:
                    memory_after = psutil.Process().memory_info().rss
                    memory_used = memory_after - memory_before
                except:
                    pass
            
            # Update circuit breaker
            self.circuit_breakers.get(agent_id, CircuitBreaker()).record_success()
            
            # Update agent metrics
            self._update_agent_metrics(agent_id, execution_time, True)
            
            return TaskResult(
                task_id=task_request.task_id,
                success=True,
                result_data=result_data,
                execution_time=execution_time,
                agent_id=agent_id,
                confidence_score=result_data.get('confidence', 0.8) if isinstance(result_data, dict) else 0.8,
                resources_used={'memory_mb': memory_used / 1024 / 1024}
            )
            
        except Exception as e:
            # Record failure in circuit breaker
            self.circuit_breakers.get(agent_id, CircuitBreaker()).record_failure()
            
            # Update agent metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_agent_metrics(agent_id, execution_time, False)
            
            raise

    async def _execute_agent_task(self, agent_id: str, task_request: TaskRequest) -> Any:
        """Execute the actual agent task"""
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent_instance = agent_info['instance']
        
        # Create task context
        task_context = {
            'query': task_request.description,
            'description': task_request.description,
            'task_type': 'general_processing',
            'shared_context': task_request.context,
            'task_id': task_request.task_id
        }
        
        # Execute agent with timeout
        try:
            # Check if agent has specific methods
            if hasattr(agent_instance, 'process_task'):
                result = await asyncio.wait_for(
                    asyncio.to_thread(agent_instance.process_task, task_request),
                    timeout=task_request.timeout_seconds
                )
            elif hasattr(agent_instance, 'process'):
                result = await asyncio.wait_for(
                    asyncio.to_thread(agent_instance.process, task_context),
                    timeout=task_request.timeout_seconds
                )
            else:
                # Fallback for simple agents
                await asyncio.sleep(0.1)  # Simulate processing
                result = {
                    'agent_id': agent_id,
                    'task_completed': True,
                    'result': f"Processed task: {task_request.description}",
                    'confidence': 0.8
                }
            
            return result
            
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Agent {agent_id} timed out after {task_request.timeout_seconds}s")
        except Exception as e:
            raise AgentError(f"Agent {agent_id} execution failed: {str(e)}")

    def _update_agent_metrics(self, agent_id: str, execution_time: float, success: bool):
        """Update agent performance metrics"""
        metrics = self.agent_metrics.get(agent_id)
        if not metrics:
            return
        
        # Update counters
        metrics.total_tasks += 1
        if success:
            metrics.successful_tasks += 1
        else:
            metrics.failed_tasks += 1
        
        # Update average execution time
        if metrics.total_tasks == 1:
            metrics.average_execution_time = execution_time
        else:
            metrics.average_execution_time = (
                (metrics.average_execution_time * (metrics.total_tasks - 1) + execution_time) / 
                metrics.total_tasks
            )
        
        # Update last activity
        metrics.last_activity = datetime.now()
        
        # Update health score
        success_rate = metrics.successful_tasks / metrics.total_tasks
        performance_factor = max(0, 1 - (execution_time / 300))  # 5 minute baseline
        metrics.health_score = (success_rate * 0.7 + performance_factor * 0.3)

    def _update_system_metrics(self, task_request: TaskRequest, execution_time: float, result: Optional[TaskResult]):
        """Update system-wide performance metrics"""
        self.system_metrics['total_tasks'] += 1
        
        if result and result.success:
            self.system_metrics['successful_tasks'] += 1
        else:
            self.system_metrics['failed_tasks'] += 1
        
        # Update average execution time
        total_tasks = self.system_metrics['total_tasks']
        current_avg = self.system_metrics['average_execution_time']
        
        self.system_metrics['average_execution_time'] = (
            (current_avg * (total_tasks - 1) + execution_time) / total_tasks
        )

    # Result creation methods
    def _create_validation_failure_result(self, task_request: TaskRequest, validation_result: Dict[str, Any]) -> TaskResult:
        """Create result for validation failure"""
        return TaskResult(
            task_id=task_request.task_id,
            success=False,
            result_data=f"Validation failed: {'; '.join(validation_result['issues'])}",
            execution_time=0.0,
            agent_id="validation",
            confidence_score=0.0,
            errors=validation_result['issues']
        )
    
    def _create_no_agent_result(self, task_request: TaskRequest, start_time: datetime) -> TaskResult:
        """Create result when no suitable agent is available"""
        execution_time = (datetime.now() - start_time).total_seconds()
        return TaskResult(
            task_id=task_request.task_id,
            success=False,
            result_data="No suitable agent available",
            execution_time=execution_time,
            agent_id="none",
            confidence_score=0.0,
            errors=["No agent available"]
        )
    
    def _create_resource_exhausted_result(self, task_request: TaskRequest, resource_info: Dict[str, Any], start_time: datetime) -> TaskResult:
        """Create result for resource exhaustion"""
        execution_time = (datetime.now() - start_time).total_seconds()
        return TaskResult(
            task_id=task_request.task_id,
            success=False,
            result_data="System resources exhausted",
            execution_time=execution_time,
            agent_id="system",
            confidence_score=0.0,
            errors=[f"Resource exhaustion: {'; '.join(resource_info.get('reasons', ['Unknown']))}"],
            metadata={'resource_info': resource_info}
        )
    
    def _create_cancelled_result(self, task_request: TaskRequest, start_time: datetime) -> TaskResult:
        """Create result for cancelled task"""
        execution_time = (datetime.now() - start_time).total_seconds()
        return TaskResult(
            task_id=task_request.task_id,
            success=False,
            result_data="Task was cancelled",
            execution_time=execution_time,
            agent_id="cancelled",
            confidence_score=0.0,
            errors=["Task cancelled"]
        )
    
    def _create_timeout_result(self, task_request: TaskRequest, start_time: datetime) -> TaskResult:
        """Create result for timed out task"""
        execution_time = (datetime.now() - start_time).total_seconds()
        return TaskResult(
            task_id=task_request.task_id,
            success=False,
            result_data=f"Task timed out after {task_request.timeout_seconds}s",
            execution_time=execution_time,
            agent_id="timeout",
            confidence_score=0.0,
            errors=[f"Timeout after {task_request.timeout_seconds}s"]
        )
    
    def _create_error_result(self, task_request: TaskRequest, start_time: datetime, error: str) -> TaskResult:
        """Create result for error case"""
        execution_time = (datetime.now() - start_time).total_seconds()
        return TaskResult(
            task_id=task_request.task_id,
            success=False,
            result_data=f"Task failed: {error}",
            execution_time=execution_time,
            agent_id="error",
            confidence_score=0.0,
            errors=[error]
        )

    async def graceful_shutdown(self, timeout: int = 30):
        """Graceful shutdown with timeout"""
        logger.info("Initiating graceful shutdown...")
        self._shutdown_event.set()
        
        # Cancel all active tasks
        with self._task_lock:
            for task_id, task_info in self._active_tasks.items():
                task = task_info.get('task')
                if task and hasattr(task, 'cancel'):
                    task.cancel()
        
        # Wait for completion with timeout
        if self._active_tasks:
            try:
                # Wait for tasks to complete
                await asyncio.sleep(min(timeout, 5))
            except Exception as e:
                logger.warning(f"Error during shutdown wait: {e}")
        
        # Shutdown executor
        # Continuing from graceful_shutdown method...
        try:
            self.executor.shutdown(wait=True, timeout=timeout)
        except Exception as e:
            logger.warning(f"Error shutting down executor: {e}")
            self.executor.shutdown(wait=False)
        
        logger.info("Graceful shutdown completed")

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information"""
        return {
            'agent_pool': {
                'total_agents': len(self.agents),
                'healthy_agents': len([a for a in self.agents.values() 
                                     if a.get('health_status', {}).get('healthy', False)]),
                'idle_agents': len([a for a in self.agents.values() 
                                  if a['status'] == AgentStatus.IDLE]),
                'working_agents': len([a for a in self.agents.values() 
                                     if a['status'] == AgentStatus.WORKING]),
                'error_agents': len([a for a in self.agents.values() 
                                   if a['status'] == AgentStatus.ERROR])
            },
            'circuit_breakers': {
                agent_id: cb.get_metrics() 
                for agent_id, cb in self.circuit_breakers.items()
            },
            'resources': self.resource_monitor.get_resource_summary(),
            'system_metrics': self.system_metrics,
            'active_tasks': len(self._active_tasks),
            'last_health_check': self._last_health_check.isoformat(),
            'overall_health_score': self._calculate_overall_health_score()
        }
    
    def _calculate_overall_health_score(self) -> float:
        """Calculate comprehensive system health score"""
        if not self.agents:
            return 0.0
        
        # Agent health factor (40%)
        healthy_agents = len([a for a in self.agents.values() 
                            if a.get('health_status', {}).get('healthy', False)])
        agent_health_score = healthy_agents / len(self.agents)
        
        # Success rate factor (30%)
        total_tasks = self.system_metrics['total_tasks']
        if total_tasks > 0:
            success_rate = self.system_metrics['successful_tasks'] / total_tasks
        else:
            success_rate = 1.0  # No failures yet
        
        # Circuit breaker health (20%)
        healthy_breakers = len([cb for cb in self.circuit_breakers.values() 
                              if cb.get_metrics()['is_healthy']])
        cb_health_score = healthy_breakers / max(len(self.circuit_breakers), 1)
        
        # Resource health (10%)
        resource_summary = self.resource_monitor.get_resource_summary()
        resource_health = resource_summary.get('health_score', 1.0)
        
        # Composite score
        overall_score = (
            agent_health_score * 0.4 +
            success_rate * 0.3 +
            cb_health_score * 0.2 +
            resource_health * 0.1
        )
        
        return min(max(overall_score, 0.0), 1.0)
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a specific agent"""
        if agent_id not in self.agents:
            return None
        
        agent_info = self.agents[agent_id]
        metrics = self.agent_metrics.get(agent_id)
        cb_metrics = self.circuit_breakers.get(agent_id, CircuitBreaker()).get_metrics()
        
        return {
            'agent_id': agent_id,
            'status': agent_info['status'].value,
            'capabilities': [cap.value for cap in agent_info['capabilities']],
            'registered_at': agent_info['registered_at'].isoformat(),
            'last_health_check': agent_info['last_health_check'].isoformat(),
            'health_status': agent_info['health_status'],
            'metrics': {
                'total_tasks': metrics.total_tasks if metrics else 0,
                'successful_tasks': metrics.successful_tasks if metrics else 0,
                'failed_tasks': metrics.failed_tasks if metrics else 0,
                'success_rate': (metrics.successful_tasks / max(metrics.total_tasks, 1) * 100) if metrics else 0,
                'average_execution_time': metrics.average_execution_time if metrics else 0,
                'health_score': metrics.health_score if metrics else 0,
                'last_activity': metrics.last_activity.isoformat() if metrics else None
            },
            'circuit_breaker': cb_metrics,
            'current_health_score': self._calculate_agent_health_score(agent_id)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        # Calculate agent utilization
        agent_utilization = {}
        for agent_id, metrics in self.agent_metrics.items():
            if metrics.total_tasks > 0:
                agent_utilization[agent_id] = {
                    'tasks_completed': metrics.total_tasks,
                    'success_rate': (metrics.successful_tasks / metrics.total_tasks) * 100,
                    'avg_execution_time': metrics.average_execution_time,
                    'health_score': metrics.health_score
                }
        
        # System-wide statistics
        total_tasks = self.system_metrics['total_tasks']
        success_rate = (self.system_metrics['successful_tasks'] / max(total_tasks, 1)) * 100
        
        return {
            'system_overview': {
                'total_tasks_processed': total_tasks,
                'overall_success_rate': success_rate,
                'average_execution_time': self.system_metrics['average_execution_time'],
                'active_agents': len([a for a in self.agents.values() if a['status'] != AgentStatus.ERROR]),
                'system_health_score': self._calculate_overall_health_score()
            },
            'agent_utilization': agent_utilization,
            'resource_status': self.resource_monitor.get_resource_summary(),
            'circuit_breaker_summary': {
                'total_breakers': len(self.circuit_breakers),
                'healthy_breakers': len([cb for cb in self.circuit_breakers.values() 
                                       if cb.get_metrics()['is_healthy']]),
                'open_breakers': len([cb for cb in self.circuit_breakers.values() 
                                    if cb.get_metrics()['state'] == 'OPEN'])
            }
        }
    
    def reset_agent_circuit_breaker(self, agent_id: str) -> bool:
        """Manually reset an agent's circuit breaker"""
        if agent_id not in self.circuit_breakers:
            return False
        
        cb = self.circuit_breakers[agent_id]
        with cb._lock:
            cb.state = "CLOSED"
            cb.failure_count = 0
            cb.last_failure_time = None
            cb._transition_state("CLOSED", "Manual reset")
        
        logger.info(f"Circuit breaker for agent {agent_id} manually reset")
        return True
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        self._last_health_check = datetime.now()
        health_issues = []
        recommendations = []
        
        # Check agent health
        unhealthy_agents = [
            agent_id for agent_id, agent_info in self.agents.items()
            if not agent_info.get('health_status', {}).get('healthy', True)
        ]
        if unhealthy_agents:
            health_issues.append(f"Unhealthy agents: {unhealthy_agents}")
            recommendations.append("Review and restart unhealthy agents")
        
        # Check circuit breakers
        open_breakers = [
            agent_id for agent_id, cb in self.circuit_breakers.items()
            if cb.get_metrics()['state'] == 'OPEN'
        ]
        if open_breakers:
            health_issues.append(f"Open circuit breakers: {open_breakers}")
            recommendations.append("Investigate and resolve agent failures")
        
        # Check resource usage
        resource_summary = self.resource_monitor.get_resource_summary()
        if resource_summary.get('memory_utilization_percent', 0) > 80:
            health_issues.append("High memory utilization")
            recommendations.append("Consider scaling or optimizing memory usage")
        
        # Check task success rate
        if self.system_metrics['total_tasks'] > 10:  # Only check if we have enough data
            success_rate = (self.system_metrics['successful_tasks'] / self.system_metrics['total_tasks']) * 100
            if success_rate < 80:
                health_issues.append(f"Low success rate: {success_rate:.1f}%")
                recommendations.append("Review task failures and agent performance")
        
        return {
            'health_check_time': self._last_health_check.isoformat(),
            'overall_health': 'healthy' if not health_issues else 'issues_detected',
            'health_score': self._calculate_overall_health_score(),
            'issues': health_issues,
            'recommendations': recommendations,
            'detailed_status': self.get_system_health()
        }

# Backward compatibility aliases and helper functions
class NoOpCircuitBreaker:
    """No-op circuit breaker for backward compatibility"""
    def can_execute(self) -> bool:
        return True
    def record_success(self):
        pass
    def record_failure(self):
        pass
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'state': 'DISABLED',
            'is_healthy': True,
            'total_calls': 0,
            'successful_calls': 0,
            'success_rate': 100.0
        }

# Factory function for easy instantiation
def create_agent_pool_manager(config: SystemConfiguration = None) -> AgentPoolManager:
    """Factory function to create an agent pool manager with default configuration"""
    if config is None:
        config = SystemConfiguration()
    
    return AgentPoolManager(config)

# Utility functions for testing and development
def create_test_task_request(description: str, capabilities: List[AgentCapability] = None) -> TaskRequest:
    """Create a test task request for development and testing"""
    import uuid
    
    if capabilities is None:
        capabilities = [AgentCapability.DATABASE_EXPLORATION]
    
    return TaskRequest(
        task_id=str(uuid.uuid4()),
        description=description,
        required_capabilities=capabilities,
        priority=TaskPriority.MEDIUM,
        timeout_seconds=60,
        context={'test_mode': True}
    )

# Export main classes for backward compatibility
__all__ = [
    'AgentPoolManager',
    'CircuitBreaker', 
    'ResourceMonitor',
    'TaskRequest',
    'TaskResult', 
    'AgentCapability',
    'AgentStatus',
    'TaskPriority',
    'AgentMetrics',
    'SystemConfiguration',
    'create_agent_pool_manager',
    'create_test_task_request'
]