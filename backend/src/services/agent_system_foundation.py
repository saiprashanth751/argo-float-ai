# src/services/agent_system_foundation.py
"""
AGENT SYSTEM FOUNDATION LAYER - Single Source of Truth for Core Components
PRODUCTION-HARDENED with all missing critical components
"""

import logging
import sqlite3
import time
import threading
import weakref
import asyncio
import gc
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from sqlalchemy import text, Engine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
from collections import defaultdict
import uuid

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# CRITICAL MISSING COMPONENTS - Added to Foundation
# ============================================================================

class MemoryMonitor:
    """Production-grade memory monitoring and leak prevention"""
    
    def __init__(self, threshold_mb=2048, cleanup_interval=300):
        self.threshold_mb = threshold_mb
        self.cleanup_interval = cleanup_interval
        self.active_objects = weakref.WeakSet()
        self.memory_history = []
        self.last_cleanup = time.time()
        self._lock = threading.RLock()
        
        # Start monitoring thread
        self._start_monitoring()
    
    def register_object(self, obj):
        """Register object for memory tracking"""
        with self._lock:
            try:
                self.active_objects.add(obj)
            except TypeError:
                # Object doesn't support weak references
                pass
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits"""
        try:
            if not PSUTIL_AVAILABLE:
                return True
            
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            with self._lock:
                self.memory_history.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb,
                    'active_objects': len(self.active_objects)
                })
                
                # Keep only last 100 entries
                if len(self.memory_history) > 100:
                    self.memory_history = self.memory_history[-100:]
            
            if memory_mb > self.threshold_mb:
                logger.warning(f"High memory usage: {memory_mb:.1f}MB (threshold: {self.threshold_mb}MB)")
                self._emergency_cleanup()
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return True  # Don't block on monitoring failure
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup"""
        logger.info("Initiating emergency memory cleanup")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Clean up memory history
        with self._lock:
            self.memory_history = self.memory_history[-10:]
        
        self.last_cleanup = time.time()
    
    def _start_monitoring(self):
        """Start background memory monitoring"""
        def monitor_loop():
            while True:
                try:
                    current_time = time.time()
                    if current_time - self.last_cleanup > self.cleanup_interval:
                        self.check_memory_usage()
                        self.last_cleanup = current_time
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Memory monitor error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(
            target=monitor_loop, 
            daemon=True, 
            name="memory-monitor"
        )
        monitor_thread.start()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        try:
            if not PSUTIL_AVAILABLE:
                return {"available": False, "reason": "psutil not installed"}
            
            memory_info = psutil.Process().memory_info()
            
            with self._lock:
                return {
                    "current_mb": memory_info.rss / 1024 / 1024,
                    "threshold_mb": self.threshold_mb,
                    "active_objects": len(self.active_objects),
                    "history_points": len(self.memory_history),
                    "last_cleanup": self.last_cleanup,
                    "within_limits": memory_info.rss / 1024 / 1024 < self.threshold_mb
                }
        except Exception as e:
            return {"error": str(e), "available": False}

class CircuitBreakerRegistry:
    """Centralized circuit breaker management for preventing cascade failures"""
    
    def __init__(self):
        self.breakers: Dict[str, 'CircuitBreaker'] = {}
        self.global_failure_count = 0
        self.emergency_mode = False
        self.emergency_threshold = 10
        self._lock = threading.RLock()
        
        # Reset global failure count periodically
        self._start_reset_thread()
    
    def get_breaker(self, service_name: str, failure_threshold: int = 5, 
                   recovery_timeout: int = 60) -> 'CircuitBreaker':
        """Get or create circuit breaker for service"""
        with self._lock:
            if service_name not in self.breakers:
                self.breakers[service_name] = CircuitBreaker(
                    service_name, failure_threshold, recovery_timeout, self
                )
            return self.breakers[service_name]
    
    def record_global_failure(self):
        """Record a global system failure"""
        with self._lock:
            self.global_failure_count += 1
            if self.global_failure_count >= self.emergency_threshold:
                self.emergency_mode = True
                logger.critical("EMERGENCY MODE ACTIVATED - Too many global failures")
    
    def reset_emergency_mode(self):
        """Reset emergency mode"""
        with self._lock:
            self.emergency_mode = False
            self.global_failure_count = 0
            logger.info("Emergency mode deactivated")
    
    def _start_reset_thread(self):
        """Start thread to periodically reset failure counts"""
        def reset_loop():
            while True:
                try:
                    time.sleep(300)  # 5 minutes
                    with self._lock:
                        # Decay global failure count
                        self.global_failure_count = max(0, self.global_failure_count - 1)
                        
                        # If failures are low, exit emergency mode
                        if self.emergency_mode and self.global_failure_count < 3:
                            self.reset_emergency_mode()
                            
                except Exception as e:
                    logger.error(f"Circuit breaker reset thread error: {e}")
        
        reset_thread = threading.Thread(
            target=reset_loop, 
            daemon=True, 
            name="circuit-breaker-reset"
        )
        reset_thread.start()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        with self._lock:
            return {
                "emergency_mode": self.emergency_mode,
                "global_failure_count": self.global_failure_count,
                "breakers": {
                    name: breaker.get_status() 
                    for name, breaker in self.breakers.items()
                }
            }

class CircuitBreaker:
    """Individual circuit breaker for service protection"""
    
    def __init__(self, service_name: str, failure_threshold: int = 5, 
                 recovery_timeout: int = 60, registry: CircuitBreakerRegistry = None):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.registry = registry
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.success_count_in_half_open = 0
        self._lock = threading.RLock()
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        with self._lock:
            if self.state == "CLOSED":
                return True
            elif self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.success_count_in_half_open = 0
                    logger.info(f"Circuit breaker {self.service_name} entering HALF_OPEN state")
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def record_success(self):
        """Record successful execution"""
        with self._lock:
            if self.state == "HALF_OPEN":
                self.success_count_in_half_open += 1
                if self.success_count_in_half_open >= 3:  # 3 successes to close
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.service_name} returning to CLOSED state")
            elif self.state == "CLOSED":
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed execution"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.registry:
                self.registry.record_global_failure()
            
            if self.state == "HALF_OPEN":
                # Failure in half-open goes back to open
                self.state = "OPEN"
                logger.warning(f"Circuit breaker {self.service_name} returning to OPEN state")
            elif self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker {self.service_name} OPENED due to {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from OPEN to HALF_OPEN"""
        if not self.last_failure_time:
            return False
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        with self._lock:
            return {
                "service": self.service_name,
                "state": self.state,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "last_failure": self.last_failure_time,
                "can_execute": self.can_execute()
            }

class CorrelationTracker:
    """Track requests across all system components with correlation IDs"""
    
    def __init__(self):
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_metrics = defaultdict(list)
        self._lock = threading.RLock()
        
        # Cleanup old requests periodically
        self._start_cleanup_thread()
    
    def start_request(self, correlation_id: str = None, operation: str = "unknown") -> str:
        """Start tracking a new request"""
        if not correlation_id:
            correlation_id = f"req_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        with self._lock:
            self.active_requests[correlation_id] = {
                'operation': operation,
                'start_time': time.time(),
                'components': [],
                'errors': [],
                'metadata': {}
            }
        
        return correlation_id
    
    def add_component(self, correlation_id: str, component: str, 
                     duration: float = None, success: bool = True, 
                     metadata: Dict[str, Any] = None):
        """Add component execution to request trace"""
        with self._lock:
            if correlation_id in self.active_requests:
                component_info = {
                    'component': component,
                    'timestamp': time.time(),
                    'success': success,
                    'duration': duration,
                    'metadata': metadata or {}
                }
                
                self.active_requests[correlation_id]['components'].append(component_info)
                
                if not success:
                    error_info = {
                        'component': component,
                        'timestamp': time.time(),
                        'metadata': metadata or {}
                    }
                    self.active_requests[correlation_id]['errors'].append(error_info)
    
    def finish_request(self, correlation_id: str, success: bool = True, 
                      result_metadata: Dict[str, Any] = None):
        """Finish tracking a request"""
        with self._lock:
            if correlation_id in self.active_requests:
                request_data = self.active_requests[correlation_id]
                total_duration = time.time() - request_data['start_time']
                
                # Store metrics
                metrics = {
                    'correlation_id': correlation_id,
                    'operation': request_data['operation'],
                    'total_duration': total_duration,
                    'components_count': len(request_data['components']),
                    'errors_count': len(request_data['errors']),
                    'success': success,
                    'timestamp': time.time(),
                    'result_metadata': result_metadata or {}
                }
                
                operation = request_data['operation']
                self.request_metrics[operation].append(metrics)
                
                # Keep only last 1000 metrics per operation
                if len(self.request_metrics[operation]) > 1000:
                    self.request_metrics[operation] = self.request_metrics[operation][-1000:]
                
                # Remove from active requests
                del self.active_requests[correlation_id]
                
                return metrics
        
        return None
    
    def get_request_status(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of active request"""
        with self._lock:
            return self.active_requests.get(correlation_id)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of request metrics"""
        with self._lock:
            summary = {}
            
            for operation, metrics_list in self.request_metrics.items():
                if not metrics_list:
                    continue
                
                durations = [m['total_duration'] for m in metrics_list]
                success_rate = sum(1 for m in metrics_list if m['success']) / len(metrics_list)
                
                summary[operation] = {
                    'total_requests': len(metrics_list),
                    'success_rate': success_rate,
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'recent_errors': sum(1 for m in metrics_list[-100:] if not m['success'])
                }
            
            return {
                'operations': summary,
                'active_requests': len(self.active_requests),
                'total_operations': len(self.request_metrics)
            }
    
    def _start_cleanup_thread(self):
        """Start thread to cleanup old requests"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(600)  # 10 minutes
                    current_time = time.time()
                    
                    with self._lock:
                        # Remove requests older than 1 hour
                        expired_requests = [
                            req_id for req_id, req_data in self.active_requests.items()
                            if current_time - req_data['start_time'] > 3600
                        ]
                        
                        for req_id in expired_requests:
                            logger.warning(f"Cleaning up expired request: {req_id}")
                            del self.active_requests[req_id]
                            
                except Exception as e:
                    logger.error(f"Correlation tracker cleanup error: {e}")
        
        cleanup_thread = threading.Thread(
            target=cleanup_loop, 
            daemon=True, 
            name="correlation-cleanup"
        )
        cleanup_thread.start()

class ResourceLimiter:
    """Prevent resource exhaustion with configurable limits"""
    
    def __init__(self, max_concurrent=5, queue_size=100, memory_limit_mb=4096):
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        self.memory_limit_mb = memory_limit_mb
        
        self.active_requests = 0
        self.queue = []
        self.rejected_count = 0
        self.total_processed = 0
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
        self._thread_lock = threading.RLock()
    
    async def acquire(self, operation_id: str = None) -> bool:
        """Acquire resources for operation"""
        try:
            # Check memory before acquiring
            if not self._check_memory_available():
                self.rejected_count += 1
                return False
            
            # Try to acquire semaphore (non-blocking check)
            if self._semaphore.locked() and len(self.queue) >= self.queue_size:
                self.rejected_count += 1
                return False
            
            await self._semaphore.acquire()
            
            with self._thread_lock:
                self.active_requests += 1
                if operation_id:
                    logger.debug(f"Resource acquired for operation: {operation_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Resource acquisition failed: {e}")
            return False
    
    def release(self, operation_id: str = None):
        """Release resources"""
        try:
            self._semaphore.release()
            
            with self._thread_lock:
                self.active_requests = max(0, self.active_requests - 1)
                self.total_processed += 1
                if operation_id:
                    logger.debug(f"Resource released for operation: {operation_id}")
                    
        except Exception as e:
            logger.error(f"Resource release failed: {e}")
    
    @contextmanager
    async def resource_context(self, operation_id: str = None):
        """Context manager for resource acquisition/release"""
        acquired = await self.acquire(operation_id)
        if not acquired:
            raise ResourceExhaustionError("Unable to acquire resources")
        
        try:
            yield
        finally:
            self.release(operation_id)
    
    def _check_memory_available(self) -> bool:
        """Check if memory is available"""
        try:
            if not PSUTIL_AVAILABLE:
                return True
            
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            return memory_mb < self.memory_limit_mb
            
        except Exception:
            return True  # Don't block if check fails
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource limiter statistics"""
        with self._thread_lock:
            return {
                "active_requests": self.active_requests,
                "max_concurrent": self.max_concurrent,
                "queue_size_limit": self.queue_size,
                "rejected_count": self.rejected_count,
                "total_processed": self.total_processed,
                "utilization": self.active_requests / self.max_concurrent
            }

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class ResourceExhaustionError(Exception):
    """Raised when system resources are exhausted"""
    pass

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass

class SystemOverloadError(Exception):
    """Raised when system is overloaded"""
    pass

# ============================================================================
# UPDATED CORE CLASSES WITH PRODUCTION FIXES
# ============================================================================

class MCPToolsManager:
    """Core MCP Tools Manager - PRODUCTION HARDENED"""
    
    def __init__(self, db_engine: Engine):
        self.db_engine = db_engine
        
        # Initialize monitoring and protection systems
        self.memory_monitor = MemoryMonitor()
        self.circuit_registry = CircuitBreakerRegistry()
        self.correlation_tracker = CorrelationTracker()
        
        # Initialize storage paths
        self.storage_path = Path("storage")
        self.knowledge_db_path = self.storage_path / "knowledge" / "oceanographic.db"
        
        # Create storage directories
        self.storage_path.mkdir(exist_ok=True)
        self.knowledge_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components with protection
        self._setup_core_components()
        
        # Register for memory monitoring
        self.memory_monitor.register_object(self)
        
        logger.info("PRODUCTION-HARDENED MCP Tools Manager initialized")
    
    def _setup_core_components(self):
        """Setup core components with circuit breaker protection"""
        db_breaker = self.circuit_registry.get_breaker("database_connection")
        
        try:
            if db_breaker.can_execute():
                with self.db_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                db_breaker.record_success()
                self.db_available = True
            else:
                logger.warning("Database circuit breaker is OPEN")
                self.db_available = False
                
        except Exception as e:
            db_breaker.record_failure()
            logger.error(f"Database connection failed: {e}")
            self.db_available = False
    
    def get_database_connection(self) -> Engine:
        """Get database connection with circuit breaker protection"""
        db_breaker = self.circuit_registry.get_breaker("database_connection")
        
        if not db_breaker.can_execute():
            raise CircuitBreakerOpenError("Database circuit breaker is OPEN")
        
        try:
            # Test connection
            with self.db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            db_breaker.record_success()
            return self.db_engine
            
        except Exception as e:
            db_breaker.record_failure()
            raise
    
    def is_ready(self) -> bool:
        """Check if tools manager is ready with health checks"""
        return (
            self.db_available and 
            self.memory_monitor.check_memory_usage() and 
            not self.circuit_registry.emergency_mode
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            "db_available": self.db_available,
            "memory_stats": self.memory_monitor.get_memory_stats(),
            "circuit_breakers": self.circuit_registry.get_status(),
            "correlation_metrics": self.correlation_tracker.get_metrics_summary(),
            "emergency_mode": self.circuit_registry.emergency_mode,
            "ready": self.is_ready()
        }

class DatabaseExplorerTool:
    """Database exploration tool - PRODUCTION HARDENED"""
    
    def __init__(self, db_engine: Engine, tools_manager: MCPToolsManager):
        self.db_engine = db_engine
        self.tools_manager = tools_manager
        
        # Get monitoring components from tools manager
        self.circuit_registry = tools_manager.circuit_registry
        self.correlation_tracker = tools_manager.correlation_tracker
        self.memory_monitor = tools_manager.memory_monitor
        
        # Cache for schema information with TTL
        self.schema_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self._cache_lock = threading.RLock()
        
        # Register for memory monitoring
        self.memory_monitor.register_object(self)
        
        logger.info("PRODUCTION-HARDENED Database Explorer Tool initialized")
    
    def explore_database_schema(self, table_name: str = None, 
                               include_sample_data: bool = False,
                               correlation_id: str = None) -> str:
        """Explore database schema with production-grade protection"""
        
        # Start correlation tracking
        if not correlation_id:
            correlation_id = self.correlation_tracker.start_request("schema_exploration")
        
        start_time = time.time()
        
        try:
            # Check circuit breaker
            db_breaker = self.circuit_registry.get_breaker("database_schema")
            if not db_breaker.can_execute():
                return "Database schema exploration temporarily unavailable (circuit breaker OPEN)"
            
            # Check memory
            if not self.memory_monitor.check_memory_usage():
                return "Schema exploration temporarily limited due to high memory usage"
            
            # Check cache first
            cache_key = f"schema_{table_name or 'all'}_{include_sample_data}"
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result:
                self.correlation_tracker.add_component(
                    correlation_id, "schema_cache", time.time() - start_time, True
                )
                db_breaker.record_success()
                return cached_result
            
            # Execute schema query
            result = self._execute_schema_query(table_name, include_sample_data)
            
            # Cache successful result
            self._set_cache(cache_key, result)
            
            # Record success
            duration = time.time() - start_time
            self.correlation_tracker.add_component(
                correlation_id, "schema_query", duration, True, 
                {"table_name": table_name, "include_sample": include_sample_data}
            )
            db_breaker.record_success()
            
            return result
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            self.correlation_tracker.add_component(
                correlation_id, "schema_query", duration, False, 
                {"error": str(e), "table_name": table_name}
            )
            
            db_breaker = self.circuit_registry.get_breaker("database_schema")
            db_breaker.record_failure()
            
            error_msg = f"Schema exploration failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _execute_schema_query(self, table_name: str, include_sample_data: bool) -> str:
        """Execute schema query with connection management"""
        
        # Use tools manager's protected connection
        with self.tools_manager.get_database_connection().connect() as conn:
            if table_name:
                return self._get_table_schema(conn, table_name, include_sample_data)
            else:
                return self._get_all_tables(conn)
    
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Get from cache with TTL check"""
        with self._cache_lock:
            if cache_key in self.schema_cache:
                cache_entry = self.schema_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    return cache_entry['data']
                else:
                    del self.schema_cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, data: str):
        """Set cache with cleanup"""
        with self._cache_lock:
            # Simple cache cleanup - remove oldest entries if cache is too large
            if len(self.schema_cache) > 100:
                oldest_key = min(
                    self.schema_cache.keys(), 
                    key=lambda k: self.schema_cache[k]['timestamp']
                )
                del self.schema_cache[oldest_key]
            
            self.schema_cache[cache_key] = {
                'data': data,
                'timestamp': time.time()
            }
    
    def _get_table_schema(self, conn, table_name: str, include_sample_data: bool) -> str:
        """Get schema for specific table with validation"""
        
        # Validate table name to prevent injection
        if not self._is_valid_table_name(table_name):
            return f"Invalid table name: {table_name}"
        
        schema_query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = :table_name
        ORDER by ordinal_position
        """
        
        result = conn.execute(text(schema_query), {'table_name': table_name})
        columns = result.fetchall()
        
        if not columns:
            return f"Table '{table_name}' not found or no columns available"
        
        schema_info = f"Schema for table '{table_name}':\n"
        schema_info += "-" * 40 + "\n"
        
        for col in columns:
            nullable = "NULL" if col[2] == "YES" else "NOT NULL"
            default = f" DEFAULT {col[3]}" if col[3] else ""
            schema_info += f"  {col[0]} ({col[1]}) {nullable}{default}\n"
        
        # Add sample data if requested
        if include_sample_data:
            sample_data = self._get_safe_sample_data(conn, table_name)
            if sample_data:
                schema_info += f"\nSample data:\n{sample_data}"
        
        return schema_info
    
    def _get_all_tables(self, conn) -> str:
        """Get all available tables"""
        
        tables_query = """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER by table_name
        """
        
        result = conn.execute(text(tables_query))
        tables = result.fetchall()
        
        if not tables:
            return "No tables found in the database"
        
        tables_info = "Available tables in database:\n"
        tables_info += "=" * 40 + "\n"
        
        for table in tables:
            tables_info += f"  {table[0]} ({table[1]})\n"
        
        tables_info += f"\nTotal tables: {len(tables)}\n"
        
        return tables_info
    
    def _is_valid_table_name(self, table_name: str) -> bool:
        """Validate table name to prevent SQL injection"""
        import re
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name))
    
    def _get_safe_sample_data(self, conn, table_name: str) -> Optional[str]:
        """Get sample data safely"""
        try:
            sample_query = f"SELECT * FROM {table_name} LIMIT 3"
            result = conn.execute(text(sample_query))
            rows = result.fetchall()
            
            if not rows:
                return None
            
            sample_lines = []
            for i, row in enumerate(rows):
                sample_lines.append(f"Row {i+1}: {dict(row)}")
            
            return "\n".join(sample_lines)
            
        except Exception as e:
            logger.warning(f"Sample data query failed: {e}")
            return f"Sample data unavailable: {str(e)}"

# CrewAI availability check - SINGLE source for this flag
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logging.warning("CrewAI not available - agent system will use fallback implementations")