# src/services/llm_enhanced_sql_generator.py
"""
CLEAN LLM-Enhanced SQL Generator - Production Implementation
Fixed async/sync issues, removed redundant validation, simplified architecture
"""

import re
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Import required components
try:
    from services.template_sql_generator import ProductionSQLGenerator, GeneratedSQL
    from services.oceanographic_intelligence_engine import (
        QueryIntent, ComplexityLevel, QueryClassification, OceanographicContext
    )
except ImportError:
    # Clean fallback without duplicating classes
    logging.warning("Core dependencies not available - running in limited mode")
    ProductionSQLGenerator = None
    QueryClassification = None

logger = logging.getLogger(__name__)

@dataclass
class LLMEnhancementResult:
    """Clean result structure for LLM enhancement"""
    enhanced_sql: str
    enhancement_applied: bool
    enhancement_reasoning: str
    llm_response_time: float
    confidence_score: float
    context_utilized: bool = False

class DeepSeekSQLService:
    """Clean DeepSeek integration - removed redundant validation layers"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        if not OpenAI:
            raise ImportError("OpenAI package required for DeepSeek integration")
        
        # Handle OpenAI client initialization with compatibility issues
        try:
            # Try minimal initialization first
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            logger.info("DeepSeek SQL service initialized successfully")
        except Exception as e:
            logger.error(f"DeepSeek client initialization failed: {e}")
            # Try alternative initialization
            try:
                self.client = OpenAI(api_key=api_key)
                logger.warning("DeepSeek client initialized without base_url - may not work properly")
            except Exception as e2:
                logger.error(f"Alternative DeepSeek initialization also failed: {e2}")
                raise ImportError(f"Cannot initialize DeepSeek client: {e}")
        
        self.model = model
        self.timeout = 15  # Simplified timeout
        self.max_tokens = 800
        self.temperature = 0.0
    
    async def generate_sql_from_context(self, query: str, vector_context: str, 
                                      classification) -> LLMEnhancementResult:
        """Generate SQL using context - simplified and cleaned"""
        
        start_time = time.time()
        
        try:
            # Build focused prompt
            prompt = self._build_prompt(query, vector_context, classification)
            
            # Single LLM call with proper async
            response = await self._call_deepseek(prompt)
            
            # Parse response
            sql_result = self._parse_response(response)
            
            # Basic validation only
            is_valid = self._basic_validate(sql_result['sql'])
            
            response_time = time.time() - start_time
            
            return LLMEnhancementResult(
                enhanced_sql=sql_result['sql'],
                enhancement_applied=is_valid,
                enhancement_reasoning=sql_result.get('reasoning', 'LLM generation'),
                llm_response_time=response_time,
                confidence_score=0.8 if is_valid else 0.0,
                context_utilized=len(vector_context) > 100
            )
            
        except Exception as e:
            logger.warning(f"LLM SQL generation failed: {e}")
            response_time = time.time() - start_time
            
            return LLMEnhancementResult(
                enhanced_sql="",
                enhancement_applied=False,
                enhancement_reasoning=f"Generation failed: {str(e)}",
                llm_response_time=response_time,
                confidence_score=0.0
            )
    
    def _build_prompt(self, query: str, vector_context: str, classification) -> str:
        """Clean, focused prompt building"""
        
        return f"""You are an expert SQL generator for oceanographic ARGO float databases.

USER QUERY: "{query}"

DATABASE CONTEXT:
{vector_context}

CRITICAL REQUIREMENTS:
1. Use ONLY these exact table names: argo_profiles, argo_measurements
2. Use ONLY these exact column names from argo_profiles:
   - surface_temp (NOT temperature)
   - surface_salinity (NOT salinity) 
   - platform_number, profile_date, latitude, longitude
   - mixed_layer_depth, max_pressure, n_levels
3. Use ONLY these exact column names from argo_measurements:
   - temperature, salinity, pressure, depth, profile_id
4. For surface data: Use argo_profiles table only with surface_temp, surface_salinity
5. For depth profiles: JOIN argo_profiles p INNER JOIN argo_measurements m ON p.id = m.profile_id
6. Always include WHERE clauses for safety and performance
7. Add LIMIT clause (typically 100-5000)
8. Handle NULL values with IS NOT NULL checks
9. Use proper geographic bounds for Indian Ocean queries

IMPORTANT: For surface temperature queries, use surface_temp column from argo_profiles table.

Generate PostgreSQL SQL that directly answers the user's question.

Respond with JSON:
{{
  "sql": "your SQL query here",
  "reasoning": "brief explanation of approach",
  "confidence": 0.85
}}"""

    async def _call_deepseek(self, prompt: str) -> Dict[str, Any]:
        """Single, clean API call"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert SQL generator. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON in response")
    
    def _parse_response(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Clean response parsing"""
        
        sql = llm_response.get('sql', '').strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        return {
            'sql': sql,
            'reasoning': llm_response.get('reasoning', 'SQL generated'),
            'confidence': llm_response.get('confidence', 0.5)
        }
    
    def _basic_validate(self, sql: str) -> bool:
        """Simple, essential validation only"""
        
        if not sql:
            return False
        
        sql_lower = sql.lower().strip()
        
        # Must be SELECT query
        if not sql_lower.startswith('select'):
            return False
        
        # Must have FROM
        if 'from' not in sql_lower:
            return False
        
        # No dangerous operations
        dangerous = ['drop', 'delete', 'truncate', 'alter', 'update']
        if any(op in sql_lower for op in dangerous):
            return False
        
        return True

class LLMEnhancedSQLGenerator:
    """Clean LLM-Enhanced SQL Generator - removed complexity"""
    
    def __init__(self, deepseek_api_key: str):
        # Initialize base generator
        if ProductionSQLGenerator:
            self.base_generator = ProductionSQLGenerator()
        else:
            self.base_generator = None
        
        # Initialize DeepSeek service
        try:
            self.deepseek_service = DeepSeekSQLService(deepseek_api_key)
            self.llm_available = True
            logger.info("LLM-Enhanced SQL Generator initialized")
        except Exception as e:
            logger.warning(f"DeepSeek initialization failed: {e}")
            self.deepseek_service = None
            self.llm_available = False
        
        # Simple stats
        self.stats = {
            'total_queries': 0,
            'llm_used': 0,
            'fallback_used': 0
        }
    
    async def generate_sql_enhanced(self, classification, query_text: str, 
                                  vector_context: str = "") -> GeneratedSQL:
        """Main generation method - clean and simple"""
        
        self.stats['total_queries'] += 1
        
        # Try LLM enhancement first
        if self.llm_available and self._should_use_llm(query_text):
            try:
                enhancement_result = await self.deepseek_service.generate_sql_from_context(
                    query_text, vector_context, classification
                )
                
                if enhancement_result.enhancement_applied:
                    self.stats['llm_used'] += 1
                    return self._create_enhanced_result(enhancement_result, query_text)
                
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")
        
        # Fallback to base generator
        self.stats['fallback_used'] += 1
        
        if self.base_generator:
            return self.base_generator.generate_sql(classification, query_text)
        else:
            return self._create_emergency_fallback(query_text)
    
    def _should_use_llm(self, query_text: str) -> bool:
        """Intelligent decision logic for LLM usage"""
        
        query_lower = query_text.lower()
        
        # Use LLM for complex analytical queries
        complex_indicators = [
            'calculate', 'analyze', 'compare', 'correlate',
            'average', 'maximum', 'minimum', 'trend',
            'gradient', 'anomaly', 'pattern', 'seasonal',
            'variability', 'distribution', 'relationship'
        ]
        
        # Use LLM for queries that need custom logic
        custom_logic_indicators = [
            'between', 'different regions', 'strong vertical mixing',
            'maximum depth reached', 'areas with', 'find all'
        ]
        
        # Use LLM for statistical queries
        statistical_indicators = [
            'average', 'mean', 'median', 'standard deviation',
            'variance', 'percentile', 'statistics'
        ]
        
        # Check if query needs LLM intelligence
        needs_llm = (
            any(indicator in query_lower for indicator in complex_indicators) or
            any(indicator in query_lower for indicator in custom_logic_indicators) or
            any(indicator in query_lower for indicator in statistical_indicators)
        )
        
        # Don't use LLM for simple queries that templates handle well
        simple_indicators = [
            'show me', 'latest', 'recent', 'count', 'total number'
        ]
        
        is_simple = any(indicator in query_lower for indicator in simple_indicators)
        
        return needs_llm and not is_simple
    
    def _create_enhanced_result(self, enhancement_result: LLMEnhancementResult, 
                              query_text: str) -> GeneratedSQL:
        """Create GeneratedSQL from LLM result"""
        
        return GeneratedSQL(
            sql=enhancement_result.enhanced_sql,
            template_id="llm_enhanced",
            parameters_used={'llm_confidence': enhancement_result.confidence_score},
            estimated_performance="good",
            recommended_timeout=30,
            index_requirements=[],
            adaptations_made=[enhancement_result.enhancement_reasoning],
            validation_passed=True,
            warnings=[]
        )
    
    def _create_emergency_fallback(self, query_text: str) -> GeneratedSQL:
        """Ultra-safe emergency fallback"""
        
        safe_sql = """
        SELECT platform_number, profile_date, latitude, longitude, surface_temp
        FROM argo_profiles 
        WHERE profile_date >= NOW() - INTERVAL '1 year'
        AND surface_temp IS NOT NULL
        ORDER BY profile_date DESC 
        LIMIT 100;
        """
        
        return GeneratedSQL(
            sql=safe_sql,
            template_id='emergency_fallback',
            parameters_used={'reason': 'all_systems_failed'},
            estimated_performance='fast',
            recommended_timeout=15,
            index_requirements=['idx_profiles_date'],
            adaptations_made=['Emergency fallback - safe query used'],
            validation_passed=True,
            warnings=['Emergency fallback used - query may not match request exactly']
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simple statistics"""
        
        total = max(self.stats['total_queries'], 1)
        return {
            'total_queries': self.stats['total_queries'],
            'llm_usage_rate': (self.stats['llm_used'] / total) * 100,
            'fallback_rate': (self.stats['fallback_used'] / total) * 100,
            'llm_available': self.llm_available
        }

# Clean integration helper
def integrate_llm_enhanced_sql_generator(rag_system, deepseek_api_key: str):
    """
    Simplified integration helper
    """
    
    enhanced_generator = LLMEnhancedSQLGenerator(deepseek_api_key)
    
    # Store reference
    rag_system.enhanced_sql_generator = enhanced_generator
    rag_system.llm_enhancement_enabled = enhanced_generator.llm_available
    
    # Replace the SQL generation method
    original_method = getattr(rag_system, '_generate_sql_with_intelligence', None)
    
    async def enhanced_generate_sql(classification, query_text, domain_context):
        """Enhanced async SQL generation"""
        
        if rag_system.llm_enhancement_enabled:
            try:
                return await rag_system.enhanced_sql_generator.generate_sql_enhanced(
                    classification, query_text, domain_context
                )
            except Exception as e:
                logger.warning(f"Enhanced generation failed: {e}")
        
        # Fallback to original method
        if original_method:
            return original_method(classification, query_text, domain_context)
        else:
            # Ultimate fallback
            return rag_system.enhanced_sql_generator._create_emergency_fallback(query_text)
    
    rag_system._generate_sql_with_intelligence = enhanced_generate_sql
    
    logger.info("LLM-Enhanced SQL Generator integrated successfully")
    return enhanced_generator