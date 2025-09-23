# src/services/llm_enhanced_sql_generator.py
"""
LLM-Enhanced SQL Generator - Production Implementation
Integrates with existing vector-context-aware architecture while adding intelligent SQL generation
"""

import re
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .template_sql_generator import ProductionSQLGenerator, SQLTemplate, GeneratedSQL
from .oceanographic_intelligence_engine import QueryClassification, OceanographicContext

logger = logging.getLogger(__name__)

@dataclass
class LLMEnhancementResult:
    """Result of LLM enhancement process"""
    enhanced_components: Dict[str, str]
    enhancement_applied: bool
    enhancement_reasoning: str
    llm_response_time: float
    fallback_used: bool
    confidence_score: float

class DeepSeekSQLService:
    """Production-grade DeepSeek integration for SQL component enhancement"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        if not OpenAI:
            raise ImportError("OpenAI package required for DeepSeek integration")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model
        
        # Production settings
        self.max_retries = 3
        self.timeout = 15
        self.rate_limit_delay = 0.5
        
        # Token management
        self.max_tokens = 1500
        self.temperature = 0.1  # Low temperature for SQL generation
        
        logger.info(f"DeepSeek SQL service initialized with model: {model}")
    
    async def enhance_sql_components(self, 
                                   query: str,
                                   classification: QueryClassification,
                                   vector_context: str,
                                   base_components: Dict[str, str],
                                   template_info: Dict[str, Any]) -> LLMEnhancementResult:
        """
        Enhance SQL components using DeepSeek while maintaining safety constraints
        """
        
        start_time = time.time()
        
        try:
            # Build intelligent prompt with context
            prompt = self._build_enhancement_prompt(
                query, classification, vector_context, base_components, template_info
            )
            
            # Make LLM call with retries
            llm_response = await self._call_deepseek_with_retry(prompt)
            
            # Parse and validate response
            enhanced_components = self._parse_llm_response(llm_response, base_components)
            
            # Validate against safety constraints
            validation_result = self._validate_enhanced_components(
                enhanced_components, base_components, template_info
            )
            
            response_time = time.time() - start_time
            
            return LLMEnhancementResult(
                enhanced_components=validation_result['components'],
                enhancement_applied=validation_result['valid'],
                enhancement_reasoning=llm_response.get('reasoning', 'LLM enhancement applied'),
                llm_response_time=response_time,
                fallback_used=not validation_result['valid'],
                confidence_score=validation_result['confidence']
            )
            
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            response_time = time.time() - start_time
            
            return LLMEnhancementResult(
                enhanced_components=base_components,
                enhancement_applied=False,
                enhancement_reasoning=f"Enhancement failed: {str(e)}",
                llm_response_time=response_time,
                fallback_used=True,
                confidence_score=0.0
            )
    
    def _build_enhancement_prompt(self, 
                                query: str,
                                classification: QueryClassification,
                                vector_context: str,
                                base_components: Dict[str, str],
                                template_info: Dict[str, Any]) -> str:
        """
        Build intelligent prompt for SQL component enhancement
        """
        
        prompt = f"""You are an expert oceanographic database specialist. Enhance SQL query components based on the user's natural language query while strictly following the provided template structure and database schema.

USER QUERY: "{query}"

QUERY CLASSIFICATION:
- Intent: {classification.intent.value}
- Complexity: {classification.complexity.value}
- Parameters: {classification.context.parameters}
- Confidence: {classification.confidence}

DATABASE SCHEMA CONTEXT:
{vector_context}

TEMPLATE STRUCTURE: {template_info.get('template_id', 'unknown')}
Required components: {list(base_components.keys())}

CURRENT RULE-BASED COMPONENTS:
{json.dumps(base_components, indent=2)}

ENHANCEMENT INSTRUCTIONS:
1. Improve component accuracy based on query intent and domain knowledge
2. Add missing spatial/temporal filters if implied by query
3. Enhance parameter selection based on oceanographic context
4. Maintain template structure and safety constraints
5. Use only tables: argo_profiles (p), argo_measurements (m)
6. Ensure JOIN consistency if measurements table referenced

RESPONSE FORMAT (JSON):
{{
  "enhanced_components": {{
    "select_columns": "enhanced column selection",
    "spatial_filters": "enhanced spatial filters",
    "temporal_filters": "enhanced temporal filters",
    "parameter_filters": "enhanced parameter filters",
    "ordering": "enhanced ordering",
    "limit": "appropriate limit"
  }},
  "reasoning": "Brief explanation of enhancements made",
  "confidence": 0.85
}}

Only enhance components that need improvement. Return original component if no enhancement needed."""

        return prompt
    
    async def _call_deepseek_with_retry(self, prompt: str) -> Dict[str, Any]:
        """
        Call DeepSeek API with retry logic and error handling
        """
        
        for attempt in range(self.max_retries):
            try:
                # Add rate limiting delay
                if attempt > 0:
                    await asyncio.sleep(self.rate_limit_delay * attempt)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert SQL generator for oceanographic databases. Always respond with valid JSON."
                        },
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
                    # Extract JSON from response if wrapped in other text
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        raise ValueError("No valid JSON in response")
                
            except Exception as e:
                logger.warning(f"DeepSeek call attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise e
        
        raise RuntimeError("All DeepSeek API attempts failed")
    
    def _parse_llm_response(self, 
                           llm_response: Dict[str, Any], 
                           base_components: Dict[str, str]) -> Dict[str, str]:
        """
        Parse and merge LLM response with base components
        """
        
        enhanced_components = base_components.copy()
        
        if 'enhanced_components' in llm_response:
            llm_components = llm_response['enhanced_components']
            
            # Only update components that exist in base and are properly enhanced
            for key, value in llm_components.items():
                if key in base_components and value and value != base_components[key]:
                    enhanced_components[key] = value
        
        return enhanced_components
    
    def _validate_enhanced_components(self, 
                                    enhanced_components: Dict[str, str],
                                    base_components: Dict[str, str],
                                    template_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate enhanced components against safety constraints
        """
        
        validation_issues = []
        confidence = 1.0
        
        # Check for dangerous SQL patterns
        dangerous_patterns = [
            r';\s*(drop|delete|truncate|alter)',
            r'union\s+select',
            r'--',
            r'/\*.*\*/',
            r'xp_cmdshell',
            r'sp_executesql'
        ]
        
        all_components_text = ' '.join(enhanced_components.values()).lower()
        
        for pattern in dangerous_patterns:
            if re.search(pattern, all_components_text, re.IGNORECASE):
                validation_issues.append(f"Dangerous SQL pattern detected: {pattern}")
                confidence = 0.0
        
        # Check table reference consistency
        has_measurements_reference = any('m.' in comp for comp in enhanced_components.values())
        has_join_clause = any('join' in comp.lower() for comp in enhanced_components.values())
        
        if has_measurements_reference and not has_join_clause:
            # Check if template handles JOIN automatically
            template_has_join = template_info.get('requires_measurements_join', False)
            if not template_has_join:
                validation_issues.append("References measurements table without JOIN")
                confidence *= 0.5
        
        # Check for empty critical components
        critical_components = ['select_columns', 'ordering', 'limit']
        for comp in critical_components:
            if comp in enhanced_components and not enhanced_components[comp].strip():
                validation_issues.append(f"Empty critical component: {comp}")
                confidence *= 0.7
        
        # Validate limit values
        if 'limit' in enhanced_components:
            try:
                limit_val = int(enhanced_components['limit'])
                if limit_val <= 0 or limit_val > 100000:
                    validation_issues.append(f"Invalid limit value: {limit_val}")
                    enhanced_components['limit'] = base_components.get('limit', '5000')
                    confidence *= 0.8
            except ValueError:
                validation_issues.append("Invalid limit format")
                enhanced_components['limit'] = base_components.get('limit', '5000')
                confidence *= 0.8
        
        # If significant validation issues, fall back to base components
        if confidence < 0.5:
            logger.warning(f"LLM enhancement validation failed: {validation_issues}")
            return {
                'components': base_components,
                'valid': False,
                'confidence': confidence,
                'issues': validation_issues
            }
        
        return {
            'components': enhanced_components,
            'valid': True,
            'confidence': confidence,
            'issues': validation_issues
        }


class LLMEnhancedSQLGenerator:
    """
    Production-grade LLM-Enhanced SQL Generator
    Extends existing ProductionSQLGenerator with intelligent component enhancement
    """
    
    def __init__(self, deepseek_api_key: str):
        # Initialize base generator
        self.base_generator = ProductionSQLGenerator()
        
        # Initialize DeepSeek service
        try:
            self.deepseek_service = DeepSeekSQLService(deepseek_api_key)
            self.llm_available = True
            logger.info("LLM-Enhanced SQL Generator initialized with DeepSeek")
        except Exception as e:
            logger.warning(f"DeepSeek initialization failed: {e}")
            self.deepseek_service = None
            self.llm_available = False
            logger.info("LLM-Enhanced SQL Generator running in fallback mode")
        
        # Performance tracking
        self.enhancement_stats = {
            'total_queries': 0,
            'enhancements_applied': 0,
            'fallback_used': 0,
            'avg_llm_response_time': 0.0,
            'success_rate': 0.0
        }
    
    async def generate_sql_enhanced(self, 
                                  classification: QueryClassification,
                                  query_text: str,
                                  vector_context: str = "") -> GeneratedSQL:
        """
        Generate SQL with LLM enhancement while maintaining safety and fallbacks
        """
        
        self.enhancement_stats['total_queries'] += 1
        start_time = time.time()
        
        try:
            # Step 1: Generate base SQL using existing rule-based system
            base_result = self.base_generator.generate_sql(classification, query_text)
            
            # Step 2: Check if LLM enhancement is beneficial and available
            should_enhance = self._should_apply_llm_enhancement(
                classification, query_text, base_result
            )
            
            if should_enhance and self.llm_available:
                # Step 3: Apply LLM enhancement
                enhancement_result = await self._apply_llm_enhancement(
                    classification, query_text, vector_context, base_result
                )
                
                if enhancement_result.enhancement_applied:
                    # Step 4: Create enhanced GeneratedSQL
                    enhanced_result = self._create_enhanced_result(
                        base_result, enhancement_result
                    )
                    
                    self.enhancement_stats['enhancements_applied'] += 1
                    logger.info(f"LLM enhancement applied successfully in {enhancement_result.llm_response_time:.2f}s")
                    
                    return enhanced_result
                else:
                    self.enhancement_stats['fallback_used'] += 1
                    logger.info("LLM enhancement failed, using base result")
            
            # Return base result (either LLM not needed or not available)
            base_result.adaptations_made.append("Base rule-based generation used")
            return base_result
            
        except Exception as e:
            logger.error(f"Enhanced SQL generation failed: {e}")
            self.enhancement_stats['fallback_used'] += 1
            
            # Ultimate fallback to base generator
            try:
                return self.base_generator.generate_sql(classification, query_text)
            except Exception as base_error:
                logger.error(f"Even base SQL generation failed: {base_error}")
                return self._create_emergency_fallback(query_text, str(e))
        
        finally:
            # Update performance statistics
            total_time = time.time() - start_time
            self._update_performance_stats(total_time)
    
    def _should_apply_llm_enhancement(self, 
                                    classification: QueryClassification,
                                    query_text: str,
                                    base_result: GeneratedSQL) -> bool:
        """
        Determine if LLM enhancement would be beneficial
        """
        
        # Skip enhancement for very simple queries
        if (classification.complexity.value == 'basic' and 
            classification.confidence > 0.8 and
            len(query_text.split()) < 8):
            return False
        
        # Skip if base generation already failed
        if not base_result.validation_passed:
            return False
        
        # Apply enhancement for:
        # 1. Complex queries
        # 2. Low confidence classifications
        # 3. Queries with unknown terms
        # 4. Advanced analysis requirements
        
        should_enhance = (
            classification.complexity.value in ['advanced', 'expert'] or
            classification.confidence < 0.7 or
            len(classification.required_calculations) > 2 or
            any(word in query_text.lower() for word in [
                'calculate', 'analyze', 'correlate', 'compare', 
                'relationship', 'pattern', 'trend', 'gradient'
            ])
        )
        
        logger.debug(f"LLM enhancement decision: {should_enhance} for query: {query_text[:50]}...")
        return should_enhance
    
    async def _apply_llm_enhancement(self,
                                   classification: QueryClassification,
                                   query_text: str,
                                   vector_context: str,
                                   base_result: GeneratedSQL) -> LLMEnhancementResult:
        """
        Apply LLM enhancement to base SQL components
        """
        
        # Extract components from base result SQL
        base_components = self._extract_components_from_sql(base_result)
        
        # Prepare template information
        template_info = {
            'template_id': base_result.template_id,
            'requires_measurements_join': 'argo_measurements m' in base_result.sql,
            'performance_notes': base_result.estimated_performance
        }
        
        # Call DeepSeek for enhancement
        enhancement_result = await self.deepseek_service.enhance_sql_components(
            query=query_text,
            classification=classification,
            vector_context=vector_context,
            base_components=base_components,
            template_info=template_info
        )
        
        return enhancement_result
    
    def _extract_components_from_sql(self, base_result: GeneratedSQL) -> Dict[str, str]:
        """
        Extract components from base SQL result for enhancement
        """
        
        sql = base_result.sql
        components = {}
        
        # Extract SELECT columns
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            components['select_columns'] = select_match.group(1).strip()
        
        # Extract WHERE conditions (spatial, temporal, parameter filters)
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER|\s+GROUP|\s+LIMIT|;|$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            components['spatial_filters'] = ''
            components['temporal_filters'] = ''
            components['parameter_filters'] = where_clause
        
        # Extract ORDER BY
        order_match = re.search(r'ORDER BY\s+(.*?)(?:\s+LIMIT|;|$)', sql, re.IGNORECASE)
        if order_match:
            components['ordering'] = order_match.group(1).strip()
        
        # Extract LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
        if limit_match:
            components['limit'] = limit_match.group(1)
        
        return components
    
    def _create_enhanced_result(self, 
                              base_result: GeneratedSQL,
                              enhancement_result: LLMEnhancementResult) -> GeneratedSQL:
        """
        Create enhanced GeneratedSQL from base result and LLM enhancement
        """
        
        # Rebuild SQL with enhanced components
        enhanced_sql = self._rebuild_sql_with_components(
            base_result.sql, enhancement_result.enhanced_components
        )
        
        # Create enhanced result
        enhanced_result = GeneratedSQL(
            sql=enhanced_sql,
            template_id=base_result.template_id + "_llm_enhanced",
            parameters_used=enhancement_result.enhanced_components,
            estimated_performance=base_result.estimated_performance,
            recommended_timeout=base_result.recommended_timeout + 5,  # Slightly longer timeout
            index_requirements=base_result.index_requirements,
            adaptations_made=base_result.adaptations_made + [
                f"LLM enhancement applied (confidence: {enhancement_result.confidence_score:.2f})",
                enhancement_result.enhancement_reasoning
            ],
            validation_passed=True,
            warnings=base_result.warnings
        )
        
        return enhanced_result
    
    def _rebuild_sql_with_components(self, 
                                   base_sql: str,
                                   enhanced_components: Dict[str, str]) -> str:
        """
        Rebuild SQL with enhanced components
        """
        
        # This is a simplified rebuild - in production you might want more sophisticated parsing
        sql = base_sql
        
        # Replace SELECT clause
        if 'select_columns' in enhanced_components:
            sql = re.sub(
                r'SELECT\s+.*?\s+FROM',
                f"SELECT {enhanced_components['select_columns']} FROM",
                sql,
                flags=re.IGNORECASE | re.DOTALL
            )
        
        # Replace ORDER BY
        if 'ordering' in enhanced_components:
            if 'ORDER BY' in sql.upper():
                sql = re.sub(
                    r'ORDER BY\s+.*?(?=\s+LIMIT|;|$)',
                    f"ORDER BY {enhanced_components['ordering']}",
                    sql,
                    flags=re.IGNORECASE
                )
            else:
                sql = sql.replace(';', f" ORDER BY {enhanced_components['ordering']};")
        
        # Replace LIMIT
        if 'limit' in enhanced_components:
            if 'LIMIT' in sql.upper():
                sql = re.sub(
                    r'LIMIT\s+\d+',
                    f"LIMIT {enhanced_components['limit']}",
                    sql,
                    flags=re.IGNORECASE
                )
            else:
                sql = sql.replace(';', f" LIMIT {enhanced_components['limit']};")
        
        return sql
    
    def _create_emergency_fallback(self, query_text: str, error: str) -> GeneratedSQL:
        """
        Create emergency fallback SQL when all else fails
        """
        
        logger.error(f"Creating emergency fallback for: {query_text}")
        
        # Ultra-safe SQL
        safe_sql = """
        SELECT platform_number, profile_date, latitude, longitude, surface_temp
        FROM argo_profiles 
        WHERE profile_date >= NOW() - INTERVAL '1 year'
        ORDER BY profile_date DESC 
        LIMIT 100;
        """
        
        return GeneratedSQL(
            sql=safe_sql,
            template_id='emergency_fallback',
            parameters_used={'emergency_reason': error},
            estimated_performance='fast',
            recommended_timeout=15,
            index_requirements=['idx_profiles_date'],
            adaptations_made=[f'Emergency fallback due to: {error}'],
            validation_passed=True,
            warnings=[f'Emergency fallback used: {error}']
        )
    
    def _update_performance_stats(self, total_time: float):
        """
        Update performance statistics
        """
        
        # Update success rate
        if self.enhancement_stats['total_queries'] > 0:
            self.enhancement_stats['success_rate'] = (
                (self.enhancement_stats['enhancements_applied'] / 
                 self.enhancement_stats['total_queries']) * 100
            )
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """
        Get LLM enhancement performance statistics
        """
        
        return {
            'total_queries': self.enhancement_stats['total_queries'],
            'enhancements_applied': self.enhancement_stats['enhancements_applied'],
            'enhancement_rate': (
                (self.enhancement_stats['enhancements_applied'] / 
                 max(self.enhancement_stats['total_queries'], 1)) * 100
            ),
            'fallback_rate': (
                (self.enhancement_stats['fallback_used'] / 
                 max(self.enhancement_stats['total_queries'], 1)) * 100
            ),
            'success_rate': self.enhancement_stats['success_rate'],
            'llm_service_available': self.llm_available
        }


# Integration helper for existing RAG system
def integrate_llm_enhanced_sql_generator(enhanced_rag_system, deepseek_api_key: str):
    """
    Helper function to integrate LLM-enhanced SQL generator into existing RAG system
    """
    
    # Create enhanced generator
    enhanced_generator = LLMEnhancedSQLGenerator(deepseek_api_key)
    
    # Replace the SQL generator in the existing RAG system
    enhanced_rag_system.enhanced_sql_generator = enhanced_generator
    
    # Modify the process_oceanographic_query method to use enhanced generation
    original_method = enhanced_rag_system.process_oceanographic_query
    
    async def enhanced_process_query(natural_language_query: str):
        """Enhanced process query method with LLM SQL generation"""
        
        # Get domain context (existing flow)
        domain_context = enhanced_rag_system._get_domain_context(natural_language_query)
        
        # Classify query (existing flow)  
        classification = enhanced_rag_system.ocean_intelligence.classify_query(natural_language_query)
        
        # Use enhanced SQL generation instead of base generator
        generated_sql = await enhanced_generator.generate_sql_enhanced(
            classification=classification,
            query_text=natural_language_query,
            vector_context=domain_context
        )
        
        # Continue with existing flow (execution, insights, etc.)
        results_df = enhanced_rag_system._execute_sql_safely(generated_sql)
        
        if results_df is None:
            return enhanced_rag_system._build_error_response(
                natural_language_query, "Enhanced SQL execution failed", {}
            )
        
        insights = enhanced_rag_system.ocean_intelligence.generate_insights(
            natural_language_query, results_df, classification
        )
        
        return {
            'success': True,
            'query': natural_language_query,
            'sql_query': generated_sql.sql,
            'template_used': generated_sql.template_id,
            'classification': enhanced_rag_system._classification_to_dict(classification),
            'results': results_df,
            'result_count': len(results_df),
            'insights': insights,
            'llm_enhancement': {
                'applied': 'llm_enhanced' in generated_sql.template_id,
                'adaptations': generated_sql.adaptations_made
            }
        }
    
    # Replace the method
    enhanced_rag_system.process_oceanographic_query_enhanced = enhanced_process_query
    
    logger.info("LLM-Enhanced SQL Generator integrated successfully")
    return enhanced_generator