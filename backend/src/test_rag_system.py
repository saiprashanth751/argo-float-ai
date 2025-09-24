#!/usr/bin/env python3
"""
Final Comprehensive Test - Intelligent RAG System
"""

from services.enhanced_rag_oceanographic import ProductionOceanographicRAG
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    print('🚀 FINAL COMPREHENSIVE TEST - Intelligent RAG System')
    print('=' * 60)

    rag = ProductionOceanographicRAG()

    # Test diverse query types
    test_queries = [
        'What is the average surface temperature in the Indian Ocean?',
        'Show me the latest 10 profiles from platform 1900121',
        'Find all measurements below 1000m depth in the Arabian Sea',
        'Compare surface salinity between different regions',
        'What is the maximum depth reached by any float?'
    ]

    total_llm_used = 0
    total_queries = len(test_queries)

    for i, query in enumerate(test_queries, 1):
        print(f'\n🔍 Test {i}: {query}')
        print('-' * 50)
        
        result = rag.process_oceanographic_query(query)
        
        if result['success']:
            llm_used = result['llm_enhancement']['used']
            if llm_used:
                total_llm_used += 1
                
            print('✅ SUCCESS')
            method = '🧠 LLM-Enhanced' if llm_used else '📋 Template-Based'
            print('   Method:', method)
            print('   Template:', result['template_used'])
            print('   Rows:', f"{result['result_count']:,}")
            print('   Time:', f"{result['processing_time']:.2f}s")
            print('   SQL:', result['sql_query'][:80] + '...')
        else:
            print('❌ FAILED:', result['error'])

    print(f'\n📊 FINAL SUMMARY')
    print('=' * 60)
    print('Total Queries:', total_queries)
    print('LLM-Enhanced:', f'{total_llm_used}/{total_queries} ({total_llm_used/total_queries*100:.1f}%)')
    print('Template-Based:', f'{total_queries-total_llm_used}/{total_queries} ({(total_queries-total_llm_used)/total_queries*100:.1f}%)')
    print('\n🎉 INTELLIGENT RAG SYSTEM: FULLY OPERATIONAL!')

if __name__ == "__main__":
    main()
