# backend\test_api.py - Test script for FloatChat FastAPI backend
import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_stats_endpoint():
    """Test the database statistics endpoint"""
    print("\n🔍 Testing stats endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/stats")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Stats test failed: {e}")
        return False

def test_floats_endpoint():
    """Test the floats metadata endpoint"""
    print("\n🔍 Testing floats endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/floats")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Found {len(data)} floats")
        if data:
            print("First float:")
            print(json.dumps(data[0], indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Floats test failed: {e}")
        return False

def test_query_endpoint():
    """Test the main query processing endpoint"""
    print("\n🔍 Testing query endpoint...")
    
    # Test queries based on your single float data
    test_queries = [
        "Show me all available floats",
        "What is the temperature profile for this float?",
        "Show surface temperature measurements",
        "How many measurements do we have?",
        "Show me salinity data"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing Query: {query} ---")
        try:
            payload = {
                "query": query,
                "include_sql": True,
                "limit": 100
            }
            
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/api/query", json=payload)
            end_time = time.time()
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Time: {end_time - start_time:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Success: {data['success']}")
                print(f"Result Count: {data['result_count']}")
                print(f"Processing Time: {data['processing_time']:.2f}s")
                
                if data['sql_query']:
                    print(f"Generated SQL: {data['sql_query']}")
                
                if data['results'] and len(data['results']) > 0:
                    print(f"Sample results: {data['results'][:2]}")
                
                if data.get('error'):
                    print(f"Error: {data['error']}")
            else:
                print(f"Request failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Query test failed: {e}")

def test_websocket_connection():
    """Test WebSocket connection (basic connectivity only)"""
    print("\n🔍 Testing WebSocket endpoint availability...")
    try:
        # Just test if the WebSocket endpoint responds
        response = requests.get(f"{BASE_URL}/ws/chat")
        # WebSocket endpoints typically return 426 for HTTP requests
        if response.status_code in [426, 400]:
            print("✅ WebSocket endpoint is available (returns expected error for HTTP request)")
            return True
        else:
            print(f"WebSocket endpoint response: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

def run_full_api_test():
    """Run comprehensive API test suite"""
    print("🚀 Starting FloatChat API Test Suite")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(BASE_URL)
        print(f"✅ Server is running at {BASE_URL}")
    except Exception as e:
        print(f"❌ Server not accessible at {BASE_URL}")
        print("Make sure to start the server with: uvicorn main:app --reload")
        return
    
    # Run tests
    tests = [
        ("Health Check", test_health_endpoint),
        ("Database Stats", test_stats_endpoint),
        ("Floats Metadata", test_floats_endpoint),
        ("Query Processing", test_query_endpoint),
        ("WebSocket Availability", test_websocket_connection)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    if passed == total:
        print(f"\n🎉 All tests passed! Your API is ready for frontend integration.")
    else:
        print(f"\n⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    run_full_api_test()