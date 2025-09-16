'use client';

import { useState } from 'react';

export default function SimpleTestPage() {
  const [testResult, setTestResult] = useState<string>('Click button to test');

  const testBackend = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/health');
      const data = await response.json();
      setTestResult(`✅ Backend is working! Status: ${data.status}`);
    } catch (error) {
      setTestResult(`❌ Backend error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const testQuery = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: 'Show me all ARGO floats',
          include_sql: true,
          limit: 5
        }),
      });
      const data = await response.json();
      setTestResult(`✅ Query successful! Found ${data.result_count} results. SQL: ${data.sql_query}`);
    } catch (error) {
      setTestResult(`❌ Query error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Simple Backend Test</h1>
        
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="space-y-4">
            <button
              onClick={testBackend}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Test Backend Health
            </button>
            
            <button
              onClick={testQuery}
              className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 ml-4"
            >
              Test Query
            </button>
            
            <div className="mt-4 p-4 bg-gray-100 rounded-lg">
              <pre className="text-sm">{testResult}</pre>
            </div>
          </div>
        </div>

        <div className="mt-8 bg-blue-50 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-blue-900 mb-2">Quick Test Prompts</h2>
          <div className="space-y-2 text-blue-800">
            <p>Try these queries in the main app:</p>
            <ul className="list-disc list-inside space-y-1">
              <li>"Show me all ARGO floats"</li>
              <li>"Count total profiles"</li>
              <li>"What is the average temperature?"</li>
              <li>"Show temperature profiles for platform 1900121"</li>
              <li>"Show me salinity distribution in the Arabian Sea"</li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  );
}
