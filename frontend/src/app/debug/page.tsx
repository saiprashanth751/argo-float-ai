'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api';

export default function DebugPage() {
  const [backendStatus, setBackendStatus] = useState<string>('Checking...');
  const [testQuery, setTestQuery] = useState<string>('Show me all ARGO floats');
  const [queryResult, setQueryResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    checkBackendStatus();
  }, []);

  const checkBackendStatus = async () => {
    try {
      const health = await api.checkHealth();
      setBackendStatus(`✅ Backend is healthy: ${JSON.stringify(health, null, 2)}`);
    } catch (error) {
      setBackendStatus(`❌ Backend error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const testQueryExecution = async () => {
    setIsLoading(true);
    try {
      const result = await api.processQuery({
        query: testQuery,
        include_sql: true,
        limit: 10
      });
      setQueryResult(result);
    } catch (error) {
      setQueryResult({ error: error instanceof Error ? error.message : 'Unknown error' });
    } finally {
      setIsLoading(false);
    }
  };

  const testStats = async () => {
    try {
      const stats = await api.getStats();
      setQueryResult(stats);
    } catch (error) {
      setQueryResult({ error: error instanceof Error ? error.message : 'Unknown error' });
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Debug Page - Backend Connection</h1>
        
        {/* Backend Status */}
        <div className="bg-white rounded-lg shadow-sm border p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Backend Status</h2>
          <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto">
            {backendStatus}
          </pre>
          <button
            onClick={checkBackendStatus}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Refresh Status
          </button>
        </div>

        {/* Test Query */}
        <div className="bg-white rounded-lg shadow-sm border p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Test Query</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Query:</label>
              <input
                type="text"
                value={testQuery}
                onChange={(e) => setTestQuery(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter your query..."
              />
            </div>
            <div className="flex space-x-4">
              <button
                onClick={testQueryExecution}
                disabled={isLoading}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400"
              >
                {isLoading ? 'Testing...' : 'Test Query'}
              </button>
              <button
                onClick={testStats}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
              >
                Test Stats
              </button>
            </div>
          </div>
        </div>

        {/* Query Result */}
        {queryResult && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Query Result</h2>
            <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto max-h-96">
              {JSON.stringify(queryResult, null, 2)}
            </pre>
          </div>
        )}

        {/* Quick Test Prompts */}
        <div className="bg-white rounded-lg shadow-sm border p-6 mt-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Test Prompts</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              'Show me all ARGO floats',
              'Count total profiles',
              'What is the average temperature?',
              'Show temperature profiles for platform 1900121',
              'Show me salinity distribution in the Arabian Sea',
              'What are the nearest ARGO floats to this location?'
            ].map((prompt, index) => (
              <button
                key={index}
                onClick={() => {
                  setTestQuery(prompt);
                  testQueryExecution();
                }}
                className="p-3 text-left bg-gray-50 hover:bg-gray-100 rounded-lg border transition-colors"
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
