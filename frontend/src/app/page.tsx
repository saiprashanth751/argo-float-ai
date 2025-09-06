'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api';
import { QueryResponse, DatabaseStats } from '@/lib/types';
import ProfileChart from '@/components/ProfileChart';
import ChatInterface from '@/components/ChatInterface';
import dynamic from 'next/dynamic';
const MapView = dynamic(() => import('@/components/MapView'), {
  ssr: false, // Disable server-side rendering
});
import 'leaflet/dist/leaflet.css';

export default function FloatChatPage() {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentResult, setCurrentResult] = useState<QueryResponse | null>(null);
  const [stats, setStats] = useState<DatabaseStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'table' | 'profile' | 'map'>('table');
  const [interfaceMode, setInterfaceMode] = useState<'simple' | 'chat'>('chat'); // Default to chat

  // Load stats on component mount
  useEffect(() => {
    const loadStats = async () => {
      try {
        const dbStats = await api.getStats();
        setStats(dbStats);
      } catch (error) {
        console.error('Failed to load stats:', error);
      }
    };
    
    loadStats();
  }, []);

  const handleQuerySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await api.processQuery({
        query: query.trim(),
        include_sql: true,
        limit: 1000
      });
      
      setCurrentResult(result);
      setActiveTab('table'); // Default to table view
      
    } catch (error) {
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle result from ChatInterface
  const handleChatQueryResult = (result: QueryResponse) => {
    setCurrentResult(result);
    setActiveTab('table');
    setError(null);
  };

  // Check what visualizations are possible
  const canShowProfile = currentResult?.results && currentResult.columns?.some(col => 
    col.toLowerCase().includes('pressure') || col.toLowerCase().includes('depth')
  );

  const canShowMap = currentResult?.results && 
    currentResult.columns?.some(col => col.toLowerCase().includes('lat')) &&
    currentResult.columns?.some(col => col.toLowerCase().includes('lon'));

  return (
    <main className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">FloatChat</h1>
              <p className="text-sm text-gray-600">Advanced oceanographic data analysis</p>
            </div>
            
            {/* Mode Toggle */}
            <div className="flex items-center space-x-4">
              <div className="flex bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setInterfaceMode('chat')}
                  className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                    interfaceMode === 'chat'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  Chat Mode
                </button>
                <button
                  onClick={() => setInterfaceMode('simple')}
                  className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                    interfaceMode === 'simple'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  Simple Mode
                </button>
              </div>
            </div>

            {/* Stats Display */}
            {stats && (
              <div className="flex space-x-6 text-sm">
                <div className="text-center">
                  <div className="font-semibold text-gray-900">{stats.total_floats?.toLocaleString() || '0'}</div>
                  <div className="text-gray-600">Floats</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-gray-900">{stats.total_measurements?.toLocaleString() || '0'}</div>
                  <div className="text-gray-600">Measurements</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {interfaceMode === 'chat' ? (
          /* Chat Interface Layout */
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Chat Panel */}
            <div className="space-y-6">
              <div className="bg-white rounded-lg shadow-sm border">
                <div className="p-4 border-b">
                  <h2 className="text-lg font-semibold text-gray-900">AI Assistant</h2>
                  <p className="text-sm text-gray-600">Ask questions about ARGO data in natural language</p>
                </div>
                <div className="h-96">
                  <ChatInterface onQueryResult={handleChatQueryResult} />
                </div>
              </div>
            </div>

            {/* Results Panel */}
            <div className="space-y-6">
              {currentResult && currentResult.success && currentResult.results && currentResult.results.length > 0 ? (
                <div className="bg-white rounded-lg shadow-sm border">
                  {/* Quick Results Summary */}
                  <div className="p-4 border-b">
                    <div className="flex items-center justify-between">
                      <h2 className="text-lg font-semibold text-gray-900">Results</h2>
                      <div className="flex space-x-4 text-sm text-gray-600">
                        <span>{currentResult.result_count} rows</span>
                        <span>{currentResult.processing_time?.toFixed(2)}s</span>
                      </div>
                    </div>
                  </div>

                  {/* Mini Tabs */}
                  <div className="border-b">
                    <nav className="flex space-x-4 px-4" aria-label="Tabs">
                      <button
                        onClick={() => setActiveTab('table')}
                        className={`py-2 px-1 border-b-2 font-medium text-sm ${
                          activeTab === 'table'
                            ? 'border-blue-500 text-blue-600'
                            : 'border-transparent text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        Table
                      </button>
                      {canShowProfile && (
                        <button
                          onClick={() => setActiveTab('profile')}
                          className={`py-2 px-1 border-b-2 font-medium text-sm ${
                            activeTab === 'profile'
                              ? 'border-blue-500 text-blue-600'
                              : 'border-transparent text-gray-500 hover:text-gray-700'
                          }`}
                        >
                          Profile
                        </button>
                      )}
                      {canShowMap && (
                        <button
                          onClick={() => setActiveTab('map')}
                          className={`py-2 px-1 border-b-2 font-medium text-sm ${
                            activeTab === 'map'
                              ? 'border-blue-500 text-blue-600'
                              : 'border-transparent text-gray-500 hover:text-gray-700'
                          }`}
                        >
                          Map
                        </button>
                      )}
                    </nav>
                  </div>

                  {/* Compact Results Display */}
                  <div className="max-h-96 overflow-auto">
                    {activeTab === 'table' && (
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead className="bg-gray-50">
                            <tr>
                              {currentResult.columns?.slice(0, 4).map((column) => (
                                <th key={column} className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                                  {column}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-200">
                            {currentResult.results.slice(0, 10).map((row, index) => (
                              <tr key={index}>
                                {currentResult.columns?.slice(0, 4).map((column) => (
                                  <td key={column} className="px-3 py-2 text-gray-900">
                                    {typeof row[column] === 'number' 
                                      ? row[column].toFixed(2)
                                      : row[column] || '-'
                                    }
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}

                    {activeTab === 'profile' && canShowProfile && (
                      <div className="p-4">
                        <ProfileChart 
                          data={currentResult.results} 
                          columns={currentResult.columns || []}
                        />
                      </div>
                    )}

                    {activeTab === 'map' && canShowMap && (
                      <div className="p-4">
                        <MapView 
                          data={currentResult.results} 
                          columns={currentResult.columns || []}
                        />
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-lg shadow-sm border p-8 text-center">
                  <div className="text-gray-400">
                    <svg className="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Results Yet</h3>
                    <p className="text-gray-600">Start a conversation to see your data analysis results here.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Simple Interface (Original) */
          <div className="space-y-6">
            {/* Query Input */}
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <form onSubmit={handleQuerySubmit} className="space-y-4">
                <div>
                  <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
                    Ask about oceanographic data:
                  </label>
                  <input
                    type="text"
                    id="query"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g., Show temperature profile for float 1900121"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    disabled={isLoading}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-500">
                    Try: "Show all floats in Arabian Sea" or "Average temperature by depth"
                  </div>
                  <button
                    type="submit"
                    disabled={!query.trim() || isLoading}
                    className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isLoading ? 'Processing...' : 'Ask'}
                  </button>
                </div>
              </form>
            </div>

            {/* Error Display */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex">
                  <div className="text-red-800">
                    <strong>Error:</strong> {error}
                  </div>
                </div>
              </div>
            )}

            {/* Results Display - Full Original Layout */}
            {currentResult && (
              <div className="space-y-6">
                {/* Query Info */}
                <div className="bg-white rounded-lg shadow-sm border p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-gray-900">Query Results</h2>
                    <div className="flex space-x-4 text-sm text-gray-600">
                      <span>{currentResult.result_count} results</span>
                      <span>{currentResult.processing_time?.toFixed(2)}s</span>
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        currentResult.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                      }`}>
                        {currentResult.success ? 'Success' : 'Failed'}
                      </span>
                    </div>
                  </div>

                  {/* SQL Query Display */}
                  {currentResult.sql_query && (
                    <div className="mb-4">
                      <h3 className="text-sm font-medium text-gray-700 mb-2">Generated SQL:</h3>
                      <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto text-gray-800">
                        {currentResult.sql_query}
                      </pre>
                    </div>
                  )}

                  {/* Error Message */}
                  {!currentResult.success && currentResult.error && (
                    <div className="text-red-600">
                      <strong>Error:</strong> {currentResult.error}
                    </div>
                  )}
                </div>

                {/* Full Visualization Tabs */}
                {currentResult.success && currentResult.results && currentResult.results.length > 0 && (
                  <div className="bg-white rounded-lg shadow-sm border">
                    {/* Tab Navigation */}
                    <div className="border-b">
                      <nav className="flex space-x-8 px-6" aria-label="Tabs">
                        <button
                          onClick={() => setActiveTab('table')}
                          className={`py-4 px-1 border-b-2 font-medium text-sm ${
                            activeTab === 'table'
                              ? 'border-blue-500 text-blue-600'
                              : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                          }`}
                        >
                          Data Table
                        </button>
                        {canShowProfile && (
                          <button
                            onClick={() => setActiveTab('profile')}
                            className={`py-4 px-1 border-b-2 font-medium text-sm ${
                              activeTab === 'profile'
                                ? 'border-blue-500 text-blue-600'
                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                            }`}
                          >
                            Profile Chart
                          </button>
                        )}
                        {canShowMap && (
                          <button
                            onClick={() => setActiveTab('map')}
                            className={`py-4 px-1 border-b-2 font-medium text-sm ${
                              activeTab === 'map'
                                ? 'border-blue-500 text-blue-600'
                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                            }`}
                          >
                            Map View
                          </button>
                        )}
                      </nav>
                    </div>

                    {/* Tab Content */}
                    <div>
                      {activeTab === 'table' && (
                        <div className="overflow-x-auto">
                          <table className="w-full">
                            <thead className="bg-gray-50">
                              <tr>
                                {currentResult.columns?.map((column) => (
                                  <th
                                    key={column}
                                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                                  >
                                    {column}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-200">
                              {currentResult.results.slice(0, 20).map((row, index) => (
                                <tr key={index} className="hover:bg-gray-50">
                                  {currentResult.columns?.map((column) => (
                                    <td key={column} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                      {typeof row[column] === 'number' 
                                        ? row[column].toFixed(3)
                                        : row[column] || '-'
                                      }
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {currentResult.results.length > 20 && (
                            <div className="p-4 text-center text-gray-600 bg-gray-50">
                              Showing first 20 of {currentResult.results.length} results
                            </div>
                          )}
                        </div>
                      )}

                      {activeTab === 'profile' && canShowProfile && (
                        <div className="p-6">
                          <ProfileChart 
                            data={currentResult.results} 
                            columns={currentResult.columns || []}
                          />
                        </div>
                      )}

                      {activeTab === 'map' && canShowMap && (
                        <div className="p-6">
                          <MapView 
                            data={currentResult.results} 
                            columns={currentResult.columns || []}
                          />
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}