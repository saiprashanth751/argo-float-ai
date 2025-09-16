'use client';

import { useState } from 'react';
import { FloatChatTestRunner, TestSuite, TestResult, runQuickTest, runVisualizationTest } from '@/utils/testRunner';

export default function TestPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [testResults, setTestResults] = useState<TestSuite[]>([]);
  const [currentTest, setCurrentTest] = useState<string>('');
  const [report, setReport] = useState<string>('');

  const runFullTestSuite = async () => {
    setIsRunning(true);
    setTestResults([]);
    setCurrentTest('Initializing test suite...');
    
    try {
      const runner = new FloatChatTestRunner();
      const suites = await runner.runFullTestSuite();
      
      setTestResults(suites);
      setCurrentTest('Generating report...');
      
      const testReport = runner.generateReport(suites);
      setReport(testReport);
      
      setCurrentTest('Test suite completed!');
    } catch (error) {
      setCurrentTest(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsRunning(false);
    }
  };

  const runQuickTest = async () => {
    setIsRunning(true);
    setCurrentTest('Running quick test...');
    
    try {
      const results = await runQuickTest();
      setCurrentTest(`Quick test completed: ${results.filter(r => r.success).length}/${results.length} passed`);
    } catch (error) {
      setCurrentTest(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsRunning(false);
    }
  };

  const runVizTest = async () => {
    setIsRunning(true);
    setCurrentTest('Testing visualizations...');
    
    try {
      const results = await runVisualizationTest();
      setCurrentTest(`Visualization test completed: ${results.filter(r => r.success).length}/${results.length} passed`);
    } catch (error) {
      setCurrentTest(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsRunning(false);
    }
  };

  const downloadReport = () => {
    if (!report) return;
    
    const blob = new Blob([report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `floatchat-test-report-${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <main className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">FloatChat Test Suite</h1>
          <p className="text-gray-600 mt-2">
            Comprehensive testing for the FloatChat oceanographic data visualization system
          </p>
        </div>

        {/* Test Controls */}
        <div className="bg-white rounded-lg shadow-sm border p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Test Controls</h2>
          
          <div className="flex flex-wrap gap-4">
            <button
              onClick={runQuickTest}
              disabled={isRunning}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              Quick Test (5 prompts)
            </button>
            
            <button
              onClick={runVizTest}
              disabled={isRunning}
              className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              Visualization Test
            </button>
            
            <button
              onClick={runFullTestSuite}
              disabled={isRunning}
              className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              Full Test Suite
            </button>
            
            {report && (
              <button
                onClick={downloadReport}
                className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                Download Report
              </button>
            )}
          </div>
          
          {isRunning && (
            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
              <div className="flex items-center">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-3"></div>
                <span className="text-blue-800">{currentTest}</span>
              </div>
            </div>
          )}
        </div>

        {/* Test Results */}
        {testResults.length > 0 && (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold text-gray-900">Test Results</h2>
            
            {testResults.map((suite, index) => (
              <div key={index} className="bg-white rounded-lg shadow-sm border p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">{suite.name}</h3>
                  <div className="flex space-x-4 text-sm">
                    <span className={`px-3 py-1 rounded-full ${
                      suite.summary.successRate >= 80 
                        ? 'bg-green-100 text-green-800' 
                        : suite.summary.successRate >= 60 
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {suite.summary.successRate.toFixed(1)}% Success
                    </span>
                    <span className="px-3 py-1 bg-gray-100 text-gray-800 rounded-full">
                      {suite.summary.averageTime.toFixed(2)}s Avg
                    </span>
                  </div>
                </div>
                
                <p className="text-gray-600 mb-4">{suite.description}</p>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-gray-900">{suite.summary.total}</div>
                    <div className="text-sm text-gray-600">Total Tests</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">{suite.summary.passed}</div>
                    <div className="text-sm text-gray-600">Passed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-red-600">{suite.summary.failed}</div>
                    <div className="text-sm text-gray-600">Failed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{suite.summary.averageTime.toFixed(2)}s</div>
                    <div className="text-sm text-gray-600">Avg Time</div>
                  </div>
                </div>
                
                {/* Failed tests */}
                {suite.summary.failed > 0 && (
                  <div className="mt-4">
                    <h4 className="font-medium text-gray-900 mb-2">Failed Tests:</h4>
                    <div className="space-y-2">
                      {suite.results.filter(r => !r.success).map((result, idx) => (
                        <div key={idx} className="p-3 bg-red-50 border border-red-200 rounded-lg">
                          <div className="font-medium text-red-800">{result.prompt}</div>
                          <div className="text-sm text-red-600 mt-1">
                            Error: {result.error || 'Unknown error'}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Report Display */}
        {report && (
          <div className="mt-8 bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900">Test Report</h2>
              <button
                onClick={downloadReport}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors text-sm"
              >
                Download
              </button>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-4 overflow-auto max-h-96">
              <pre className="text-sm text-gray-800 whitespace-pre-wrap">{report}</pre>
            </div>
          </div>
        )}

        {/* Instructions */}
        <div className="mt-8 bg-blue-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-2">Testing Instructions</h3>
          <div className="text-blue-800 space-y-2">
            <p><strong>Quick Test:</strong> Tests 5 representative prompts to verify basic functionality</p>
            <p><strong>Visualization Test:</strong> Focuses on queries that should generate visualizations</p>
            <p><strong>Full Test Suite:</strong> Comprehensive testing of all prompt categories and difficulty levels</p>
            <p><strong>Note:</strong> Tests require the backend server to be running and accessible</p>
          </div>
        </div>
      </div>
    </main>
  );
}
