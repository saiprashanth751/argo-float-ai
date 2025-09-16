'use client';

import { useState } from 'react';
import AdvancedCharts from '@/components/AdvancedCharts';
import GlobeVisualization from '@/components/GlobeVisualization';

export default function TestVizPage() {
  const [testData] = useState({
    success: true,
    result_count: 10,
    processing_time: 1.5,
    columns: ['latitude', 'longitude', 'temperature', 'salinity', 'pressure', 'date'],
    results: [
      { latitude: 15.5, longitude: 75.0, temperature: 28.5, salinity: 35.2, pressure: 10, date: '2025-01-01' },
      { latitude: 16.0, longitude: 75.5, temperature: 27.8, salinity: 35.1, pressure: 20, date: '2025-01-02' },
      { latitude: 15.8, longitude: 74.8, temperature: 29.1, salinity: 35.3, pressure: 15, date: '2025-01-03' },
      { latitude: 16.2, longitude: 75.2, temperature: 28.2, salinity: 35.0, pressure: 25, date: '2025-01-04' },
      { latitude: 15.9, longitude: 75.1, temperature: 28.8, salinity: 35.2, pressure: 18, date: '2025-01-05' },
    ]
  });

  return (
    <main className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Visualization Test Page</h1>
        
        <div className="space-y-8">
          {/* Test Advanced Charts */}
          <div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Advanced Charts Test</h2>
            <AdvancedCharts 
              data={testData}
              chartType="scatter"
              selectedParameters={['temperature', 'salinity']}
            />
          </div>

          {/* Test Globe Visualization */}
          <div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">3D Globe Test</h2>
            <GlobeVisualization 
              data={testData.results}
              columns={testData.columns}
              visualizationType="distribution"
              selectedParameters={['temperature', 'salinity', 'pressure']}
            />
          </div>
        </div>
      </div>
    </main>
  );
}
