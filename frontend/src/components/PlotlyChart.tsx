'use client';

import dynamic from 'next/dynamic';

// Create a wrapper component for Plotly to avoid SSR issues
const PlotlyChart = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <p className="text-gray-600">Loading chart...</p>
      </div>
    </div>
  )
});

export default PlotlyChart;
