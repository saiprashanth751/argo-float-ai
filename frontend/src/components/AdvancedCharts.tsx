'use client';

import { useState, useEffect } from 'react';
import { QueryResponse } from '@/lib/types';
import PlotlyChart from './PlotlyChart';

interface AdvancedChartsProps {
  data: QueryResponse;
  chartType?: 'timeSeries' | 'heatmap' | 'profile' | 'scatter' | 'histogram' | 'box';
  selectedParameters?: string[];
}

interface ChartConfig {
  type: string;
  title: string;
  description: string;
  requiredColumns: string[];
  recommendedFor: string[];
}

const CHART_CONFIGS: { [key: string]: ChartConfig } = {
  timeSeries: {
    type: 'timeSeries',
    title: 'Time Series Analysis',
    description: 'Temporal trends of oceanographic parameters',
    requiredColumns: ['date', 'time'],
    recommendedFor: ['temperature', 'salinity', 'pressure']
  },
  heatmap: {
    type: 'heatmap',
    title: 'Spatial Heatmap',
    description: 'Geographic distribution of ocean parameters',
    requiredColumns: ['latitude', 'longitude'],
    recommendedFor: ['temperature', 'salinity', 'pressure']
  },
  profile: {
    type: 'profile',
    title: 'Depth Profile',
    description: 'Vertical distribution of ocean parameters',
    requiredColumns: ['pressure', 'depth'],
    recommendedFor: ['temperature', 'salinity', 'density']
  },
  scatter: {
    type: 'scatter',
    title: 'Parameter Correlation',
    description: 'Relationship between different ocean parameters',
    requiredColumns: [],
    recommendedFor: ['temperature', 'salinity', 'pressure', 'density']
  },
  histogram: {
    type: 'histogram',
    title: 'Parameter Distribution',
    description: 'Statistical distribution of ocean parameters',
    requiredColumns: [],
    recommendedFor: ['temperature', 'salinity', 'pressure']
  },
  box: {
    type: 'box',
    title: 'Statistical Summary',
    description: 'Box plots showing parameter statistics',
    requiredColumns: [],
    recommendedFor: ['temperature', 'salinity', 'pressure']
  }
};

export default function AdvancedCharts({ 
  data, 
  chartType = 'timeSeries',
  selectedParameters = ['temperature', 'salinity', 'pressure']
}: AdvancedChartsProps) {
  const [plotData, setPlotData] = useState<any[]>([]);
  const [plotLayout, setPlotLayout] = useState<any>({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Process data based on chart type
  useEffect(() => {
    if (!data?.results || data.results.length === 0) {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Limit data points for performance (max 2000 points)
      const results = data.results.length > 2000 ? data.results.slice(0, 2000) : data.results;
      const columns = data.columns || [];

      switch (chartType) {
        case 'timeSeries':
          generateTimeSeriesChart(results, columns);
          break;
        case 'heatmap':
          generateHeatmapChart(results, columns);
          break;
        case 'profile':
          generateProfileChart(results, columns);
          break;
        case 'scatter':
          generateScatterChart(results, columns);
          break;
        case 'histogram':
          generateHistogramChart(results, columns);
          break;
        case 'box':
          generateBoxChart(results, columns);
          break;
        default:
          generateTimeSeriesChart(results, columns);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate chart');
    } finally {
      setIsLoading(false);
    }
  }, [data, chartType, selectedParameters]);

  const generateTimeSeriesChart = (results: any[], columns: string[]) => {
    const dateColumn = columns.find(col => 
      col.toLowerCase().includes('date') || col.toLowerCase().includes('time')
    );
    
    if (!dateColumn) {
      setError('No date/time column found for time series chart');
      return;
    }

    const traces: any[] = [];
    
    selectedParameters.forEach(param => {
      const paramColumn = columns.find(col => 
        col.toLowerCase().includes(param.toLowerCase())
      );
      
      if (paramColumn) {
        const validData = results
          .filter(row => row[dateColumn] && row[paramColumn] !== null && row[paramColumn] !== undefined)
          .map(row => ({
            x: new Date(row[dateColumn]),
            y: parseFloat(row[paramColumn])
          }))
          .sort((a, b) => a.x.getTime() - b.x.getTime());

        if (validData.length > 0) {
          traces.push({
            x: validData.map(d => d.x),
            y: validData.map(d => d.y),
            type: 'scatter',
            mode: 'lines+markers',
            name: paramColumn,
            line: { width: 2 },
            marker: { size: 4 }
          });
        }
      }
    });

    setPlotData(traces);
    setPlotLayout({
      title: 'Temporal Trends of Ocean Parameters',
      xaxis: { title: 'Time' },
      yaxis: { title: 'Parameter Value' },
      hovermode: 'closest',
      showlegend: true,
      margin: { t: 50, r: 50, b: 50, l: 50 }
    });
  };

  const generateHeatmapChart = (results: any[], columns: string[]) => {
    const latColumn = columns.find(col => col.toLowerCase().includes('lat'));
    const lonColumn = columns.find(col => col.toLowerCase().includes('lon'));
    
    if (!latColumn || !lonColumn) {
      setError('No latitude/longitude columns found for heatmap chart');
      return;
    }

    const paramColumn = columns.find(col => 
      selectedParameters.some(param => col.toLowerCase().includes(param.toLowerCase()))
    );

    if (!paramColumn) {
      setError('No parameter column found for heatmap chart');
      return;
    }

    const validData = results.filter(row => 
      row[latColumn] !== null && row[lonColumn] !== null && row[paramColumn] !== null
    );

    if (validData.length === 0) {
      setError('No valid data points for heatmap chart');
      return;
    }

    const trace = {
      x: validData.map(row => parseFloat(row[lonColumn])),
      y: validData.map(row => parseFloat(row[latColumn])),
      z: validData.map(row => parseFloat(row[paramColumn])),
      type: 'heatmap',
      colorscale: 'Viridis',
      showscale: true,
      colorbar: { title: paramColumn }
    };

    setPlotData([trace]);
    setPlotLayout({
      title: `Geographic Distribution of ${paramColumn}`,
      xaxis: { title: 'Longitude' },
      yaxis: { title: 'Latitude' },
      margin: { t: 50, r: 50, b: 50, l: 50 }
    });
  };

  const generateProfileChart = (results: any[], columns: string[]) => {
    const depthColumn = columns.find(col => 
      col.toLowerCase().includes('pressure') || col.toLowerCase().includes('depth')
    );
    
    if (!depthColumn) {
      setError('No depth/pressure column found for profile chart');
      return;
    }

    const traces: any[] = [];
    
    selectedParameters.forEach(param => {
      const paramColumn = columns.find(col => 
        col.toLowerCase().includes(param.toLowerCase())
      );
      
      if (paramColumn && paramColumn !== depthColumn) {
        const validData = results
          .filter(row => row[depthColumn] !== null && row[paramColumn] !== null)
          .map(row => ({
            x: parseFloat(row[paramColumn]),
            y: parseFloat(row[depthColumn])
          }))
          .sort((a, b) => b.y - a.y); // Sort by depth (deepest first)

        if (validData.length > 0) {
          traces.push({
            x: validData.map(d => d.x),
            y: validData.map(d => d.y),
            type: 'scatter',
            mode: 'lines+markers',
            name: paramColumn,
            line: { width: 2 },
            marker: { size: 4 }
          });
        }
      }
    });

    setPlotData(traces);
    setPlotLayout({
      title: 'Depth Profile of Ocean Parameters',
      xaxis: { title: 'Parameter Value' },
      yaxis: { title: 'Depth (m)', autorange: 'reversed' },
      hovermode: 'closest',
      showlegend: true,
      margin: { t: 50, r: 50, b: 50, l: 50 }
    });
  };

  const generateScatterChart = (results: any[], columns: string[]) => {
    if (selectedParameters.length < 2) {
      setError('At least 2 parameters required for scatter chart');
      return;
    }

    const param1Column = columns.find(col => 
      col.toLowerCase().includes(selectedParameters[0].toLowerCase())
    );
    const param2Column = columns.find(col => 
      col.toLowerCase().includes(selectedParameters[1].toLowerCase())
    );

    if (!param1Column || !param2Column) {
      setError('Required parameter columns not found for scatter chart');
      return;
    }

    const validData = results.filter(row => 
      row[param1Column] !== null && row[param2Column] !== null
    );

    if (validData.length === 0) {
      setError('No valid data points for scatter chart');
      return;
    }

    const trace = {
      x: validData.map(row => parseFloat(row[param1Column])),
      y: validData.map(row => parseFloat(row[param2Column])),
      type: 'scatter',
      mode: 'markers',
      marker: {
        size: 6,
        color: validData.map((_, index) => index),
        colorscale: 'Viridis',
        showscale: true,
        colorbar: { title: 'Data Point Index' }
      },
      text: validData.map((row, index) => 
        `Point ${index + 1}<br>${param1Column}: ${row[param1Column]}<br>${param2Column}: ${row[param2Column]}`
      ),
      hovertemplate: '%{text}<extra></extra>'
    };

    setPlotData([trace]);
    setPlotLayout({
      title: `Correlation: ${param1Column} vs ${param2Column}`,
      xaxis: { title: param1Column },
      yaxis: { title: param2Column },
      hovermode: 'closest',
      margin: { t: 50, r: 50, b: 50, l: 50 }
    });
  };

  const generateHistogramChart = (results: any[], columns: string[]) => {
    const traces: any[] = [];
    
    selectedParameters.forEach(param => {
      const paramColumn = columns.find(col => 
        col.toLowerCase().includes(param.toLowerCase())
      );
      
      if (paramColumn) {
        const validData = results
          .filter(row => row[paramColumn] !== null && row[paramColumn] !== undefined)
          .map(row => parseFloat(row[paramColumn]));

        if (validData.length > 0) {
          traces.push({
            x: validData,
            type: 'histogram',
            name: paramColumn,
            opacity: 0.7,
            nbinsx: 30
          });
        }
      }
    });

    setPlotData(traces);
    setPlotLayout({
      title: 'Distribution of Ocean Parameters',
      xaxis: { title: 'Parameter Value' },
      yaxis: { title: 'Frequency' },
      barmode: 'overlay',
      showlegend: true,
      margin: { t: 50, r: 50, b: 50, l: 50 }
    });
  };

  const generateBoxChart = (results: any[], columns: string[]) => {
    const traces: any[] = [];
    
    selectedParameters.forEach(param => {
      const paramColumn = columns.find(col => 
        col.toLowerCase().includes(param.toLowerCase())
      );
      
      if (paramColumn) {
        const validData = results
          .filter(row => row[paramColumn] !== null && row[paramColumn] !== undefined)
          .map(row => parseFloat(row[paramColumn]));

        if (validData.length > 0) {
          traces.push({
            y: validData,
            type: 'box',
            name: paramColumn,
            boxpoints: 'outliers'
          });
        }
      }
    });

    setPlotData(traces);
    setPlotLayout({
      title: 'Statistical Summary of Ocean Parameters',
      yaxis: { title: 'Parameter Value' },
      showlegend: true,
      margin: { t: 50, r: 50, b: 50, l: 50 }
    });
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-8">
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-600">Generating chart...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-8 text-center">
        <div className="text-red-500">
          <svg className="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h3 className="text-lg font-medium text-gray-900 mb-2">Chart Error</h3>
          <p className="text-gray-600">{error}</p>
        </div>
      </div>
    );
  }

  if (plotData.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-8 text-center">
        <div className="text-gray-500">
          <svg className="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Data Available</h3>
          <p className="text-gray-600">Insufficient data to generate {chartType} chart</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      {/* Header */}
      <div className="p-6 border-b">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              {CHART_CONFIGS[chartType]?.title || 'Advanced Chart'}
            </h3>
            <p className="text-sm text-gray-600">
              {CHART_CONFIGS[chartType]?.description || 'Data visualization'}
            </p>
          </div>
          
          {/* Chart type selector */}
          <div className="flex space-x-2">
            {Object.keys(CHART_CONFIGS).map(type => (
              <button
                key={type}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  chartType === type
                    ? 'bg-blue-100 text-blue-700'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="p-6">
        <PlotlyChart
          data={plotData}
          layout={plotLayout}
          style={{ width: '100%', height: '500px' }}
          config={{
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
          }}
        />
      </div>
    </div>
  );
}
