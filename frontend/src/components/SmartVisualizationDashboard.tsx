'use client';

import { useState, useEffect } from 'react';
import { QueryResponse } from '@/lib/types';
import dynamic from 'next/dynamic';

// Dynamic imports to avoid SSR issues
const GlobeVisualization = dynamic(() => import('./GlobeVisualization'), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg">Loading 3D globe...</div>
});

const AdvancedCharts = dynamic(() => import('./AdvancedCharts'), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg">Loading charts...</div>
});

const MapView = dynamic(() => import('./MapView'), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg">Loading map...</div>
});

const ProfileChart = dynamic(() => import('./ProfileChart'), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg">Loading profile chart...</div>
});

interface SmartVisualizationDashboardProps {
  data: QueryResponse;
  queryIntent?: string;
  queryComplexity?: string;
}

interface VisualizationRecommendation {
  type: 'globe' | 'chart' | 'map' | 'profile';
  chartType?: string;
  priority: number;
  title: string;
  description: string;
  reasoning: string;
}

export default function SmartVisualizationDashboard({ 
  data, 
  queryIntent = 'exploration',
  queryComplexity = 'intermediate'
}: SmartVisualizationDashboardProps) {
  const [recommendations, setRecommendations] = useState<VisualizationRecommendation[]>([]);
  const [activeVisualizations, setActiveVisualizations] = useState<string[]>([]);
  const [selectedParameters, setSelectedParameters] = useState<string[]>(['temperature', 'salinity', 'pressure']);

  // Analyze data and generate visualization recommendations
  useEffect(() => {
    if (!data?.results || data.results.length === 0) {
      setRecommendations([]);
      return;
    }

    const newRecommendations = generateVisualizationRecommendations(data, queryIntent, queryComplexity);
    setRecommendations(newRecommendations);
    
    // Auto-select top 2 recommendations
    const topRecommendations = newRecommendations
      .sort((a, b) => b.priority - a.priority)
      .slice(0, 2)
      .map(rec => `${rec.type}-${rec.chartType || 'default'}`);
    
    setActiveVisualizations(topRecommendations);
  }, [data, queryIntent, queryComplexity]);

  const generateVisualizationRecommendations = (
    data: QueryResponse, 
    intent: string, 
    complexity: string
  ): VisualizationRecommendation[] => {
    const results = data.results || [];
    const columns = data.columns || [];
    const recommendations: VisualizationRecommendation[] = [];

    // Check for geographic data
    const hasLatitude = columns.some(col => col.toLowerCase().includes('lat'));
    const hasLongitude = columns.some(col => col.toLowerCase().includes('lon'));
    const hasGeographicData = hasLatitude && hasLongitude;

    // Check for temporal data
    const hasDateColumn = columns.some(col => 
      col.toLowerCase().includes('date') || col.toLowerCase().includes('time')
    );

    // Check for depth/pressure data
    const hasDepthData = columns.some(col => 
      col.toLowerCase().includes('pressure') || col.toLowerCase().includes('depth')
    );

    // Check for oceanographic parameters
    const oceanParams = ['temperature', 'salinity', 'pressure', 'density'];
    const availableParams = oceanParams.filter(param => 
      columns.some(col => col.toLowerCase().includes(param))
    );

    // Generate recommendations based on intent and data availability
    switch (intent) {
      case 'spatial_mapping':
        if (hasGeographicData) {
          recommendations.push({
            type: 'globe',
            priority: 10,
            title: '3D Global Distribution',
            description: 'Interactive 3D globe showing ARGO float locations and parameter distributions',
            reasoning: 'Perfect for spatial analysis with geographic coordinates'
          });
          
          recommendations.push({
            type: 'map',
            priority: 8,
            title: '2D Geographic Map',
            description: 'Traditional map view with markers and parameter-based coloring',
            reasoning: 'Complements 3D view with detailed geographic information'
          });

          if (availableParams.length > 0) {
            recommendations.push({
              type: 'chart',
              chartType: 'heatmap',
              priority: 7,
              title: 'Spatial Heatmap',
              description: 'Heatmap showing parameter distribution across geographic regions',
              reasoning: 'Excellent for identifying spatial patterns and hotspots'
            });
          }
        }
        break;

      case 'temporal_trends':
        if (hasDateColumn && availableParams.length > 0) {
          recommendations.push({
            type: 'chart',
            chartType: 'timeSeries',
            priority: 10,
            title: 'Time Series Analysis',
            description: 'Temporal trends of ocean parameters over time',
            reasoning: 'Essential for understanding temporal patterns and trends'
          });

          if (hasGeographicData) {
            recommendations.push({
              type: 'globe',
              priority: 6,
              title: 'Temporal Globe Animation',
              description: '3D globe showing parameter changes over time',
              reasoning: 'Combines spatial and temporal analysis for comprehensive insights'
            });
          }
        }
        break;

      case 'profile_analysis':
        if (hasDepthData && availableParams.length > 0) {
          recommendations.push({
            type: 'chart',
            chartType: 'profile',
            priority: 10,
            title: 'Depth Profile Analysis',
            description: 'Vertical distribution of ocean parameters with depth',
            reasoning: 'Core visualization for understanding vertical ocean structure'
          });

          recommendations.push({
            type: 'profile',
            priority: 8,
            title: 'Ocean Profile Chart',
            description: 'Traditional oceanographic profile visualization',
            reasoning: 'Standard oceanographic visualization for depth profiles'
          });
        }
        break;

      case 'statistical_summary':
        if (availableParams.length > 0) {
          recommendations.push({
            type: 'chart',
            chartType: 'histogram',
            priority: 9,
            title: 'Parameter Distribution',
            description: 'Statistical distribution of ocean parameters',
            reasoning: 'Essential for understanding data distribution and outliers'
          });

          recommendations.push({
            type: 'chart',
            chartType: 'box',
            priority: 8,
            title: 'Statistical Summary',
            description: 'Box plots showing parameter statistics and variability',
            reasoning: 'Provides comprehensive statistical overview'
          });

          if (availableParams.length >= 2) {
            recommendations.push({
              type: 'chart',
              chartType: 'scatter',
              priority: 7,
              title: 'Parameter Correlation',
              description: 'Scatter plot showing relationships between parameters',
              reasoning: 'Reveals correlations and relationships between ocean parameters'
            });
          }
        }
        break;

      case 'comparative_analysis':
        if (availableParams.length >= 2) {
          recommendations.push({
            type: 'chart',
            chartType: 'scatter',
            priority: 9,
            title: 'Comparative Analysis',
            description: 'Compare relationships between different ocean parameters',
            reasoning: 'Perfect for comparing multiple parameters simultaneously'
          });

          recommendations.push({
            type: 'chart',
            chartType: 'box',
            priority: 8,
            title: 'Statistical Comparison',
            description: 'Compare statistical distributions across parameters',
            reasoning: 'Shows statistical differences between parameters'
          });
        }
        break;

      default: // exploration
        // Default recommendations for general exploration
        if (hasGeographicData) {
          recommendations.push({
            type: 'globe',
            priority: 8,
            title: 'Global Ocean Overview',
            description: '3D globe showing ARGO float distribution worldwide',
            reasoning: 'Best starting point for exploring global ocean data'
          });
        }

        if (availableParams.length > 0) {
          recommendations.push({
            type: 'chart',
            chartType: 'histogram',
            priority: 7,
            title: 'Data Overview',
            description: 'Distribution of available ocean parameters',
            reasoning: 'Quick overview of data characteristics and quality'
          });
        }

        if (hasDepthData) {
          recommendations.push({
            type: 'chart',
            chartType: 'profile',
            priority: 6,
            title: 'Depth Analysis',
            description: 'Vertical structure of ocean parameters',
            reasoning: 'Understanding vertical ocean structure'
          });
        }
    }

    // Sort by priority
    return recommendations.sort((a, b) => b.priority - a.priority);
  };

  const toggleVisualization = (visualizationId: string) => {
    setActiveVisualizations(prev => 
      prev.includes(visualizationId)
        ? prev.filter(id => id !== visualizationId)
        : [...prev, visualizationId]
    );
  };

  const renderVisualization = (recommendation: VisualizationRecommendation) => {
    const visualizationId = `${recommendation.type}-${recommendation.chartType || 'default'}`;
    
    if (!activeVisualizations.includes(visualizationId)) {
      return null;
    }

    switch (recommendation.type) {
      case 'globe':
        return (
          <GlobeVisualization
            key={visualizationId}
            data={data.results || []}
            columns={data.columns || []}
            visualizationType="distribution"
            selectedParameters={selectedParameters}
          />
        );
      
      case 'chart':
        return (
          <AdvancedCharts
            key={visualizationId}
            data={data}
            chartType={recommendation.chartType as any}
            selectedParameters={selectedParameters}
          />
        );
      
      case 'map':
        return (
          <MapView
            key={visualizationId}
            data={data.results || []}
            columns={data.columns || []}
          />
        );
      
      case 'profile':
        return (
          <ProfileChart
            key={visualizationId}
            data={data.results || []}
            columns={data.columns || []}
          />
        );
      
      default:
        return null;
    }
  };

  if (!data?.results || data.results.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-8 text-center">
        <div className="text-gray-500">
          <svg className="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Data Available</h3>
          <p className="text-gray-600">Query results are empty or invalid</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Visualization Recommendations */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Smart Visualizations</h2>
            <p className="text-sm text-gray-600">
              AI-recommended visualizations based on your query: "{queryIntent}"
            </p>
          </div>
          
          {/* Parameter selector */}
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-700">Parameters:</label>
            <div className="flex space-x-2">
              {['temperature', 'salinity', 'pressure', 'density'].map(param => (
                <label key={param} className="flex items-center space-x-1">
                  <input
                    type="checkbox"
                    checked={selectedParameters.includes(param)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedParameters(prev => [...prev, param]);
                      } else {
                        setSelectedParameters(prev => prev.filter(p => p !== param));
                      }
                    }}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-600 capitalize">{param}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Recommendation cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {recommendations.map((rec, index) => {
            const visualizationId = `${rec.type}-${rec.chartType || 'default'}`;
            const isActive = activeVisualizations.includes(visualizationId);
            
            return (
              <div
                key={visualizationId}
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  isActive
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 bg-white hover:border-gray-300'
                }`}
                onClick={() => toggleVisualization(visualizationId)}
              >
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-medium text-gray-900">{rec.title}</h3>
                  <div className={`w-4 h-4 rounded-full border-2 ${
                    isActive ? 'border-blue-500 bg-blue-500' : 'border-gray-300'
                  }`}>
                    {isActive && <div className="w-full h-full rounded-full bg-white scale-50"></div>}
                  </div>
                </div>
                <p className="text-sm text-gray-600 mb-2">{rec.description}</p>
                <div className="text-xs text-gray-500">
                  <span className="font-medium">Why:</span> {rec.reasoning}
                </div>
                <div className="mt-2 text-xs text-gray-400">
                  Priority: {rec.priority}/10
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Active Visualizations */}
      <div className="space-y-6">
        {recommendations.map(rec => renderVisualization(rec))}
      </div>

      {/* No visualizations selected */}
      {activeVisualizations.length === 0 && (
        <div className="bg-gray-50 rounded-lg p-8 text-center">
          <div className="text-gray-500">
            <svg className="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Select Visualizations</h3>
            <p className="text-gray-600">Choose visualizations from the recommendations above to explore your data</p>
          </div>
        </div>
      )}
    </div>
  );
}
