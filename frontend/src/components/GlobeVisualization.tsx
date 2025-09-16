'use client';

import { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';

// Dynamically import Globe to avoid SSR issues
const Globe = dynamic(() => import('react-globe.gl'), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg">Loading 3D Globe...</div>
});

interface GlobeVisualizationProps {
  data: Array<Record<string, any>>;
  columns: string[];
  visualizationType?: 'trajectories' | 'distribution' | 'heatmap';
  selectedParameters?: string[];
}

interface GlobeDataPoint {
  lat: number;
  lng: number;
  alt?: number;
  color?: string;
  size?: number;
  label?: string;
  platform?: string;
  temperature?: number;
  salinity?: number;
  pressure?: number;
  date?: string;
}

export default function GlobeVisualization({ 
  data, 
  columns, 
  visualizationType = 'distribution',
  selectedParameters = ['temperature', 'salinity', 'pressure']
}: GlobeVisualizationProps) {
  const globeRef = useRef<any>(null);
  const [globeData, setGlobeData] = useState<GlobeDataPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedParameter, setSelectedParameter] = useState<string>(selectedParameters[0] || 'temperature');

  // Color scales for different parameters
  const colorScales = {
    temperature: {
      min: 0,
      max: 35,
      colors: ['#0066cc', '#00aa00', '#ffaa00', '#ff4444'], // Blue to Red
      label: 'Temperature (°C)'
    },
    salinity: {
      min: 30,
      max: 40,
      colors: ['#0066cc', '#00aa00', '#ffaa00', '#ff4444'],
      label: 'Salinity (PSU)'
    },
    pressure: {
      min: 0,
      max: 2000,
      colors: ['#ff4444', '#ffaa00', '#00aa00', '#0066cc'], // Red to Blue (surface to deep)
      label: 'Pressure (dbar)'
    }
  };

  // Process data for globe visualization
  useEffect(() => {
    if (!data || data.length === 0) {
      setGlobeData([]);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    
    // Limit data points for performance (max 1000 points)
    const limitedData = data.length > 1000 ? data.slice(0, 1000) : data;

    // Find coordinate columns
    const latColumn = columns.find(col => col.toLowerCase().includes('lat'));
    const lonColumn = columns.find(col => col.toLowerCase().includes('lon'));
    const platformColumn = columns.find(col => col.toLowerCase().includes('platform'));
    const dateColumn = columns.find(col => col.toLowerCase().includes('date') || col.toLowerCase().includes('time'));

    if (!latColumn || !lonColumn) {
      setIsLoading(false);
      return;
    }

    const processedData: GlobeDataPoint[] = [];
    const platformGroups: { [key: string]: GlobeDataPoint[] } = {};

    // Group data by platform for trajectory visualization
    limitedData.forEach((row) => {
      const lat = parseFloat(row[latColumn]);
      const lon = parseFloat(row[lonColumn]);
      
      if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
        return;
      }

      const platform = platformColumn ? row[platformColumn] : 'unknown';
      const date = dateColumn ? row[dateColumn] : null;

      // Get parameter values
      const temperature = row.temperature ? parseFloat(row.temperature) : null;
      const salinity = row.salinity ? parseFloat(row.salinity) : null;
      const pressure = row.pressure ? parseFloat(row.pressure) : null;

      const point: GlobeDataPoint = {
        lat,
        lng: lon,
        alt: pressure ? pressure / 10 : 0, // Convert pressure to altitude (negative for depth)
        platform,
        temperature,
        salinity,
        pressure,
        date,
        label: `${platform || 'Unknown'} - ${date || 'No date'}`
      };

      processedData.push(point);

      // Group by platform for trajectories
      if (platform) {
        if (!platformGroups[platform]) {
          platformGroups[platform] = [];
        }
        platformGroups[platform].push(point);
      }
    });

    setGlobeData(processedData);
    setIsLoading(false);
  }, [data, columns, visualizationType]);

  // Get color for data point based on selected parameter
  const getPointColor = (point: GlobeDataPoint): string => {
    const scale = colorScales[selectedParameter as keyof typeof colorScales];
    if (!scale || !point[selectedParameter as keyof GlobeDataPoint]) {
      return '#3388ff'; // Default blue
    }

    const value = point[selectedParameter as keyof GlobeDataPoint] as number;
    const normalizedValue = Math.max(0, Math.min(1, (value - scale.min) / (scale.max - scale.min)));
    
    // Simple color interpolation
    if (normalizedValue < 0.33) return scale.colors[0];
    if (normalizedValue < 0.66) return scale.colors[1];
    if (normalizedValue < 0.9) return scale.colors[2];
    return scale.colors[3];
  };

  // Generate trajectory arcs for platform paths
  const generateTrajectories = () => {
    const platformGroups: { [key: string]: GlobeDataPoint[] } = {};
    
    globeData.forEach(point => {
      if (point.platform) {
        if (!platformGroups[point.platform]) {
          platformGroups[point.platform] = [];
        }
        platformGroups[point.platform].push(point);
      }
    });

    const arcs: Array<{ startLat: number; startLng: number; endLat: number; endLng: number; color: string }> = [];
    
    Object.values(platformGroups).forEach(platformData => {
      if (platformData.length < 2) return;
      
      // Sort by date if available
      platformData.sort((a, b) => {
        if (a.date && b.date) {
          return new Date(a.date).getTime() - new Date(b.date).getTime();
        }
        return 0;
      });

      // Create arcs between consecutive points
      for (let i = 0; i < platformData.length - 1; i++) {
        const start = platformData[i];
        const end = platformData[i + 1];
        
        arcs.push({
          startLat: start.lat,
          startLng: start.lng,
          endLat: end.lat,
          endLng: end.lng,
          color: getPointColor(start)
        });
      }
    });

    return arcs;
  };

  const trajectories = visualizationType === 'trajectories' ? generateTrajectories() : [];

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-8">
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-600">Processing oceanographic data...</p>
          </div>
        </div>
      </div>
    );
  }

  if (globeData.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-8 text-center">
        <div className="text-gray-500">
          <svg className="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Geographic Data</h3>
          <p className="text-gray-600">No valid geographic coordinates found in the dataset</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      {/* Header with controls */}
      <div className="p-6 border-b">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">3D Ocean Globe</h3>
            <p className="text-sm text-gray-600">
              {globeData.length} data points • {visualizationType} view
            </p>
          </div>
          
          {/* Parameter selector */}
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-700">Color by:</label>
            <select
              value={selectedParameter}
              onChange={(e) => setSelectedParameter(e.target.value)}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {selectedParameters.map(param => (
                <option key={param} value={param}>
                  {colorScales[param as keyof typeof colorScales]?.label || param}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Visualization type selector */}
        <div className="flex space-x-2">
          {['distribution', 'trajectories', 'heatmap'].map(type => (
            <button
              key={type}
              onClick={() => {/* This would be passed as prop to parent */}}
              className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                visualizationType === type
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {type.charAt(0).toUpperCase() + type.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Globe container */}
      <div className="relative">
        <div style={{ height: '500px', width: '100%' }}>
          <Globe
            ref={globeRef}
            globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
            backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"
            pointsData={globeData}
            pointLat="lat"
            pointLng="lng"
            pointAltitude="alt"
            pointColor={() => getPointColor(globeData[0])}
            pointRadius={2}
            pointResolution={8}
            arcsData={trajectories}
            arcStartLat="startLat"
            arcStartLng="startLng"
            arcEndLat="endLat"
            arcEndLng="endLng"
            arcColor="color"
            arcDashLength={0.4}
            arcDashGap={0.2}
            arcDashAnimateTime={2000}
            onPointClick={(point: any) => {
              console.log('Clicked point:', point);
            }}
            onPointHover={(point: any) => {
              if (point) {
                console.log('Hovered point:', point);
              }
            }}
          />
        </div>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-white bg-opacity-90 rounded-lg p-3 shadow-sm">
          <div className="text-sm font-medium text-gray-700 mb-2">
            {colorScales[selectedParameter as keyof typeof colorScales]?.label}
          </div>
          <div className="flex items-center space-x-2">
            {colorScales[selectedParameter as keyof typeof colorScales]?.colors.map((color, index) => (
              <div key={index} className="flex items-center space-x-1">
                <div 
                  className="w-4 h-4 rounded-full border border-gray-300"
                  style={{ backgroundColor: color }}
                />
                <span className="text-xs text-gray-600">
                  {index === 0 ? 'Low' : index === 3 ? 'High' : ''}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
