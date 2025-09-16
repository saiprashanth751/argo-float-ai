// components/visualizations/GlobeVisualization.tsx
'use client';

import { useEffect, useRef, useState } from 'react';
import Globe from 'react-globe.gl';

interface GlobeVisualizationProps {
  data: any;
}

interface GlobePoint {
  lat: number;
  lng: number;
  value?: number;
  [key: string]: any;
}

export default function GlobeVisualization({ data }: GlobeVisualizationProps) {
  const globeRef = useRef<any>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    if (!globeRef.current || !data) return;

    // Process data for the globe
    const points = processDataForGlobe(data);
    
    // Configure globe
    globeRef.current
      .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
      .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
      .pointsData(points)
      .pointAltitude(0.1)
      .pointRadius(0.5)
      .pointColor(() => '#ff0000')
      .pointLabel((d: any) => `
        <div class="p-2 bg-white rounded shadow-lg">
          <strong>Lat:</strong> ${d.lat.toFixed(2)}<br/>
          <strong>Lon:</strong> ${d.lng.toFixed(2)}<br/>
          <strong>Value:</strong> ${d.value || 'N/A'}
        </div>
      `);

    setIsLoaded(true);
  }, [data]);

  const processDataForGlobe = (rawData: any): GlobePoint[] => {
    // Handle different data structures
    const processedData = Array.isArray(rawData) ? rawData : rawData.results || rawData.data || [];
    
    return processedData
      .filter((item: any) => item.latitude && item.longitude)
      .map((item: any) => ({
        lat: item.latitude,
        lng: item.longitude,
        value: item.temperature || item.salinity || item.value,
        ...item
      }));
  };

  return (
    <div className="w-full h-full relative">
      <Globe
        ref={globeRef}
        width={800}
        height={600}
        backgroundColor="rgba(0,0,0,0)"
        onGlobeReady={() => console.log('Globe ready')}
      />
      
      {!isLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-80">
          <div className="text-gray-600">Loading globe...</div>
        </div>
      )}
    </div>
  );
}