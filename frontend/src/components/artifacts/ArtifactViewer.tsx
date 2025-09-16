// components/artifacts/ArtifactViewer.tsx
'use client';

import { Artifact } from '@/lib/types';
import { useState } from 'react';
import dynamic from 'next/dynamic';

// Dynamically import all visualization components with no SSR
const PlotlyChart = dynamic(() => import('../visualizations/PlotlyChart'), { 
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-full">Loading chart...</div>
});

const DataTable = dynamic(() => import('../visualizations/DataTable'), { 
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-full">Loading table...</div>
});

const MapView = dynamic(() => import('../visualizations/MapView'), { 
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-full">Loading map...</div>
});

const GlobeVisualization = dynamic(() => import('../visualizations/GlobeVisualization'), { 
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-full">Loading globe...</div>
});

interface ArtifactViewerProps {
  artifact: Artifact;
}

export default function ArtifactViewer({ artifact }: ArtifactViewerProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  const renderArtifact = () => {
    if (!artifact) {
      return (
        <div className="flex items-center justify-center h-full text-gray-500">
          Select an artifact to view details
        </div>
      );
    }

    switch (artifact.component) {
      case 'PlotlyChart':
        return <PlotlyChart data={artifact.data} title={artifact.title} />;
      case 'DataTable':
        return <DataTable data={artifact.data} />;
      case 'MapView':
        return <MapView data={artifact.data} />;
      case 'Globe':
        return <GlobeVisualization data={artifact.data} />;
      default:
        return (
          <div className="flex items-center justify-center h-full">
            <p className="text-gray-500">Unsupported visualization type: {artifact.component}</p>
          </div>
        );
    }
  };

  const handleExport = () => {
    const exportData = JSON.stringify(artifact.data, null, 2);
    const blob = new Blob([exportData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `${artifact.title.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  return (
    <div className={`h-full flex flex-col ${isFullscreen ? 'fixed inset-0 z-50 bg-white' : ''}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200 flex justify-between items-center">
        <div>
          <h3 className="text-lg font-semibold text-gray-800">{artifact.title}</h3>
          {artifact.metadata?.description && (
            <p className="text-sm text-gray-600 mt-1">{artifact.metadata.description}</p>
          )}
        </div>
        
        {isFullscreen && (
          <button
            onClick={handleFullscreen}
            className="px-3 py-1 text-sm bg-gray-100 rounded hover:bg-gray-200"
          >
            Exit Fullscreen
          </button>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 p-4 overflow-auto">
        {renderArtifact()}
      </div>

      {/* Footer with actions */}
      <div className="p-4 border-t border-gray-200 flex justify-between">
        <div className="text-sm text-gray-500">
          {artifact.metadata?.createdAt && (
            <span>Created: {new Date(artifact.metadata.createdAt).toLocaleString()}</span>
          )}
        </div>
        <div className="space-x-2">
          <button 
            onClick={handleExport}
            className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
          >
            Export JSON
          </button>
          <button 
            onClick={handleFullscreen}
            className="px-3 py-1 text-sm bg-gray-100 rounded hover:bg-gray-200"
          >
            {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          </button>
        </div>
      </div>
    </div>
  );
}