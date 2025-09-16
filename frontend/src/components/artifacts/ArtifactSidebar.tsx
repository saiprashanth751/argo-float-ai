// components/artifacts/ArtifactSidebar.tsx
'use client';

import { Artifact } from '@/lib/types';
import ArtifactTabs from './ArtifactTabs';
import ArtifactViewer from './ArtifactViewer';

interface ArtifactSidebarProps {
  activeArtifact: Artifact | null;
  artifacts: Artifact[];
  onArtifactSelect: (artifact: Artifact) => void;
}

export default function ArtifactSidebar({ 
  activeArtifact, 
  artifacts, 
  onArtifactSelect 
}: ArtifactSidebarProps) {
  // Group artifacts by type
  const artifactsByType = artifacts.reduce((acc, artifact) => {
    if (!acc[artifact.type]) {
      acc[artifact.type] = [];
    }
    acc[artifact.type].push(artifact);
    return acc;
  }, {} as Record<string, Artifact[]>);

  return (
    <div className="h-full bg-white border-l border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-800">Analysis Artifacts</h2>
        <p className="text-sm text-gray-600">Visualizations, maps, and data from your queries</p>
      </div>

      {/* Tabs */}
      <ArtifactTabs 
        artifactsByType={artifactsByType}
        activeArtifact={activeArtifact}
        onArtifactSelect={onArtifactSelect}
      />

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {activeArtifact ? (
          <ArtifactViewer artifact={activeArtifact} />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            Select an artifact to view details
          </div>
        )}
      </div>
    </div>
  );
}