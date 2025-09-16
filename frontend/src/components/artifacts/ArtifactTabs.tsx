// components/artifacts/ArtifactTabs.tsx
'use client';

import { Artifact } from '@/lib/types';
import { useState } from 'react';

interface ArtifactTabsProps {
  artifactsByType: Record<string, Artifact[]>;
  activeArtifact: Artifact | null;
  onArtifactSelect: (artifact: Artifact) => void;
}

const tabLabels: Record<string, string> = {
  'visualization': 'Visualizations',
  'table': 'Data',
  'map': 'Maps',
  'analysis': 'Analysis'
};

export default function ArtifactTabs({ 
  artifactsByType, 
  activeArtifact, 
  onArtifactSelect 
}: ArtifactTabsProps) {
  const [activeTab, setActiveTab] = useState<string>('visualization');

  return (
    <div className="border-b border-gray-200">
      {/* Tab headers */}
      <div className="flex px-4">
        {Object.entries(tabLabels).map(([type, label]) => (
          <button
            key={type}
            onClick={() => setActiveTab(type)}
            className={`px-4 py-2 text-sm font-medium border-b-2 ${
              activeTab === type
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            {label} ({artifactsByType[type]?.length || 0})
          </button>
        ))}
      </div>

      {/* Tab content - artifact list */}
      <div className="p-4 bg-gray-50 max-h-40 overflow-y-auto">
        {artifactsByType[activeTab]?.length > 0 ? (
          <div className="space-y-2">
            {artifactsByType[activeTab].map((artifact) => (
              <button
                key={artifact.id}
                onClick={() => onArtifactSelect(artifact)}
                className={`w-full text-left p-2 rounded text-sm ${
                  activeArtifact?.id === artifact.id
                    ? 'bg-blue-100 text-blue-800'
                    : 'hover:bg-gray-100'
                }`}
              >
                {artifact.title}
              </button>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500">No {tabLabels[activeTab].toLowerCase()} available</p>
        )}
      </div>
    </div>
  );
}