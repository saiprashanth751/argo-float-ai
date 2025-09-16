// components/layout/ChatLayout.tsx
'use client';

import { useState } from 'react';
import ChatInterface from './ChatInterface';
import ArtifactSidebar from '../artifacts/ArtifactSidebar';
import { Artifact, ChatMessage } from '@/lib/types';

export default function ChatLayout() {
  const [activeArtifact, setActiveArtifact] = useState<Artifact | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);

  const handleNewMessage = (message: ChatMessage) => {
    setMessages(prev => [...prev, message]);
    
    // If message has artifacts, add them to the artifacts list
    if (message.artifacts && message.artifacts.length > 0) {
      setArtifacts(prev => [...prev, ...message.artifacts!]);
    }
  };

  const handleArtifactSelect = (artifact: Artifact) => {
    setActiveArtifact(artifact);
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Chat Interface */}
      <div className="flex-1 flex flex-col border-r border-gray-200">
        <ChatInterface 
          messages={messages}
          onNewMessage={handleNewMessage}
          onArtifactSelect={handleArtifactSelect}
        />
      </div>

      {/* Artifact Sidebar */}
      <div className="w-1/2 min-w-[500px]">
        <ArtifactSidebar
          activeArtifact={activeArtifact}
          artifacts={artifacts}
          onArtifactSelect={handleArtifactSelect}
        />
      </div>
    </div>
  );
}