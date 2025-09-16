// components/chat/MessageBubble.tsx
import { ChatMessage, Artifact } from '@/lib/types';

interface MessageBubbleProps {
  message: ChatMessage;
  onArtifactSelect: (artifact: Artifact) => void;
}

export default function MessageBubble({ message, onArtifactSelect }: MessageBubbleProps) {
  const isUser = message.type === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-xs lg:max-w-md xl:max-w-lg rounded-lg p-3 ${
        isUser 
          ? 'bg-blue-600 text-white' 
          : 'bg-gray-100 text-gray-800'
      }`}>
        {/* Message content */}
        <p className="whitespace-pre-wrap">{message.content}</p>
        
        {/* Artifact badges */}
        {message.artifacts && message.artifacts.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {message.artifacts.map((artifact) => (
              <button
                key={artifact.id}
                onClick={() => onArtifactSelect(artifact)}
                className={`text-xs px-2 py-1 rounded-full ${
                  isUser
                    ? 'bg-blue-500 text-white hover:bg-blue-400'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {getArtifactIcon(artifact.type)} {artifact.title}
              </button>
            ))}
          </div>
        )}
        
        {/* Metadata */}
        {message.metadata && (
          <div className="text-xs mt-1 opacity-70">
            {message.metadata.processingTime && (
              <span>{message.metadata.processingTime.toFixed(2)}s</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

const getArtifactIcon = (type: string) => {
  const icons: Record<string, string> = {
    'visualization': '📊',
    'table': '📋',
    'map': '🗺️',
    'analysis': '🔍'
  };
  return icons[type] || '📄';
};