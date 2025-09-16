// components/chat/MessageList.tsx
import { ChatMessage, Artifact } from '@/lib/types';
import MessageBubble from './MessageBubble';

interface MessageListProps {
  messages: ChatMessage[];
  onArtifactSelect: (artifact: Artifact) => void;
  isLoading: boolean;
}

export default function MessageList({ messages, onArtifactSelect, isLoading }: MessageListProps) {
  return (
    <div className="space-y-4">
      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          message={message}
          onArtifactSelect={onArtifactSelect}
        />
      ))}
      
      {isLoading && (
        <div className="flex justify-start">
          <div className="bg-gray-100 rounded-lg p-3 max-w-xs">
            <div className="flex space-x-2">
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}