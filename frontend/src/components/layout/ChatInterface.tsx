// components/chat/ChatInterface.tsx
'use client';

import { useState, useRef, useEffect } from 'react';
import MessageList from '../chat/MessageList';
import MessageInput from '../chat/MessageInput';
import { ChatMessage, Artifact } from '@/lib/types';
import { WebSocketChatClient } from '@/lib/api';

interface ChatInterfaceProps {
    messages: ChatMessage[];
    onNewMessage: (message: ChatMessage) => void;
    onArtifactSelect: (artifact: Artifact) => void;
}

export default function ChatInterface({
    messages,
    onNewMessage,
    onArtifactSelect
}: ChatInterfaceProps) {
    const [isConnected, setIsConnected] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const wsClientRef = useRef<WebSocketChatClient | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Initialize WebSocket connection
        wsClientRef.current = new WebSocketChatClient(
            handleWebSocketMessage,
            handleConnectionChange
        );
        wsClientRef.current.connect();

        return () => {
            if (wsClientRef.current) {
                wsClientRef.current.disconnect();
            }
        };
    }, []);

    useEffect(() => {
        // Scroll to bottom when new messages arrive
        scrollToBottom();
    }, [messages]);

    const handleConnectionChange = (connected: boolean) => {
        setIsConnected(connected);
    };

    const handleWebSocketMessage = (message: any) => {
        switch (message.type) {
            case 'status':
                // Handle status updates (e.g., "Processing your query...")
                if (message.stage === 'starting') {
                    setIsLoading(true);
                }
                break;

            case 'result':
                setIsLoading(false);
                if (message.data) {
                    processBotResponse(message.data);
                }
                break;

            case 'error':
                setIsLoading(false);
                // Create error message
                const errorMessage: ChatMessage = {
                    id: `error-${Date.now()}`,
                    type: 'bot',
                    content: `Error: ${message.message || 'Unknown error occurred'}`,
                    timestamp: new Date()
                };
                onNewMessage(errorMessage);
                break;
        }
    };

    const processBotResponse = (response: any) => {
        // Create the main bot message
        const botMessage: ChatMessage = {
            id: `msg-${Date.now()}`,
            type: 'bot',
            content: response.narrative_response || 'Here are your results:',
            timestamp: new Date(),
            metadata: {
                processingTime: response.processing_time,
                confidence: response.classification?.confidence,
                intent: response.classification?.intent,
                complexity: response.classification?.complexity
            }
        };

        // Create artifacts from visualizations
        if (response.visualizations && Array.isArray(response.visualizations)) {
            botMessage.artifacts = response.visualizations.map((viz: any, index: number) => ({
                id: `artifact-${Date.now()}-${index}`,
                type: mapVizTypeToArtifactType(viz.type),
                title: viz.title || `Visualization ${index + 1}`,
                data: viz,
                component: mapVizTypeToComponent(viz.type),
                metadata: {
                    createdAt: new Date(),
                    queryId: response.metadata?.queryId
                }
            }));
        }

        onNewMessage(botMessage);
    };

    const mapVizTypeToArtifactType = (vizType: string): Artifact['type'] => {
        const typeMap: Record<string, Artifact['type']> = {
            'profile': 'visualization',
            'spatial_profiles': 'map',
            'statistical': 'analysis',
            'temporal': 'visualization',
            'basic': 'visualization'
        };
        return typeMap[vizType] || 'visualization';
    };

    const mapVizTypeToComponent = (vizType: string): Artifact['component'] => {
        const componentMap: Record<string, Artifact['component']> = {
            'profile': 'PlotlyChart',
            'spatial_profiles': 'MapView',
            'statistical': 'PlotlyChart',
            'temporal': 'PlotlyChart',
            'basic': 'PlotlyChart'
        };
        return componentMap[vizType] || 'PlotlyChart';
    };

    const handleSendMessage = (content: string) => {
        if (!wsClientRef.current || !isConnected) return;

        // Create user message
        const userMessage: ChatMessage = {
            id: `user-${Date.now()}`,
            type: 'user',
            content,
            timestamp: new Date()
        };

        onNewMessage(userMessage);
        setIsLoading(true);

        // Send via WebSocket
        try {
            wsClientRef.current.sendQuery(content);
        } catch (error) {
            console.error('Failed to send message:', error);
            setIsLoading(false);

            const errorMessage: ChatMessage = {
                id: `error-${Date.now()}`,
                type: 'bot',
                content: 'Connection error. Please try again.',
                timestamp: new Date()
            };
            onNewMessage(errorMessage);
        }
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    return (
        <div className="flex flex-col h-full bg-white">
            {/* Header */}
            <div className="p-4 border-b border-gray-200">
                <h1 className="text-xl font-semibold text-gray-800">FloatChat</h1>
                <div className="flex items-center mt-1">
                    <div className={`w-2 h-2 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'
                        }`} />
                    <span className="text-sm text-gray-600">
                        {isConnected ? 'Connected' : 'Disconnected'}
                    </span>
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4">
                <MessageList
                    messages={messages}
                    onArtifactSelect={onArtifactSelect}
                    isLoading={isLoading}
                />
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-4 border-t border-gray-200">
                <MessageInput
                    onSendMessage={handleSendMessage}
                    disabled={!isConnected || isLoading}
                />
            </div>
        </div>
    );
}