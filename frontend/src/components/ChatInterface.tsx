'use client';

import { useState, useEffect, useRef } from 'react';
import { WebSocketChatClient, WebSocketMessage } from '@/lib/api';

interface ChatMessage {
    id: string;
    type: 'user' | 'ai' | 'status' | 'error';
    content: string;
    timestamp: Date;
    data?: any;
}

interface ChatInterfaceProps {
    onQueryResult?: (result: any) => void;
}

export default function ChatInterface({ onQueryResult }: ChatInterfaceProps) {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [isConnected, setIsConnected] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const wsClient = useRef<WebSocketChatClient | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Initialize WebSocket connection
        wsClient.current = new WebSocketChatClient(
            handleWebSocketMessage,
            setIsConnected
        );

        wsClient.current.connect();

        // Add welcome message
        setMessages([{
            id: 'welcome',
            type: 'ai',
            content: 'Hello! I\'m FloatChat, your oceanographic data assistant. Ask me about ARGO float data, temperature profiles, or spatial distributions.',
            timestamp: new Date()
        }]);

        // Cleanup on unmount
        return () => {
            wsClient.current?.disconnect();
        };
    }, []);

    useEffect(() => {
        // Auto-scroll to bottom
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleWebSocketMessage = (message: WebSocketMessage) => {
        const messageId = message.message_id;

        if (message.type === 'status') {
            // Update or add status message
            setMessages(prev => {
                const existingIndex = prev.findIndex(msg => msg.id === `status-${messageId}`);
                const statusMessage: ChatMessage = {
                    id: `status-${messageId}`,
                    type: 'status',
                    content: message.message || 'Processing...',
                    timestamp: new Date()
                };

                if (existingIndex >= 0) {
                    // Update existing status message
                    const updated = [...prev];
                    updated[existingIndex] = statusMessage;
                    return updated;
                } else {
                    // Add new status message
                    return [...prev, statusMessage];
                }
            });

        } else if (message.type === 'result') {
            setIsLoading(false);

            // Remove status message and add result
            setMessages(prev => prev.filter(msg => msg.id !== `status-${messageId}`));

            const result = message.data;
            let resultContent = '';

            if (result?.success) {
                resultContent = `Found ${result.result_count || 0} results in ${result.processing_time?.toFixed(2) || '0'}s`;
                if (result.sql_query) {
                    resultContent += `\n\nGenerated SQL: ${result.sql_query}`;
                }
            } else {
                resultContent = `Error: ${result?.error || 'Unknown error occurred'}`;
            }

            setMessages(prev => [...prev, {
                id: `result-${messageId}`,
                type: result?.success ? 'ai' : 'error',
                content: resultContent,
                timestamp: new Date(),
                data: result
            }]);

            // Pass result to parent component
            if (result && onQueryResult) {
                onQueryResult(result);
            }

        } else if (message.type === 'error') {
            setIsLoading(false);

            // Remove status message and add error
            setMessages(prev => prev.filter(msg => msg.id !== `status-${messageId}`));

            setMessages(prev => [...prev, {
                id: `error-${messageId}`,
                type: 'error',
                content: message.message || 'An error occurred',
                timestamp: new Date()
            }]);
        }
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();

        if (!input.trim()) return;
        if (!wsClient.current?.isConnected()) {
            alert('Not connected to server. Please wait for connection.');
            return;
        }

        // Add user message
        const userMessage: ChatMessage = {
            id: Date.now().toString(),
            type: 'user',
            content: input.trim(),
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setIsLoading(true);

        // Send query via WebSocket with correct format
        try {
            const messageId = wsClient.current.sendQuery(input.trim());

            // Add loading message with the correct message_id format
            const loadingMessage: ChatMessage = {
                id: `status-${messageId}`, // This matches the backend expectation
                type: 'status',
                content: 'Processing your query...',
                timestamp: new Date()
            };
            setMessages(prev => [...prev, loadingMessage]);

            setInput('');
        } catch (error) {
            console.error('Failed to send query:', error);
            setIsLoading(false);
            setMessages(prev => [...prev, {
                id: Date.now().toString(),
                type: 'error',
                content: 'Failed to send query. Please try again.',
                timestamp: new Date()
            }]);
        }
    };

    return (
        <div className="flex flex-col h-full max-h-96 border border-gray-200 rounded-lg bg-white shadow-sm">
            {/* Connection status */}
            <div className="px-4 py-2 border-b border-gray-100 bg-gray-50">
                <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-gray-800">FloatChat</h3>
                    <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                        <span className="text-xs text-gray-600">
                            {isConnected ? 'Connected' : 'Disconnected'}
                        </span>
                    </div>
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {messages.map((message) => (
                    <div
                        key={message.id}
                        className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                        <div
                            className={`max-w-[80%] p-3 rounded-lg ${message.type === 'user'
                                    ? 'bg-blue-500 text-white'
                                    : message.type === 'error'
                                        ? 'bg-red-100 text-red-800 border border-red-200'
                                        : message.type === 'status'
                                            ? 'bg-yellow-100 text-yellow-800 border border-yellow-200'
                                            : 'bg-gray-100 text-gray-800'
                                }`}
                        >
                            <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                            <div className="text-xs opacity-70 mt-1">
                                {message.timestamp.toLocaleTimeString()}
                            </div>
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form onSubmit={handleSubmit} className="p-4 border-t border-gray-100">
                <div className="flex space-x-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask about ARGO data..."
                        className="flex-1 px-3 py-2 border border-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-black"
                        disabled={isLoading || !isConnected}
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !isConnected || !input.trim()}
                        className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                    >
                        {isLoading ? 'Processing...' : 'Send'}
                    </button>
                </div>
            </form>
        </div>
    );
}