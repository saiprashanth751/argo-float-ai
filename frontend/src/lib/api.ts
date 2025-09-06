// lib/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export interface QueryRequest {
    query: string;
    include_sql?: boolean;
    limit?: number;
}

export interface QueryResponse {
    success: boolean;
    query: string;
    sql_query?: string;
    results?: Array<Record<string, any>>;
    result_count: number;
    columns: string[];
    processing_time: number;
    error?: string;
    metadata: Record<string, any>;
}

export interface FloatInfo {
    platform_number: string;
    cycle_number: number;
    date: string;
    latitude: number;
    longitude: number;
    project_name: string;
    institution: string;
    measurement_count: number;
}

export interface DatabaseStats {
    total_floats: number;
    total_measurements: number;
    total_projects: number;
    date_range: {
        earliest: string | null;
        latest: string | null;
    };
    averages: {
        temperature: number | null;
        salinity: number | null;
    };
    timestamp: string;
}

// WebSocket message types
export interface WebSocketMessage {
    type: 'status' | 'result' | 'error' | 'connection';
    message?: string;
    stage?: string;
    data?: QueryResponse;
    message_id?: string;
    timestamp?: string;
}

export interface WebSocketQuery {
    type: 'query';
    message: string;
    message_id: string;
}

class FloatChatAPI {
    private baseURL: string;

    constructor(baseURL: string = API_BASE_URL) {
        this.baseURL = baseURL;
    }

    async processQuery(request: QueryRequest): Promise<QueryResponse> {
        const response = await fetch(`${this.baseURL}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    async getFloats(filters?: {
        region?: string;
        date_start?: string;
        date_end?: string;
        platform_numbers?: string[];
        limit?: number;
    }): Promise<FloatInfo[]> {
        const params = new URLSearchParams();

        if (filters) {
            Object.entries(filters).forEach(([key, value]) => {
                if (value !== undefined) {
                    if (Array.isArray(value)) {
                        value.forEach(v => params.append(key, v));
                    } else {
                        params.append(key, String(value));
                    }
                }
            });
        }

        const response = await fetch(`${this.baseURL}/api/floats?${params}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    async getStats(): Promise<DatabaseStats> {
        const response = await fetch(`${this.baseURL}/api/stats`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    async checkHealth(): Promise<{ status: string; database: string; rag_system: string }> {
        const response = await fetch(`${this.baseURL}/api/health`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }
}

export class WebSocketChatClient {
    private ws: WebSocket | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 1000;
    private connectionUrl: string;

    constructor(
        private onMessage: (message: WebSocketMessage) => void,
        private onConnectionChange: (connected: boolean) => void
    ) {
        this.connectionUrl = WS_BASE_URL.replace('http', 'ws').replace('https', 'wss') + '/ws/chat';
    }

    connect(): void {
        try {
            console.log(`Connecting to WebSocket: ${this.connectionUrl}`);
            this.ws = new WebSocket(this.connectionUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected successfully');
                this.reconnectAttempts = 0;
                this.onConnectionChange(true);
            };

            this.ws.onmessage = (event) => {
                try {
                    const message: WebSocketMessage = JSON.parse(event.data);
                    this.onMessage(message);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error, event.data);
                }
            };

            this.ws.onclose = (event) => {
                console.log(`WebSocket disconnected. Code: ${event.code}, Reason: ${event.reason}`);
                this.onConnectionChange(false);
                this.ws = null;
                this.attemptReconnect();
            };

            this.ws.onerror = (error) => {
                // Browser often sends empty error objects - this is normal
                console.log('WebSocket connection event (may be false error)');

                // Try to get more info if available
                if (this.ws) {
                    console.log('WebSocket readyState:', this.ws.readyState);
                }
            };

        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.onConnectionChange(false);
            this.attemptReconnect();
        }
    }

    private attemptReconnect(): void {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        }
    }

    sendQuery(query: string): string {
        const messageId = Date.now().toString() + Math.random().toString(36).substr(2, 9);

        if (this.ws?.readyState === WebSocket.OPEN) {
            const message = {
                query: query,
                message_id: messageId
            };

            this.ws.send(JSON.stringify(message));
            return messageId;
        } else {
            throw new Error('WebSocket is not connected');
        }
    }

    disconnect(): void {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }
}

export const api = new FloatChatAPI();