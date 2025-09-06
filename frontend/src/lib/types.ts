// lib/types.ts

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

export interface ChatMessage {
  id: string;
  type: 'user' | 'bot' | 'system';
  content: string;
  timestamp: Date;
  query_result?: QueryResponse;
}

export interface MeasurementData {
  pressure: number;
  temperature?: number;
  salinity?: number;
  oxygen?: number;
  chlorophyll?: number;
  backscatter?: number;
}