// lib/types.ts

// API Request/Response Types
export interface QueryRequest {
  query: string;
  include_sql?: boolean;
  limit?: number;
  response_format?: ResponseFormat;
}

export interface ResponseFormat {
  target_audience?: 'government_official' | 'researcher' | 'maritime_industry' | 'general_public';
  complexity_level?: 'basic' | 'intermediate' | 'advanced' | 'expert';
  include_visualizations?: boolean;
  include_raw_data?: boolean;
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
  // Intelligent response fields
  narrative_response?: string;
  classification?: QueryClassification;
  insights?: ScientificInsights;
  visualizations?: VisualizationData[];
  recommendations?: Recommendation[];
  export_data?: ExportData;
}

export interface QueryClassification {
  intent: QueryIntent;
  complexity: ComplexityLevel;
  confidence: number;
  parameters: string[];
  context: QueryContext;
  suggested_approach: string;
  required_calculations: string[];
}

export interface QueryContext {
  depth_range?: [number, number];
  spatial_bounds?: {
    min_lat: number;
    max_lat: number;
    min_lon: number;
    max_lon: number;
  };
  temporal_range?: [string, string];
  physical_processes: string[];
  data_quality_requirements: string;
}

export interface ScientificInsights {
  key_findings: string[];
  summary: string;
  physical_interpretation: string;
  data_quality_notes: string;
  statistical_significance?: string;
  uncertainty_estimation?: string;
}

export interface VisualizationData {
  type: string;
  parameter: string;
  title: string;
  plotly_json: string;
  data_points: number;
  platforms?: number;
  depth_range?: [number, number];
  statistics?: Record<string, number>;
  time_range?: [string, string];
}

export interface Recommendation {
  type: string;
  priority: 'low' | 'medium' | 'high';
  title: string;
  description: string;
  action: string;
}

export interface ExportData {
  csv?: ExportFormat;
  json?: ExportFormat;
  statistics?: ExportFormat;
}

export interface ExportFormat {
  format: string;
  data: string;
  filename: string;
  size_bytes: number;
}

// Database Types
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
  total_profiles: number;
  total_measurements: number;
  unique_platforms: number;
  date_range: {
    earliest: string | null;
    latest: string | null;
  };
  averages: {
    temperature: number | null;
    salinity: number | null;
  };
  timestamp: string;
  system_version: string;
}

export interface MeasurementData {
  pressure: number;
  temperature?: number;
  salinity?: number;
  oxygen?: number;
  chlorophyll?: number;
  backscatter?: number;
  profile_id?: string;
  platform_number?: string;
  profile_date?: string;
  latitude?: number;
  longitude?: number;
}

// Chat & Artifact Types
export interface ChatMessage {
  id: string;
  type: 'user' | 'bot' | 'system';
  content: string;
  artifacts?: Artifact[];
  metadata?: ProcessingMetadata;
  timestamp: Date;
  query_result?: QueryResponse;
}

export interface Artifact {
  id: string;
  type: ArtifactType;
  title: string;
  data: any;
  component: ArtifactComponent;
  metadata?: ArtifactMetadata;
}

export interface ArtifactMetadata {
  queryId?: string;
  createdAt: Date;
  description?: string;
  processingTime?: number;
  confidence?: number;
  parameters?: string[];
  spatialBounds?: {
    minLat: number;
    maxLat: number;
    minLon: number;
    maxLon: number;
  };
  depthRange?: {
    min: number;
    max: number;
  };
  timeRange?: {
    start: Date;
    end: Date;
  };
  dataQuality?: {
    score: number;
    flags: string[];
    notes: string;
  };
}

export interface ProcessingMetadata {
  processingTime?: number;
  confidence?: number;
  intent?: QueryIntent;
  complexity?: ComplexityLevel;
  parameters?: string[];
  sqlValidation?: SQLValidationResult;
  dataStatistics?: DataStatistics;
}

export interface SQLValidationResult {
  original_sql: string;
  corrected_sql: string;
  is_valid: boolean;
  errors_found: string[];
  corrections_applied: string[];
  schema_compatibility: string;
}

export interface DataStatistics {
  total_records: number;
  column_stats: Record<string, ColumnStatistics>;
  quality_metrics: QualityMetrics;
}

export interface ColumnStatistics {
  data_type: string;
  min: number;
  max: number;
  mean: number;
  std: number;
  null_count: number;
  unique_count: number;
}

export interface QualityMetrics {
  completeness: number;
  consistency: number;
  accuracy: number;
  timeliness: number;
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'status' | 'result' | 'error' | 'connection' | 'query';
  message?: string;
  stage?: string;
  data?: QueryResponse;
  message_id?: string;
  timestamp?: string;
  query?: string;
}

export interface WebSocketQuery {
  type: 'query';
  message: string;
  message_id: string;
}

// Enums and Type Guards
export type QueryIntent = 
  | 'profile_analysis' 
  | 'spatial_mapping' 
  | 'temporal_trends' 
  | 'statistical_summary' 
  | 'exploration'
  | 'quality_assessment'
  | 'comparative_analysis';

export type ComplexityLevel = 
  | 'basic' 
  | 'intermediate' 
  | 'advanced' 
  | 'expert';

export type ArtifactType = 
  | 'visualization' 
  | 'table' 
  | 'map' 
  | 'analysis'
  | 'recommendation'
  | 'export';

export type ArtifactComponent = 
  | 'PlotlyChart' 
  | 'DataTable' 
  | 'MapView' 
  | 'Globe'
  | 'StatisticsTable'
  | 'RecommendationList'
  | 'ExportPanel';

export type VisualizationType =
  | 'profile'
  | 'spatial_profiles'
  | 'statistical'
  | 'temporal'
  | 'basic'
  | 'comparison'
  | 'quality'
  | 'correlation';

// Type Guards
export function isProfileVisualization(data: any): data is { pressure: number; parameter_value: number } {
  return data && typeof data.pressure === 'number' && typeof data.parameter_value === 'number';
}

export function isSpatialData(data: any): data is { latitude: number; longitude: number } {
  return data && typeof data.latitude === 'number' && typeof data.longitude === 'number';
}

export function isTemporalData(data: any): data is { date: string; value: number } {
  return data && data.date && typeof data.value === 'number';
}

// Helper Types for Component Props
export interface VisualizationProps {
  data: any;
  title?: string;
  config?: Record<string, any>;
  onExport?: (format: string) => void;
}

export interface ChartConfig {
  type: VisualizationType;
  xAxis?: string;
  yAxis?: string;
  colorBy?: string;
  sizeBy?: string;
  title: string;
  description?: string;
  interactive?: boolean;
  exportable?: boolean;
}

// API Response Types for Frontend
export interface APIHealthStatus {
  status: string;
  timestamp: string;
  version: string;
  components: {
    database: {
      status: string;
      total_profiles?: number;
      total_measurements?: number;
      error?: string;
    };
    rag_system: {
      status: string;
    };
    intelligent_system: {
      status: string;
    };
    fallback_system: {
      status: string;
    };
  };
}

// Error Types
export interface APIError {
  code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
}

export interface ValidationError {
  field: string;
  message: string;
  code: string;
}

// Event Types for Analytics
export interface ChatEvent {
  type: 'query_submitted' | 'response_received' | 'artifact_viewed' | 'export_triggered';
  timestamp: Date;
  details: Record<string, any>;
  session_id: string;
  user_id?: string;
}

// Configuration Types
export interface AppConfig {
  api: {
    base_url: string;
    websocket_url: string;
    timeout: number;
    retry_attempts: number;
  };
  visualization: {
    max_data_points: number;
    default_chart_height: number;
    color_schemes: string[];
    animation_enabled: boolean;
  };
  performance: {
    cache_enabled: boolean;
    cache_ttl: number;
    debounce_time: number;
  };
}
