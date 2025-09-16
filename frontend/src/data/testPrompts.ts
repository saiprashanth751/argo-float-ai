// Comprehensive test prompts for FloatChat visualization system
// These prompts test different query intents and visualization scenarios

export interface TestPrompt {
  id: string;
  category: string;
  query: string;
  expectedIntent: string;
  expectedComplexity: string;
  expectedVisualizations: string[];
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

export const TEST_PROMPTS: TestPrompt[] = [
  // === SPATIAL MAPPING PROMPTS ===
  {
    id: 'spatial-001',
    category: 'Spatial Mapping',
    query: 'Show me all ARGO floats in the Arabian Sea',
    expectedIntent: 'spatial_mapping',
    expectedComplexity: 'basic',
    expectedVisualizations: ['globe', 'map', 'heatmap'],
    description: 'Basic spatial query for geographic distribution',
    difficulty: 'beginner'
  },
  {
    id: 'spatial-002',
    category: 'Spatial Mapping',
    query: 'Map temperature distribution across the Indian Ocean',
    expectedIntent: 'spatial_mapping',
    expectedComplexity: 'intermediate',
    expectedVisualizations: ['globe', 'heatmap', 'map'],
    description: 'Spatial analysis with parameter mapping',
    difficulty: 'intermediate'
  },
  {
    id: 'spatial-003',
    category: 'Spatial Mapping',
    query: 'Where are the warmest waters in the Indian Ocean region?',
    expectedIntent: 'spatial_mapping',
    expectedComplexity: 'intermediate',
    expectedVisualizations: ['globe', 'heatmap', 'scatter'],
    description: 'Spatial analysis with parameter extremes',
    difficulty: 'intermediate'
  },
  {
    id: 'spatial-004',
    category: 'Spatial Mapping',
    query: 'Compare salinity patterns between the Bay of Bengal and Arabian Sea',
    expectedIntent: 'spatial_mapping',
    expectedComplexity: 'advanced',
    expectedVisualizations: ['globe', 'heatmap', 'scatter', 'box'],
    description: 'Comparative spatial analysis',
    difficulty: 'advanced'
  },

  // === TEMPORAL TRENDS PROMPTS ===
  {
    id: 'temporal-001',
    category: 'Temporal Trends',
    query: 'Show temperature trends over the last 6 months',
    expectedIntent: 'temporal_trends',
    expectedComplexity: 'intermediate',
    expectedVisualizations: ['timeSeries', 'histogram'],
    description: 'Basic temporal analysis',
    difficulty: 'beginner'
  },
  {
    id: 'temporal-002',
    category: 'Temporal Trends',
    query: 'How has ocean temperature changed from January to August 2025?',
    expectedIntent: 'temporal_trends',
    expectedComplexity: 'intermediate',
    expectedVisualizations: ['timeSeries', 'scatter', 'histogram'],
    description: 'Specific temporal range analysis',
    difficulty: 'intermediate'
  },
  {
    id: 'temporal-003',
    category: 'Temporal Trends',
    query: 'Analyze seasonal variations in salinity across different regions',
    expectedIntent: 'temporal_trends',
    expectedComplexity: 'advanced',
    expectedVisualizations: ['timeSeries', 'heatmap', 'box', 'scatter'],
    description: 'Complex temporal-spatial analysis',
    difficulty: 'advanced'
  },
  {
    id: 'temporal-004',
    category: 'Temporal Trends',
    query: 'What are the long-term trends in ocean temperature and salinity?',
    expectedIntent: 'temporal_trends',
    expectedComplexity: 'advanced',
    expectedVisualizations: ['timeSeries', 'scatter', 'histogram'],
    description: 'Long-term trend analysis',
    difficulty: 'advanced'
  },

  // === PROFILE ANALYSIS PROMPTS ===
  {
    id: 'profile-001',
    category: 'Profile Analysis',
    query: 'Show me temperature profiles for float 1900121',
    expectedIntent: 'profile_analysis',
    expectedComplexity: 'basic',
    expectedVisualizations: ['profile', 'scatter'],
    description: 'Single float profile analysis',
    difficulty: 'beginner'
  },
  {
    id: 'profile-002',
    category: 'Profile Analysis',
    query: 'What is the temperature at 1000 meters depth?',
    expectedIntent: 'profile_analysis',
    expectedComplexity: 'intermediate',
    expectedVisualizations: ['profile', 'histogram', 'scatter'],
    description: 'Depth-specific parameter query',
    difficulty: 'intermediate'
  },
  {
    id: 'profile-003',
    category: 'Profile Analysis',
    query: 'Compare vertical temperature structure between different regions',
    expectedIntent: 'profile_analysis',
    expectedComplexity: 'advanced',
    expectedVisualizations: ['profile', 'scatter', 'box'],
    description: 'Comparative profile analysis',
    difficulty: 'advanced'
  },
  {
    id: 'profile-004',
    category: 'Profile Analysis',
    query: 'Show me mixed layer depth variations across the Indian Ocean',
    expectedIntent: 'profile_analysis',
    expectedComplexity: 'advanced',
    expectedVisualizations: ['profile', 'heatmap', 'scatter'],
    description: 'Advanced oceanographic parameter analysis',
    difficulty: 'advanced'
  },

  // === STATISTICAL SUMMARY PROMPTS ===
  {
    id: 'statistical-001',
    category: 'Statistical Summary',
    query: 'What is the average temperature in the dataset?',
    expectedIntent: 'statistical_summary',
    expectedComplexity: 'basic',
    expectedVisualizations: ['histogram', 'box'],
    description: 'Basic statistical query',
    difficulty: 'beginner'
  },
  {
    id: 'statistical-002',
    category: 'Statistical Summary',
    query: 'Count how many ARGO floats we have data for',
    expectedIntent: 'statistical_summary',
    expectedComplexity: 'basic',
    expectedVisualizations: ['histogram'],
    description: 'Data count query',
    difficulty: 'beginner'
  },
  {
    id: 'statistical-003',
    category: 'Statistical Summary',
    query: 'What are the temperature and salinity statistics for different depth ranges?',
    expectedIntent: 'statistical_summary',
    expectedComplexity: 'intermediate',
    expectedVisualizations: ['histogram', 'box', 'scatter'],
    description: 'Multi-parameter statistical analysis',
    difficulty: 'intermediate'
  },
  {
    id: 'statistical-004',
    category: 'Statistical Summary',
    query: 'Analyze the distribution and variability of ocean parameters',
    expectedIntent: 'statistical_summary',
    expectedComplexity: 'advanced',
    expectedVisualizations: ['histogram', 'box', 'scatter'],
    description: 'Comprehensive statistical analysis',
    difficulty: 'advanced'
  },

  // === COMPARATIVE ANALYSIS PROMPTS ===
  {
    id: 'comparative-001',
    category: 'Comparative Analysis',
    query: 'Compare water temperatures between 2020 and 2023',
    expectedIntent: 'comparative_analysis',
    expectedComplexity: 'intermediate',
    expectedVisualizations: ['scatter', 'box', 'timeSeries'],
    description: 'Temporal comparison',
    difficulty: 'intermediate'
  },
  {
    id: 'comparative-002',
    category: 'Comparative Analysis',
    query: 'How do temperature and salinity correlate?',
    expectedIntent: 'comparative_analysis',
    expectedComplexity: 'intermediate',
    expectedVisualizations: ['scatter', 'heatmap'],
    description: 'Parameter correlation analysis',
    difficulty: 'intermediate'
  },
  {
    id: 'comparative-003',
    category: 'Comparative Analysis',
    query: 'Compare BGC parameters in the Arabian Sea for the last 6 months',
    expectedIntent: 'comparative_analysis',
    expectedComplexity: 'advanced',
    expectedVisualizations: ['scatter', 'box', 'timeSeries', 'heatmap'],
    description: 'Complex multi-parameter comparison',
    difficulty: 'advanced'
  },
  {
    id: 'comparative-004',
    category: 'Comparative Analysis',
    query: 'What are the differences between surface and deep ocean conditions?',
    expectedIntent: 'comparative_analysis',
    expectedComplexity: 'advanced',
    expectedVisualizations: ['profile', 'scatter', 'box'],
    description: 'Depth-based comparison',
    difficulty: 'advanced'
  },

  // === EXPLORATION PROMPTS ===
  {
    id: 'exploration-001',
    category: 'Exploration',
    query: 'Show me salinity profiles near the equator in March 2023',
    expectedIntent: 'exploration',
    expectedComplexity: 'intermediate',
    expectedVisualizations: ['profile', 'map', 'scatter'],
    description: 'Multi-criteria exploration',
    difficulty: 'intermediate'
  },
  {
    id: 'exploration-002',
    category: 'Exploration',
    query: 'What are the nearest ARGO floats to this location?',
    expectedIntent: 'exploration',
    expectedComplexity: 'intermediate',
    expectedVisualizations: ['map', 'globe'],
    description: 'Proximity-based exploration',
    difficulty: 'intermediate'
  },
  {
    id: 'exploration-003',
    category: 'Exploration',
    query: 'Explore ocean data for the Indian Ocean region',
    expectedIntent: 'exploration',
    expectedComplexity: 'basic',
    expectedVisualizations: ['globe', 'map', 'histogram'],
    description: 'General exploration query',
    difficulty: 'beginner'
  },
  {
    id: 'exploration-004',
    category: 'Exploration',
    query: 'What interesting patterns can you find in the ARGO data?',
    expectedIntent: 'exploration',
    expectedComplexity: 'advanced',
    expectedVisualizations: ['globe', 'heatmap', 'scatter', 'timeSeries'],
    description: 'Open-ended exploration',
    difficulty: 'advanced'
  },

  // === ADVANCED VISUALIZATION PROMPTS ===
  {
    id: 'advanced-001',
    category: 'Advanced Visualization',
    query: 'Create a 3D visualization of ARGO float trajectories',
    expectedIntent: 'spatial_mapping',
    expectedComplexity: 'advanced',
    expectedVisualizations: ['globe'],
    description: 'Explicit 3D visualization request',
    difficulty: 'advanced'
  },
  {
    id: 'advanced-002',
    category: 'Advanced Visualization',
    query: 'Show me a heatmap of temperature anomalies',
    expectedIntent: 'spatial_mapping',
    expectedComplexity: 'advanced',
    expectedVisualizations: ['heatmap', 'globe'],
    description: 'Anomaly analysis visualization',
    difficulty: 'advanced'
  },
  {
    id: 'advanced-003',
    category: 'Advanced Visualization',
    query: 'Generate a comprehensive dashboard showing all ocean parameters',
    expectedIntent: 'exploration',
    expectedComplexity: 'advanced',
    expectedVisualizations: ['globe', 'heatmap', 'timeSeries', 'profile', 'scatter'],
    description: 'Dashboard generation request',
    difficulty: 'advanced'
  },
  {
    id: 'advanced-004',
    category: 'Advanced Visualization',
    query: 'Show me interactive charts for temperature, salinity, and pressure',
    expectedIntent: 'exploration',
    expectedComplexity: 'intermediate',
    expectedVisualizations: ['scatter', 'profile', 'histogram'],
    description: 'Multi-chart visualization request',
    difficulty: 'intermediate'
  },

  // === ERROR HANDLING PROMPTS ===
  {
    id: 'error-001',
    category: 'Error Handling',
    query: 'Show me data for platform INVALID123456789',
    expectedIntent: 'statistical_summary',
    expectedComplexity: 'basic',
    expectedVisualizations: [],
    description: 'Invalid platform ID test',
    difficulty: 'beginner'
  },
  {
    id: 'error-002',
    category: 'Error Handling',
    query: '',
    expectedIntent: 'exploration',
    expectedComplexity: 'basic',
    expectedVisualizations: [],
    description: 'Empty query test',
    difficulty: 'beginner'
  },
  {
    id: 'error-003',
    category: 'Error Handling',
    query: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    expectedIntent: 'exploration',
    expectedComplexity: 'basic',
    expectedVisualizations: [],
    description: 'Nonsensical query test',
    difficulty: 'beginner'
  }
];

// Helper functions for test prompt management
export const getPromptsByCategory = (category: string): TestPrompt[] => {
  return TEST_PROMPTS.filter(prompt => prompt.category === category);
};

export const getPromptsByDifficulty = (difficulty: string): TestPrompt[] => {
  return TEST_PROMPTS.filter(prompt => prompt.difficulty === difficulty);
};

export const getPromptsByIntent = (intent: string): TestPrompt[] => {
  return TEST_PROMPTS.filter(prompt => prompt.expectedIntent === intent);
};

export const getRandomPrompt = (): TestPrompt => {
  const randomIndex = Math.floor(Math.random() * TEST_PROMPTS.length);
  return TEST_PROMPTS[randomIndex];
};

export const getPromptById = (id: string): TestPrompt | undefined => {
  return TEST_PROMPTS.find(prompt => prompt.id === id);
};

// Test scenarios for different user types
export const USER_SCENARIOS = {
  researcher: {
    name: 'Ocean Researcher',
    description: 'Experienced oceanographer looking for detailed analysis',
    preferredPrompts: ['profile-003', 'comparative-003', 'advanced-001', 'temporal-003'],
    expectedComplexity: 'advanced'
  },
  student: {
    name: 'Graduate Student',
    description: 'Student learning oceanography concepts',
    preferredPrompts: ['spatial-001', 'profile-001', 'statistical-001', 'exploration-001'],
    expectedComplexity: 'intermediate'
  },
  general_public: {
    name: 'General Public',
    description: 'Curious individual interested in ocean data',
    preferredPrompts: ['exploration-003', 'spatial-001', 'statistical-002'],
    expectedComplexity: 'basic'
  },
  government_official: {
    name: 'Government Official',
    description: 'Policy maker needing ocean data insights',
    preferredPrompts: ['statistical-003', 'comparative-001', 'temporal-002'],
    expectedComplexity: 'intermediate'
  }
};
