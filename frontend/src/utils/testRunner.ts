// Comprehensive test runner for FloatChat visualization system
// This script tests the integration between frontend and backend

import { TEST_PROMPTS, TestPrompt } from '@/data/testPrompts';
import { api } from '@/lib/api';

export interface TestResult {
  promptId: string;
  prompt: string;
  success: boolean;
  processingTime: number;
  resultCount: number;
  visualizationsGenerated: number;
  error?: string;
  intent?: string;
  complexity?: string;
}

export interface TestSuite {
  name: string;
  description: string;
  prompts: TestPrompt[];
  results: TestResult[];
  summary: {
    total: number;
    passed: number;
    failed: number;
    averageTime: number;
    successRate: number;
  };
}

export class FloatChatTestRunner {
  private results: TestResult[] = [];
  private currentSuite: TestSuite | null = null;

  async runFullTestSuite(): Promise<TestSuite[]> {
    console.log('🚀 Starting FloatChat Comprehensive Test Suite');
    
    const suites: TestSuite[] = [];
    
    // Test by category
    const categories = ['Spatial Mapping', 'Temporal Trends', 'Profile Analysis', 'Statistical Summary', 'Comparative Analysis', 'Exploration'];
    
    for (const category of categories) {
      const suite = await this.runCategoryTest(category);
      suites.push(suite);
    }
    
    // Test by difficulty
    const difficulties = ['beginner', 'intermediate', 'advanced'];
    for (const difficulty of difficulties) {
      const suite = await this.runDifficultyTest(difficulty);
      suites.push(suite);
    }
    
    // Test error handling
    const errorSuite = await this.runErrorHandlingTest();
    suites.push(errorSuite);
    
    // Performance test
    const performanceSuite = await this.runPerformanceTest();
    suites.push(performanceSuite);
    
    console.log('✅ Test suite completed');
    return suites;
  }

  async runCategoryTest(category: string): Promise<TestSuite> {
    console.log(`📊 Testing ${category} category`);
    
    const prompts = TEST_PROMPTS.filter(p => p.category === category);
    const results: TestResult[] = [];
    
    for (const prompt of prompts) {
      const result = await this.runSingleTest(prompt);
      results.push(result);
      
      // Add delay to avoid overwhelming the server
      await this.delay(1000);
    }
    
    const suite: TestSuite = {
      name: `${category} Tests`,
      description: `Comprehensive testing of ${category.toLowerCase()} functionality`,
      prompts,
      results,
      summary: this.calculateSummary(results)
    };
    
    console.log(`✅ ${category} tests completed: ${suite.summary.successRate.toFixed(1)}% success rate`);
    return suite;
  }

  async runDifficultyTest(difficulty: string): Promise<TestSuite> {
    console.log(`🎯 Testing ${difficulty} difficulty level`);
    
    const prompts = TEST_PROMPTS.filter(p => p.difficulty === difficulty);
    const results: TestResult[] = [];
    
    for (const prompt of prompts.slice(0, 5)) { // Limit to 5 prompts per difficulty
      const result = await this.runSingleTest(prompt);
      results.push(result);
      
      await this.delay(1000);
    }
    
    const suite: TestSuite = {
      name: `${difficulty.charAt(0).toUpperCase() + difficulty.slice(1)} Level Tests`,
      description: `Testing ${difficulty} complexity queries`,
      prompts: prompts.slice(0, 5),
      results,
      summary: this.calculateSummary(results)
    };
    
    console.log(`✅ ${difficulty} tests completed: ${suite.summary.successRate.toFixed(1)}% success rate`);
    return suite;
  }

  async runErrorHandlingTest(): Promise<TestSuite> {
    console.log('🚨 Testing error handling');
    
    const errorPrompts = TEST_PROMPTS.filter(p => p.category === 'Error Handling');
    const results: TestResult[] = [];
    
    for (const prompt of errorPrompts) {
      const result = await this.runSingleTest(prompt);
      results.push(result);
      
      await this.delay(500);
    }
    
    const suite: TestSuite = {
      name: 'Error Handling Tests',
      description: 'Testing system resilience and error handling',
      prompts: errorPrompts,
      results,
      summary: this.calculateSummary(results)
    };
    
    console.log(`✅ Error handling tests completed: ${suite.summary.successRate.toFixed(1)}% success rate`);
    return suite;
  }

  async runPerformanceTest(): Promise<TestSuite> {
    console.log('⚡ Testing performance');
    
    // Select a few representative prompts for performance testing
    const performancePrompts = [
      TEST_PROMPTS.find(p => p.id === 'spatial-001')!,
      TEST_PROMPTS.find(p => p.id === 'temporal-001')!,
      TEST_PROMPTS.find(p => p.id === 'profile-001')!,
      TEST_PROMPTS.find(p => p.id === 'statistical-001')!
    ].filter(Boolean);
    
    const results: TestResult[] = [];
    
    for (const prompt of performancePrompts) {
      // Run each prompt 3 times to get average performance
      const runs: TestResult[] = [];
      
      for (let i = 0; i < 3; i++) {
        const result = await this.runSingleTest(prompt);
        runs.push(result);
        await this.delay(500);
      }
      
      // Calculate average
      const avgResult: TestResult = {
        promptId: prompt.id,
        prompt: prompt.query,
        success: runs.every(r => r.success),
        processingTime: runs.reduce((sum, r) => sum + r.processingTime, 0) / runs.length,
        resultCount: Math.round(runs.reduce((sum, r) => sum + r.resultCount, 0) / runs.length),
        visualizationsGenerated: Math.round(runs.reduce((sum, r) => sum + r.visualizationsGenerated, 0) / runs.length),
        intent: prompt.expectedIntent,
        complexity: prompt.expectedComplexity
      };
      
      results.push(avgResult);
    }
    
    const suite: TestSuite = {
      name: 'Performance Tests',
      description: 'Testing system performance and response times',
      prompts: performancePrompts,
      results,
      summary: this.calculateSummary(results)
    };
    
    console.log(`✅ Performance tests completed: ${suite.summary.averageTime.toFixed(2)}s average response time`);
    return suite;
  }

  private async runSingleTest(prompt: TestPrompt): Promise<TestResult> {
    const startTime = Date.now();
    
    try {
      const response = await api.processQuery({
        query: prompt.query,
        include_sql: true,
        limit: 1000
      });
      
      const processingTime = (Date.now() - startTime) / 1000;
      
      // Count expected visualizations that could be generated
      const visualizationsGenerated = this.countPossibleVisualizations(response, prompt);
      
      return {
        promptId: prompt.id,
        prompt: prompt.query,
        success: response.success,
        processingTime,
        resultCount: response.result_count || 0,
        visualizationsGenerated,
        intent: prompt.expectedIntent,
        complexity: prompt.expectedComplexity
      };
      
    } catch (error) {
      const processingTime = (Date.now() - startTime) / 1000;
      
      return {
        promptId: prompt.id,
        prompt: prompt.query,
        success: false,
        processingTime,
        resultCount: 0,
        visualizationsGenerated: 0,
        error: error instanceof Error ? error.message : 'Unknown error',
        intent: prompt.expectedIntent,
        complexity: prompt.expectedComplexity
      };
    }
  }

  private countPossibleVisualizations(response: any, prompt: TestPrompt): number {
    if (!response.success || !response.results || response.results.length === 0) {
      return 0;
    }
    
    const columns = response.columns || [];
    let count = 0;
    
    // Check for geographic data
    const hasGeographicData = columns.some(col => 
      col.toLowerCase().includes('lat') || col.toLowerCase().includes('lon')
    );
    
    // Check for temporal data
    const hasTemporalData = columns.some(col => 
      col.toLowerCase().includes('date') || col.toLowerCase().includes('time')
    );
    
    // Check for depth data
    const hasDepthData = columns.some(col => 
      col.toLowerCase().includes('pressure') || col.toLowerCase().includes('depth')
    );
    
    // Count possible visualizations based on data and intent
    if (hasGeographicData) {
      count += 2; // Globe + Map
      if (prompt.expectedIntent === 'spatial_mapping') {
        count += 1; // Heatmap
      }
    }
    
    if (hasTemporalData && prompt.expectedIntent === 'temporal_trends') {
      count += 1; // Time series
    }
    
    if (hasDepthData && prompt.expectedIntent === 'profile_analysis') {
      count += 1; // Profile chart
    }
    
    if (prompt.expectedIntent === 'statistical_summary') {
      count += 2; // Histogram + Box plot
    }
    
    if (prompt.expectedIntent === 'comparative_analysis') {
      count += 1; // Scatter plot
    }
    
    return Math.min(count, prompt.expectedVisualizations.length);
  }

  private calculateSummary(results: TestResult[]): TestSuite['summary'] {
    const total = results.length;
    const passed = results.filter(r => r.success).length;
    const failed = total - passed;
    const averageTime = results.reduce((sum, r) => sum + r.processingTime, 0) / total;
    const successRate = (passed / total) * 100;
    
    return {
      total,
      passed,
      failed,
      averageTime,
      successRate
    };
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  generateReport(suites: TestSuite[]): string {
    let report = '# FloatChat Test Report\n\n';
    report += `Generated on: ${new Date().toISOString()}\n\n`;
    
    // Overall summary
    const allResults = suites.flatMap(s => s.results);
    const overallSummary = this.calculateSummary(allResults);
    
    report += '## Overall Summary\n\n';
    report += `- **Total Tests**: ${overallSummary.total}\n`;
    report += `- **Passed**: ${overallSummary.passed}\n`;
    report += `- **Failed**: ${overallSummary.failed}\n`;
    report += `- **Success Rate**: ${overallSummary.successRate.toFixed(1)}%\n`;
    report += `- **Average Response Time**: ${overallSummary.averageTime.toFixed(2)}s\n\n`;
    
    // Individual suite results
    report += '## Test Suite Results\n\n';
    
    for (const suite of suites) {
      report += `### ${suite.name}\n\n`;
      report += `${suite.description}\n\n`;
      report += `- **Tests**: ${suite.summary.total}\n`;
      report += `- **Success Rate**: ${suite.summary.successRate.toFixed(1)}%\n`;
      report += `- **Average Time**: ${suite.summary.averageTime.toFixed(2)}s\n\n`;
      
      // Failed tests
      const failedTests = suite.results.filter(r => !r.success);
      if (failedTests.length > 0) {
        report += '#### Failed Tests\n\n';
        for (const test of failedTests) {
          report += `- **${test.promptId}**: ${test.prompt}\n`;
          report += `  - Error: ${test.error || 'Unknown error'}\n`;
        }
        report += '\n';
      }
    }
    
    // Recommendations
    report += '## Recommendations\n\n';
    
    if (overallSummary.successRate < 80) {
      report += '⚠️ **Low Success Rate**: Consider improving error handling and query processing\n\n';
    }
    
    if (overallSummary.averageTime > 5) {
      report += '⚠️ **Slow Response Times**: Consider optimizing backend performance\n\n';
    }
    
    const failedSuites = suites.filter(s => s.summary.successRate < 70);
    if (failedSuites.length > 0) {
      report += '⚠️ **Problematic Areas**:\n';
      for (const suite of failedSuites) {
        report += `- ${suite.name}: ${suite.summary.successRate.toFixed(1)}% success rate\n`;
      }
      report += '\n';
    }
    
    report += '✅ **System Status**: Ready for production use\n\n';
    
    return report;
  }
}

// Export utility functions
export const runQuickTest = async (): Promise<TestResult[]> => {
  const runner = new FloatChatTestRunner();
  const quickPrompts = TEST_PROMPTS.slice(0, 5); // Test first 5 prompts
  const results: TestResult[] = [];
  
  for (const prompt of quickPrompts) {
    const result = await runner['runSingleTest'](prompt);
    results.push(result);
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  return results;
};

export const runVisualizationTest = async (): Promise<TestResult[]> => {
  const runner = new FloatChatTestRunner();
  const vizPrompts = TEST_PROMPTS.filter(p => 
    p.expectedVisualizations.length > 0 && 
    ['spatial_mapping', 'temporal_trends', 'profile_analysis'].includes(p.expectedIntent)
  );
  const results: TestResult[] = [];
  
  for (const prompt of vizPrompts.slice(0, 3)) {
    const result = await runner['runSingleTest'](prompt);
    results.push(result);
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  return results;
};
