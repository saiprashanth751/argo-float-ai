'use client';

import { useState } from 'react';
import { TEST_PROMPTS, TestPrompt, getPromptsByCategory, getPromptsByDifficulty, USER_SCENARIOS } from '@/data/testPrompts';

interface TestInterfaceProps {
  onPromptSelect: (prompt: string) => void;
  isVisible: boolean;
  onClose: () => void;
}

export default function TestInterface({ onPromptSelect, isVisible, onClose }: TestInterfaceProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('all');
  const [selectedUserType, setSelectedUserType] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');

  if (!isVisible) return null;

  // Filter prompts based on selections
  const filteredPrompts = TEST_PROMPTS.filter(prompt => {
    const matchesCategory = selectedCategory === 'all' || prompt.category === selectedCategory;
    const matchesDifficulty = selectedDifficulty === 'all' || prompt.difficulty === selectedDifficulty;
    const matchesSearch = searchTerm === '' || 
      prompt.query.toLowerCase().includes(searchTerm.toLowerCase()) ||
      prompt.description.toLowerCase().includes(searchTerm.toLowerCase());
    
    // Filter by user type preferences
    let matchesUserType = true;
    if (selectedUserType !== 'all') {
      const userScenario = USER_SCENARIOS[selectedUserType as keyof typeof USER_SCENARIOS];
      matchesUserType = userScenario.preferredPrompts.includes(prompt.id);
    }
    
    return matchesCategory && matchesDifficulty && matchesSearch && matchesUserType;
  });

  const categories = ['all', ...Array.from(new Set(TEST_PROMPTS.map(p => p.category)))];
  const difficulties = ['all', 'beginner', 'intermediate', 'advanced'];
  const userTypes = ['all', ...Object.keys(USER_SCENARIOS)];

  const handlePromptClick = (prompt: TestPrompt) => {
    onPromptSelect(prompt.query);
    onClose();
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getIntentColor = (intent: string) => {
    switch (intent) {
      case 'spatial_mapping': return 'bg-blue-100 text-blue-800';
      case 'temporal_trends': return 'bg-purple-100 text-purple-800';
      case 'profile_analysis': return 'bg-indigo-100 text-indigo-800';
      case 'statistical_summary': return 'bg-green-100 text-green-800';
      case 'comparative_analysis': return 'bg-orange-100 text-orange-800';
      case 'exploration': return 'bg-pink-100 text-pink-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-6xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="p-6 border-b">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Test Prompts</h2>
              <p className="text-sm text-gray-600 mt-1">
                Comprehensive test scenarios for FloatChat visualization system
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="p-6 border-b bg-gray-50">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Search */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Search</label>
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search prompts..."
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Category Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {categories.map(category => (
                  <option key={category} value={category}>
                    {category === 'all' ? 'All Categories' : category}
                  </option>
                ))}
              </select>
            </div>

            {/* Difficulty Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Difficulty</label>
              <select
                value={selectedDifficulty}
                onChange={(e) => setSelectedDifficulty(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {difficulties.map(difficulty => (
                  <option key={difficulty} value={difficulty}>
                    {difficulty === 'all' ? 'All Levels' : difficulty.charAt(0).toUpperCase() + difficulty.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            {/* User Type Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">User Type</label>
              <select
                value={selectedUserType}
                onChange={(e) => setSelectedUserType(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {userTypes.map(userType => (
                  <option key={userType} value={userType}>
                    {userType === 'all' ? 'All Users' : USER_SCENARIOS[userType as keyof typeof USER_SCENARIOS]?.name || userType}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Results count */}
          <div className="mt-4 text-sm text-gray-600">
            Showing {filteredPrompts.length} of {TEST_PROMPTS.length} prompts
          </div>
        </div>

        {/* Prompts List */}
        <div className="overflow-y-auto max-h-96">
          <div className="p-6 space-y-4">
            {filteredPrompts.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <svg className="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.29-1.009-5.824-2.709M15 6.291A7.962 7.962 0 0012 4c-2.34 0-4.29 1.009-5.824 2.709" />
                </svg>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Prompts Found</h3>
                <p className="text-gray-600">Try adjusting your filters to see more prompts</p>
              </div>
            ) : (
              filteredPrompts.map((prompt) => (
                <div
                  key={prompt.id}
                  onClick={() => handlePromptClick(prompt)}
                  className="p-4 border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 cursor-pointer transition-all group"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <h3 className="font-medium text-gray-900 group-hover:text-blue-900">
                        {prompt.query}
                      </h3>
                      <p className="text-sm text-gray-600 mt-1">{prompt.description}</p>
                    </div>
                    <div className="flex space-x-2 ml-4">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(prompt.difficulty)}`}>
                        {prompt.difficulty}
                      </span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getIntentColor(prompt.expectedIntent)}`}>
                        {prompt.expectedIntent.replace('_', ' ')}
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <div className="flex space-x-4">
                      <span>Category: {prompt.category}</span>
                      <span>Complexity: {prompt.expectedComplexity}</span>
                    </div>
                    <div className="flex space-x-1">
                      {prompt.expectedVisualizations.slice(0, 3).map(viz => (
                        <span key={viz} className="px-2 py-1 bg-gray-100 rounded text-xs">
                          {viz}
                        </span>
                      ))}
                      {prompt.expectedVisualizations.length > 3 && (
                        <span className="px-2 py-1 bg-gray-100 rounded text-xs">
                          +{prompt.expectedVisualizations.length - 3}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t bg-gray-50">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              Click any prompt to test it in the chat interface
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => {
                  const randomPrompt = filteredPrompts[Math.floor(Math.random() * filteredPrompts.length)];
                  if (randomPrompt) {
                    handlePromptClick(randomPrompt);
                  }
                }}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm"
                disabled={filteredPrompts.length === 0}
              >
                Random Prompt
              </button>
              <button
                onClick={onClose}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400 transition-colors text-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
