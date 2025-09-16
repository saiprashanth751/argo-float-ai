// app/page.tsx
'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { api } from '@/lib/api';
import { DatabaseStats } from '@/lib/types';

// Dynamically import ChatLayout to avoid SSR issues with visualizations
const ChatLayout = dynamic(() => import('@/components/layout/ChatLayout'), {
  ssr: false,
  loading: () => (
    <div className="h-[calc(100vh-80px)] flex items-center justify-center">
      <div className="text-gray-500">Loading chat interface...</div>
    </div>
  )
});

export default function FloatChatPage() {
  const [stats, setStats] = useState<DatabaseStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadStats = async () => {
      try {
        setLoading(true);
        setError(null);
        const dbStats = await api.getStats();
        setStats(dbStats);
      } catch (error) {
        console.error('Failed to load stats:', error);
        setError('Failed to load database statistics');
      } finally {
        setLoading(false);
      }
    };
    
    loadStats();
  }, []);

  return (
    <main className="h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">FloatChat</h1>
              <p className="text-sm text-gray-600">Advanced oceanographic data analysis</p>
            </div>
            
            {/* Stats Display */}
            {loading && (
              <div className="flex space-x-6 text-sm">
                <div className="text-center">
                  <div className="h-6 bg-gray-200 rounded w-12 animate-pulse"></div>
                  <div className="text-gray-600">Loading...</div>
                </div>
              </div>
            )}
            
            {error && (
              <div className="text-sm text-red-600">{error}</div>
            )}
            
            {stats && !loading && (
              <div className="flex space-x-6 text-sm">
                <div className="text-center">
                  <div className="font-semibold text-gray-900">
                    {stats.total_profiles.toLocaleString()}
                  </div>
                  <div className="text-gray-600">Profiles</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-gray-900">
                    {stats.total_measurements.toLocaleString()}
                  </div>
                  <div className="text-gray-600">Measurements</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-gray-900">
                    {stats.unique_platforms.toLocaleString()}
                  </div>
                  <div className="text-gray-600">Platforms</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Chat Layout */}
      <div className="h-[calc(100vh-80px)]">
        <ChatLayout />
      </div>
    </main>
  );
}