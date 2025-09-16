// components/visualizations/PlotlyChart.tsx
'use client';

import dynamic from 'next/dynamic';
import { VisualizationProps } from '@/lib/types';

// Dynamically import the client-side only component with no SSR
const ClientSidePlotly = dynamic(
  () => import('./ClientSidePlotly'),
  { 
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex items-center justify-center bg-gray-100">
        <div className="text-gray-500">Loading chart...</div>
      </div>
    )
  }
);

export default function PlotlyChart(props: VisualizationProps) {
  return <ClientSidePlotly {...props} />;
}