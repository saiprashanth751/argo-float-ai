// components/visualizations/ClientSidePlotly.tsx
'use client';

import { useEffect, useRef } from 'react';

interface ClientSidePlotlyProps {
  data: any;
  layout?: any;
  config?: any;
}

export default function ClientSidePlotly({ data, layout, config }: ClientSidePlotlyProps) {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Dynamically import Plotly only on client side
    import('plotly.js-dist-min').then((Plotly) => {
      if (!chartRef.current || !data) return;

      const plotData = data.plotly_json ? JSON.parse(data.plotly_json) : data;

      Plotly.react(chartRef.current, plotData.data || plotData, {
        ...plotData.layout,
        ...layout,
        autosize: true,
        responsive: true,
      }, {
        ...config,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        modeBarButtonsToAdd: ['toggleHover', 'resetViews'],
      });
    });

    // Cleanup
    return () => {
      if (chartRef.current && typeof window !== 'undefined' && (window as any).Plotly) {
        (window as any).Plotly.purge(chartRef.current);
      }
    };
  }, [data, layout, config]);

  return <div ref={chartRef} className="w-full h-full" />;
}