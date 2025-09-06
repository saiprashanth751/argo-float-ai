'use client';

import { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ScatterController,
} from 'chart.js';
import { Chart } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ScatterController
);

interface ProfileChartProps {
  data: Array<Record<string, any>>;
  columns: string[];
}

export default function ProfileChart({ data, columns }: ProfileChartProps) {
  const chartRef = useRef<ChartJS>(null);

  // Check if this data is suitable for profile visualization
  const hasDepthData = columns.some(col => 
    col.toLowerCase().includes('pressure') || col.toLowerCase().includes('depth')
  );
  
  const hasParameterData = columns.some(col => 
    col.toLowerCase().includes('temperature') || 
    col.toLowerCase().includes('salinity') ||
    col.toLowerCase().includes('oxygen')
  );

  if (!hasDepthData || !hasParameterData || data.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-8 text-center">
        <div className="text-gray-500">
          Profile visualization requires pressure/depth and parameter data (temperature, salinity, etc.)
        </div>
      </div>
    );
  }

  // Find the depth/pressure column
  const depthColumn = columns.find(col => 
    col.toLowerCase().includes('pressure') || col.toLowerCase().includes('depth')
  );

  // Find parameter columns
  const parameterColumns = columns.filter(col => 
    col.toLowerCase().includes('temperature') || 
    col.toLowerCase().includes('salinity') ||
    col.toLowerCase().includes('oxygen') ||
    col.toLowerCase().includes('chlorophyll')
  );

  if (!depthColumn || parameterColumns.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-8 text-center">
        <div className="text-gray-500">Cannot identify depth and parameter columns for profile</div>
      </div>
    );
  }

  // Prepare datasets for each parameter
  const datasets = parameterColumns.map((param, index) => {
    const color = [
      'rgb(255, 99, 132)',    // Red for temperature
      'rgb(54, 162, 235)',    // Blue for salinity  
      'rgb(255, 205, 86)',    // Yellow for oxygen
      'rgb(75, 192, 192)',    // Green for chlorophyll
    ][index % 4];

    const profileData = data
      .filter(row => row[depthColumn] != null && row[param] != null)
      .map(row => ({
        x: parseFloat(row[param]),
        y: -Math.abs(parseFloat(row[depthColumn])) // Negative for depth (oceanographic convention)
      }))
      .sort((a, b) => b.y - a.y); // Sort by depth (surface to bottom)

    return {
      label: param.charAt(0).toUpperCase() + param.slice(1),
      data: profileData,
      borderColor: color,
      backgroundColor: color + '20',
      showLine: true,
      tension: 0.1,
      pointRadius: 2,
    };
  });

  const chartData = {
    datasets: datasets
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Oceanographic Profile',
        font: {
          size: 16,
          weight: 'bold' as const
        }
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            return `${context.dataset.label}: ${context.parsed.x.toFixed(3)} at ${Math.abs(context.parsed.y).toFixed(1)}m depth`;
          }
        }
      }
    },
    scales: {
      x: {
        type: 'linear' as const,
        position: 'top' as const,
        title: {
          display: true,
          text: parameterColumns.length === 1 ? parameterColumns[0] : 'Parameter Value'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        }
      },
      y: {
        type: 'linear' as const,
        title: {
          display: true,
          text: 'Depth (m)'
        },
        reverse: false, // Don't reverse since we're using negative values
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
        ticks: {
          callback: function(value: any) {
            return Math.abs(value) + 'm';
          }
        }
      },
    },
    interaction: {
      intersect: false,
      mode: 'nearest' as const,
    },
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="p-6 border-b">
        <h3 className="text-lg font-semibold text-gray-900">Profile Visualization</h3>
        <p className="text-sm text-gray-600">
          {data.length} measurements across {parameterColumns.length} parameters
        </p>
      </div>
      <div className="p-6">
        <div style={{ height: '400px' }}>
          <Chart 
            ref={chartRef}
            type='scatter' 
            data={chartData} 
            options={options} 
          />
        </div>
      </div>
    </div>
  );
}