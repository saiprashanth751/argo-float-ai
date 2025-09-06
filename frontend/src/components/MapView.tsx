'use client';

import { useEffect, useRef } from 'react';
import L from 'leaflet';

// Fix for default markers in Leaflet with Next.js
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface MapViewProps {
  data: Array<Record<string, any>>;
  columns: string[];
}

export default function MapView({ data, columns }: MapViewProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);

  // Check if data has geographic coordinates
  const hasLatitude = columns.some(col => col.toLowerCase().includes('lat'));
  const hasLongitude = columns.some(col => col.toLowerCase().includes('lon'));

  useEffect(() => {
    if (!mapRef.current || !hasLatitude || !hasLongitude || data.length === 0) {
      return;
    }

    // Find coordinate columns
    const latColumn = columns.find(col => col.toLowerCase().includes('lat'));
    const lonColumn = columns.find(col => col.toLowerCase().includes('lon'));

    if (!latColumn || !lonColumn) return;

    // Cleanup existing map
    if (mapInstanceRef.current) {
      mapInstanceRef.current.remove();
    }

    // Create new map
    const map = L.map(mapRef.current).setView([15, 75], 5); // Center on Indian Ocean

    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Add markers for each data point
    const validPoints = data.filter(row => 
      row[latColumn] != null && 
      row[lonColumn] != null &&
      !isNaN(parseFloat(row[latColumn])) &&
      !isNaN(parseFloat(row[lonColumn]))
    );

    if (validPoints.length === 0) {
      return;
    }

    const bounds = L.latLngBounds([]);

    validPoints.forEach((point, index) => {
      const lat = parseFloat(point[latColumn]);
      const lon = parseFloat(point[lonColumn]);

      if (lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {
        // Create popup content
        let popupContent = '<div class="text-sm">';
        
        // Add key information to popup
        Object.entries(point).forEach(([key, value]) => {
          if (value != null && value !== '') {
            if (key.toLowerCase().includes('platform')) {
              popupContent += `<div><strong>Float:</strong> ${value}</div>`;
            } else if (key.toLowerCase().includes('temp')) {
              popupContent += `<div><strong>Temperature:</strong> ${parseFloat(value).toFixed(2)}°C</div>`;
            } else if (key.toLowerCase().includes('sal')) {
              popupContent += `<div><strong>Salinity:</strong> ${parseFloat(value).toFixed(2)} PSU</div>`;
            } else if (key.toLowerCase().includes('pressure') || key.toLowerCase().includes('depth')) {
              popupContent += `<div><strong>Depth:</strong> ${parseFloat(value).toFixed(1)} m</div>`;
            } else if (key.toLowerCase().includes('date')) {
              popupContent += `<div><strong>Date:</strong> ${value}</div>`;
            }
          }
        });
        
        popupContent += '</div>';

        // Choose marker color based on data
        let markerColor = '#3388ff'; // Default blue
        
        if (point.temperature) {
          const temp = parseFloat(point.temperature);
          if (temp > 28) markerColor = '#ff4444'; // Hot - red
          else if (temp > 20) markerColor = '#ffaa00'; // Warm - orange
          else if (temp > 10) markerColor = '#00aa00'; // Cool - green
          else markerColor = '#0066cc'; // Cold - blue
        }

        // Create custom icon
        const customIcon = L.divIcon({
          className: 'custom-marker',
          html: `<div style="background-color: ${markerColor}; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.3);"></div>`,
          iconSize: [12, 12],
          iconAnchor: [6, 6]
        });

        const marker = L.marker([lat, lon], { icon: customIcon })
          .bindPopup(popupContent)
          .addTo(map);

        bounds.extend([lat, lon]);
      }
    });

    // Fit map to show all markers
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [20, 20] });
    }

    mapInstanceRef.current = map;

    // Cleanup function
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [data, columns, hasLatitude, hasLongitude]);

  if (!hasLatitude || !hasLongitude) {
    return (
      <div className="bg-gray-50 rounded-lg p-8 text-center">
        <div className="text-gray-500">
          Map visualization requires latitude and longitude coordinates
        </div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-8 text-center">
        <div className="text-gray-500">
          No geographic data available to display
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="p-6 border-b">
        <h3 className="text-lg font-semibold text-gray-900">Geographic Distribution</h3>
        <p className="text-sm text-gray-600">
          {data.length} data points on map
        </p>
      </div>
      <div className="relative">
        <div 
          ref={mapRef} 
          style={{ height: '400px', width: '100%' }}
          className="rounded-b-lg"
        />
      </div>
    </div>
  );
}