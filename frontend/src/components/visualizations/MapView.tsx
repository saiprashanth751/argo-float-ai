// components/visualizations/MapView.tsx
'use client';

import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default markers in Leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface MapViewProps {
  data: any;
}

interface MapItem {
  latitude?: number;
  longitude?: number;
  lat?: number;
  lon?: number;
  lng?: number;
  [key: string]: any;
}

export default function MapView({ data }: MapViewProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    if (!mapRef.current || !data) return;

    // Initialize map
    const map = L.map(mapRef.current).setView([20, 0], 2);
    mapInstanceRef.current = map;

    // Add base tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Process and add data
    const markers: L.Marker[] = [];
    
    // Handle different data structures
    const mapData: MapItem[] = Array.isArray(data) ? data : data.results || data.data || [];
    
    mapData.forEach((item: MapItem) => {
      const lat = item.latitude || item.lat;
      const lng = item.longitude || item.lon || item.lng;
      
      if (lat && lng) {
        const marker = L.marker([lat, lng]).addTo(map);
        
        // Create popup content
        let popupContent = `<div class="text-sm">`;
        Object.entries(item).forEach(([key, value]) => {
          if (key !== 'latitude' && key !== 'longitude' && key !== 'lat' && key !== 'lon' && key !== 'lng') {
            popupContent += `<div><strong>${key}:</strong> ${value}</div>`;
          }
        });
        popupContent += `</div>`;
        
        marker.bindPopup(popupContent);
        markers.push(marker);
      }
    });

    // Fit map to markers if we have any
    if (markers.length > 0) {
      const group = L.featureGroup(markers);
      map.fitBounds(group.getBounds().pad(0.1));
    }

    setIsLoaded(true);

    // Cleanup
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [data]);

  return (
    <div className="w-full h-full relative">
      <div 
        ref={mapRef} 
        className="w-full h-full"
        style={{ minHeight: '400px' }}
      />
      
      {!isLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-80">
          <div className="text-gray-600">Loading map...</div>
        </div>
      )}
    </div>
  );
}