'use client';

export default function StyleTestPage() {
  return (
    <main className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Test different text colors */}
        <div className="space-y-6">
          <h1 className="text-4xl font-bold text-gray-900">Style Test Page</h1>
          <p className="text-lg text-gray-700">This page tests if the styling is working correctly.</p>
          
          {/* Color test grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white p-6 rounded-lg shadow-sm border">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">White Background</h2>
              <p className="text-gray-700 mb-4">This should have dark text on white background.</p>
              <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                Blue Button
              </button>
            </div>
            
            <div className="bg-gray-100 p-6 rounded-lg shadow-sm border">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Gray Background</h2>
              <p className="text-gray-700 mb-4">This should have dark text on gray background.</p>
              <button className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">
                Green Button
              </button>
            </div>
          </div>
          
          {/* Text color tests */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Text Color Tests</h2>
            <div className="space-y-2">
              <p className="text-gray-900">Gray 900 - Darkest text</p>
              <p className="text-gray-800">Gray 800 - Very dark text</p>
              <p className="text-gray-700">Gray 700 - Dark text</p>
              <p className="text-gray-600">Gray 600 - Medium text</p>
              <p className="text-gray-500">Gray 500 - Light text</p>
              <p className="text-blue-600">Blue 600 - Blue text</p>
              <p className="text-green-600">Green 600 - Green text</p>
              <p className="text-red-600">Red 600 - Red text</p>
            </div>
          </div>
          
          {/* Form elements test */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Form Elements Test</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Input Field</label>
                <input 
                  type="text" 
                  placeholder="Type something here..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Textarea</label>
                <textarea 
                  placeholder="Enter your message..."
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Select</label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                  <option>Option 1</option>
                  <option>Option 2</option>
                  <option>Option 3</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
