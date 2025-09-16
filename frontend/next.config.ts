import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  // Turbopack-compatible configuration
  transpilePackages: ['react-plotly.js', 'plotly.js-dist-min'],
};

export default nextConfig;
