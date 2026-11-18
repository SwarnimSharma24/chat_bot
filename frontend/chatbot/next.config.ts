import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    NEXT_API_BASE_URL: process.env.NEXT_API_BASE_URL,
  },
  allowedDevOrigins: ["192.168.1.6"],
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
