import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    NEXT_API_BASE_URL: process.env.NEXT_API_BASE_URL,
  },
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
