import type { NextConfig } from "next";

/**
 * On Vercel, set API_PROXY_TARGET + NEXT_PUBLIC_API_BASE_URL=/api so the
 * browser talks same-origin (avoids CORS / "Failed to fetch" to Render).
 */
const apiProxyTarget = (process.env.API_PROXY_TARGET || "").replace(/\/$/, "");

const nextConfig: NextConfig = {
  // standalone is for Docker images; Vercel uses its own build output
  ...(process.env.DOCKER_BUILD === "1" ? { output: "standalone" as const } : {}),
  async rewrites() {
    if (!apiProxyTarget) return [];
    return [
      {
        source: "/api/:path*",
        destination: `${apiProxyTarget}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
