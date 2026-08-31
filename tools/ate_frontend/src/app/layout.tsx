import type { Metadata } from "next";
import { QueryProvider } from "@/components/providers/QueryProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "VERILUMEN — ATE Intelligence",
  description: "VERILUMEN ATE Intelligence — real-time semiconductor test-floor yield and optimization",
};

/**
 * Fonts load via CSS link (runtime) instead of next/font/google so Vercel
 * builds do not fail when fonts.gstatic.com is unreachable at build time.
 */
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  );
}
