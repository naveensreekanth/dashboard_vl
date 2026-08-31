# ATE Analytics Dashboard - Standalone Next.js Frontend

A modern Next.js 15 frontend application built with React 19, TypeScript, Tailwind CSS, and Zustand.

## Features

- Real-time Wafer & Die test metrics dashboard
- Shmoo plot viewer & interactive test limits management
- Event filtering, alerts, and KPI tracking
- Configurable external backend API integration

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Backend API Connection

Copy `.env.example` to `.env.local` and set your backend API URL:

```bash
cp .env.example .env.local
```

Set the backend API endpoint in `.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api
```

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Available Scripts

- `npm run dev` - Start Next.js development server
- `npm run build` - Build production bundle
- `npm run start` - Run production build server
- `npm run lint` - Run ESLint checks

## Connecting with an External Backend

The API client (`src/services/api.ts`) resolves backend requests using `NEXT_PUBLIC_API_BASE_URL`. Ensure your backend handles standard CORS requests from `http://localhost:3000` or configure `API_PROXY_TARGET` in Next.js rewrites (`next.config.ts`).
