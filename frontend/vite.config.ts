import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The dev proxy forwards /api/* to the FastAPI backend and injects the API key
// server-side, so the key never reaches the browser. In production, use a
// backend-for-frontend (BFF) or serverless proxy that does the same.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const API_TARGET = env.API_TARGET || 'http://localhost:8077'
  const API_KEY = env.API_KEY || ''

  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      // the workspace path contains a ':' which breaks Vite's fs allow-list matching;
      // relax it for local dev (dev server only).
      fs: { strict: false },
      proxy: {
        '/api': {
          target: API_TARGET,
          changeOrigin: true,
          rewrite: (p) => p.replace(/^\/api/, ''),
          configure: (proxy) => {
            proxy.on('proxyReq', (proxyReq) => {
              if (API_KEY) proxyReq.setHeader('X-API-Key', API_KEY)
            })
          },
        },
      },
    },
  }
})
