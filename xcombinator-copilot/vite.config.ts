import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Production build is served at /copilot/ on the infineon-fab-copilot Worker
// (Cloudflare Static Assets), so the build base must match that path. Dev stays
// at / for convenience.
export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === 'build' ? '/copilot/' : '/',
  server: { port: 5173 },
}))
