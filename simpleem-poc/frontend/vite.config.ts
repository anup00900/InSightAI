import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        // Allow large file uploads (up to 10 minutes for 2GB files)
        timeout: 600000,
        proxyTimeout: 600000,
      },
      '/uploads': 'http://127.0.0.1:8000',
      '/ws': { target: 'http://127.0.0.1:8000', ws: true },
    },
  },
})
