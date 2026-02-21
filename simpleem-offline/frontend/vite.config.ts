import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 4000,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:9000',
        timeout: 600000,
        proxyTimeout: 600000,
      },
      '/uploads': 'http://127.0.0.1:9000',
    },
  },
})
