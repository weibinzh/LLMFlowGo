import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
 // Configure a proxy to resolve cross-origin issues during front-end development
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
    },
  },
  build: {
// Specify the build output directory as the backend's static folder
    outDir: '../static',
    emptyOutDir: true,
  }
})
