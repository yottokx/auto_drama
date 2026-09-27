import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const coordinator = process.env.AUTO_DRAMA_COORDINATOR_URL || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: { '/api': coordinator, '/player': coordinator },
  },
})
