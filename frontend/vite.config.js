import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
  ],
  server: {
    port: 5180,
    proxy: {
      '/predict': 'http://localhost:8000',
      '/shap':    'http://localhost:8000',
      '/3d':      'http://localhost:8000',
    },
  },
})
