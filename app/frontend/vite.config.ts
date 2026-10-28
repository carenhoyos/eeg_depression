import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// El prefijo de la API y el backend de desarrollo se leen de variables de entorno (ver .env.example).
export default defineConfig(({ mode }) => {
  const apiPrefix = process.env.VITE_API_PREFIX ?? '/api/v1'
  const backend = process.env.VITE_DEV_BACKEND ?? 'http://localhost:8000'
  return {
    plugins: [react()],
    server: { proxy: { [apiPrefix]: { target: backend, changeOrigin: true } } },
    build: { outDir: 'dist', sourcemap: mode !== 'production' },
  }
})
