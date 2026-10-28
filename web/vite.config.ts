import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Runtime verification uses the compiled UI served by proofloop.server.
  // Keep preview development local; the API intentionally enforces one origin.
  server: { host: '127.0.0.1' },
})
