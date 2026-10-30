import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Development only: proxy /api (with WebSocket upgrade for /api/session)
// and /stream to a locally running backend (`uvicorn egocap_web.app:app
// --port 8000`) so `npm run dev` can hit the real FastAPI endpoints
// without a separate CORS setup. Production serves both from the same
// FastAPI origin -- see backend/egocap_web/app.py.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", ws: true },
      "/stream": "http://127.0.0.1:8000",
    },
  },
});
