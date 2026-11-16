import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run build` writes web/dist, which `python -m sfs.web.server` serves at /.
// `npm run dev` serves the UI on :5173 and forwards the API and media to the running back end (:8000).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000", "/media": "http://127.0.0.1:8000" },
  },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 900 },
});
