// AetherForge v1.0 — frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  root: path.resolve(__dirname, "frontend"),
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "frontend/src"),
    },
  },
  server: {
    port: 1420,
    strictPort: true,
    // Proxy API calls to FastAPI backend during dev
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://127.0.0.1:8765",
        ws: true,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: path.resolve(__dirname, "frontend/dist"),
    emptyOutDir: true,
    sourcemap: false, // Disable in production for security
    rollupOptions: {
      output: {
        manualChunks: {
          // Split large deps for faster initial load
          vendor: ["react", "react-dom"],
          flow: ["@xyflow/react", "reactflow"],
          monaco: ["@monaco-editor/react"],
        },
      },
    },
  },
});
