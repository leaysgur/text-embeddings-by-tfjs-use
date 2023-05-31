import { defineConfig } from "vite";
import solidPlugin from "vite-plugin-solid";

export default defineConfig(({ mode }) => ({
  base: mode === "production" ? "/tfjs-embeddings-by-use/" : "/",
  plugins: [solidPlugin()],
  build: {
    outDir: "docs",
  },
}));
