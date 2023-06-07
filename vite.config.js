import { defineConfig } from "vite";
import solidPlugin from "vite-plugin-solid";

export default defineConfig(({ mode }) => ({
  base: mode === "production" ? "/text-embeddings-by-tfjs-use/" : "/",
  plugins: [solidPlugin()],
  build: {
    outDir: "docs",
  },
}));
