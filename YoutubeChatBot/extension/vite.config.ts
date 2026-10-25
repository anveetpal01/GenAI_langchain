import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import preact from "@preact/preset-vite";
import { defineConfig } from "vite";

const root = fileURLToPath(new URL(".", import.meta.url));
const src = resolve(root, "src");

// One build, four entry points. Output names are fixed (no hashes) because
// manifest.json refers to them. The content script must stay self-contained
// (Chrome loads it as a classic script, so it cannot import shared chunks):
// it may only `import type` from other modules.
export default defineConfig({
  root: src,
  publicDir: resolve(root, "public"),
  plugins: [preact()],
  server: { port: 5173, open: "/sidepanel/index.html" },
  build: {
    outDir: resolve(root, "dist"),
    emptyOutDir: true,
    target: "chrome114",
    rollupOptions: {
      input: {
        sidepanel: resolve(src, "sidepanel/index.html"),
        options: resolve(src, "options/index.html"),
        background: resolve(src, "background/index.ts"),
        content: resolve(src, "content/index.ts"),
      },
      output: {
        entryFileNames: "[name].js",
        chunkFileNames: "chunks/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});
