import type { NextConfig } from "next";

/** Static export for Cloudflare Pages: no server, no image service.
 *  The ONNX WASM binaries are served from public/ort/, copied there by
 *  scripts/gather.mjs so onnxruntime-web never needs a CDN. */
const config: NextConfig = {
  output: "export",
  reactStrictMode: true,
  images: { unoptimized: true },
};

export default config;
