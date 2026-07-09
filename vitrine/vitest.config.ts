import { defineConfig } from "vitest/config";

/** The gates run in Node against the crates in ../export, before any
 *  gathering into public/. Long timeouts: Gate W renders 256 scenes. */
export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
    testTimeout: 300_000,
    hookTimeout: 120_000,
  },
});
