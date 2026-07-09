/** The gathering: copies the atelier's exported crates into public/,
 *  verbatim, never edited by hand. Run before dev, build and test.
 *  Sources: ../export/out (models, data, results incl. parity.json)
 *  and the onnxruntime-web WASM binaries out of node_modules, so the
 *  static site serves its own runtime. */
import { cpSync, mkdirSync, existsSync, readdirSync, copyFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const crates = join(here, "..", "..", "export", "out");
const pub = join(here, "..", "public");

if (!existsSync(crates)) {
  console.error("no export/out found: run `python -m export.for_the_vitrine` in the atelier repo first");
  process.exit(1);
}
for (const leaf of ["models", "data", "results"]) {
  cpSync(join(crates, leaf), join(pub, leaf), { recursive: true });
}
copyFileSync(join(here, "..", "..", "results", "report.md"),
             join(pub, "results", "report.md"));

const wasmHome = join(here, "..", "node_modules", "onnxruntime-web", "dist");
mkdirSync(join(pub, "ort"), { recursive: true });
for (const file of readdirSync(wasmHome)) {
  if (file.endsWith(".wasm") || file.endsWith(".mjs")) {
    copyFileSync(join(wasmHome, file), join(pub, "ort", file));
  }
}
console.log("gathered: models, data, results, ort wasm into public/");
