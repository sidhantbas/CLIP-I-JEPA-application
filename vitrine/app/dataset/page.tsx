import type { Metadata } from "next";
import DatasetExplorer from "@/components/dataset/explorer";
import DatasetMapping from "@/components/dataset/mapping";

export const metadata: Metadata = {
  title: "The dataset: a controlled synthetic environment",
  description:
    "Procedurally generated scenes with exact ground truth: four factors " +
    "per shape, templated captions, and one deliberate supervision gap.",
};

export default function DatasetPage() {
  return (
    <main className="mx-auto max-w-5xl px-6 pb-20 pt-16">
      <p className="ui-sans text-xs uppercase tracking-[0.25em] text-[var(--ink-soft)]">
        the environment
      </p>
      <h1 className="mt-2 max-w-2xl text-4xl leading-tight">
        A world small enough to know completely
      </h1>
      <p className="mt-5 max-w-xl text-lg leading-relaxed text-[var(--ink-soft)]">
        Every scene is generated, so every scene arrives with exact ground
        truth: the form, color, position and orientation of each shape,
        and a caption produced from those same factors by a fixed
        template. No annotation noise, no distribution drift, and full
        control over what the captions choose to say. The generator below
        is the training generator, running in your browser.
      </p>

      <section aria-labelledby="explore-title" className="mt-14">
        <h2 id="explore-title" className="text-2xl">Inspect the scenes</h2>
        <p className="mb-8 mt-3 max-w-xl leading-relaxed text-[var(--ink-soft)]">
          Pick any scene, then hover the ground-truth rows to see which
          caption words each factor produced.
        </p>
        <DatasetExplorer />
      </section>

      <section aria-labelledby="mapping-title" className="mt-16 border-t hairline pt-12">
        <h2 id="mapping-title" className="text-2xl">The factor mapping</h2>
        <p className="mb-8 mt-3 max-w-xl leading-relaxed text-[var(--ink-soft)]">
          Each factor exists twice: once in pixels, once in words. The
          table records both encodings and the coverage of each.
        </p>
        <DatasetMapping />
      </section>
    </main>
  );
}
