"use client";

/** The overview: a hero that states the question, the ambient stream of
 *  generated scenes as evidence that the world is live, and four cards
 *  routing to the dataset and the three applications. Each application
 *  now lives on its own page; this page is the map. */

import Prologue from "@/components/prologue/stream";
import OverviewCards from "@/components/overview/cards";

export default function Page() {
  return (
    <main>
      <Prologue />
      <section aria-labelledby="apps-title" className="mx-auto max-w-4xl px-6 pb-24">
        <h2 id="apps-title" className="text-2xl">Three applications, one environment</h2>
        <p className="mb-8 mt-3 max-w-xl leading-relaxed text-[var(--ink-soft)]">
          Each runs entirely in your browser on exported models, verified
          against the original PyTorch to within 1e-4. Start with the
          dataset, or go straight to a model.
        </p>
        <OverviewCards />
        <p className="ui-sans mt-10 max-w-xl text-xs leading-relaxed text-[var(--ink-soft)]">
          Every number on these pages is read from the recorded experiment
          files or computed live by the exported models in your browser;
          nothing is hardcoded. The atelier repository holds the training
          code, frozen.
        </p>
      </section>
    </main>
  );
}
