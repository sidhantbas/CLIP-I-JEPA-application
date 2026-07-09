import type { Metadata } from "next";
import Link from "next/link";
import LatentGame from "@/components/latent/game";

export const metadata: Metadata = {
  title: "Latent Space: The Architect's Dilemma",
  description:
    "A deduction game whose mechanics are the I-JEPA architecture: read " +
    "the visible context, predict the redacted zone's semantic embedding, " +
    "and minimize the L2 distance to the Ghost.",
};

const ROWS = [
  ["Context Encoder (x → z_x)", "Observation Module", "Reads the visible zones into semantic feature blocks."],
  ["Target Encoder (y → z_y)", "The Ghost", "Holds the ground-truth embedding of the redacted zone; drifts by EMA."],
  ["Predictor (z_x, Δy → ẑ_y)", "Predictor Terminal", "Turns your deduction plus a positional offset into a predicted embedding."],
  ["L2 distance loss", "Alignment Engine", "Scores the strict structural distance between your prediction and the Ghost."],
];

export default function LatentSpacePage() {
  return (
    <main className="mx-auto max-w-5xl px-6 pb-20 pt-16">
      <p className="ui-sans text-xs uppercase tracking-[0.25em] text-[var(--ink-soft)]">
        application 5 · the architecture, as a game
      </p>
      <h1 className="mt-2 max-w-3xl text-4xl leading-tight">
        Latent Space: The Architect&rsquo;s Dilemma
      </h1>
      <p className="mt-5 max-w-2xl text-lg leading-relaxed text-[var(--ink-soft)]">
        The other sections observe the model from outside. This one puts
        you inside its objective. One block in a row is redacted &mdash; a
        Void Zone that returns only informational static. You cannot see
        its pixels; nothing here evaluates pixels. You read the visible{" "}
        <Link href="/ijepa" className="underline decoration-dotted underline-offset-4">
          context
        </Link>
        , deduce the hidden block&rsquo;s semantic embedding, and predict
        it. The Alignment Engine measures your prediction&rsquo;s L2
        distance to the Ghost; land inside the threshold and the static
        dissipates into a pristine wireframe. You are doing, by hand, what
        the predictor learns to do: infer a representation you were never
        shown.
      </p>

      <div className="mt-8 overflow-x-auto">
        <table className="w-full min-w-[560px] max-w-3xl text-sm">
          <thead>
            <tr className="ui-sans text-left text-[10px] uppercase tracking-[0.16em] text-[var(--ink-soft)]">
              <th className="border-b hairline py-2 pr-4 font-normal">I-JEPA component</th>
              <th className="border-b hairline py-2 pr-4 font-normal">game system</th>
              <th className="border-b hairline py-2 font-normal">responsibility</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r[0]}>
                <td className="ui-sans border-b hairline py-2.5 pr-4 text-xs text-[var(--ink-soft)]">{r[0]}</td>
                <td className="border-b hairline py-2.5 pr-4 text-xs font-medium">{r[1]}</td>
                <td className="border-b hairline py-2.5 text-xs text-[var(--ink-soft)]">{r[2]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section aria-labelledby="game" className="mt-10">
        <h2 id="game" className="sr-only">The game</h2>
        <LatentGame />
      </section>

      <section aria-labelledby="notes" className="mt-16 border-t hairline pt-12">
        <h2 id="notes" className="text-2xl">How the mechanics stay faithful</h2>
        <p className="mt-4 max-w-2xl leading-relaxed text-[var(--ink-soft)]">
          Every block is a semantic embedding &mdash; a vector encoding
          geometry, material, and dynamics &mdash; never an image. Your
          prediction is scored by the exact loss the real objective uses:
          squared L2 distance in embedding space. Alignment is a distance
          below the threshold ε, not a pixel match. The Ghost is a target
          encoder: it holds the ground truth and its stability updates by
          an exponential moving average of your mastery, so a run of clean
          alignments makes it more trustworthy and a run of failures makes
          it drift. Correct alignments pay compute energy in proportion to
          how tight the alignment was; dissonant ones spend it.
        </p>
        <p className="ui-sans mt-8 max-w-2xl text-xs leading-relaxed text-[var(--ink-soft)]">
          Scope. The engine &mdash; embeddings, L2 loss, threshold, EMA,
          reward &mdash; runs live in the browser and follows the design
          specification exactly. The &ldquo;encoders&rdquo; are behavioral
          stand-ins for the exported I-JEPA network characterized in the
          preceding sections, not that network executing inference. Every
          puzzle is constructed so the context uniquely determines the
          redacted block: a correct deduction reaches L2 distance zero, and
          a wrong geometry or material provably exceeds the threshold.
        </p>
      </section>
    </main>
  );
}
