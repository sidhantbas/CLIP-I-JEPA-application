import type { Metadata } from "next";
import Link from "next/link";
import IJepaArchitecture from "@/components/ijepa/architecture";
import IJepaLab from "@/components/ijepa/lab";

export const metadata: Metadata = {
  title: "I-JEPA: self-supervised latent prediction",
  description:
    "A miniature I-JEPA trained from scratch: masked latent prediction " +
    "with a live mask-and-predict demo and probing figure in the browser.",
};

export default function IJepaPage() {
  return (
    <main className="mx-auto max-w-5xl px-6 pb-20 pt-16">
      <p className="ui-sans text-xs uppercase tracking-[0.25em] text-[var(--ink-soft)]">
        application 2 · self-supervised latent prediction
      </p>
      <h1 className="mt-2 max-w-2xl text-4xl leading-tight">
        I-JEPA: learning by predicting what is hidden
      </h1>
      <p className="mt-5 max-w-xl text-lg leading-relaxed text-[var(--ink-soft)]">
        No captions, no labels, no pixel reconstruction and no noise. A
        region of the image is masked; an encoder sees only the visible
        remainder; a predictor must produce the internal representation
        of the hidden region, as judged by a slowly averaged copy of the
        encoder that saw everything. The model is never asked to draw
        the missing pixels, only to predict what they would mean.
      </p>
      <p className="mt-3 max-w-xl leading-relaxed text-[var(--ink-soft)]">
        Because the loss lives in representation space, the encoder may
        discard what does not help prediction. On this{" "}
        <Link href="/dataset" className="underline decoration-dotted underline-offset-4">
          dataset
        </Link>{" "}
        it keeps shape, color and position, and drops orientation
        entirely; the probing figure below makes both facts measurable.
      </p>

      <section aria-labelledby="jepa-arch-title" className="mt-14">
        <h2 id="jepa-arch-title" className="text-2xl">The architecture</h2>
        <p className="mb-8 mt-3 max-w-xl leading-relaxed text-[var(--ink-soft)]">
          Two paths through the same architecture: the online path is
          trained, the target path is its exponential moving average.
          Press a block for its purpose.
        </p>
        <IJepaArchitecture />
      </section>

      <IJepaLab />
    </main>
  );
}
