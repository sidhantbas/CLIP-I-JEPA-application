import type { Metadata } from "next";
import Link from "next/link";
import ClipArchitecture from "@/components/clip/architecture";
import NamerLab from "@/components/namer/lab";

export const metadata: Metadata = {
  title: "CLIP: contrastive vision-language learning",
  description:
    "A miniature CLIP trained from scratch: dual encoders, a contrastive " +
    "objective, and a live retrieval demo running in the browser.",
};

export default function ClipPage() {
  return (
    <main className="mx-auto max-w-5xl px-6 pb-20 pt-16">
      <p className="ui-sans text-xs uppercase tracking-[0.25em] text-[var(--ink-soft)]">
        application 1 · contrastive vision-language learning
      </p>
      <h1 className="mt-2 max-w-2xl text-4xl leading-tight">
        CLIP: learning by matching images to captions
      </h1>
      <p className="mt-5 max-w-xl text-lg leading-relaxed text-[var(--ink-soft)]">
        Two encoders, one shared embedding space. The image encoder never
        reads text and the text encoder never sees pixels; the only
        training signal is which caption belongs to which image. From
        that matching game alone, the model reaches 97% retrieval@1
        within training batches of 256.
      </p>
      <p className="mt-3 max-w-xl leading-relaxed text-[var(--ink-soft)]">
        The consequence to watch for: the model can only learn
        distinctions its captions make. The{" "}
        <Link href="/dataset" className="underline decoration-dotted underline-offset-4">
          dataset
        </Link>{" "}
        names orientation in only half of its captions, and the demo
        below shows exactly that gap in the learned space.
      </p>

      <section aria-labelledby="clip-arch-title" className="mt-14">
        <h2 id="clip-arch-title" className="text-2xl">The architecture</h2>
        <p className="mb-8 mt-3 max-w-xl leading-relaxed text-[var(--ink-soft)]">
          Every block, with its output shape and parameter count, exactly
          as trained. Press a block for its purpose.
        </p>
        <ClipArchitecture />
      </section>

      <NamerLab />
    </main>
  );
}
