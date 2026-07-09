import type { Metadata } from "next";
import Link from "next/link";
import HandshakeLab from "@/components/handshake/lab";
import Ledger from "@/components/handshake/ledger";

export const metadata: Metadata = {
  title: "Combined: language-conditioned retrieval and control",
  description:
    "Bridging CLIP's text embeddings into I-JEPA's visual space to drive " +
    "cross-model retrieval and a language-conditioned planner, live.",
};

export default function ControlPage() {
  return (
    <main className="mx-auto max-w-5xl px-6 pb-20 pt-16">
      <p className="ui-sans text-xs uppercase tracking-[0.25em] text-[var(--ink-soft)]">
        application 3 · combining CLIP and I-JEPA
      </p>
      <h1 className="mt-2 max-w-2xl text-4xl leading-tight">
        Reading language into a space that has none
      </h1>
      <p className="mt-5 max-w-xl text-lg leading-relaxed text-[var(--ink-soft)]">
        The two models shared no training and no vocabulary. A two-layer
        MLP is trained to map{" "}
        <Link href="/clip" className="underline decoration-dotted underline-offset-4">
          CLIP
        </Link>
        &apos;s text embeddings into{" "}
        <Link href="/ijepa" className="underline decoration-dotted underline-offset-4">
          I-JEPA
        </Link>
        &apos;s visual representation, which contains no language at all.
        If that bridge carries meaning, a caption can retrieve images and
        even direct a planner, using only the wordless features.
      </p>

      <HandshakeLab />
      <Ledger />
    </main>
  );
}
