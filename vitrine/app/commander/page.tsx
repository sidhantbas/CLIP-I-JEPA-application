import type { Metadata } from "next";
import Link from "next/link";
import CommanderLab from "@/components/commander/lab";

export const metadata: Metadata = {
  title: "Instruction-conditioned layout",
  description:
    "A text-driven layout language over a 4x4 canvas: any color, any " +
    "shape and a rich spatial vocabulary (diagonals, halves, border, " +
    "checkerboard, letter-shapes), with additive step-through composition.",
};

export default function CommanderPage() {
  return (
    <main className="mx-auto max-w-5xl px-6 pb-20 pt-16">
      <p className="ui-sans text-xs uppercase tracking-[0.25em] text-[var(--ink-soft)]">
        application 4 · instruction-conditioned layout
      </p>
      <h1 className="mt-2 max-w-3xl text-4xl leading-tight">
        Constructing a scene from language, cell by cell
      </h1>
      <p className="mt-5 max-w-2xl text-lg leading-relaxed text-[var(--ink-soft)]">
        The preceding sections measured what each encoder represents. This
        section turns representation into construction: a natural-language
        instruction is parsed into a structured layout specification, and a
        compiler realizes it exactly on a 4&times;4 canvas. You hold total
        control from text over three factors at once &mdash; color, shape,
        and spatial arrangement.
      </p>
      <p className="mt-4 max-w-2xl leading-relaxed text-[var(--ink-soft)]">
        The spatial vocabulary is deliberately broad. Beyond filling the
        whole grid, an instruction may target a diagonal or anti-diagonal,
        a half or a row or a column, the border, the center, a
        checkerboard, the corners, or a letter-shape such as L, H, T, X, O
        or plus. Any color and any shape may be bound to any of these
        regions. Parsing a sentence into this joint specification, and
        grounding it precisely in geometry, is a compact test of
        compositional spatial reasoning.
      </p>
      <p className="mt-4 max-w-2xl leading-relaxed text-[var(--ink-soft)]">
        Instructions also compose in sequence. Prefixed with{" "}
        <em>add</em> or <em>serially add</em>, an instruction stacks onto
        the current canvas rather than replacing it, and the new cells
        draw in one at a time &mdash; the behavior of a sequential
        predictor completing a scene, in the spirit of the{" "}
        <Link href="/ijepa" className="underline decoration-dotted underline-offset-4">
          masked-prediction encoder
        </Link>
        . A layout can therefore be built up layer by layer: an L in one
        color, an anti-diagonal in another, a border on top.
      </p>

      <section aria-labelledby="control-lab" className="mt-12">
        <h2 id="control-lab" className="sr-only">Interactive layout control</h2>
        <CommanderLab />
      </section>

      <section aria-labelledby="interpretation" className="mt-16 border-t hairline pt-12">
        <h2 id="interpretation" className="text-2xl">Interpretation</h2>
        <p className="mt-4 max-w-2xl leading-relaxed text-[var(--ink-soft)]">
          Total text control over color, shape, and arrangement is more
          than a convenience: it is the operational form of compositional
          grounding. A single sentence selects one value along each of
          three independent axes and binds them to a set of grid cells,
          and the system must honor all three jointly. Sequential,
          additive construction extends this from a static specification to
          a process, where each instruction is interpreted against the
          state the previous instructions produced. The canvas is small
          enough that every placement can be verified by eye, and rich
          enough that the space of expressible layouts is large.
        </p>
        <p className="ui-sans mt-8 max-w-2xl text-xs leading-relaxed text-[var(--ink-soft)]">
          Scope. The parsing, compilation, and rendering run live in the
          browser and are exact by construction. The &ldquo;text
          encoder&rdquo; and layout compiler are behavioral stand-ins for
          the exported CLIP and I-JEPA networks characterized in the
          preceding sections, not those networks executing inference; this
          section demonstrates the instruction-to-geometry interface, not
          neural inference over it. No ground-truth layout is consulted:
          the canvas is constructed toward the stated specification alone.
        </p>
      </section>
    </main>
  );
}
