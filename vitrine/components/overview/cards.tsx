import Link from "next/link";

const CARDS = [
  {
    href: "/dataset", eyebrow: "the environment",
    title: "The dataset",
    body: "Procedural scenes with exact ground truth. Inspect any scene, " +
      "see which caption words each factor produced, and find the one " +
      "deliberate gap in supervision.",
    cta: "Explore the world",
  },
  {
    href: "/clip", eyebrow: "application 1",
    title: "CLIP",
    body: "Contrastive vision-language learning. Two encoders meet in one " +
      "embedding space; compose a query and watch retrieval re-rank live. " +
      "The model learns only what its captions say.",
    cta: "Retrieve by text",
  },
  {
    href: "/ijepa", eyebrow: "application 2",
    title: "I-JEPA",
    body: "Self-supervised latent prediction. Mask a region and watch the " +
      "model predict its representation, then probe what the encoder kept " +
      "and what its objective threw away.",
    cta: "Mask and predict",
  },
  {
    href: "/control", eyebrow: "application 3",
    title: "Combined",
    body: "A learned bridge carries a caption from CLIP's text space into " +
      "I-JEPA's wordless visual space, driving cross-model retrieval and a " +
      "language-conditioned planner.",
    cta: "Bridge the two",
  },
];

export default function OverviewCards() {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {CARDS.map((card) => (
        <Link key={card.href} href={card.href}
          className="group flex flex-col border hairline bg-[var(--paper)] p-6 transition-colors hover:border-[var(--ink)]">
          <p className="ui-sans text-[10px] uppercase tracking-[0.2em] text-[var(--ink-soft)]">
            {card.eyebrow}
          </p>
          <h3 className="mt-2 text-2xl">{card.title}</h3>
          <p className="mt-3 flex-1 text-sm leading-relaxed text-[var(--ink-soft)]">
            {card.body}
          </p>
          <p className="ui-sans mt-5 text-xs text-[var(--ink)]">
            {card.cta}
            <span aria-hidden className="ml-1 inline-block transition-transform group-hover:translate-x-1">
              &rarr;
            </span>
          </p>
        </Link>
      ))}
    </div>
  );
}
