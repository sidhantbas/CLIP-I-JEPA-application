"use client";

/** Lab I: the Namer. To know is to name. The visitor composes captions
 *  from the closed vocabulary and watches the grid rerank live; a flip
 *  reverses the direction. The quiet lesson sits beneath the grid:
 *  words the captions always spoke move the ranking hard, words spoken
 *  only half the time barely move it. */

import { useCallback, useEffect, useState } from "react";
import Composer, { Composition, EMPTY, phrase } from "@/components/namer/composer";
import AccordDiagram from "@/components/namer/diagram";
import RerankGrid from "@/components/namer/grid";
import { prepare } from "@/lib/inference";

export default function NamerLab({ onPass }: { onPass?: () => void }) {
  const [composition, setComposition] = useState<Composition>(EMPTY);
  const [flipped, setFlipped] = useState<number | null>(null);
  const [sims, setSims] = useState<number[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [composedOnce, setComposedOnce] = useState(false);
  const [flippedOnce, setFlippedOnce] = useState(false);

  useEffect(() => { prepare("gaze", "tongue"); }, []);
  useEffect(() => {
    if (composedOnce && flippedOnce) onPass?.();
  }, [composedOnce, flippedOnce, onPass]);

  const caption = phrase(composition);

  const handlePick = (slot: keyof Composition, word: string,
                      optional: boolean) => {
    setComposition((old) => {
      const next = { ...old,
        [slot]: old[slot] === word && optional ? null : word };
      if (phrase(next)) { setComposedOnce(true); setBusy(true); }
      return next;
    });
  };
  const handleSims = useCallback((next: number[] | null) => {
    setSims(next);
    setBusy(false);
  }, []);
  const handleTongue = useCallback((ms: number) => {
    if (process.env.NODE_ENV === "development") {
      console.log(`[vitrine] tongue forward ${ms.toFixed(1)}ms`);
    }
  }, []);
  const handleFlip = (index: number | null) => {
    setFlipped(index);
    if (index !== null) setFlippedOnce(true);
  };

  return (
    <section id="namer" aria-labelledby="namer-title"
             className="mt-16 border-t hairline pt-12">
      <h2 id="namer-title" className="text-2xl">Live demo: text-based image retrieval</h2>
      <p className="mt-3 max-w-xl leading-relaxed text-[var(--ink-soft)]">
        Compose a query from the closed vocabulary; the text encoder runs
        in your browser on every edit (about 2 ms), and the sixteen
        scenes re-rank by cosine similarity in the shared embedding space.
        Click any scene to reverse the direction and retrieve captions for
        an image instead.
      </p>

      <div className="mt-12 grid gap-12 lg:grid-cols-[1fr_1.4fr]">
        <div className="space-y-10">
          <AccordDiagram sims={sims} tongueBusy={busy} />
          <Composer value={composition} onPick={handlePick} />
        </div>
        <div>
          <RerankGrid caption={caption} flipped={flipped} onFlip={handleFlip}
                      onSims={handleSims} onTongueRan={handleTongue} />
          <p className="mt-6 max-w-lg text-sm leading-relaxed text-[var(--ink-soft)]">
            Observation: attributes present in every training caption
            (color, shape, region) move the ranking strongly. The facing
            attribute appeared in only half the training captions, and
            its words barely move it. The representation mirrors the
            coverage of its supervision.
          </p>
          <p className="ui-sans mt-4 text-xs text-[var(--ink-soft)]">
            {composedOnce ? "query composed" : "compose one full query"}
            {" · "}
            {flippedOnce ? "direction reversed" : "click a scene to reverse the direction"}
          </p>
        </div>
      </div>
    </section>
  );
}
