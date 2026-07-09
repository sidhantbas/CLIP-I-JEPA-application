"use client";

/** The prologue: the world, streaming. Scenes conjure themselves on a
 *  slow pulse while their captions type alongside; ten seconds of
 *  ambience establish the whole domain. Deterministic from seed 7, so
 *  every visitor meets the same opening scenes. */

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";
import SceneCanvas from "@/components/stage/scene-canvas";
import { theTokenizerSpec } from "@/lib/results";
import { Scene, conjure, mulberry32 } from "@/lib/world";

const KEEP = 4;

function TypedCaption({ text, still }: { text: string; still: boolean }) {
  const [shown, setShown] = useState(still ? text.length : 0);
  useEffect(() => {
    if (still) { setShown(text.length); return; }
    setShown(0);
    const beat = window.setInterval(
      () => setShown((n) => (n >= text.length ? n : n + 1)), 28);
    return () => window.clearInterval(beat);
  }, [text, still]);
  return (
    <p aria-label={text}
       className="h-32 overflow-hidden text-sm leading-relaxed text-[var(--ink-soft)]">
      {text.slice(0, shown)}
      <span aria-hidden className="opacity-40">{shown < text.length ? "|" : ""}</span>
    </p>
  );
}

export default function Prologue() {
  const rng = useRef(mulberry32(7));
  const [scenes, setScenes] = useState<Scene[]>(() => []);
  const [words, setWords] = useState<number | null>(null);
  const still = useReducedMotion() ?? false;

  useEffect(() => {
    theTokenizerSpec().then((spec) => setWords(spec.vocabulary.length));
    setScenes([conjure(rng.current)]);
    const pulse = window.setInterval(() => {
      setScenes((seen) => [...seen.slice(-(KEEP - 1)), conjure(rng.current)]);
    }, still ? 3600 : 2600);
    return () => window.clearInterval(pulse);
  }, [still]);

  return (
    <section id="prologue" aria-labelledby="prologue-title" className="mx-auto max-w-4xl px-6 pt-24 pb-20">
      <p className="ui-sans text-xs uppercase tracking-[0.25em] text-[var(--ink-soft)]">
        an interactive demonstration
      </p>
      <h1 id="prologue-title" className="mt-4 max-w-2xl text-4xl leading-tight sm:text-5xl">
        Two ways of learning visual representations.
      </h1>
      <p className="mt-6 max-w-xl text-lg leading-relaxed text-[var(--ink-soft)]">
        Two models, each about two million parameters, were trained from
        scratch on the controlled environment below. The first is a
        miniature CLIP: it learns by matching images to their captions.
        The second is a miniature I-JEPA: it learns with no text at all,
        by masking parts of an image and predicting the internal
        representation of what is hidden. Both run live in this page,
        and every number shown is either computed here or read from the
        recorded experiments.
      </p>

      <div className="mt-14 grid grid-cols-2 gap-6 sm:grid-cols-4">
        {scenes.map((scene, i) => (
          <figure key={scene.caption + i} className="space-y-3">
            <SceneCanvas
              factors={scene.factors}
              caption={scene.caption}
              scale={2}
              className="border hairline"
            />
            {i === scenes.length - 1 ? (
              <TypedCaption text={scene.caption} still={still} />
            ) : (
              <figcaption className="h-32 overflow-hidden text-sm leading-relaxed text-[var(--ink-soft)]">
                {scene.caption}
              </figcaption>
            )}
          </figure>
        ))}
      </div>

      <p className="mt-12 text-base italic text-[var(--ink-soft)]">
        {words === null
          ? " "
          : `Four ground-truth factors per shape; every caption is built from a vocabulary of ${words} words.`}
      </p>
    </section>
  );
}
