"use client";

/** The ledger: the whole argument as a three-rung ladder, present,
 *  reachable, trainable at budget, each rung standing on its numbers
 *  from the shipped JSONs. Orientation's casualty note, the full-scale
 *  implication, and the honest footnote about session tallies. */

import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useState } from "react";
import { theBarman, theInterrogation, theUnpooled } from "@/lib/results";

interface Rung { title: string; line: string; numbers: string }

export default function Ledger() {
  const still = useReducedMotion() ?? false;
  const [rungs, setRungs] = useState<Rung[] | null>(null);
  const [casualty, setCasualty] = useState<string>("");

  useEffect(() => {
    Promise.all([theInterrogation(), theUnpooled(), theBarman()])
      .then(([pooled, unpooled, barman]) => {
        const s = unpooled.sculptor;
        setRungs([
          {
            title: "present",
            line: "The information must exist in the representation at all.",
            numbers: `position: MLP probe on pooled I-JEPA features ${pooled.sculptor.position.scrutiny.mean.toFixed(3)}; linear probe at token level ${s.position.glance_tokens.mean.toFixed(3)}, above the raw-pixel baseline of ${unpooled.pixels.position.glance_tokens.mean.toFixed(3)}`,
          },
          {
            title: "accessible",
            line: "A readout the application can afford must be able to reach it.",
            numbers: `planning success with an attention-based cost head: ${(barman.sculptor.attend * 100).toFixed(1)}% of ${barman.episodes} episodes, against ${(barman.sculptor.glance_pooled * 100).toFixed(1)}% with pooled features`,
          },
          {
            title: "trainable at budget",
            line: "Accessible is not enough; the readout must also be learnable on the available data.",
            numbers: `the highest-capacity heads planned worse: bilinear over all tokens ${(barman.sculptor.token_glance * 100).toFixed(1)}%, token-level MLP ${(barman.sculptor.token_scrutiny * 100).toFixed(1)}%`,
          },
        ]);
        setCasualty(
          `Orientation, under every probe and every pooling: ${s.orientation.glance_tokens.mean.toFixed(3)} against chance ${unpooled.chance.orientation.toFixed(3)}.`);
      });
  }, []);

  return (
    <section id="ledger" aria-labelledby="ledger-title"
             className="mt-16 max-w-3xl border-t hairline pt-12">
      <h2 id="ledger-title" className="text-3xl">Conclusions</h2>
      <div className="mt-10 space-y-8">
        {rungs?.map((rung, i) => (
          <motion.div key={rung.title}
            initial={still ? false : { opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ delay: still ? 0 : i * 0.25, duration: 0.5 }}
            className="border-l-2 pl-5" style={{ borderColor: "var(--ink)" }}>
            <h3 className="text-xl">{i + 1}. {rung.title}</h3>
            <p className="mt-1 text-[var(--ink-soft)]">{rung.line}</p>
            <p className="figure-number ui-sans mt-2 text-xs text-[var(--ink-soft)]">
              {rung.numbers}
            </p>
          </motion.div>
        ))}
      </div>

      <p className="mt-12 max-w-xl leading-relaxed">
        {casualty} That is the one genuine blind spot of the I-JEPA
        objective in this study. Everything else the model had learned;
        the evaluation was asking badly.
      </p>
      <p className="mt-4 max-w-xl leading-relaxed text-[var(--ink-soft)]">
        The broader implication, in one sentence: when a self-supervised
        encoder such as I-JEPA appears not to know something, examine the
        readout before indicting the objective, because average pooling
        alone can hide information a downstream application could have
        used.
      </p>
      <p className="ui-sans mt-8 max-w-xl text-xs leading-relaxed text-[var(--ink-soft)]">
        A methodological note: your session runs the same models as the
        recorded experiments, but over far fewer episodes, so small
        samples will vary. The recorded rates come from 500 fixed
        episodes per condition.
      </p>
      <p className="ui-sans mt-6 text-xs">
        <a href="/results/report.md" className="underline decoration-dotted underline-offset-4">
          the full report
        </a>
        {" · "}
        <a href="/results/parity.json" className="underline decoration-dotted underline-offset-4">
          the parity manifest
        </a>
        <span className="text-[var(--ink-soft)]">
          {" "}· the atelier repository holds the training code, frozen
        </span>
      </p>
    </section>
  );
}
