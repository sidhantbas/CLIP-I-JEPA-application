"use client";

/** Lab III: the Handshake and the Barman. One composition serves both
 *  panels: the caption crosses the bridge and lands in the wordless
 *  space, then stands as the barman's order. No gate here; the ledger
 *  is the exit. */

import { useEffect, useState } from "react";
import Composer, { Composition, EMPTY, phrase } from "@/components/namer/composer";
import Barman from "@/components/handshake/barman";
import Bridge from "@/components/handshake/bridge";
import { prepare } from "@/lib/inference";

export default function HandshakeLab() {
  const [order, setOrder] = useState<Composition>(EMPTY);
  useEffect(() => {
    prepare("tongue", "bridge_pooled", "bridge_attentive", "eye");
  }, []);
  const caption = phrase(order);

  return (
    <section id="handshake" aria-labelledby="handshake-title" className="mt-14">
      <h2 id="handshake-title" className="text-2xl">
        Language-conditioned retrieval and control
      </h2>
      <p className="mt-4 max-w-xl leading-relaxed text-[var(--ink-soft)]">
        The two models never shared any training. A two-layer MLP is
        trained to map CLIP's text embeddings into I-JEPA's visual
        representation space, which contains no language at all. Compose
        a goal below; watch it land among 512 real scenes; then hand it
        to a planner that must move a shape to the described position
        using only I-JEPA features.
      </p>

      <div className="mt-10 max-w-xl">
        <Composer value={order}
          onPick={(slot, word, optional) => setOrder((old) => ({
            ...old, [slot]: old[slot] === word && optional ? null : word }))} />
      </div>

      <div className="mt-12 grid gap-14 lg:grid-cols-2">
        <div>
          <h3 className="mb-4 text-xl">Cross-model retrieval</h3>
          <Bridge caption={caption} />
        </div>
        <div>
          <h3 className="mb-4 text-xl">Language-conditioned planning</h3>
          <Barman order={order} />
        </div>
      </div>
    </section>
  );
}
