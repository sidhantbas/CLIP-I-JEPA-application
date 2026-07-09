"use client";

/** The caption composer: word chips constrained to the closed
 *  vocabulary, arranged as the grammar's slots so every composition is
 *  a sentence the tongue has actually heard. The facing slot is
 *  optional, exactly as it was in training, where captions mentioned
 *  it only half the time. */

import { COLORS, COMPASS_8, FORMS, GRID_NAMES } from "@/lib/world";

export interface Composition {
  size: string | null; color: string | null; form: string | null;
  region: string | null; facing: string | null;
}

export const EMPTY: Composition = {
  size: null, color: null, form: null, region: null, facing: null,
};

export function phrase(c: Composition): string | null {
  if (!c.size || !c.color || !c.form || !c.region) return null;
  const base = `a ${c.size} ${c.color} ${c.form} rests in the ${c.region}`;
  return c.facing ? `${base}, facing ${c.facing}` : base;
}

const SLOTS: Array<{ name: keyof Composition; words: string[]; optional?: boolean }> = [
  { name: "size", words: ["small", "large"] },
  { name: "color", words: Object.keys(COLORS) },
  { name: "form", words: [...FORMS] },
  { name: "region", words: GRID_NAMES.flat() as string[] },
  { name: "facing", words: [...COMPASS_8], optional: true },
];

function Chip({ word, active, onPick }: {
  word: string; active: boolean; onPick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onPick}
      aria-pressed={active}
      className={`ui-sans rounded-full border px-3 py-1 text-xs transition-colors
        ${active
          ? "border-[var(--ink)] bg-[var(--ink)] text-[var(--paper)]"
          : "hairline bg-[var(--paper)] text-[var(--ink-soft)] hover:border-[var(--ink-soft)]"}`}
    >
      {word}
    </button>
  );
}

export default function Composer({ value, onPick }: {
  value: Composition;
  onPick: (slot: keyof Composition, word: string, optional: boolean) => void;
}) {
  const caption = phrase(value);
  return (
    <div className="space-y-5">
      {SLOTS.map((slot) => (
        <fieldset key={slot.name}>
          <legend className="ui-sans mb-2 text-[10px] uppercase tracking-[0.2em] text-[var(--ink-soft)]">
            {slot.name === "form" ? "shape" : slot.name}{slot.optional ? " (mentioned in only half of training captions)" : ""}
          </legend>
          <div className="flex flex-wrap gap-2">
            {slot.words.map((word) => (
              <Chip
                key={word}
                word={word}
                active={value[slot.name] === word}
                onPick={() => onPick(slot.name, word, slot.optional ?? false)}
              />
            ))}
          </div>
        </fieldset>
      ))}
      <p aria-live="polite" className="min-h-[1.75rem] border-l-2 pl-3 text-base italic"
         style={{ borderColor: "var(--hairline)" }}>
        {caption ?? "choose a size, a color, a shape and a region to form a query"}
      </p>
    </div>
  );
}
