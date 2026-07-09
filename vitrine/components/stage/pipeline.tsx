"use client";

/** Shared primitives for the architecture diagrams: a column of
 *  selectable stage blocks joined by labeled arrows, and the card that
 *  explains the selected block. Selection is an ordinary button press,
 *  so the diagrams read fully with a keyboard and a screen reader. */

export interface Stage {
  id: string;
  label: string;      // the block's name, e.g. "conv stage 2"
  output: string;     // the tensor leaving the block
  params: string;     // approximate learned parameter count, "0" if none
  detail: string;     // what the block does and why it is there
}

export function StageColumn({ title, stages, selected, onSelect }: {
  title: string;
  stages: Stage[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div>
      <h4 className="ui-sans mb-3 text-[10px] uppercase tracking-[0.2em] text-[var(--ink-soft)]">
        {title}
      </h4>
      <ol className="ui-sans">
        {stages.map((stage, i) => {
          const active = selected === stage.id;
          return (
            <li key={stage.id}>
              <button type="button" onClick={() => onSelect(stage.id)}
                aria-pressed={active}
                aria-label={`${stage.label}, output ${stage.output}, ${stage.params} parameters. ${stage.detail}`}
                className={`flex w-full items-baseline justify-between gap-3 border px-3 py-2 text-left text-xs transition-colors ${
                  active
                    ? "border-[var(--ink)] bg-[var(--ink)] text-[var(--paper)]"
                    : "hairline bg-[var(--paper)] hover:border-[var(--ink-soft)]"}`}>
                <span>{stage.label}</span>
                <span className={`figure-number whitespace-nowrap text-[10px] ${
                  active ? "text-[var(--paper)]/70" : "text-[var(--ink-soft)]"}`}>
                  {stage.params}
                </span>
              </button>
              {i < stages.length - 1 && (
                <p aria-hidden
                   className="figure-number py-1 pl-3 text-[10px] leading-4 text-[var(--ink-soft)]">
                  &darr; {stage.output}
                </p>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export function DetailCard({ stage, hint }: { stage: Stage | null; hint: string }) {
  return (
    <div aria-live="polite"
         className="border hairline bg-[var(--wash)] px-5 py-4">
      {stage ? (
        <>
          <p className="ui-sans flex flex-wrap items-baseline gap-x-4 gap-y-1 text-xs">
            <span className="font-medium">{stage.label}</span>
            <span className="figure-number text-[var(--ink-soft)]">
              output {stage.output}
            </span>
            <span className="figure-number text-[var(--ink-soft)]">
              {stage.params} parameters
            </span>
          </p>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed">{stage.detail}</p>
        </>
      ) : (
        <p className="text-sm leading-relaxed text-[var(--ink-soft)]">{hint}</p>
      )}
    </div>
  );
}
