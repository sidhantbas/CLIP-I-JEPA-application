/** The factor mapping, stated once as a table: how each generative
 *  factor is encoded in pixels, how it surfaces in the caption, and how
 *  often. The asymmetry in the last column is the experiment. */

const ROWS = [
  {
    factor: "form", values: "4: triangle, square, circle, star",
    pixels: "the silhouette", words: "a noun", rate: "always",
  },
  {
    factor: "color", values: "5: red, blue, green, yellow, violet",
    pixels: "the fill", words: "an adjective", rate: "always",
  },
  {
    factor: "position", values: "continuous x, y",
    pixels: "placement on the canvas",
    words: "one of 9 region names (3x3 grid)", rate: "always, primary shape",
  },
  {
    factor: "orientation", values: "continuous angle, binned to 8",
    pixels: "a notch toward the facing direction",
    words: "a compass phrase, 'facing north-east'", rate: "50% of captions",
  },
  {
    factor: "size", values: "radius 8 to 15 px",
    pixels: "the circumradius", words: "'small' or 'large'",
    rate: "always (not probed)",
  },
];

const SCALE = [
  { n: "20,000", what: "scenes per training run, for each model" },
  { n: "10,000 + 2,000", what: "probe training and test scenes, disjoint" },
  { n: "26", what: "words in the closed caption vocabulary" },
  { n: "1 to 3", what: "shapes per scene; captions describe the largest fully" },
];

export default function DatasetMapping() {
  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] max-w-3xl text-sm">
          <thead>
            <tr className="ui-sans text-left text-[10px] uppercase tracking-[0.2em] text-[var(--ink-soft)]">
              <th className="border-b hairline py-2 pr-4 font-normal">factor</th>
              <th className="border-b hairline py-2 pr-4 font-normal">values</th>
              <th className="border-b hairline py-2 pr-4 font-normal">in the image</th>
              <th className="border-b hairline py-2 pr-4 font-normal">in the caption</th>
              <th className="border-b hairline py-2 font-normal">mentioned</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => (
              <tr key={row.factor}>
                <td className="ui-sans border-b hairline py-2.5 pr-4 text-xs">{row.factor}</td>
                <td className="figure-number border-b hairline py-2.5 pr-4 text-xs">{row.values}</td>
                <td className="border-b hairline py-2.5 pr-4 text-xs text-[var(--ink-soft)]">{row.pixels}</td>
                <td className="border-b hairline py-2.5 pr-4 text-xs text-[var(--ink-soft)]">{row.words}</td>
                <td className={`border-b hairline py-2.5 text-xs ${
                  row.factor === "orientation" ? "font-medium" : "text-[var(--ink-soft)]"}`}
                    style={row.factor === "orientation" ? { color: "var(--accent)" } : undefined}>
                  {row.rate}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-6 max-w-2xl leading-relaxed text-[var(--ink-soft)]">
        The image always shows all four probed factors; the caption names
        orientation only half the time. Because CLIP learns exclusively
        from image-caption agreement while I-JEPA never reads a caption,
        this one controlled gap lets the experiments separate what a
        model saw from what its objective taught it to keep.
      </p>

      <dl className="mt-8 grid max-w-3xl grid-cols-2 gap-6 sm:grid-cols-4">
        {SCALE.map((item) => (
          <div key={item.what}>
            <dt className="figure-number text-xl">{item.n}</dt>
            <dd className="ui-sans mt-1 text-xs leading-relaxed text-[var(--ink-soft)]">
              {item.what}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
