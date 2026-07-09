"use client";

/** The shared window into the world: renders ShapeFactors to a crisp,
 *  pixel-doubled canvas. Rendering is the same float64 arithmetic the
 *  Python world used; only the frame differs. */

import { useEffect, useRef } from "react";
import { SIZE, ShapeFactors, render } from "@/lib/world";

export function paintInto(canvas: HTMLCanvasElement,
                          factors: ShapeFactors[]): void {
  const rgb = render(factors);
  const image = new ImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    image.data[i * 4] = Math.round(rgb[i * 3] * 255);
    image.data[i * 4 + 1] = Math.round(rgb[i * 3 + 1] * 255);
    image.data[i * 4 + 2] = Math.round(rgb[i * 3 + 2] * 255);
    image.data[i * 4 + 3] = 255;
  }
  canvas.getContext("2d")?.putImageData(image, 0, 0);
}

export default function SceneCanvas({
  factors, caption, scale = 2, className = "",
}: {
  factors: ShapeFactors[];
  caption: string;
  scale?: number;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (ref.current) paintInto(ref.current, factors);
  }, [factors]);
  return (
    <canvas
      ref={ref}
      width={SIZE}
      height={SIZE}
      role="img"
      aria-label={caption}
      className={`pixelated ${className}`}
      style={{ width: SIZE * scale, height: SIZE * scale }}
    />
  );
}
