"use client";

/** The site navigation: one fixed bar naming the overview, the dataset
 *  and the three applications. The current page is read from the
 *  pathname; every link is a plain anchor, so the bar works without
 *  JavaScript once the static export is served. */

import Link from "next/link";
import { usePathname } from "next/navigation";

const PAGES = [
  { href: "/", label: "Overview" },
  { href: "/dataset", label: "Dataset" },
  { href: "/clip", label: "CLIP" },
  { href: "/ijepa", label: "I-JEPA" },
  { href: "/control", label: "Combined" },
  { href: "/commander", label: "Control" },
  { href: "/latent-space", label: "Game" },
];

export default function Nav() {
  const pathname = usePathname();
  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b hairline bg-[var(--paper)]/95 backdrop-blur-sm">
      <nav aria-label="site"
           className="ui-sans mx-auto flex h-12 max-w-5xl items-center gap-1 overflow-x-auto px-4 text-xs sm:px-6">
        <span className="mr-4 hidden whitespace-nowrap tracking-wide text-[var(--ink-soft)] sm:inline">
          CLIP and I-JEPA, live
        </span>
        {PAGES.map((page) => {
          const here = pathname === page.href;
          return (
            <Link key={page.href} href={page.href}
              aria-current={here ? "page" : undefined}
              className={`whitespace-nowrap rounded-full px-3 py-1.5 transition-colors ${
                here
                  ? "bg-[var(--ink)] text-[var(--paper)]"
                  : "text-[var(--ink-soft)] hover:text-[var(--ink)]"}`}>
              {page.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
