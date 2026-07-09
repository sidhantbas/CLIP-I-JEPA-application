import type { Metadata } from "next";
import type { ReactNode } from "react";
import Nav from "@/components/stage/nav";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "CLIP and I-JEPA, live in the browser",
    template: "%s · CLIP and I-JEPA, live",
  },
  description:
    "Interactive demonstrations of contrastive vision-language learning " +
    "(CLIP) and self-supervised latent prediction (I-JEPA), trained on a " +
    "controlled synthetic environment and running entirely in the browser.",
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen pt-12">
        <Nav />
        {children}
      </body>
    </html>
  );
}
