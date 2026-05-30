import type { Metadata } from "next";
import "./globals.css";

import { NavBar } from "@/components/NavBar";

export const metadata: Metadata = {
  title: "Zero One — Industrial AI",
  description: "Fab process-logic modeling — training, OOD eval, and a recipe copilot",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="flex items-center justify-between gap-4 border-b border-neutral-800 px-6 py-3">
          <a href="/" className="text-lg font-semibold tracking-tight">
            Zero One <span className="text-neutral-500">/ Infineon fab-route</span>
          </a>
          <NavBar />
        </header>
        <main className="px-6 py-6">{children}</main>
      </body>
    </html>
  );
}
