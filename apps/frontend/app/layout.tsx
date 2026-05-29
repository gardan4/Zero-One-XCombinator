import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Zero One — Runs",
  description: "Finetuning / RL / eval / agent dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="border-b border-neutral-800 px-6 py-4">
          <a href="/" className="text-lg font-semibold tracking-tight">
            Zero One <span className="text-neutral-500">/ runs</span>
          </a>
        </header>
        <main className="px-6 py-6">{children}</main>
      </body>
    </html>
  );
}
