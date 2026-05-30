"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Runs" },
  { href: "/compare", label: "Compare" },
  { href: "/copilot", label: "Copilot" },
];

export function NavBar() {
  const path = usePathname();
  return (
    <nav className="flex items-center gap-1 text-sm">
      {LINKS.map((l) => {
        const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
        return (
          <Link
            key={l.href}
            href={l.href}
            className={`rounded-md px-3 py-1.5 transition ${
              active
                ? "bg-neutral-800 text-neutral-100"
                : "text-neutral-400 hover:bg-neutral-900 hover:text-neutral-200"
            }`}
          >
            {l.label}
          </Link>
        );
      })}
    </nav>
  );
}
