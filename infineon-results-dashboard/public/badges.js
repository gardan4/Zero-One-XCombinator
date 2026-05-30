/** Data-source badges for mock vs live vs demo sections. */

const BADGE_META = {
  mock: { label: "MOCK", title: "Static placeholder from results.js — not from the registry" },
  partial: {
    label: "PARTIAL",
    title: "Mix of measured baselines and illustrative finetuned values in results.js",
  },
  live: { label: "LIVE", title: "Real metrics from the FastAPI experiment registry" },
  demo: { label: "DEMO", title: "Live Worker retrieval baseline on this page" },
  proxy: { label: "PROXY", title: "Kickoff proxy metrics only — run with --gold for full task scores" },
};

export function dataBadge(kind) {
  const m = BADGE_META[kind] || { label: kind.toUpperCase(), title: "" };
  return `<span class="data-badge ${kind}" title="${m.title}">${m.label}</span>`;
}

export function cardSub(label, badgeKind) {
  return `<p class="card-sub">${label} ${dataBadge(badgeKind)}</p>`;
}
