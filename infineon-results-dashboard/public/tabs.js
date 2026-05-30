/** Simple tab switching for the dashboard page. */

export function initTabs() {
  const nav = document.getElementById("pageTabs");
  if (!nav) return;

  const panels = [...document.querySelectorAll(".tab-panel")];
  const buttons = [...nav.querySelectorAll(".page-tab")];

  function show(tabId) {
    buttons.forEach((b) => b.classList.toggle("active", b.dataset.tab === tabId));
  panels.forEach((p) => {
    const on = p.dataset.tab === tabId;
    p.classList.toggle("hidden", !on);
    p.classList.toggle("active", on);
  });
    try {
      history.replaceState(null, "", `#${tabId}`);
    } catch {
      /* ignore */
    }
  }

  nav.addEventListener("click", (e) => {
    const btn = e.target.closest(".page-tab");
    if (!btn) return;
    show(btn.dataset.tab);
  });

  const hash = (location.hash || "").replace(/^#/, "");
  if (hash && buttons.some((b) => b.dataset.tab === hash)) {
    show(hash);
  }
}
