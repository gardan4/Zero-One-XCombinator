# `public/` — the results dashboard

Static frontend served by the Worker. No build step, no framework — plain HTML/CSS/JS.

| File | Role |
| --- | --- |
| `index.html` | Page structure and section anchors. |
| `styles.css` | Infineon-brand light theme (Hanken Grotesk + IBM Plex Mono). |
| `app.js` | Renders every section from the `RESULTS` object and wires the live-demo strip to the backend API. |
| `results.js` | **The single source of truth for every number on the page.** |

## `results.js` is the only file you edit to update results

The entire dashboard renders from one exported object, `RESULTS`. Swap it and the page
re-renders — no changes to `app.js` or `index.html` needed.

> ⚠️ The values shipped today are **ILLUSTRATIVE placeholders** standing in for the
> trained model. Replace them with real eval-harness output to go live.

### Shape of `RESULTS`

| Key | Contents |
| --- | --- |
| `copy` | All hero/section copy and the model name + blurb. |
| `families` / `datasetFamilies` | The three trained families and their dataset stats. |
| `rules` | The process rules the anomaly task checks. |
| `nextstep` | Top-1/3/5 + MRR, baseline vs fine-tuned, overall and per family. |
| `completion` | Exact match, normalized edit distance, token/block accuracy. |
| `anomaly` | Binary + rule-attribution metrics, confusion matrix, per-rule attribution. |
| `training` | Param count, epochs, final loss, and the logged loss curve (`steps`). |
| `ood` | Held-out fourth family — ID vs OOD F1 / Top-1 / exact match. |

Conventions: fractions are in `[0, 1]`; `normEditDistance` is **lower-is-better**.

See the top-level [`README.md`](../README.md) for how the dashboard relates to the
live-demo backend.
