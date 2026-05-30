# XCombinator team additions in vendored / organizer files

The Lumos hackathon pack and upstream `tracks/industrial-infineon/` materials are vendored **as-is**
where possible. When we need to correct or extend them, we use this **visible marker** (search the repo
for `XCombinator-TEAM-START`):

```
========== XCombinator-TEAM-START ==========
(not in original organizer / Lumos materials)
...
========== XCombinator-TEAM-END ==========
```

In Markdown we use the same labels in a blockquote or horizontal-rule block.

**Files with inline team additions**

| File | What we added |
|------|----------------|
| `docs/Track One Assignment.txt` | Errata block at top (scoring, rubrics, self-eval) |
| `data/industrial-infineon/Track_industrial_en.md` | Errata block at top |
| `data/industrial-infineon/Track_industrial.md` | Errata block at top (DE) |
| `data/industrial-infineon/README.md` | Eval section: team block + **original** self-eval text preserved |
| `data/industrial-infineon/eval/README.md` | Kickoff eval inputs + usage |
| `docs/eval-and-artifacts.md` | Eval workflow, tagging, HF model cards, command reference |
| `docs/submission/SUBMISSION.md` | Team block after original rubrics link (Industrial AI scoring) |
| `docs/submission/REPORT_TEMPLATE.md` | Team blocks for Results + Industrial checklist (original lines kept) |

**Team-authored docs** (entire file is ours, not upstream): [eval-and-artifacts.md](eval-and-artifacts.md), `docs/track-industrial-sources.md`, this file, root `README.md` track section, `.claude/knowledge/*` updates.
