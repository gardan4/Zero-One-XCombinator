---
description: Record a learning into the knowledge base (topic file + INDEX log)
argument-hint: <what you learned>
allowed-tools: Read, Edit, Write
---
Capture this learning for the team: $ARGUMENTS

This is how the knowledge base stays current — do it carefully.

1. Decide where it belongs: one of
   `.claude/knowledge/{stack,cluster,training,eval,agents,hackathon}.md`. If it's a genuine
   architectural decision, create/append an ADR under `.claude/knowledge/decisions/` instead.
2. Append a short, concrete entry to that file (use its "Append below as you learn" section). If the
   learning **corrects** an existing note, fix the stale line in place rather than leaving a
   contradiction.
3. Add a one-line dated entry to the Learnings log at the bottom of `.claude/knowledge/INDEX.md`:
   `YYYY-MM-DD — <one line> — (topic)`. Use today's actual date.
4. Keep it terse and factual — a teammate's fresh Claude session should be able to act on it cold.
   Confirm what you wrote and where.
