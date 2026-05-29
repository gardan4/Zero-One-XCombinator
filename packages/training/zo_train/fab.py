"""Fab-step sequence data for the Industrial AI (Infineon) track.

Stdlib only — **no torch / pandas / datasets** — so it imports fine on a laptop and is
cheap to call from training, eval, and the agent harness alike.

The track data lives in `data/industrial-infineon/training_data/`:

- ``{FAMILY}_variants.csv`` — long format ``SEQUENCE_ID,STEP``; 1000 process sequences per
  family (IC / IGBT / MOSFET), one step per row, in order. **This is the training data.**
- ``{FAMILY}_Longdescr.csv`` — ``STEP,DESCRIPTION``; human-readable descriptions for steps.
  Useful reference, but **not** a complete step list — some steps in the variants never appear
  here. So the vocabulary is the *union* of canonical (Longdescr) and observed (variants) steps;
  that way every step a model actually sees has an id (no silent ``<unk>`` on real data).

Everything that touches token ids (a trainer, an eval scorer, an anomaly detector) MUST use
the *same* vocabulary or ids won't line up. Two ways to guarantee that:

1. ``build_vocab()`` is **deterministic** — specials, then family tokens, then the sorted
   union of all known steps (canonical ∪ observed). Same inputs → same ids, on any machine.
2. A run can pin its vocab to disk with ``Vocab.save()`` and everyone downstream loads the
   exact same file with ``Vocab.load()``. Prefer this once a run exists.

Quick start::

    from zo_train.fab import build_vocab, load_encoded
    vocab, seqs = load_encoded("MOSFET")          # vocab + integer-encoded sequences
    ids = vocab.encode(["RECEIVE WAFER LOT"], family="MOSFET", add_special=True)
    vocab.decode(ids)                              # -> ['RECEIVE WAFER LOT']
"""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from zo_common.paths import repo_root

# Three product families shipped in the track data.
FAMILIES: tuple[str, ...] = ("IC", "IGBT", "MOSFET")

# Special tokens with FIXED ids — keep PAD at 0 so it's the natural padding/ignore value.
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD, UNK, BOS, EOS)
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def default_data_dir() -> Path:
    """`<repo>/data/industrial-infineon/training_data` — the vendored track data."""
    return repo_root() / "data" / "industrial-infineon" / "training_data"


def family_token(family: str) -> str:
    """A control token that conditions a single model on the product family, e.g. ``<fam:IGBT>``."""
    return f"<fam:{family.upper()}>"


# --- raw CSV readers ---------------------------------------------------------------------
# All files are read with utf-8-sig so a leading BOM (present in the Longdescr CSVs) is
# stripped, and via csv.* so quoted fields are handled correctly.


def _read_column(path: Path, column: str) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return [row[column].strip() for row in reader if row.get(column, "").strip()]


def canonical_steps(family: str, data_dir: Path | None = None) -> list[str]:
    """The authoritative list of valid steps for a family (from ``{FAMILY}_Longdescr.csv``).

    Order-preserving and de-duplicated (so it doubles as the family's reference recipe).
    """
    data_dir = data_dir or default_data_dir()
    path = data_dir / f"{family}_Longdescr.csv"
    return list(OrderedDict.fromkeys(_read_column(path, "STEP")))


def all_steps(family: str, data_dir: Path | None = None) -> set[str]:
    """Every distinct step for a family: canonical (Longdescr) ∪ observed (variants).

    The Longdescr files are incomplete, so the variants must be scanned too or the vocab would
    miss steps that actually occur in the sequences.
    """
    steps = set(canonical_steps(family, data_dir))
    for seq in read_sequences(family, data_dir):
        steps.update(seq)
    return steps


def read_sequences(family: str, data_dir: Path | None = None) -> list[list[str]]:
    """Load ``{FAMILY}_variants.csv`` into a list of step-name sequences.

    Rows are grouped by ``SEQUENCE_ID`` with both the steps within a sequence and the
    sequences themselves kept in first-seen order.
    """
    data_dir = data_dir or default_data_dir()
    path = data_dir / f"{family}_variants.csv"
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            seq_id = (row.get("SEQUENCE_ID") or "").strip()
            step = (row.get("STEP") or "").strip()
            if seq_id and step:
                grouped.setdefault(seq_id, []).append(step)
    return list(grouped.values())


# --- vocabulary --------------------------------------------------------------------------


@dataclass
class Vocab:
    """A deterministic step <-> id mapping shared across training, eval, and agents.

    Layout: ``SPECIAL_TOKENS`` (ids 0-3), then one ``<fam:*>`` token per family, then the
    sorted union of canonical steps. Keep that ordering stable — saved checkpoints and
    pinned vocab files depend on it.
    """

    itos: list[str]

    def __post_init__(self) -> None:
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}
        self._special: frozenset[str] = frozenset(SPECIAL_TOKENS) | {
            t for t in self.itos if t.startswith("<fam:")
        }

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, step: str) -> bool:
        return step in self.stoi

    def id_of(self, token: str) -> int:
        return self.stoi.get(token, UNK_ID)

    def encode(
        self,
        steps: Iterable[str],
        family: str | None = None,
        add_special: bool = False,
    ) -> list[int]:
        """Step names -> ids. Unknown steps map to ``UNK``.

        With ``add_special`` the sequence is wrapped ``[BOS, (<fam>), ...steps, EOS]``; the
        family token is inserted when ``family`` is given.
        """
        out: list[int] = []
        if add_special:
            out.append(BOS_ID)
        if family is not None:
            out.append(self.id_of(family_token(family)))
        out.extend(self.stoi.get(s, UNK_ID) for s in steps)
        if add_special:
            out.append(EOS_ID)
        return out

    def decode(self, ids: Iterable[int], keep_special: bool = False) -> list[str]:
        """Ids -> step names. Drops special/family tokens unless ``keep_special``."""
        toks = [self.itos[i] if 0 <= i < len(self.itos) else UNK for i in ids]
        if keep_special:
            return toks
        return [t for t in toks if t not in self._special]

    def save(self, path: str | Path) -> Path:
        """Pin the vocab to JSON so every downstream task can load the identical mapping."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos}, indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Vocab:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(itos=data["itos"])


def build_vocab(
    data_dir: Path | None = None,
    families: Iterable[str] = FAMILIES,
) -> Vocab:
    """Build the deterministic shared vocabulary covering every step in the data.

    ``SPECIAL_TOKENS`` + ``<fam:*>`` tokens + the sorted union of every family's known steps
    (canonical ∪ observed, via ``all_steps``). Deterministic: same data → same ids on any
    machine. Scans the variants once; pin the result with ``Vocab.save()`` to avoid re-scanning.
    """
    families = list(families)
    steps: set[str] = set()
    for fam in families:
        steps.update(all_steps(fam, data_dir))
    itos = [*SPECIAL_TOKENS, *(family_token(f) for f in families), *sorted(steps)]
    return Vocab(itos=itos)


# --- encoded datasets --------------------------------------------------------------------


def load_encoded(
    families: str | Iterable[str] = FAMILIES,
    data_dir: Path | None = None,
    vocab: Vocab | None = None,
    add_family_token: bool = True,
    add_special: bool = True,
) -> tuple[Vocab, list[list[int]]]:
    """Load one or more families as integer-encoded sequences.

    Returns ``(vocab, sequences)``. Pass a pinned ``vocab`` to guarantee id alignment with a
    specific run; otherwise a fresh deterministic one is built.
    """
    if isinstance(families, str):
        families = [families]
    families = list(families)
    data_dir = data_dir or default_data_dir()
    vocab = vocab or build_vocab(data_dir)
    out: list[list[int]] = []
    for fam in families:
        for seq in read_sequences(fam, data_dir):
            out.append(
                vocab.encode(
                    seq,
                    family=fam if add_family_token else None,
                    add_special=add_special,
                )
            )
    return vocab, out


def next_step_pairs(seq: list[int]) -> Iterator[tuple[list[int], int]]:
    """Yield ``(prefix, target)`` for every position — the core next-step-prediction signal."""
    for i in range(1, len(seq)):
        yield seq[:i], seq[i]


def summarize(data_dir: Path | None = None) -> dict[str, object]:
    """Quick health check: vocab size, per-family sequence counts, length stats, UNK rate."""
    data_dir = data_dir or default_data_dir()
    vocab = build_vocab(data_dir)
    per_family: dict[str, dict[str, float]] = {}
    unknown_steps: set[str] = set()
    for fam in FAMILIES:
        seqs = read_sequences(fam, data_dir)
        lengths = [len(s) for s in seqs]
        for s in seqs:
            unknown_steps.update(step for step in s if step not in vocab)
        per_family[fam] = {
            "sequences": len(seqs),
            "min_len": min(lengths) if lengths else 0,
            "max_len": max(lengths) if lengths else 0,
            "mean_len": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        }
    return {
        "vocab_size": len(vocab),
        "families": per_family,
        "unknown_steps": sorted(unknown_steps),  # should be empty if Longdescr is complete
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import pprint

    pprint.pp(summarize())
