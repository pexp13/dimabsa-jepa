"""Loading DimABSA / DimStance files as flat (text, aspect) -> (V, A) instances."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SPLIT_ALIASES = {"train": "train", "dev": "dev", "test": "test"}


@dataclass
class Instance:
    id: str
    text: str
    aspect: str
    va: tuple[float, float] | None = None  # None for unlabeled data
    opinion: str | None = None  # only in *_train_alltasks files
    source: str = ""  # e.g. "eng_restaurant"


def _parse_va(s: str) -> tuple[float, float]:
    v, a = s.split("#")
    return float(v), float(a)


def read_file(path: str | Path) -> list[Instance]:
    """Read one jsonl file in any of the task formats.

    Handles `Quadruplet` / `Triplet` (train_alltasks), `Aspect_VA` (subtask 1
    train/dev/gold test) and bare `Aspect` lists (unlabeled test input).
    Implicit aspects ("NULL") are skipped: subtask 1 only scores explicit ones.
    """
    path = Path(path)
    source = "_".join(path.name.split("_")[:-2])
    out: list[Instance] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items = row.get("Quadruplet") or row.get("Triplet") or row.get("Aspect_VA")
            if items is None:
                items = [{"Aspect": a} for a in row.get("Aspect", [])]
            for it in items:
                aspect = it["Aspect"]
                if aspect == "NULL":
                    continue
                opinion = it.get("Opinion")
                out.append(
                    Instance(
                        id=str(row["ID"]),
                        text=row["Text"],
                        aspect=aspect,
                        va=_parse_va(it["VA"]) if "VA" in it else None,
                        opinion=None if opinion in (None, "NULL") else opinion,
                        source=source,
                    )
                )
    return out


def merge_duplicates(instances: list[Instance]) -> list[Instance]:
    """Average VA over repeated (ID, aspect) pairs.

    train_alltasks files list one entry per (aspect, opinion) pair, while
    subtask 1 scores a single VA per aspect.
    """
    groups: dict[tuple[str, str], list[Instance]] = {}
    for inst in instances:
        groups.setdefault((inst.id, inst.aspect), []).append(inst)
    out = []
    for group in groups.values():
        first = group[0]
        if len(group) > 1 and all(g.va is not None for g in group):
            va = tuple(float(np.mean([g.va[k] for g in group])) for k in range(2))
            first = Instance(first.id, first.text, first.aspect, va, first.opinion, first.source)
        out.append(first)
    return out


def find_files(root: str | Path, split: str, track: str = "a",
               langs: list[str] | None = None, domains: list[str] | None = None) -> list[Path]:
    """Subtask-1 files for a split, optionally filtered by language / domain."""
    base = Path(root) / "task-dataset" / f"track_{track}" / "subtask_1"
    files = []
    for p in sorted(base.glob(f"*/*_{split}_*.jsonl")):
        lang = p.parent.name
        domain = p.name[len(lang) + 1: p.name.index(f"_{split}_")]
        if langs and lang not in langs:
            continue
        if domains and domain not in domains:
            continue
        files.append(p)
    return files


def find_span(text: str, phrase: str | None) -> tuple[int, int] | None:
    """Character span of `phrase` in `text` (case-insensitive), or None."""
    if not phrase:
        return None
    i = text.lower().find(phrase.lower())
    return None if i < 0 else (i, i + len(phrase))


def sample_block_mask(n: int, ratio: float, max_span: int, rng: np.random.Generator) -> np.ndarray:
    """Mask ~ratio of n positions with contiguous blocks; keeps >=1 visible if n > 1."""
    mask = np.zeros(n, dtype=bool)
    if n <= 1:
        return mask
    target = int(min(max(1, round(ratio * n)), n - 1))
    for _ in range(4 * n):
        remaining = target - int(mask.sum())
        if remaining <= 0:
            break
        length = int(min(rng.integers(1, max_span + 1), remaining))
        start = int(rng.integers(0, n - length + 1))
        mask[start:start + length] = True
    if mask.sum() > target:  # overlapping blocks cannot overshoot, but stay safe
        on = np.flatnonzero(mask)
        mask[on[target:]] = False
    return mask


class Collator:
    """Tokenizes (aspect, text) pairs and builds the JEPA target mask.

    jepa_mask marks *text* tokens hidden from the context encoder and predicted
    in latent space. strategy:
      random  - random contiguous blocks covering ~mask_ratio of the text
      opinion - the annotated opinion span when available, else random blocks
    """

    def __init__(self, tokenizer, max_length: int = 128, mask_strategy: str = "random",
                 mask_ratio: float = 0.3, max_span: int = 5, seed: int = 0):
        self.tok = tokenizer
        self.max_length = max_length
        self.mask_strategy = mask_strategy
        self.mask_ratio = mask_ratio
        self.max_span = max_span
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch: list[Instance]) -> dict[str, torch.Tensor]:
        enc = self.tok(
            [b.aspect for b in batch], [b.text for b in batch],
            truncation="only_second", max_length=self.max_length,
            padding=True, return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping").numpy()
        n, length = enc["input_ids"].shape
        jepa_mask = np.zeros((n, length), dtype=bool)
        for i, b in enumerate(batch):
            seq = enc.sequence_ids(i)
            text_pos = np.array([j for j, s in enumerate(seq) if s == 1], dtype=int)
            if len(text_pos) <= 1:
                continue
            chosen = None
            if self.mask_strategy == "opinion":
                span = find_span(b.text, b.opinion)
                if span is not None:
                    s, e = span
                    hit = [j for j in text_pos if offsets[i, j, 0] < e and offsets[i, j, 1] > s]
                    if 0 < len(hit) < len(text_pos):
                        chosen = np.array(hit, dtype=int)
            if chosen is None:
                m = sample_block_mask(len(text_pos), self.mask_ratio, self.max_span, self.rng)
                chosen = text_pos[m]
            jepa_mask[i, chosen] = True
        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "jepa_mask": torch.from_numpy(jepa_mask),
        }
        if all(b.va is not None for b in batch):
            out["labels"] = torch.tensor([b.va for b in batch], dtype=torch.float)
        return out


def write_predictions(path: str | Path, instances: list[Instance], preds: np.ndarray) -> None:
    """Write subtask-1 submission jsonl (ID, Aspect_VA), preserving input order."""
    rows: dict[str, list[dict]] = {}
    for inst, (v, a) in zip(instances, preds):
        rows.setdefault(inst.id, []).append({"Aspect": inst.aspect, "VA": f"{v:.2f}#{a:.2f}"})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for id_, items in rows.items():
            f.write(json.dumps({"ID": id_, "Aspect_VA": items}, ensure_ascii=False) + "\n")
