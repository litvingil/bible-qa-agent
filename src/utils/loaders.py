from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .types import Chunk, Verse


def load_verses(path: str | Path) -> List[Verse]:
    """Load a flat list of verses (verses.json)."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    verses: List[Verse] = []
    for v in raw:
        verses.append(
            Verse(
                id=v["id"],
                book=v["book"],
                book_en=v.get("book_en", ""),
                book_id=v["book_id"],
                chapter=int(v["chapter"]),
                verse=int(v["verse"]),
                text=v["text"],
                page=int(v.get("page", -1)),
            )
        )
    verses.sort(key=lambda x: (x.book_id, x.chapter, x.verse))
    return verses


def compute_coverage(verses: Sequence[Verse]) -> dict:
    """Compute basic coverage info to help the agent abstain correctly."""
    by_book = defaultdict(list)
    for v in verses:
        by_book[v.book_id].append(v)

    coverage = {}
    for book_id, lst in by_book.items():
        lst_sorted = sorted(lst, key=lambda x: (x.chapter, x.verse))
        first = lst_sorted[0]
        last = lst_sorted[-1]
        coverage[book_id] = {
            "book_he": first.book,
            "book_en": first.book_en,
            "chapters_present": sorted({v.chapter for v in lst_sorted}),
            "first_ref": {"chapter": first.chapter, "verse": first.verse, "page": first.page},
            "last_ref": {"chapter": last.chapter, "verse": last.verse, "page": last.page},
            "total_verses": len(lst_sorted),
        }
    return coverage


def make_chunks(
    verses: Sequence[Verse],
    verses_per_chunk: int = 6,
    overlap: int = 2,
) -> List[Chunk]:
    """
    Turn verses into small overlapping chunks for retrieval.

    For narrative text, chunks of ~4-8 verses tend to work well.
    Overlap helps follow-up queries that refer to 'the paragraph above'.
    """
    if verses_per_chunk < 1:
        raise ValueError("verses_per_chunk must be >= 1")
    if overlap < 0 or overlap >= verses_per_chunk:
        raise ValueError("overlap must be in [0, verses_per_chunk-1]")

    by_chapter: Dict[Tuple[str, int], List[Verse]] = defaultdict(list)
    for v in verses:
        by_chapter[(v.book_id, v.chapter)].append(v)

    chunks: List[Chunk] = []
    step = max(1, verses_per_chunk - overlap)

    for (book_id, chapter), lst in sorted(by_chapter.items(), key=lambda x: (x[0][0], x[0][1])):
        lst = sorted(lst, key=lambda x: x.verse)
        for start in range(0, len(lst), step):
            window = lst[start : start + verses_per_chunk]
            if not window:
                continue
            verse_ids = tuple(v.id for v in window)
            verse_start = window[0].verse
            verse_end = window[-1].verse
            page_start = min(v.page for v in window if v.page is not None)
            page_end = max(v.page for v in window if v.page is not None)
            chunk_id = f"{book_id}-{chapter}-{verse_start}-{verse_end}"
            text = "\n".join([v.text for v in window])
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    book_id=book_id,
                    book=window[0].book,
                    book_en=window[0].book_en,
                    chapter=chapter,
                    verse_start=verse_start,
                    verse_end=verse_end,
                    page_start=page_start,
                    page_end=page_end,
                    verse_ids=verse_ids,
                    text=text,
                )
            )
        # Also add a full-chapter chunk (useful for broad "summarize chapter X" queries).
        if len(lst) > verses_per_chunk:
            verse_ids = tuple(v.id for v in lst)
            chunk_id = f"{book_id}-{chapter}-FULL"
            text = "\n".join([v.text for v in lst])
            page_start = min(v.page for v in lst if v.page is not None)
            page_end = max(v.page for v in lst if v.page is not None)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    book_id=book_id,
                    book=lst[0].book,
                    book_en=lst[0].book_en,
                    chapter=chapter,
                    verse_start=lst[0].verse,
                    verse_end=lst[-1].verse,
                    page_start=page_start,
                    page_end=page_end,
                    verse_ids=verse_ids,
                    text=text,
                )
            )

    return chunks


def save_chunks(chunks: Sequence[Chunk], path: str | Path) -> None:
    p = Path(path)
    data = []
    for c in chunks:
        data.append(
            {
                "chunk_id": c.chunk_id,
                "book_id": c.book_id,
                "book": c.book,
                "book_en": c.book_en,
                "chapter": c.chapter,
                "verse_start": c.verse_start,
                "verse_end": c.verse_end,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "verse_ids": list(c.verse_ids),
                "text": c.text,
            }
        )
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(path: str | Path) -> List[Chunk]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    chunks: List[Chunk] = []
    for c in raw:
        chunks.append(
            Chunk(
                chunk_id=c["chunk_id"],
                book_id=c["book_id"],
                book=c["book"],
                book_en=c.get("book_en", ""),
                chapter=int(c["chapter"]),
                verse_start=int(c["verse_start"]),
                verse_end=int(c["verse_end"]),
                page_start=int(c["page_start"]),
                page_end=int(c["page_end"]),
                verse_ids=tuple(c["verse_ids"]),
                text=c["text"],
            )
        )
    return chunks


def save_verses(verses: Sequence[Verse], path: str | Path) -> None:
    p = Path(path)
    data = []
    for v in verses:
        data.append(
            {
                "id": v.id,
                "book": v.book,
                "book_en": v.book_en,
                "book_id": v.book_id,
                "chapter": v.chapter,
                "verse": v.verse,
                "text": v.text,
                "page": v.page,
            }
        )
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
