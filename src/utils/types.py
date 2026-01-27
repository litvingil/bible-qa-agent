from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class Verse:
    """A verse with full metadata."""
    id: str
    book: str
    book_en: str
    book_id: str
    chapter: int
    verse: int
    text: str
    page: int

@dataclass
class Chunk:
    """A chunk of verses with metadata."""
    chunk_id: str
    book: str
    book_en: str
    book_id: str
    chapter: int
    verse_start: int
    verse_end: int
    page_start: int
    page_end: int
    verse_ids: List[str]
    text: str

@dataclass
class SearchHit:
    rank: int
    score: float
    chunk: Chunk

@dataclass
class SearchResult:
    """A single search result."""
    verse_id: str
    book: str
    book_en: str
    chapter: int
    verse: int
    text: str
    page: int
    score: float
    verse_ids: Optional[List[str]] = None

JsonDict = Dict[str, Any]