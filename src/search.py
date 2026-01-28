"""
Search infrastructure for Hebrew Bible Q&A.
Provides keyword (BM25) and semantic search capabilities.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .utils.book_mappings import BOOK_ID_ALIASES, HEBREW_BOOK_MAP
from .utils.types import Chunk, Verse, SearchResult

import faiss
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv


def normalize_book_id(book_id: Optional[str]) -> Optional[str]:
    """Normalize a book ID to its canonical form."""
    if book_id is None:
        return None
    book_id_lower = book_id.lower()
    return BOOK_ID_ALIASES.get(book_id_lower, book_id)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize vectors for cosine similarity."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms

class BibleSearch:
    """Search engine for the Hebrew Bible."""

    def __init__(self, data_dir: str = None):
        """Initialize the search engine."""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data" / "bible"
        self.data_dir = Path(data_dir)

        self.verses: List[Verse] = []
        self.verse_index: dict = {}  # verse_id -> Verse
        self.books: dict = {}
        self.metadata: dict = {}

        # BM25 index
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

        # Semantic search
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self.faiss_index_path = self._resolve_faiss_path()
        self.chunk_index_meta_path = self._resolve_chunk_meta_path()
        self.chunks_path = self._resolve_chunks_path()
        self.chunk_ids: List[str] = []
        self.chunks: List[Chunk] = []
        self.chunks_by_id: Dict[str, Chunk] = {}
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self._load_data()

    def _resolve_faiss_path(self) -> Path:
        env_path = os.getenv("BIBLE_FAISS_INDEX_PATH")
        if env_path:
            return Path(env_path)
        candidate = self.data_dir / "index" / "index.faiss"
        if candidate.exists():
            return candidate
        candidate = self.data_dir / "index.faiss"
        if candidate.exists():
            return candidate
        fallback = self.data_dir.parent / "index.faiss"
        return fallback

    def _resolve_chunk_meta_path(self) -> Path:
        env_path = os.getenv("BIBLE_INDEX_META_PATH")
        if env_path and Path(env_path).exists():
            return Path(env_path)
        candidate = self.data_dir / "index" / "index_meta.json"
        if candidate.exists():
            return candidate
        else:
            raise FileNotFoundError(f"Index meta file not found at {candidate}")

    def _resolve_chunks_path(self) -> Path:
        env_path = os.getenv("BIBLE_CHUNKS_PATH")
        if env_path and Path(env_path).exists():
            return Path(env_path)
        candidate = self.data_dir / "index" / "chunks.json"
        if candidate.exists():
            return candidate
        else:
            raise FileNotFoundError(f"Chunks file not found at {candidate}")

    def _load_data(self):
        """Load all Bible data from JSON files."""
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        # Load verses
        verses_path = self.data_dir / "index" / "verses.json"
        if verses_path.exists():
            with open(verses_path, "r", encoding="utf-8") as f:
                verses_data = json.load(f)
                for v in verses_data:
                    verse = Verse(
                        id=v["id"],
                        book=v["book"],
                        book_en=v["book_en"],
                        book_id=v["book_id"],
                        chapter=v["chapter"],
                        verse=v["verse"],
                        text=v["text"],
                        page=v["page"]
                    )
                    self.verses.append(verse)
                    self.verse_index[verse.id] = verse

        # Load book data
        books_dir = self.data_dir / "books"
        if books_dir.exists():
            for book_file in books_dir.glob("*.json"):
                with open(book_file, "r", encoding="utf-8") as f:
                    book_data = json.load(f)
                    self.books[book_data["id"]] = book_data

        # Build BM25 index
        self._build_bm25_index()

        # Load FAISS index if present; otherwise build from embeddings
        self._load_chunks()
        self._load_faiss_index()
        self._load_embeddings()

    def _load_chunks(self) -> None:
        """Load chunk metadata and index meta for chunk mapping (required)."""
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {self.chunks_path}")

        raw_chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))

        chunks: List[Chunk] = []
        for c in raw_chunks:
            chunks.append(Chunk(
                chunk_id=c["chunk_id"],
                book=c["book"],
                book_en=c["book_en"],
                book_id=c["book_id"],
                chapter=int(c["chapter"]),
                verse_start=int(c["verse_start"]),
                verse_end=int(c["verse_end"]),
                page_start=int(c["page_start"]),
                page_end=int(c["page_end"]),
                verse_ids=list(c.get("verse_ids") or []),
                text=c.get("text") or "",
            ))

        self.chunks = chunks
        self.chunks_by_id = {c.chunk_id: c for c in chunks}

        if not self.chunk_index_meta_path.exists():
            raise FileNotFoundError(f"Chunk index meta not found at {self.chunk_index_meta_path}")

        meta = json.loads(self.chunk_index_meta_path.read_text(encoding="utf-8"))
        self.chunk_ids = list(meta.get("chunk_ids") or [])

    def _load_faiss_index(self):
        """Load the FAISS index (required)."""
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.faiss_index_path}")
        self.faiss_index = faiss.read_index(str(self.faiss_index_path))

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize Hebrew text for BM25."""
        # Remove nikud (vowel points) and cantillation marks
        text = re.sub(r'[\u0591-\u05C7]', '', text)
        # Remove other common problematic characters from PDF extraction
        text = text.replace('˄', 'ל')  # Common PDF extraction error
        text = text.replace('׃', '')   # Sof pasuq
        # Find Hebrew words - include both Hebrew block (0590-05FF) and
        # Hebrew Presentation Forms (FB00-FB4F) which contain letters with dagesh/shin dots
        tokens = re.findall(r'[\u0590-\u05FF\uFB00-\uFB4F]+', text)
        return tokens

    def _build_bm25_index(self):
        """Build BM25 index from verses."""
        if not self.verses:
            return

        self.tokenized_corpus = [self._tokenize(v.text) for v in self.verses]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _parse_verse_reference(self, query: str) -> Optional[dict]:
        """
        Parse a verse reference from a query string.

        Returns dict with book_id, chapter, verse_start, verse_end if found.
        Handles formats like:
        - "בראשית 12:2"
        - "בראשית 12:2-4"
        - "בראשית פרק 12 פסוק 2"
        """
        # Build pattern with all book names, sorted by length (longest first)
        # to match multi-word names like "שיר השירים" before shorter ones
        book_names = sorted(HEBREW_BOOK_MAP.keys(), key=len, reverse=True)
        book_pattern = '|'.join(re.escape(name) for name in book_names)

        # Pattern 1: "בראשית 12:2" or "בראשית 12:2-4"
        pattern1 = rf'({book_pattern})\s+(\d+):(\d+)(?:-(\d+))?'
        match = re.search(pattern1, query)
        if match:
            book_name = match.group(1)
            chapter = int(match.group(2))
            verse_start = int(match.group(3))
            verse_end = int(match.group(4)) if match.group(4) else verse_start
            return {
                'book_id': HEBREW_BOOK_MAP[book_name],
                'chapter': chapter,
                'verse_start': verse_start,
                'verse_end': verse_end
            }

        # Pattern 2: "בראשית פרק 12 פסוק 2" or "בראשית פרק 12 פסוקים 2-4"
        pattern2 = rf'({book_pattern})\s+פרק\s+(\d+)\s+פסוק(?:ים)?\s+(\d+)(?:-(\d+))?'
        match = re.search(pattern2, query)
        if match:
            book_name = match.group(1)
            chapter = int(match.group(2))
            verse_start = int(match.group(3))
            verse_end = int(match.group(4)) if match.group(4) else verse_start
            return {
                'book_id': HEBREW_BOOK_MAP[book_name],
                'chapter': chapter,
                'verse_start': verse_start,
                'verse_end': verse_end
            }

        return None

    def _expand_verse_range(self, ref: dict) -> List[str]:
        """
        Expand a verse reference to a list of verse IDs.

        Args:
            ref: Dict with book_id, chapter, verse_start, verse_end

        Returns:
            List of verse IDs that exist in the index
        """
        verse_ids = []
        book_id = ref['book_id']
        chapter = ref['chapter']
        start = ref['verse_start']
        end = ref['verse_end']

        for verse_num in range(start, end + 1):
            verse_id = f"{book_id}-{chapter}-{verse_num}"
            if verse_id in self.verse_index:
                verse_ids.append(verse_id)

        return verse_ids

    def _try_expand_verse_id(self, verse_id: str) -> List[str]:
        """
        Try to expand a verse ID that might be a range.

        Examples:
            "bereshit-17-4" -> ["bereshit-17-4"]
            "bereshit-17-4-6" -> ["bereshit-17-4", "bereshit-17-5", "bereshit-17-6"]

        Returns:
            List of verse IDs, or empty list if cannot parse
        """
        # If it exists as-is, return it
        if verse_id in self.verse_index:
            return [verse_id]

        parts = verse_id.split('-')

        # Handle 4-part format: book-chapter-start_verse-end_verse
        if len(parts) == 4:
            book_id, chapter, start, end = parts
            try:
                start_num, end_num = int(start), int(end)
                expanded = []
                for v in range(start_num, end_num + 1):
                    vid = f"{book_id}-{chapter}-{v}"
                    if vid in self.verse_index:
                        expanded.append(vid)
                return expanded
            except ValueError:
                pass

        # Handle books with underscores that might have 5+ parts
        # e.g., "shmuel_alef-1-1-3" -> book=shmuel_alef, chapter=1, verses 1-3
        # Find the book_id by trying progressively longer prefixes
        for i in range(len(parts) - 3, 0, -1):
            potential_book = '-'.join(parts[:i]).replace('-', '_')
            remaining = parts[i:]
            if len(remaining) >= 3:
                try:
                    chapter = int(remaining[0])
                    start_num = int(remaining[1])
                    end_num = int(remaining[2])
                    expanded = []
                    for v in range(start_num, end_num + 1):
                        vid = f"{potential_book}-{chapter}-{v}"
                        if vid in self.verse_index:
                            expanded.append(vid)
                    if expanded:
                        return expanded
                except ValueError:
                    continue

        return []

    def _verse_to_search_result(self, verse_id: str) -> Optional[SearchResult]:
        """Convert a verse ID to a SearchResult object."""
        verse = self.verse_index.get(verse_id)
        if not verse:
            return None
        return SearchResult(
            verse_id=verse.id,
            book=verse.book,
            book_en=verse.book_en,
            chapter=verse.chapter,
            verse=verse.verse,
            text=verse.text,
            page=verse.page,
            score=1.0,  # Direct reference, highest score
            verse_ids=[verse.id]
        )

    def _handle_direct_verse_reference(
        self,
        query: str,
        book: Optional[str],
        chapter: Optional[int],
        top_k: int
    ) -> Optional[List[SearchResult]]:
        """
        Check if query is a direct verse reference and return results if so.

        Args:
            query: Search query that might contain a verse reference
            book: Optional book filter
            chapter: Optional chapter filter
            top_k: Max results to return

        Returns:
            List of SearchResults if query is a verse reference, None otherwise
        """
        ref = self._parse_verse_reference(query)
        if not ref:
            return None

        # Apply book filter if provided
        if book and ref['book_id'] != book:
            return []
        # Apply chapter filter if provided
        if chapter and ref['chapter'] != chapter:
            return []

        # Directly retrieve the verse(s) instead of searching
        verse_ids = self._expand_verse_range(ref)
        results = []
        for vid in verse_ids[:top_k]:
            result = self._verse_to_search_result(vid)
            if result:
                results.append(result)
        return results

    def _load_embeddings(self):
        """Load precomputed embeddings if FAISS index not already loaded."""
        if self.faiss_index is not None:
            return
        embeddings_path = self.data_dir / "index" / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(str(embeddings_path)).astype(np.float32)
            self.embeddings = l2_normalize(self.embeddings)
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(self.embeddings)

    def _embed_text(self, text: str) -> np.ndarray:
        """Get embedding for text using OpenAI."""
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        response = self.openai_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # Validate embedding dimension matches the FAISS index
        if self.faiss_index is not None:
            index_dim = self.faiss_index.d
            if len(embedding) != index_dim:
                raise ValueError(
                    f"Embedding dimension mismatch!\n"
                    f"  FAISS index expects: {index_dim} dimensions\n"
                    f"  Current model '{embedding_model}' produces: {len(embedding)} dimensions\n\n"
                    f"The index was likely built with a different embedding model.\n"
                    f"To fix this, set EMBEDDING_MODEL in your .env file to match the model used to build the index."
                )
        
        return embedding

    def search_keyword(
        self,
        query: str,
        book: Optional[str] = None,
        chapter: Optional[int] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search using BM25 keyword matching.

        Args:
            query: Search query in Hebrew
            book: Optional book ID to filter by
            chapter: Optional chapter number to filter by
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        # Normalize book ID if provided
        book = normalize_book_id(book)

        # Check if query is a direct verse reference
        direct_results = self._handle_direct_verse_reference(query, book, chapter, top_k)
        if direct_results is not None:
            return direct_results

        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Get indices sorted by score
        indices = np.argsort(scores)[::-1]

        results = []
        for idx in indices:
            if len(results) >= top_k:
                break

            verse = self.verses[idx]
            score = scores[idx]

            if score <= 0:
                continue

            # Apply filters
            if book and verse.book_id != book:
                continue
            if chapter and verse.chapter != chapter:
                continue

            results.append(SearchResult(
                verse_id=verse.id,
                book=verse.book,
                book_en=verse.book_en,
                chapter=verse.chapter,
                verse=verse.verse,
                text=verse.text,
                page=verse.page,
                score=float(score),
                verse_ids=[verse.id]
            ))

        return results

    def search_semantic(
        self,
        query: str,
        book: Optional[str] = None,
        chapter: Optional[int] = None,
        top_k: int = 10,
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search using semantic similarity.

        Args:
            query: Search query
            book: Optional book ID to filter by
            chapter: Optional chapter number to filter by
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        # Normalize book ID if provided
        book = normalize_book_id(book)

        # Check if query is a direct verse reference
        direct_results = self._handle_direct_verse_reference(query, book, chapter, top_k)
        if direct_results is not None:
            return direct_results

        # Get embedding for the query
        query_embedding = self._embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        query_embedding = l2_normalize(query_embedding)

        # Search using chunk-level FAISS index
        candidate_k = min(max(top_k * 5, top_k), len(self.chunk_ids))
        scores, indices = self.faiss_index.search(query_embedding, candidate_k)

        hits: List[tuple[float, Chunk]] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[idx]
            chunk = self.chunks_by_id.get(chunk_id)
            if not chunk:
                continue
            if book and chunk.book_id != book:
                continue
            if chapter and chunk.chapter != chapter:
                continue
            if min_score is not None and score < min_score:
                continue
            hits.append((float(score), chunk))

        hits = hits[:top_k]

        results: List[SearchResult] = []
        for hit in hits:
            results.append(SearchResult(
                verse_id=hit[1].chunk_id,
                book=hit[1].book,
                book_en=hit[1].book_en,
                chapter=hit[1].chapter,
                verse=hit[1].verse_start,
                text=hit[1].text,
                page=hit[1].page_start,
                score=hit[0],
                verse_ids=hit[1].verse_ids
            ))
            if len(results) >= top_k:
                break

        return results

    def search(
        self,
        query: str,
        method: str = "hybrid",
        book: Optional[str] = None,
        chapter: Optional[int] = None,
        top_k: int = 10,
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for verses matching the query.

        Args:
            query: Search query in Hebrew
            method: "keyword", "semantic", or "hybrid"
            book: Optional book ID to filter by
            chapter: Optional chapter number to filter by
            top_k: Number of results to return

        Returns:
            List of SearchResult objects sorted by relevance
        """
        # Normalize book ID if provided
        book = normalize_book_id(book)

        if method == "keyword":
            return self.search_keyword(query, book, chapter, top_k)
        elif method == "semantic":
            return self.search_semantic(query, book, chapter, top_k, min_score=min_score)
        else:  # hybrid
            # Combine keyword and semantic results
            keyword_results = self.search_keyword(query, book, chapter, top_k)
            semantic_results = self.search_semantic(query, book, chapter, top_k, min_score=min_score)
            # Merge results using reciprocal rank fusion
            scores = {}
            k = 60  # RRF constant

            for rank, result in enumerate(keyword_results):
                scores[result.verse_id] = scores.get(result.verse_id, 0) + 1 / (k + rank)

            for rank, result in enumerate(semantic_results):
                scores[result.verse_id] = scores.get(result.verse_id, 0) + 1 / (k + rank)

            # Build result map
            result_map = {r.verse_id: r for r in keyword_results + semantic_results}

            # Sort by combined score
            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

            results = []
            for verse_id in sorted_ids[:top_k]:
                result = result_map[verse_id]
                result.score = scores[verse_id]
                results.append(result)

            return results

    def get_verse(self, verse_id: str) -> Optional[Verse]:
        """Get a specific verse by ID."""
        return self.verse_index.get(verse_id)

    def get_verses(
        self,
        verse_ids: List[str],
        context: int = 0
    ) -> List[Verse]:
        """
        Get specific verses with optional surrounding context.

        Supports both single verse IDs (e.g., "bereshit-1-1") and range IDs
        (e.g., "bereshit-17-4-6" for verses 4-6 of chapter 17).

        Args:
            verse_ids: List of verse IDs to retrieve
            context: Number of verses before/after to include

        Returns:
            List of Verse objects
        """
        result_verses = []
        seen_ids = set()

        # Expand any range-style IDs first
        expanded_ids = []
        for verse_id in verse_ids:
            expanded = self._try_expand_verse_id(verse_id)
            if expanded:
                expanded_ids.extend(expanded)
            # If expansion returned nothing, keep the original ID
            # (it might still be found directly)
            elif verse_id in self.verse_index:
                expanded_ids.append(verse_id)

        for verse_id in expanded_ids:
            if verse_id not in self.verse_index:
                continue

            verse = self.verse_index[verse_id]

            # Find index in verses list
            idx = None
            for i, v in enumerate(self.verses):
                if v.id == verse_id:
                    idx = i
                    break

            if idx is None:
                continue

            # Get context verses
            start = max(0, idx - context)
            end = min(len(self.verses), idx + context + 1)

            for i in range(start, end):
                v = self.verses[i]
                # Only include context from same chapter
                if v.book_id == verse.book_id and v.chapter == verse.chapter:
                    if v.id not in seen_ids:
                        result_verses.append(v)
                        seen_ids.add(v.id)

        # Sort by position
        result_verses.sort(key=lambda v: (v.book_id, v.chapter, v.verse))
        return result_verses

    def get_chapter(self, book_id: str, chapter: int) -> List[Verse]:
        """
        Get all verses in a specific chapter.

        Args:
            book_id: Book identifier
            chapter: Chapter number

        Returns:
            List of Verse objects in order
        """
        # Normalize book ID if provided
        book_id = normalize_book_id(book_id)
        return [v for v in self.verses
                if v.book_id == book_id and v.chapter == chapter]

    def list_books(self) -> List[dict]:
        """
        List all available books.

        Returns:
            List of book info dicts with id, name_he, name_en, chapter_count
        """
        return self.metadata.get("books", [])


if __name__ == "__main__":
    load_dotenv()
    search = BibleSearch()
    
    embeddings_path = Path(__file__).parent.parent / "data" / "bible" / "index" / "embeddings.npy"
    
    # Test search
    print(f"Loaded {len(search.verses)} verses")

    if search.verses:
        results = search.search("בראשית ברא אלהים", top_k=5)
        print("\nSearch results for 'בראשית ברא אלהים':")
        for r in results:
            print(f"{r.verse_id}  {r.book} {r.chapter}:{r.verse} - {r.text[:50]}... (score: {r.score:.3f})")
