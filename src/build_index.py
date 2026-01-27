from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Sequence
from openai import OpenAI
import openai
import random

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from .utils.indexing import FaissVectorIndex
from .utils.loaders import load_verses, make_chunks, save_chunks, save_verses


def _sleep_backoff(attempt: int) -> None:
    # Exponential backoff with jitter.
    base = min(2 ** attempt, 30)
    jitter = random.random()
    time.sleep(base + jitter)


def _is_retryable_error(err: Exception) -> bool:
    if openai is None:
        return False

    retryable = (
        getattr(openai, "RateLimitError", None),
        getattr(openai, "APITimeoutError", None),
        getattr(openai, "InternalServerError", None),
        getattr(openai, "APIConnectionError", None),
        getattr(openai, "APIStatusError", None),
    )
    return isinstance(err, tuple([e for e in retryable if e is not None]))


def embed_texts(
    client: "OpenAI",
    texts: Sequence[str],
    model: str,
    *,
    dimensions: Optional[int] = None,
    max_retries: int = 6,
) -> np.ndarray:
    """
    Embed a batch of texts. Returns float32 matrix shape (n, d).
    Uses OpenAI embeddings endpoint. See docs for token limits. 
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {"model": model, "input": list(texts)}
            if dimensions is not None:
                kwargs["dimensions"] = dimensions
            resp = client.embeddings.create(**kwargs)  # type: ignore[attr-defined]
            # `resp.data` order matches input order.
            embs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
            return np.vstack(embs)
        except Exception as e:
            if attempt >= max_retries - 1 or not _is_retryable_error(e):
                raise
            _sleep_backoff(attempt)
    raise RuntimeError("unreachable")


def build_embeddings(
    client,
    texts: list[str],
    model: str,
    *,
    batch_size: int = 64,
    dimensions: Optional[int] = None,
    max_retries: int = 6,
) -> np.ndarray:
    embs: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i : i + batch_size]
        batch_embs = embed_texts(
            client,
            batch,
            model=model,
            dimensions=dimensions,
            max_retries=max_retries,
        )
        embs.append(batch_embs)
    if not embs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(embs).astype(np.float32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a local FAISS index for Bible Q&A.")
    ap.add_argument("--verses", required=True, help="Path to verses.json (flat list).")
    ap.add_argument("--out_dir", required=True, help="Output directory for index artifacts.")
    ap.add_argument("--embedding_model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))
    ap.add_argument("--verses_per_chunk", type=int, default=6)
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()


    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    verses = load_verses(args.verses)
    chunks = make_chunks(verses, verses_per_chunk=args.verses_per_chunk, overlap=args.overlap)
    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    embeddings = build_embeddings(
        client,
        texts,
        model=args.embedding_model,
        batch_size=args.batch_size,
    )


    index = FaissVectorIndex.build(embeddings, chunk_ids=chunk_ids)

    index_path = out_dir / "index.faiss"
    index_meta_path = out_dir / "index_meta.json"
    chunks_path = out_dir / "chunks.json"
    verses_path = out_dir / "verses.json"

    index.save(index_path=index_path, meta_path=index_meta_path)
    save_chunks(chunks, chunks_path)
    save_verses(verses, verses_path)


    print("Built index artifacts successfully:")
    print(f"- {index_path}")
    print(f"- {index_meta_path}")
    print(f"- {chunks_path}")
    print(f"- {verses_path}")


if __name__ == "__main__":
    main()
