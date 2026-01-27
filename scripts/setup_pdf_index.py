#!/usr/bin/env python3
"""
Setup script to extract verses from a PDF and build the FAISS index.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.utils.extract_pdf import extract_verses
from src import build_index


def build_index_from_verses(
    verses_path: Path,
    out_dir: Path,
    *,
    embedding_model: str,
    verses_per_chunk: int,
    overlap: int,
    batch_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    client = build_index.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    verses = build_index.load_verses(str(verses_path))
    chunks = build_index.make_chunks(
        verses, verses_per_chunk=verses_per_chunk, overlap=overlap
    )
    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    embeddings = build_index.build_embeddings(
        client,
        texts,
        model=embedding_model,
        batch_size=batch_size,
    )

    index = build_index.FaissVectorIndex.build(embeddings, chunk_ids=chunk_ids)

    index_path = out_dir / "index.faiss"
    index_meta_path = out_dir / "index_meta.json"
    chunks_path = out_dir / "chunks.json"
    verses_out_path = out_dir / "verses.json"

    index.save(index_path=index_path, meta_path=index_meta_path)
    build_index.save_chunks(chunks, chunks_path)
    build_index.save_verses(verses, verses_out_path)

    print("Built index artifacts successfully:")
    print(f"- {index_path}")
    print(f"- {index_meta_path}")
    print(f"- {chunks_path}")
    print(f"- {verses_out_path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Extract verses from a PDF, then build the FAISS index."
    )
    parser.add_argument("pdf_path", help="Path to input PDF file.")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT_DIR / "data" / "bible"),
        help="Output directory for extracted data (default: data/bible).",
    )
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))
    parser.add_argument("--verses-per-chunk", type=int, default=6)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).expanduser()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_dir = Path(args.out_dir).expanduser()
    verses_output_dir = out_dir
    index_output_dir = out_dir / "index"

    print("Extracting verses...")
    extract_verses(pdf_path, verses_output_dir)

    verses_path = index_output_dir / "verses.json"
    if not verses_path.exists():
        raise SystemExit(f"Expected verses file not found: {verses_path}")

    print("Building index...")
    build_index_from_verses(
        verses_path,
        index_output_dir,
        embedding_model=args.embedding_model,
        verses_per_chunk=args.verses_per_chunk,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
