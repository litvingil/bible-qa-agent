import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import faiss

from .types import Chunk, SearchHit


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


@dataclass
class FaissIndexBundle:
    index: "faiss.Index"
    dim: int
    chunk_ids: List[str]


class FaissVectorIndex:
    """
    Simple cosine-similarity FAISS index.

    We store unit-normalized vectors and use inner product (IP),
    which is equivalent to cosine similarity after normalization.
    """

    def __init__(self, bundle: FaissIndexBundle):
        
        self.bundle = bundle

    @property
    def dim(self) -> int:
        return self.bundle.dim

    @property
    def chunk_ids(self) -> List[str]:
        return self.bundle.chunk_ids

    @classmethod
    def build(cls, embeddings: np.ndarray, chunk_ids: Sequence[str]) -> "FaissVectorIndex":
        
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if embeddings.shape[0] != len(chunk_ids):
            raise ValueError("embeddings rows must match len(chunk_ids)")

        embs = l2_normalize(embeddings.astype(np.float32, copy=False))
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        return cls(FaissIndexBundle(index=index, dim=dim, chunk_ids=list(chunk_ids)))

    def search(self, query_embedding: np.ndarray, chunks_by_id: Dict[str, Chunk], top_k: int = 6) -> List[SearchHit]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        q = l2_normalize(query_embedding.astype(np.float32, copy=False))
        scores, idxs = self.bundle.index.search(q, top_k)
        hits: List[SearchHit] = []
        for rank, (score, i) in enumerate(zip(scores[0].tolist(), idxs[0].tolist()), start=1):
            if i < 0 or i >= len(self.bundle.chunk_ids):
                continue
            cid = self.bundle.chunk_ids[i]
            chunk = chunks_by_id[cid]
            hits.append(SearchHit(rank=rank, score=float(score), chunk=chunk))
        return hits

    def save(self, index_path: str | Path, meta_path: str | Path) -> None:
        
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        faiss.write_index(self.bundle.index, str(index_path))
        meta = {"dim": self.bundle.dim, "chunk_ids": self.bundle.chunk_ids}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: str | Path, meta_path: str | Path) -> "FaissVectorIndex":
        
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bundle = FaissIndexBundle(index=index, dim=int(meta["dim"]), chunk_ids=list(meta["chunk_ids"]))
        return cls(bundle)
