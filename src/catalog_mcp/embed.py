"""Text embeddings (sentence-transformers) and the FAISS vector index."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol, Sequence

import faiss
import numpy as np

from .config import DEFAULT_MODEL, QUERY_INSTRUCTION


class Embedder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def encode_docs(self, texts: Sequence[str], batch_size: int = 256, progress: bool = False) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. The model loads lazily on first use.

    Weights come from the Hugging Face cache (HF_HOME), so nothing is hard-coded here.
    """

    def __init__(self, name: str = DEFAULT_MODEL, device: str | None = None, query_instruction: str | None = None):
        self.name = name
        self.device = device
        self.query_instruction = QUERY_INSTRUCTION if query_instruction is None and "bge" in name.lower() else (query_instruction or "")
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    try:  # keep stderr quiet: no weight-loading progress bar on every start
                        from transformers.utils import logging as hf_logging

                        hf_logging.disable_progress_bar()
                    except ImportError:
                        pass

                    try:  # use the cached copy without a network round-trip when we have it
                        model = SentenceTransformer(self.name, device=self.device, local_files_only=True)
                    except Exception:  # noqa: BLE001 - not cached yet: download it
                        model = SentenceTransformer(self.name, device=self.device)
                    model.max_seq_length = 256
                    if model.device.type == "cuda":
                        model.half()
                    self._model = model
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_embedding_dimension())

    def encode_docs(self, texts: Sequence[str], batch_size: int = 256, progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=progress and len(texts) > batch_size,
        )
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vec = self.model.encode([self.query_instruction + text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.ascontiguousarray(vec, dtype=np.float32)


def build_index(vectors: np.ndarray) -> faiss.Index:
    """Exact inner-product index. Vectors are L2-normalised, so inner product = cosine."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))
