"""
Embedding-based RAG retrieval using multilingual-e5-small (CPU-only).
Group 5:
- Loads intfloat/multilingual-e5-small via sentence-transformers at startup
- Builds index from 800-row history CSV; rebuilds only if CSV hash changes
- BM25 as optional hybrid re-ranker (reciprocal rank fusion)
- Taluka soft boost (+0.05 cosine) for same taluka
- Real cosine similarity (not BM25-mapped scores)
- Deduplicates near-identical rows before indexing
- Returns grievance_id, similarity, category, resolution_text, etc.
"""
import hashlib
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

_INDEX_DIR = settings.DATA_DIR / ".embedding_cache"
_INDEX_FILE = _INDEX_DIR / "faiss_index.pkl"
_HASH_FILE = _INDEX_DIR / "csv_hash.txt"
_CSV_PATH = settings.DATA_DIR / "mock_grievance_history.csv"


def _csv_hash() -> str:
    """SHA256 of (model name + history CSV) so a model change also invalidates the cache."""
    h = hashlib.sha256()
    h.update(settings.EMBEDDING_MODEL.encode("utf-8"))
    try:
        with open(_CSV_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except FileNotFoundError:
        return ""
    return h.hexdigest()


class EmbeddingRetrievalEngine:
    def __init__(self):
        self.model = None
        self.embeddings: Optional[np.ndarray] = None
        self.corpus_records: List[Dict[str, Any]] = []
        self.is_ready = False
        self._load_or_build()

    def _load_or_build(self):
        """Load cached index or build from CSV."""
        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        current_hash = _csv_hash()
        cached_hash = ""
        if _HASH_FILE.exists():
            cached_hash = _HASH_FILE.read_text().strip()

        if _INDEX_FILE.exists() and current_hash and current_hash == cached_hash:
            try:
                with open(_INDEX_FILE, "rb") as f:
                    cached = pickle.load(f)
                self.embeddings = cached["embeddings"]
                self.corpus_records = cached["records"]
                self._load_model()
                self.is_ready = True
                logger.info(
                    "Loaded embedding index from cache (%d records).", len(self.corpus_records)
                )
                return
            except Exception as exc:
                logger.warning("Cache load failed (%s), rebuilding.", exc)

        self._build_index(current_hash)

    def _load_model(self):
        """
        Load the sentence-transformers model.
        * EMBEDDING_ALLOW_DOWNLOAD=true (default): download on first use if not cached.
        * EMBEDDING_ALLOW_DOWNLOAD=false: only use a local copy; otherwise retrieval falls back to BM25.
        """
        if self.model is not None:
            return
        import os
        from pathlib import Path

        model_path = Path(settings.EMBEDDING_MODEL)
        hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
        model_slug = "models--" + settings.EMBEDDING_MODEL.replace("/", "--")
        is_local = model_path.is_dir() or (hf_home / model_slug).is_dir()

        if not is_local and not settings.EMBEDDING_ALLOW_DOWNLOAD:
            logger.info(
                "Embedding model '%s' is not cached and downloads are disabled "
                "(EMBEDDING_ALLOW_DOWNLOAD=false). Retrieval will use BM25 only.",
                settings.EMBEDDING_MODEL,
            )
            return
        try:
            if is_local:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
            logger.info("Embedding model ready: %s (%s)", settings.EMBEDDING_MODEL,
                        "local cache" if is_local else "downloaded")
        except Exception as exc:
            logger.warning("Could not load embedding model %s: %s. Retrieval will use BM25 only.",
                           settings.EMBEDDING_MODEL, exc)
            self.model = None

    def _build_index(self, current_hash: str):
        """Build embedding index from CSV."""
        import csv

        self._load_model()
        if self.model is None:
            logger.warning("No embedding model — embedding retrieval disabled.")
            return

        records = []
        try:
            with open(_CSV_PATH, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
        except FileNotFoundError:
            logger.error("History CSV not found at %s", _CSV_PATH)
            return

        if not records:
            logger.warning("History CSV is empty.")
            return

        # Deduplicate near-identical complaint texts (keep one per unique text)
        seen_texts: Dict[str, int] = {}
        deduped_records = []
        for rec in records:
            text = rec.get("complaint_text", "").strip().lower()
            if text not in seen_texts:
                seen_texts[text] = len(deduped_records)
                deduped_records.append(rec)
        logger.info(
            "Deduplication: %d records → %d unique.", len(records), len(deduped_records)
        )

        # Build text corpus for embedding (category + subcategory + complaint)
        corpus_texts = []
        self.corpus_records = []
        for rec in deduped_records:
            cat = rec.get("category", "")
            subcat = rec.get("subcategory", "")
            text = rec.get("complaint_text", "")
            # e5-small prefers "query: " and "passage: " prefixes
            search_str = f"passage: {cat} {subcat} {text}"
            corpus_texts.append(search_str)
            self.corpus_records.append({
                "grievance_id": rec.get("grievance_id", ""),
                "complaint_text": text,
                "category_id": rec.get("category_id", ""),
                "category": cat,
                "subcategory": subcat,
                "department_id": rec.get("department_id", ""),
                "taluka": rec.get("broad_location_taluka", ""),
                "locality": rec.get("broad_location_locality", ""),
                "status": rec.get("status", ""),
                "resolution_text": rec.get("resolution_text") or "No resolution text",
                "citizen_feedback": rec.get("citizen_feedback") or "No feedback recorded",
                "sla_days": int(rec.get("sla_days", 7)),
                "sensitive": rec.get("sensitive", "").lower() in ("true", "1", "yes"),
                "identity_hiding": rec.get("identity_hiding", "").lower() in ("true", "1", "yes"),
                "ai_auto_closed": rec.get("ai_auto_closed", "").lower() in ("true", "1", "yes"),
            })

        logger.info("Building embeddings for %d records (CPU)…", len(corpus_texts))
        try:
            self.embeddings = self.model.encode(
                corpus_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )
            # Normalise for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self.embeddings = self.embeddings / norms

            # Cache
            with open(_INDEX_FILE, "wb") as f:
                pickle.dump({"embeddings": self.embeddings, "records": self.corpus_records}, f)
            _HASH_FILE.write_text(current_hash)
            self.is_ready = True
            logger.info("Embedding index built and cached (%d vectors).", len(self.embeddings))
        except Exception as exc:
            logger.error("Embedding index build failed: %s", exc)

    def score_all(self, queries: List[str]) -> Optional[Dict[str, float]]:
        """
        Cosine similarity of the query to every indexed (deduplicated) history text.
        Multiple query texts (original language + English normalisation) -> max per record.
        Returns {lowercased complaint_text: cosine}, or None when embeddings are unavailable.
        """
        if not self.is_ready or self.model is None or self.embeddings is None or not queries:
            return None
        try:
            q = self.model.encode([f"query: {t}" for t in queries], convert_to_numpy=True)
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            q = q / np.where(norms == 0, 1, norms)
            sims = (self.embeddings @ q.T).max(axis=1)
            return {
                rec["complaint_text"].strip().lower(): float(sims[i])
                for i, rec in enumerate(self.corpus_records)
            }
        except Exception as exc:
            logger.error("Embedding scoring failed: %s", exc)
            return None

    def search(
        self,
        query: str,
        top_k: int = 5,
        taluka_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns top_k results with cosine similarity.
        Taluka match gets +0.05 boost.
        """
        if not self.is_ready or self.model is None or self.embeddings is None:
            return []

        try:
            q_emb = self.model.encode(
                [f"query: {query}"], convert_to_numpy=True
            )
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 0:
                q_emb = q_emb / q_norm

            sims = (self.embeddings @ q_emb.T).flatten()

            # Taluka boost
            if taluka_filter:
                for i, rec in enumerate(self.corpus_records):
                    if rec["taluka"].lower() == taluka_filter.lower():
                        sims[i] = min(sims[i] + 0.05, 1.0)

            top_indices = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in top_indices:
                rec = self.corpus_records[int(idx)]
                results.append({**rec, "similarity": round(float(sims[idx]), 4)})
            return results
        except Exception as exc:
            logger.error("Embedding search failed: %s", exc)
            return []


_ENGINE: Optional["EmbeddingRetrievalEngine"] = None
_ENGINE_INIT_DONE = False


def get_embedding_engine() -> Optional["EmbeddingRetrievalEngine"]:
    """Lazy singleton. Returns None (not an exception) if the engine cannot be created."""
    global _ENGINE, _ENGINE_INIT_DONE
    if not _ENGINE_INIT_DONE:
        _ENGINE_INIT_DONE = True
        try:
            _ENGINE = EmbeddingRetrievalEngine()
        except Exception as exc:
            logger.warning("Could not initialise EmbeddingRetrievalEngine: %s", exc)
            _ENGINE = None
    return _ENGINE


def __getattr__(name: str):
    # Backward compatibility: `from app.nlp.embedding_retrieval import embedding_engine`
    if name == "embedding_engine":
        return get_embedding_engine()
    raise AttributeError(name)
