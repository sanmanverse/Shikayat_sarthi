"""
Case Memory & RAG Retrieval Engine (Module 03).
Hybrid search over the 800-case Official Mock Grievance History:
- Dense multilingual embeddings (multilingual-e5-small, CPU) with REAL cosine similarity, fused with
  BM25 by reciprocal rank fusion. Falls back to BM25-only (flagged retrieval_mode="bm25") when the
  embedding model is unavailable.
- BM25 token-based ranking for keyword precision
- Multilingual normalization bridge (translates Marathi & Hindi to English domain keywords before retrieval)
- Calibrated similarity scoring reflecting genuine lexical overlap
- Returns top-k precedents with resolution summaries, citizen feedback, and sensitivity signals
- Detects sensitive neighbors to feed the Policy Gate's conservative escalation rule.
"""
import re
import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session
from app.db.models import HistoryGrievance, Category
from app.services.normalizer import normalize_to_english
from app.core.config import settings

logger = logging.getLogger(__name__)

def normalize_token(t: str) -> str:
    """Lightweight suffix normalization for English tokens."""
    if len(t) > 4:
        if t.endswith("ing"):
            return t[:-3]
        if t.endswith("ies"):
            return t[:-3] + "y"
        if t.endswith("es"):
            return t[:-2]
        if t.endswith("s") and not t.endswith("ss"):
            return t[:-1]
    return t

def tokenize(text: str) -> List[str]:
    """Unicode-aware tokenization supporting English, Hindi, and Marathi with light stemming."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [normalize_token(tok.strip()) for tok in cleaned.split() if len(tok.strip()) > 1]
    return tokens


class CaseRetrievalEngine:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_records: List[Dict[str, Any]] = []
        self._is_indexed = False
        self._embedding_unavailable_logged = False

    @staticmethod
    def _text_key(text: str) -> str:
        return (text or "").strip().lower()

    def _embedding_scores(self, texts: List[str]) -> Optional[Dict[str, float]]:
        """Cosine similarity of the query (max over `texts`) to every deduplicated history text."""
        try:
            from app.nlp.embedding_retrieval import get_embedding_engine
            engine = get_embedding_engine()
        except Exception as exc:  # pragma: no cover - import problems must not break retrieval
            engine = None
            logger.warning("Embedding engine import failed: %s", exc)
        if engine is None or not engine.is_ready:
            if not self._embedding_unavailable_logged:
                logger.warning("Embedding retrieval unavailable; using BM25 only (retrieval_mode=bm25).")
                self._embedding_unavailable_logged = True
            return None
        return engine.score_all([t for t in texts if t and t.strip()])

    def build_index(self, db: Session):
        """Indexes the 800 history records into BM25 memory corpus."""
        records = db.query(HistoryGrievance).all()
        self.corpus_records = []
        tokenized_corpus = []

        for r in records:
            item = {
                "grievance_id": r.grievance_id,
                "complaint_text": r.complaint_text,
                "category_id": r.category_id,
                "category": r.category,
                "subcategory": r.subcategory,
                "department_id": r.department_id,
                "taluka": r.broad_location_taluka,
                "locality": r.broad_location_locality,
                "status": r.status,
                "resolution_text": r.resolution_text or "No resolution text",
                "citizen_feedback": r.citizen_feedback or "No feedback recorded",
                "sla_days": r.sla_days,
                "sensitive": r.sensitive,
                "identity_hiding": r.identity_hiding,
                "ai_auto_closed": r.ai_auto_closed
            }
            # Search over combination of category, subcategory, locality, and complaint text
            search_str = f"{r.category} {r.subcategory} {r.broad_location_locality} {r.complaint_text}"
            tokenized_corpus.append(tokenize(search_str))
            self.corpus_records.append(item)

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._is_indexed = True
            logger.info(f"Indexed {len(self.corpus_records)} official history cases into Case Memory BM25.")

    def search(
        self,
        query: str,
        db: Session,
        top_k: int = 5,
        taluka_filter: Optional[str] = None,
        raw_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves top-k similar historical cases (hybrid: BM25 + embeddings via RRF).
        `query` is normalised to English for BM25; `raw_query` (original-language text) is embedded
        directly by the multilingual model, so vernacular wording the dictionary does not know
        can still match.
        """
        empty = {
            "neighbors": [], "sensitive_neighbor_count": 0, "retrieval_quality": "WEAK",
            "top_category_vote": None, "retrieval_mode": "none", "sensitive_best": None,
        }
        if not self._is_indexed or not self.corpus_records:
            self.build_index(db)
        if not self.bm25 or not self.corpus_records:
            return empty

        normalized_query = normalize_to_english(query)
        query_tokens = tokenize(normalized_query)

        bm25_scores = self.bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(self.corpus_records)

        emb_map = self._embedding_scores([raw_query or query, normalized_query])
        hybrid = emb_map is not None

        n = len(self.corpus_records)
        taluka_l = (taluka_filter or "").lower()

        # --- embedding similarity per record (soft taluka boost; no hard filter) -----------------
        emb_sim: List[float] = [0.0] * n
        if hybrid:
            for i, rec in enumerate(self.corpus_records):
                v = emb_map.get(self._text_key(rec["complaint_text"]))
                if v is None:
                    continue
                if taluka_l and (rec["taluka"] or "").lower() == taluka_l:
                    v = min(v + 0.05, 1.0)
                emb_sim[i] = float(v)

        if not query_tokens and not hybrid:
            return {**empty, "normalized_query": normalized_query}

        # --- ranking ----------------------------------------------------------------------------
        if hybrid:
            k = settings.RRF_K
            fused = [0.0] * n
            bm25_order = [i for i in sorted(range(n), key=lambda i: bm25_scores[i], reverse=True) if bm25_scores[i] > 0][:50]
            for rank, i in enumerate(bm25_order):
                fused[i] += 1.0 / (k + rank + 1)
            emb_order = sorted(range(n), key=lambda i: emb_sim[i], reverse=True)[:50]
            for rank, i in enumerate(emb_order):
                fused[i] += 1.0 / (k + rank + 1)
            order = sorted(range(n), key=lambda i: (fused[i], emb_sim[i]), reverse=True)[:max(top_k, 10)]
        else:
            order = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)[:max(top_k, 10)]

        def _bm25_calibrated(score: float) -> float:
            if score <= 0.0:
                return 0.0
            if score >= 10.0:
                return round(min(0.85 + (score - 10.0) * 0.01, 0.99), 3)
            if score >= 5.0:
                return round(0.70 + (score - 5.0) * 0.03, 3)
            if score >= 2.0:
                return round(0.45 + (score - 2.0) * 0.08, 3)
            return round(max(score * 0.20, 0.10), 3)

        candidates = []
        for i in order:
            rec = self.corpus_records[i]
            raw = float(bm25_scores[i])
            if hybrid:
                sim = round(emb_sim[i], 4)
                source = "embedding_cosine"
            else:
                sim = _bm25_calibrated(raw)
                source = "bm25_calibrated"
            candidates.append({
                **rec,
                "similarity": sim,
                "similarity_source": source,
                "raw_score": round(raw, 2),
            })

        results = candidates[:top_k]
        top_raw_score = max((float(x) for x in bm25_scores), default=0.0)

        sensitive_count = sum(1 for r in results if r["sensitive"] or r["identity_hiding"])

        # Best sensitive precedent among the wider top-10 (used by the proximity safety net)
        sensitive_best = None
        if candidates:
            top_sim = candidates[0]["similarity"] or 0.0
            for r in candidates[:10]:
                if (r["sensitive"] or r["identity_hiding"]) and top_sim > 0:
                    ratio = round((r["similarity"] or 0.0) / top_sim, 3)
                    if sensitive_best is None or ratio > sensitive_best["ratio"]:
                        sensitive_best = {
                            "category_id": r["category_id"],
                            "grievance_id": r["grievance_id"],
                            "similarity": r["similarity"],
                            "ratio": ratio,
                            "gap": round(top_sim - (r["similarity"] or 0.0), 4),
                            "similarity_source": r.get("similarity_source"),
                        }

        # --- quality ----------------------------------------------------------------------------
        if not results:
            quality = "WEAK"
        elif hybrid:
            top = results[0]["similarity"]
            if top >= settings.EMBED_STRONG_SIM:
                quality = "STRONG"
            elif top >= settings.EMBED_WEAK_SIM:
                quality = "MEDIUM"
            else:
                quality = "WEAK"
        elif top_raw_score < 2.5 or results[0]["similarity"] < 0.45:
            quality = "WEAK"
        elif results[0]["similarity"] >= 0.70 and top_raw_score >= 5.0:
            quality = "STRONG"
        else:
            quality = "MEDIUM"

        # --- category consensus vote ------------------------------------------------------------
        cat_counts: Dict[str, int] = {}
        for r in results:
            qualifies = (r["similarity"] >= settings.EMBED_WEAK_SIM) if hybrid else (r["raw_score"] > 1.0)
            if qualifies:
                cat_counts[r["category_id"]] = cat_counts.get(r["category_id"], 0) + 1
        top_vote = (
            max(cat_counts.items(), key=lambda x: x[1])[0] if cat_counts
            else (results[0]["category_id"] if results else None)
        )

        return {
            "neighbors": results,
            "sensitive_neighbor_count": sensitive_count,
            "retrieval_quality": quality,
            "top_category_vote": top_vote,
            "normalized_query": normalized_query,
            "retrieval_mode": "hybrid" if hybrid else "bm25",
            "sensitive_best": sensitive_best,
        }


retrieval_engine = CaseRetrievalEngine()
