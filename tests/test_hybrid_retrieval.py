"""Hybrid retrieval wiring (BM25 + embeddings via RRF). The embedding model is faked: offline, deterministic."""
import re
from app.nlp import embedding_retrieval as emb
from app.nlp.retrieval import retrieval_engine


class FakeEngine:
    """Cosine-like scores in [0.70, 0.95] from token overlap; same interface as the real engine."""
    is_ready = True

    def __init__(self, texts):
        self.texts = texts

    @staticmethod
    def _tok(s):
        return set(re.findall(r"[a-z]+", s.lower()))

    def score_all(self, queries):
        qs = [self._tok(q) for q in queries]
        out = {}
        for t in self.texts:
            tt = self._tok(t)
            best = max((len(q & tt) / max(len(q | tt), 1)) for q in qs)
            out[t.strip().lower()] = 0.70 + 0.25 * best
        return out


def _engine_from_index(db):
    retrieval_engine.build_index(db)
    return FakeEngine([r["complaint_text"] for r in retrieval_engine.corpus_records])


def test_hybrid_mode_reports_real_cosine_and_source(db, monkeypatch):
    eng = _engine_from_index(db)
    monkeypatch.setattr(emb, "get_embedding_engine", lambda: eng)
    res = retrieval_engine.search("Pothole on the main road near the bus stand", db, top_k=5,
                                  raw_query="Pothole on the main road near the bus stand")
    assert res["retrieval_mode"] == "hybrid" and len(res["neighbors"]) == 5
    assert all(n["similarity_source"] == "embedding_cosine" for n in res["neighbors"])
    assert all(0.70 <= n["similarity"] <= 1.0 for n in res["neighbors"])
    assert res["retrieval_quality"] in ("STRONG", "MEDIUM", "WEAK")


def test_falls_back_to_bm25_and_says_so(db, monkeypatch):
    retrieval_engine.build_index(db)
    monkeypatch.setattr(emb, "get_embedding_engine", lambda: None)
    res = retrieval_engine.search("Pothole on the main road", db, top_k=5)
    assert res["retrieval_mode"] == "bm25"
    assert all(n["similarity_source"] == "bm25_calibrated" for n in res["neighbors"])


def test_taluka_is_a_soft_boost_not_a_filter(db, monkeypatch):
    eng = _engine_from_index(db)
    monkeypatch.setattr(emb, "get_embedding_engine", lambda: eng)
    plain = retrieval_engine.search("Garbage not collected", db, top_k=5)
    boosted = retrieval_engine.search("Garbage not collected", db, top_k=5, taluka_filter="Nanded")
    assert len(boosted["neighbors"]) == 5                      # nothing filtered out
    if any(n["taluka"] == "Nanded" for n in boosted["neighbors"]):
        assert max(n["similarity"] for n in boosted["neighbors"]) >= max(n["similarity"] for n in plain["neighbors"]) - 1e-9


def test_sensitive_best_exposes_gap_and_source(db, monkeypatch):
    eng = _engine_from_index(db)
    monkeypatch.setattr(emb, "get_embedding_engine", lambda: eng)
    res = retrieval_engine.search("Exposed live wire hanging near school gate electric shock", db, top_k=5)
    sb = res["sensitive_best"]
    assert sb is None or {"category_id", "ratio", "gap", "similarity_source"} <= set(sb)
