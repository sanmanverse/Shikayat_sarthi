"""
Calibrate the embedding-retrieval thresholds on the 800-row history (leave-one-out).

Run once on a machine that has the model (internet or a local copy):
    python scripts/calibrate_retrieval.py
It prints, per cosine bin, how often the top-1 neighbour (excluding the row itself and exact-duplicate
texts) has the same category, then recommends EMBED_WEAK_SIM / EMBED_STRONG_SIM for your .env.
The defaults in config.py (0.78 / 0.84) are UNCALIBRATED guesses until you do this.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402


def main():
    from sentence_transformers import SentenceTransformer

    rows = list(csv.DictReader(open(settings.DATA_DIR / "mock_grievance_history.csv", encoding="utf-8-sig")))
    texts = [f"passage: {r['category']} {r['subcategory']} {r['complaint_text']}" for r in rows]
    queries = [f"query: {r['complaint_text']}" for r in rows]
    model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
    P = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    Q = model.encode(queries, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    sims = Q @ P.T

    keys = [r["complaint_text"].strip().lower() for r in rows]
    top_sim, correct = [], []
    for i, r in enumerate(rows):
        s = sims[i].copy()
        for j, k in enumerate(keys):
            if j == i or k == keys[i]:
                s[j] = -1
        j = int(np.argmax(s))
        top_sim.append(float(s[j]))
        correct.append(rows[j]["category_id"] == r["category_id"])
    top_sim, correct = np.array(top_sim), np.array(correct)

    print(f"\nLeave-one-out top-1 category accuracy: {correct.mean():.2%} over {len(rows)} rows\n")
    print("cosine >=   n     precision")
    best_weak = best_strong = None
    for th in np.arange(0.70, 0.96, 0.02):
        m = top_sim >= th
        if m.sum() < 15:
            continue
        p = correct[m].mean()
        print(f"  {th:.2f}   {int(m.sum()):4d}   {p:.2%}")
        if best_weak is None and p >= 0.80:
            best_weak = th
        if best_strong is None and p >= 0.92:
            best_strong = th
    print("\nSuggested .env values (precision targets: weak>=80%, strong>=92%):")
    print(f"EMBED_WEAK_SIM={best_weak if best_weak is not None else 'n/a (raise target data)'}")
    print(f"EMBED_STRONG_SIM={best_strong if best_strong is not None else 'n/a'}")


if __name__ == "__main__":
    main()
