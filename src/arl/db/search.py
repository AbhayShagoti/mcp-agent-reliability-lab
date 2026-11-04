"""BM25 plus hashing-trick vectors. Postgres also stores the vector in pgvector."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

TOKEN = re.compile(r"[a-z0-9$]+")
STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "and",
        "or",
        "for",
        "on",
        "in",
        "is",
        "are",
        "be",
        "only",
        "it",
        "this",
        "that",
        "with",
        "by",
        "before",
        "after",
        "may",
        "must",
        "do",
        "not",
        "other",
        "own",
        "when",
        "what",
        "how",
        "can",
        "i",
        "my",
        "me",
        "you",
        "your",
    }
)
VECTOR_DIM = 64


def tokenize(text: str) -> list[str]:
    return [tok for tok in TOKEN.findall(text.lower()) if tok not in STOPWORDS]


def hash_embed(text: str, dim: int = VECTOR_DIM) -> list[float]:
    """Signed hashing trick. Deterministic and free — not a neural encoder."""
    vec = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.sha256(tok.encode()).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


@dataclass(frozen=True)
class IndexedDoc:
    doc_id: str
    text: str


def bm25_scores(
    query: str,
    docs: list[IndexedDoc],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    """Okapi BM25 over a tiny in-memory corpus."""
    if not docs:
        return {}
    tokenized = [tokenize(doc.text) for doc in docs]
    n_docs = len(docs)
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    avgdl = sum(len(tokens) for tokens in tokenized) / n_docs
    if avgdl == 0:
        avgdl = 1.0
    query_tokens = tokenize(query)
    scores: dict[str, float] = {}
    for doc, tokens in zip(docs, tokenized, strict=True):
        tf = Counter(tokens)
        score = 0.0
        dl = len(tokens) or 1
        for term in query_tokens:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (freq * (k1 + 1.0)) / (freq + k1 * (1.0 - b + b * dl / avgdl))
        scores[doc.doc_id] = score
    return scores


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi == lo:
        return {key: 1.0 if value > 0 else 0.0 for key, value in scores.items()}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def hybrid_scores(
    query: str,
    docs: list[IndexedDoc],
    embeddings: dict[str, list[float]],
    *,
    bm25_weight: float = 0.85,
    vector_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    """Blend lexical BM25 with cosine similarity.

    `vector_scores`, when supplied, come from pgvector. Otherwise cosine runs in process
    against `embeddings`.
    """
    lexical = _minmax(bm25_scores(query, docs))
    if vector_scores is None:
        query_vec = hash_embed(query)
        vector_scores = {doc.doc_id: cosine(query_vec, embeddings[doc.doc_id]) for doc in docs}
    dense = _minmax(vector_scores)
    keys = set(lexical) | set(dense)
    weight = bm25_weight
    return {key: weight * lexical.get(key, 0.0) + (1.0 - weight) * dense.get(key, 0.0) for key in keys}


def rank(scores: dict[str, float], k: int) -> list[tuple[str, float]]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:k]
