"""32k via retrieve-then-decide for Laya.

Keeps single-pass 32ms behavior: chunk long state, TF-IDF rank against
questions, run Router.predict on top-k chunks only.
"""
import re
from typing import Any, Dict, List, Tuple

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def chunk_text(text: str, chunk_words: int = 100, overlap: int = 20) -> List[str]:
    toks = text.split()
    if len(toks) <= chunk_words:
        return [text]
    out, i, step = [], 0, max(1, chunk_words - overlap)
    while i < len(toks):
        out.append(" ".join(toks[i:i + chunk_words]))
        if i + chunk_words >= len(toks):
            break
        i += step
    return out


def _query_text(questions: Dict[str, Any]) -> str:
    parts = []
    for q in questions.values():
        if not isinstance(q, dict):
            continue
        if q.get("instructions"):
            parts.append(str(q["instructions"]))
        c = q.get("criteria")
        if isinstance(c, dict):
            parts.extend(f"{k} {v}" for k, v in c.items())
        elif isinstance(c, list):
            parts.extend(map(str, c))
    return " ".join(parts)


def rank_chunks(chunks: List[str], query: str) -> List[Tuple[int, float]]:
    """TF-IDF cosine rank. Returns [(idx, score)] descending."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not chunks:
        return []
    if not query.strip():
        return [(i, 0.0) for i in range(len(chunks))]
    vec = TfidfVectorizer()
    try:
        m = vec.fit_transform([query] + chunks)
    except ValueError:  # empty vocab
        return [(i, 0.0) for i in range(len(chunks))]
    sims = cosine_similarity(m[0:1], m[1:]).ravel()
    order = sorted(range(len(chunks)), key=lambda i: (-float(sims[i]), i))
    return [(i, float(sims[i])) for i in order]


def select_state(state: Any, questions: Dict[str, Any], chunk_words: int = 100,
                 overlap: int = 20, top_k: int = 3, state_budget_chars: int = 1280) -> Dict[str, Any]:
    """Reduce arbitrary state to top-k chunks fitting Laya's ~512 budget.
    AUDIT FIX (L1): true state room after head (~192) + question text is
    ~320 tokens ≈ 1280 chars — previous 1800-char budget silently overflowed."""
    import json

    if isinstance(state, str):
        text = state
    else:
        text = json.dumps(state, ensure_ascii=False)
    approx_tokens = len(text) // 4
    chunks = chunk_text(text, chunk_words, overlap)
    ranked = rank_chunks(chunks, _query_text(questions))
    kept_idx = [i for i, _ in ranked[:top_k]]
    kept_idx.sort()  # preserve document order
    reduced = "\n\n---\n\n".join(chunks[i] for i in kept_idx)
    truncated = False
    if len(reduced) > state_budget_chars:  # ~320 tokens: fits 512 after question+head
        reduced = reduced[:state_budget_chars]
        truncated = True
    return {"reduced": reduced, "chunks": len(chunks),
            "kept": kept_idx, "scores": {str(i): s for i, s in ranked[:top_k]},
            "approx_input_tokens": approx_tokens, "truncated": truncated}


class LongContextRouter:
    """Wrapper: Router + retrieve-then-decide. Drop-in for >512 inputs."""

    def __init__(self, router=None, **router_kwargs):
        if router is None:
            from laya import Router
            router = Router(**router_kwargs)
        self.router = router

    def predict(self, state: Any, questions: Dict[str, Any], top_k: int = 3,
                chunk_words: int = 100, overlap: int = 20, **kw) -> Dict[str, Any]:
        import json
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        # fast path: fits natively, no retrieval overhead
        if len(text) // 4 <= 900:
            res = self.router.predict(state, questions, **kw)
            res["longctx"] = {"retrieved": False, "chunks": 1}
            return res
        sel = select_state(state, questions, chunk_words, overlap, top_k)
        res = self.router.predict(sel["reduced"], questions, **kw)
        res["longctx"] = {"retrieved": True, **{k: v for k, v in sel.items() if k != "reduced"}}
        return res
