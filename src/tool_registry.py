"""
Dynamic Tool Routing Engine — Phase 2.

Indexes all skill blocks from config/skills.md into a cloud-hosted Qdrant
collection using hybrid (dense + sparse) retrieval.

Embedding backends (in priority order):
  1. HuggingFace BAAI/bge-base-en-v1.5  — 768-dim, truly local, state-of-the-art retrieval.
     BGE requires a query prefix at search time ("Represent this sentence for
     searching relevant passages: ") but NOT at document indexing time.
  2. OpenAI text-embedding-3-small       — 768-dim, cloud fallback.
     Uses the Matryoshka `dimensions=768` parameter for exact size match.

Dense vectors : 768-dim COSINE similarity.
Sparse vectors: BM25-style TF-IDF with per-skill keyword boost (keywords from
                the "Keywords:" line in each skill block are repeated 2× in the
                document token stream, raising their TF weight at retrieval time).

Qdrant defaults to an embedded local store under ``$TASK_CODER_HOME/qdrant``.
Cloud / HTTP Qdrant remains available via settings ``qdrant.mode=http``.

Usage:
    from src.tool_registry import DynamicToolRouter

    router = DynamicToolRouter("config/skills.md")
    tools_md = router.fetch_curated_skills("write a new Python module with tests", top_k=3)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from src.settings import get_settings, models_cache_path, qdrant_path, secrets_path
from src.settings.store import _read_json

HF_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
KEYWORD_BOOST = 2

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        SparseVectorParams,
        SparseVector,
        PointStruct,
    )
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


class DynamicToolRouter:
    """
    Indexes config/skills.md into cloud Qdrant and returns the best-matched
    skill blocks for any given task description via hybrid retrieval + RRF.
    """

    def __init__(self, skills_path: Optional[str | Path] = None) -> None:
        self._skills: list[dict] = []
        self._client: Optional[object] = None
        self._encoder_backend: str = "none"  # "hf" | "openai" | "none"
        self._hf_model = None
        self._oai_client = None
        self._qdrant_mode = "off"
        self._qdrant_error: Optional[str] = None
        self._collection = "agent_skills"
        self._dense_name = "dense"
        self._sparse_name = "sparse"
        self._dense_dims = 768
        self._skills_path: Optional[Path] = None

        # BM25 stats cache — populated once after indexing, invalidated on re-index
        self._vocab:    dict[str, int] = {}
        self._doc_freq: Counter = Counter()
        self._avg_len:  float = 0.0
        self._stats_valid: bool = False

        self._init_encoder()
        self._init_qdrant()

        if skills_path is not None:
            self.parse_and_index_skills(skills_path)

    # ------------------------------------------------------------------
    # Encoder initialisation (HuggingFace → OpenAI fallback)
    # ------------------------------------------------------------------

    def _qdrant_cfg(self):
        return get_settings().qdrant

    def _embedding_cfg(self):
        return get_settings().embeddings

    def _secret(self, key: str) -> str:
        return str(_read_json(secrets_path()).get(key, "") or "")

    def _init_encoder(self) -> None:
        cfg = self._embedding_cfg()
        backend = (cfg.backend or "auto").strip().lower()
        if backend == "none":
            print("[ToolRegistry] Embeddings disabled — using keyword-only fallback.")
            self._encoder_backend = "none"
            return

        hf_token = self._secret("hf") or os.getenv("HF_TOKEN", "")
        cache = models_cache_path() / "hf"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache))

        want_hf = backend in {"auto", "hf"}
        want_openai = backend in {"auto", "openai"}

        if want_hf and _ST_AVAILABLE:
            try:
                self._hf_model = SentenceTransformer(
                    cfg.hf_model, token=hf_token or None
                )
                self._encoder_backend = "hf"
                print(f"[ToolRegistry] Encoder: HuggingFace {cfg.hf_model} (768-dim, local)")
                return
            except Exception as exc:
                print(f"[ToolRegistry] HuggingFace encoder unavailable: {exc}")
                if backend == "hf":
                    self._encoder_backend = "none"
                    return

        openai_key = (
            get_settings().provider("openai")
            and get_settings().provider("openai").api_key
        ) or (
            get_settings().provider("gpt4o")
            and get_settings().provider("gpt4o").api_key
        ) or os.getenv("OPENAI_API_KEY", "")
        if want_openai and openai_key:
            try:
                from openai import OpenAI
                self._oai_client = OpenAI(api_key=openai_key)
                self._encoder_backend = "openai"
                print(f"[ToolRegistry] Encoder: OpenAI {cfg.openai_model} (fallback)")
                return
            except Exception as exc:
                print(f"[ToolRegistry] OpenAI encoder unavailable: {exc}")

        print("[ToolRegistry] No vector encoder available — using keyword-only fallback.")
        self._encoder_backend = "none"

    def _encode_doc(self, text: str) -> list[float]:
        """Embed a skill document (no query prefix for BGE)."""
        if self._encoder_backend == "hf":
            return self._hf_model.encode(text, normalize_embeddings=True).tolist()
        if self._encoder_backend == "openai":
            cfg = self._embedding_cfg()
            r = self._oai_client.embeddings.create(
                model=cfg.openai_model,
                input=text,
                dimensions=cfg.openai_dims,
            )
            return r.data[0].embedding
        return []

    def _encode_query(self, text: str) -> list[float]:
        """
        Embed a search query.

        BGE requires the asymmetric query prefix for retrieval; OpenAI does not.
        """
        if self._encoder_backend == "hf":
            return self._hf_model.encode(
                HF_QUERY_PREFIX + text, normalize_embeddings=True
            ).tolist()
        if self._encoder_backend == "openai":
            cfg = self._embedding_cfg()
            r = self._oai_client.embeddings.create(
                model=cfg.openai_model,
                input=text,
                dimensions=cfg.openai_dims,
            )
            return r.data[0].embedding
        return []

    # ------------------------------------------------------------------
    # Qdrant client initialisation
    # ------------------------------------------------------------------

    def _init_qdrant(self) -> None:
        cfg = self._qdrant_cfg()
        self._qdrant_mode = cfg.mode
        self._collection = cfg.collection or "agent_skills"
        self._dense_name = cfg.dense_name or "dense"
        self._sparse_name = cfg.sparse_name or "sparse"
        self._dense_dims = int(cfg.dense_dims or 768)
        if cfg.mode == "off":
            print("[ToolRegistry] Qdrant disabled — keyword fallback active.")
            return
        if not _QDRANT_AVAILABLE:
            self._qdrant_error = "qdrant-client not installed"
            print("[ToolRegistry] qdrant-client not installed — vector search disabled.")
            return
        try:
            if cfg.mode == "http":
                if not cfg.url:
                    raise RuntimeError("qdrant.mode=http but url is empty")
                self._client = QdrantClient(
                    url=cfg.url,
                    api_key=cfg.api_key or None,
                )
                location = cfg.url
            else:
                path = cfg.path or str(qdrant_path())
                Path(path).mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=path)
                location = path
            self._ensure_collection()
            print(
                f"[ToolRegistry] Connected to Qdrant ({cfg.mode}): {location} "
                f"| collection={self._collection}"
            )
        except Exception as exc:
            self._qdrant_error = str(exc)
            print(f"[ToolRegistry] Qdrant connection failed: {exc} — keyword fallback active.")
            self._client = None

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection only if it does not already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    self._dense_name: VectorParams(
                        size=self._dense_dims, distance=Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    self._sparse_name: SparseVectorParams(),
                },
            )
            print(f"[ToolRegistry] Created Qdrant collection '{self._collection}'.")

    def status(self) -> dict:
        return {
            "mode": self._qdrant_mode,
            "reachable": self._client is not None,
            "error": self._qdrant_error,
            "encoder": self._encoder_backend,
            "collection": self._collection,
            "skill_count": len(self._skills),
        }

    def _skills_hash(self, file_path: Path) -> str:
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        return f"{digest}:{self._encoder_backend}:{self._collection}"

    def _index_meta_path(self) -> Path:
        cfg = self._qdrant_cfg()
        root = Path(cfg.path or qdrant_path())
        return root / "skills_index.json"

    def _should_reindex(self, file_path: Path) -> bool:
        meta = _read_json(self._index_meta_path())
        return meta.get("hash") != self._skills_hash(file_path)

    # ------------------------------------------------------------------
    # Parsing & indexing
    # ------------------------------------------------------------------

    def parse_and_index_skills(self, file_path: str | Path) -> int:
        """
        Parse SKILL_START/SKILL_END blocks from skills.md and upsert each
        block into Qdrant with both dense and sparse vectors plus keyword
        metadata payload.

        Keyword enrichment strategy:
          - Each skill block contains a "Keywords: kw1, kw2, kw3" line.
          - Keywords are stored in the Qdrant payload for optional hard-filter queries.
          - Keywords are repeated KEYWORD_BOOST times in the BM25 document token
            stream, raising their TF weight so keyword-exact milestone descriptions
            score higher at sparse retrieval time.

        BM25 stats are computed once over ALL skills after parsing, then cached
        for O(1) access at query time and during vector construction.

        Returns:
            Number of skills indexed.
        """
        path = Path(file_path)
        self._skills_path = path
        content = path.read_text(encoding="utf-8")
        blocks = re.findall(
            r"<!-- SKILL_START:\s*(\S+?)\s*-->(.*?)<!-- SKILL_END -->",
            content,
            re.DOTALL,
        )

        self._skills = []

        for idx, (skill_name, block_text) in enumerate(blocks):
            clean = block_text.strip()

            kw_match = re.search(r"\*\*Keywords:\*\*\s*([^\n]+)", clean)
            keywords: list[str] = []
            if kw_match:
                keywords = [k.strip() for k in kw_match.group(1).split(",") if k.strip()]

            base_tokens = self._tokenize(clean)
            enriched_tokens = base_tokens + [k.lower() for k in keywords] * KEYWORD_BOOST

            self._skills.append({
                "id": idx,
                "name": skill_name,
                "raw_markdown": clean,
                "keywords": keywords,
                "base_tokens": base_tokens,
                "enriched_tokens": enriched_tokens,
            })

        # Build BM25 stats once over the complete skill set
        self._rebuild_bm25_stats()

        # Upsert into Qdrant using the now-stable cached stats
        if self._client and self._encoder_backend != "none" and self._should_reindex(path):
            points = []
            for skill in self._skills:
                dense_vec = self._encode_doc(skill["raw_markdown"])
                sparse_indices, sparse_values = self._bm25_doc_vector(skill["enriched_tokens"])
                points.append(
                    PointStruct(
                        id=skill["id"],
                        vector={
                            self._dense_name: dense_vec,
                            self._sparse_name: SparseVector(
                                indices=sparse_indices,
                                values=sparse_values,
                            ),
                        },
                        payload={
                            "name": skill["name"],
                            "raw_markdown": skill["raw_markdown"],
                            "keywords": skill["keywords"],
                        },
                    )
                )
            if points:
                try:
                    self._client.upsert(collection_name=self._collection, points=points)
                    meta_path = self._index_meta_path()
                    meta_path.parent.mkdir(parents=True, exist_ok=True)
                    meta_path.write_text(
                        json.dumps({
                            "hash": self._skills_hash(path),
                            "count": len(points),
                        }) + "\n",
                        encoding="utf-8",
                    )
                except Exception as exc:
                    print(f"[ToolRegistry] Qdrant upsert failed: {exc} — keyword fallback active.")
                    self._qdrant_error = str(exc)
        elif self._client and self._encoder_backend != "none":
            print("[ToolRegistry] Skills index is current — skipping re-embed.")

        print(f"[ToolRegistry] Indexed {len(self._skills)} skills from {file_path}.")
        return len(self._skills)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def fetch_curated_skills(
        self,
        task_description: str,
        top_k: int = 3,
        rrf_k: int = 60,
    ) -> str:
        """
        Return the top-k skill blocks as a single markdown string, ranked by
        Reciprocal Rank Fusion (RRF) over dense and sparse Qdrant search results.

        Falls back to keyword-only ranking if Qdrant is unavailable or raises.

        Args:
            task_description: The milestone intent string to match against.
            top_k:            Number of skills to return (2 or 3 recommended).
            rrf_k:            RRF smoothing constant (standard value: 60).

        Returns:
            Concatenated markdown text of the matched skill blocks.
        """
        if not self._skills:
            return ""

        if self._client and self._encoder_backend != "none":
            try:
                return self._hybrid_search(task_description, top_k, rrf_k)
            except Exception as exc:
                print(
                    f"[ToolRegistry] Qdrant query failed: {exc} — "
                    "falling back to keyword search."
                )

        return self._keyword_fallback(task_description, top_k)

    def search_tools(
        self,
        query: str,
        top_k: int = 3,
        rrf_k: int = 60,
    ) -> dict:
        """
        Structured progressive-disclosure lookup for the Worker meta-tool.

        Retrieval remains deterministic: no secondary LLM is invoked. The
        existing Markdown method is retained for compatibility, while this
        method also returns the canonical tool names so the runtime can expand
        its active-tool allowlist after discovery.
        """
        top_k = max(1, min(int(top_k), 5))
        documentation = self.fetch_curated_skills(query, top_k=top_k, rrf_k=rrf_k)
        selected: list[dict] = []
        for skill in self._skills:
            if skill["raw_markdown"] in documentation:
                selected.append({
                    "name": skill["name"],
                    "documentation": skill["raw_markdown"],
                })
        return {
            "success": True,
            "query": query,
            "tools": [item["name"] for item in selected],
            "documentation": documentation,
            "count": len(selected),
        }

    def _hybrid_search(self, query: str, top_k: int, rrf_k: int) -> str:
        dense_vec = self._encode_query(query)
        query_tokens = self._tokenize(query)
        sparse_indices, sparse_values = self._bm25_query_vector(query_tokens)

        dense_hits = self._client.query_points(
            collection_name=self._collection,
            query=dense_vec,
            using=self._dense_name,
            limit=top_k * 2,
        ).points
        sparse_hits = self._client.query_points(
            collection_name=self._collection,
            query=SparseVector(indices=sparse_indices, values=sparse_values),
            using=self._sparse_name,
            limit=top_k * 2,
        ).points

        # Reciprocal Rank Fusion
        rrf_scores: dict[int, float] = {}
        for rank, hit in enumerate(dense_hits):
            rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, hit in enumerate(sparse_hits):
            rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + 1.0 / (rrf_k + rank + 1)

        top_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

        results = []
        for sid in top_ids:
            skill = next((s for s in self._skills if s["id"] == sid), None)
            if skill:
                results.append(skill["raw_markdown"])
        return "\n\n---\n\n".join(results)

    def _keyword_fallback(self, query: str, top_k: int) -> str:
        """
        Pure-Python TF-IDF keyword fallback when Qdrant / encoder is unavailable.

        Also boosts skills whose keywords appear in the query string directly.
        """
        query_tokens = set(self._tokenize(query))
        query_lower = query.lower()

        scored = []
        for skill in self._skills:
            overlap   = len(query_tokens & set(skill["base_tokens"]))
            kw_bonus  = sum(1 for kw in skill["keywords"] if kw in query_lower)
            scored.append((overlap + kw_bonus * 2, skill["raw_markdown"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return "\n\n---\n\n".join(md for _, md in scored[:top_k])

    # ------------------------------------------------------------------
    # BM25 sparse vector computation
    # ------------------------------------------------------------------

    def _rebuild_bm25_stats(self) -> None:
        """
        Compute and cache vocabulary, document-frequency, and average document
        length from the current skill set.

        Called once after all skills are parsed. Uses enriched_tokens so the
        keyword boost is reflected in IDF weights.
        """
        vocab: dict[str, int] = {}
        doc_freq: Counter = Counter()
        total_tokens = 0

        for skill in self._skills:
            unique_toks = set(skill["enriched_tokens"])
            total_tokens += len(skill["enriched_tokens"])
            for t in unique_toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
                doc_freq[t] += 1

        n_docs = max(len(self._skills), 1)
        self._vocab     = vocab
        self._doc_freq  = doc_freq
        self._avg_len   = total_tokens / n_docs
        self._stats_valid = True

    def _bm25_doc_vector(
        self,
        tokens: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> tuple[list[int], list[float]]:
        """BM25 sparse vector for a document using the cached vocabulary stats."""
        n_docs  = max(len(self._skills), 1)
        doc_len = len(tokens)
        tf_counter = Counter(tokens)

        indices, values = [], []
        for token, tf in tf_counter.items():
            if token not in self._vocab:
                continue
            idf = math.log(
                (n_docs - self._doc_freq[token] + 0.5)
                / (self._doc_freq[token] + 0.5) + 1
            )
            tf_norm = (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * doc_len / max(self._avg_len, 1))
            )
            indices.append(self._vocab[token])
            values.append(float(idf * tf_norm))
        return indices, values

    def _bm25_query_vector(
        self,
        tokens: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> tuple[list[int], list[float]]:
        """
        BM25 sparse vector for a query using the cached vocabulary stats.

        Query vectors use plain (non-enriched) tokens; IDF is computed from
        the document collection so the two vectors share the same term space.
        """
        if not self._skills or not self._stats_valid:
            return [], []

        n_docs  = max(len(self._skills), 1)
        doc_len = len(tokens)
        tf_counter = Counter(tokens)

        indices, values = [], []
        for token, tf in tf_counter.items():
            if token not in self._vocab:
                continue
            idf = math.log(
                (n_docs - self._doc_freq[token] + 0.5)
                / (self._doc_freq[token] + 0.5) + 1
            )
            tf_norm = (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * doc_len / max(self._avg_len, 1))
            )
            indices.append(self._vocab[token])
            values.append(float(idf * tf_norm))
        return indices, values

    # ------------------------------------------------------------------
    # Tokenisation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, split on non-alphanumeric chars, drop single-char tokens."""
        return [t for t in re.split(r"[^a-z0-9_]+", text.lower()) if len(t) > 1]

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        for closer in ("close", "stop"):
            fn = getattr(client, closer, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass
        inner = getattr(client, "_client", None)
        if inner is not None and hasattr(inner, "close"):
            try:
                inner.close()
            except Exception:
                pass

    def list_skills(self) -> list[str]:
        return [s["name"] for s in self._skills]

    def get_skill_by_name(self, name: str) -> Optional[str]:
        for s in self._skills:
            if s["name"] == name:
                return s["raw_markdown"]
        return None

    def get_skill_keywords(self, name: str) -> list[str]:
        for s in self._skills:
            if s["name"] == name:
                return s["keywords"]
        return []
