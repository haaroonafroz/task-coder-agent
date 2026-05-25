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

Qdrant connection reads from .env:
  QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
  QDRANT_DENSE_VECTOR_NAME, QDRANT_SPARSE_VECTOR_NAME, QDRANT_DENSE_VECTOR_DIMS

Usage:
    from src.tool_registry import DynamicToolRouter

    router = DynamicToolRouter("config/skills.md")
    tools_md = router.fetch_curated_skills("write a new Python module with tests", top_k=3)
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Qdrant config (from .env)
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("QDRANT_COLLECTION", "agent_skills")
DENSE_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
SPARSE_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
DENSE_DIMS = int(os.getenv("QDRANT_DENSE_VECTOR_DIMS", "768"))

# ---------------------------------------------------------------------------
# Embedding config (from .env)
# ---------------------------------------------------------------------------
HF_MODEL = "BAAI/bge-base-en-v1.5"
# BGE asymmetric retrieval: prefix queries only, NOT documents.
HF_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
HF_TOKEN = os.getenv("HF_TOKEN", "")

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIMS = int(os.getenv("OPENAI_EMBEDDING_DIMS", "768"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Keyword repetition factor for BM25 boost (keywords appear N extra times
# in the document token stream to increase their term-frequency weight).
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
        NamedVector,
        NamedSparseVector,
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

        self._init_encoder()
        self._init_qdrant()

        if skills_path is not None:
            self.parse_and_index_skills(skills_path)

    # ------------------------------------------------------------------
    # Encoder initialisation (HuggingFace → OpenAI fallback)
    # ------------------------------------------------------------------

    def _init_encoder(self) -> None:
        if _ST_AVAILABLE:
            try:
                self._hf_model = SentenceTransformer(
                    HF_MODEL, token=HF_TOKEN or None
                )
                self._encoder_backend = "hf"
                print(f"[ToolRegistry] Encoder: HuggingFace {HF_MODEL} (768-dim, local)")
                return
            except Exception as exc:
                print(f"[ToolRegistry] HuggingFace encoder unavailable: {exc}")

        if OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self._oai_client = OpenAI(api_key=OPENAI_API_KEY)
                self._encoder_backend = "openai"
                print(f"[ToolRegistry] Encoder: OpenAI {OPENAI_EMBEDDING_MODEL} (fallback)")
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
            r = self._oai_client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=text,
                dimensions=OPENAI_EMBEDDING_DIMS,
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
            r = self._oai_client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=text,
                dimensions=OPENAI_EMBEDDING_DIMS,
            )
            return r.data[0].embedding
        return []

    # ------------------------------------------------------------------
    # Qdrant client initialisation
    # ------------------------------------------------------------------

    def _init_qdrant(self) -> None:
        if not _QDRANT_AVAILABLE:
            print("[ToolRegistry] qdrant-client not installed — vector search disabled.")
            return
        if not QDRANT_URL:
            print("[ToolRegistry] QDRANT_URL not set — vector search disabled.")
            return
        try:
            self._client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
            self._ensure_collection()
            print(f"[ToolRegistry] Connected to Qdrant: {QDRANT_URL} | collection={COLLECTION}")
        except Exception as exc:
            print(f"[ToolRegistry] Qdrant connection failed: {exc} — keyword fallback active.")
            self._client = None

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection only if it does not already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION not in existing:
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config={
                    DENSE_NAME: VectorParams(size=DENSE_DIMS, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    SPARSE_NAME: SparseVectorParams(),
                },
            )
            print(f"[ToolRegistry] Created Qdrant collection '{COLLECTION}'.")

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

        Returns:
            Number of skills indexed.
        """
        content = Path(file_path).read_text(encoding="utf-8")
        blocks = re.findall(
            r"<!-- SKILL_START:\s*(\S+?)\s*-->(.*?)<!-- SKILL_END -->",
            content,
            re.DOTALL,
        )

        self._skills = []
        points = []

        for idx, (skill_name, block_text) in enumerate(blocks):
            clean = block_text.strip()

            # Extract the Keywords line: "- **Keywords:** kw1, kw2, kw3, kw4"
            kw_match = re.search(r"\*\*Keywords:\*\*\s*([^\n]+)", clean)
            keywords: list[str] = []
            if kw_match:
                keywords = [k.strip() for k in kw_match.group(1).split(",") if k.strip()]

            # Base tokens from full text
            base_tokens = self._tokenize(clean)
            # Enriched token stream: keywords repeated KEYWORD_BOOST extra times
            enriched_tokens = base_tokens + [k.lower() for k in keywords] * KEYWORD_BOOST

            entry = {
                "id": idx,
                "name": skill_name,
                "raw_markdown": clean,
                "keywords": keywords,
                "base_tokens": base_tokens,
                "enriched_tokens": enriched_tokens,
            }
            self._skills.append(entry)

            if self._client and self._encoder_backend != "none":
                dense_vec = self._encode_doc(clean)
                sparse_indices, sparse_values = self._bm25_doc_vector(enriched_tokens, idx)
                points.append(
                    PointStruct(
                        id=idx,
                        vector={
                            DENSE_NAME: dense_vec,
                            SPARSE_NAME: SparseVector(
                                indices=sparse_indices,
                                values=sparse_values,
                            ),
                        },
                        payload={
                            "name": skill_name,
                            "raw_markdown": clean,
                            "keywords": keywords,
                        },
                    )
                )

        if self._client and points:
            self._client.upsert(collection_name=COLLECTION, points=points)

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
            return self._hybrid_search(task_description, top_k, rrf_k)
        return self._keyword_fallback(task_description, top_k)

    def _hybrid_search(self, query: str, top_k: int, rrf_k: int) -> str:
        dense_vec = self._encode_query(query)
        query_tokens = self._tokenize(query)
        sparse_indices, sparse_values = self._bm25_query_vector(query_tokens)
        dense_hits = self._client.query_points(
            collection_name=COLLECTION,
            query=dense_vec,
            using=DENSE_NAME,
            limit=top_k * 2,
        ).points
        sparse_hits = self._client.query_points(
            collection_name=COLLECTION,
            query=SparseVector(indices=sparse_indices, values=sparse_values),
            using=SPARSE_NAME,
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
            # Overlap of query tokens with base document tokens
            overlap = len(query_tokens & set(skill["base_tokens"]))
            # Extra bonus if any keyword appears literally in the query string
            kw_bonus = sum(1 for kw in skill["keywords"] if kw in query_lower)
            scored.append((overlap + kw_bonus * 2, skill["raw_markdown"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return "\n\n---\n\n".join(md for _, md in scored[:top_k])

    # ------------------------------------------------------------------
    # BM25 sparse vector computation
    # ------------------------------------------------------------------

    def _global_vocab_and_stats(self) -> tuple[dict[str, int], Counter, float]:
        """
        Build a vocabulary and document-frequency counter from all indexed skills.
        Uses `enriched_tokens` so keyword boost is reflected in IDF too.
        """
        vocab: dict[str, int] = {}
        doc_freq: Counter = Counter()
        for skill in self._skills:
            unique_toks = set(skill["enriched_tokens"])
            for t in unique_toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
                doc_freq[t] += 1
        n_docs = max(len(self._skills), 1)
        avg_len = (
            sum(len(s["enriched_tokens"]) for s in self._skills) / n_docs
        )
        return vocab, doc_freq, avg_len

    def _bm25_doc_vector(
        self,
        tokens: list[str],
        doc_idx: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> tuple[list[int], list[float]]:
        """BM25 sparse vector for a document (uses enriched tokens)."""
        vocab, doc_freq, avg_len = self._global_vocab_and_stats()
        n_docs = max(len(self._skills), 1)
        doc_len = len(tokens)
        tf_counter = Counter(tokens)

        indices, values = [], []
        for token, tf in tf_counter.items():
            if token not in vocab:
                continue
            idf = math.log(
                (n_docs - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5) + 1
            )
            tf_norm = (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * doc_len / max(avg_len, 1))
            )
            indices.append(vocab[token])
            values.append(float(idf * tf_norm))
        return indices, values

    def _bm25_query_vector(
        self,
        tokens: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> tuple[list[int], list[float]]:
        """
        BM25 sparse vector for a query.

        Query vectors use plain (non-enriched) tokens; IDF is still computed from
        the document collection so the two vectors live in the same term space.
        """
        if not self._skills:
            return [], []
        vocab, doc_freq, avg_len = self._global_vocab_and_stats()
        n_docs = max(len(self._skills), 1)
        # Queries are typically short; treat query length as its actual length
        doc_len = len(tokens)
        tf_counter = Counter(tokens)

        indices, values = [], []
        for token, tf in tf_counter.items():
            if token not in vocab:
                continue
            idf = math.log(
                (n_docs - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5) + 1
            )
            tf_norm = (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * doc_len / max(avg_len, 1))
            )
            indices.append(vocab[token])
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
