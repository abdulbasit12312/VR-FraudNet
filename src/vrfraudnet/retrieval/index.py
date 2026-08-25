"""Training-only retrieval index (manuscript Section 4.3.4).

Manuscript specification
------------------------
* The index is built per dataset from the *training partition only*.
* Each record holds a standardised transaction summary, temporal and graph
  context, the Stage 1 probability band, the structured rationale, the claims
  and the evidence references.
* Only schema-valid and verifier-accepted training examples are indexed.
* Validation, test, transfer, temporal-shock and adversarial-evaluation examples
  are excluded.
* For datasets with explicit timestamps, an indexed example is eligible only if
  it occurred strictly before the query transaction.
* Examples sharing a transaction identifier or a direct entity identifier with
  the query are excluded, to prevent identifier-based retrieval leakage.
* Summaries are encoded with ``all-MiniLM-L6-v2`` and stored in FAISS under
  cosine similarity; the four most similar eligible examples are retrieved and
  ordered by similarity. Fewer than four is allowed; retrieval is omitted
  entirely when no eligible example exists.

The leakage guarantees are enforced structurally in :class:`RetrievalIndex`, not
left to caller discipline: adding a record whose partition is not ``train``
raises immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from vrfraudnet.errors import LeakageError, MissingArtefactError

#: Manuscript S4.3.4.
ENCODER_MODEL = "all-MiniLM-L6-v2"
N_RETRIEVED = 4


@dataclass
class RetrievalRecord:
    """One indexed training example."""

    record_id: str
    partition: str
    timestamp: float | None
    summary: str
    probability_band: str
    rationale: dict[str, Any]
    entity_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.partition != "train":
            raise LeakageError(
                f"retrieval record {self.record_id!r} belongs to partition "
                f"{self.partition!r}; manuscript S4.3.4 permits training examples only"
            )


class RetrievalIndex:
    """Cosine-similarity index over verifier-accepted training rationales."""

    def __init__(
        self,
        *,
        encoder: Callable[[Sequence[str]], np.ndarray] | None = None,
        n_retrieved: int = N_RETRIEVED,
    ) -> None:
        self.records: list[RetrievalRecord] = []
        self.embeddings: np.ndarray | None = None
        self.n_retrieved = int(n_retrieved)
        self._encoder = encoder

    # -- construction -------------------------------------------------------
    def add(self, record: RetrievalRecord, *, verifier_accepted: bool) -> None:
        """Add one record. Only verifier-accepted, schema-valid examples qualify."""
        if not verifier_accepted:
            return
        self.records.append(record)
        self.embeddings = None  # invalidate

    def build(self) -> "RetrievalIndex":
        """Encode every summary and normalise for cosine similarity."""
        if not self.records:
            self.embeddings = np.zeros((0, 1), dtype=np.float32)
            return self
        encoder = self._encoder or default_encoder()
        matrix = np.asarray(encoder([r.summary for r in self.records]), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.embeddings = matrix / np.maximum(norms, 1e-12)
        return self

    # -- query --------------------------------------------------------------
    def query(
        self,
        summary: str,
        *,
        query_timestamp: float | None = None,
        query_transaction_id: str | None = None,
        query_entity_ids: frozenset[str] | None = None,
    ) -> list[RetrievalRecord]:
        """Return up to ``n_retrieved`` eligible examples, ordered by similarity.

        Eligibility (manuscript S4.3.4):
        1. training partition only (guaranteed at insertion time);
        2. strictly earlier timestamp than the query, when timestamps exist;
        3. no shared transaction identifier;
        4. no shared direct entity identifier.
        """
        if self.embeddings is None:
            self.build()
        if not self.records:
            return []  # "Retrieval is omitted when no eligible training example exists."

        eligible = [
            i
            for i, record in enumerate(self.records)
            if _is_eligible(
                record, query_timestamp, query_transaction_id, query_entity_ids or frozenset()
            )
        ]
        if not eligible:
            return []

        encoder = self._encoder or default_encoder()
        vector = np.asarray(encoder([summary]), dtype=np.float32)[0]
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        similarities = self.embeddings[eligible] @ vector
        order = np.argsort(-similarities, kind="stable")[: self.n_retrieved]
        return [self.records[eligible[int(i)]] for i in order]

    def __len__(self) -> int:
        return len(self.records)


def _is_eligible(
    record: RetrievalRecord,
    query_timestamp: float | None,
    query_transaction_id: str | None,
    query_entity_ids: frozenset[str],
) -> bool:
    if query_transaction_id is not None and record.record_id == query_transaction_id:
        return False
    if query_entity_ids and record.entity_ids & query_entity_ids:
        return False
    if query_timestamp is not None and record.timestamp is not None:
        if record.timestamp >= query_timestamp:
            return False
    return True


def default_encoder() -> Callable[[Sequence[str]], np.ndarray]:
    """Return the ``all-MiniLM-L6-v2`` sentence encoder (manuscript S4.3.4).

    Raises when ``sentence-transformers`` is unavailable rather than silently
    substituting a different embedding, because a different encoder would change
    which demonstrations the language model sees.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingArtefactError(
            f"the manuscript's retrieval encoder ({ENCODER_MODEL}) requires "
            "sentence-transformers. Install it with:\n"
            "    pip install -r requirements-optional.txt\n"
            "No substitute encoder is used, because a different embedding would "
            "change the retrieved demonstrations."
        ) from exc

    model = SentenceTransformer(ENCODER_MODEL)

    def encode(texts: Sequence[str]) -> np.ndarray:
        return model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)

    return encode


def hashing_encoder(dim: int = 256) -> Callable[[Sequence[str]], np.ndarray]:
    """Deterministic bag-of-character-trigrams encoder for tests only.

    Never used in a manuscript reproduction path: it exists so the index's
    leakage and ordering logic can be unit-tested without downloading a model.
    """

    def encode(texts: Sequence[str]) -> np.ndarray:
        import hashlib

        out = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f"  {text}  "
            for i in range(len(padded) - 2):
                trigram = padded[i : i + 3]
                digest = hashlib.blake2b(trigram.encode("utf-8"), digest_size=4).digest()
                out[row, int.from_bytes(digest, "big") % dim] += 1.0
        return out

    return encode
