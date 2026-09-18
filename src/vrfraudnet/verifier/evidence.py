"""Trusted evidence available to the Stage 3 verifier.

Manuscript S4.4: "the verifier receives the structured rationale r_i, the
transaction feature vector x_i, the temporal graph context G_t when graph
structure is available, timestamp information t, and the closed set of
admissible claim primitives and consistency rules."

The evidence dictionary is *trusted*: it is built from the dataset, never from
model output. Anything the language model asserts must be checked against this
object, and a claim that cites an identifier absent from it is unverifiable by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

EvidenceKind = Literal[
    "numeric_field",
    "categorical_field",
    "entity_field",
    "timestamp",
    "temporal_aggregate",
    "graph_snapshot",
]


@dataclass(frozen=True)
class EvidenceEntry:
    """One trusted fact about the transaction under evaluation."""

    evidence_id: str
    kind: EvidenceKind
    value: Any
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence_id must be a non-empty string")


@dataclass
class GraphEvidence:
    """The temporal graph snapshot observable at the transaction time.

    ``adjacency`` maps an entity identifier to the set of entity identifiers it
    is connected to *as of* ``as_of_time``. Building it from a snapshot rather
    than from the full graph is what makes graph-path verification honest: the
    verifier cannot confirm a path that only exists because of a future edge.
    """

    adjacency: dict[str, set[str]]
    as_of_time: float
    flagged_entities: set[str] = field(default_factory=set)

    def neighbours(self, node: str) -> set[str]:
        return self.adjacency.get(node, set())

    def path_within(self, source: str, target: str, max_hops: int) -> tuple[bool, int | None]:
        """Breadth-first search for a path of at most ``max_hops`` edges.

        Returns ``(found, hops)``. Bounded search keeps verification cost on the
        escalation path deterministic.
        """
        if source not in self.adjacency and target not in self.adjacency:
            return False, None
        if source == target:
            return True, 0
        frontier = {source}
        seen = {source}
        for hop in range(1, max_hops + 1):
            nxt: set[str] = set()
            for node in frontier:
                for neighbour in self.neighbours(node):
                    if neighbour == target:
                        return True, hop
                    if neighbour not in seen:
                        seen.add(neighbour)
                        nxt.add(neighbour)
            if not nxt:
                break
            frontier = nxt
        return False, None


@dataclass
class TransactionEvidence:
    """Everything the verifier is allowed to consult for one transaction."""

    transaction_id: str
    timestamp: float
    entries: dict[str, EvidenceEntry]
    graph: GraphEvidence | None = None

    def get(self, evidence_id: str) -> EvidenceEntry | None:
        return self.entries.get(evidence_id)

    def has(self, evidence_id: str) -> bool:
        return evidence_id in self.entries

    @property
    def graph_available(self) -> bool:
        return self.graph is not None

    @classmethod
    def from_row(
        cls,
        transaction_id: str,
        timestamp: float,
        numeric: Mapping[str, float] | None = None,
        categorical: Mapping[str, str] | None = None,
        entities: Mapping[str, str] | None = None,
        temporal_aggregates: Mapping[str, float] | None = None,
        graph: GraphEvidence | None = None,
    ) -> "TransactionEvidence":
        """Build an evidence dictionary from typed field groups.

        Field groups are kept explicit rather than inferred from dtypes so that
        a numeric identifier (a card number, say) is never silently treated as a
        quantity a numeric_comparison claim may reason about.
        """
        entries: dict[str, EvidenceEntry] = {}
        for name, value in (numeric or {}).items():
            entries[name] = EvidenceEntry(name, "numeric_field", float(value))
        for name, value in (categorical or {}).items():
            entries[name] = EvidenceEntry(name, "categorical_field", str(value))
        for name, value in (entities or {}).items():
            entries[name] = EvidenceEntry(name, "entity_field", str(value))
        for name, value in (temporal_aggregates or {}).items():
            entries[name] = EvidenceEntry(name, "temporal_aggregate", float(value), timestamp)
        entries["transaction_time"] = EvidenceEntry(
            "transaction_time", "timestamp", float(timestamp), float(timestamp)
        )
        if graph is not None:
            entries["graph_snapshot"] = EvidenceEntry(
                "graph_snapshot", "graph_snapshot", graph, graph.as_of_time
            )
        return cls(
            transaction_id=transaction_id,
            timestamp=float(timestamp),
            entries=entries,
            graph=graph,
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "TransactionEvidence":
        """Build evidence from the JSON layout of ``examples/evidence_example.json``.

        The layout is the on-disk form of a Stage 2 corpus record's ``evidence``
        field (docs/STAGE2_LORA.md). Typed groups are mandatory for the same
        reason as in :meth:`from_row`: nothing is inferred from a value's type.
        """
        graph_payload = payload.get("graph")
        graph = None
        if graph_payload:
            graph = GraphEvidence(
                adjacency={
                    str(node): {str(n) for n in neighbours}
                    for node, neighbours in dict(graph_payload.get("adjacency", {})).items()
                },
                as_of_time=float(graph_payload.get("as_of_time", payload["timestamp"])),
                flagged_entities={str(n) for n in graph_payload.get("flagged_entities", [])},
            )
        return cls.from_row(
            str(payload["transaction_id"]),
            float(payload["timestamp"]),
            numeric=payload.get("numeric"),
            categorical=payload.get("categorical"),
            entities=payload.get("entities"),
            temporal_aggregates=payload.get("temporal_aggregates"),
            graph=graph,
        )


def build_graph_evidence(
    edge_index: np.ndarray,
    edge_time: np.ndarray,
    node_names: list[str],
    *,
    as_of_time: float,
    flagged: set[str] | None = None,
) -> GraphEvidence:
    """Materialise a :class:`GraphEvidence` snapshot restricted to ``as_of_time``.

    Edges with ``edge_time > as_of_time`` are dropped before the adjacency map
    is built, which is the operational form of manuscript S4.1's "future edges
    ... are excluded".
    """
    keep = edge_time <= as_of_time
    src = edge_index[0][keep]
    dst = edge_index[1][keep]
    adjacency: dict[str, set[str]] = {}
    for s, d in zip(src.tolist(), dst.tolist()):
        a, b = node_names[s], node_names[d]
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return GraphEvidence(
        adjacency=adjacency, as_of_time=float(as_of_time), flagged_entities=flagged or set()
    )
