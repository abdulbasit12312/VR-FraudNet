"""The five claim primitives of manuscript Section 4.3.2 / 4.4.

Each checker returns a :class:`ClaimOutcome` with one of three statuses:

``supported``
    The evidence exists and the assertion holds.
``contradicted``
    The evidence exists and the assertion is false.
``unverifiable``
    The required evidence is absent (for example a graph_path claim on a
    dataset with no graph). Manuscript Table 4, example E2 treats this as a
    rejection, and so does the engine.

Only ``supported`` counts towards acceptance. The distinction between
``contradicted`` and ``unverifiable`` is kept because the audit trace is meant
to tell a human investigator *why* a rationale failed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from vrfraudnet.verifier.evidence import GraphEvidence, TransactionEvidence

ClaimStatus = Literal["supported", "contradicted", "unverifiable"]


@dataclass(frozen=True)
class ClaimOutcome:
    """Result of checking one claim, retained in the claim-level audit trace."""

    index: int
    claim_type: str
    field: str
    operator: str
    status: ClaimStatus
    detail: str
    observed: Any = None

    @property
    def passed(self) -> bool:
        return self.status == "supported"


_NUMERIC_OPS = {
    "lt": lambda a, b, tol: a < b - tol,
    "le": lambda a, b, tol: a <= b + tol,
    "gt": lambda a, b, tol: a > b + tol,
    "ge": lambda a, b, tol: a >= b - tol,
    "eq": lambda a, b, tol: abs(a - b) <= tol,
    "ne": lambda a, b, tol: abs(a - b) > tol,
}


def check_numeric_comparison(
    index: int, claim: dict[str, Any], evidence: TransactionEvidence, rule: dict[str, Any]
) -> ClaimOutcome:
    """Ordering or equality between a numeric field and a numeric constant."""
    entry = evidence.get(claim["evidence_id"])
    if entry is None or entry.kind != "numeric_field":
        return ClaimOutcome(
            index, "numeric_comparison", claim["field"], claim["operator"], "unverifiable",
            f"no numeric evidence entry named {claim['evidence_id']!r}",
        )
    if not isinstance(claim["value"], (int, float)) or isinstance(claim["value"], bool):
        return ClaimOutcome(
            index, "numeric_comparison", claim["field"], claim["operator"], "contradicted",
            "numeric_comparison requires a numeric asserted value",
        )
    tol = float(rule.get("float_tolerance", 1e-9))
    op = _NUMERIC_OPS[claim["operator"]]
    holds = bool(op(float(entry.value), float(claim["value"]), tol))
    return ClaimOutcome(
        index, "numeric_comparison", claim["field"], claim["operator"],
        "supported" if holds else "contradicted",
        f"observed {entry.value!r} {claim['operator']} asserted {claim['value']!r}",
        observed=entry.value,
    )


def check_set_membership(
    index: int, claim: dict[str, Any], evidence: TransactionEvidence, rule: dict[str, Any]
) -> ClaimOutcome:
    """Membership of a categorical field in an enumerated admissible set."""
    entry = evidence.get(claim["evidence_id"])
    if entry is None or entry.kind != "categorical_field":
        return ClaimOutcome(
            index, "set_membership", claim["field"], claim["operator"], "unverifiable",
            f"no categorical evidence entry named {claim['evidence_id']!r}",
        )
    value = claim["value"]
    members = [str(v) for v in (value if isinstance(value, list) else [value])]
    if len(members) > int(rule.get("max_set_size", 32)):
        return ClaimOutcome(
            index, "set_membership", claim["field"], claim["operator"], "contradicted",
            f"asserted set of size {len(members)} exceeds the permitted maximum",
        )
    present = str(entry.value) in members
    holds = present if claim["operator"] == "in" else not present
    return ClaimOutcome(
        index, "set_membership", claim["field"], claim["operator"],
        "supported" if holds else "contradicted",
        f"observed {entry.value!r}; asserted {claim['operator']} {members}",
        observed=entry.value,
    )


def check_temporal_relation(
    index: int, claim: dict[str, Any], evidence: TransactionEvidence, rule: dict[str, Any]
) -> ClaimOutcome:
    """Ordering, window, or count claim over past-only temporal context."""
    entry = evidence.get(claim["evidence_id"])
    if entry is None or entry.kind not in {"timestamp", "temporal_aggregate"}:
        return ClaimOutcome(
            index, "temporal_relation", claim["field"], claim["operator"], "unverifiable",
            f"no temporal evidence entry named {claim['evidence_id']!r}",
        )

    # Past-only enforcement: a temporal claim must not reference an event after
    # the transaction under evaluation (manuscript S4.1 / consistency rule CR5).
    if rule.get("enforce_past_only", True) and entry.timestamp is not None:
        if entry.timestamp > evidence.timestamp + 1e-9:
            return ClaimOutcome(
                index, "temporal_relation", claim["field"], claim["operator"], "unverifiable",
                f"evidence timestamp {entry.timestamp!r} is after the transaction "
                f"timestamp {evidence.timestamp!r}; future evidence is not admissible",
            )

    operator = claim["operator"]
    observed = float(entry.value)
    asserted = claim["value"]
    if operator in {"count_ge", "count_le", "within_hours"} and not isinstance(
        asserted, (int, float)
    ):
        return ClaimOutcome(
            index, "temporal_relation", claim["field"], operator, "contradicted",
            f"operator {operator} requires a numeric asserted value",
        )

    if operator == "count_ge":
        holds = observed >= float(asserted)
    elif operator == "count_le":
        holds = observed <= float(asserted)
    elif operator == "within_hours":
        holds = abs(evidence.timestamp - observed) <= float(asserted) * 3600.0
    elif operator == "before":
        holds = observed < evidence.timestamp
    elif operator == "after":
        # "after" relative to a reference that is itself in the past.
        holds = observed > float(asserted) if isinstance(asserted, (int, float)) else False
    else:  # pragma: no cover - the schema restricts the operator set
        return ClaimOutcome(
            index, "temporal_relation", claim["field"], operator, "contradicted",
            f"operator {operator!r} is not admissible for temporal_relation",
        )

    return ClaimOutcome(
        index, "temporal_relation", claim["field"], operator,
        "supported" if holds else "contradicted",
        f"observed {observed!r}; asserted {operator} {asserted!r}",
        observed=observed,
    )


def check_graph_path(
    index: int, claim: dict[str, Any], evidence: TransactionEvidence, rule: dict[str, Any]
) -> ClaimOutcome:
    """Existence or absence of a bounded path in the temporal graph snapshot."""
    if not evidence.graph_available:
        # Manuscript Table 4, E2: an unverifiable graph claim causes rejection.
        return ClaimOutcome(
            index, "graph_path", claim["field"], claim["operator"], "unverifiable",
            "graph evidence is unavailable for this dataset; the claim cannot be checked",
        )
    graph: GraphEvidence = evidence.graph  # type: ignore[assignment]

    value = claim["value"]
    if isinstance(value, dict):
        target = str(value["target"])
        max_hops = int(value.get("max_hops", rule.get("max_hops", 4)))
    elif claim["operator"] == "hops_le" and isinstance(value, (int, float)):
        target = str(claim["field"])
        max_hops = int(value)
    else:
        return ClaimOutcome(
            index, "graph_path", claim["field"], claim["operator"], "contradicted",
            "graph_path claims require an object value with a 'target', or an "
            "integer hop bound for the hops_le operator",
        )

    max_hops = min(max_hops, int(rule.get("max_hops", 4)))
    source = str(evidence.get(claim["evidence_id"]).value) if evidence.has(claim["evidence_id"]) else None
    if source is None:
        source = evidence.transaction_id

    found, hops = graph.path_within(source, target, max_hops)
    operator = claim["operator"]
    if operator == "path_exists":
        holds = found
    elif operator == "path_absent":
        holds = not found
    else:  # hops_le
        holds = found and hops is not None and hops <= max_hops

    return ClaimOutcome(
        index, "graph_path", claim["field"], operator,
        "supported" if holds else "contradicted",
        f"path from {source!r} to {target!r} within {max_hops} hop(s): "
        f"{'found at ' + str(hops) + ' hop(s)' if found else 'not found'} "
        f"(snapshot as of t={graph.as_of_time})",
        observed=hops,
    )


def check_entity_match(
    index: int, claim: dict[str, Any], evidence: TransactionEvidence, rule: dict[str, Any]
) -> ClaimOutcome:
    """Equality between an asserted entity identifier and the recorded one."""
    entry = evidence.get(claim["evidence_id"])
    if entry is None or entry.kind != "entity_field":
        return ClaimOutcome(
            index, "entity_match", claim["field"], claim["operator"], "unverifiable",
            f"no entity evidence entry named {claim['evidence_id']!r}",
        )
    if not isinstance(claim["value"], str):
        return ClaimOutcome(
            index, "entity_match", claim["field"], claim["operator"], "contradicted",
            "entity_match requires a string asserted value",
        )
    observed = str(entry.value)
    asserted = claim["value"]
    if not rule.get("case_sensitive", True):
        observed, asserted = observed.lower(), asserted.lower()
    equal = observed == asserted
    holds = equal if claim["operator"] == "matches" else not equal
    return ClaimOutcome(
        index, "entity_match", claim["field"], claim["operator"],
        "supported" if holds else "contradicted",
        f"observed {entry.value!r}; asserted {claim['operator']} {claim['value']!r}",
        observed=entry.value,
    )


#: Dispatch table wired by :class:`vrfraudnet.verifier.engine.DeterministicVerifier`.
CHECKERS = {
    "numeric_comparison": check_numeric_comparison,
    "set_membership": check_set_membership,
    "temporal_relation": check_temporal_relation,
    "graph_path": check_graph_path,
    "entity_match": check_entity_match,
}
