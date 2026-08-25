"""Stage 3: the fixed deterministic verifier (manuscript Section 4.4).

    V(r_i, x_i, G_t) in {0, 1}                                        (Eq. 6)
    V = 1 if all claim checks and consistency rules are satisfied,
    V = 0 otherwise.                                                  (Eq. 7)

Properties this implementation preserves, because they are the paper's central
claim rather than an implementation detail:

* **Fixed and non-trainable.** No parameter of this module is updated by the
  predictive, adversarial or counterfactual objectives (manuscript S4.6.4). The
  rule set lives in ``verifier/rules/claim_primitives.yaml`` and is loaded, not
  learned.
* **Closed primitive set.** A claim whose type is outside the five primitives is
  rejected, never "interpreted".
* **Evidence-bounded.** A claim can only be supported by the current
  transaction's evidence dictionary. A retrieved training example can never be
  cited (manuscript S4.3.4).
* **Deterministic.** Identical inputs give an identical verdict and an identical
  audit trace. There is no sampling anywhere in this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vrfraudnet.errors import ConfigurationError
from vrfraudnet.verifier.primitives import CHECKERS, ClaimOutcome
from vrfraudnet.verifier.evidence import TransactionEvidence

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[3] / "verifier" / "rules" / "claim_primitives.yaml"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "rationale_schema.json"


@dataclass
class VerifierResult:
    """Binary verdict plus the claim-level audit trace (manuscript S4.3.5, S4.4)."""

    accepted: bool
    claim_outcomes: list[ClaimOutcome] = field(default_factory=list)
    consistency_failures: list[str] = field(default_factory=list)
    schema_failures: list[str] = field(default_factory=list)

    @property
    def status(self) -> int:
        """``V`` in {0, 1} as defined by manuscript Eq. 6-7."""
        return 1 if self.accepted else 0

    @property
    def failure_reasons(self) -> list[str]:
        reasons = list(self.schema_failures) + list(self.consistency_failures)
        reasons += [
            f"claim[{o.index}] {o.claim_type}.{o.operator}: {o.status} - {o.detail}"
            for o in self.claim_outcomes
            if not o.passed
        ]
        return reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier_status": self.status,
            "accepted": self.accepted,
            "claims": [
                {
                    "index": o.index,
                    "type": o.claim_type,
                    "field": o.field,
                    "operator": o.operator,
                    "status": o.status,
                    "detail": o.detail,
                }
                for o in self.claim_outcomes
            ],
            "consistency_failures": self.consistency_failures,
            "schema_failures": self.schema_failures,
        }


class DeterministicVerifier:
    """Loads the declarative rule set and evaluates rationales against evidence."""

    def __init__(
        self,
        rules_path: str | Path = DEFAULT_RULES_PATH,
        schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    ) -> None:
        self.rules = _load_rules(rules_path)
        self.schema = _load_schema(schema_path)
        self.primitives: dict[str, dict[str, Any]] = self.rules["primitives"]
        self.risk_direction: dict[str, dict[str, str]] = self.rules.get("risk_direction", {})
        unknown = set(self.primitives) - set(CHECKERS)
        if unknown:
            raise ConfigurationError(
                f"rule file declares primitives with no checker implementation: {sorted(unknown)}"
            )

    # -- public API ---------------------------------------------------------
    def verify(self, rationale: dict[str, Any], evidence: TransactionEvidence) -> VerifierResult:
        """Return ``V(r_i, x_i, G_t)`` and the audit trace."""
        schema_failures = self.validate_schema(rationale)
        if schema_failures:
            # A malformed object cannot be checked claim by claim.
            return VerifierResult(accepted=False, schema_failures=schema_failures)

        outcomes: list[ClaimOutcome] = []
        for i, claim in enumerate(rationale["claims"]):
            outcomes.append(self._check_claim(i, claim, evidence))

        consistency = self._check_consistency(rationale, evidence, outcomes)
        accepted = all(o.passed for o in outcomes) and not consistency
        return VerifierResult(
            accepted=accepted,
            claim_outcomes=outcomes,
            consistency_failures=consistency,
        )

    def validate_schema(self, rationale: Any) -> list[str]:
        """Structural validation against ``schemas/rationale_schema.json``.

        Uses :mod:`jsonschema` when available and a self-contained fallback
        validator otherwise, so the verifier never silently skips this step just
        because an optional dependency is missing.
        """
        try:
            import jsonschema
        except ImportError:
            return _fallback_schema_check(rationale, self.schema)
        validator = jsonschema.Draft202012Validator(self.schema)
        return [
            f"schema: {e.json_path} {e.message}"
            for e in sorted(validator.iter_errors(rationale), key=lambda e: list(e.path))
        ]

    # -- internals ----------------------------------------------------------
    def _check_claim(
        self, index: int, claim: dict[str, Any], evidence: TransactionEvidence
    ) -> ClaimOutcome:
        claim_type = claim.get("type")
        rule = self.primitives.get(claim_type)
        if rule is None:
            return ClaimOutcome(
                index, str(claim_type), str(claim.get("field")), str(claim.get("operator")),
                "contradicted", f"claim type {claim_type!r} is outside the closed primitive set",
            )
        if claim.get("operator") not in rule["operators"]:
            return ClaimOutcome(
                index, str(claim_type), str(claim.get("field")), str(claim.get("operator")),
                "contradicted",
                f"operator {claim.get('operator')!r} is not admissible for {claim_type}; "
                f"admissible operators are {rule['operators']}",
            )
        return CHECKERS[claim_type](index, claim, evidence, rule)

    def _check_consistency(
        self,
        rationale: dict[str, Any],
        evidence: TransactionEvidence,
        outcomes: list[ClaimOutcome],
    ) -> list[str]:
        """Evaluate the cross-claim consistency rules CR1-CR6."""
        failures: list[str] = []
        claims: list[dict[str, Any]] = rationale["claims"]
        references = set(rationale.get("evidence_references", []))

        # CR6: claim count.
        if not 1 <= len(claims) <= 8:
            failures.append("CR6: claim count outside the permitted range [1, 8]")

        # CR2: every evidence_id declared and resolvable.
        for i, claim in enumerate(claims):
            eid = claim.get("evidence_id")
            if eid not in references:
                failures.append(
                    f"CR2: claim[{i}] cites evidence_id {eid!r} which is absent from "
                    f"evidence_references"
                )
            elif not evidence.has(eid):
                failures.append(
                    f"CR2: claim[{i}] cites evidence_id {eid!r} which does not resolve in "
                    f"the transaction's evidence dictionary"
                )

        # CR4: exact duplicates.
        signatures = [
            (c.get("type"), c.get("field"), c.get("operator"), json.dumps(c.get("value"), sort_keys=True))
            for c in claims
        ]
        if len(set(signatures)) != len(signatures):
            failures.append("CR4: duplicate claims detected")

        # CR1: contradictory numeric claims over the same field.
        failures.extend(_contradiction_failures(claims))

        # CR5: past-only temporal claims (the primitive checker already marks
        # these unverifiable; this rule reports them as a consistency failure so
        # the audit trace names the rule the paper describes).
        for outcome in outcomes:
            if outcome.claim_type == "temporal_relation" and "future evidence" in outcome.detail:
                failures.append(f"CR5: claim[{outcome.index}] references future evidence")

        # CR3: verdict must be supported by the verified claims.
        failures.extend(self._verdict_support_failures(rationale["verdict"], claims, outcomes))
        return failures

    def _verdict_support_failures(
        self, verdict: str, claims: list[dict[str, Any]], outcomes: list[ClaimOutcome]
    ) -> list[str]:
        directions = [
            self.risk_direction.get(c["type"], {}).get(c["operator"], "neutral")
            for c, o in zip(claims, outcomes)
            if o.passed
        ]
        increasing = sum(1 for d in directions if d == "increasing")
        if verdict == "fraud" and increasing == 0:
            return [
                "CR3: verdict 'fraud' is not supported by any verified risk-increasing claim"
            ]
        if verdict == "legitimate" and increasing > 0:
            return [
                f"CR3: verdict 'legitimate' contradicts {increasing} verified "
                f"risk-increasing claim(s)"
            ]
        return []


def _contradiction_failures(claims: list[dict[str, Any]]) -> list[str]:
    """CR1: detect mutually unsatisfiable numeric bounds on the same field."""
    failures: list[str] = []
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for claim in claims:
        if claim.get("type") != "numeric_comparison":
            continue
        value = claim.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        field_name = str(claim.get("field"))
        op = claim.get("operator")
        if op in {"gt", "ge"}:
            lower[field_name] = max(lower.get(field_name, float("-inf")), float(value))
        elif op in {"lt", "le"}:
            upper[field_name] = min(upper.get(field_name, float("inf")), float(value))
    for field_name in set(lower) & set(upper):
        if lower[field_name] >= upper[field_name]:
            failures.append(
                f"CR1: claims on field {field_name!r} assert an empty interval "
                f"({lower[field_name]} .. {upper[field_name]})"
            )
    return failures


def _load_rules(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"verifier rule file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        rules = yaml.safe_load(handle)
    if "primitives" not in rules:
        raise ConfigurationError(f"{path} does not declare a 'primitives' section")
    return rules


def _load_schema(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"rationale schema not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fallback_schema_check(rationale: Any, schema: dict[str, Any]) -> list[str]:
    """Minimal structural validation used when jsonschema is not installed.

    Deliberately conservative: it checks the constraints the verifier depends on
    (required keys, enums, ranges, claim count, claim shape) rather than the
    whole JSON Schema specification.
    """
    failures: list[str] = []
    if not isinstance(rationale, dict):
        return ["schema: rationale must be a JSON object"]

    for key in schema["required"]:
        if key not in rationale:
            failures.append(f"schema: missing required field {key!r}")
    if failures:
        return failures

    verdict_enum = schema["properties"]["verdict"]["enum"]
    if rationale["verdict"] not in verdict_enum:
        failures.append(f"schema: verdict {rationale['verdict']!r} not in {verdict_enum}")

    probability = rationale["rationale_probability"]
    if not isinstance(probability, (int, float)) or isinstance(probability, bool):
        failures.append("schema: rationale_probability must be a number")
    elif not 0.0 <= float(probability) <= 1.0:
        failures.append(f"schema: rationale_probability {probability} outside [0, 1]")

    claims = rationale["claims"]
    if not isinstance(claims, list) or not 1 <= len(claims) <= 8:
        failures.append("schema: claims must be a list of between 1 and 8 items")
        return failures

    claim_required = schema["$defs"]["claim"]["required"]
    type_enum = schema["$defs"]["claim"]["properties"]["type"]["enum"]
    operator_enum = schema["$defs"]["claim"]["properties"]["operator"]["enum"]
    for i, claim in enumerate(claims):
        if not isinstance(claim, dict):
            failures.append(f"schema: claims[{i}] must be an object")
            continue
        for key in claim_required:
            if key not in claim:
                failures.append(f"schema: claims[{i}] missing required field {key!r}")
        if claim.get("type") not in type_enum:
            failures.append(f"schema: claims[{i}].type {claim.get('type')!r} not in {type_enum}")
        if claim.get("operator") not in operator_enum:
            failures.append(f"schema: claims[{i}].operator {claim.get('operator')!r} is invalid")
        extra = set(claim) - set(schema["$defs"]["claim"]["properties"])
        if extra:
            failures.append(f"schema: claims[{i}] has unsupported field(s) {sorted(extra)}")

    references = rationale["evidence_references"]
    if not isinstance(references, list) or not references:
        failures.append("schema: evidence_references must be a non-empty list")

    extra_top = set(rationale) - set(schema["properties"])
    if extra_top:
        failures.append(f"schema: unsupported top-level field(s) {sorted(extra_top)}")
    return failures
