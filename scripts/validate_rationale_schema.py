"""Validate the rationale JSON schema, the verifier rule set, and the examples.

Usage
-----
    python scripts/validate_rationale_schema.py
    python scripts/validate_rationale_schema.py --json

Checks
------
1. ``schemas/rationale_schema.json`` is a valid JSON Schema (Draft 2020-12).
   Uses ``jsonschema.Draft202012Validator.check_schema`` when the optional
   dependency is installed; otherwise a structural check of the fields the
   runtime depends on.
2. ``verifier/rules/claim_primitives.yaml`` parses and declares exactly the
   five primitives of manuscript S4.3.2; its operator set equals the schema's
   operator enum (no drift in either direction); every primitive has a checker
   implementation; every consistency rule CR1-CR6 is declared.
3. ``examples/rationale_accepted.json`` validates against the schema AND is
   accepted by the deterministic verifier against
   ``examples/evidence_example.json`` (V = 1).
4. ``examples/rationale_rejected.json`` validates against the schema (it is
   well-formed) AND is rejected by the verifier (V = 0), with the audit trace
   naming the failing claims and consistency rules.
5. The decoding grammar built from the schema parses the accepted example as
   COMPLETE and rejects an illegal enum, an illegal key, an illegal operator and
   an incomplete object.

Exit code 0 when every check passes, 1 otherwise. Nothing here reads a dataset
or a model weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrfraudnet.models.grammar import GrammarSpec, ParseState, RationaleGrammar  # noqa: E402
from vrfraudnet.verifier import DeterministicVerifier  # noqa: E402
from vrfraudnet.verifier.evidence import TransactionEvidence  # noqa: E402
from vrfraudnet.verifier.primitives import CHECKERS  # noqa: E402

SCHEMA = REPO_ROOT / "schemas" / "rationale_schema.json"
RULES = REPO_ROOT / "verifier" / "rules" / "claim_primitives.yaml"
EXAMPLES = REPO_ROOT / "examples"
MANUSCRIPT_PRIMITIVES = {
    "numeric_comparison", "set_membership", "temporal_relation", "graph_path", "entity_match",
}
CONSISTENCY_RULES = {"CR1", "CR2", "CR3", "CR4", "CR5", "CR6"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    results: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        results.append({"check": name, "passed": bool(ok), "detail": detail})

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    try:
        import jsonschema

        jsonschema.Draft202012Validator.check_schema(schema)
        record("schema_is_valid_draft_2020_12", True, "jsonschema.check_schema passed")
        validator = jsonschema.Draft202012Validator(schema)

        def schema_errors(obj) -> list[str]:
            return [f"{e.json_path}: {e.message}" for e in validator.iter_errors(obj)]
    except ImportError:
        needed = {"$schema", "type", "required", "properties", "$defs"}
        ok = needed <= set(schema) and schema.get("additionalProperties") is False
        record("schema_is_valid_draft_2020_12", ok,
               "jsonschema not installed; structural check only (pip install jsonschema)")
        verifier_for_schema = DeterministicVerifier()

        def schema_errors(obj) -> list[str]:
            return verifier_for_schema.validate_schema(obj)

    record("schema_version_declared",
           schema.get("$schema", "").endswith("2020-12/schema") and "$id" in schema,
           f"$schema={schema.get('$schema')} $id={schema.get('$id')}")
    record("schema_rejects_free_form_fields",
           schema.get("additionalProperties") is False
           and schema["$defs"]["claim"].get("additionalProperties") is False,
           "additionalProperties is false at the top level and inside every claim")

    rules = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    primitives = rules["primitives"]
    record("rules_declare_five_manuscript_primitives",
           set(primitives) == MANUSCRIPT_PRIMITIVES,
           f"declared: {sorted(primitives)}")
    record("every_primitive_has_a_checker",
           set(primitives) <= set(CHECKERS), f"checkers: {sorted(CHECKERS)}")
    rule_operators = {op for p in primitives.values() for op in p["operators"]}
    schema_operators = set(schema["$defs"]["claim"]["properties"]["operator"]["enum"])
    record("schema_and_verifier_operator_sets_agree",
           rule_operators == schema_operators,
           f"schema-only: {sorted(schema_operators - rule_operators)}; "
           f"rules-only: {sorted(rule_operators - schema_operators)}")
    schema_types = set(schema["$defs"]["claim"]["properties"]["type"]["enum"])
    record("schema_and_verifier_claim_types_agree", schema_types == set(primitives),
           f"schema: {sorted(schema_types)}")
    declared_rules = {r["id"] for r in rules.get("consistency_rules", [])}
    record("consistency_rules_cr1_to_cr6_declared", declared_rules == CONSISTENCY_RULES,
           f"declared: {sorted(declared_rules)}")
    record("float_tolerance_declared",
           "float_tolerance" in primitives["numeric_comparison"],
           f"float_tolerance={primitives['numeric_comparison'].get('float_tolerance')} (L-13)")
    record("unavailable_graph_outcome_declared",
           primitives["graph_path"].get("unavailable_graph_outcome") == "reject",
           "graph_path.unavailable_graph_outcome=reject (manuscript Table 4, E2)")
    record("rules_version_recorded", "version" in rules and "source" in rules,
           f"version={rules.get('version')} source={rules.get('source')!r}")

    verifier = DeterministicVerifier()
    evidence = TransactionEvidence.from_json(
        json.loads((EXAMPLES / "evidence_example.json").read_text(encoding="utf-8"))
    )
    accepted = json.loads((EXAMPLES / "rationale_accepted.json").read_text(encoding="utf-8"))
    rejected = json.loads((EXAMPLES / "rationale_rejected.json").read_text(encoding="utf-8"))
    errs = schema_errors(accepted)
    record("accepted_example_is_schema_valid", not errs, "; ".join(errs) or "no schema errors")
    result = verifier.verify(accepted, evidence)
    record("accepted_example_verifier_status_1", result.status == 1,
           f"V={result.status}; failures={result.failure_reasons}")
    errs = schema_errors(rejected)
    record("rejected_example_is_schema_valid", not errs, "; ".join(errs) or "no schema errors")
    result = verifier.verify(rejected, evidence)
    record("rejected_example_verifier_status_0", result.status == 0,
           f"V={result.status}; failures={result.failure_reasons}")

    grammar = RationaleGrammar(GrammarSpec.from_schema(SCHEMA))
    ordered = json.dumps(
        {"verdict": accepted["verdict"], "rationale_probability": accepted["rationale_probability"],
         "claims": [{k: c[k] for k in ("type", "field", "operator", "value", "evidence_id")}
                    for c in accepted["claims"]],
         "evidence_references": accepted["evidence_references"]},
        separators=(",", ":"),
    )
    record("grammar_accepts_the_accepted_example",
           grammar.parse(ordered) is ParseState.COMPLETE, "ParseState.COMPLETE")
    record("grammar_valid_prefix_is_partial",
           grammar.parse('{"verdict":"fra') is ParseState.PARTIAL, '{"verdict":"fra -> PARTIAL')
    record("grammar_rejects_invalid_enum",
           grammar.parse('{"verdict":"maybe"') is ParseState.INVALID, '"maybe" is not a verdict')
    record("grammar_rejects_illegal_key",
           grammar.parse('{"explanation":') is ParseState.INVALID, '"explanation" is not a key')
    record("grammar_rejects_illegal_operator",
           grammar.parse('{"verdict":"fraud","rationale_probability":0.5,"claims":[{"type":'
                         '"numeric_comparison","field":"amount","operator":"approx"')
           is ParseState.INVALID, '"approx" is not an operator')
    record("grammar_incomplete_object_is_not_complete",
           grammar.parse(ordered[:-1]) is ParseState.PARTIAL, "missing closing brace -> PARTIAL")

    failed = [r for r in results if not r["passed"]]
    if args.json:
        print(json.dumps({"results": results, "passed": not failed}, indent=2))
    else:
        for r in results:
            print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['check']}: {r['detail']}")
        print(f"\n{len(results) - len(failed)} of {len(results)} checks passed.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
