"""Grammar-masked decoding must exclude every schema-violating token (S4.3.3)."""

from __future__ import annotations

import json

import pytest

from vrfraudnet.models.grammar import GrammarSpec, ParseState, RationaleGrammar


@pytest.fixture(scope="module")
def grammar(repo_root) -> RationaleGrammar:
    return RationaleGrammar(GrammarSpec.from_schema(repo_root / "schemas" / "rationale_schema.json"))


def _valid_object() -> dict:
    return {
        "verdict": "fraud",
        "rationale_probability": 0.81,
        "claims": [
            {"type": "numeric_comparison", "field": "amount", "operator": "gt",
             "value": 5000, "evidence_id": "amount"},
            {"type": "graph_path", "field": "counterparty", "operator": "path_exists",
             "value": {"target": "ACC-1", "max_hops": 2}, "evidence_id": "counterparty"},
        ],
        "evidence_references": ["amount", "counterparty"],
    }


def test_valid_object_parses_as_complete(grammar):
    text = json.dumps(_valid_object(), separators=(",", ":"))
    assert grammar.parse(text) is ParseState.COMPLETE


def test_every_prefix_of_a_valid_object_is_partial(grammar):
    text = json.dumps(_valid_object(), separators=(",", ":"))
    for k in range(1, len(text)):
        assert grammar.parse(text[:k]) is not ParseState.INVALID, f"prefix rejected at {k}: {text[:k]!r}"


@pytest.mark.parametrize(
    "prefix",
    [
        '{"verdict":"maybe"',                       # verdict outside the enum
        '{"verdict":"fraud","rationale_probability":1.5,',   # probability above 1
        '{"verdict":"fraud","rationale_probability":-0.1',   # probability below 0
        '{"verdict":"fraud","rationale_probability":0.5,"notes"',  # unsupported field
        '{"claims"',                                # wrong key order
        '["fraud"]',                                # not an object
    ],
)
def test_invalid_prefixes_are_rejected(grammar, prefix):
    assert grammar.parse(prefix) is ParseState.INVALID


def test_claim_type_enum_is_enforced(grammar):
    prefix = ('{"verdict":"fraud","rationale_probability":0.5,'
              '"claims":[{"type":"vibes"')
    assert grammar.parse(prefix) is ParseState.INVALID


def test_operator_enum_is_enforced(grammar):
    prefix = ('{"verdict":"fraud","rationale_probability":0.5,'
              '"claims":[{"type":"numeric_comparison","field":"amount","operator":"approximately"')
    assert grammar.parse(prefix) is ParseState.INVALID


def test_more_than_eight_claims_is_rejected(grammar):
    claim = ('{"type":"numeric_comparison","field":"a","operator":"gt",'
             '"value":1,"evidence_id":"a"}')
    prefix = ('{"verdict":"fraud","rationale_probability":0.5,"claims":['
              + ",".join([claim] * 8) + ",")
    assert grammar.parse(prefix) is ParseState.INVALID


def test_allowed_characters_narrow_to_the_enum(grammar):
    allowed = grammar.allowed_characters('{"verdict":"')
    assert allowed == {"f", "l", "u"}, allowed


def test_allowed_characters_after_a_partial_verdict(grammar):
    assert grammar.allowed_characters('{"verdict":"fr') == {"a"}


def test_token_mask_blocks_violating_tokens(grammar):
    vocabulary = ['"fraud"', '"maybe"', '"legitimate"', "12345", "}"]
    mask = grammar.token_mask('{"verdict":', vocabulary)
    assert mask == [True, False, True, False, False]


def test_max_hops_upper_bound_is_enforced(grammar):
    prefix = ('{"verdict":"fraud","rationale_probability":0.5,"claims":['
              '{"type":"graph_path","field":"c","operator":"path_exists",'
              '"value":{"target":"A","max_hops":9')
    assert grammar.parse(prefix) is ParseState.INVALID


def test_is_complete_only_for_a_closed_object(grammar):
    text = json.dumps(_valid_object(), separators=(",", ":"))
    assert grammar.is_complete(text)
    assert not grammar.is_complete(text[:-1])
    assert not grammar.is_complete(text + "trailing")
