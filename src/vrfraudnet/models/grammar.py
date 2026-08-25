"""Grammar-masked decoding for the Stage 2 rationale schema (manuscript S4.3.3).

    "The predefined JSON schema is converted into a deterministic decoding
     grammar that represents the permitted object structure, field names, data
     types, verdict labels, claim categories, operators, and numerical ranges.
     At each generation step, an incremental parser tracks the current grammar
     state. Tokens that would violate JSON syntax, introduce an unsupported
     field, produce an invalid categorical value, or break the required object
     structure are masked from the model output."

Implementation
--------------
A hand-written incremental parser decides, for any generated prefix, whether
that prefix can still be extended into a schema-valid object. The token mask at
each step is then simply the set of vocabulary tokens whose surface form keeps
the prefix extendable. This is the standard construction, and it makes the
masking behaviour directly testable without a language model in the loop
(``tests/test_grammar_masking.py``).

The manuscript does not name a grammar-masking library, so this repository
ships its own rather than pretending a particular third-party backend was used
(docs/KNOWN_LIMITATIONS.md L-07).

ASSUMPTION L-15: the grammar fixes the order of object keys
(``verdict``, ``rationale_probability``, ``claims``, ``evidence_references``,
and within a claim ``type``, ``field``, ``operator``, ``value``,
``evidence_id``). A fixed key order is the usual choice for constrained
decoding and strictly narrows the accepted language; the manuscript does not
state whether key order was constrained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

WHITESPACE = " \t\n\r"


class ParseState(str, Enum):
    """Outcome of parsing a generated prefix."""

    INVALID = "invalid"
    PARTIAL = "partial"
    COMPLETE = "complete"


@dataclass(frozen=True)
class GrammarSpec:
    """The enumerations and bounds the grammar enforces, loaded from the schema."""

    verdicts: tuple[str, ...]
    claim_types: tuple[str, ...]
    operators: tuple[str, ...]
    max_claims: int
    min_claims: int
    max_evidence_refs: int

    @classmethod
    def from_schema(cls, schema_path: str | Path) -> "GrammarSpec":
        with Path(schema_path).open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        claim = schema["$defs"]["claim"]["properties"]
        return cls(
            verdicts=tuple(schema["properties"]["verdict"]["enum"]),
            claim_types=tuple(claim["type"]["enum"]),
            operators=tuple(claim["operator"]["enum"]),
            max_claims=int(schema["properties"]["claims"]["maxItems"]),
            min_claims=int(schema["properties"]["claims"]["minItems"]),
            max_evidence_refs=int(schema["properties"]["evidence_references"]["maxItems"]),
        )


class RationaleGrammar:
    """Incremental parser over the fixed rationale-object grammar."""

    def __init__(self, spec: GrammarSpec) -> None:
        self.spec = spec

    # -- public API ---------------------------------------------------------
    def parse(self, text: str) -> ParseState:
        """Classify a generated prefix as invalid, partial, or complete."""
        rest = _skip_ws(text)
        state, rest = self._object(rest)
        if state is ParseState.INVALID:
            return ParseState.INVALID
        if state is ParseState.PARTIAL:
            return ParseState.PARTIAL
        return ParseState.COMPLETE if _skip_ws(rest) == "" else ParseState.INVALID

    def allowed_characters(self, prefix: str, alphabet: str | None = None) -> set[str]:
        """Characters that may legally follow ``prefix``."""
        alphabet = alphabet or _DEFAULT_ALPHABET
        return {c for c in alphabet if self.parse(prefix + c) is not ParseState.INVALID}

    def token_mask(self, prefix: str, vocabulary: Sequence[str]) -> list[bool]:
        """Boolean mask over a token vocabulary for the current decoding step.

        ``True`` means the token may be emitted. A token is admissible when the
        extended prefix is still a valid prefix of the grammar; the end-of-object
        condition is handled by the caller, which stops when
        :meth:`parse` returns :attr:`ParseState.COMPLETE`.
        """
        return [self.parse(prefix + token) is not ParseState.INVALID for token in vocabulary]

    def is_complete(self, text: str) -> bool:
        return self.parse(text) is ParseState.COMPLETE

    # -- grammar productions ------------------------------------------------
    def _object(self, s: str) -> tuple[ParseState, str]:
        if s == "":
            return ParseState.PARTIAL, s
        if s[0] != "{":
            return ParseState.INVALID, s
        s = _skip_ws(s[1:])

        for i, (key, parser) in enumerate(
            (
                ("verdict", lambda t: _enum_string(t, self.spec.verdicts)),
                ("rationale_probability", lambda t: _number(t, 0.0, 1.0)),
                ("claims", self._claims_array),
                ("evidence_references", self._evidence_array),
            )
        ):
            if i > 0:
                state, s = _literal(s, ",")
                if state is not ParseState.COMPLETE:
                    return state, s
                s = _skip_ws(s)
            state, s = _exact_string(s, key)
            if state is not ParseState.COMPLETE:
                return state, s
            s = _skip_ws(s)
            state, s = _literal(s, ":")
            if state is not ParseState.COMPLETE:
                return state, s
            s = _skip_ws(s)
            state, s = parser(s)
            if state is not ParseState.COMPLETE:
                return state, s
            s = _skip_ws(s)

        return _literal(s, "}")

    def _claims_array(self, s: str) -> tuple[ParseState, str]:
        return _array(s, self._claim_object, self.spec.min_claims, self.spec.max_claims)

    def _evidence_array(self, s: str) -> tuple[ParseState, str]:
        return _array(s, _any_string, 1, self.spec.max_evidence_refs)

    def _claim_object(self, s: str) -> tuple[ParseState, str]:
        if s == "":
            return ParseState.PARTIAL, s
        if s[0] != "{":
            return ParseState.INVALID, s
        s = _skip_ws(s[1:])
        fields = (
            ("type", lambda t: _enum_string(t, self.spec.claim_types)),
            ("field", _any_string),
            ("operator", lambda t: _enum_string(t, self.spec.operators)),
            ("value", _claim_value),
            ("evidence_id", _any_string),
        )
        for i, (key, parser) in enumerate(fields):
            if i > 0:
                state, s = _literal(s, ",")
                if state is not ParseState.COMPLETE:
                    return state, s
                s = _skip_ws(s)
            state, s = _exact_string(s, key)
            if state is not ParseState.COMPLETE:
                return state, s
            s = _skip_ws(s)
            state, s = _literal(s, ":")
            if state is not ParseState.COMPLETE:
                return state, s
            s = _skip_ws(s)
            state, s = parser(s)
            if state is not ParseState.COMPLETE:
                return state, s
            s = _skip_ws(s)
        return _literal(s, "}")


# ---------------------------------------------------------------------------
# Primitive production helpers. Each returns (state, remaining_text).
# ---------------------------------------------------------------------------

_DEFAULT_ALPHABET = (
    "{}[]:,\"" + "abcdefghijklmnopqrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "0123456789.-" + " "
)


def _skip_ws(s: str) -> str:
    i = 0
    while i < len(s) and s[i] in WHITESPACE:
        i += 1
    return s[i:]


def _literal(s: str, token: str) -> tuple[ParseState, str]:
    if s == "":
        return ParseState.PARTIAL, s
    if s.startswith(token):
        return ParseState.COMPLETE, s[len(token) :]
    return ParseState.INVALID, s


def _exact_string(s: str, literal: str) -> tuple[ParseState, str]:
    """Match a quoted literal key, allowing any proper prefix of it."""
    target = f'"{literal}"'
    if target.startswith(s):
        return (ParseState.COMPLETE, "") if s == target else (ParseState.PARTIAL, "")
    if s.startswith(target):
        return ParseState.COMPLETE, s[len(target) :]
    return ParseState.INVALID, s


def _enum_string(s: str, options: Iterable[str]) -> tuple[ParseState, str]:
    """Match a quoted value drawn from a closed enumeration."""
    if s == "":
        return ParseState.PARTIAL, s
    if s[0] != '"':
        return ParseState.INVALID, s
    partial = False
    for option in options:
        target = f'"{option}"'
        if s.startswith(target):
            return ParseState.COMPLETE, s[len(target) :]
        if target.startswith(s):
            partial = True
    return (ParseState.PARTIAL, s) if partial else (ParseState.INVALID, s)


def _any_string(s: str) -> tuple[ParseState, str]:
    """Match a JSON string over the identifier alphabet permitted by the schema."""
    if s == "":
        return ParseState.PARTIAL, s
    if s[0] != '"':
        return ParseState.INVALID, s
    i = 1
    while i < len(s):
        ch = s[i]
        if ch == '"':
            return ParseState.COMPLETE, s[i + 1 :]
        if not (ch.isalnum() or ch in "_.:- "):
            return ParseState.INVALID, s
        if i > 128:
            return ParseState.INVALID, s
        i += 1
    return ParseState.PARTIAL, s


def _number(s: str, low: float | None = None, high: float | None = None) -> tuple[ParseState, str]:
    """Match a JSON number, enforcing the schema's numeric range when complete."""
    if s == "":
        return ParseState.PARTIAL, s
    i = 0
    if s[0] == "-":
        i = 1
    seen_digit = False
    seen_dot = False
    while i < len(s):
        ch = s[i]
        if ch.isdigit():
            seen_digit = True
        elif ch == "." and seen_digit and not seen_dot:
            seen_dot = True
        else:
            break
        i += 1
    if not seen_digit:
        return (ParseState.PARTIAL, s) if i >= len(s) else (ParseState.INVALID, s)
    if i >= len(s):
        # The literal may still be extended, so it cannot be range-checked
        # exactly. It CAN be bounded: appending digits to a positive literal
        # never decreases it, and appending digits to a negative literal never
        # increases it. That is enough to reject a prefix no extension could
        # rescue (for example "-0.1" against a [0, 1] range).
        partial_value = float(s[:i]) if not s[:i].endswith(".") else float(s[: i - 1])
        if partial_value < 0:
            if low is not None and partial_value < low:
                return ParseState.INVALID, s
        else:
            if high is not None and partial_value > high:
                return ParseState.INVALID, s
        return ParseState.PARTIAL, s
    literal = s[:i]
    if literal.endswith("."):
        return ParseState.INVALID, s
    value = float(literal)
    if low is not None and value < low:
        return ParseState.INVALID, s
    if high is not None and value > high:
        return ParseState.INVALID, s
    return ParseState.COMPLETE, s[i:]


def _claim_value(s: str) -> tuple[ParseState, str]:
    """Match the polymorphic ``value`` field of a claim."""
    if s == "":
        return ParseState.PARTIAL, s
    head = s[0]
    if head == '"':
        return _any_string(s)
    if head == "[":
        return _array(s, _scalar, 0, 32)
    if head == "{":
        return _graph_target_object(s)
    if head == "t":
        return _keyword(s, "true")
    if head == "f":
        return _keyword(s, "false")
    if head == "-" or head.isdigit():
        return _number(s)
    return ParseState.INVALID, s


def _scalar(s: str) -> tuple[ParseState, str]:
    if s == "":
        return ParseState.PARTIAL, s
    if s[0] == '"':
        return _any_string(s)
    return _number(s)


def _keyword(s: str, word: str) -> tuple[ParseState, str]:
    if word.startswith(s):
        return (ParseState.COMPLETE, "") if s == word else (ParseState.PARTIAL, "")
    if s.startswith(word):
        return ParseState.COMPLETE, s[len(word) :]
    return ParseState.INVALID, s


def _graph_target_object(s: str) -> tuple[ParseState, str]:
    """Match ``{"target": "...", "max_hops": N}`` with ``max_hops`` optional."""
    if s == "":
        return ParseState.PARTIAL, s
    if s[0] != "{":
        return ParseState.INVALID, s
    s = _skip_ws(s[1:])
    state, s = _exact_string(s, "target")
    if state is not ParseState.COMPLETE:
        return state, s
    s = _skip_ws(s)
    state, s = _literal(s, ":")
    if state is not ParseState.COMPLETE:
        return state, s
    s = _skip_ws(s)
    state, s = _any_string(s)
    if state is not ParseState.COMPLETE:
        return state, s
    s = _skip_ws(s)
    if s == "":
        return ParseState.PARTIAL, s
    if s[0] == ",":
        s = _skip_ws(s[1:])
        state, s = _exact_string(s, "max_hops")
        if state is not ParseState.COMPLETE:
            return state, s
        s = _skip_ws(s)
        state, s = _literal(s, ":")
        if state is not ParseState.COMPLETE:
            return state, s
        s = _skip_ws(s)
        state, s = _number(s, 1.0, 4.0)
        if state is not ParseState.COMPLETE:
            return state, s
        s = _skip_ws(s)
    return _literal(s, "}")


def _array(s: str, item_parser, min_items: int, max_items: int) -> tuple[ParseState, str]:
    """Match a JSON array with a bounded number of items."""
    if s == "":
        return ParseState.PARTIAL, s
    if s[0] != "[":
        return ParseState.INVALID, s
    s = _skip_ws(s[1:])
    if s == "":
        return ParseState.PARTIAL, s
    count = 0
    if s[0] == "]":
        return (ParseState.COMPLETE, s[1:]) if min_items == 0 else (ParseState.INVALID, s)
    while True:
        state, s = item_parser(s)
        if state is not ParseState.COMPLETE:
            return state, s
        count += 1
        if count > max_items:
            return ParseState.INVALID, s
        s = _skip_ws(s)
        if s == "":
            return ParseState.PARTIAL, s
        if s[0] == ",":
            if count == max_items:
                return ParseState.INVALID, s
            s = _skip_ws(s[1:])
            if s == "":
                return ParseState.PARTIAL, s
            continue
        if s[0] == "]":
            if count < min_items:
                return ParseState.INVALID, s
            return ParseState.COMPLETE, s[1:]
        return ParseState.INVALID, s
