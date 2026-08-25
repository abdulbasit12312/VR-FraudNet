"""Which adversarial edit families are defined for which dataset and model.

Manuscript Section 4.6.2 lists five edit families: transaction splitting,
execution delay, mule-routing modification, prompt injection, and bounded
feature perturbation. Section 5.6 reports an evasion rate for every family on
every model, including LightGBM and TabTransformer.

An edit family is only meaningful when (a) the dataset carries the structure the
edit manipulates and (b) the model under attack actually consumes that
structure. This module states those preconditions explicitly and refuses to
generate an edit that fails them, because an evasion rate for an inapplicable
edit is not a weak result - it is an undefined one.

AUDIT FINDING A-06: prompt injection
------------------------------------
Manuscript S4.6.2 states that "prompt-injection strings are introduced only into
untrusted text fields". None of D1-D5 contains an untrusted free-text field:

* D1 BAF is a tabular account-application dataset of 30 numeric and categorical
  attributes; there is no free-text column.
* D2 AMLworld carries amounts, currencies, bank and account identifiers and a
  payment format; no free text.
* D3 IEEE-CIS is fully anonymised numeric/categorical (``V``, ``C``, ``D``,
  ``M``, ``id_*``); ``P_emaildomain``/``R_emaildomain`` are closed-vocabulary
  categoricals, not attacker-controlled prose.
* D4 Elliptic++ features are anonymised floats.
* D5 DGraph-Fin ships 17 anonymised node attributes.

Moreover, LightGBM, XGBoost, Random Forest, MLP, 1D-CNN, LSTM, TabTransformer,
Logistic Regression and Isolation Forest consume no text at any point, so an
injected string cannot reach them. Table 10(a)-(c) nevertheless report
prompt-injection evasion rates of 0.5234 (LightGBM, B=1) up to 0.8123
(LightGBM, B=4). Those numbers are not reproducible under the protocol the
manuscript describes.

This repository therefore refuses to synthesise a text field in order to make
the experiment run. Running the injection family requires the operator to
declare, per dataset, the name of the real untrusted text field, and requires
the target model to be text-consuming.

AUDIT FINDING A-07: transaction splitting on D1
-----------------------------------------------
BAF is a bank-account-*application* dataset: each row is an application, not a
payment. "Transaction splitting edits preserve the original total amount" has no
validity-preserving interpretation for an application record. The family is
disabled for D1 pending author confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vrfraudnet.errors import NotApplicableError

#: The five edit families of manuscript S4.6.2, using the paper's epsilon labels.
EDIT_FAMILIES: tuple[str, ...] = ("split", "delay", "mule", "inject", "feat")

EPSILON_LABELS: dict[str, str] = {
    "split": "eps1 split",
    "delay": "eps2 delay",
    "mule": "eps3 mule",
    "inject": "eps4 inject",
    "feat": "eps5 feat",
}


@dataclass(frozen=True)
class DatasetCapabilities:
    """Structural facts that determine which edits a dataset can carry."""

    dataset_id: str
    has_transaction_amount: bool
    row_is_a_payment: bool
    has_fine_grained_timestamp: bool
    has_graph: bool
    untrusted_text_field: str | None


DATASET_CAPABILITIES: dict[str, DatasetCapabilities] = {
    "D1": DatasetCapabilities("D1", False, False, False, False, None),
    "D2": DatasetCapabilities("D2", True, True, True, True, None),
    "D3": DatasetCapabilities("D3", True, True, True, False, None),
    "D4": DatasetCapabilities("D4", False, True, True, True, None),
    "D5": DatasetCapabilities("D5", False, False, True, True, None),
}

#: Models that consume text at inference time. Only the Stage 2 rationale
#: pathway does; every tabular baseline in manuscript S4.7 does not.
TEXT_CONSUMING_MODELS: frozenset[str] = frozenset({"vr_fraudnet_stage2", "rationale_llm"})


def check_applicable(
    family: str,
    dataset_id: str,
    model_name: str,
    *,
    declared_text_field: str | None = None,
) -> None:
    """Raise :class:`NotApplicableError` when this edit is undefined here.

    Parameters
    ----------
    declared_text_field:
        Operator-supplied name of a genuine untrusted free-text column. Required
        before the injection family will run at all.
    """
    if family not in EDIT_FAMILIES:
        raise NotApplicableError(f"unknown edit family {family!r}; expected {EDIT_FAMILIES}")
    caps = DATASET_CAPABILITIES.get(dataset_id.upper())
    if caps is None:
        raise NotApplicableError(f"no capability record for dataset {dataset_id!r}")

    if family == "split":
        if not caps.row_is_a_payment or not caps.has_transaction_amount:
            raise NotApplicableError(
                f"transaction splitting is undefined for {dataset_id}: "
                f"row_is_a_payment={caps.row_is_a_payment}, "
                f"has_transaction_amount={caps.has_transaction_amount}. "
                "Splitting must preserve an original total amount across two payment "
                "records (manuscript S4.6.2); see audit finding A-07."
            )
    elif family == "delay":
        if not caps.has_fine_grained_timestamp:
            raise NotApplicableError(
                f"execution delay is undefined for {dataset_id}: the dataset carries no "
                "timestamp finer than its partition granularity, so a delay edit cannot "
                "preserve chronological validity while changing model input."
            )
    elif family == "mule":
        if not caps.has_graph:
            raise NotApplicableError(
                f"mule-routing modification is undefined for {dataset_id}: the dataset "
                "provides no entity graph, so admissible entity and edge types cannot be "
                "drawn from a training graph (manuscript S4.6.2)."
            )
    elif family == "inject":
        text_field = declared_text_field or caps.untrusted_text_field
        if text_field is None:
            raise NotApplicableError(
                f"prompt injection is undefined for {dataset_id}: the dataset contains no "
                "untrusted free-text field into which a payload could be placed "
                "(manuscript S4.6.2). See audit finding A-06. If such a field does exist, "
                "declare it explicitly with --injection-text-field."
            )
        if model_name not in TEXT_CONSUMING_MODELS:
            raise NotApplicableError(
                f"prompt injection is undefined against model {model_name!r}: it consumes "
                "no text, so an injected string cannot reach it. Manuscript Table 10 "
                "reports injection evasion rates for LightGBM and TabTransformer; see "
                "audit finding A-06."
            )


def applicable_families(
    dataset_id: str, model_name: str, *, declared_text_field: str | None = None
) -> tuple[str, ...]:
    """Every edit family that is defined for this (dataset, model) pair."""
    out: list[str] = []
    for family in EDIT_FAMILIES:
        try:
            check_applicable(
                family, dataset_id, model_name, declared_text_field=declared_text_field
            )
        except NotApplicableError:
            continue
        out.append(family)
    return tuple(out)


def applicability_report(
    dataset_ids: Iterable[str], model_names: Iterable[str]
) -> dict[str, dict[str, list[str]]]:
    """Matrix of applicable families, for inclusion in the robustness report."""
    return {
        dataset: {
            model: list(applicable_families(dataset, model)) for model in model_names
        }
        for dataset in dataset_ids
    }
