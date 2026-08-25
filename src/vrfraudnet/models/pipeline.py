"""End-to-end VR-FraudNet inference pipeline (manuscript Section 4, Figure 2).

For each transaction the pipeline returns three artefacts, exactly as the
manuscript specifies in Section 1: a calibrated fraud probability, a structured
rationale when applicable, and a verifier-acceptance flag - plus the routing
decision from Section 4.2.

The control flow encodes the paper's central claim: **the language model
proposes, the verifier disposes**. There is no path by which a rationale-side
probability reaches the Stage 4 mixer without a Stage 3 acceptance, and there is
no path by which a verifier rejection results in an automated action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from vrfraudnet.models.stage1_triage import Route, ThresholdGate, TriageClassifier
from vrfraudnet.models.stage2_rationale import RationaleModel, RationaleOutput
from vrfraudnet.models.stage4_calibration import (
    IsotonicMixer,
    MixerInputs,
    SplitConformalCalibrator,
)
from vrfraudnet.verifier import DeterministicVerifier, TransactionEvidence, VerifierResult


@dataclass
class TransactionDecision:
    """The per-transaction output of VR-FraudNet."""

    transaction_id: str
    calibrated_probability: float
    route: str
    verifier_accepted: bool | None
    rationale: dict[str, Any] | None
    rationale_released: bool
    escalated_to_human: bool
    prediction_set: set[int] = field(default_factory=set)
    audit_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "calibrated_probability": self.calibrated_probability,
            "route": self.route,
            "verifier_accepted": self.verifier_accepted,
            "rationale_released": self.rationale_released,
            "escalated_to_human": self.escalated_to_human,
            "prediction_set": sorted(self.prediction_set),
            "audit_trace": self.audit_trace,
        }


class VRFraudNet:
    """Stages 0-4 wired together.

    Stage 0 is supplied as a precomputed per-node embedding rather than being
    invoked inside this class, because on the large graphs (D2/D5) the spectral
    filters are computed once per evaluation-time snapshot and reused, which is
    also what makes the common path fast enough to measure.
    """

    def __init__(
        self,
        triage: TriageClassifier,
        gate: ThresholdGate,
        verifier: DeterministicVerifier,
        mixer: IsotonicMixer,
        *,
        rationale_model: RationaleModel | None = None,
        conformal: SplitConformalCalibrator | None = None,
    ) -> None:
        self.triage = triage
        self.gate = gate
        self.verifier = verifier
        self.mixer = mixer
        self.rationale_model = rationale_model
        self.conformal = conformal

    # -- Stage 1 ------------------------------------------------------------
    def triage_probabilities(
        self, features: np.ndarray, graph_embedding: np.ndarray | None = None
    ) -> np.ndarray:
        """``P_t = T_theta([x ; z_g])`` (manuscript Eq. 3-4)."""
        stacked = (
            features
            if graph_embedding is None
            else np.hstack([features, graph_embedding])
        )
        return self.triage.predict_proba(stacked)

    # -- Stages 2-4 ---------------------------------------------------------
    def decide(
        self,
        transaction_ids: Sequence[str],
        features: np.ndarray,
        evidence: Sequence[TransactionEvidence] | None = None,
        graph_embedding: np.ndarray | None = None,
        prompts: Sequence[str] | None = None,
    ) -> list[TransactionDecision]:
        """Run the full pipeline over a batch of transactions."""
        triage_p = self.triage_probabilities(features, graph_embedding)
        routes = self.gate.route(triage_p)

        rationale_p = np.zeros_like(triage_p)
        verifier_status = np.full(triage_p.shape, -1, dtype=int)
        rationales: list[dict[str, Any] | None] = [None] * len(triage_p)
        traces: list[dict[str, Any]] = [{} for _ in triage_p]

        for i, route in enumerate(routes):
            if route != Route.ESCALATE.value:
                continue
            output, result = self._escalate(i, evidence, prompts)
            traces[i] = {
                "stage2": {
                    "schema_valid_generation": output.schema_valid_generation if output else False,
                    "parse_error": output.parse_error if output else "stage 2 not available",
                },
                "stage3": result.to_dict() if result else {},
            }
            if output is not None and output.parsed is not None:
                rationales[i] = output.parsed
                if result is not None and result.accepted:
                    rationale_p[i] = float(output.parsed["rationale_probability"])
                    verifier_status[i] = 1
                else:
                    verifier_status[i] = 0
            else:
                verifier_status[i] = 0

        mixer_inputs = MixerInputs(
            triage_probability=triage_p,
            rationale_probability=rationale_p,
            verifier_status=np.where(verifier_status < 0, 1, verifier_status),
        )
        calibrated = self.mixer.predict(mixer_inputs)

        decisions: list[TransactionDecision] = []
        for i, tid in enumerate(transaction_ids):
            escalated = routes[i] == Route.ESCALATE.value
            accepted = None if verifier_status[i] < 0 else bool(verifier_status[i] == 1)
            prediction_set: set[int] = set()
            if self.conformal is not None and self.conformal.threshold_ is not None:
                prediction_set = self.conformal.prediction_set(
                    float(calibrated[i]),
                    verifier_status=None if accepted is None else int(accepted),
                )
            decisions.append(
                TransactionDecision(
                    transaction_id=str(tid),
                    calibrated_probability=float(calibrated[i]),
                    route=str(routes[i]),
                    verifier_accepted=accepted,
                    rationale=rationales[i],
                    # A rationale may only be released when the verifier accepted it.
                    rationale_released=bool(accepted),
                    # Verifier rejection routes to human review, never to automated action.
                    escalated_to_human=bool(escalated and accepted is not True),
                    prediction_set=prediction_set,
                    audit_trace=traces[i],
                )
            )
        return decisions

    def _escalate(
        self,
        index: int,
        evidence: Sequence[TransactionEvidence] | None,
        prompts: Sequence[str] | None,
    ) -> tuple[RationaleOutput | None, VerifierResult | None]:
        """Stage 2 generation followed by Stage 3 verification for one transaction."""
        if self.rationale_model is None or evidence is None or prompts is None:
            return None, None
        output = self.rationale_model.generate(prompts[index])
        if output.parsed is None:
            # Manuscript S4.3.3: unparseable or truncated output IS a verification
            # failure; it is not retried and not repaired.
            return output, VerifierResult(
                accepted=False,
                schema_failures=[output.parse_error or "generation failed"],
            )
        return output, self.verifier.verify(output.parsed, evidence[index])


def escalation_statistics(decisions: Sequence[TransactionDecision]) -> dict[str, float]:
    """Routing and verifier-safeguard rates over a batch of decisions."""
    n = len(decisions)
    if n == 0:
        return {}
    escalated = [d for d in decisions if d.route == Route.ESCALATE.value]
    verified = [d for d in escalated if d.verifier_accepted is not None]
    accepted = [d for d in verified if d.verifier_accepted]
    return {
        "n": float(n),
        "accept_rate": float(sum(d.route == Route.ACCEPT.value for d in decisions) / n),
        "reject_rate": float(sum(d.route == Route.REJECT.value for d in decisions) / n),
        "escalation_rate": float(len(escalated) / n),
        "verifier_acceptance_rate": float(len(accepted) / len(verified)) if verified else float("nan"),
        "unverified_rationale_rate": (
            float((len(verified) - len(accepted)) / len(verified)) if verified else float("nan")
        ),
        "human_review_rate": float(sum(d.escalated_to_human for d in decisions) / n),
    }
