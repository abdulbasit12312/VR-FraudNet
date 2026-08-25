"""Per-transaction latency measurement (manuscript Section 4.8.2).

Manuscript protocol, reproduced exactly
---------------------------------------
* Batch size 1, representing transaction-level online inference.
* Data loading and offline preprocessing excluded; model computation, threshold
  routing, retrieval, constrained generation, deterministic verification and
  probability mixing included in whichever pathway they belong to.
* Common path = Stage 0 (when applicable), Stage 1 triage, routing, and Stage 4
  probability processing, with no language-model invocation.
* Escalation path additionally includes retrieval, Stage 2 generation and
  Stage 3 verification.
* Retrieval index and model weights loaded into memory before measurement.
* 1,000 warm-up inferences, then 10,000 timed transactions per component.
* GPU synchronisation immediately before and after every timed inference.
* Percentiles computed from individual transaction times, not batch averages.

AUDIT FINDING A-17: the reported escalation-path p99 is not consistent with the
reported generation length
--------------------------------------------------------------------------------
The manuscript reports a p99 of 175.23 ms on the escalation path, which includes
autoregressive generation from Meta-Llama-3.1-8B-Instruct at batch size 1 with a
384-token output cap and greedy decoding. An 8B-parameter model in bfloat16 on a
single A100-80GB sustains roughly 30-60 tokens/s in single-stream decoding
(memory-bandwidth bound: ~16 GB of weights read per token against ~2 TB/s of
HBM gives a floor near 8 ms/token before any framework overhead). Producing even
a short rationale object of 60-80 tokens would therefore take on the order of
1-3 s, an order of magnitude above the reported p99, before adding retrieval and
verification.

Reproducing 175.23 ms would require at least one of: a much smaller generator, a
much shorter output, speculative decoding, an optimised serving stack such as
vLLM or TensorRT-LLM, or batching - none of which the manuscript describes.
This module therefore records the *measured token count* alongside every timing,
so any future measurement is self-explaining, and
:func:`check_escalation_plausibility` reports the implied tokens-per-second so a
reader can see immediately whether a number is physically attainable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

#: Manuscript S4.8.2.
N_WARMUP = 1_000
N_TIMED = 10_000

#: Manuscript Figure 1: "<= 200 ms" stated latency budget.
LATENCY_BUDGET_MS = 200.0


@dataclass
class LatencyMeasurement:
    """Timing statistics for one pathway."""

    pathway: str
    n_timed: int
    n_warmup: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    tokens_generated_mean: float | None = None
    hardware: dict[str, Any] = field(default_factory=dict)

    @property
    def within_budget(self) -> bool:
        return self.p99_ms <= LATENCY_BUDGET_MS

    @property
    def implied_tokens_per_second(self) -> float | None:
        """Decoding throughput implied by this measurement, if tokens were counted."""
        if not self.tokens_generated_mean or self.mean_ms <= 0:
            return None
        return float(self.tokens_generated_mean / (self.mean_ms / 1000.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathway": self.pathway,
            "n_timed": self.n_timed,
            "n_warmup": self.n_warmup,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
            "tokens_generated_mean": self.tokens_generated_mean,
            "implied_tokens_per_second": self.implied_tokens_per_second,
            "latency_budget_ms": LATENCY_BUDGET_MS,
            "within_budget": self.within_budget,
            "hardware": self.hardware,
        }


def _synchronise() -> None:
    """GPU synchronisation before and after every timed inference (S4.8.2)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover
        pass


def measure_pathway(
    pathway: str,
    inference: Callable[[int], Any],
    *,
    n_timed: int = N_TIMED,
    n_warmup: int = N_WARMUP,
    token_counter: Callable[[Any], int] | None = None,
    hardware: dict[str, Any] | None = None,
) -> LatencyMeasurement:
    """Time ``inference`` once per transaction at batch size 1.

    ``inference(i)`` must perform the full pathway for transaction ``i`` and
    return its output; ``token_counter`` extracts the number of generated tokens
    from that output when the pathway includes generation.
    """
    for i in range(n_warmup):
        inference(i % max(1, n_timed))

    durations = np.empty(n_timed, dtype=float)
    token_counts: list[int] = []
    for i in range(n_timed):
        _synchronise()
        start = time.perf_counter()
        output = inference(i)
        _synchronise()
        durations[i] = (time.perf_counter() - start) * 1000.0
        if token_counter is not None:
            token_counts.append(int(token_counter(output)))

    return LatencyMeasurement(
        pathway=pathway,
        n_timed=n_timed,
        n_warmup=n_warmup,
        mean_ms=float(durations.mean()),
        p50_ms=float(np.percentile(durations, 50)),
        p95_ms=float(np.percentile(durations, 95)),
        p99_ms=float(np.percentile(durations, 99)),
        max_ms=float(durations.max()),
        tokens_generated_mean=float(np.mean(token_counts)) if token_counts else None,
        hardware=hardware or collect_hardware_facts(),
    )


def collect_hardware_facts() -> dict[str, Any]:
    """Record the hardware a latency measurement was taken on.

    Manuscript S4.8.2 states the reference configuration as an AMD EPYC 7543
    32-core CPU, 256 GB RAM and one NVIDIA A100 80GB. Latency numbers are only
    interpretable next to the machine that produced them.
    """
    import platform

    facts: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            facts["gpu"] = torch.cuda.get_device_name(0)
            facts["cuda"] = torch.version.cuda
    except ImportError:  # pragma: no cover
        facts["torch"] = "not-installed"
    return facts


def check_escalation_plausibility(
    p99_ms: float, tokens_generated: float, *, model_parameters_b: float = 8.0
) -> dict[str, Any]:
    """Report whether an escalation-path latency is physically attainable.

    Uses the memory-bandwidth floor for single-stream autoregressive decoding:
    each token requires reading the full weight matrix once, so

        ms_per_token_floor = (2 bytes/param * params) / HBM_bandwidth

    For an 8B model in bfloat16 on an A100-80GB (about 2.0 TB/s) this is roughly
    8 ms per token before any framework overhead. Returns the implied rate and a
    verdict; it does not attempt to be exact, only to distinguish "plausible"
    from "off by an order of magnitude".
    """
    bytes_per_param = 2.0
    hbm_bandwidth_bytes_per_s = 2.0e12
    floor_ms_per_token = (bytes_per_param * model_parameters_b * 1e9) / hbm_bandwidth_bytes_per_s * 1000.0
    implied_ms_per_token = p99_ms / max(tokens_generated, 1.0)
    return {
        "p99_ms": p99_ms,
        "tokens_generated": tokens_generated,
        "implied_ms_per_token": implied_ms_per_token,
        "bandwidth_floor_ms_per_token": floor_ms_per_token,
        "plausible": implied_ms_per_token >= floor_ms_per_token,
        "note": (
            "Implied decoding speed is faster than the memory-bandwidth floor for a "
            f"{model_parameters_b:.0f}B bfloat16 model at batch size 1. Reproducing it "
            "requires a smaller generator, a shorter output, speculative decoding, or an "
            "optimised serving stack. See audit finding A-17."
            if implied_ms_per_token < floor_ms_per_token
            else "Implied decoding speed is above the bandwidth floor."
        ),
    }
