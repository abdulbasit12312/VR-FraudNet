"""Stage 2: schema-constrained rationale language model (manuscript S4.3).

Manuscript specification
------------------------
* Base model: ``Meta-Llama-3.1-8B-Instruct``, frozen, adapted with LoRA on the
  query, key, value and output projections; rank 16, alpha 32, dropout 0.05.
* Fine-tuned for 3 epochs with AdamW, learning rate 2e-4, weight decay 0.01,
  linear warm-up over the first 5% of steps, max input 1024 tokens, max output
  384 tokens, bfloat16, effective batch size 32 via gradient accumulation.
* Checkpoint selected on validation loss, with schema-valid generation rate and
  verifier acceptance rate as secondary criteria.
* Trained only on review-band transactions from the corresponding training
  partition. Targets that are malformed, internally inconsistent, or rejected by
  the verifier are excluded from training.
* Inference: greedy decoding, temperature 0, sampling disabled; generation stops
  at a complete JSON object or the 384-token limit; incomplete or unparseable
  output is treated as a verification failure.

Artefact status
---------------
No LoRA adapter is distributed with this repository, and none is fabricated.
:class:`RationaleModel` loads an adapter from a path the operator supplies; if
that path does not exist it raises :class:`~vrfraudnet.errors.MissingArtefactError`
naming the training command. Everything that does *not* require weights - the
prompt construction, the schema, the grammar mask, the stopping rule, the
failure handling - is implemented and unit-tested here.

The manuscript's Section 4.3.1 states the learning rate twice, once as
"2 x 10^4" (Section 4.3.1) and once as "0.0002" (Section 4.8.1). The former is
a typesetting artefact of ``2e-4``; the value used is 2e-4. Recorded as audit
note A-11.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from vrfraudnet.errors import MissingArtefactError
from vrfraudnet.models.grammar import GrammarSpec, ParseState, RationaleGrammar


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA settings from manuscript S4.3.1 / S4.8.1."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class Stage2Config:
    """Stage 2 training and decoding configuration."""

    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_input_tokens: int = 1024
    max_output_tokens: int = 384
    effective_batch_size: int = 32
    precision: str = "bfloat16"
    temperature: float = 0.0
    do_sample: bool = False
    n_retrieved_examples: int = 4  # MANUSCRIPT S4.3.4


@dataclass
class RationaleOutput:
    """A Stage 2 generation together with its own failure status.

    Stage 2 never judges its own validity beyond parseability; the object is
    handed unchanged to the Stage 3 verifier (manuscript S4.3.5).
    """

    raw_text: str
    parsed: dict[str, Any] | None
    truncated: bool
    parse_error: str | None = None

    @property
    def schema_valid_generation(self) -> bool:
        """Whether decoding produced a complete, parseable object.

        Reported as the "schema-valid generation rate" used as a secondary
        checkpoint-selection criterion (manuscript S4.3.1).
        """
        return self.parsed is not None and not self.truncated


def build_prompt(
    evidence_summary: str,
    triage_probability: float,
    timestamp: float,
    retrieved_examples: Sequence[str],
    *,
    schema_hint: str,
) -> str:
    """Assemble the Stage 2 prompt (manuscript S4.3, S4.3.4).

    Order matters and follows the manuscript: structured evidence for the
    current transaction first, retrieved demonstrations after it. Retrieved
    examples are labelled as demonstrations of *form*, never as evidence, which
    is the textual counterpart of the verifier rule that a retrieved example can
    never be cited as an ``evidence_id``.
    """
    blocks = [
        "You are a fraud-analysis rationale generator.",
        "Emit exactly one JSON object conforming to the schema below. Emit nothing else.",
        "Every claim must cite an evidence_id from the CURRENT transaction's evidence "
        "dictionary. Retrieved examples below illustrate output form only and must "
        "never be cited as evidence.",
        f"SCHEMA:\n{schema_hint}",
        f"CURRENT TRANSACTION EVIDENCE:\n{evidence_summary}",
        f"TRIAGE PROBABILITY: {triage_probability:.6f}",
        f"TRANSACTION TIMESTAMP: {timestamp}",
    ]
    if retrieved_examples:
        blocks.append(
            "DEMONSTRATIONS (form only, not evidence):\n" + "\n".join(retrieved_examples)
        )
    blocks.append("OUTPUT:")
    return "\n\n".join(blocks)


class RationaleModel:
    """Wrapper around the LoRA-adapted rationale generator.

    The class is deliberately thin. Its job is to (a) refuse to run without real
    weights, (b) enforce grammar-masked greedy decoding, and (c) convert a
    generation failure into the exact status the manuscript defines.
    """

    def __init__(
        self,
        config: Stage2Config,
        schema_path: str | Path,
        *,
        adapter_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.schema_path = Path(schema_path)
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.grammar = RationaleGrammar(GrammarSpec.from_schema(self.schema_path))
        self._model = None
        self._tokenizer = None

    def load(self) -> "RationaleModel":
        """Load the base model and LoRA adapter.

        Raises
        ------
        vrfraudnet.errors.MissingArtefactError
            When no adapter path was supplied or the path does not exist. This
            repository ships no weights and creates none.
        """
        if self.adapter_path is None or not self.adapter_path.exists():
            raise MissingArtefactError(
                "no Stage 2 LoRA adapter available at "
                f"{self.adapter_path!r}.\n"
                "This repository does not distribute model weights and will not "
                "fabricate them. Train an adapter with:\n"
                "    python scripts/train.py --dataset D1 --config configs/d1.yaml "
                "--seed 42 --stage stage2\n"
                "or point --adapter at an adapter you trained yourself."
            )
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingArtefactError(
                "transformers and peft are required for Stage 2. Install them with:\n"
                "    pip install -r requirements-optional.txt"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            self.config.base_model, torch_dtype=self.config.precision
        )
        self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self._model.eval()
        return self

    def generate(self, prompt: str) -> RationaleOutput:
        """Greedy, grammar-masked decoding of one rationale object."""
        if self._model is None or self._tokenizer is None:
            raise MissingArtefactError(
                "RationaleModel.generate called before load(); no weights are loaded"
            )
        return self._decode(prompt)

    # -- decoding -----------------------------------------------------------
    def _decode(self, prompt: str) -> RationaleOutput:  # pragma: no cover - needs weights
        import torch

        tokenizer, model = self._tokenizer, self._model
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=self.config.max_input_tokens).input_ids
        vocabulary = _decoded_vocabulary(tokenizer)
        generated = ""
        emitted = 0
        while emitted < self.config.max_output_tokens:
            with torch.no_grad():
                logits = model(input_ids).logits[0, -1]
            mask = torch.tensor(self.grammar.token_mask(generated, vocabulary))
            logits = logits.masked_fill(~mask, float("-inf"))
            token_id = int(torch.argmax(logits))
            piece = vocabulary[token_id]
            generated += piece
            emitted += 1
            input_ids = torch.cat(
                [input_ids, torch.tensor([[token_id]], device=input_ids.device)], dim=1
            )
            if self.grammar.parse(generated) is ParseState.COMPLETE:
                return finalise_generation(generated, truncated=False)
        return finalise_generation(generated, truncated=True)


def finalise_generation(text: str, *, truncated: bool) -> RationaleOutput:
    """Convert raw decoded text into a :class:`RationaleOutput`.

    Manuscript S4.3.3: "Outputs that remain incomplete or cannot be parsed are
    treated as verification failures."
    """
    if truncated:
        return RationaleOutput(
            raw_text=text, parsed=None, truncated=True,
            parse_error="generation reached the 384-token output limit before completing",
        )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return RationaleOutput(text, None, False, f"JSON decode error: {exc}")
    if not isinstance(parsed, dict):
        return RationaleOutput(text, None, False, "top-level value is not a JSON object")
    return RationaleOutput(text, parsed, False, None)


def _decoded_vocabulary(tokenizer) -> list[str]:  # pragma: no cover - needs a tokenizer
    """Surface forms of every vocabulary token, used for grammar masking."""
    size = len(tokenizer)
    return [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(size)]


def schema_hint(schema_path: str | Path) -> str:
    """Compact schema rendering embedded in the prompt."""
    with Path(schema_path).open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    claim = schema["$defs"]["claim"]["properties"]
    return json.dumps(
        {
            "verdict": schema["properties"]["verdict"]["enum"],
            "rationale_probability": "number in [0, 1]",
            "claims": {
                "maxItems": schema["properties"]["claims"]["maxItems"],
                "item": {
                    "type": claim["type"]["enum"],
                    "field": "string",
                    "operator": claim["operator"]["enum"],
                    "value": "number | string | boolean | array | {target, max_hops}",
                    "evidence_id": "string",
                },
            },
            "evidence_references": ["string"],
        },
        indent=None,
    )


def filter_training_targets(
    targets: Sequence[dict[str, Any]],
    verifier,
    evidence_lookup,
) -> list[dict[str, Any]]:
    """Drop malformed, inconsistent or verifier-rejected training targets (S4.3.1).

    Returns only the targets the deterministic verifier accepts against their own
    evidence, which is what "Targets that are malformed, internally inconsistent,
    or rejected by the deterministic verifier are excluded from training" means
    operationally.
    """
    kept: list[dict[str, Any]] = []
    for target in targets:
        evidence = evidence_lookup(target)
        if evidence is None:
            continue
        if verifier.verify(target, evidence).accepted:
            kept.append(target)
    return kept
