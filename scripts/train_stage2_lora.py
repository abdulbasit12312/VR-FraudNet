"""Fine-tune the Stage 2 rationale language model with LoRA (manuscript S4.3.1).

Usage
-----
    python scripts/train_stage2_lora.py \\
        --dataset D1 --config configs/d1.yaml \\
        --stage2-config configs/stage2_lora.yaml \\
        --corpus <rationale_corpus_train.jsonl> --seed 42

    # Validate the corpus, apply every filter, print the training plan and stop
    # before any model weight is touched (no GPU, no download):
    python scripts/train_stage2_lora.py ... --dry-run

What this script does
---------------------
1. Loads the canonical Stage 2 configuration (``configs/stage2_lora.yaml``)
   and the dataset configuration (for the routing thresholds that define the
   review band).
2. Loads a rationale corpus in the JSONL format described in
   ``docs/STAGE2_LORA.md``. Every record must belong to the ``train``
   partition of the named dataset; anything else raises ``LeakageError``.
3. Applies the manuscript's three training-set filters (S4.3.1):
   review-band only, verifier-accepted targets only, grammar-serialisable
   targets only. Counts of everything dropped are written to the run manifest.
4. Generates one label-preserving counterfactual per record (S4.6.3) from the
   training partition only.
5. Builds prompts with ``vrfraudnet.models.stage2_rationale.build_prompt``.
6. Fine-tunes ``meta-llama/Meta-Llama-3.1-8B-Instruct`` with PEFT LoRA
   (r=16, alpha=32, dropout=0.05, q/k/v/o projections), AdamW, lr 2e-4,
   weight decay 0.01, 5% linear warm-up, 3 epochs, bfloat16, effective batch
   32, checkpoint selected on validation loss, and saves the adapter.

What this script does NOT do
----------------------------
* It does not ship, download or fabricate a rationale corpus. The manuscript
  does not describe how target rationales were authored (L-25), so the corpus
  is an operator input.
* It does not produce the adapter used for the manuscript. That adapter is not
  present in the accessible project materials (artifacts/lora/README.md). An
  adapter trained here is a *reconstruction under the documented
  configuration*, and its manifest says so.
* It does not guess values the manuscript omits. ``--per-device-batch-size``
  must be given because only the effective batch size is stated (L-30); the
  script derives the gradient-accumulation steps and records both.

Exit codes: 0 success, 2 missing input, 3 leakage or filter failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from _common import REPO_ROOT, add_common_arguments, bootstrap, make_manifest

from vrfraudnet.errors import ConfigurationError, LeakageError, MissingArtefactError
from vrfraudnet.io_utils import sha256_of_file, write_result
from vrfraudnet.losses.counterfactual import generate_counterfactual
from vrfraudnet.models.grammar import ParseState
from vrfraudnet.models.stage2_rationale import (
    RationaleModel,
    Stage2Config,
    build_prompt,
    schema_hint,
)
from vrfraudnet.verifier import DeterministicVerifier
from vrfraudnet.verifier.evidence import TransactionEvidence

log = logging.getLogger("train_stage2")

#: Key order the decoding grammar expects (configs/stage2_lora.yaml, L-15).
RATIONALE_KEY_ORDER = ("verdict", "rationale_probability", "claims", "evidence_references")
CLAIM_KEY_ORDER = ("type", "field", "operator", "value", "evidence_id")


@dataclass
class CorpusRecord:
    """One rationale-training example as read from the JSONL corpus."""

    record_id: str
    dataset: str
    partition: str
    timestamp: float
    triage_probability: float
    evidence: dict[str, Any]
    evidence_summary: str
    target: dict[str, Any]
    entity_ids: tuple[str, ...]


@dataclass
class TrainingExample:
    prompt: str
    target_text: str
    kind: str  # "original" or "counterfactual"
    source_id: str


def serialise_rationale(rationale: dict[str, Any]) -> str:
    """Serialise a rationale with the exact key order the grammar accepts."""
    ordered: dict[str, Any] = {}
    for key in RATIONALE_KEY_ORDER:
        if key == "claims":
            ordered[key] = [
                {k: claim[k] for k in CLAIM_KEY_ORDER if k in claim} for claim in rationale[key]
            ]
        else:
            ordered[key] = rationale[key]
    return json.dumps(ordered, separators=(",", ":"), ensure_ascii=True)


def load_corpus(path: Path, *, dataset: str, partition: str = "train") -> list[CorpusRecord]:
    """Read a JSONL corpus and enforce the partition and dataset constraints."""
    if not path.exists():
        raise MissingArtefactError(
            f"rationale corpus not found: {path}\n"
            "No corpus is shipped with this repository (docs/KNOWN_LIMITATIONS.md L-25). "
            "The expected JSONL format is documented in docs/STAGE2_LORA.md and "
            "illustrated by examples/stage2_corpus_example.jsonl (synthetic)."
        )
    records: list[CorpusRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            for key in ("record_id", "dataset", "partition", "timestamp",
                        "triage_probability", "evidence", "evidence_summary", "target"):
                if key not in raw:
                    raise ConfigurationError(f"{path}:{line_no}: missing field {key!r}")
            if raw["partition"] != partition:
                raise LeakageError(
                    f"{path}:{line_no}: record {raw['record_id']!r} belongs to partition "
                    f"{raw['partition']!r}; Stage 2 may only be trained on {partition!r} "
                    "(manuscript S4.3.1)"
                )
            if str(raw["dataset"]).upper() != dataset.upper():
                raise ConfigurationError(
                    f"{path}:{line_no}: record is for dataset {raw['dataset']!r}, "
                    f"but --dataset is {dataset!r}; the index and the adapter are per dataset"
                )
            records.append(
                CorpusRecord(
                    record_id=str(raw["record_id"]),
                    dataset=str(raw["dataset"]).upper(),
                    partition=str(raw["partition"]),
                    timestamp=float(raw["timestamp"]),
                    triage_probability=float(raw["triage_probability"]),
                    evidence=dict(raw["evidence"]),
                    evidence_summary=str(raw["evidence_summary"]),
                    target=dict(raw["target"]),
                    entity_ids=tuple(str(e) for e in raw.get("entity_ids", [])),
                )
            )
    return records


def apply_manuscript_filters(
    records: list[CorpusRecord],
    *,
    t_low: float,
    t_high: float,
    verifier: DeterministicVerifier,
    grammar,
) -> tuple[list[CorpusRecord], Counter]:
    """Keep only review-band, verifier-accepted, grammar-serialisable targets."""
    kept: list[CorpusRecord] = []
    dropped: Counter = Counter()
    for record in records:
        if not t_low <= record.triage_probability <= t_high:
            dropped["outside_review_band"] += 1
            continue
        evidence = TransactionEvidence.from_json(record.evidence)
        if not verifier.verify(record.target, evidence).accepted:
            dropped["verifier_rejected"] += 1
            continue
        if grammar.parse(serialise_rationale(record.target)) is not ParseState.COMPLETE:
            dropped["not_grammar_serialisable"] += 1
            continue
        kept.append(record)
    return kept, dropped


def build_counterfactual(
    record: CorpusRecord, *, verifier: DeterministicVerifier, rng: np.random.Generator
) -> tuple[CorpusRecord | None, str]:
    """One label-preserving counterfactual per record (manuscript S4.6.3).

    Operationalisation (ASSUMPTION L-32): one admissible evidence value is
    perturbed; claims that still verify against the modified evidence are kept
    with their original form and references, claims that no longer verify are
    removed, and the pair is discarded when no claim survives or the verdict is
    no longer supported. The manuscript does not state the exact rule.
    """
    flat: dict[str, Any] = {"transaction_id": record.record_id}
    groups = ("numeric", "categorical", "entities", "temporal_aggregates")
    for group in groups:
        flat.update({k: v for k, v in dict(record.evidence.get(group, {})).items()})
    mutable = [k for k in flat if k != "transaction_id"]
    if not mutable:
        return None, "no_mutable_evidence"
    cf_flat, example = generate_counterfactual(
        flat, record.target["claims"], rng=rng, partition=record.partition, mutable_fields=mutable
    )
    cf_evidence = json.loads(json.dumps(record.evidence))
    for group in groups:
        if example.modified_field in cf_evidence.get(group, {}):
            cf_evidence[group][example.modified_field] = cf_flat[example.modified_field]
    evidence_obj = TransactionEvidence.from_json(cf_evidence)
    trial = dict(record.target)
    outcome = verifier.verify(trial, evidence_obj)
    surviving = [
        claim for claim, o in zip(trial["claims"], outcome.claim_outcomes) if o.passed
    ] if outcome.claim_outcomes else []
    if not surviving:
        return None, "no_claim_survives"
    cf_target = {
        "verdict": trial["verdict"],
        "rationale_probability": trial["rationale_probability"],
        "claims": surviving,
        "evidence_references": sorted({c["evidence_id"] for c in surviving}),
    }
    if not verifier.verify(cf_target, evidence_obj).accepted:
        return None, "verdict_unsupported_after_edit"
    return (
        CorpusRecord(
            record_id=f"{record.record_id}#cf",
            dataset=record.dataset,
            partition=record.partition,
            timestamp=record.timestamp,
            triage_probability=record.triage_probability,
            evidence=cf_evidence,
            evidence_summary=record.evidence_summary
            + f" [counterfactual: {example.modified_field} -> {example.counterfactual_value}]",
            target=cf_target,
            entity_ids=record.entity_ids,
        ),
        "ok",
    )


def make_examples(
    records: list[CorpusRecord],
    *,
    hint: str,
    kind: str,
    retrieve=None,
) -> list[TrainingExample]:
    out: list[TrainingExample] = []
    for record in records:
        demonstrations = retrieve(record) if retrieve is not None else []
        prompt = build_prompt(
            record.evidence_summary,
            record.triage_probability,
            record.timestamp,
            demonstrations,
            schema_hint=hint,
        )
        out.append(
            TrainingExample(prompt, serialise_rationale(record.target), kind, record.record_id)
        )
    return out


def _corpus_digest(records: list[CorpusRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.record_id):
        digest.update(record.record_id.encode("utf-8"))
        digest.update(serialise_rationale(record.target).encode("utf-8"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--stage2-config", type=Path,
                        default=REPO_ROOT / "configs" / "stage2_lora.yaml")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="JSONL rationale corpus, TRAIN partition only (docs/STAGE2_LORA.md)")
    parser.add_argument("--validation-corpus", type=Path, default=None,
                        help="optional JSONL corpus from the VALIDATION partition, used only "
                             "for checkpoint selection (validation loss)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="adapter output directory (default checkpoints/stage2/<dataset>/seed<seed>)")
    parser.add_argument("--per-device-batch-size", type=int, default=None,
                        help="REQUIRED for a real run; the manuscript states only the effective "
                             "batch size of 32 (L-30). Gradient accumulation is derived.")
    parser.add_argument("--gradient-clip-norm", type=float, default=None,
                        help="not stated by the manuscript for Stage 2 (L-30); recorded if given")
    parser.add_argument("--base-model-revision", default=None,
                        help="immutable Hugging Face revision of the base model (L-29); recorded")
    parser.add_argument("--retrieval-index", type=Path, default=None,
                        help="directory written by scripts/build_retrieval_index.py; when given, "
                             "the four most similar eligible TRAIN examples are prepended as "
                             "demonstrations (manuscript S4.3.4)")
    parser.add_argument("--use-manuscript-thresholds", action="store_true", default=True,
                        help="define the review band with the per-dataset (T_low, T_high) of "
                             "S4.8.1 (default; the only option without a trained Stage 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate, filter, build prompts, write the plan, and stop before "
                             "loading any model weight")
    args = parser.parse_args()

    config, _ = bootstrap(args)
    stage2 = Stage2Config.from_yaml(args.stage2_config)
    if args.base_model_revision:
        stage2 = Stage2Config(**{**stage2.__dict__, "base_model_revision": args.base_model_revision})
    log.info("Stage 2 config: %s @ %s, LoRA r=%d alpha=%d dropout=%.3f targets=%s",
             stage2.base_model, stage2.revision_label(), stage2.lora.rank, stage2.lora.alpha,
             stage2.lora.dropout, list(stage2.lora.target_modules))

    t_low = float(config.require("stage1.threshold_low"))
    t_high = float(config.require("stage1.threshold_high"))
    verifier = DeterministicVerifier()
    grammar = RationaleModel(stage2, REPO_ROOT / "schemas" / "rationale_schema.json").grammar
    hint = schema_hint(REPO_ROOT / "schemas" / "rationale_schema.json")

    records = load_corpus(args.corpus, dataset=args.dataset, partition="train")
    kept, dropped = apply_manuscript_filters(
        records, t_low=t_low, t_high=t_high, verifier=verifier, grammar=grammar
    )
    log.info("corpus: %d records, %d kept after S4.3.1 filters, dropped=%s",
             len(records), len(kept), dict(dropped))
    if not kept:
        log.error("no training example survives the manuscript filters; refusing to train")
        return 3

    rng = np.random.default_rng(args.seed)
    counterfactuals: list[CorpusRecord] = []
    cf_dropped: Counter = Counter()
    for record in kept:
        cf, status = build_counterfactual(record, verifier=verifier, rng=rng)
        if cf is None:
            cf_dropped[status] += 1
        else:
            counterfactuals.append(cf)
    log.info("counterfactuals: %d generated, dropped=%s", len(counterfactuals), dict(cf_dropped))

    retrieve = None
    if args.retrieval_index is not None:
        from vrfraudnet.retrieval.index import RetrievalIndex, default_encoder, hashing_encoder

        index = RetrievalIndex.load(args.retrieval_index)
        manifest = index.manifest or {}
        if manifest.get("reproduction_grade", True):
            revision = manifest.get("encoder_revision")
            index._encoder = default_encoder(None if revision in (None, "unpinned") else revision)
        else:
            log.warning("retrieval index %s was built with the TEST-ONLY encoder; demonstrations "
                        "drawn from it are not reproduction-grade", args.retrieval_index)
            index._encoder = hashing_encoder()

        def retrieve(record: CorpusRecord) -> list[str]:
            hits = index.query(
                record.evidence_summary,
                query_timestamp=record.timestamp,
                query_transaction_id=record.record_id,
                query_entity_ids=frozenset(record.entity_ids),
            )
            return [serialise_rationale(h.rationale) for h in hits]

    examples = make_examples(kept, hint=hint, kind="original", retrieve=retrieve)
    examples += make_examples(counterfactuals, hint=hint, kind="counterfactual", retrieve=retrieve)

    validation_examples: list[TrainingExample] = []
    if args.validation_corpus is not None:
        val_records = load_corpus(args.validation_corpus, dataset=args.dataset, partition="validation")
        val_kept, val_dropped = apply_manuscript_filters(
            val_records, t_low=t_low, t_high=t_high, verifier=verifier, grammar=grammar
        )
        log.info("validation corpus: %d records, %d kept, dropped=%s",
                 len(val_records), len(val_kept), dict(val_dropped))
        validation_examples = make_examples(val_kept, hint=hint, kind="original", retrieve=retrieve)

    output_dir = args.output_dir or (
        REPO_ROOT / "checkpoints" / "stage2" / args.dataset / f"seed{args.seed}"
    )
    plan = {
        "adapter_status": "RECONSTRUCTION under configs/stage2_lora.yaml; NOT the manuscript adapter",
        "base_model": stage2.base_model,
        "base_model_revision": stage2.revision_label(),
        "stage2_config_sha256": sha256_of_file(args.stage2_config),
        "corpus_path": args.corpus.name,
        "corpus_sha256": sha256_of_file(args.corpus),
        "kept_records_digest": _corpus_digest(kept),
        "review_band": {"t_low": t_low, "t_high": t_high, "source": "manuscript S4.8.1"},
        "records_loaded": len(records),
        "records_kept": len(kept),
        "records_dropped": dict(dropped),
        "counterfactuals_generated": len(counterfactuals),
        "counterfactuals_dropped": dict(cf_dropped),
        "counterfactual_weight": stage2.counterfactual_weight,
        "training_examples": len(examples),
        "validation_examples": len(validation_examples),
        "retrieval_demonstrations": args.retrieval_index is not None,
        "lora": stage2.lora.to_peft_kwargs(),
        "optimisation": {
            "optimizer": "adamw",
            "learning_rate": stage2.learning_rate,
            "weight_decay": stage2.weight_decay,
            "warmup_ratio": stage2.warmup_ratio,
            "epochs": stage2.epochs,
            "effective_batch_size": stage2.effective_batch_size,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": (
                stage2.effective_batch_size // args.per_device_batch_size
                if args.per_device_batch_size else None
            ),
            "gradient_clip_norm": args.gradient_clip_norm,
            "precision": stage2.precision,
            "max_input_tokens": stage2.max_input_tokens,
            "max_output_tokens": stage2.max_output_tokens,
        },
        "checkpoint_selection": "validation loss; schema-valid rate and verifier acceptance secondary",
        "output_dir": (output_dir.relative_to(REPO_ROOT).as_posix()
                       if output_dir.is_relative_to(REPO_ROOT) else output_dir.name),
        "dry_run": bool(args.dry_run),
    }
    manifest_path = Path(args.results_dir) / "stage2" / args.dataset / f"train_plan_seed{args.seed}.json"
    write_result(
        manifest_path,
        make_manifest(args, experiment="train_stage2_lora", model="vr_fraudnet_stage2",
                      partition="train", notes="Stage 2 LoRA training plan / run record"),
        plan,
        overwrite=True,
    )
    log.info("wrote %s", manifest_path)

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print("\nDRY RUN: no model weight was loaded, downloaded or written.")
        return 0

    if args.per_device_batch_size is None:
        log.error("--per-device-batch-size is required for a real run: the manuscript states only "
                  "the effective batch size of 32 (docs/KNOWN_LIMITATIONS.md L-30)")
        return 2
    if stage2.effective_batch_size % args.per_device_batch_size:
        log.error("effective batch size %d is not divisible by per-device batch size %d",
                  stage2.effective_batch_size, args.per_device_batch_size)
        return 2

    return _train(stage2, examples, validation_examples, output_dir, args, plan)


def _train(stage2, examples, validation_examples, output_dir, args, plan) -> int:  # pragma: no cover - needs GPU + gated weights
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise MissingArtefactError(
            "transformers, peft, datasets and torch are required for Stage 2 training:\n"
            "    pip install -r requirements-optional.txt"
        ) from exc

    revision = stage2.base_model_revision
    tokenizer = AutoTokenizer.from_pretrained(stage2.base_model, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        stage2.base_model, torch_dtype=getattr(torch, stage2.precision), revision=revision
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)  # MANUSCRIPT S4.3.1: base weights frozen
    model = get_peft_model(model, LoraConfig(**stage2.lora.to_peft_kwargs()))
    model.print_trainable_parameters()

    def encode(example: TrainingExample) -> dict[str, Any]:
        prompt_ids = tokenizer(example.prompt, add_special_tokens=True,
                               truncation=True, max_length=stage2.max_input_tokens).input_ids
        target_ids = tokenizer(example.target_text, add_special_tokens=False,
                               truncation=True, max_length=stage2.max_output_tokens).input_ids
        target_ids = target_ids + [tokenizer.eos_token_id]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        weight = stage2.counterfactual_weight if example.kind == "counterfactual" else 1.0
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
                "labels": labels, "loss_weight": weight}

    train_ds = Dataset.from_list([encode(e) for e in examples])
    eval_ds = Dataset.from_list([encode(e) for e in validation_examples]) if validation_examples else None

    class WeightedTrainer(Trainer):
        """Applies the counterfactual weight (S4.6.3, 0.25) as a per-example loss weight."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("loss_weight")
            outputs = model(**{k: v for k, v in inputs.items() if k != "loss_weight"})
            logits = outputs.logits[:, :-1, :]
            labels = inputs["labels"][:, 1:]
            per_token = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                ignore_index=-100, reduction="none",
            ).view(labels.shape)
            mask = (labels != -100).float()
            per_example = (per_token * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            loss = (per_example * weights.to(per_example.dtype)).sum() / weights.sum()
            return (loss, outputs) if return_outputs else loss

    grad_accum = stage2.effective_batch_size // args.per_device_batch_size
    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=stage2.epochs,
        learning_rate=stage2.learning_rate,
        weight_decay=stage2.weight_decay,
        warmup_ratio=stage2.warmup_ratio,
        lr_scheduler_type="linear",  # linear warm-up is stated; post-warm-up decay is L-30
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=(stage2.precision == "bfloat16"),
        max_grad_norm=args.gradient_clip_norm if args.gradient_clip_norm is not None else 1.0,
        evaluation_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=10,
        report_to=[],
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    def collate(features):
        weights = torch.tensor([f.pop("loss_weight") for f in features], dtype=torch.float32)
        batch = collator(features)
        batch["loss_weight"] = weights
        return batch

    trainer = WeightedTrainer(model=model, args=training_args, train_dataset=train_ds,
                              eval_dataset=eval_ds, data_collator=collate)
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    record = dict(plan)
    record["adapter_files"] = {
        p.name: sha256_of_file(p) for p in sorted(output_dir.iterdir()) if p.is_file()
    }
    (output_dir / "vrfraudnet_training_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    log.info("adapter written to %s (reconstruction; not the manuscript adapter)", output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MissingArtefactError, ConfigurationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except LeakageError as exc:
        print(f"leakage: {exc}", file=sys.stderr)
        raise SystemExit(3)
