"""Reviewer reproducibility artefacts: every item the reviewer asked for is where
the documentation says it is, agrees with the code, and cannot silently drift.

Covers: reviewer matrix paths; seed file vs code constant; Stage 2 canonical
config vs dataclass vs base.yaml; adapter absence and validator behaviour;
schema/verifier/grammar drift; grammar reviewer cases; retrieval config vs
runtime constants and partition refusal; edit-family registry vs code vs
documentation; D3 fold definition; manifest and archive build/verify; README
coverage; CI wiring.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from vrfraudnet.adversarial.applicability import EDIT_FAMILIES, applicable_families
from vrfraudnet.adversarial.edits import EDIT_FUNCTIONS
from vrfraudnet.data.splits import MANUSCRIPT_D3_CUT_QUANTILES, expanding_window_folds
from vrfraudnet.errors import ConfigurationError, LeakageError, MissingArtefactError
from vrfraudnet.models.grammar import GrammarSpec, ParseState, RationaleGrammar
from vrfraudnet.models.stage2_rationale import (
    RationaleModel,
    Stage2Config,
    check_adapter_metadata,
)
from vrfraudnet.reproducibility import (
    COMPONENT_PATHS,
    FORBIDDEN_SUFFIXES,
    MANIFEST_FILE,
    REQUIRED_IN_ARCHIVE,
    BundleFile,
    build_manifest,
    collect_bundle_files,
    compare_manifest_files,
    forbidden_content_problems,
    verify_archive,
    write_archive,
)
from vrfraudnet.retrieval.index import (
    ENCODER_DIMENSION,
    ENCODER_REPOSITORY,
    N_RETRIEVED,
    RetrievalIndex,
    RetrievalRecord,
    hashing_encoder,
)
from vrfraudnet.seeds import MANUSCRIPT_SEEDS
from vrfraudnet.verifier.primitives import CHECKERS


# ---------------------------------------------------------------------------
# 1. reviewer matrix paths and README coverage
# ---------------------------------------------------------------------------

def test_reviewer_matrix_paths_exist(repo_root):
    text = (repo_root / "docs" / "REVIEWER_REPRODUCIBILITY.md").read_text(encoding="utf-8")
    pattern = re.compile(r"`([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-/]*(?:\.(?:py|md|yaml|yml|json|jsonl|txt|toml|cff|sh)|/))`")
    tokens = {m.group(1).split("::")[0] for m in pattern.finditer(text)}
    skip = ("dist/", "artifacts/lora/manuscript", "/tmp/")
    missing = sorted(t for t in tokens if not t.startswith(skip) and not (repo_root / t.rstrip("/")).exists())
    assert not missing, f"reviewer matrix references paths that do not exist: {missing}"
    assert len(tokens) >= 25


def test_reviewer_audit_script_passes(repo_root):
    result = subprocess.run(
        [sys.executable, "scripts/audit_reviewer_artifacts.py"],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_readme_has_reproducing_section_and_links(repo_root):
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "## Reproducing the study" in readme
    for target in (
        "verifier/rules/claim_primitives.yaml", "schemas/rationale_schema.json",
        "docs/DECODING_GRAMMAR.md", "docs/STAGE2_LORA.md", "artifacts/lora/README.md",
        "configs/stage2_lora.yaml", "docs/RETRIEVAL_SETUP.md", "docs/EDIT_GENERATORS.md",
        "configs/manuscript_seeds.yaml", "docs/RELEASE.md", "docs/ARTIFACT_AVAILABILITY.md",
        "docs/REVIEWER_REPRODUCIBILITY.md", "docs/KNOWN_LIMITATIONS.md",
    ):
        assert f"]({target}" in readme, f"README does not link to {target}"
    assert "NOT" in readme and "adapter" in readme.lower()


# ---------------------------------------------------------------------------
# 2. seeds
# ---------------------------------------------------------------------------

def test_seed_file_matches_code_constant(repo_root):
    data = yaml.safe_load((repo_root / "configs" / "manuscript_seeds.yaml").read_text(encoding="utf-8"))
    assert data["schema_version"] == "vrfraudnet-seeds/1"
    assert [int(s) for s in data["seeds"]] == list(MANUSCRIPT_SEEDS)
    assert data["n_repeated_runs"] == len(MANUSCRIPT_SEEDS) == 10


def test_base_config_seeds_match_seed_file(repo_root):
    base = yaml.safe_load((repo_root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    assert list(base["seeds"]) == list(MANUSCRIPT_SEEDS)


# ---------------------------------------------------------------------------
# 3. Stage 2 configuration and adapter
# ---------------------------------------------------------------------------

def test_stage2_canonical_config_matches_dataclass_defaults(repo_root):
    loaded = Stage2Config.from_yaml(repo_root / "configs" / "stage2_lora.yaml")
    assert loaded == Stage2Config()


def test_base_config_stage2_block_matches_canonical(repo_root):
    base = yaml.safe_load((repo_root / "configs" / "base.yaml").read_text(encoding="utf-8"))["stage2"]
    canon = Stage2Config.from_yaml(repo_root / "configs" / "stage2_lora.yaml")
    assert base["base_model"] == canon.base_model
    assert base["lora_rank"] == canon.lora.rank
    assert base["lora_alpha"] == canon.lora.alpha
    assert base["lora_dropout"] == canon.lora.dropout
    assert tuple(base["target_modules"]) == canon.lora.target_modules
    assert base["epochs"] == canon.epochs
    assert base["learning_rate"] == canon.learning_rate
    assert base["weight_decay"] == canon.weight_decay
    assert base["warmup_ratio"] == canon.warmup_ratio
    assert base["max_input_tokens"] == canon.max_input_tokens
    assert base["max_output_tokens"] == canon.max_output_tokens
    assert base["effective_batch_size"] == canon.effective_batch_size
    assert base["precision"] == canon.precision
    assert base["n_retrieved_examples"] == canon.n_retrieved_examples


def test_stage2_config_provenance_cites_existing_limitations(repo_root):
    raw = yaml.safe_load((repo_root / "configs" / "stage2_lora.yaml").read_text(encoding="utf-8"))
    limitations = (repo_root / "docs" / "KNOWN_LIMITATIONS.md").read_text(encoding="utf-8")
    for key, prov in raw["_provenance"].items():
        assert prov["tag"] in {"MANUSCRIPT", "ASSUMPTION", "DERIVED"}, key
        if prov["tag"] == "ASSUMPTION":
            assert f"### {prov['limitation']} " in limitations, f"{key} cites unknown {prov['limitation']}"
        if prov["tag"] == "MANUSCRIPT":
            assert prov.get("source"), key
    assert raw["base_model"]["revision"] is None  # L-29: not fabricated
    assert raw["adapter"]["authentic_manuscript_adapter"] == "NOT_AVAILABLE"


def test_no_adapter_is_shipped(repo_root):
    lora_dir = repo_root / "artifacts" / "lora"
    files = [p for p in lora_dir.rglob("*") if p.is_file()]
    assert [p.name for p in files] == ["README.md"]
    assert "NOT" in (lora_dir / "README.md").read_text(encoding="utf-8")


def test_rationale_model_refuses_to_load_without_adapter(repo_root):
    model = RationaleModel(Stage2Config(), repo_root / "schemas" / "rationale_schema.json",
                           adapter_path=repo_root / "artifacts" / "lora" / "manuscript")
    with pytest.raises(MissingArtefactError) as excinfo:
        model.load()
    message = str(excinfo.value)
    assert "train_stage2_lora.py" in message and "artifacts/lora/README.md" in message
    with pytest.raises(MissingArtefactError):
        model.generate("anything")


def test_adapter_metadata_check_detects_mismatch(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 8, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "bias": "none",
        "base_model_name_or_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    }), encoding="utf-8")
    problems = check_adapter_metadata(adapter, Stage2Config())
    assert any("r: adapter declares 8" in p for p in problems)
    assert any("no adapter weight file" in p for p in problems)
    (adapter / "adapter_model.safetensors").write_bytes(b"")  # interface fixture only
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "bias": "none",
        "base_model_name_or_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    }), encoding="utf-8")
    assert check_adapter_metadata(adapter, Stage2Config()) == []


def test_validate_adapter_script_exit_codes(repo_root, tmp_path):
    missing = subprocess.run(
        [sys.executable, "scripts/validate_stage2_adapter.py", "--adapter", str(tmp_path / "nope")],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert missing.returncode == 2 and "MISSING" in missing.stdout
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 4}', encoding="utf-8")
    mismatch = subprocess.run(
        [sys.executable, "scripts/validate_stage2_adapter.py", "--adapter", str(bad)],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert mismatch.returncode == 1 and "MISMATCH" in mismatch.stdout


def test_train_stage2_dry_run_on_synthetic_corpus(repo_root, tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/train_stage2_lora.py", "--dataset", "D2",
         "--config", "configs/d2.yaml", "--corpus", "examples/stage2_corpus_example.jsonl",
         "--seed", "42", "--dry-run", "--results-dir", str(tmp_path)],
        cwd=repo_root, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRY RUN" in result.stdout
    plan = json.loads((tmp_path / "stage2" / "D2" / "train_plan_seed42.json").read_text(encoding="utf-8"))
    assert plan["results"]["records_kept"] == 2
    assert plan["results"]["adapter_status"].startswith("RECONSTRUCTION")
    assert plan["results"]["base_model_revision"] == "unpinned"


def test_train_stage2_refuses_non_training_partition(repo_root, tmp_path):
    lines = (repo_root / "examples" / "stage2_corpus_example.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["partition"] = "test"
    corpus = tmp_path / "leak.jsonl"
    corpus.write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/train_stage2_lora.py", "--dataset", "D2",
         "--config", "configs/d2.yaml", "--corpus", str(corpus), "--seed", "42", "--dry-run",
         "--results-dir", str(tmp_path)],
        cwd=repo_root, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 3 and "leakage" in result.stderr.lower()


# ---------------------------------------------------------------------------
# 4. schema / verifier / grammar drift
# ---------------------------------------------------------------------------

def test_schema_and_verifier_rules_do_not_drift(repo_root):
    schema = json.loads((repo_root / "schemas" / "rationale_schema.json").read_text(encoding="utf-8"))
    rules = yaml.safe_load((repo_root / "verifier" / "rules" / "claim_primitives.yaml").read_text(encoding="utf-8"))
    schema_ops = set(schema["$defs"]["claim"]["properties"]["operator"]["enum"])
    rule_ops = {op for p in rules["primitives"].values() for op in p["operators"]}
    assert schema_ops == rule_ops
    assert set(schema["$defs"]["claim"]["properties"]["type"]["enum"]) == set(rules["primitives"]) == set(CHECKERS)
    assert {r["id"] for r in rules["consistency_rules"]} == {"CR1", "CR2", "CR3", "CR4", "CR5", "CR6"}
    grammar_spec = GrammarSpec.from_schema(repo_root / "schemas" / "rationale_schema.json")
    assert set(grammar_spec.operators) == schema_ops
    assert grammar_spec.max_claims == 8 and grammar_spec.min_claims == 1


def test_schema_validation_script_passes(repo_root):
    result = subprocess.run(
        [sys.executable, "scripts/validate_rationale_schema.py"],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout


@pytest.fixture
def grammar(repo_root):
    return RationaleGrammar(GrammarSpec.from_schema(repo_root / "schemas" / "rationale_schema.json"))


def test_grammar_reviewer_cases(grammar, repo_root):
    accepted = json.loads((repo_root / "examples" / "rationale_accepted.json").read_text(encoding="utf-8"))
    ordered = json.dumps({
        "verdict": accepted["verdict"],
        "rationale_probability": accepted["rationale_probability"],
        "claims": [{k: c[k] for k in ("type", "field", "operator", "value", "evidence_id")}
                   for c in accepted["claims"]],
        "evidence_references": accepted["evidence_references"],
    }, separators=(",", ":"))
    assert grammar.parse(ordered) is ParseState.COMPLETE                       # complete valid
    assert grammar.parse('{"verdict":"fra') is ParseState.PARTIAL               # valid prefix
    assert grammar.parse('{"verdict":"maybe"') is ParseState.INVALID            # invalid enum
    assert grammar.parse('{"explanation":') is ParseState.INVALID               # illegal key
    assert grammar.parse('{"verdict":"fraud","rationale_probability":0.5,"claims":[{"type":'
                         '"numeric_comparison","field":"amount","operator":"approx"') is ParseState.INVALID
    assert grammar.parse(ordered[:-1]) is ParseState.PARTIAL                    # incomplete JSON
    assert grammar.parse('{"verdict":"fraud","rationale_probability":1.5') is ParseState.INVALID


# ---------------------------------------------------------------------------
# 5. retrieval
# ---------------------------------------------------------------------------

def test_retrieval_config_matches_runtime_constants(repo_root):
    cfg = yaml.safe_load((repo_root / "retrieval" / "config" / "retrieval.yaml").read_text(encoding="utf-8"))
    assert cfg["encoder"]["repository"] == ENCODER_REPOSITORY
    assert cfg["encoder"]["embedding_dimension"] == ENCODER_DIMENSION
    assert cfg["encoder"]["revision"] is None  # L-29: not fabricated
    assert cfg["retrieval"]["n_retrieved"] == N_RETRIEVED == 4
    assert cfg["eligibility"]["partition"]["allowed"] == ["train"]
    assert cfg["similarity"]["approximate_search"] is False


@pytest.mark.parametrize("partition", ["validation", "test", "transfer", "temporal_shock",
                                       "strict_inductive", "adversarial", "calibration"])
def test_retrieval_index_refuses_every_non_training_partition(partition):
    with pytest.raises(LeakageError):
        RetrievalRecord("r", partition, 1.0, "summary", "review", {"verdict": "fraud"})


def test_retrieval_index_round_trip_verifies_checksums(tmp_path):
    index = RetrievalIndex(encoder=hashing_encoder())
    index.add(RetrievalRecord("a", "train", 1.0, "wire 9000 new account", "review",
                              {"verdict": "fraud"}, frozenset({"E1"})), verifier_accepted=True)
    index.add(RetrievalRecord("b", "train", 2.0, "card 12 grocery", "review",
                              {"verdict": "legitimate"}), verifier_accepted=True)
    index.save(tmp_path)
    loaded = RetrievalIndex.load(tmp_path, encoder=hashing_encoder())
    assert len(loaded) == 2
    assert [r.record_id for r in loaded.query("wire 9500 new account", query_timestamp=5.0)] == ["a", "b"]
    # future example excluded, shared entity excluded
    assert [r.record_id for r in loaded.query("x", query_timestamp=1.5, query_entity_ids=frozenset({"E1"}))] == []
    (tmp_path / "records.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        RetrievalIndex.load(tmp_path, encoder=hashing_encoder())


def test_build_retrieval_index_script(repo_root, tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/build_retrieval_index.py", "--dataset", "D2",
         "--config", "configs/d2.yaml", "--corpus", "examples/stage2_corpus_example.jsonl",
         "--seed", "42", "--encoder", "test-hashing", "--output-dir", str(tmp_path / "idx")],
        cwd=repo_root, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((tmp_path / "idx" / "index_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_records"] == 2 and manifest["partition"] == "train"
    assert manifest["reproduction_grade"] is False


# ---------------------------------------------------------------------------
# 6. edit generators
# ---------------------------------------------------------------------------

def test_edit_family_registry_documentation_and_code_agree(repo_root):
    registry = yaml.safe_load((repo_root / "adversarial" / "families" / "edit_families.yaml").read_text(encoding="utf-8"))
    doc = (repo_root / "docs" / "EDIT_GENERATORS.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"^## ε[₁₂₃₄₅] (\w+) ", doc, flags=re.MULTILINE))
    assert set(registry["families"]) == set(EDIT_FAMILIES) == set(EDIT_FUNCTIONS) == documented
    assert set(registry["predictive_path_families"]) == {"split", "delay", "mule", "feat"}
    for family, spec in registry["families"].items():
        assert set(spec["status_by_dataset"]) == {"D1", "D2", "D3", "D4", "D5"}, family
        for dataset, status in spec["status_by_dataset"].items():
            applicable = family in applicable_families(dataset, "lightgbm")
            assert applicable == (status == "applicable"), (family, dataset, status)


# ---------------------------------------------------------------------------
# 7. D3 expanding window per revised Table 8(d)
# ---------------------------------------------------------------------------

def test_d3_cut_quantiles_match_manuscript_table_8d(repo_root):
    cfg = yaml.safe_load((repo_root / "configs" / "d3.yaml").read_text(encoding="utf-8"))
    assert tuple(cfg["split"]["cv_cut_quantiles"]) == MANUSCRIPT_D3_CUT_QUANTILES
    assert cfg["_provenance"]["split.cv_cut_quantiles"]["tag"] == "MANUSCRIPT"
    times = np.arange(1000.0)
    folds = expanding_window_folds(times, n_folds=5, cut_quantiles=MANUSCRIPT_D3_CUT_QUANTILES)
    for q, fold in zip(MANUSCRIPT_D3_CUT_QUANTILES, folds):
        assert fold.test.size == round((1 - q) * 1000)              # test-to-end
        assert times[fold.test].min() > times[fold.validation].max()
        assert times[fold.validation].min() > times[fold.train].max()
    assert folds[2].test.size == 200                                    # Fold 3 = primary 0.80 holdout


# ---------------------------------------------------------------------------
# 8. manifest and archive
# ---------------------------------------------------------------------------

def test_manifest_components_exist_and_have_real_hashes(repo_root):
    manifest = build_manifest(repo_root, version="test")
    for key, entries in manifest["components"].items():
        assert entries, key
        for rel, digest in entries.items():
            assert (repo_root / rel).exists(), rel
            assert re.fullmatch(r"[0-9a-f]{64}", digest), rel
    assert manifest["seeds"]["values"] == list(MANUSCRIPT_SEEDS)
    assert manifest["stage2_adapter"]["status"] == "NOT_AVAILABLE"
    assert manifest["stage2_adapter"]["sha256"] is None
    assert manifest["base_model"]["revision"] is None
    assert set(COMPONENT_PATHS) == set(manifest["components"])


def test_committed_manifest_is_current(repo_root):
    committed = json.loads((repo_root / MANIFEST_FILE).read_text(encoding="utf-8"))
    fresh = build_manifest(repo_root, version=committed["release_version"],
                           created_utc=committed["created_utc"])
    assert compare_manifest_files(committed, fresh) == []
    assert re.fullmatch(r"[0-9a-f]{40}", committed["git"]["commit"] or "")


def test_bundle_excludes_data_weights_and_secrets(repo_root):
    files = collect_bundle_files(repo_root)
    assert forbidden_content_problems(files) == []
    relatives = {f.relative for f in files}
    assert not any(Path(r).suffix.lower() in FORBIDDEN_SUFFIXES for r in relatives)
    assert not any(r.startswith(("dist/", "retrieval/index/", "datasets/raw/", "results/predictions")) for r in relatives)
    assert "src/vrfraudnet/models/grammar.py" in relatives and "docs/REVIEWER_REPRODUCIBILITY.md" in relatives


def test_forbidden_content_scanner_catches_planted_secret(tmp_path):
    planted = tmp_path / "notes.md"
    planted.write_text("token = ghp_" + "A" * 36, encoding="utf-8")
    problems = forbidden_content_problems([BundleFile("notes.md", planted)])
    assert any("GitHub personal access token" in p for p in problems)
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"")
    assert forbidden_content_problems([BundleFile("adapter_model.safetensors", weights)])


def test_archive_builds_deterministically_and_verifies(repo_root, tmp_path):
    files = collect_bundle_files(repo_root)
    manifest = build_manifest(repo_root, version="9.9.9-test", files=files,
                              created_utc="2026-01-01T00:00:00+00:00")
    first = write_archive(files, manifest=manifest, output=tmp_path / "a.tar.gz", mtime=1_700_000_000)
    second = write_archive(files, manifest=manifest, output=tmp_path / "b.tar.gz", mtime=1_700_000_000)
    assert first["sha256"] == second["sha256"]
    report = verify_archive(tmp_path / "a.tar.gz", expected_seeds=list(MANUSCRIPT_SEEDS))
    assert report["ok"], report["problems"]
    assert report["file_count"] == len(files) + 2
    for required in REQUIRED_IN_ARCHIVE:
        assert required in manifest["files"] or required in {MANIFEST_FILE, "SHA256SUMS"}


def test_archive_verifier_detects_tampering(repo_root, tmp_path):
    import tarfile
    import gzip
    import io

    files = collect_bundle_files(repo_root)
    manifest = build_manifest(repo_root, version="9.9.9-test", files=files,
                              created_utc="2026-01-01T00:00:00+00:00")
    write_archive(files, manifest=manifest, output=tmp_path / "a.tar.gz", mtime=1)
    with gzip.open(tmp_path / "a.tar.gz", "rb") as gz:
        raw = gz.read()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as src, tarfile.open(fileobj=buffer, mode="w") as dst:
        for member in src.getmembers():
            data = src.extractfile(member).read() if member.isfile() else b""
            if member.name.endswith("configs/manuscript_seeds.yaml"):
                data = data.replace(b"- 42\n", b"- 43\n")
                member.size = len(data)
            dst.addfile(member, io.BytesIO(data))
    with gzip.open(tmp_path / "tampered.tar.gz", "wb") as gz:
        gz.write(buffer.getvalue())
    report = verify_archive(tmp_path / "tampered.tar.gz", expected_seeds=list(MANUSCRIPT_SEEDS))
    assert not report["ok"]
    assert any("manuscript_seeds.yaml" in p for p in report["problems"])


# ---------------------------------------------------------------------------
# 9. CI wiring
# ---------------------------------------------------------------------------

def test_ci_runs_the_reviewer_audits(repo_root):
    ci = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for command in (
        "scripts/audit_reviewer_artifacts.py",
        "scripts/validate_rationale_schema.py",
        "scripts/build_reproducibility_bundle.py --check-manifest",
        "scripts/verify_reproducibility_bundle.py",
        "scripts/train_stage2_lora.py",
        "scripts/audit_manuscript.py",
    ):
        assert command in ci, f"CI does not run {command}"
