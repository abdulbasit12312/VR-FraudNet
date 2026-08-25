"""Config provenance, dataset registry, and result-file integrity."""

from __future__ import annotations

import json

import pytest

from vrfraudnet.config import Config, Provenance, load_config, summarise_assumptions
from vrfraudnet.data.registry import (
    DATASETS,
    PRIMARY_DATASETS,
    STRESS_DATASETS,
    get_spec,
    graph_datasets,
)
from vrfraudnet.errors import ConfigurationError, MissingArtefactError
from vrfraudnet.io_utils import RunManifest, read_result, require_file, write_result
from vrfraudnet.models.stage1_triage import MANUSCRIPT_THRESHOLDS


DATASET_CONFIGS = ("d1", "d2", "d3", "d4", "d5")


@pytest.mark.parametrize("name", DATASET_CONFIGS)
def test_every_dataset_config_loads(repo_root, name):
    config = load_config(repo_root / "configs" / f"{name}.yaml")
    assert config.require("dataset.id") == name.upper()


@pytest.mark.parametrize("name", DATASET_CONFIGS)
def test_configs_inherit_the_base_stage1_settings(repo_root, name):
    config = load_config(repo_root / "configs" / f"{name}.yaml")
    assert config.require("stage1.n_estimators") == 500
    assert config.require("stage1.num_leaves") == 64
    assert config.require("stage1.max_depth") == 8


@pytest.mark.parametrize("name", DATASET_CONFIGS)
def test_routing_thresholds_match_the_manuscript(repo_root, name):
    config = load_config(repo_root / "configs" / f"{name}.yaml")
    gate = MANUSCRIPT_THRESHOLDS[name.upper()]
    assert config.require("stage1.threshold_low") == pytest.approx(gate.t_low)
    assert config.require("stage1.threshold_high") == pytest.approx(gate.t_high)


@pytest.mark.parametrize("name", DATASET_CONFIGS)
def test_every_assumption_names_a_limitation_id(repo_root, name):
    config = load_config(repo_root / "configs" / f"{name}.yaml")
    for dotted, record in config.assumptions().items():
        assert record.limitation.startswith("L-"), f"{dotted} has no limitation id"


@pytest.mark.parametrize("name", DATASET_CONFIGS)
def test_every_assumption_appears_in_known_limitations(repo_root, name):
    config = load_config(repo_root / "configs" / f"{name}.yaml")
    text = (repo_root / "docs" / "KNOWN_LIMITATIONS.md").read_text(encoding="utf-8")
    for dotted, record in config.assumptions().items():
        assert record.limitation in text, (
            f"{dotted} cites {record.limitation}, which is missing from "
            f"docs/KNOWN_LIMITATIONS.md"
        )


def test_manuscript_provenance_requires_a_source():
    with pytest.raises(ConfigurationError, match="must name the manuscript section"):
        Provenance(tag="MANUSCRIPT")


def test_assumption_provenance_requires_a_limitation_id():
    with pytest.raises(ConfigurationError, match="KNOWN_LIMITATIONS"):
        Provenance(tag="ASSUMPTION")


def test_unknown_provenance_tag_is_rejected():
    with pytest.raises(ConfigurationError, match="unknown provenance tag"):
        Provenance(tag="VIBES")


def test_missing_config_key_raises():
    config = Config({"a": {"b": 1}}, {})
    assert config.require("a.b") == 1
    with pytest.raises(ConfigurationError, match="missing configuration key"):
        config.require("a.c")


def test_summarise_assumptions_is_informative(repo_root):
    config = load_config(repo_root / "configs" / "d3.yaml")
    text = summarise_assumptions(config)
    assert "NOT specified by the manuscript" in text
    assert "L-" in text


def test_registry_covers_all_five_datasets():
    assert set(DATASETS) == {"D1", "D2", "D3", "D4", "D5"}
    assert PRIMARY_DATASETS == ("D1", "D2", "D3")
    assert STRESS_DATASETS == ("D4", "D5")


def test_graph_datasets_match_the_manuscript():
    assert set(graph_datasets()) == {"D2", "D4", "D5"}


def test_recall_ceiling_is_binding_only_for_d3():
    binding = {d for d in DATASETS if get_spec(d).recall_at_top_1pct_ceiling < 0.5}
    assert binding == {"D3", "D4"}
    assert get_spec("D3").recall_at_top_1pct_ceiling == pytest.approx(0.2857, abs=1e-4)


def test_no_dataset_declares_an_untrusted_text_field():
    """Underpins audit finding A-06."""
    assert all(not spec.has_free_text_field for spec in DATASETS.values())


def test_result_files_round_trip(tmp_path):
    manifest = RunManifest(
        experiment="unit-test", dataset="D1", model="vr_fraudnet", seed=42,
        config_path="configs/d1.yaml", config_sha256="0" * 64, partition="test",
    )
    path = write_result(tmp_path / "r.json", manifest, {"auprc": 0.5})
    document = read_result(path)
    assert document["results"]["auprc"] == 0.5
    assert document["manifest"]["seed"] == 42


def test_result_files_are_not_silently_overwritten(tmp_path):
    manifest = RunManifest(
        experiment="unit-test", dataset="D1", model="m", seed=1,
        config_path="c", config_sha256="0" * 64, partition="test",
    )
    write_result(tmp_path / "r.json", manifest, {"a": 1})
    with pytest.raises(FileExistsError, match="never silently clobbered"):
        write_result(tmp_path / "r.json", manifest, {"a": 2})
    write_result(tmp_path / "r.json", manifest, {"a": 2}, overwrite=True)


def test_missing_result_file_names_the_producing_command(tmp_path):
    with pytest.raises(MissingArtefactError, match="never fabricates results"):
        read_result(tmp_path / "absent.json")


def test_require_file_names_the_producing_command(tmp_path):
    with pytest.raises(MissingArtefactError, match="python scripts/train.py"):
        require_file(tmp_path / "nope.json", produced_by="python scripts/train.py ...")


def test_manifest_excludes_user_identifying_fields(tmp_path):
    manifest = RunManifest(
        experiment="e", dataset="D1", model="m", seed=1,
        config_path="c", config_sha256="0" * 64, partition="test",
    )
    payload = json.dumps(manifest.to_dict())
    for forbidden in ("username", "hostname", "home", "user"):
        assert forbidden not in payload.lower(), f"{forbidden} leaked into the manifest"
