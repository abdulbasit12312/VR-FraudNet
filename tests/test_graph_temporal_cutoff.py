"""Stage 0 must never see an edge from after the evaluation time (S4.1)."""

from __future__ import annotations

import numpy as np
import pytest

from vrfraudnet.data.types import TemporalGraph
from vrfraudnet.errors import LeakageError
from vrfraudnet.models.stage0_spectral import (
    Stage0Config,
    apply_polynomial_filter,
    assert_no_future_edges,
    beta_kernel_coefficients,
    normalised_laplacian,
    precompute_filter_bank,
    time_embedding,
)


@pytest.fixture
def graph() -> TemporalGraph:
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_time = np.array([1.0, 2.0, 50.0, 90.0])
    return TemporalGraph(edge_index=edge_index, edge_time=edge_time, num_nodes=5)


def test_snapshot_drops_future_edges(graph):
    snapshot = graph.snapshot(10.0)
    assert snapshot.edge_index.shape[1] == 2
    assert snapshot.edge_time.max() <= 10.0


def test_snapshot_is_monotone(graph):
    sizes = [graph.snapshot(t).edge_index.shape[1] for t in (0.0, 1.5, 60.0, 100.0)]
    assert sizes == sorted(sizes)
    assert sizes[-1] == graph.edge_index.shape[1]


def test_assert_no_future_edges_trips_on_leakage(graph):
    with pytest.raises(LeakageError, match="later than the evaluation time"):
        assert_no_future_edges(graph.edge_time, evaluation_time=10.0)


def test_assert_no_future_edges_passes_on_a_snapshot(graph):
    snapshot = graph.snapshot(10.0)
    assert_no_future_edges(snapshot.edge_time, evaluation_time=10.0)


def test_graph_rejects_out_of_range_node_ids():
    with pytest.raises(ValueError, match="outside"):
        TemporalGraph(
            edge_index=np.array([[0], [99]]), edge_time=np.array([1.0]), num_nodes=5
        )


def test_graph_rejects_mismatched_edge_time_length():
    with pytest.raises(ValueError, match="edge_time length"):
        TemporalGraph(
            edge_index=np.array([[0, 1], [1, 2]]), edge_time=np.array([1.0]), num_nodes=3
        )


def test_normalised_laplacian_has_unit_diagonal():
    edge_index = np.array([[0, 1], [1, 2]])
    indices, values = normalised_laplacian(edge_index, num_nodes=3)
    dense = np.zeros((3, 3))
    np.add.at(dense, (indices[0], indices[1]), values)
    assert np.allclose(np.diag(dense), 1.0)


def test_normalised_laplacian_is_symmetric():
    edge_index = np.array([[0, 1, 0], [1, 2, 2]])
    indices, values = normalised_laplacian(edge_index, num_nodes=3)
    dense = np.zeros((3, 3))
    np.add.at(dense, (indices[0], indices[1]), values)
    assert np.allclose(dense, dense.T)


def test_normalised_laplacian_eigenvalues_lie_in_zero_two():
    rng = np.random.default_rng(0)
    src = rng.integers(0, 20, size=60)
    dst = rng.integers(0, 20, size=60)
    keep = src != dst
    indices, values = normalised_laplacian(np.vstack([src[keep], dst[keep]]), num_nodes=20)
    dense = np.zeros((20, 20))
    np.add.at(dense, (indices[0], indices[1]), values)
    eigenvalues = np.linalg.eigvalsh((dense + dense.T) / 2)
    assert eigenvalues.min() > -1e-8
    assert eigenvalues.max() < 2.0 + 1e-6


def test_beta_kernel_coefficients_have_the_expected_degree():
    coefficients = beta_kernel_coefficients(1, 3)
    assert coefficients.shape[0] == 5  # degree p + q, plus the constant term
    assert coefficients[0] == 0.0  # (L/2)^1 leaves no constant term


def test_beta_kernel_is_zero_at_zero_frequency():
    # A band-pass Beta wavelet with p >= 1 must annihilate the DC component.
    for p in (1, 2, 3):
        coefficients = beta_kernel_coefficients(p, 4 - p)
        assert coefficients[0] == pytest.approx(0.0)


def test_polynomial_filter_is_linear():
    edge_index = np.array([[0, 1], [1, 2]])
    indices, values = normalised_laplacian(edge_index, num_nodes=3)
    coefficients = beta_kernel_coefficients(1, 3)
    a = np.random.default_rng(1).normal(size=(3, 4))
    b = np.random.default_rng(2).normal(size=(3, 4))
    fa = apply_polynomial_filter(indices, values, 3, a, coefficients)
    fb = apply_polynomial_filter(indices, values, 3, b, coefficients)
    fab = apply_polynomial_filter(indices, values, 3, a + b, coefficients)
    assert np.allclose(fa + fb, fab)


def test_filter_bank_shape_matches_the_config():
    config = Stage0Config(n_filters=4, beta_order=4)
    features = np.random.default_rng(3).normal(size=(6, 5))
    edge_index = np.array([[0, 1, 2], [1, 2, 3]])
    bank = precompute_filter_bank(edge_index, 6, features, config)
    assert bank.shape == (4, 6, 5)


def test_filter_bank_depends_only_on_visible_edges(graph):
    config = Stage0Config()
    features = np.random.default_rng(4).normal(size=(5, 3))
    early = precompute_filter_bank(
        graph.snapshot(10.0).edge_index, 5, features, config
    )
    late = precompute_filter_bank(graph.edge_index, 5, features, config)
    assert not np.allclose(early, late), "later edges must change the representation"


def test_time_embedding_is_deterministic_and_bounded():
    embedded = time_embedding(np.array([0.0, 1.0, 100.0]), dim=32)
    again = time_embedding(np.array([0.0, 1.0, 100.0]), dim=32)
    assert embedded.shape == (3, 32)
    assert np.array_equal(embedded, again)
    assert np.all(np.abs(embedded) <= 1.0 + 1e-12)


def test_time_embedding_requires_an_even_dimension():
    with pytest.raises(ValueError, match="even"):
        time_embedding(np.array([1.0]), dim=31)
