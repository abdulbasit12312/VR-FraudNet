"""Stage 4: verifier gating, isotonic mixing and split-conformal calibration."""

from __future__ import annotations

import numpy as np
import pytest

from vrfraudnet.errors import ConfigurationError, LeakageError
from vrfraudnet.models.stage4_calibration import (
    IsotonicMixer,
    MixerInputs,
    SplitConformalCalibrator,
)


@pytest.fixture
def calibration_data():
    rng = np.random.default_rng(4)
    n = 4000
    y = (rng.uniform(size=n) < 0.15).astype(int)
    p = np.clip(rng.beta(2, 8, size=n) + 0.5 * y, 0.0, 1.0)
    return p, y


def test_rejected_rationale_contributes_no_probability_mass():
    triage = np.array([0.2, 0.3, 0.4])
    rationale = np.array([0.9, 0.9, 0.9])
    status = np.array([1, 0, 0])
    gated = MixerInputs(triage, rationale, status).gated_rationale()
    assert gated.tolist() == [0.9, 0.0, 0.0]


def test_rejection_status_is_still_a_mixer_input():
    design = MixerInputs(
        np.array([0.2, 0.3]), np.array([0.8, 0.8]), np.array([1, 0])
    ).design_matrix()
    assert design.shape == (2, 3)
    assert design[:, 2].tolist() == [0.0, 1.0]


def test_mixer_refuses_to_fit_on_test_labels():
    inputs = MixerInputs(np.array([0.1, 0.9]), None, None)
    with pytest.raises(LeakageError, match="no test labels"):
        IsotonicMixer().fit(inputs, np.array([0, 1]), partition="test")


def test_mixer_output_is_monotone_in_the_triage_score(calibration_data):
    p, y = calibration_data
    mixer = IsotonicMixer().fit(MixerInputs(p, None, None), y)
    grid = np.linspace(0.0, 1.0, 50)
    calibrated = mixer.predict(MixerInputs(grid, None, None))
    assert np.all(np.diff(calibrated) >= -1e-9)


def test_mixer_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        IsotonicMixer().predict(MixerInputs(np.array([0.5]), None, None))


def test_conformal_alpha_must_be_a_probability():
    with pytest.raises(ConfigurationError):
        SplitConformalCalibrator(alpha=0.0)
    with pytest.raises(ConfigurationError):
        SplitConformalCalibrator(alpha=1.0)


def test_conformal_refuses_a_tiny_calibration_set():
    calibrator = SplitConformalCalibrator(alpha=0.05)
    with pytest.raises(ConfigurationError, match="meaningless quantile"):
        calibrator.calibrate(np.linspace(0, 1, 10), np.array([0, 1] * 5))


def test_conformal_empirical_coverage_is_near_nominal(calibration_data):
    p, y = calibration_data
    half = p.size // 2
    calibrator = SplitConformalCalibrator(alpha=0.10)
    calibrator.calibrate(p[:half], y[:half])
    coverage = calibrator.evaluate_coverage(p[half:], y[half:])
    assert coverage["empirical_coverage"] == pytest.approx(0.90, abs=0.04)


def test_conformal_miscoverage_on_the_calibration_set_respects_alpha(calibration_data):
    p, y = calibration_data
    result = SplitConformalCalibrator(alpha=0.05).calibrate(p, y)
    assert result.empirical_miscoverage <= 0.05 + 1e-9


def test_nonconformity_is_one_minus_observed_class_probability():
    calibrator = SplitConformalCalibrator(alpha=0.05)
    scores = calibrator.nonconformity(np.array([0.8, 0.8]), np.array([1, 0]))
    assert scores.tolist() == pytest.approx([0.2, 0.8])


def test_verifier_aware_variant_widens_rejected_cases():
    calibrator = SplitConformalCalibrator(alpha=0.05, verifier_penalty=0.2)
    plain = calibrator.nonconformity(np.array([0.8]), np.array([1]))
    penalised = calibrator.nonconformity(
        np.array([0.8]), np.array([1]), verifier_status=np.array([0])
    )
    assert penalised[0] > plain[0]


def test_prediction_set_is_ambiguous_near_the_boundary(calibration_data):
    p, y = calibration_data
    calibrator = SplitConformalCalibrator(alpha=0.01)
    calibrator.calibrate(p, y)
    sizes = {len(calibrator.prediction_set(float(v))) for v in np.linspace(0.05, 0.95, 40)}
    assert sizes, "prediction sets should be computable across the probability range"


def test_prediction_set_before_calibration_raises():
    with pytest.raises(RuntimeError, match="before calibrate"):
        SplitConformalCalibrator().prediction_set(0.5)
