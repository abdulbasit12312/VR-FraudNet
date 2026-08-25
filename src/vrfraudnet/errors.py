"""Typed errors used across the package.

Distinct error types matter here: a reviewer auditing the repository should be
able to tell at a glance whether a failure is a bug, a missing artefact, a
manuscript specification problem, or a deliberate refusal to run an invalid
experiment.
"""

from __future__ import annotations


class VRFraudNetError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(VRFraudNetError):
    """A configuration value is missing, malformed, or mutually inconsistent."""


class MissingArtefactError(VRFraudNetError):
    """A required input artefact (dataset, checkpoint, prediction file) is absent.

    Raised instead of silently substituting synthetic data. The message must
    always name the exact path expected and the command that produces it.
    """


class LeakageError(VRFraudNetError):
    """A leakage guard tripped.

    Raised when a transformation would fit on non-training data, when a future
    observation would enter a training partition, or when a retrieval index
    would contain a validation or test record.
    """


class NotApplicableError(VRFraudNetError):
    """An operation is not defined for the given dataset or model.

    Example: the prompt-injection edit family is not applicable to a dataset
    with no untrusted free-text field, nor to a model that does not consume
    text. See docs/MANUSCRIPT_AUDIT.md finding A-06.
    """


class ManuscriptInconsistencyError(VRFraudNetError):
    """The manuscript specification of this experiment is internally inconsistent.

    Raised deliberately rather than implementing a guess. The message names the
    audit finding ID in docs/MANUSCRIPT_AUDIT.md.
    """

    def __init__(self, finding_id: str, message: str) -> None:
        super().__init__(
            f"[{finding_id}] {message}\n"
            f"See docs/MANUSCRIPT_AUDIT.md, finding {finding_id}, for the full "
            f"analysis and the clarification required from the authors."
        )
        self.finding_id = finding_id
