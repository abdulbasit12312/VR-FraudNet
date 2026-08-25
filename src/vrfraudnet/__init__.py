"""VR-FraudNet: verifier-grounded fraud detection.

Reference implementation accompanying the manuscript
"Grounding language models with deterministic verifiers for fraud detection".

Scientific-integrity policy for this package
--------------------------------------------
1. No manuscript result is hard-coded anywhere in the runtime path. Table
   builders read *only* result files produced by an actual run.
2. Settings the manuscript states explicitly are marked MANUSCRIPT in the
   configuration files. Settings the manuscript does not state are marked
   ASSUMPTION and are listed in docs/KNOWN_LIMITATIONS.md.
3. Experiments whose specification is internally inconsistent in the manuscript
   are not silently implemented. They raise
   :class:`vrfraudnet.errors.ManuscriptInconsistencyError` with a pointer to
   docs/MANUSCRIPT_AUDIT.md.
"""

__version__ = "0.1.0"

from vrfraudnet.errors import (  # noqa: F401
    ConfigurationError,
    LeakageError,
    ManuscriptInconsistencyError,
    MissingArtefactError,
    NotApplicableError,
)

__all__ = [
    "__version__",
    "ConfigurationError",
    "LeakageError",
    "ManuscriptInconsistencyError",
    "MissingArtefactError",
    "NotApplicableError",
]
