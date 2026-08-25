"""Neural baselines: MLP, 1D-CNN, LSTM and TabTransformer (manuscript S4.7).

Every architectural choice in this file is an ASSUMPTION (L-18): the manuscript
names these baselines but publishes no hyperparameters for them. The
implementations are deliberately small, conventional and readable, so that when
the authors supply their real configurations only the numbers in
``configs/baselines.yaml`` need to change.

All four share the same training loop: focal loss with class-balanced weights
(matching the imbalance handling the manuscript applies uniformly in Table 2),
AdamW, early stopping on validation AUPRC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vrfraudnet.errors import MissingArtefactError


@dataclass
class NeuralTrainingConfig:
    """Shared training loop settings (ASSUMPTION L-18)."""

    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 50
    patience: int = 10
    focal_gamma: float = 2.0
    n_heads: int = 8  # TabTransformer only
    grad_clip_norm: float = 1.0


class TorchTabularBaseline:
    """A scikit-learn-shaped wrapper around a small torch model."""

    def __init__(self, kind: str, seed: int, config: NeuralTrainingConfig) -> None:
        self.kind = kind
        self.seed = seed
        self.config = config
        self.model_: Any = None
        self.in_dim_: int | None = None

    # -- sklearn-compatible surface ----------------------------------------
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> "TorchTabularBaseline":
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from vrfraudnet.determinism import set_global_seed
        from vrfraudnet.losses.cost_sensitive import class_balanced_weights
        from vrfraudnet.losses.focal import torch_focal_loss

        set_global_seed(self.seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.in_dim_ = int(x.shape[1])
        self.model_ = _build(self.kind, self.in_dim_, self.config)

        weights = class_balanced_weights(y.astype(int)).astype(np.float32)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(weights)
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        optimiser = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_score, best_state, stale = -np.inf, None, 0
        for _ in range(self.config.max_epochs):
            self.model_.train()
            for xb, yb, wb in loader:
                optimiser.zero_grad()
                logits = self.model_(xb).squeeze(-1)
                loss = torch_focal_loss(logits, yb, gamma=self.config.focal_gamma, weight=wb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), self.config.grad_clip_norm)
                optimiser.step()

            if x_valid is None or y_valid is None:
                continue
            from sklearn.metrics import average_precision_score

            score = float(average_precision_score(y_valid, self.predict_proba(x_valid)[:, 1]))
            if score > best_score:
                best_score, stale = score, 0
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
            else:
                stale += 1
                if stale >= self.config.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch

        if self.model_ is None:
            raise RuntimeError("model used before fit")
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(np.asarray(x, dtype=np.float32))).squeeze(-1)
            p = torch.sigmoid(logits).numpy()
        return np.column_stack([1.0 - p, p])


def _build(kind: str, in_dim: int, config: NeuralTrainingConfig):
    """Construct the requested architecture."""
    from torch import nn

    if kind == "mlp":
        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(config.n_layers):
            layers += [nn.Linear(width, config.hidden_dim), nn.ReLU(), nn.Dropout(config.dropout)]
            width = config.hidden_dim
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    if kind == "cnn1d":
        class Conv1DBaseline(nn.Module):
            """Treats the feature vector as a length-F single-channel sequence."""

            def __init__(self) -> None:
                super().__init__()
                self.body = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.head = nn.Sequential(
                    nn.Flatten(), nn.Dropout(config.dropout), nn.Linear(64, 1)
                )

            def forward(self, x):
                return self.head(self.body(x.unsqueeze(1)))

        return Conv1DBaseline()

    if kind == "lstm":
        class LSTMBaseline(nn.Module):
            """Reads the feature vector as a length-F sequence of scalars."""

            def __init__(self) -> None:
                super().__init__()
                self.rnn = nn.LSTM(
                    input_size=1,
                    hidden_size=config.hidden_dim,
                    num_layers=config.n_layers,
                    batch_first=True,
                    dropout=config.dropout if config.n_layers > 1 else 0.0,
                )
                self.head = nn.Linear(config.hidden_dim, 1)

            def forward(self, x):
                out, _ = self.rnn(x.unsqueeze(-1))
                return self.head(out[:, -1])

        return LSTMBaseline()

    if kind == "tabtransformer":
        class TabTransformerBaseline(nn.Module):
            """Self-attention over per-feature embeddings.

            The original TabTransformer applies attention to *categorical*
            embeddings and concatenates continuous features. After the
            manuscript's preprocessing, every column reaching the model is
            numeric, so each feature is projected to an embedding and attended
            over. That divergence is deliberate and recorded as L-18.
            """

            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(1, config.hidden_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.n_heads,
                    dim_feedforward=config.hidden_dim * 2,
                    dropout=config.dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
                self.head = nn.Sequential(
                    nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 1)
                )

            def forward(self, x):
                tokens = self.embed(x.unsqueeze(-1))
                encoded = self.encoder(tokens)
                return self.head(encoded.mean(dim=1))

        return TabTransformerBaseline()

    raise ValueError(f"unknown neural baseline kind {kind!r}")


def build_neural_baseline(kind: str, *, seed: int, **params: Any) -> TorchTabularBaseline:
    """Factory used by :func:`vrfraudnet.baselines.registry.build_baseline`."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingArtefactError(
            f"the {kind} baseline requires PyTorch. Install it with: pip install torch"
        ) from exc
    config = NeuralTrainingConfig(**params)
    return TorchTabularBaseline(kind, seed, config)
