"""PyTorch family-aware tabular encoder for the composite backbone."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .constants import CLASS_10
from .feature_families import FeatureFamilySpec


@dataclass(frozen=True)
class FamilyEncoderTrainConfig:
    token_dim: int = 64
    hidden_dim: int = 192
    dropout: float = 0.10
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hard_label_weight: float = 0.20
    distill_weight: float = 1.00
    patience: int = 15


class FamilyEncoderNet(nn.Module):
    def __init__(
        self,
        family_dims: Mapping[str, int],
        *,
        token_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.family_names = tuple(family_dims.keys())
        self.family_encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, token_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(token_dim, token_dim),
                    nn.ReLU(),
                )
                for name, dim in family_dims.items()
            }
        )
        fused_dim = token_dim * len(self.family_names) + len(self.family_names)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.global_head = nn.Linear(hidden_dim, len(CLASS_10))

    def forward(
        self,
        families: Mapping[str, torch.Tensor],
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        for index, name in enumerate(self.family_names):
            token = self.family_encoders[name](families[name])
            token = token * masks[:, index : index + 1]
            tokens.append(token)
        fused = torch.cat([*tokens, masks], dim=1)
        z = self.fusion(fused)
        return self.global_head(z), z


def fit_family_encoder(
    train_arrays: Mapping[str, np.ndarray],
    train_masks: np.ndarray,
    y_train: np.ndarray,
    *,
    val_arrays: Mapping[str, np.ndarray] | None = None,
    val_masks: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    teacher_train: np.ndarray | None = None,
    teacher_mask_train: np.ndarray | None = None,
    teacher_val: np.ndarray | None = None,
    teacher_mask_val: np.ndarray | None = None,
    config: FamilyEncoderTrainConfig | None = None,
    seed: int = 42,
) -> FamilyEncoderNet:
    """Fit a family-aware encoder on split-local tabular feature blocks.

    Parameters
    ----------
    train_arrays
        Mapping from family name to normalized training feature matrix.
    train_masks
        Binary matrix indicating which feature families are present per row.
    y_train
        Integer class labels in ``CLASS_10`` order.
    val_arrays, val_masks, y_val
        Optional validation split used for early stopping.
    teacher_train, teacher_mask_train, teacher_val, teacher_mask_val
        Optional teacher probabilities and availability masks for distillation.
    config
        Training hyperparameters. Defaults to ``FamilyEncoderTrainConfig``.
    seed
        Torch RNG seed used for initialization and batch ordering.

    Returns
    -------
    FamilyEncoderNet
        Trained model restored to the best validation checkpoint.
    """

    cfg = config or FamilyEncoderTrainConfig()
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    family_dims = {name: values.shape[1] for name, values in train_arrays.items()}
    model = FamilyEncoderNet(
        family_dims,
        token_dim=cfg.token_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    counts = np.bincount(y_train, minlength=len(CLASS_10)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    kl = nn.KLDivLoss(reduction="batchmean")

    train_tensors = {
        name: torch.from_numpy(np.ascontiguousarray(train_arrays[name], dtype=np.float32))
        for name in model.family_names
    }
    train_mask_tensor = torch.from_numpy(np.ascontiguousarray(train_masks, dtype=np.float32))
    y_tensor = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.int64))
    if teacher_train is not None:
        teacher_tensor = torch.from_numpy(np.ascontiguousarray(teacher_train, dtype=np.float32))
        teacher_mask_tensor = torch.from_numpy(
            np.ascontiguousarray(
                np.ones(len(y_train), dtype=np.float32)
                if teacher_mask_train is None
                else teacher_mask_train.astype(np.float32),
            )
        )
    else:
        teacher_tensor = None
        teacher_mask_tensor = None

    best_state = None
    best_loss = float("inf")
    stale = 0
    for _epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(y_tensor))
        for start in range(0, len(y_tensor), cfg.batch_size):
            batch_idx = order[start : start + cfg.batch_size]
            family_batch = {name: tensor[batch_idx] for name, tensor in train_tensors.items()}
            mask_batch = train_mask_tensor[batch_idx]
            y_batch = y_tensor[batch_idx]
            logits, _z = model(family_batch, mask_batch)
            loss = cfg.hard_label_weight * ce(logits, y_batch)
            if teacher_tensor is not None and teacher_mask_tensor is not None:
                teacher_batch = teacher_tensor[batch_idx]
                teacher_mask_batch = teacher_mask_tensor[batch_idx] > 0.5
                if bool(teacher_mask_batch.any()):
                    loss = loss + cfg.distill_weight * kl(
                        torch.log_softmax(logits[teacher_mask_batch], dim=1),
                        teacher_batch[teacher_mask_batch],
                    )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = _validation_loss(
            model,
            val_arrays,
            val_masks,
            y_val,
            ce,
            kl,
            hard_label_weight=cfg.hard_label_weight,
            distill_weight=cfg.distill_weight,
            teacher_val=teacher_val,
            teacher_mask_val=teacher_mask_val,
        )
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_family_encoder_proba(
    model: FamilyEncoderNet,
    arrays: Mapping[str, np.ndarray],
    masks: np.ndarray,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    """Predict class probabilities with a trained family encoder.

    Parameters
    ----------
    model
        Trained family encoder.
    arrays
        Mapping from family name to normalized feature matrix.
    masks
        Binary family-presence mask aligned to ``arrays``.
    batch_size
        Number of rows to score per forward pass.

    Returns
    -------
    np.ndarray
        Probability matrix with columns in ``CLASS_10`` order.
    """

    proba, _z = predict_family_encoder_outputs(
        model,
        arrays,
        masks,
        batch_size=batch_size,
    )
    return proba


def predict_family_encoder_outputs(
    model: FamilyEncoderNet,
    arrays: Mapping[str, np.ndarray],
    masks: np.ndarray,
    *,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict class probabilities and latent embeddings.

    Parameters
    ----------
    model
        Trained family encoder.
    arrays
        Mapping from family name to normalized feature matrix.
    masks
        Binary family-presence mask aligned to ``arrays``.
    batch_size
        Number of rows to score per forward pass.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(probabilities, embeddings)`` where probabilities follow
        ``CLASS_10`` order and embeddings are the fused hidden representation.
    """

    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(arrays[name], dtype=np.float32))
        for name in model.family_names
    }
    mask_tensor = torch.from_numpy(np.ascontiguousarray(masks, dtype=np.float32))
    proba_chunks: list[np.ndarray] = []
    z_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(mask_tensor), batch_size):
            stop = start + batch_size
            families = {name: tensor[start:stop] for name, tensor in tensors.items()}
            logits, _z = model(families, mask_tensor[start:stop])
            proba_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
            z_chunks.append(_z.cpu().numpy())
    return (
        np.vstack(proba_chunks).astype(np.float64),
        np.vstack(z_chunks).astype(np.float32),
    )


def _validation_loss(
    model: FamilyEncoderNet,
    val_arrays: Mapping[str, np.ndarray] | None,
    val_masks: np.ndarray | None,
    y_val: np.ndarray | None,
    ce: nn.CrossEntropyLoss,
    kl: nn.KLDivLoss,
    *,
    hard_label_weight: float,
    distill_weight: float,
    teacher_val: np.ndarray | None,
    teacher_mask_val: np.ndarray | None,
) -> float:
    if val_arrays is None or val_masks is None or y_val is None or len(y_val) == 0:
        return 0.0
    with torch.no_grad():
        families = {
            name: torch.from_numpy(np.ascontiguousarray(val_arrays[name], dtype=np.float32))
            for name in model.family_names
        }
        logits, _z = model(
            families,
            torch.from_numpy(np.ascontiguousarray(val_masks, dtype=np.float32)),
        )
        loss = hard_label_weight * ce(
            logits,
            torch.from_numpy(np.ascontiguousarray(y_val, dtype=np.int64)),
        )
        if teacher_val is not None:
            teacher_tensor = torch.from_numpy(np.ascontiguousarray(teacher_val, dtype=np.float32))
            if teacher_mask_val is None:
                mask = torch.ones(len(y_val), dtype=torch.bool)
            else:
                mask = (
                    torch.from_numpy(np.ascontiguousarray(teacher_mask_val.astype(np.float32)))
                    > 0.5
                )
            if bool(mask.any()):
                loss = loss + distill_weight * kl(
                    torch.log_softmax(logits[mask], dim=1),
                    teacher_tensor[mask],
                )
        return float(loss.item())


def family_dims_from_specs(specs: list[FeatureFamilySpec]) -> dict[str, int]:
    """Return model input dimensions for a resolved feature-family list.

    Parameters
    ----------
    specs
        Feature family specifications used by a composite backbone.

    Returns
    -------
    dict[str, int]
        Mapping from family name to number of input columns.
    """

    return {spec.name: len(spec.columns) for spec in specs}
