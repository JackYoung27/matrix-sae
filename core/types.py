from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch

# Config value type: JSON-safe scalars written by train() and read by build_sae_from_config().
ConfigValue = str | int | float | bool | None

@dataclass
class SAEOutput:
    reconstruction: torch.Tensor
    coefficients: torch.Tensor
    loss: torch.Tensor
    mse: torch.Tensor
    aux_loss: torch.Tensor
    n_dead: int

class SAEConfig(TypedDict, total=False):
    """Shape of the ``config`` dict that train() writes and build_sae_from_config() reads.

    All fields optional because consumers (e.g. ``infer_*``) tolerate missing keys and
    fall back to inference from the state dict or defaults.
    """
    sae_type: str
    layer: int
    head: int
    n_features: int
    d_k: int
    d_v: int
    d_in: int
    expansion_factor: int
    k: int
    rank: int
    use_batchtopk: bool
    seed: int
    lr: float
    lr_min: float
    batch_size: int
    epochs: int
    total_steps: int
    n_params: int
    device: str
    code_sha: str

class SAECheckpoint(TypedDict, total=False):
    """Payload format written by ``torch.save`` in core.train.

    ``model_state_dict`` is always present in checkpoints produced by train(); the
    other keys depend on which writer path produced the file (``_save`` vs the
    resume writer).
    """
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, object]
    config: SAEConfig
    step: int
    epoch: int
    val_mse: float
    best_val_mse: float

class TrainResult(TypedDict):
    """Return value of ``core.train.train``."""
    sae_type: str
    layer: int
    head: int
    expansion_factor: int
    k: int
    rank: int
    seed: int
    code_sha: str
    n_features: int
    n_samples: int
    best_mse: float
    final_mse: float
    final_n_dead: int
    total_time_s: float
