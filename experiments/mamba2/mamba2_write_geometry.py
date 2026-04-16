#!/usr/bin/env python3
"""Mamba-2 vs GDN write-geometry analysis.

Stage 1: extract exact Mamba-2 (k_t, v_t) write factors via a forward-pre hook.
Stage 2: extract exact GDN (k_t, delta_t) write factors by replaying the recurrence.
Stage 3 & 4: compute write-reuse score, paired compactness, temporal persistence,
and correlate with per-layer rank-1 SAE advantage.

No Modal: runs locally on the user's GPU. Paths configurable via CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

MODEL_NAME = "AntonV/mamba2-780m-hf"
GDN_MODEL_NAME = "Qwen/Qwen3.5-0.8B"

MAMBA2_DEFAULT_N_SEQS = 500
GDN_DEFAULT_N_SEQS = 500
DEFAULT_SEQ_LEN = 1024
MAMBA2_BATCH_SIZE = 2
GDN_BATCH_SIZE = 2
HEAD_IDX = 0
WRITE_ANALYSIS_SAMPLE_SIZE = 2048
TEMPORAL_LAGS = (1, 4, 16, 64)

MAMBA2_LAYERS = [0, 6, 14, 31, 46, 47]
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]

# Packaged bundled reference data (spectral predictors). These are needed to
# correlate write-geometry metrics with rank-1 advantages. Under Modal they were
# copied to /root; locally they live in <script_dir>/results/data.
_SPECTRAL_MAMBA2_DEFAULT = _SCRIPT_DIR / "results" / "data" / "spectral_audit_mamba2.json"
_SPECTRAL_18LAYER_DEFAULT = _SCRIPT_DIR / "results" / "data" / "spectral_18layer_correlation.json"

def _code_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_SCRIPT_DIR,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (FileNotFoundError, OSError):
        return "unknown"

CODE_SHA = _code_sha()

def _apply_padding_mask(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[0] > 1 and attention_mask.shape[1] > 1:
        return (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    return hidden_states

def _effective_rank(eigenvalues: np.ndarray, eps: float = 1e-12) -> float:
    vals = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    total = float(vals.sum())
    if total <= eps:
        return 0.0
    p = vals / total
    entropy = -(p * np.log(p + eps)).sum()
    return float(np.exp(entropy))

def _write_reuse_score(k_vectors: np.ndarray, v_vectors: np.ndarray) -> dict[str, Any]:
    n = k_vectors.shape[0]
    k = k_vectors.astype(np.float64)
    v = v_vectors.astype(np.float64)

    k_normed = k / np.clip(np.linalg.norm(k, axis=1, keepdims=True), 1e-12, None)
    v_normed = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)

    c_k = (k_normed.T @ k_normed) / n
    c_v = (v_normed.T @ v_normed) / n

    eigs_k = np.linalg.eigvalsh(c_k)
    eigs_v = np.linalg.eigvalsh(c_v)

    erank_k = _effective_rank(eigs_k)
    erank_v = _effective_rank(eigs_v)
    score = 1.0 / np.sqrt(max(erank_k, 1e-12) * max(erank_v, 1e-12))

    return {
        "erank_k": erank_k,
        "erank_v": erank_v,
        "write_reuse_score": float(score),
        "top5_eigs_k": sorted(eigs_k.tolist(), reverse=True)[:5],
        "top5_eigs_v": sorted(eigs_v.tolist(), reverse=True)[:5],
        "n_tokens": n,
        "d_k": int(k_vectors.shape[1]),
        "d_v": int(v_vectors.shape[1]),
    }

def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)

def _write_process_metrics(
    k_vectors: np.ndarray,
    v_vectors: np.ndarray,
    *,
    n_sequences: int | None,
    seq_len: int | None,
    sample_size: int = WRITE_ANALYSIS_SAMPLE_SIZE,
    temporal_lags: tuple[int, ...] = TEMPORAL_LAGS,
    seed: int = 0,
) -> dict[str, Any]:
    valid = (np.linalg.norm(k_vectors, axis=1) > 1e-8) & (np.linalg.norm(v_vectors, axis=1) > 1e-8)
    k_valid = k_vectors[valid].astype(np.float64, copy=False)
    v_valid = v_vectors[valid].astype(np.float64, copy=False)
    if k_valid.shape[0] < 2:
        return {}

    rng = np.random.default_rng(seed)
    sample_n = min(sample_size, k_valid.shape[0])
    if sample_n < k_valid.shape[0]:
        sample_idx = np.sort(rng.choice(k_valid.shape[0], size=sample_n, replace=False))
    else:
        sample_idx = np.arange(sample_n)

    k_sample = _normalize_rows(k_valid[sample_idx])
    v_sample = _normalize_rows(v_valid[sample_idx])

    # For x_t = vec(v_t k_t^T), the Gram matrix factorizes exactly:
    # <x_i, x_j> = <k_i, k_j> * <v_i, v_j>.
    gram_k = k_sample @ k_sample.T
    gram_v = v_sample @ v_sample.T
    gram = (gram_k * gram_v) / sample_n
    eigs = np.linalg.eigvalsh((gram + gram.T) * 0.5)
    eigs = np.clip(eigs, 0.0, None)
    eig_sum = float(eigs.sum())
    write_vec_erank = _effective_rank(eigs)
    write_vec_top1 = float(eigs[-1] / eig_sum) if eig_sum > 1e-12 else 0.0

    metrics: dict[str, Any] = {
        "write_vec_erank": write_vec_erank,
        "write_vec_top1_energy_fraction": write_vec_top1,
        "write_vec_sample_size": int(sample_n),
    }

    if n_sequences is None or seq_len is None:
        return metrics

    usable = min(k_vectors.shape[0], v_vectors.shape[0], n_sequences * seq_len)
    if usable < 2:
        return metrics

    k_full = k_vectors[:usable].astype(np.float64, copy=False)
    v_full = v_vectors[:usable].astype(np.float64, copy=False)
    valid_full = (np.linalg.norm(k_full, axis=1) > 1e-8) & (np.linalg.norm(v_full, axis=1) > 1e-8)

    n_seq_usable = usable // seq_len
    usable = n_seq_usable * seq_len
    if usable < 2:
        return metrics

    k_full = _normalize_rows(k_full[:usable]).reshape(n_seq_usable, seq_len, -1)
    v_full = _normalize_rows(v_full[:usable]).reshape(n_seq_usable, seq_len, -1)
    valid_full = valid_full[:usable].reshape(n_seq_usable, seq_len)

    persistence_vals: list[float] = []
    for lag in temporal_lags:
        if lag >= seq_len:
            continue
        k_cos = np.sum(k_full[:, :-lag, :] * k_full[:, lag:, :], axis=-1)
        v_cos = np.sum(v_full[:, :-lag, :] * v_full[:, lag:, :], axis=-1)
        joint_cos = k_cos * v_cos
        pair_valid = valid_full[:, :-lag] & valid_full[:, lag:]
        if not np.any(pair_valid):
            continue
        value = float(np.mean(np.abs(joint_cos[pair_valid])))
        metrics[f"temporal_joint_abs_cos_lag{lag}"] = value
        persistence_vals.append(value)

    if persistence_vals:
        metrics["temporal_joint_persistence_mean"] = float(np.mean(persistence_vals))

    return metrics

def _correlation_stat(xs: list[float | None], ys: list[float | None]) -> dict[str, Any]:
    from scipy.stats import spearmanr

    valid = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(valid) < 3:
        return {"spearman_rho": 0.0, "p_value": 1.0, "n": len(valid)}
    x_vals, y_vals = zip(*valid)
    result = spearmanr(x_vals, y_vals)
    rho = float(result.statistic)  # type: ignore[union-attr]
    p = float(result.pvalue)  # type: ignore[union-attr]
    return {"spearman_rho": rho, "p_value": p, "n": len(valid)}

def _stream_openwebtext_batches(tokenizer, n_sequences: int, seq_len: int, batch_size: int):
    from datasets import load_dataset

    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    token_ids: list[int] = []
    target = n_sequences * seq_len * 2
    for row in ds:
        token_ids.extend(tokenizer.encode(row["text"], add_special_tokens=False))
        if len(token_ids) >= target:
            break

    actual_sequences = min(n_sequences, len(token_ids) // seq_len)
    batches = []
    for start in range(0, actual_sequences, batch_size):
        end = min(start + batch_size, actual_sequences)
        seqs = [token_ids[i * seq_len : (i + 1) * seq_len] for i in range(start, end)]
        batches.append(torch.tensor(seqs, dtype=torch.long))
    return batches, actual_sequences, len(token_ids)

def _get_mamba2_layers(model) -> list[Any]:
    if hasattr(model, "backbone") and hasattr(model.backbone, "layers"):
        return list(model.backbone.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise AttributeError("Could not find Mamba-2 layers on model")

def _get_qwen_layers(model) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return list(model.language_model.layers)
    raise AttributeError("Could not find Qwen layers on model")

def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

class Mamba2WriteCapture:
    def __init__(self, model, layer_idx: int, head_idx: int = HEAD_IDX):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.mixer = _get_mamba2_layers(model)[layer_idx].mixer
        self.hidden_states: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None
        self.handle = self.mixer.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError(f"Could not capture hidden_states for Mamba-2 layer {self.layer_idx}")
        self.hidden_states = hidden_states.detach()
        self.attention_mask = kwargs.get("attention_mask")

    @torch.no_grad()
    def pop_write_factors(self) -> tuple[np.ndarray, np.ndarray]:
        if self.hidden_states is None:
            raise RuntimeError(f"No Mamba-2 activations captured for layer {self.layer_idx}")

        hidden_states = _apply_padding_mask(self.hidden_states, self.attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        mixer = self.mixer

        projected_states = mixer.in_proj(hidden_states)
        d_mlp = (
            projected_states.shape[-1]
            - 2 * mixer.intermediate_size
            - 2 * mixer.n_groups * mixer.ssm_state_size
            - mixer.num_heads
        ) // 2
        _, _, _, hidden_states_b_c, dt = projected_states.split(
            [d_mlp, d_mlp, mixer.intermediate_size, mixer.conv_dim, mixer.num_heads],
            dim=-1,
        )

        hidden_states_b_c = hidden_states_b_c.transpose(1, 2)
        hidden_states_b_c = mixer.act(mixer.conv1d(hidden_states_b_c)[..., :seq_len].transpose(1, 2))
        hidden_states_b_c = _apply_padding_mask(hidden_states_b_c, self.attention_mask)

        hidden_ssm, b_factor, _ = torch.split(
            hidden_states_b_c,
            [
                mixer.intermediate_size,
                mixer.n_groups * mixer.ssm_state_size,
                mixer.n_groups * mixer.ssm_state_size,
            ],
            dim=-1,
        )

        dt = F.softplus(dt + mixer.dt_bias)
        dt = torch.clamp(dt, mixer.time_step_limit[0], mixer.time_step_limit[1])

        v_factor = hidden_ssm.reshape(batch_size, seq_len, -1, mixer.head_dim).float()
        b_factor = b_factor.reshape(batch_size, seq_len, -1, mixer.ssm_state_size).float()
        b_factor = b_factor.repeat_interleave(mixer.num_heads // mixer.n_groups, dim=2, output_size=mixer.num_heads)

        # Exact rank-1 write: dBx = (v_factor * dt[..., None]) outer b_factor.
        v_factor = v_factor * dt[..., None]

        k = b_factor[:, :, self.head_idx, :].detach().cpu().numpy()
        v = v_factor[:, :, self.head_idx, :].detach().cpu().numpy()

        self.hidden_states = None
        self.attention_mask = None
        return k.reshape(-1, k.shape[-1]), v.reshape(-1, v.shape[-1])

    def remove(self) -> None:
        self.handle.remove()

class GDNWriteCapture:
    def __init__(self, model, layer_idx: int, head_idx: int = HEAD_IDX):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.gdn = _get_qwen_layers(model)[layer_idx].linear_attn
        self.hidden_states: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None
        self.handle = self.gdn.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError(f"Could not capture hidden_states for GDN layer {self.layer_idx}")
        self.hidden_states = hidden_states.detach()
        self.attention_mask = kwargs.get("attention_mask")

    @torch.no_grad()
    def pop_write_factors(self) -> tuple[np.ndarray, np.ndarray]:
        if self.hidden_states is None:
            raise RuntimeError(f"No GDN activations captured for layer {self.layer_idx}")

        hidden_states = _apply_padding_mask(self.hidden_states, self.attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        gdn = self.gdn

        mixed_qkv = gdn.in_proj_qkv(hidden_states).transpose(1, 2)
        b = gdn.in_proj_b(hidden_states)
        a = gdn.in_proj_a(hidden_states)

        if gdn.causal_conv1d_fn is not None:
            mixed_qkv = gdn.causal_conv1d_fn(
                x=mixed_qkv,
                weight=gdn.conv1d.weight.squeeze(1),
                bias=gdn.conv1d.bias,
                activation=gdn.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(gdn.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        _, key, value = torch.split(
            mixed_qkv,
            [gdn.key_dim, gdn.key_dim, gdn.value_dim],
            dim=-1,
        )

        key = key.reshape(batch_size, seq_len, -1, gdn.head_k_dim).float()
        value = value.reshape(batch_size, seq_len, -1, gdn.head_v_dim).float()
        beta = b.sigmoid().float()
        g = (-gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)).float()

        if gdn.num_v_heads // gdn.num_k_heads > 1:
            key = key.repeat_interleave(gdn.num_v_heads // gdn.num_k_heads, dim=2)

        key = _l2norm(key, dim=-1)
        key_h = key[:, :, self.head_idx, :]
        value_h = value[:, :, self.head_idx, :]
        beta_h = beta[:, :, self.head_idx]
        g_h = g[:, :, self.head_idx]

        recurrent = torch.zeros(
            batch_size,
            gdn.head_k_dim,
            gdn.head_v_dim,
            dtype=torch.float32,
            device=key_h.device,
        )
        k_writes = torch.empty(batch_size, seq_len, gdn.head_k_dim, dtype=torch.float32, device=key_h.device)
        v_writes = torch.empty(batch_size, seq_len, gdn.head_v_dim, dtype=torch.float32, device=key_h.device)

        for t in range(seq_len):
            recurrent = recurrent * g_h[:, t].exp().unsqueeze(-1).unsqueeze(-1)
            kv_mem = (recurrent * key_h[:, t].unsqueeze(-1)).sum(dim=-2)
            delta = (value_h[:, t] - kv_mem) * beta_h[:, t].unsqueeze(-1)
            recurrent = recurrent + key_h[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
            k_writes[:, t] = key_h[:, t]
            v_writes[:, t] = delta

        self.hidden_states = None
        self.attention_mask = None
        k = k_writes.detach().cpu().numpy()
        v = v_writes.detach().cpu().numpy()
        return k.reshape(-1, k.shape[-1]), v.reshape(-1, v.shape[-1])

    def remove(self) -> None:
        self.handle.remove()

def _save_factor_memmaps(
    write_dir: Path,
    layers: list[int],
    k_memmaps: dict[int, np.memmap],
    v_memmaps: dict[int, np.memmap],
    actual_tokens: int,
) -> None:
    for layer_idx in layers:
        layer_dir = write_dir / f"layer_{layer_idx}"
        k_arr = np.array(k_memmaps[layer_idx][:actual_tokens], dtype=np.float32)
        v_arr = np.array(v_memmaps[layer_idx][:actual_tokens], dtype=np.float32)
        np.save(str(layer_dir / "k_vectors.npy"), k_arr)
        np.save(str(layer_dir / "v_vectors.npy"), v_arr)
        del k_memmaps[layer_idx], v_memmaps[layer_idx]
        for tmp in layer_dir.glob("*_tmp.dat"):
            tmp.unlink()

def extract_mamba2_writes(
    *,
    data_root: Path,
    model_cache_dir: Path | None,
    layers: list[int],
    n_sequences: int,
    seq_len: int,
    batch_size: int,
    model_name: str = MODEL_NAME,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    if model_cache_dir is not None:
        os.environ["HF_HOME"] = str(model_cache_dir)
    device = "cuda"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    model_layers = _get_mamba2_layers(model)
    bad_layers = [layer for layer in layers if layer < 0 or layer >= len(model_layers)]
    if bad_layers:
        raise ValueError(f"Invalid layers {bad_layers}; model has layers 0..{len(model_layers) - 1}")

    config = model.config
    head_dim = int(getattr(config, "head_dim", 64))
    state_size = int(getattr(config, "state_size", getattr(config, "ssm_state_size", 128)))
    print(f"Model: {len(model_layers)} layers, state=({head_dim}, {state_size})")

    batches, actual_sequences, n_tokens = _stream_openwebtext_batches(tokenizer, n_sequences, seq_len, batch_size)
    print(f"Collected {n_tokens} tokens, processing {actual_sequences} sequences in {len(batches)} batches")

    write_dir = data_root / "write_vectors"
    write_dir.mkdir(parents=True, exist_ok=True)
    total_tokens = actual_sequences * seq_len

    k_memmaps: dict[int, np.memmap] = {}
    v_memmaps: dict[int, np.memmap] = {}
    for layer_idx in layers:
        layer_dir = write_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        k_memmaps[layer_idx] = np.memmap(
            str(layer_dir / "k_tmp.dat"),
            dtype=np.float32,
            mode="w+",
            shape=(total_tokens, state_size),
        )
        v_memmaps[layer_idx] = np.memmap(
            str(layer_dir / "v_tmp.dat"),
            dtype=np.float32,
            mode="w+",
            shape=(total_tokens, head_dim),
        )

    captures = {layer_idx: Mamba2WriteCapture(model, layer_idx, head_idx=HEAD_IDX) for layer_idx in layers}

    token_offset = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc="Mamba-2 batches")):
            input_ids = batch.to(device)
            batch_tokens = input_ids.shape[0] * input_ids.shape[1]
            _ = model(input_ids=input_ids, use_cache=False)

            for layer_idx, capture in captures.items():
                k_batch, v_batch = capture.pop_write_factors()
                k_memmaps[layer_idx][token_offset : token_offset + batch_tokens] = k_batch
                v_memmaps[layer_idx][token_offset : token_offset + batch_tokens] = v_batch

            token_offset += batch_tokens
            torch.cuda.empty_cache()

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = token_offset / max(elapsed, 1e-6)
                eta = (total_tokens - token_offset) / max(rate, 1e-6)
                print(
                    f"  Batch {batch_idx + 1}/{len(batches)}, "
                    f"{token_offset}/{total_tokens} tokens, "
                    f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                )

    for capture in captures.values():
        capture.remove()

    elapsed = time.time() - t0
    _save_factor_memmaps(write_dir, layers, k_memmaps, v_memmaps, token_offset)

    metadata = {
        "model": model_name,
        "layers": layers,
        "n_sequences": actual_sequences,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "n_write_vectors": token_offset,
        "head": HEAD_IDX,
        "head_dim": head_dim,
        "state_size": state_size,
        "extraction_time_s": elapsed,
        "method": "exact_mamba2_factors_from_full_sequence_forward",
        "code_sha": CODE_SHA,
    }
    with open(write_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Extracted {token_offset} exact Mamba-2 write vectors from {len(layers)} layers "
        f"in {elapsed:.0f}s ({elapsed / max(actual_sequences, 1):.1f}s/seq)"
    )
    return metadata

def _load_mamba2_sae_from_checkpoints(checkpoint_dir: Path, layers: list[int]) -> dict[str, Any]:
    per_layer: dict[str, Any] = {}
    for layer_idx in layers:
        ckpt_base = checkpoint_dir / f"layer_{layer_idx}"
        mse_by_type = {}
        for sae_type in ["bilinear", "flat", "rank1"]:
            mses = []
            for seed in [0, 1, 42]:
                ckpt_path = ckpt_base / f"{sae_type}_s{seed}" / "best.pt"
                if ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    val_mse = ckpt.get("val_mse", ckpt.get("best_val_mse"))
                    if val_mse is not None:
                        mses.append(float(val_mse))
            if mses:
                mse_by_type[sae_type] = {"mean": float(np.mean(mses)), "std": float(np.std(mses))}

        rank1_adv = None
        if "rank1" in mse_by_type and "flat" in mse_by_type:
            r1 = mse_by_type["rank1"]["mean"]
            fl = mse_by_type["flat"]["mean"]
            rank1_adv = (fl - r1) / fl * 100.0 if abs(fl) > 1e-12 else 0.0

        bilinear_adv = None
        if "bilinear" in mse_by_type and "flat" in mse_by_type:
            bi = mse_by_type["bilinear"]["mean"]
            fl = mse_by_type["flat"]["mean"]
            bilinear_adv = (fl - bi) / fl * 100.0 if abs(fl) > 1e-12 else 0.0

        per_layer[str(layer_idx)] = {
            "mse_by_type": mse_by_type,
            "rank1_advantage_pct": rank1_adv,
            "bilinear_advantage_pct": bilinear_adv,
        }
    return per_layer

def analyze_mamba2(
    *,
    data_root: Path,
    checkpoint_dir: Path | None,
    spectral_mamba2_path: Path,
) -> dict:
    write_dir = data_root / "write_vectors"
    meta_path = write_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing write vector metadata: {meta_path}. Run extract first.")
    with open(meta_path) as f:
        meta = json.load(f)

    layers = meta["layers"]
    print(f"Analyzing exact write geometry for {len(layers)} Mamba-2 layers")
    print(f"  {meta['n_write_vectors']} write vectors total per layer")

    write_metrics: dict[int, dict[str, Any]] = {}
    for layer_idx in layers:
        layer_dir = write_dir / f"layer_{layer_idx}"
        k = np.load(str(layer_dir / "k_vectors.npy"))
        v = np.load(str(layer_dir / "v_vectors.npy"))

        valid = (np.linalg.norm(k, axis=1) > 1e-8) & (np.linalg.norm(v, axis=1) > 1e-8)
        k_valid = k[valid]
        v_valid = v[valid]
        if k_valid.shape[0] < 100:
            print(f"  Layer {layer_idx}: only {k_valid.shape[0]} valid writes, skipping")
            continue

        metrics = _write_reuse_score(k_valid, v_valid)
        metrics.update(
            _write_process_metrics(
                k,
                v,
                n_sequences=meta.get("n_sequences"),
                seq_len=meta.get("seq_len"),
            )
        )
        metrics["paired_compactness"] = (
            metrics["erank_k"] * metrics["erank_v"] / max(metrics.get("write_vec_erank", 1e-12), 1e-12)
        )
        metrics["persistent_paired_score"] = (
            metrics["paired_compactness"] * metrics.get("temporal_joint_persistence_mean", 0.0)
        )
        metrics["n_valid"] = int(k_valid.shape[0])
        metrics["n_total"] = int(k.shape[0])
        metrics["frac_valid"] = float(k_valid.shape[0] / k.shape[0])
        write_metrics[layer_idx] = metrics
        print(
            f"  Layer {layer_idx}: erank_k={metrics['erank_k']:.1f}, "
            f"erank_v={metrics['erank_v']:.1f}, "
            f"write_reuse={metrics['write_reuse_score']:.4f}, "
            f"write_vec_erank={metrics.get('write_vec_erank', 0.0):.1f}, "
            f"persist={metrics.get('temporal_joint_persistence_mean', 0.0):.4f}"
        )

    sae_results_path = data_root / "mamba2_sae_results.json"
    if sae_results_path.exists():
        with open(sae_results_path) as f:
            sae_results = json.load(f)
        per_layer_sae = sae_results.get("per_layer", {})
    else:
        print("WARNING: No Mamba-2 SAE results found; loading from checkpoints instead.")
        ckpt_dir = checkpoint_dir or (data_root / "mamba2_checkpoints")
        per_layer_sae = _load_mamba2_sae_from_checkpoints(ckpt_dir, layers)

    spectral_ref: dict = {}
    for path in [data_root / "mamba2_states" / "spectral_audit_mamba2.json", spectral_mamba2_path]:
        if path.exists():
            with open(path) as f:
                payload = json.load(f)
            spectral_ref = payload.get("per_layer", payload)
            break

    rank1_advantages = []
    bilinear_advantages = []
    sv1_sv2_ratios = []
    layer_ids = []

    for layer_idx in layers:
        if layer_idx not in write_metrics:
            continue
        layer_key = str(layer_idx)
        sae_info = per_layer_sae.get(layer_key, {})
        mse_by_type = sae_info.get("mse_by_type", {})

        rank1_adv = sae_info.get("rank1_advantage_pct")
        if rank1_adv is None and "rank1" in mse_by_type and "flat" in mse_by_type:
            r1 = float(mse_by_type["rank1"]["mean"])
            fl = float(mse_by_type["flat"]["mean"])
            rank1_adv = (fl - r1) / fl * 100.0 if abs(fl) > 1e-12 else 0.0

        bilinear_adv = sae_info.get("bilinear_advantage_pct")
        if bilinear_adv is None and "bilinear" in mse_by_type and "flat" in mse_by_type:
            bi = float(mse_by_type["bilinear"]["mean"])
            fl = float(mse_by_type["flat"]["mean"])
            bilinear_adv = (fl - bi) / fl * 100.0 if abs(fl) > 1e-12 else 0.0

        sv_ratio = spectral_ref.get(layer_key, {}).get("sv1_sv2", sae_info.get("sv1_sv2"))

        rank1_advantages.append(rank1_adv)
        bilinear_advantages.append(bilinear_adv)
        sv1_sv2_ratios.append(sv_ratio)
        layer_ids.append(layer_idx)

    metric_keys = [
        "write_reuse_score",
        "write_vec_erank",
        "write_vec_top1_energy_fraction",
        "paired_compactness",
        "persistent_paired_score",
        "temporal_joint_persistence_mean",
        "temporal_joint_abs_cos_lag1",
        "temporal_joint_abs_cos_lag4",
        "temporal_joint_abs_cos_lag16",
        "temporal_joint_abs_cos_lag64",
    ]

    correlations: dict[str, Any] = {
        "sv1sv2_vs_rank1_adv": _correlation_stat(sv1_sv2_ratios, rank1_advantages),
        "sv1sv2_vs_bilinear_adv": _correlation_stat(sv1_sv2_ratios, bilinear_advantages),
    }
    for metric_key in metric_keys:
        metric_values = [write_metrics[layer_idx].get(metric_key) for layer_idx in layer_ids]
        correlations[f"{metric_key}_vs_rank1_adv"] = _correlation_stat(metric_values, rank1_advantages)
        correlations[f"{metric_key}_vs_bilinear_adv"] = _correlation_stat(metric_values, bilinear_advantages)

    output = {
        "model": MODEL_NAME,
        "architecture": "mamba2",
        "layers": layer_ids,
        "write_metrics": {str(l): write_metrics[l] for l in layer_ids},
        "rank1_advantages": {str(l): adv for l, adv in zip(layer_ids, rank1_advantages)},
        "bilinear_advantages": {str(l): adv for l, adv in zip(layer_ids, bilinear_advantages)},
        "sv1_sv2_ratios": {str(l): sv for l, sv in zip(layer_ids, sv1_sv2_ratios)},
        "correlations": correlations,
    }

    out_path = data_root / "write_geometry_mamba2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    _print_arch_summary("MAMBA-2", output)
    return output

def extract_gdn_writes(
    *,
    data_root: Path,
    model_cache_dir: Path | None,
    layers: list[int],
    n_sequences: int,
    seq_len: int,
    batch_size: int,
    model_name: str = GDN_MODEL_NAME,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    if model_cache_dir is not None:
        os.environ["HF_HOME"] = str(model_cache_dir)
    device = "cuda"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    model_layers = _get_qwen_layers(model)
    gdn_layers = [idx for idx, layer in enumerate(model_layers) if hasattr(layer, "linear_attn")]
    bad_layers = [layer for layer in layers if layer < 0 or layer >= len(model_layers)]
    if bad_layers:
        raise ValueError(f"Invalid layers {bad_layers}; model has layers 0..{len(model_layers) - 1}")

    requested_gdn = [layer for layer in layers if layer in gdn_layers]
    skipped_attn = [layer for layer in layers if layer not in gdn_layers]
    if skipped_attn:
        print(f"Skipping attention layers (not GDN): {skipped_attn}")
    if not requested_gdn:
        raise ValueError("No requested layers are GDN layers")

    sample_gdn = model_layers[requested_gdn[0]].linear_attn
    head_k_dim = int(sample_gdn.head_k_dim)
    head_v_dim = int(sample_gdn.head_v_dim)
    print(f"GDN model: {len(model_layers)} layers, processing {len(requested_gdn)} GDN layers")
    print(f"State per head: ({head_k_dim}, {head_v_dim})")

    batches, actual_sequences, n_tokens = _stream_openwebtext_batches(tokenizer, n_sequences, seq_len, batch_size)
    print(f"Collected {n_tokens} tokens, processing {actual_sequences} sequences in {len(batches)} batches")

    write_dir = data_root / "write_vectors"
    write_dir.mkdir(parents=True, exist_ok=True)
    total_tokens = actual_sequences * seq_len

    k_memmaps: dict[int, np.memmap] = {}
    v_memmaps: dict[int, np.memmap] = {}
    for layer_idx in requested_gdn:
        layer_dir = write_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        k_memmaps[layer_idx] = np.memmap(
            str(layer_dir / "k_tmp.dat"),
            dtype=np.float32,
            mode="w+",
            shape=(total_tokens, head_k_dim),
        )
        v_memmaps[layer_idx] = np.memmap(
            str(layer_dir / "v_tmp.dat"),
            dtype=np.float32,
            mode="w+",
            shape=(total_tokens, head_v_dim),
        )

    captures = {layer_idx: GDNWriteCapture(model, layer_idx, head_idx=HEAD_IDX) for layer_idx in requested_gdn}

    token_offset = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc="GDN batches")):
            input_ids = batch.to(device)
            batch_tokens = input_ids.shape[0] * input_ids.shape[1]
            _ = model(input_ids=input_ids, use_cache=False)

            for layer_idx, capture in captures.items():
                k_batch, v_batch = capture.pop_write_factors()
                k_memmaps[layer_idx][token_offset : token_offset + batch_tokens] = k_batch
                v_memmaps[layer_idx][token_offset : token_offset + batch_tokens] = v_batch

            token_offset += batch_tokens
            torch.cuda.empty_cache()

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = token_offset / max(elapsed, 1e-6)
                eta = (total_tokens - token_offset) / max(rate, 1e-6)
                print(
                    f"  Batch {batch_idx + 1}/{len(batches)}, "
                    f"{token_offset}/{total_tokens} tokens, "
                    f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                )

    for capture in captures.values():
        capture.remove()

    elapsed = time.time() - t0
    _save_factor_memmaps(write_dir, requested_gdn, k_memmaps, v_memmaps, token_offset)

    metadata = {
        "model": model_name,
        "layers": requested_gdn,
        "n_sequences": actual_sequences,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "n_write_vectors": token_offset,
        "head": HEAD_IDX,
        "head_k_dim": head_k_dim,
        "head_v_dim": head_v_dim,
        "extraction_time_s": elapsed,
        "method": "exact_gdn_factors_from_full_sequence_forward",
        "code_sha": CODE_SHA,
    }
    with open(write_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Extracted {token_offset} exact GDN write vectors from {len(requested_gdn)} layers "
        f"in {elapsed:.0f}s ({elapsed / max(actual_sequences, 1):.1f}s/seq)"
    )
    return metadata

def analyze_gdn(*, data_root: Path, spectral_18layer_path: Path) -> dict:
    write_dir = data_root / "write_vectors"
    meta_path = write_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing GDN write vector metadata: {meta_path}. Run extract-gdn first.")
    with open(meta_path) as f:
        meta = json.load(f)

    layers = meta["layers"]
    print(f"Analyzing exact write geometry for {len(layers)} GDN layers")

    write_metrics: dict[int, dict[str, Any]] = {}
    for layer_idx in layers:
        layer_dir = write_dir / f"layer_{layer_idx}"
        k = np.load(str(layer_dir / "k_vectors.npy"))
        v = np.load(str(layer_dir / "v_vectors.npy"))

        valid = (np.linalg.norm(k, axis=1) > 1e-8) & (np.linalg.norm(v, axis=1) > 1e-8)
        k_valid = k[valid]
        v_valid = v[valid]
        if k_valid.shape[0] < 100:
            print(f"  Layer {layer_idx}: only {k_valid.shape[0]} valid writes, skipping")
            continue

        metrics = _write_reuse_score(k_valid, v_valid)
        metrics.update(
            _write_process_metrics(
                k,
                v,
                n_sequences=meta.get("n_sequences"),
                seq_len=meta.get("seq_len"),
            )
        )
        metrics["paired_compactness"] = (
            metrics["erank_k"] * metrics["erank_v"] / max(metrics.get("write_vec_erank", 1e-12), 1e-12)
        )
        metrics["persistent_paired_score"] = (
            metrics["paired_compactness"] * metrics.get("temporal_joint_persistence_mean", 0.0)
        )
        metrics["n_valid"] = int(k_valid.shape[0])
        metrics["n_total"] = int(k.shape[0])
        write_metrics[layer_idx] = metrics
        print(
            f"  Layer {layer_idx}: erank_k={metrics['erank_k']:.1f}, "
            f"erank_v={metrics['erank_v']:.1f}, "
            f"write_reuse={metrics['write_reuse_score']:.4f}, "
            f"write_vec_erank={metrics.get('write_vec_erank', 0.0):.1f}, "
            f"persist={metrics.get('temporal_joint_persistence_mean', 0.0):.4f}"
        )

    if not spectral_18layer_path.exists():
        raise FileNotFoundError(f"Missing spectral reference: {spectral_18layer_path}")
    with open(spectral_18layer_path) as f:
        gdn_ref = json.load(f)
    gdn_layers_ref = gdn_ref.get("layers", {})

    rank1_advantages = []
    sv1_sv2_ratios = []
    layer_ids = []

    for layer_idx in layers:
        if layer_idx not in write_metrics:
            continue
        ref = gdn_layers_ref.get(str(layer_idx), {})
        rank1_advantages.append(ref.get("rank1_advantage_pct"))
        sv1_sv2_ratios.append(ref.get("sv1_sv2"))
        layer_ids.append(layer_idx)

    metric_keys = [
        "write_reuse_score",
        "write_vec_erank",
        "write_vec_top1_energy_fraction",
        "paired_compactness",
        "persistent_paired_score",
        "temporal_joint_persistence_mean",
        "temporal_joint_abs_cos_lag1",
        "temporal_joint_abs_cos_lag4",
        "temporal_joint_abs_cos_lag16",
        "temporal_joint_abs_cos_lag64",
    ]

    correlations: dict[str, Any] = {
        "sv1sv2_vs_rank1_adv": _correlation_stat(sv1_sv2_ratios, rank1_advantages),
    }
    for metric_key in metric_keys:
        metric_values = [write_metrics[layer_idx].get(metric_key) for layer_idx in layer_ids]
        correlations[f"{metric_key}_vs_rank1_adv"] = _correlation_stat(metric_values, rank1_advantages)

    output = {
        "model": GDN_MODEL_NAME,
        "architecture": "gdn",
        "layers": layer_ids,
        "write_metrics": {str(l): write_metrics[l] for l in layer_ids},
        "rank1_advantages": {str(l): adv for l, adv in zip(layer_ids, rank1_advantages)},
        "sv1_sv2_ratios": {str(l): sv for l, sv in zip(layer_ids, sv1_sv2_ratios)},
        "correlations": correlations,
    }

    out_path = data_root / "write_geometry_gdn.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    _print_arch_summary("GDN", output)
    return output

def _print_arch_summary(name: str, output: dict[str, Any]) -> None:
    print(f"\n{'=' * 72}")
    print(f"{name} WRITE GEOMETRY ANALYSIS")
    print(f"{'=' * 72}")
    print(
        f"{'Layer':>5}  {'erank_k':>8}  {'erank_v':>8}  {'write_reuse':>12}  "
        f"{'rank1_adv%':>11}  {'sv1/sv2':>8}"
    )
    print("-" * 72)

    write_metrics = output["write_metrics"]
    rank1_adv = output.get("rank1_advantages", {})
    sv_ratios = output.get("sv1_sv2_ratios", {})
    for layer_key in sorted(write_metrics.keys(), key=lambda x: int(x)):
        metrics = write_metrics[layer_key]
        adv = rank1_adv.get(layer_key)
        sv = sv_ratios.get(layer_key)
        adv_str = f"{adv:+.1f}" if adv is not None else "n/a"
        sv_str = f"{sv:.1f}" if sv is not None else "n/a"
        print(
            f"L{layer_key:>4}  {metrics['erank_k']:>8.1f}  {metrics['erank_v']:>8.1f}  "
            f"{metrics['write_reuse_score']:>12.4f}  {adv_str:>11}  {sv_str:>8}"
        )

    print("\nCorrelations:")
    for name_key, stats in output.get("correlations", {}).items():
        print(f"  {name_key}: rho={stats['spearman_rho']:+.3f}, p={stats['p_value']:.4f}, N={stats['n']}")

def combined_summary(*, mamba2_data_root: Path, gdn_data_root: Path) -> dict:
    from scipy.stats import spearmanr

    results = {}
    m2_path = mamba2_data_root / "write_geometry_mamba2.json"
    if m2_path.exists():
        with open(m2_path) as f:
            results["mamba2"] = json.load(f)

    gdn_path = gdn_data_root / "write_geometry_gdn.json"
    if gdn_path.exists():
        with open(gdn_path) as f:
            results["gdn"] = json.load(f)

    if not results:
        print("No results found. Run extraction and analysis stages first.")
        return {}

    print(f"\n{'=' * 80}")
    print("WRITE-REUSE METRIC: CROSS-ARCHITECTURE COMPARISON")
    print(f"{'=' * 80}")

    for arch_name, data in results.items():
        print(f"\n--- {arch_name.upper()} ({data['model']}) ---")
        print(f"{'Layer':>5}  {'erank_k':>8}  {'erank_v':>8}  {'write_reuse':>12}  {'rank1_adv%':>11}")
        print("-" * 52)

        write_metrics = data["write_metrics"]
        rank1_adv = data.get("rank1_advantages", {})
        for layer_key in sorted(write_metrics.keys(), key=lambda x: int(x)):
            metrics = write_metrics[layer_key]
            adv = rank1_adv.get(layer_key)
            adv_str = f"{adv:+.1f}" if adv is not None else "n/a"
            print(
                f"L{layer_key:>4}  {metrics['erank_k']:>8.1f}  {metrics['erank_v']:>8.1f}  "
                f"{metrics['write_reuse_score']:>12.4f}  {adv_str:>11}"
            )

        corr = data.get("correlations", {})
        wr_corr = corr.get("write_reuse_vs_rank1_adv", {})
        sv_corr = corr.get("sv1sv2_vs_rank1_adv", {})
        print(
            f"\n  write_reuse vs rank1_adv: rho={wr_corr.get('spearman_rho', 0):+.3f}, "
            f"p={wr_corr.get('p_value', 1):.4f}, N={wr_corr.get('n', 0)}"
        )
        if sv_corr:
            print(
                f"  sv1/sv2 vs rank1_adv:    rho={sv_corr.get('spearman_rho', 0):+.3f}, "
                f"p={sv_corr.get('p_value', 1):.4f}, N={sv_corr.get('n', 0)}"
            )

    all_ws = []
    all_adv = []
    for data in results.values():
        write_metrics = data["write_metrics"]
        rank1_adv = data.get("rank1_advantages", {})
        for layer_key, metrics in write_metrics.items():
            adv = rank1_adv.get(layer_key)
            if adv is not None:
                all_ws.append(metrics["write_reuse_score"])
                all_adv.append(adv)

    pooled_rho: float | None = None
    pooled_p: float | None = None
    if len(all_ws) >= 3:
        _pooled_result = spearmanr(all_ws, all_adv)
        pooled_rho = float(_pooled_result.statistic)  # type: ignore[union-attr]
        pooled_p = float(_pooled_result.pvalue)  # type: ignore[union-attr]
        print("\n--- POOLED CROSS-ARCHITECTURE ---")
        print(
            f"  write_reuse vs rank1_adv (all layers): "
            f"rho={pooled_rho:+.3f}, p={pooled_p:.4f}, N={len(all_ws)}"
        )

    combined = {
        "per_architecture": results,
        "pooled": {
            "write_reuse_scores": all_ws,
            "rank1_advantages": all_adv,
            "spearman_rho": float(pooled_rho) if pooled_rho is not None else None,
            "p_value": float(pooled_p) if pooled_p is not None else None,
            "n": len(all_ws),
        },
    }

    for root in [mamba2_data_root, gdn_data_root]:
        out_path = root / "write_geometry_combined.json"
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2, default=str)

    return combined

def _parse_layer_list(s: str | None, default: list[int]) -> list[int]:
    if s is None:
        return default
    return [int(x) for x in s.split(",") if x.strip()]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        help="One of: extract-mamba2, extract-gdn, analyze-mamba2, analyze-gdn, summary, all")
    parser.add_argument("--mamba2-data-root", type=Path, default=Path("./data/mamba2"),
                        help="Root for Mamba-2 write vectors + results (default: ./data/mamba2)")
    parser.add_argument("--gdn-data-root", type=Path, default=Path("./data/gdn"),
                        help="Root for GDN write vectors + results (default: ./data/gdn)")
    # Generic CLI aliases for the same paths.
    parser.add_argument("--states-dir", type=Path, default=None,
                        help="(Unused: see --mamba2-data-root / --gdn-data-root) kept for CLI consistency")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="(Unused: outputs go under --*-data-root) kept for CLI consistency")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Mamba-2 SAE checkpoint root (default: <mamba2-data-root>/mamba2_checkpoints)")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"Mamba-2 HF model name (default: {MODEL_NAME})")
    parser.add_argument("--gdn-model", default=GDN_MODEL_NAME,
                        help=f"GDN HF model name (default: {GDN_MODEL_NAME})")
    parser.add_argument("--model-cache-dir", type=Path, default=None,
                        help="HF_HOME cache dir")
    parser.add_argument("--layer", type=int, default=None,
                        help="(Unused: use --mamba2-layers / --gdn-layers)")
    parser.add_argument("--head", type=int, default=HEAD_IDX,
                        help=f"Head idx for single-head write capture (default: {HEAD_IDX})")
    parser.add_argument("--seed", type=int, default=0,
                        help="(Unused for extraction; kept for CLI consistency)")
    parser.add_argument("--n-sequences", type=int, default=None,
                        help="(Use --mamba2-n-seqs / --gdn-n-seqs) kept for CLI consistency")
    parser.add_argument("--mamba2-layers", type=str, default=None,
                        help=f"Comma-separated Mamba-2 layer indices (default: {MAMBA2_LAYERS})")
    parser.add_argument("--gdn-layers", type=str, default=None,
                        help=f"Comma-separated GDN layer indices (default: {GDN_LAYERS})")
    parser.add_argument("--mamba2-n-seqs", type=int, default=MAMBA2_DEFAULT_N_SEQS,
                        help=f"Mamba-2 sequences (default: {MAMBA2_DEFAULT_N_SEQS})")
    parser.add_argument("--gdn-n-seqs", type=int, default=GDN_DEFAULT_N_SEQS,
                        help=f"GDN sequences (default: {GDN_DEFAULT_N_SEQS})")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help=f"Sequence length (default: {DEFAULT_SEQ_LEN})")
    parser.add_argument("--mamba2-batch-size", type=int, default=MAMBA2_BATCH_SIZE,
                        help=f"Mamba-2 forward batch size (default: {MAMBA2_BATCH_SIZE})")
    parser.add_argument("--gdn-batch-size", type=int, default=GDN_BATCH_SIZE,
                        help=f"GDN forward batch size (default: {GDN_BATCH_SIZE})")
    parser.add_argument("--spectral-mamba2", type=Path, default=_SPECTRAL_MAMBA2_DEFAULT,
                        help="Path to spectral_audit_mamba2.json")
    parser.add_argument("--spectral-18layer", type=Path, default=_SPECTRAL_18LAYER_DEFAULT,
                        help="Path to spectral_18layer_correlation.json")
    args = parser.parse_args()

    mamba2_layers = _parse_layer_list(args.mamba2_layers, MAMBA2_LAYERS)
    gdn_layers = _parse_layer_list(args.gdn_layers, GDN_LAYERS)
    mamba2_root = args.mamba2_data_root.resolve()
    gdn_root = args.gdn_data_root.resolve()
    mamba2_root.mkdir(parents=True, exist_ok=True)
    gdn_root.mkdir(parents=True, exist_ok=True)

    stage = args.stage

    if stage in ("extract", "extract-mamba2", "all"):
        print("=== Stage 1: Extract Mamba-2 write vectors ===")
        meta = extract_mamba2_writes(
            data_root=mamba2_root,
            model_cache_dir=args.model_cache_dir,
            layers=mamba2_layers,
            n_sequences=args.mamba2_n_seqs,
            seq_len=args.seq_len,
            batch_size=args.mamba2_batch_size,
            model_name=args.model,
        )
        print(
            f"Mamba-2 extraction done: {meta['n_write_vectors']} vectors from "
            f"{meta['n_sequences']} sequences"
        )

    if stage in ("extract-gdn", "all"):
        print("=== Stage 2: Extract GDN write vectors ===")
        meta = extract_gdn_writes(
            data_root=gdn_root,
            model_cache_dir=args.model_cache_dir,
            layers=gdn_layers,
            n_sequences=args.gdn_n_seqs,
            seq_len=args.seq_len,
            batch_size=args.gdn_batch_size,
            model_name=args.gdn_model,
        )
        print(
            f"GDN extraction done: {meta['n_write_vectors']} vectors from "
            f"{meta['n_sequences']} sequences"
        )

    if stage in ("analyze", "analyze-mamba2", "all"):
        print("=== Stage 3: Analyze Mamba-2 write geometry ===")
        output = analyze_mamba2(
            data_root=mamba2_root,
            checkpoint_dir=args.checkpoint_dir,
            spectral_mamba2_path=args.spectral_mamba2,
        )
        corr = output.get("correlations", {}).get("write_reuse_vs_rank1_adv", {})
        print(f"Mamba-2: rho={corr.get('spearman_rho', 'n/a')}, p={corr.get('p_value', 'n/a')}")

    if stage in ("analyze-gdn", "all"):
        print("=== Stage 4: Analyze GDN write geometry ===")
        output = analyze_gdn(
            data_root=gdn_root,
            spectral_18layer_path=args.spectral_18layer,
        )
        corr = output.get("correlations", {}).get("write_reuse_vs_rank1_adv", {})
        print(f"GDN: rho={corr.get('spearman_rho', 'n/a')}, p={corr.get('p_value', 'n/a')}")

    if stage in ("summary", "all"):
        print("=== Combined Summary ===")
        combined_summary(mamba2_data_root=mamba2_root, gdn_data_root=gdn_root)

if __name__ == "__main__":
    main()
