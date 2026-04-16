#!/usr/bin/env python3
"""GDN multi-head SAE sweep.

Stage 1 (scan): score every (layer, head) by write_vec_top1_energy_fraction.
Stage 2 (extract): full-state extraction for the selected pairs.
Stage 3 (train): train flat + rank1 SAEs across pairs x types x seeds.
Stage 4 (analyze): correlate write_vec_top1 with rank-1 advantage.

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

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

# Allow flat-layout imports used by core/train.py etc.
for _p in (
    _REPO_ROOT,
    _REPO_ROOT / "core",
    _REPO_ROOT / "experiments" / "analysis",
    _REPO_ROOT / "experiments" / "extraction",
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from core.train import train as train_sae_fn  # noqa: E402

def _code_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_SCRIPT_DIR),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "unknown"

CODE_SHA = _code_sha()

GDN_MODEL_NAME = "Qwen/Qwen3.5-0.8B"

# 7 GDN layers spanning sv1/sv2 from 1.9 to 16.4.
# From the 18-layer spectral predictor study:
#   L0  sv1/sv2=16.36   L2  sv1/sv2=7.91   L5  sv1/sv2=4.45
#   L9  sv1/sv2=6.81    L13 sv1/sv2=9.24   L17 sv1/sv2=2.96
#   L21 sv1/sv2=2.54
LAYERS = [0, 2, 5, 9, 13, 17, 21]
N_HEADS = 16  # Qwen3.5-0.8B has 16 GDN value heads

# Scan stage: quick pass to compute write metrics per head.
SCAN_N_SEQUENCES = 512
SCAN_SEQ_LEN = 1024
SCAN_BATCH_SIZE = 2
SCAN_WRITE_SAMPLE_SIZE = 512

# Pair selection.
N_SWEEP_PAIRS = 20
MAX_PAIRS_PER_LAYER = 4

# Extract stage: full state extraction for SAE training.
EXTRACT_N_SEQUENCES = 5000
EXTRACT_SEQ_LEN = 1024
EXTRACT_BATCH_SIZE = 2
EXTRACT_WRITE_SAMPLE_SIZE = 2048

# Train stage.
SAE_N_FEATURES = 2048
SAE_K = 32
SEEDS = [0, 1, 42]
SAE_TYPES = ["flat", "rank1"]
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 256
TRAIN_LR = 3e-4
TRAIN_LR_MIN = 3e-5
TRAIN_WARMUP_STEPS = 50
TRAIN_RESAMPLE_EVERY = 250

# Bundled reference data (under <script_dir>/../analysis or /results/data).
_SPECTRAL_18LAYER_DEFAULT = _REPO_ROOT / "experiments" / "mamba2" / "results" / "data" / "spectral_18layer_correlation.json"

class _Paths:
    """Bundle of paths computed from --data-root / --checkpoint-dir."""

    def __init__(self, data_root: Path, checkpoint_dir: Path | None = None):
        self.data_root = data_root
        self.head_scan = data_root / "gdn_head_scan_metrics.json"
        self.selected_pairs = data_root / "gdn_head_sweep_pairs.json"
        self.state_dir = data_root / "gdn_states_multihead"
        self.write_sample_dir = data_root / "gdn_write_samples_multihead"
        self.checkpoint_dir = checkpoint_dir or (data_root / "gdn_checkpoints_multihead")
        self.results = data_root / "gdn_sae_results_multihead.json"

# Utility functions

def _pair_key(layer: int, head: int) -> str:
    return f"L{layer}_H{head}"

def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num / den) if abs(den) > 1e-12 else float(default)

def _get_qwen_layers(model) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return list(model.language_model.layers)
    raise AttributeError("Could not find Qwen layers on model")

def _l2norm(x, dim: int = -1, eps: float = 1e-6):
    import torch
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

def _apply_padding_mask(hidden_states, attention_mask):
    if (
        attention_mask is not None
        and attention_mask.ndim == 2
        and attention_mask.shape[0] > 1
        and attention_mask.shape[1] > 1
    ):
        return (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    return hidden_states

def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)

def _effective_rank(eigenvalues: np.ndarray, eps: float = 1e-12) -> float:
    vals = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    total = float(vals.sum())
    if total <= eps:
        return 0.0
    p = vals / total
    entropy = -(p * np.log(p + eps)).sum()
    return float(np.exp(entropy))

def _write_metric_summary(
    k_vectors: np.ndarray,
    v_vectors: np.ndarray,
    sample_size: int = 2048,
) -> dict[str, float]:
    """Compute write_vec_top1_energy_fraction and related metrics for one head."""
    if k_vectors.shape[0] == 0 or v_vectors.shape[0] == 0:
        return {}

    valid = (np.linalg.norm(k_vectors, axis=1) > 1e-8) & (
        np.linalg.norm(v_vectors, axis=1) > 1e-8
    )
    k_valid = k_vectors[valid].astype(np.float64, copy=False)
    v_valid = v_vectors[valid].astype(np.float64, copy=False)
    if k_valid.shape[0] < 2:
        return {}

    rng = np.random.default_rng(0)
    sample_n = min(sample_size, k_valid.shape[0])
    if sample_n < k_valid.shape[0]:
        sample_idx = np.sort(rng.choice(k_valid.shape[0], size=sample_n, replace=False))
    else:
        sample_idx = np.arange(sample_n)

    k_sample = _normalize_rows(k_valid[sample_idx])
    v_sample = _normalize_rows(v_valid[sample_idx])

    # Gram matrix of rank-1 write vectors: <x_i, x_j> = <k_i, k_j> * <v_i, v_j>.
    gram_k = k_sample @ k_sample.T
    gram_v = v_sample @ v_sample.T
    gram = (gram_k * gram_v) / sample_n
    eigs = np.linalg.eigvalsh((gram + gram.T) * 0.5)
    eigs = np.clip(eigs, 0.0, None)
    eig_sum = float(eigs.sum())
    write_vec_erank = _effective_rank(eigs)
    write_vec_top1 = float(eigs[-1] / eig_sum) if eig_sum > 1e-12 else 0.0

    # Individual key/value effective ranks.
    n = k_sample.shape[0]
    c_k = (k_sample.T @ k_sample) / n
    c_v = (v_sample.T @ v_sample) / n
    eigs_k = np.linalg.eigvalsh(c_k)
    eigs_v = np.linalg.eigvalsh(c_v)
    erank_k = _effective_rank(eigs_k)
    erank_v = _effective_rank(eigs_v)
    write_reuse = 1.0 / np.sqrt(max(erank_k, 1e-12) * max(erank_v, 1e-12))

    return {
        "write_vec_top1_energy_fraction": write_vec_top1,
        "write_vec_erank": write_vec_erank,
        "write_reuse_score": float(write_reuse),
        "erank_k": erank_k,
        "erank_v": erank_v,
        "n_valid_writes": int(k_valid.shape[0]),
        "sample_size": int(sample_n),
    }

def _stream_openwebtext_batches(tokenizer, n_sequences: int, seq_len: int, batch_size: int):
    """Tokenize OpenWebText into fixed-length batches."""
    import torch
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

def _load_selected_pairs_payload(paths: _Paths) -> dict[str, Any]:
    if not paths.selected_pairs.exists():
        raise FileNotFoundError(
            f"Missing selected pairs file: {paths.selected_pairs}. Run stage=scan first."
        )
    with open(paths.selected_pairs) as f:
        return json.load(f)

def _load_selected_pairs(paths: _Paths) -> list[dict[str, Any]]:
    payload = _load_selected_pairs_payload(paths)
    return payload.get("selected_pairs", [])

# GDN multi-head write capture

class GDNMultiHeadWriteCapture:
    """Capture exact write factors for multiple GDN heads in a single forward pass.

    Replays the GDN recurrence to extract (k_t, delta_t) per head per timestep,
    where delta_t = beta_t * (v_t - S_{t-1} @ k_t) is the value-side write factor.
    """

    def __init__(self, model, layer_idx: int, heads: list[int]):
        self.layer_idx = layer_idx
        self.heads = heads
        self.gdn = _get_qwen_layers(model)[layer_idx].linear_attn
        self.hidden_states = None
        self.attention_mask = None
        self.handle = self.gdn.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError(f"Could not capture hidden_states for GDN layer {self.layer_idx}")
        self.hidden_states = hidden_states.detach()
        self.attention_mask = kwargs.get("attention_mask")

    def pop_write_factors(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Return {head_idx: (k_writes, v_writes)} where each is (batch*seq_len, dim)."""
        import torch
        import torch.nn.functional as F

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

        head_indices = torch.tensor(self.heads, device=key.device, dtype=torch.long)
        n_h = len(self.heads)
        key_sel = key.index_select(2, head_indices)
        value_sel = value.index_select(2, head_indices)
        beta_sel = beta.index_select(2, head_indices)
        g_sel = g.index_select(2, head_indices)

        recurrent = torch.zeros(
            batch_size, n_h, gdn.head_k_dim, gdn.head_v_dim,
            dtype=torch.float32, device=key.device,
        )
        k_writes = torch.empty(
            batch_size, seq_len, n_h, gdn.head_k_dim,
            dtype=torch.float32, device=key.device,
        )
        v_writes = torch.empty(
            batch_size, seq_len, n_h, gdn.head_v_dim,
            dtype=torch.float32, device=key.device,
        )

        for t in range(seq_len):
            recurrent = recurrent * g_sel[:, t].exp().unsqueeze(-1).unsqueeze(-1)
            kv_mem = (recurrent * key_sel[:, t].unsqueeze(-1)).sum(dim=-2)
            delta = (value_sel[:, t] - kv_mem) * beta_sel[:, t].unsqueeze(-1)
            recurrent = recurrent + key_sel[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
            k_writes[:, t] = key_sel[:, t]
            v_writes[:, t] = delta

        results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        k_all = k_writes.detach().cpu().numpy()
        v_all = v_writes.detach().cpu().numpy()
        for local_idx, head_idx in enumerate(self.heads):
            k_np = k_all[:, :, local_idx, :].reshape(-1, gdn.head_k_dim)
            v_np = v_all[:, :, local_idx, :].reshape(-1, gdn.head_v_dim)
            results[head_idx] = (k_np, v_np)

        self.hidden_states = None
        self.attention_mask = None
        return results

    def remove(self) -> None:
        self.handle.remove()

# Pair selection

def _select_pairs_from_metrics(
    metrics_by_pair: dict[str, dict[str, float]],
    *,
    predictor_name: str,
    n_pairs: int = N_SWEEP_PAIRS,
    max_per_layer: int = MAX_PAIRS_PER_LAYER,
) -> list[dict[str, Any]]:
    """Select pairs spanning the predictor range with a per-layer cap."""
    ranked = []
    for pair_key, metrics in metrics_by_pair.items():
        predictor = metrics.get(predictor_name)
        if predictor is None or not np.isfinite(predictor):
            continue
        ranked.append((float(predictor), pair_key, metrics))
    if len(ranked) < n_pairs:
        raise ValueError(
            f"Only {len(ranked)} valid pair metrics for {predictor_name}, need {n_pairs}"
        )

    ranked.sort(key=lambda item: item[0])
    selected: list[tuple[float, str, dict[str, float]]] = []
    used_keys: set[str] = set()
    layer_counts: dict[int, int] = {}

    def parse_pair(pair_key: str) -> tuple[int, int]:
        layer_str, head_str = pair_key.split("_")
        return int(layer_str[1:]), int(head_str[1:])

    # Spread selection evenly across the ranked list.
    for slot in range(n_pairs):
        if n_pairs == 1:
            target_idx = len(ranked) // 2
        else:
            target_idx = int(round(slot * (len(ranked) - 1) / (n_pairs - 1)))

        best_candidate = None
        best_distance = None
        for idx, item in enumerate(ranked):
            _, pair_key, _ = item
            if pair_key in used_keys:
                continue
            layer, _ = parse_pair(pair_key)
            if layer_counts.get(layer, 0) >= max_per_layer:
                continue
            distance = abs(idx - target_idx)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_candidate = item

        if best_candidate is None:
            break

        value, pair_key, metrics = best_candidate
        layer, head = parse_pair(pair_key)
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        used_keys.add(pair_key)
        selected.append((value, pair_key, metrics))

    # Fill remaining slots if the spread pass left gaps.
    if len(selected) < n_pairs:
        for value, pair_key, metrics in ranked:
            if pair_key in used_keys:
                continue
            layer, head = parse_pair(pair_key)
            if layer_counts.get(layer, 0) >= max_per_layer:
                continue
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
            used_keys.add(pair_key)
            selected.append((value, pair_key, metrics))
            if len(selected) >= n_pairs:
                break

    selected_pairs = []
    for rank_idx, (value, pair_key, metrics) in enumerate(selected, start=1):
        layer, head = parse_pair(pair_key)
        payload = dict(metrics)
        payload["layer"] = layer
        payload["head"] = head
        payload["rank_order"] = rank_idx
        selected_pairs.append(payload)

    return selected_pairs

def _append_head_samples(
    sample_store: dict[tuple[int, int], dict[str, np.ndarray]],
    counts: dict[tuple[int, int], int],
    layer: int,
    heads: list[int],
    head_results: dict[int, tuple[np.ndarray, np.ndarray]],
    batch_idx: int,
    n_batches: int,
    rng: np.random.Generator,
) -> None:
    """Spread write-factor samples across the full corpus instead of front-loading batch 1."""
    for head in heads:
        pair = (layer, head)
        if pair not in sample_store:
            continue
        limit = sample_store[pair]["k"].shape[0]
        count = counts[pair]
        if count >= limit:
            continue
        k_flat, v_flat = head_results[head]
        batches_left = max(n_batches - batch_idx, 1)
        remaining = limit - count
        take = min(int(np.ceil(remaining / batches_left)), remaining, k_flat.shape[0])
        if take <= 0:
            continue
        if take < k_flat.shape[0]:
            sample_idx = np.sort(rng.choice(k_flat.shape[0], size=take, replace=False))
            k_take = k_flat[sample_idx]
            v_take = v_flat[sample_idx]
        else:
            k_take = k_flat
            v_take = v_flat
        sample_store[pair]["k"][count : count + take] = k_take
        sample_store[pair]["v"][count : count + take] = v_take
        counts[pair] = count + take

# Stage 1: SCAN

def scan_heads(
    *,
    paths: _Paths,
    model_cache_dir: Path | None,
    model_name: str = GDN_MODEL_NAME,
    layers: list[int] = LAYERS,
    n_sequences: int = SCAN_N_SEQUENCES,
    seq_len: int = SCAN_SEQ_LEN,
    batch_size: int = SCAN_BATCH_SIZE,
    sample_size_per_head: int = SCAN_WRITE_SAMPLE_SIZE,
    n_pairs: int = N_SWEEP_PAIRS,
    max_per_layer: int = MAX_PAIRS_PER_LAYER,
    spectral_18layer_path: Path | None = None,
) -> dict:
    """Score all heads on target layers via write_vec_top1_energy_fraction."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    if model_cache_dir is not None:
        os.environ["HF_HOME"] = str(model_cache_dir)
    device = "cuda"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    # Validate layers are GDN layers.
    model_layers = _get_qwen_layers(model)
    gdn_layer_indices = [
        idx for idx, layer_mod in enumerate(model_layers)
        if hasattr(layer_mod, "linear_attn")
    ]
    bad_layers = [layer for layer in layers if layer not in gdn_layer_indices]
    if bad_layers:
        raise ValueError(
            f"Layers {bad_layers} are not GDN layers. "
            f"GDN layers: {gdn_layer_indices}"
        )

    # Get head dimensions from first target layer.
    sample_gdn = model_layers[layers[0]].linear_attn
    head_k_dim = int(sample_gdn.head_k_dim)
    head_v_dim = int(sample_gdn.head_v_dim)
    n_heads = int(sample_gdn.num_v_heads)
    print(
        f"GDN: {n_heads} heads per layer, k_dim={head_k_dim}, v_dim={head_v_dim}, "
        f"scanning {len(layers)} layers x {n_heads} heads = {len(layers) * n_heads} pairs"
    )

    batches, actual_sequences, n_tokens = _stream_openwebtext_batches(
        tokenizer, n_sequences, seq_len, batch_size,
    )
    print(f"Tokenized {n_tokens} tokens, processing {actual_sequences} sequences")

    all_heads = list(range(n_heads))
    sample_store: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    counts: dict[tuple[int, int], int] = {}
    for layer in layers:
        for head in all_heads:
            pair = (layer, head)
            sample_store[pair] = {
                "k": np.empty((sample_size_per_head, head_k_dim), dtype=np.float32),
                "v": np.empty((sample_size_per_head, head_v_dim), dtype=np.float32),
            }
            counts[pair] = 0

    captures = {
        layer: GDNMultiHeadWriteCapture(model, layer, all_heads)
        for layer in layers
    }

    t0 = time.time()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc="GDN head scan")):
            input_ids = batch.to(device)
            _ = model(input_ids=input_ids, use_cache=False)

            for layer in layers:
                head_results = captures[layer].pop_write_factors()
                _append_head_samples(
                    sample_store,
                    counts,
                    layer,
                    all_heads,
                    head_results,
                    batch_idx,
                    len(batches),
                    rng,
                )

            torch.cuda.empty_cache()

            if (batch_idx + 1) % 50 == 0:
                filled = sum(1 for c in counts.values() if c >= sample_size_per_head)
                print(
                    f"  Batch {batch_idx + 1}/{len(batches)}, "
                    f"{filled}/{len(counts)} pairs filled"
                )

    for capture in captures.values():
        capture.remove()
    elapsed = time.time() - t0
    print(f"Scan extraction: {elapsed:.0f}s")

    spectral_ref: dict[str, Any] = {}
    if spectral_18layer_path is not None and spectral_18layer_path.exists():
        with open(spectral_18layer_path) as f:
            spectral_ref = json.load(f)
    spectral_layers = spectral_ref.get("layers", {})

    metrics_by_pair: dict[str, dict[str, float]] = {}
    for layer in layers:
        for head in all_heads:
            pair = (layer, head)
            count = counts[pair]
            metrics = _write_metric_summary(
                sample_store[pair]["k"][:count],
                sample_store[pair]["v"][:count],
                sample_size=sample_size_per_head,
            )
            metrics["layer"] = float(layer)
            metrics["head"] = float(head)

            layer_spec = spectral_layers.get(str(layer), {})
            if "sv1_sv2" in layer_spec:
                metrics["spectral_sv1_sv2"] = float(layer_spec["sv1_sv2"])

            metrics_by_pair[_pair_key(layer, head)] = metrics

    selected_pairs = _select_pairs_from_metrics(
        metrics_by_pair,
        predictor_name="write_vec_top1_energy_fraction",
        n_pairs=n_pairs,
        max_per_layer=max_per_layer,
    )

    scan_payload = {
        "model": model_name,
        "architecture": "gdn",
        "layers": layers,
        "n_heads": n_heads,
        "n_sequences": actual_sequences,
        "seq_len": seq_len,
        "sample_size_per_head": sample_size_per_head,
        "head_k_dim": head_k_dim,
        "head_v_dim": head_v_dim,
        "predictor": "write_vec_top1_energy_fraction",
        "metrics_by_pair": metrics_by_pair,
        "selected_pairs": selected_pairs,
        "scan_time_s": elapsed,
        "code_sha": CODE_SHA,
    }
    paths.head_scan.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.head_scan, "w") as f:
        json.dump(scan_payload, f, indent=2)

    selected_payload = {
        "model": model_name,
        "architecture": "gdn",
        "predictor": "write_vec_top1_energy_fraction",
        "selection_source": str(paths.head_scan),
        "selected_pairs": selected_pairs,
        "n_sequences": actual_sequences,
        "seq_len": seq_len,
        "sample_size_per_head": sample_size_per_head,
        "code_sha": CODE_SHA,
    }
    with open(paths.selected_pairs, "w") as f:
        json.dump(selected_payload, f, indent=2)

    print(f"\nSelected {len(selected_pairs)} (layer, head) pairs:")
    for pair in selected_pairs:
        sv_str = f"  sv1/sv2={pair.get('spectral_sv1_sv2', '?')}" if "spectral_sv1_sv2" in pair else ""
        print(
            f"  L{pair['layer']:>2} H{pair['head']:>2}: "
            f"write_vec_top1={pair['write_vec_top1_energy_fraction']:.4f}, "
            f"write_reuse={pair['write_reuse_score']:.4f}"
            f"{sv_str}"
        )
    return selected_payload

# Stage 2: EXTRACT

def extract_states(
    *,
    paths: _Paths,
    model_cache_dir: Path | None,
    model_name: str = GDN_MODEL_NAME,
    n_sequences: int = EXTRACT_N_SEQUENCES,
    seq_len: int = EXTRACT_SEQ_LEN,
    batch_size: int = EXTRACT_BATCH_SIZE,
    write_sample_size: int = EXTRACT_WRITE_SAMPLE_SIZE,
) -> dict:
    """Extract GDN recurrent states for selected (layer, head) pairs.

    Saves per-head states as layer_{L}/head_{H}.npy with shape (n_samples, d_k, d_v).
    Also saves higher-fidelity write samples for the analyze stage.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    selected_pairs = _load_selected_pairs(paths)
    if not selected_pairs:
        raise ValueError("No selected pairs found. Run stage=scan first.")

    pair_list = [(int(p["layer"]), int(p["head"])) for p in selected_pairs]
    heads_by_layer: dict[int, list[int]] = {}
    for layer, head in pair_list:
        heads_by_layer.setdefault(layer, []).append(head)
    for layer in heads_by_layer:
        heads_by_layer[layer] = sorted(set(heads_by_layer[layer]))

    if model_cache_dir is not None:
        os.environ["HF_HOME"] = str(model_cache_dir)
    device = "cuda"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    sample_gdn = _get_qwen_layers(model)[pair_list[0][0]].linear_attn
    head_k_dim = int(sample_gdn.head_k_dim)
    head_v_dim = int(sample_gdn.head_v_dim)
    print(
        f"Extracting states for {len(pair_list)} pairs, "
        f"state shape=({head_k_dim}, {head_v_dim}), "
        f"{n_sequences} sequences x {seq_len} tokens"
    )

    batches, actual_sequences, n_tokens = _stream_openwebtext_batches(
        tokenizer, n_sequences, seq_len, batch_size,
    )
    print(f"Tokenized {n_tokens} tokens, processing {actual_sequences} sequences")

    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.write_sample_dir.mkdir(parents=True, exist_ok=True)

    memmaps: dict[tuple[int, int], np.memmap] = {}
    for layer, head in pair_list:
        layer_dir = paths.state_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = layer_dir / f"head_{head}_tmp.dat"
        memmaps[(layer, head)] = np.memmap(
            str(tmp_path),
            dtype=np.float32,
            mode="w+",
            shape=(actual_sequences, head_k_dim, head_v_dim),
        )

    sample_store: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    counts: dict[tuple[int, int], int] = {}
    for layer, head in pair_list:
        pair = (layer, head)
        sample_store[pair] = {
            "k": np.empty((write_sample_size, head_k_dim), dtype=np.float32),
            "v": np.empty((write_sample_size, head_v_dim), dtype=np.float32),
        }
        counts[pair] = 0

    captures = {
        layer: GDNMultiHeadWriteCapture(model, layer, heads)
        for layer, heads in heads_by_layer.items()
    }

    sample_offset = 0
    t0 = time.time()
    rng = np.random.default_rng(1)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc="Extract states")):
            input_ids = batch.to(device)
            bs = input_ids.shape[0]

            outputs = model(input_ids=input_ids, use_cache=True)
            cache = outputs.past_key_values

            for layer, heads in heads_by_layer.items():
                state = cache.layers[layer].recurrent_states.float().cpu().numpy()
                for head in heads:
                    memmaps[(layer, head)][sample_offset : sample_offset + bs] = state[:, head]

            for layer, heads in heads_by_layer.items():
                head_results = captures[layer].pop_write_factors()
                _append_head_samples(
                    sample_store,
                    counts,
                    layer,
                    heads,
                    head_results,
                    batch_idx,
                    len(batches),
                    rng,
                )

            sample_offset += bs
            del outputs, cache
            torch.cuda.empty_cache()

            if (batch_idx + 1) % 100 == 0:
                elapsed_so_far = time.time() - t0
                rate = sample_offset / max(elapsed_so_far, 1e-6)
                eta = (actual_sequences - sample_offset) / max(rate, 1e-6)
                print(
                    f"  Batch {batch_idx + 1}/{len(batches)}, "
                    f"{sample_offset}/{actual_sequences} samples, "
                    f"{elapsed_so_far:.0f}s elapsed, ETA {eta:.0f}s"
                )

    for capture in captures.values():
        capture.remove()

    elapsed = time.time() - t0

    for (layer, head), mm in memmaps.items():
        mm.flush()
        arr = np.array(mm, dtype=np.float32)
        layer_dir = paths.state_dir / f"layer_{layer}"
        np.save(str(layer_dir / f"head_{head}.npy"), arr)
        tmp_path = layer_dir / f"head_{head}_tmp.dat"
        del mm, arr
        if tmp_path.exists():
            tmp_path.unlink()

    write_metrics_by_pair: dict[str, dict[str, float]] = {}
    for layer, head in pair_list:
        pair = (layer, head)
        layer_dir = paths.write_sample_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        count = counts[pair]
        k_arr = np.array(sample_store[pair]["k"][:count], dtype=np.float32)
        v_arr = np.array(sample_store[pair]["v"][:count], dtype=np.float32)
        np.save(str(layer_dir / f"head_{head}_k.npy"), k_arr)
        np.save(str(layer_dir / f"head_{head}_v.npy"), v_arr)
        write_metrics_by_pair[_pair_key(layer, head)] = _write_metric_summary(k_arr, v_arr)

    metadata = {
        "model": model_name,
        "architecture": "gdn",
        "pairs": [{"layer": layer, "head": head} for layer, head in pair_list],
        "n_samples": actual_sequences,
        "seq_len": seq_len,
        "extract_batch_size": batch_size,
        "head_k_dim": head_k_dim,
        "head_v_dim": head_v_dim,
        "dtype": "float32",
        "model_dtype": "bfloat16",
        "write_sample_size": write_sample_size,
        "write_metrics_by_pair": write_metrics_by_pair,
        "extraction_time_s": elapsed,
        "code_sha": CODE_SHA,
    }
    with open(paths.state_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Extracted {actual_sequences} states for {len(pair_list)} pairs in {elapsed:.0f}s "
        f"({elapsed / max(actual_sequences, 1):.1f}s/seq)"
    )
    return metadata

# Stage 3: TRAIN

def train_sae(
    *,
    paths: _Paths,
    layer: int,
    head: int,
    sae_type: str,
    seed: int,
    n_features: int = SAE_N_FEATURES,
    k: int = SAE_K,
) -> dict:
    """Train one SAE (flat or rank1) on one (layer, head) pair."""
    metadata_path = paths.state_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing extraction metadata: {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)

    pair_set = {(int(p["layer"]), int(p["head"])) for p in metadata.get("pairs", [])}
    if (layer, head) not in pair_set:
        raise ValueError(f"Pair (L{layer}, H{head}) not in extracted data")

    ckpt_dir = paths.checkpoint_dir / f"layer_{layer}" / f"head_{head}" / f"{sae_type}_s{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    result = train_sae_fn(
        sae_type=sae_type,
        data_dir=str(paths.state_dir),
        layer=layer,
        head=head,
        n_features=n_features,
        k=k,
        seed=seed,
        output_dir=str(ckpt_dir),
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        lr=TRAIN_LR,
        lr_min=TRAIN_LR_MIN,
        warmup_steps=TRAIN_WARMUP_STEPS,
        resample_every=TRAIN_RESAMPLE_EVERY,
        log_every=25,
    )

    print(
        f"L{layer} H{head} {sae_type} s{seed}: "
        f"val_mse={result.get('best_mse', result.get('best_val_mse', '?'))}"
    )
    return {"layer": layer, "head": head, "sae_type": sae_type, "seed": seed, **result}

# Stage 4: ANALYZE

def _compute_advantage_pct(
    results: dict[str, dict[str, object]],
    better: str,
    baseline: str,
) -> float | None:
    if better not in results or baseline not in results:
        return None
    better_mse = float(results[better]["mean"])
    baseline_mse = float(results[baseline]["mean"])
    return _safe_div(baseline_mse - better_mse, baseline_mse) * 100.0

def analyze_results(*, paths: _Paths) -> dict:
    """Correlate write_vec_top1 with rank1_advantage across all pairs."""
    import torch
    from scipy.stats import spearmanr

    selected_pairs = _load_selected_pairs(paths)
    if not selected_pairs:
        raise ValueError("No selected pairs found. Run stage=scan first.")

    state_meta_path = paths.state_dir / "metadata.json"
    if not state_meta_path.exists():
        raise FileNotFoundError(f"Missing extraction metadata: {state_meta_path}")
    with open(state_meta_path) as f:
        state_meta = json.load(f)

    results_by_pair: list[dict[str, Any]] = []
    paired_rows: list[dict[str, float | int]] = []

    for pair in selected_pairs:
        layer = int(pair["layer"])
        head = int(pair["head"])
        pair_key = _pair_key(layer, head)

        k_path = paths.write_sample_dir / f"layer_{layer}" / f"head_{head}_k.npy"
        v_path = paths.write_sample_dir / f"layer_{layer}" / f"head_{head}_v.npy"
        if not k_path.exists() or not v_path.exists():
            print(f"  {pair_key}: missing write samples, skipping")
            continue
        write_metrics = _write_metric_summary(np.load(k_path), np.load(v_path))

        ckpt_base = paths.checkpoint_dir / f"layer_{layer}" / f"head_{head}"
        mse_by_type: dict[str, dict[str, object]] = {}
        missing_seeds: dict[str, list[int]] = {}
        for sae_type in SAE_TYPES:
            mses = []
            present = []
            absent = []
            for seed in SEEDS:
                ckpt_path = ckpt_base / f"{sae_type}_s{seed}" / "best.pt"
                if ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    val_mse = ckpt.get("val_mse", ckpt.get("best_val_mse"))
                    if val_mse is not None:
                        mses.append(float(val_mse))
                        present.append(seed)
                else:
                    absent.append(seed)
            if mses:
                mse_by_type[sae_type] = {
                    "mean": float(np.mean(mses)),
                    "std": float(np.std(mses)),
                    "seed_mses": dict(zip(present, mses, strict=False)),
                }
            if absent:
                missing_seeds[sae_type] = absent

        rank1_adv = _compute_advantage_pct(mse_by_type, better="rank1", baseline="flat")
        entry = {
            "layer": layer,
            "head": head,
            "pair_key": pair_key,
            "selection_predictor": float(pair.get("write_vec_top1_energy_fraction", np.nan)),
            "write_metrics": write_metrics,
            "mse_by_type": mse_by_type,
            "missing_seeds": missing_seeds,
            "rank1_advantage_pct": rank1_adv,
        }
        results_by_pair.append(entry)

        if rank1_adv is not None and write_metrics:
            predictor = write_metrics.get("write_vec_top1_energy_fraction")
            reuse = write_metrics.get("write_reuse_score")
            erank = write_metrics.get("write_vec_erank")
            if (
                predictor is not None and np.isfinite(predictor)
                and reuse is not None and np.isfinite(reuse)
                and erank is not None and np.isfinite(erank)
            ):
                paired_rows.append({
                    "layer": layer,
                    "head": head,
                    "predictor": float(predictor),
                    "selection_predictor": float(
                        pair.get("write_vec_top1_energy_fraction", np.nan)
                    ),
                    "write_reuse": float(reuse),
                    "write_vec_erank": float(erank),
                    "advantage": float(rank1_adv),
                })

    def corr(xs: list[float], ys: list[float]) -> dict[str, float]:
        if len(xs) < 3 or len(xs) != len(ys):
            return {"spearman_rho": 0.0, "p_value": 1.0, "n": len(xs)}
        rho, p = spearmanr(xs, ys)
        return {"spearman_rho": float(rho), "p_value": float(p), "n": len(xs)}

    def corr_rows(
        rows: list[dict[str, float | int]], x_key: str, y_key: str,
    ) -> dict[str, float]:
        valid = [
            (float(r[x_key]), float(r[y_key]))
            for r in rows
            if np.isfinite(float(r[x_key])) and np.isfinite(float(r[y_key]))
        ]
        if len(valid) < 3:
            return {"spearman_rho": 0.0, "p_value": 1.0, "n": len(valid)}
        xs, ys = zip(*valid)
        return corr(list(xs), list(ys))

    def layer_centered_corr(
        rows: list[dict[str, float | int]], x_key: str, y_key: str,
    ) -> dict[str, float]:
        grouped: dict[int, list[dict[str, float | int]]] = {}
        for row in rows:
            grouped.setdefault(int(row["layer"]), []).append(row)

        xs: list[float] = []
        ys: list[float] = []
        for layer_rows in grouped.values():
            if len(layer_rows) < 2:
                continue
            x_mean = float(np.mean([float(r[x_key]) for r in layer_rows]))
            y_mean = float(np.mean([float(r[y_key]) for r in layer_rows]))
            for r in layer_rows:
                xs.append(float(r[x_key]) - x_mean)
                ys.append(float(r[y_key]) - y_mean)
        return corr(xs, ys)

    def leave_one_layer_out(
        rows: list[dict[str, float | int]], x_key: str, y_key: str,
    ) -> dict[str, dict[str, float]]:
        outputs: dict[str, dict[str, float]] = {}
        layers_present = sorted({int(r["layer"]) for r in rows})
        for held_out in layers_present:
            subset = [r for r in rows if int(r["layer"]) != held_out]
            outputs[str(held_out)] = corr_rows(subset, x_key, y_key)
        return outputs

    def per_layer_corr(
        rows: list[dict[str, float | int]], x_key: str, y_key: str,
    ) -> dict[str, dict[str, float]]:
        outputs: dict[str, dict[str, float]] = {}
        layers_present = sorted({int(r["layer"]) for r in rows})
        for layer in layers_present:
            subset = [r for r in rows if int(r["layer"]) == layer]
            outputs[str(layer)] = corr_rows(subset, x_key, y_key)
        return outputs

    paired_predictor = [r["predictor"] for r in paired_rows]
    paired_reuse = [r["write_reuse"] for r in paired_rows]
    paired_erank = [r["write_vec_erank"] for r in paired_rows]
    paired_advantage = [r["advantage"] for r in paired_rows]

    output = {
        "model": GDN_MODEL_NAME,
        "architecture": "gdn",
        "predictor_name": "write_vec_top1_energy_fraction",
        "selected_pairs": selected_pairs,
        "train_config": {
            "n_samples": state_meta["n_samples"],
            "seq_len": state_meta["seq_len"],
            "batch_size": TRAIN_BATCH_SIZE,
            "epochs": TRAIN_EPOCHS,
            "warmup_steps": TRAIN_WARMUP_STEPS,
            "resample_every": TRAIN_RESAMPLE_EVERY,
            "seeds": SEEDS,
            "sae_types": SAE_TYPES,
            "n_features": SAE_N_FEATURES,
            "k": SAE_K,
        },
        "per_pair": results_by_pair,
        "correlations": {
            "write_vec_top1_energy_fraction_vs_rank1_adv": corr(
                paired_predictor, paired_advantage,
            ),
            "write_reuse_score_vs_rank1_adv": corr(paired_reuse, paired_advantage),
            "write_vec_erank_vs_rank1_adv": corr(paired_erank, paired_advantage),
            "selection_predictor_vs_recomputed_predictor": corr_rows(
                paired_rows, "selection_predictor", "predictor",
            ),
            "layer_centered_write_vec_top1_vs_rank1_adv": layer_centered_corr(
                paired_rows, "predictor", "advantage",
            ),
        },
        "stability": {
            "per_layer_write_vec_top1_vs_rank1_adv": per_layer_corr(
                paired_rows, "predictor", "advantage",
            ),
            "leave_one_layer_out_write_vec_top1_vs_rank1_adv": leave_one_layer_out(
                paired_rows, "predictor", "advantage",
            ),
        },
        "code_sha": CODE_SHA,
    }

    paths.results.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.results, "w") as f:
        json.dump(output, f, indent=2, default=str)

    main_corr = output["correlations"]["write_vec_top1_energy_fraction_vs_rank1_adv"]
    print(f"\n{'=' * 76}")
    print("GDN MULTI-HEAD SAE SWEEP RESULTS")
    print(f"{'=' * 76}")
    print(f"{'Pair':>9}  {'write_vec_top1':>14}  {'write_reuse':>12}  {'rank1_adv%':>11}")
    print("-" * 76)
    for entry in sorted(results_by_pair, key=lambda e: (e["layer"], e["head"])):
        wm = entry["write_metrics"]
        adv = entry["rank1_advantage_pct"]
        adv_str = f"{adv:+.1f}" if adv is not None else "n/a"
        print(
            f"L{entry['layer']:>2}H{entry['head']:>2}  "
            f"{wm.get('write_vec_top1_energy_fraction', float('nan')):>14.4f}  "
            f"{wm.get('write_reuse_score', float('nan')):>12.4f}  "
            f"{adv_str:>11}"
        )

    print("\nPooled correlations:")
    for name, stats in output["correlations"].items():
        print(
            f"  {name}: rho={stats['spearman_rho']:+.3f}, "
            f"p={stats['p_value']:.4f}, N={stats['n']}"
        )

    print("\nPer-layer correlations:")
    for layer_str, stats in output["stability"]["per_layer_write_vec_top1_vs_rank1_adv"].items():
        print(
            f"  layer {layer_str}: rho={stats['spearman_rho']:+.3f}, "
            f"p={stats['p_value']:.4f}, N={stats['n']}"
        )

    print("\nLeave-one-layer-out correlations:")
    for layer_str, stats in output["stability"][
        "leave_one_layer_out_write_vec_top1_vs_rank1_adv"
    ].items():
        print(
            f"  without layer {layer_str}: rho={stats['spearman_rho']:+.3f}, "
            f"p={stats['p_value']:.4f}, N={stats['n']}"
        )

    if main_corr["spearman_rho"] > 0.4 and main_corr["p_value"] < 0.05:
        print("\nRESULT: write_vec_top1 predicts rank1_advantage across GDN (layer, head) pairs.")
    else:
        print("\nRESULT: write_vec_top1 predictor transfer not established for GDN heads.")

    return output

def _parse_int_list(s: str | None, default: list[int]) -> list[int]:
    if s is None:
        return default
    return [int(x) for x in s.split(",") if x.strip()]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        help="One of: scan, extract, train, analyze, all")
    parser.add_argument("--data-root", type=Path, default=Path("./data/gdn_sweep"),
                        help="Root directory for sweep artifacts (default: ./data/gdn_sweep)")
    parser.add_argument("--states-dir", type=Path, default=None,
                        help="(Unused: derived from --data-root) kept for CLI consistency")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="(Unused: outputs go under --data-root)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Override SAE checkpoint root (default: <data-root>/gdn_checkpoints_multihead)")
    parser.add_argument("--model", default=GDN_MODEL_NAME,
                        help=f"HF model name (default: {GDN_MODEL_NAME})")
    parser.add_argument("--model-cache-dir", type=Path, default=None,
                        help="HF_HOME cache directory")
    parser.add_argument("--layer", type=int, default=None,
                        help="(Unused for full sweep; see --layers)")
    parser.add_argument("--head", type=int, default=None,
                        help="(Unused: sweep chooses pairs automatically)")
    parser.add_argument("--seed", type=int, default=None,
                        help="(Unused: seeds set by SEEDS constant)")
    parser.add_argument("--layers", type=str, default=None,
                        help=f"Comma-separated layer indices (default: {LAYERS})")
    parser.add_argument("--n-sequences", type=int, default=None,
                        help="(Use --scan-n-sequences / --extract-n-sequences)")
    parser.add_argument("--scan-n-sequences", type=int, default=SCAN_N_SEQUENCES,
                        help=f"Scan stage sequences (default: {SCAN_N_SEQUENCES})")
    parser.add_argument("--extract-n-sequences", type=int, default=EXTRACT_N_SEQUENCES,
                        help=f"Extract stage sequences (default: {EXTRACT_N_SEQUENCES})")
    parser.add_argument("--n-pairs", type=int, default=N_SWEEP_PAIRS,
                        help=f"Number of (layer, head) pairs to select (default: {N_SWEEP_PAIRS})")
    parser.add_argument("--max-per-layer", type=int, default=MAX_PAIRS_PER_LAYER,
                        help=f"Max selected pairs per layer (default: {MAX_PAIRS_PER_LAYER})")
    parser.add_argument("--n-features", type=int, default=SAE_N_FEATURES,
                        help=f"SAE dictionary size (default: {SAE_N_FEATURES})")
    parser.add_argument("--k", type=int, default=SAE_K,
                        help=f"TopK sparsity (default: {SAE_K})")
    parser.add_argument("--spectral-18layer", type=Path, default=_SPECTRAL_18LAYER_DEFAULT,
                        help="Path to spectral_18layer_correlation.json (optional)")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    paths = _Paths(data_root, args.checkpoint_dir.resolve() if args.checkpoint_dir else None)

    layers = _parse_int_list(args.layers, LAYERS)
    stage = args.stage

    if stage in ("scan", "all"):
        print("=== Stage 1: Scan all heads across GDN layers ===")
        payload = scan_heads(
            paths=paths,
            model_cache_dir=args.model_cache_dir,
            model_name=args.model,
            layers=layers,
            n_sequences=args.scan_n_sequences,
            n_pairs=args.n_pairs,
            max_per_layer=args.max_per_layer,
            spectral_18layer_path=args.spectral_18layer,
        )
        print(f"Selected {len(payload['selected_pairs'])} (layer, head) pairs")

    if stage in ("extract", "all"):
        print("=== Stage 2: Extract recurrent states for selected pairs ===")
        meta = extract_states(
            paths=paths,
            model_cache_dir=args.model_cache_dir,
            model_name=args.model,
            n_sequences=args.extract_n_sequences,
        )
        print(f"Extraction complete: {meta['n_samples']} sequences, {len(meta['pairs'])} pairs")

    if stage in ("train", "all"):
        print("=== Stage 3: Train flat + rank1 SAEs ===")
        selected_pairs = _load_selected_pairs(paths)
        jobs = []
        for pair in selected_pairs:
            layer = int(pair["layer"])
            head = int(pair["head"])
            for sae_type in SAE_TYPES:
                for seed in SEEDS:
                    jobs.append({
                        "layer": layer,
                        "head": head,
                        "sae_type": sae_type,
                        "seed": seed,
                    })
        n_jobs = len(jobs)
        print(
            f"Queued {n_jobs} training jobs "
            f"({len(selected_pairs)} pairs x {len(SAE_TYPES)} types x {len(SEEDS)} seeds)"
        )
        results = []
        failures = []
        for idx, job in enumerate(jobs, start=1):
            try:
                r = train_sae(
                    paths=paths,
                    layer=job["layer"],
                    head=job["head"],
                    sae_type=job["sae_type"],
                    seed=job["seed"],
                    n_features=args.n_features,
                    k=args.k,
                )
                results.append(r)
            except Exception as exc:
                failures.append({
                    "layer": job["layer"],
                    "head": job["head"],
                    "sae_type": job["sae_type"],
                    "seed": job["seed"],
                    "error": str(exc),
                })
                print(
                    f"  Job {idx}/{n_jobs} failed: "
                    f"L{job['layer']} H{job['head']} {job['sae_type']} s{job['seed']}: {exc}"
                )
        n_ok = sum(1 for r in results if r.get("best_mse") is not None or r.get("best_val_mse") is not None)
        print(f"Training complete: {n_ok}/{n_jobs} checkpoints saved")
        if failures:
            failure_path = paths.checkpoint_dir / "training_failures.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            with open(failure_path, "w") as f:
                json.dump(failures, f, indent=2)
            print(f"Recorded {len(failures)} failed jobs to {failure_path}; continuing to analysis")

    if stage in ("analyze", "all"):
        print("=== Stage 4: Analyze write geometry vs SAE advantage ===")
        output = analyze_results(paths=paths)

        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        local_path = os.path.join(local_out, "gdn_multihead_sweep_results.json")
        with open(local_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Saved local copy to {local_path}")

        corr = output["correlations"]["write_vec_top1_energy_fraction_vs_rank1_adv"]
        print(
            f"\nGDN multi-head sweep: rho={corr['spearman_rho']:+.3f} "
            f"(p={corr['p_value']:.4f}), N={corr['n']}"
        )

if __name__ == "__main__":
    main()
