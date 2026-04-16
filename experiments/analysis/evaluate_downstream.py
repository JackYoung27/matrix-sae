#!/usr/bin/env python3

import time
from contextlib import contextmanager
from typing import TypedDict

import numpy as np
import torch
from tqdm import tqdm  # type: ignore[import-untyped]

from core.sae import (
    BilinearEncoderFlatSAE,
    BilinearMatrixSAE,
    FlatSAE,
    MatrixSAE,
    load_sae_checkpoint,
)
from core.types import SAEConfig as _CoreSAEConfig

TrainableSAE = FlatSAE | MatrixSAE | BilinearMatrixSAE | BilinearEncoderFlatSAE

class SAEConfig(TypedDict):
    tag: str
    sae: TrainableSAE
    sae_type: str
    train_mse: float | None

def load_sae_from_checkpoint(
    ckpt_path: str,
    config_path: str | None = None,
    device: str = "cuda",
) -> tuple[TrainableSAE, _CoreSAEConfig, float | None]:
    """Load a trained SAE from checkpoint, returning (sae, cfg, train_mse)."""
    sae, cfg, ckpt = load_sae_checkpoint(
        ckpt_path, config_path=config_path, device=device, weights_only=True,
    )
    assert isinstance(sae, (FlatSAE, MatrixSAE, BilinearMatrixSAE, BilinearEncoderFlatSAE)), (
        f"load_sae_checkpoint returned unexpected type {type(sae).__name__}"
    )
    val_mse = ckpt.get("val_mse")
    return sae, cfg, float(val_mse) if val_mse is not None else None

def reconstruct_state_head(sae: TrainableSAE, state_head: torch.Tensor, sae_type: str) -> torch.Tensor:
    """Reconstruct a single head's state (d_k, d_v) through the SAE.

    Args:
        sae: trained SAE model
        state_head: (d_k, d_v) tensor, the recurrent state for one head
        sae_type: "flat", "rank1", "bilinear", or "bilinear_tied"

    Returns:
        Reconstructed (d_k, d_v) tensor
    """
    d_k, d_v = state_head.shape
    if sae_type == "flat":
        x = state_head.reshape(1, d_k * d_v)
        out = sae(x)
        return out.reconstruction.reshape(d_k, d_v)
    else:
        x = state_head.unsqueeze(0)
        out = sae(x)
        return out.reconstruction.squeeze(0)

@contextmanager
def _patch_gdn_initial_states(model, gdn_layer_indices: list[int], states: dict[int, torch.Tensor]):
    """Temporarily patch GDN layers' chunk_gated_delta_rule to inject initial states.

    The Qwen3.5 GDN layer calls chunk_gated_delta_rule with initial_state=None
    when seq_len > 1. This patches all specified GDN layers so the suffix pass
    correctly continues from the prefix's accumulated recurrent states.

    Args:
        model: HuggingFace CausalLM with model.model.layers
        gdn_layer_indices: list of layer indices to patch
        states: {layer_idx: (batch, n_heads, d_k, d_v)} tensors to inject
    """
    model_layers = model.model.layers if hasattr(model, "model") else model.layers
    originals = {}

    for idx in gdn_layer_indices:
        if idx not in states:
            continue
        gdn = model_layers[idx].linear_attn
        originals[idx] = gdn.chunk_gated_delta_rule
        state = states[idx]

        def _make_patched(fn, s):
            def patched(*args, **kwargs):
                kwargs["initial_state"] = s
                return fn(*args, **kwargs)
            return patched

        gdn.chunk_gated_delta_rule = _make_patched(originals[idx], state)

    try:
        yield
    finally:
        for idx, fn in originals.items():
            model_layers[idx].linear_attn.chunk_gated_delta_rule = fn

@torch.no_grad()
def compute_split_perplexity(
    model,
    input_ids: torch.Tensor,
    split_pos: int,
    gdn_layer_indices: list[int],
    target_layer_idx: int | None = None,
    sae: TrainableSAE | None = None,
    sae_type: str | None = None,
    head_idx: int = 0,
) -> dict[str, float | int]:
    """Compute perplexity on the suffix of a sequence, optionally with SAE state reconstruction.

    Protocol:
      1. Run prefix (tokens[:split_pos]) with use_cache=True to build all caches
      2. Extract recurrent states for ALL GDN layers from the cache
      3. If sae is provided, reconstruct target_layer's head_idx state through the SAE
      4. Patch ALL GDN layers' chunk_gated_delta_rule to inject their prefix states
      5. Run suffix with the patched model + attention KV cache
      6. Compute cross-entropy loss on suffix tokens

    All GDN layers are patched so the suffix pass correctly continues from each
    layer's accumulated state. Only the target layer gets SAE reconstruction.

    Args:
        model: HuggingFace CausalLM model
        input_ids: (batch, seq_len) token ids
        split_pos: position to split prefix/suffix
        gdn_layer_indices: all GDN layer indices in the model
        target_layer_idx: GDN layer to apply SAE on (None = no SAE, pure baseline)
        sae: trained SAE model (None = no modification)
        sae_type: SAE type string for dispatch
        head_idx: which head to reconstruct (default 0)

    Returns:
        dict with keys: loss, perplexity, n_tokens
    """
    batch_size, seq_len = input_ids.shape
    assert split_pos > 0 and split_pos < seq_len

    prefix = input_ids[:, :split_pos]
    suffix = input_ids[:, split_pos:]

    prefix_out = model(input_ids=prefix, use_cache=True)
    cache = prefix_out.past_key_values

    gdn_states = {}
    for idx in gdn_layer_indices:
        layer_cache = cache.layers[idx]
        if hasattr(layer_cache, "recurrent_states") and layer_cache.recurrent_states is not None:
            gdn_states[idx] = layer_cache.recurrent_states.clone()

    if sae is not None and target_layer_idx is not None and sae_type is not None:
        if target_layer_idx in gdn_states:
            state = gdn_states[target_layer_idx]
            for b in range(batch_size):
                original_head = state[b, head_idx].float()
                reconstructed = reconstruct_state_head(sae, original_head, sae_type)
                state[b, head_idx] = reconstructed.to(state.dtype)

    with _patch_gdn_initial_states(model, gdn_layer_indices, gdn_states):
        suffix_out = model(
            input_ids=suffix,
            past_key_values=cache,
            use_cache=False,
            labels=suffix,
        )

    loss = suffix_out.loss
    n_tokens = suffix.shape[1] - 1

    return {
        "loss": loss.item(),
        "perplexity": float(torch.exp(loss).item()),
        "n_tokens": n_tokens,
    }

def _get_gdn_layer_indices(model) -> list[int]:
    """Get indices of GDN (linear_attention) layers from model config."""
    config = getattr(model.config, "text_config", model.config)
    return [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]

@torch.no_grad()
def evaluate_downstream(
    model,
    tokenizer,
    corpus_batches: list[torch.Tensor],
    layer_idx: int,
    sae_configs: list[SAEConfig],
    head_idx: int = 0,
    split_fraction: float = 0.5,
    device: str = "cuda",
) -> dict[str, object]:
    """Run the full downstream evaluation for one layer across multiple SAEs.

    Args:
        model: HuggingFace CausalLM
        tokenizer: tokenizer (unused, kept for API consistency)
        corpus_batches: list of (batch_size, seq_len) token tensors
        layer_idx: GDN layer index to evaluate
        sae_configs: list of dicts with keys: tag, sae, sae_type, train_mse
        head_idx: which head SAEs are trained on
        split_fraction: fraction of sequence used as prefix (default 0.5)
        device: compute device

    Returns:
        dict with baseline and per-SAE results
    """
    seq_len = corpus_batches[0].shape[1]
    split_pos = int(seq_len * split_fraction)
    n_sequences = sum(b.shape[0] for b in corpus_batches)
    gdn_layers = _get_gdn_layer_indices(model)

    print(f"Downstream evaluation: layer {layer_idx}, head {head_idx}")
    print(f"  {n_sequences} sequences x {seq_len} tokens, split at position {split_pos}")
    print(f"  {len(gdn_layers)} GDN layers patched for state forwarding")
    print(f"  Evaluating {len(sae_configs)} SAE checkpoints + baseline")

    results: dict[str, object] = {
        "layer": layer_idx,
        "head": head_idx,
        "seq_len": seq_len,
        "split_pos": split_pos,
        "n_sequences": n_sequences,
        "gdn_layers": gdn_layers,
    }

    print("\nComputing baseline perplexity (original state, no SAE)...")
    baseline_losses = []
    baseline_tokens = 0
    t0 = time.time()

    for batch in tqdm(corpus_batches, desc="Baseline"):
        batch = batch.to(device)
        for i in range(batch.shape[0]):
            seq = batch[i:i+1]
            r = compute_split_perplexity(
                model, seq, split_pos,
                gdn_layer_indices=gdn_layers,
            )
            baseline_losses.append(r["loss"] * r["n_tokens"])
            baseline_tokens += r["n_tokens"]
        torch.cuda.empty_cache()

    baseline_avg_loss = sum(baseline_losses) / max(baseline_tokens, 1)
    baseline_ppl = float(np.exp(baseline_avg_loss))
    baseline_time = time.time() - t0

    results["baseline"] = {
        "loss": baseline_avg_loss,
        "perplexity": baseline_ppl,
        "n_tokens": baseline_tokens,
        "time_s": round(baseline_time, 1),
    }
    print(f"  Baseline perplexity: {baseline_ppl:.2f} (loss={baseline_avg_loss:.4f}, {baseline_time:.1f}s)")

    results["sae_results"] = {}

    for sae_cfg in sae_configs:
        tag = sae_cfg["tag"]
        sae = sae_cfg["sae"]
        sae_type = sae_cfg["sae_type"]
        train_mse = sae_cfg.get("train_mse")

        print(f"\nEvaluating: {tag}")
        sae_losses = []
        sae_tokens = 0
        t0 = time.time()

        for batch in tqdm(corpus_batches, desc=f"  {tag}"):
            batch = batch.to(device)
            for i in range(batch.shape[0]):
                seq = batch[i:i+1]
                r = compute_split_perplexity(
                    model, seq, split_pos,
                    gdn_layer_indices=gdn_layers,
                    target_layer_idx=layer_idx, sae=sae,
                    sae_type=sae_type, head_idx=head_idx,
                )
                sae_losses.append(r["loss"] * r["n_tokens"])
                sae_tokens += r["n_tokens"]
            torch.cuda.empty_cache()

        sae_avg_loss = sum(sae_losses) / max(sae_tokens, 1)
        sae_ppl = float(np.exp(sae_avg_loss))
        delta_pct = (sae_ppl - baseline_ppl) / baseline_ppl * 100
        sae_time = time.time() - t0

        results["sae_results"][tag] = {
            "loss": sae_avg_loss,
            "perplexity": sae_ppl,
            "delta_pct": delta_pct,
            "train_mse": train_mse,
            "n_tokens": sae_tokens,
            "time_s": round(sae_time, 1),
        }
        print(f"  Perplexity: {sae_ppl:.2f} (delta={delta_pct:+.2f}%, loss={sae_avg_loss:.4f}, {sae_time:.1f}s)")

    return results

@torch.no_grad()
def compute_split_perplexity_allheads(
    model,
    input_ids: torch.Tensor,
    split_pos: int,
    gdn_layer_indices: list[int],
    target_layer_idx: int | None = None,
    sae: TrainableSAE | None = None,
    sae_type: str | None = None,
    n_heads: int = 32,
) -> dict[str, float | int]:
    """Like compute_split_perplexity but reconstructs ALL heads through one SAE.

    Uses the same head-0-trained SAE for every head. This amplifies the
    structured-vs-flat MSE gap by 32x compared to single-head replacement.

    Args:
        model: HuggingFace CausalLM model
        input_ids: (batch, seq_len) token ids
        split_pos: position to split prefix/suffix
        gdn_layer_indices: all GDN layer indices in the model
        target_layer_idx: GDN layer to apply SAE on (None = no SAE, pure baseline)
        sae: trained SAE model (None = no modification)
        sae_type: SAE type string for dispatch
        n_heads: number of heads to reconstruct

    Returns:
        dict with keys: loss, perplexity, n_tokens
    """
    batch_size, seq_len = input_ids.shape
    assert split_pos > 0 and split_pos < seq_len

    prefix = input_ids[:, :split_pos]
    suffix = input_ids[:, split_pos:]

    prefix_out = model(input_ids=prefix, use_cache=True)
    cache = prefix_out.past_key_values

    gdn_states = {}
    for idx in gdn_layer_indices:
        layer_cache = cache.layers[idx]
        if hasattr(layer_cache, "recurrent_states") and layer_cache.recurrent_states is not None:
            gdn_states[idx] = layer_cache.recurrent_states.clone()

    if sae is not None and target_layer_idx is not None and sae_type is not None:
        if target_layer_idx in gdn_states:
            state = gdn_states[target_layer_idx]
            actual_n_heads = min(n_heads, state.shape[1])
            for b in range(batch_size):
                for h in range(actual_n_heads):
                    original_head = state[b, h].float()
                    reconstructed = reconstruct_state_head(sae, original_head, sae_type)
                    state[b, h] = reconstructed.to(state.dtype)

    with _patch_gdn_initial_states(model, gdn_layer_indices, gdn_states):
        suffix_out = model(
            input_ids=suffix,
            past_key_values=cache,
            use_cache=False,
            labels=suffix,
        )

    loss = suffix_out.loss
    n_tokens = suffix.shape[1] - 1

    return {
        "loss": loss.item(),
        "perplexity": float(torch.exp(loss).item()),
        "n_tokens": n_tokens,
    }

@torch.no_grad()
def evaluate_downstream_allheads(
    model,
    tokenizer,
    corpus_batches: list[torch.Tensor],
    layer_idx: int,
    sae_configs: list[SAEConfig],
    n_heads: int = 32,
    split_fraction: float = 0.5,
    device: str = "cuda",
) -> dict[str, object]:
    """Run downstream evaluation replacing ALL heads through each SAE.

    Same protocol as evaluate_downstream but reconstructs every head's state
    through the (head-0-trained) SAE. This amplifies reconstruction error
    into a larger perplexity delta.

    Args:
        model: HuggingFace CausalLM
        tokenizer: tokenizer (unused, kept for API consistency)
        corpus_batches: list of (batch_size, seq_len) token tensors
        layer_idx: GDN layer index to evaluate
        sae_configs: list of dicts with keys: tag, sae, sae_type, train_mse
        n_heads: number of heads to reconstruct (default 32)
        split_fraction: fraction of sequence used as prefix (default 0.5)
        device: compute device

    Returns:
        dict with baseline and per-SAE results
    """
    seq_len = corpus_batches[0].shape[1]
    split_pos = int(seq_len * split_fraction)
    n_sequences = sum(b.shape[0] for b in corpus_batches)
    gdn_layers = _get_gdn_layer_indices(model)

    print(f"Downstream evaluation (ALL HEADS): layer {layer_idx}, {n_heads} heads")
    print(f"  {n_sequences} sequences x {seq_len} tokens, split at position {split_pos}")
    print(f"  {len(gdn_layers)} GDN layers patched for state forwarding")
    print(f"  Evaluating {len(sae_configs)} SAE checkpoints + baseline")

    results: dict[str, object] = {
        "layer": layer_idx,
        "n_heads_replaced": n_heads,
        "all_heads": True,
        "shared_head0_sae_across_heads": True,
        "approximation_note": (
            "Each checkpoint was trained on head 0 and then applied to every head in the "
            "target layer. Treat this as a stress test, not as per-head matched reconstruction."
        ),
        "seq_len": seq_len,
        "split_pos": split_pos,
        "n_sequences": n_sequences,
        "gdn_layers": gdn_layers,
    }

    print("\nComputing baseline perplexity (original state, no SAE)...")
    baseline_losses = []
    baseline_tokens = 0
    t0 = time.time()

    for batch in tqdm(corpus_batches, desc="Baseline"):
        batch = batch.to(device)
        for i in range(batch.shape[0]):
            seq = batch[i:i+1]
            r = compute_split_perplexity_allheads(
                model, seq, split_pos,
                gdn_layer_indices=gdn_layers,
            )
            baseline_losses.append(r["loss"] * r["n_tokens"])
            baseline_tokens += r["n_tokens"]
        torch.cuda.empty_cache()

    baseline_avg_loss = sum(baseline_losses) / max(baseline_tokens, 1)
    baseline_ppl = float(np.exp(baseline_avg_loss))
    baseline_time = time.time() - t0

    results["baseline"] = {
        "loss": baseline_avg_loss,
        "perplexity": baseline_ppl,
        "n_tokens": baseline_tokens,
        "time_s": round(baseline_time, 1),
    }
    print(f"  Baseline perplexity: {baseline_ppl:.2f} (loss={baseline_avg_loss:.4f}, {baseline_time:.1f}s)")

    results["sae_results"] = {}

    for sae_cfg in sae_configs:
        tag = sae_cfg["tag"]
        sae = sae_cfg["sae"]
        sae_type = sae_cfg["sae_type"]
        train_mse = sae_cfg.get("train_mse")

        print(f"\nEvaluating (all {n_heads} heads): {tag}")
        sae_losses = []
        sae_tokens = 0
        t0 = time.time()

        for batch in tqdm(corpus_batches, desc=f"  {tag}"):
            batch = batch.to(device)
            for i in range(batch.shape[0]):
                seq = batch[i:i+1]
                r = compute_split_perplexity_allheads(
                    model, seq, split_pos,
                    gdn_layer_indices=gdn_layers,
                    target_layer_idx=layer_idx, sae=sae,
                    sae_type=sae_type, n_heads=n_heads,
                )
                sae_losses.append(r["loss"] * r["n_tokens"])
                sae_tokens += r["n_tokens"]
            torch.cuda.empty_cache()

        sae_avg_loss = sum(sae_losses) / max(sae_tokens, 1)
        sae_ppl = float(np.exp(sae_avg_loss))
        delta_pct = (sae_ppl - baseline_ppl) / baseline_ppl * 100
        sae_time = time.time() - t0

        results["sae_results"][tag] = {
            "loss": sae_avg_loss,
            "perplexity": sae_ppl,
            "delta_pct": delta_pct,
            "train_mse": train_mse,
            "n_tokens": sae_tokens,
            "time_s": round(sae_time, 1),
        }
        print(f"  Perplexity: {sae_ppl:.2f} (delta={delta_pct:+.2f}%, loss={sae_avg_loss:.4f}, {sae_time:.1f}s)")

    return results

@torch.no_grad()
def compute_split_perplexity_perhead_matched(
    model,
    input_ids: torch.Tensor,
    split_pos: int,
    gdn_layer_indices: list[int],
    target_layer_idx: int | None = None,
    head_saes: dict[int, tuple] | None = None,
    n_heads: int = 16,
) -> dict:
    """Like compute_split_perplexity_allheads but uses a MATCHED SAE per head.

    Instead of one head-0 SAE applied to all heads, each head gets its own
    SAE trained on that head's data. This gives a clean per-head reconstruction
    without cross-head spectral mismatch.

    Args:
        model: HuggingFace CausalLM model
        input_ids: (batch, seq_len) token ids
        split_pos: position to split prefix/suffix
        gdn_layer_indices: all GDN layer indices in the model
        target_layer_idx: GDN layer to apply SAEs on (None = no SAE, pure baseline)
        head_saes: {head_idx: (sae, sae_type)} dict mapping each head to its SAE
        n_heads: number of heads to reconstruct

    Returns:
        dict with keys: loss, perplexity, n_tokens
    """
    batch_size, seq_len = input_ids.shape
    assert split_pos > 0 and split_pos < seq_len

    prefix = input_ids[:, :split_pos]
    suffix = input_ids[:, split_pos:]

    prefix_out = model(input_ids=prefix, use_cache=True)
    cache = prefix_out.past_key_values

    gdn_states = {}
    for idx in gdn_layer_indices:
        layer_cache = cache.layers[idx]
        if hasattr(layer_cache, "recurrent_states") and layer_cache.recurrent_states is not None:
            gdn_states[idx] = layer_cache.recurrent_states.clone()

    if head_saes is not None and target_layer_idx is not None:
        if target_layer_idx in gdn_states:
            state = gdn_states[target_layer_idx]
            actual_n_heads = min(n_heads, state.shape[1])
            for b in range(batch_size):
                for h in range(actual_n_heads):
                    if h not in head_saes:
                        continue
                    sae, sae_type = head_saes[h]
                    original_head = state[b, h].float()
                    reconstructed = reconstruct_state_head(sae, original_head, sae_type)
                    state[b, h] = reconstructed.to(state.dtype)

    with _patch_gdn_initial_states(model, gdn_layer_indices, gdn_states):
        suffix_out = model(
            input_ids=suffix,
            past_key_values=cache,
            use_cache=False,
            labels=suffix,
        )

    loss = suffix_out.loss
    n_tokens = suffix.shape[1] - 1

    return {
        "loss": loss.item(),
        "perplexity": float(torch.exp(loss).item()),
        "n_tokens": n_tokens,
    }

@torch.no_grad()
def evaluate_downstream_perhead_matched(
    model,
    tokenizer,
    corpus_batches: list[torch.Tensor],
    layer_idx: int,
    sae_type_configs: dict[str, dict[int, tuple]],
    n_heads: int = 16,
    split_fraction: float = 0.5,
    device: str = "cuda",
    baseline_result: dict | None = None,
) -> dict:
    """Run downstream evaluation with per-head matched SAEs.

    Each SAE type gets 16 SAEs (one per head), each trained on that head's data.
    All 16 heads are reconstructed simultaneously through their matched SAEs.

    Args:
        model: HuggingFace CausalLM
        tokenizer: tokenizer (unused, kept for API consistency)
        corpus_batches: list of (batch_size, seq_len) token tensors
        layer_idx: GDN layer index to evaluate
        sae_type_configs: {sae_type_tag: {head_idx: (sae, sae_type_str)}}
            e.g. {"flat": {0: (sae0, "flat"), 1: (sae1, "flat"), ...}}
        n_heads: number of heads (default 16)
        split_fraction: fraction of sequence used as prefix (default 0.5)
        device: compute device

    Returns:
        dict with baseline and per-SAE-type results
    """
    seq_len = corpus_batches[0].shape[1]
    split_pos = int(seq_len * split_fraction)
    n_sequences = sum(b.shape[0] for b in corpus_batches)
    gdn_layers = _get_gdn_layer_indices(model)

    print(f"Downstream evaluation (PER-HEAD MATCHED): layer {layer_idx}, {n_heads} heads")
    print(f"  {n_sequences} sequences x {seq_len} tokens, split at position {split_pos}")
    print(f"  {len(gdn_layers)} GDN layers patched for state forwarding")
    print(f"  Evaluating {len(sae_type_configs)} SAE types + baseline")

    results: dict[str, object] = {
        "layer": layer_idx,
        "n_heads_replaced": n_heads,
        "all_heads": True,
        "per_head_matched": True,
        "seq_len": seq_len,
        "split_pos": split_pos,
        "n_sequences": n_sequences,
        "gdn_layers": gdn_layers,
    }

    if baseline_result is None:
        print("\nComputing baseline perplexity (original state, no SAE)...")
        baseline_losses = []
        baseline_tokens = 0
        t0 = time.time()

        for batch in tqdm(corpus_batches, desc="Baseline"):
            batch = batch.to(device)
            for i in range(batch.shape[0]):
                seq = batch[i:i+1]
                r = compute_split_perplexity_perhead_matched(
                    model, seq, split_pos,
                    gdn_layer_indices=gdn_layers,
                )
                baseline_losses.append(r["loss"] * r["n_tokens"])
                baseline_tokens += r["n_tokens"]
            torch.cuda.empty_cache()

        baseline_avg_loss = sum(baseline_losses) / max(baseline_tokens, 1)
        baseline_ppl = float(np.exp(baseline_avg_loss))
        baseline_time = time.time() - t0

        results["baseline"] = {
            "loss": baseline_avg_loss,
            "perplexity": baseline_ppl,
            "n_tokens": baseline_tokens,
            "time_s": round(baseline_time, 1),
        }
        print(
            f"  Baseline perplexity: {baseline_ppl:.2f} "
            f"(loss={baseline_avg_loss:.4f}, {baseline_time:.1f}s)"
        )
    else:
        baseline_avg_loss = float(baseline_result["loss"])
        baseline_ppl = float(baseline_result["perplexity"])
        results["baseline"] = dict(baseline_result)
        print(
            f"\nReusing baseline perplexity: {baseline_ppl:.2f} "
            f"(loss={baseline_avg_loss:.4f})"
        )

    results["sae_results"] = {}

    for tag, head_saes in sae_type_configs.items():
        n_loaded = len(head_saes)
        print(f"\nEvaluating (per-head matched, {n_loaded}/{n_heads} heads): {tag}")
        sae_losses = []
        sae_tokens = 0
        t0 = time.time()

        for batch in tqdm(corpus_batches, desc=f"  {tag}"):
            batch = batch.to(device)
            for i in range(batch.shape[0]):
                seq = batch[i:i+1]
                r = compute_split_perplexity_perhead_matched(
                    model, seq, split_pos,
                    gdn_layer_indices=gdn_layers,
                    target_layer_idx=layer_idx,
                    head_saes=head_saes,
                    n_heads=n_heads,
                )
                sae_losses.append(r["loss"] * r["n_tokens"])
                sae_tokens += r["n_tokens"]
            torch.cuda.empty_cache()

        sae_avg_loss = sum(sae_losses) / max(sae_tokens, 1)
        sae_ppl = float(np.exp(sae_avg_loss))
        delta_pct = (sae_ppl - baseline_ppl) / baseline_ppl * 100
        sae_time = time.time() - t0

        results["sae_results"][tag] = {
            "loss": sae_avg_loss,
            "perplexity": sae_ppl,
            "delta_pct": delta_pct,
            "train_mse": None,
            "n_heads_loaded": n_loaded,
            "n_tokens": sae_tokens,
            "time_s": round(sae_time, 1),
        }
        print(f"  Perplexity: {sae_ppl:.2f} (delta={delta_pct:+.2f}%, loss={sae_avg_loss:.4f}, {sae_time:.1f}s)")

    return results

def format_results_table(results: dict) -> str:
    """Format evaluation results as a readable table."""
    lines = []
    layer = results["layer"]
    n_seq = results["n_sequences"]
    seq_len = results["seq_len"]
    split_pos = results["split_pos"]

    if results.get("all_heads"):
        n_heads = results.get("n_heads_replaced", "?")
        if results.get("per_head_matched"):
            lines.append(f"Downstream Evaluation (PER-HEAD MATCHED, {n_heads} HEADS): layer {layer}, {n_seq} sequences x {seq_len} tokens (split at {split_pos})")
        else:
            lines.append(f"Downstream Evaluation (ALL {n_heads} HEADS): layer {layer}, {n_seq} sequences x {seq_len} tokens (split at {split_pos})")
        if results.get("approximation_note"):
            lines.append(f"Note: {results['approximation_note']}")
    else:
        lines.append(f"Downstream Evaluation: layer {layer}, {n_seq} sequences x {seq_len} tokens (split at {split_pos})")
    lines.append(f"Baseline perplexity: {results['baseline']['perplexity']:.2f}")
    lines.append("")
    lines.append(f"{'SAE':<45} | {'Perplexity':>10} | {'Delta (%)':>10} | {'MSE (train)':>12}")
    lines.append("-" * 85)

    for tag, r in sorted(results["sae_results"].items()):
        mse_str = f"{r['train_mse']:.2e}" if r["train_mse"] is not None else "N/A"
        lines.append(
            f"{tag:<45} | {r['perplexity']:>10.2f} | {r['delta_pct']:>+9.2f}% | {mse_str:>12}"
        )

    return "\n".join(lines)
