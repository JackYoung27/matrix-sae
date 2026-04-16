#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.sae import BilinearMatrixSAE, load_sae_checkpoint

# Model + tokenizer

def load_model(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

# SAE loader

def load_sae(ckpt_path: str, device: str) -> BilinearMatrixSAE:
    sae, _, _ = load_sae_checkpoint(ckpt_path, device=device)
    assert isinstance(sae, BilinearMatrixSAE), f"Expected BilinearMatrixSAE, got {type(sae)}"
    return sae

# Corpus loading (streaming openwebtext)

def load_corpus_batches(tokenizer, n_seqs: int, seq_len: int, batch_size: int):
    from datasets import load_dataset

    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    token_ids = []
    target = n_seqs * seq_len * 2
    for example in ds:
        token_ids.extend(tokenizer.encode(example["text"], add_special_tokens=False))
        if len(token_ids) >= target:
            break

    n_actual = min(n_seqs, len(token_ids) // seq_len)
    batches = []
    for start in range(0, n_actual, batch_size):
        end = min(start + batch_size, n_actual)
        seqs = [token_ids[i * seq_len : (i + 1) * seq_len] for i in range(start, end)]
        batches.append(torch.tensor(seqs, dtype=torch.long))
    return batches

# GDN hook: capture k, v, beta, g from layer forward

class GDNWriteCapture:
    """Register a forward hook on a GDN layer to capture k, v, beta, g."""

    def __init__(self, model, layer_idx: int):
        self.layer_idx = layer_idx
        self.k = None  # (batch, seq_len, n_heads, head_k_dim)
        self.v = None
        self.beta = None  # (batch, seq_len, n_heads)
        self.g = None  # (batch, seq_len, n_heads)
        self._original_forward = None
        self._model_layer = None

        layers = model.model.layers if hasattr(model, "model") else model.layers
        self._model_layer = layers[layer_idx]
        gdn = self._model_layer.linear_attn
        self._gdn = gdn
        self._original_forward = gdn.forward
        gdn.forward = self._hooked_forward

    def _hooked_forward(self, hidden_states, cache_params=None, attention_mask=None):
        # Reproduce the GDN forward, capturing k, v, beta, g
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

        gdn = self._gdn
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = gdn.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = gdn.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, gdn.head_v_dim)

        b = gdn.in_proj_b(hidden_states)
        a = gdn.in_proj_a(hidden_states)

        # Conv1d path (non-cached)
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (gdn.conv_kernel_size - mixed_qkv.shape[-1], 0))
            conv_state = cache_params.update_conv_state(conv_state, gdn.layer_idx)

        if gdn.causal_conv1d_fn is not None:
            mixed_qkv = gdn.causal_conv1d_fn(
                x=mixed_qkv, weight=gdn.conv1d.weight.squeeze(1),
                bias=gdn.conv1d.bias, activation=gdn.activation, seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(gdn.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, gdn.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, gdn.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, gdn.head_v_dim)

        beta = b.sigmoid()
        g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)

        if gdn.num_v_heads // gdn.num_k_heads > 1:
            query = query.repeat_interleave(gdn.num_v_heads // gdn.num_k_heads, dim=2)
            key = key.repeat_interleave(gdn.num_v_heads // gdn.num_k_heads, dim=2)

        # Store captures (detach to avoid graph retention)
        self.k = key.detach()   # (bs, seq_len, n_heads, head_k_dim)
        self.v = value.detach() # (bs, seq_len, n_heads, head_v_dim)
        self.beta = beta.detach()  # (bs, seq_len, n_heads)
        self.g = g.detach()        # (bs, seq_len, n_heads)

        # Run original chunk kernel
        core_attn_out, last_recurrent_state = gdn.chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=None, output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, gdn.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, gdn.head_v_dim)
        z = z.reshape(-1, gdn.head_v_dim)
        core_attn_out = gdn.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return gdn.out_proj(core_attn_out)

    def remove(self):
        self._gdn.forward = self._original_forward

# Simulate recurrence to get per-position states for one head

@torch.no_grad()
def simulate_recurrence(
    k: torch.Tensor,       # (seq_len, head_k_dim)
    v: torch.Tensor,       # (seq_len, head_v_dim)
    beta: torch.Tensor,    # (seq_len,) or (seq_len, 1)
    g: torch.Tensor,       # (seq_len,)
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate the GDN delta rule recurrence.

    The actual update (from fla naive reference):
        S_t = exp(g_t) * S_{t-1} + outer(k_t, beta_t * (v_t - S_{t-1} @ k_t))

    where k is L2-normalized (use_qk_l2norm_in_kernel=True).

    Returns:
        states: (seq_len, d_k, d_v) per-position recurrent states
        k_normed: (seq_len, d_k) L2-normalized key vectors (as used in the recurrence)
        write_v: (seq_len, d_v) the corrected value vectors actually written
    """
    seq_len, d_k = k.shape
    d_v = v.shape[1]

    k = k.to(dtype)
    v = v.to(dtype)
    beta_flat = beta.reshape(seq_len).to(dtype)
    g_flat = g.reshape(seq_len).to(dtype)

    # L2-normalize k (matching use_qk_l2norm_in_kernel=True)
    k_normed = F.normalize(k, dim=-1)

    states = torch.zeros(seq_len, d_k, d_v, dtype=dtype, device=k.device)
    write_v = torch.zeros(seq_len, d_v, dtype=dtype, device=k.device)
    S = torch.zeros(d_k, d_v, dtype=dtype, device=k.device)

    for t in range(seq_len):
        # Decay
        S = g_flat[t].exp() * S
        # Delta rule correction: v_corrected = v_t - S @ k_t
        v_corrected = v[t] - S @ k_normed[t]
        # Scale by beta
        v_corrected = beta_flat[t] * v_corrected
        # Write
        S = S + torch.outer(k_normed[t], v_corrected)
        states[t] = S
        write_v[t] = v_corrected

    return states, k_normed, write_v

# Main alignment computation

@torch.no_grad()
def compute_alignment(
    model,
    tokenizer,
    sae: BilinearMatrixSAE,
    layer_idx: int,
    head_idx: int,
    n_seqs: int = 100,
    seq_len: int = 512,
    batch_size: int = 4,
    top_n: int = 50,
    device: str = "cuda",
) -> dict:
    """Compute alignment between SAE decoder atoms and GDN k*v^T writes."""

    print(f"Loading {n_seqs} sequences of length {seq_len}...")
    batches = load_corpus_batches(tokenizer, n_seqs, seq_len, batch_size)
    total_seqs = sum(b.shape[0] for b in batches)
    print(f"Loaded {total_seqs} sequences in {len(batches)} batches")

    # SAE decoder atoms: V_dec (n_features, rank, d_k), W_dec (n_features, rank, d_v)
    V_dec = sae.V_dec.detach().float().cpu()  # (n_features, rank, d_k)
    W_dec = sae.W_dec.detach().float().cpu()  # (n_features, rank, d_v)
    n_features = V_dec.shape[0]
    rank = V_dec.shape[1]
    d_k = V_dec.shape[2]
    d_v = W_dec.shape[2]
    print(f"SAE: {n_features} features, rank={rank}, d_k={d_k}, d_v={d_v}")

    # For rank-1, V_dec[:,0,:] and W_dec[:,0,:] are the decoder directions
    # Normalize them for cosine computation (on CPU)
    v_dec_dir = F.normalize(V_dec[:, 0, :], dim=-1)  # (n_features, d_k)
    w_dec_dir = F.normalize(W_dec[:, 0, :], dim=-1)  # (n_features, d_v)

    # Install hook
    hook = GDNWriteCapture(model, layer_idx)

    # Accumulators: for each feature, store (activation, k_cos, v_cos) across all positions
    # We'll collect all data first, then pick top-N per feature
    # To avoid OOM, we process batch by batch and keep only top-N per feature
    feature_data = {i: [] for i in range(n_features)}
    # feature_data[i] = list of (activation, k_cosine, v_cosine)

    t_start = time.time()

    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        input_ids = batch.to(device)
        bs = input_ids.shape[0]

        # Forward pass (captures k, v, beta, g via hook)
        # use_cache=True to also get final recurrent state for verification
        outputs = model(input_ids=input_ids, use_cache=True)

        # hook.k: (bs, seq_len, n_heads, head_k_dim)
        # hook.v: (bs, seq_len, n_heads, head_v_dim)
        # hook.beta: (bs, seq_len, n_heads) or (bs, seq_len, n_heads, 1)
        # hook.g: (bs, seq_len, n_heads)

        k_all = hook.k  # (bs, seq_len, n_heads, d_k)
        v_all = hook.v  # (bs, seq_len, n_heads, d_v)
        beta_all = hook.beta  # (bs, seq_len, n_heads[, 1])
        g_all = hook.g    # (bs, seq_len, n_heads)

        for seq_idx in range(bs):
            # Extract head-specific k, v, beta, g and move to CPU for simulation
            k_head = k_all[seq_idx, :, head_idx, :].cpu()  # (seq_len, d_k)
            v_head = v_all[seq_idx, :, head_idx, :].cpu()  # (seq_len, d_v)

            # beta shape handling
            if beta_all.ndim == 4:
                beta_head = beta_all[seq_idx, :, head_idx, 0].cpu()  # (seq_len,)
            else:
                beta_head = beta_all[seq_idx, :, head_idx].cpu()  # (seq_len,)

            g_head = g_all[seq_idx, :, head_idx].cpu()  # (seq_len,)

            # Simulate recurrence to get per-position states and actual write vectors
            states, k_normed, write_v = simulate_recurrence(
                k_head, v_head, beta_head, g_head,
                dtype=torch.float32,
            )
            # states: (seq_len, d_k, d_v)
            # k_normed: (seq_len, d_k) -- L2-normalized keys used in recurrence
            # write_v: (seq_len, d_v) -- corrected value vectors actually written

            # Verify simulated final state matches actual cache (first seq only)
            if batch_idx == 0 and seq_idx == 0:
                cache = outputs.past_key_values
                actual_state = cache.layers[layer_idx].recurrent_states[0, head_idx].float().cpu()
                sim_final = states[-1].cpu()
                cos_sim = F.cosine_similarity(
                    actual_state.reshape(1, -1), sim_final.reshape(1, -1)
                ).item()
                rel_err = (actual_state - sim_final).norm() / actual_state.norm()
                print(f"Recurrence verification: cos_sim={cos_sim:.6f}, rel_error={rel_err:.6f}")
                if cos_sim < 0.9:
                    print("WARNING: simulated recurrence does not match actual state well")

            # Encode states through SAE in sub-batches to get activations
            sub_bs = 128
            all_coeffs = []
            for s in range(0, states.shape[0], sub_bs):
                e = min(s + sub_bs, states.shape[0])
                chunk = states[s:e].to(device)
                coeffs = sae.encode(chunk)  # (chunk_size, n_features)
                all_coeffs.append(coeffs.cpu())
            all_coeffs = torch.cat(all_coeffs, dim=0)  # (seq_len, n_features)

            # Normalize write vectors for cosine computation
            # k is already L2-normed; normalize write_v (use eps to avoid NaN for zero writes)
            k_dir = k_normed.cpu()  # (seq_len, d_k) already unit norm
            wv_raw = write_v.cpu()
            wv_norms = wv_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            wv_dir = wv_raw / wv_norms  # (seq_len, d_v)

            # Vectorized: compute cosines for ALL features at ALL positions at once
            # v_dec_dir: (n_features, d_k), k_dir: (seq_len, d_k)
            # k_cos_all: (seq_len, n_features)
            k_cos_all = k_dir @ v_dec_dir.t()  # (seq_len, n_features)
            v_cos_all = wv_dir @ w_dec_dir.t()  # (seq_len, n_features)

            # Collect active (position, feature) pairs efficiently
            # all_coeffs: (seq_len, n_features), sparse (k=32 active per position)
            active_pos, active_feat = (all_coeffs > 0).nonzero(as_tuple=True)
            if active_pos.numel() > 0:
                active_vals = all_coeffs[active_pos, active_feat].numpy()
                active_k_cos = k_cos_all[active_pos, active_feat].numpy()
                active_v_cos = v_cos_all[active_pos, active_feat].numpy()
                feat_np = active_feat.numpy()

                # Group by feature for efficient insertion
                for fi in np.unique(feat_np):
                    mask = feat_np == fi
                    entries = list(zip(
                        active_vals[mask].tolist(),
                        active_k_cos[mask].tolist(),
                        active_v_cos[mask].tolist(),
                    ))
                    feature_data[int(fi)].extend(entries)

        # Clear GPU memory
        del k_all, v_all, beta_all, g_all, outputs
        if device == "cuda":
            torch.cuda.empty_cache()

    hook.remove()
    elapsed = time.time() - t_start
    print(f"Data collection took {elapsed:.1f}s")

    # For each feature, pick top-N activations and compute mean alignment
    print("Computing per-feature alignment scores...")
    results = []
    alive_count = 0
    for feat_idx in range(n_features):
        data = feature_data[feat_idx]
        if len(data) == 0:
            results.append({
                "feature": feat_idx, "alive": False,
                "mean_k_cos": 0.0, "mean_v_cos": 0.0,
                "n_activations": 0,
            })
            continue

        alive_count += 1
        # Sort by activation value, take top-N
        data.sort(key=lambda x: x[0], reverse=True)
        top_data = data[:top_n]

        k_cosines = [d[1] for d in top_data]
        v_cosines = [d[2] for d in top_data]
        mean_k = float(np.mean(k_cosines))
        mean_v = float(np.mean(v_cosines))

        results.append({
            "feature": feat_idx, "alive": True,
            "mean_k_cos": mean_k, "mean_v_cos": mean_v,
            "mean_abs_k_cos": float(np.mean(np.abs(k_cosines))),
            "mean_abs_v_cos": float(np.mean(np.abs(v_cosines))),
            "n_activations": len(data),
            "top_n_used": len(top_data),
        })

    return {
        "results": results,
        "alive_count": alive_count,
        "n_features": n_features,
        "elapsed_s": elapsed,
        "n_seqs": total_seqs,
        "seq_len": seq_len,
    }

# Reporting

def report(output: dict, output_dir: str | None = None):
    results = output["results"]
    alive_count = output["alive_count"]
    n_features = output["n_features"]

    alive = [r for r in results if r["alive"]]
    dead_count = n_features - alive_count

    print(f"\n{'='*70}")
    print(f"Memory Slot Surgery: Alignment Report")
    print(f"{'='*70}")
    print(f"Features: {n_features} total, {alive_count} alive, {dead_count} dead")
    print(f"Sequences: {output['n_seqs']} x {output['seq_len']} tokens")
    print(f"Time: {output['elapsed_s']:.1f}s")

    if not alive:
        print("No alive features found.")
        return

    # Use absolute cosines (direction shouldn't matter for alignment)
    k_abs = np.array([r["mean_abs_k_cos"] for r in alive])
    v_abs = np.array([r["mean_abs_v_cos"] for r in alive])
    # Combined alignment = geometric mean of |k_cos| and |v_cos|
    combined = np.sqrt(k_abs * v_abs)

    print(f"\n--- Alignment Statistics (absolute cosines, across {len(alive)} alive features) ---")
    for name, arr in [("key alignment |cos|", k_abs), ("value alignment |cos|", v_abs), ("combined (geometric mean)", combined)]:
        print(f"  {name}:")
        print(f"    mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
              f"std={arr.std():.4f}  max={arr.max():.4f}")

    # Null baseline: expected |cosine| between random unit vectors in d dimensions
    # E[|cos|] = sqrt(2/pi) / sqrt(d) for d-dim random vectors
    d = 128
    null_expected = np.sqrt(2 / np.pi) / np.sqrt(d)
    print(f"\n--- Null Baseline ---")
    print(f"  Random 128-dim vectors: E[|cos|] = {null_expected:.4f}")
    print(f"  Random combined: E[geometric mean] = {null_expected:.4f}")

    # Fraction above thresholds
    print(f"\n--- Fraction of alive features above alignment thresholds ---")
    for thresh in [0.1, 0.2, 0.3, 0.5]:
        k_frac = (k_abs > thresh).mean()
        v_frac = (v_abs > thresh).mean()
        c_frac = (combined > thresh).mean()
        print(f"  >{thresh:.1f}: key={k_frac:.3f}  value={v_frac:.3f}  combined={c_frac:.3f}")

    # Top-10 by combined alignment
    sorted_alive = sorted(alive, key=lambda r: np.sqrt(r["mean_abs_k_cos"] * r["mean_abs_v_cos"]), reverse=True)
    print(f"\n--- Top-10 Features by Combined Alignment ---")
    print(f"  {'Feature':>8}  {'|k_cos|':>8}  {'|v_cos|':>8}  {'combined':>8}  {'n_act':>6}")
    for r in sorted_alive[:10]:
        c = np.sqrt(r["mean_abs_k_cos"] * r["mean_abs_v_cos"])
        print(f"  {r['feature']:>8}  {r['mean_abs_k_cos']:>8.4f}  {r['mean_abs_v_cos']:>8.4f}  "
              f"{c:>8.4f}  {r['n_activations']:>6}")

    # Histogram (text-based)
    print(f"\n--- Key Alignment Histogram ---")
    _text_histogram(k_abs, "key |cos|")
    print(f"\n--- Value Alignment Histogram ---")
    _text_histogram(v_abs, "value |cos|")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "alignment_results.json"), "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_dir}/alignment_results.json")

        # Save histograms as numpy arrays for plotting
        np.savez(
            os.path.join(output_dir, "alignment_histograms.npz"),
            k_abs=k_abs, v_abs=v_abs, combined=combined,
        )

def _text_histogram(values: np.ndarray, label: str, bins: int = 20):
    counts, edges = np.histogram(values, bins=bins, range=(0, 1))
    max_count = max(counts) if max(counts) > 0 else 1
    for i, c in enumerate(counts):
        bar = "#" * int(40 * c / max_count)
        print(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}) {c:>4}  {bar}")

# Entry point

def main():
    import argparse

    p = argparse.ArgumentParser(description="Memory Slot Surgery alignment check")
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--sae-checkpoint", required=True, help="Path to bilinear SAE checkpoint")
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--head", type=int, default=4)
    p.add_argument("--n-seqs", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    model, tokenizer = load_model(args.model, args.device)
    sae = load_sae(args.sae_checkpoint, args.device)

    output = compute_alignment(
        model, tokenizer, sae,
        layer_idx=args.layer, head_idx=args.head,
        n_seqs=args.n_seqs, seq_len=args.seq_len,
        batch_size=args.batch_size, top_n=args.top_n,
        device=args.device,
    )

    report(output, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
