#!/usr/bin/env python3
import argparse
import json
import os
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F

from core.sae import FlatSAE, MatrixSAE, BilinearMatrixSAE, load_sae_checkpoint

class S0Decomposition(TypedDict):
    active_feature_indices: list[int]
    coefficients: list[float]
    n_active: int
    n_total: int
    active_fraction: float
    sparsity: float

class TopExample(TypedDict):
    rank: int
    sample_idx: int
    activation: float
    text: str

class GainedFeature(TypedDict):
    feature: int
    coefficient: float

class DeltaFeature(TypedDict):
    feature: int
    delta: float
    zero_coef: float
    trained_coef: float

class S0Comparison(TypedDict):
    zero_state: S0Decomposition
    trained_state: S0Decomposition
    gained_features: list[GainedFeature]
    strengthened_features: list[DeltaFeature]
    suppressed_features: list[DeltaFeature]
    n_gained: int
    n_strengthened: int
    n_suppressed: int
    cosine_similarity: float

class ProjectionFeature(TypedDict):
    feature: int
    score: float
    abs_score: float

class DecoderDecomposition(TypedDict, total=False):
    method: str
    n_features_total: int
    projection_top_k: list[ProjectionFeature]
    s0_norm: float
    projection_scores_std: float
    projection_scores_max: float
    error: str

def top_activating_examples(
    sae: torch.nn.Module, states: torch.Tensor,
    texts: list[str], feature_idx: int, n: int = 20,
) -> list[TopExample]:
    col = _encode(sae, states)[:, feature_idx]
    active_idx = torch.nonzero(col > 0, as_tuple=True)[0]
    if active_idx.numel() == 0:
        return []
    active_vals = col[active_idx]
    topk = torch.topk(active_vals, min(n, int(active_idx.numel())))
    top_indices = active_idx[topk.indices]
    return [{"rank": r, "sample_idx": int(i), "activation": v.item(),
             "text": texts[int(i)] if int(i) < len(texts) else ""}
            for r, (v, i) in enumerate(zip(topk.values, top_indices))]

def feature_activation_stats(sae: torch.nn.Module, states: torch.Tensor) -> dict[str, np.ndarray]:
    a = _encode(sae, states).cpu().numpy()
    return {"mean": a.mean(0), "frequency": (a > 0).mean(0), "max": a.max(0)}

def dead_features(sae: torch.nn.Module, states: torch.Tensor, thr: float = 1e-8) -> np.ndarray:
    return np.where(feature_activation_stats(sae, states)["max"] < thr)[0]

def rank_analysis(states: torch.Tensor) -> dict[str, np.ndarray]:
    assert states.ndim == 3, f"Expected (N, d, d), got {states.shape}"
    N = states.shape[0]
    sv_all = np.zeros((N, min(states.shape[1], states.shape[2])), dtype=np.float32)
    eff_rank = np.zeros(N, dtype=np.float32)
    for i in range(0, N, 512):
        j = min(i + 512, N)
        S = torch.linalg.svdvals(states[i:j].float()).cpu().numpy()
        sv_all[i:j] = S
        p = S / (S.sum(1, keepdims=True) + 1e-12)
        eff_rank[i:j] = np.exp(-(p * np.log(p + 1e-12)).sum(1))
    return {"singular_values": sv_all, "effective_rank": eff_rank}

def compare_flat_vs_rank1(
    flat_sae: torch.nn.Module, rank1_sae: torch.nn.Module, states: torch.Tensor,
) -> dict[str, float]:
    flat = _flatten(states)
    total_var = flat.var().item()
    results = {}
    for name, sae in [("flat", flat_sae), ("rank1", rank1_sae)]:
        with torch.no_grad():
            recon, acts = _reconstruct(sae, flat), _encode(sae, states)
        mse = F.mse_loss(recon, flat).item()
        results[f"{name}_mse"] = mse
        results[f"{name}_l0"] = (acts > 0).float().sum(1).mean().item()
        results[f"{name}_explained_variance"] = 1.0 - mse / (total_var + 1e-12)
    return results

def s0_decomposition(sae: torch.nn.Module, s0: torch.Tensor) -> S0Decomposition:
    """Encode a single state matrix through the SAE and report active features.

    Args:
        sae: trained SAE (any variant: Flat, MatrixSAE, BilinearMatrixSAE)
        s0: state tensor, shape (d_k, d_v) or (d_k * d_v,) for a single head
    """
    x = s0.reshape(1, -1)
    acts = _encode(sae, x).squeeze(0)
    mask = acts > 0
    idx, coef = torch.where(mask)[0].cpu().numpy(), acts[mask].cpu().numpy()
    order = np.argsort(-coef)
    active_fraction = len(idx) / acts.shape[0]
    return {"active_feature_indices": idx[order].tolist(), "coefficients": coef[order].tolist(),
            "n_active": len(idx), "n_total": int(acts.shape[0]),
            "active_fraction": active_fraction, "sparsity": 1.0 - active_fraction}

def s0_compare(
    sae: torch.nn.Module,
    zero_state: torch.Tensor,
    trained_state: torch.Tensor,
) -> S0Comparison:
    """Compare feature profiles of a zero (baseline) state vs an S0-trained state.

    Returns per-state decomposition plus a diff showing which features
    task adaptation activates or suppresses.
    """
    zero_dec = s0_decomposition(sae, zero_state)
    trained_dec = s0_decomposition(sae, trained_state)

    zero_acts = _encode(sae, zero_state.reshape(1, -1)).squeeze(0).cpu().numpy()
    trained_acts = _encode(sae, trained_state.reshape(1, -1)).squeeze(0).cpu().numpy()
    diff = trained_acts - zero_acts

    gained_mask = (trained_acts > 0) & (zero_acts == 0)
    strengthened_mask = (trained_acts > zero_acts) & (zero_acts > 0)
    suppressed_mask = (trained_acts < zero_acts) & (zero_acts > 0)

    gained_idx = np.where(gained_mask)[0]
    gained_order = np.argsort(-trained_acts[gained_idx])
    gained: list[GainedFeature] = [
        {"feature": int(gained_idx[i]), "coefficient": float(trained_acts[gained_idx[i]])}
        for i in gained_order[:20]
    ]

    strengthened_idx = np.where(strengthened_mask)[0]
    str_order = np.argsort(-diff[strengthened_idx])
    strengthened: list[DeltaFeature] = [
        {"feature": int(strengthened_idx[i]),
         "delta": float(diff[strengthened_idx[i]]),
         "zero_coef": float(zero_acts[strengthened_idx[i]]),
         "trained_coef": float(trained_acts[strengthened_idx[i]])}
        for i in str_order[:20]
    ]

    suppressed_idx = np.where(suppressed_mask)[0]
    sup_order = np.argsort(diff[suppressed_idx])
    suppressed: list[DeltaFeature] = [
        {"feature": int(suppressed_idx[i]),
         "delta": float(diff[suppressed_idx[i]]),
         "zero_coef": float(zero_acts[suppressed_idx[i]]),
         "trained_coef": float(trained_acts[suppressed_idx[i]])}
        for i in sup_order[:20]
    ]

    return {
        "zero_state": zero_dec,
        "trained_state": trained_dec,
        "gained_features": gained,
        "strengthened_features": strengthened,
        "suppressed_features": suppressed,
        "n_gained": int(gained_mask.sum()),
        "n_strengthened": int(strengthened_mask.sum()),
        "n_suppressed": int(suppressed_mask.sum()),
        "cosine_similarity": float(
            np.dot(zero_acts, trained_acts)
            / (np.linalg.norm(zero_acts) * np.linalg.norm(trained_acts) + 1e-12)
        ),
    }

def s0_decoder_decomposition(
    sae: torch.nn.Module,
    s0_state: torch.Tensor,
    k: int = 32,
) -> DecoderDecomposition:
    """Decompose S0 by projecting onto SAE decoder directions (bypasses encoder bias).

    For rank-1 SAEs: score_i = <S0 - bias, v_i @ w_i^T> = v_i^T @ (S0 - bias) @ w_i
    For flat SAEs: score_i = decoder_col_i^T @ (s0_flat - bias)

    Returns top-k features by projection score, plus NNLS decomposition.
    """
    s0_state = s0_state.detach()
    with torch.no_grad():
        if isinstance(sae, (MatrixSAE, BilinearMatrixSAE)):
            bias = sae.bias  # (d_k, d_v)
            s0_centered = s0_state.float() - bias.float()

            if isinstance(sae, BilinearMatrixSAE):
                V = sae.V_dec.float()  # (n_features, d_k)
                W = sae.W_dec.float()  # (n_features, d_v)
            else:
                V = sae.V.float()
                W = sae.W.float()

            if V.ndim == 3:
                scores = torch.einsum("irk,kv,irv->i", V, s0_centered, W).detach().cpu().numpy()
            else:
                scores = torch.einsum("ik,kv,iv->i", V, s0_centered, W).detach().cpu().numpy()

            n_features = V.shape[0]
            s0_flat = s0_centered.detach().reshape(-1).cpu().numpy()

        elif isinstance(sae, FlatSAE):
            flat_bias = sae.decoder.bias.detach().float()  # (d_in,)
            s0_flat_t = s0_state.reshape(-1).float() - flat_bias
            s0_flat = s0_flat_t.cpu().numpy()

            decoder_columns = sae.decoder.weight.detach().float().T  # (n_features, d_in)
            scores = (decoder_columns @ s0_flat_t).cpu().numpy()

            n_features = decoder_columns.shape[0]
        else:
            return {"error": f"Unknown SAE type: {type(sae)}"}

    topk_idx = np.argsort(-np.abs(scores))[:k]
    projection_features: list[ProjectionFeature] = [
        {"feature": int(i), "score": float(scores[i]), "abs_score": float(abs(scores[i]))}
        for i in topk_idx
    ]

    s0_norm = float(np.linalg.norm(s0_flat))
    return {
        "method": "decoder_decomposition",
        "n_features_total": int(n_features),
        "projection_top_k": projection_features,
        "s0_norm": s0_norm,
        "projection_scores_std": float(np.std(scores)),
        "projection_scores_max": float(np.max(np.abs(scores))),
    }

def _flatten(states: torch.Tensor) -> torch.Tensor:
    return states.reshape(states.shape[0], -1) if states.ndim == 3 else states

@torch.no_grad()
def _encode(sae: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return sae.encode(x)  # type: ignore[operator]

def _reconstruct(sae: torch.nn.Module, flat: torch.Tensor) -> torch.Tensor:
    out = sae(flat)
    if isinstance(out, tuple):
        return out[0]
    return out.reconstruction if hasattr(out, "reconstruction") else out

def _load_sae(path: str) -> torch.nn.Module:
    sae, cfg, _ = load_sae_checkpoint(path)
    if not cfg:
        print(f"WARNING: no config in checkpoint {path}, using defaults (d_k=128, d_v=128, ef=4, k=32)")
    return sae

def _load_data(data_dir: str, layer: int, head: int) -> tuple[torch.Tensor, list[str]]:
    npy_path = os.path.join(data_dir, f"layer_{layer}", f"head_{head}.npy")
    pt_path = os.path.join(data_dir, f"states_layer{layer}_head{head}.pt")
    if os.path.exists(npy_path):
        states = torch.from_numpy(np.load(npy_path, mmap_mode="r").astype(np.float32))
    elif os.path.exists(pt_path):
        states = torch.load(pt_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No state data found at {npy_path} or {pt_path}")
    tp = os.path.join(data_dir, "texts.json")
    return states, json.load(open(tp)) if os.path.exists(tp) else []

def _save_json(obj: object, path: str) -> None:
    json.dump(obj, open(path, "w"), indent=2)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--head", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--rank1_checkpoint", default=None)
    p.add_argument("--s0_state_path", default=None)
    args = p.parse_args()
    os.makedirs(out := args.output_dir, exist_ok=True)
    sae = _load_sae(args.sae_checkpoint)
    states, texts = _load_data(args.data_dir, args.layer, args.head)
    print(f"Loaded states {states.shape}, {len(texts)} texts")

    stats = feature_activation_stats(sae, states)
    np.savez(os.path.join(out, "activation_stats.npz"), **stats)
    dead = dead_features(sae, states)
    np.save(os.path.join(out, "dead_features.npy"), dead)
    n_feat = len(stats["mean"])
    print(f"Dead features: {len(dead)}/{n_feat} ({100*len(dead)/n_feat:.1f}%)")

    if states.ndim == 3:
        rd = rank_analysis(states)
        np.savez(os.path.join(out, "rank_analysis.npz"), **rd)
        print(f"Effective rank: mean={rd['effective_rank'].mean():.2f}, "
              f"median={np.median(rd['effective_rank']):.2f}")

    if texts:
        top_feature_ids = [int(i) for i in np.argsort(-stats["max"]) if stats["max"][i] > 0][:min(10, n_feat)]
        top = {str(i): top_activating_examples(sae, states, texts, i) for i in top_feature_ids}
        _save_json(top, os.path.join(out, "top_examples.json"))

    if args.rank1_checkpoint:
        cmp = compare_flat_vs_rank1(sae, _load_sae(args.rank1_checkpoint), states)
        _save_json(cmp, os.path.join(out, "flat_vs_rank1.json"))
        for k, v in cmp.items():
            print(f"  {k}: {v:.6f}")

    if args.s0_state_path:
        s0_raw = torch.load(args.s0_state_path, map_location="cpu", weights_only=True)
        if isinstance(s0_raw, dict):
            key = str(args.layer) if str(args.layer) in s0_raw else list(s0_raw.keys())[0]
            s0_layer = s0_raw[key]
            if s0_layer.ndim == 3:  # (n_heads, d_k, d_v) -> pick head
                s0 = s0_layer[args.head]
            else:
                s0 = s0_layer
        elif s0_raw.ndim == 3:  # (n_heads, d_k, d_v)
            s0 = s0_raw[args.head]
        else:
            s0 = s0_raw

        s0d = s0_decomposition(sae, s0)
        _save_json(s0d, os.path.join(out, "s0_decomposition.json"))
        print(f"S0: {s0d['n_active']}/{s0d['n_total']} active ({100*s0d['active_fraction']:.1f}%)")

        zero = torch.zeros_like(s0)
        cmp_result = s0_compare(sae, zero, s0)
        _save_json(cmp_result, os.path.join(out, "s0_comparison.json"))
        print(f"S0 vs zero: gained={cmp_result['n_gained']} "
              f"strengthened={cmp_result['n_strengthened']} "
              f"suppressed={cmp_result['n_suppressed']} "
              f"cosine={cmp_result['cosine_similarity']:.4f}")

    print(f"Results saved to {out}")

if __name__ == "__main__":
    main()
