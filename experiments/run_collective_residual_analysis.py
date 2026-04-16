"""Collective residual analysis across per-head SAEs.

Two analyses:
  1. ``overlap``: alive-slot Jaccard across heads (slot-persistence proxy).
  2. ``residual``: pairwise cosine, coherence, and empirical alive-slot overlap
     computed on residuals of SAE reconstructions.

    python experiments/run_collective_residual_analysis.py \\
        --states-dir /path/to/states --checkpoint-dir /path/to/ckpts \\
        --output-dir results/data --layer 9
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

# Allow running as a plain script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from core.sae import load_sae_checkpoint

N_HEADS = 16

def _expected_random_jaccard(size_a: int, size_b: int, universe: int) -> float:
    intersection = (size_a * size_b) / max(universe, 1)
    union = size_a + size_b - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)

def _make_subset_list(k: int, rng, n_subsets: int) -> list[list[int]]:
    if k == N_HEADS:
        return [list(range(N_HEADS))]

    seen: set[tuple[int, ...]] = set()
    subsets: list[list[int]] = []
    while len(subsets) < n_subsets:
        subset = tuple(sorted(int(x) for x in rng.choice(N_HEADS, size=k, replace=False)))
        if subset in seen:
            continue
        seen.add(subset)
        subsets.append(list(subset))
    return subsets

def exp_alive_overlap_proxy(
    checkpoint_dir: Path,
    layer: int = 9,
    n_features: int = 2048,
    k: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute alive/dead mask overlap across the matched 16-head checkpoints."""

    def load_alive_mask(sae_type: str, head: int) -> tuple[np.ndarray, dict[str, Any]]:
        ckpt_dir = checkpoint_dir / f"{sae_type}_L{layer}_H{head}_nf{n_features}_k{k}_s{seed}"
        ckpt_path = ckpt_dir / "best.pt"
        cfg_path = ckpt_dir / "config.json"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        sae, _, ckpt = load_sae_checkpoint(
            ckpt_path,
            config_path=cfg_path if cfg_path.exists() else None,
            weights_only=True,
        )

        steps_since = sae.steps_since_active.detach().cpu()
        dead_threshold = int(getattr(sae, "dead_threshold", 100))
        alive_mask = (steps_since < dead_threshold).numpy().astype(bool)
        n_alive = int(alive_mask.sum())

        meta = {
            "head": head,
            "ckpt_dir": str(ckpt_dir),
            "dead_threshold": dead_threshold,
            "n_alive": n_alive,
            "n_dead": int((~alive_mask).sum()),
            "alive_pct": float(100.0 * n_alive / max(len(alive_mask), 1)),
            "val_mse": float(ckpt.get("val_mse")) if ckpt.get("val_mse") is not None else None,
        }
        return alive_mask, meta

    results: dict[str, Any] = {
        "layer": layer,
        "n_heads": N_HEADS,
        "n_features": n_features,
        "k": k,
        "seed": seed,
        "analysis_note": (
            "Pairwise Jaccard is computed on feature slots, not semantically matched features. "
            "Because these SAEs were trained independently, treat this as a slot-persistence proxy."
        ),
        "architectures": {},
    }

    for sae_type in ("flat", "bilinear"):
        alive_masks: list[np.ndarray] = []
        head_meta: list[dict[str, Any]] = []
        for head in range(N_HEADS):
            alive_mask, meta = load_alive_mask(sae_type, head)
            alive_masks.append(alive_mask)
            head_meta.append(meta)

        pair_records = []
        alive_jaccards = []
        dead_jaccards = []
        alive_excess = []
        dead_excess = []

        for i in range(N_HEADS):
            for j in range(i + 1, N_HEADS):
                alive_i = alive_masks[i]
                alive_j = alive_masks[j]
                dead_i = ~alive_i
                dead_j = ~alive_j

                alive_inter = int(np.logical_and(alive_i, alive_j).sum())
                alive_union = int(np.logical_or(alive_i, alive_j).sum())
                dead_inter = int(np.logical_and(dead_i, dead_j).sum())
                dead_union = int(np.logical_or(dead_i, dead_j).sum())

                alive_jaccard = float(alive_inter / max(alive_union, 1))
                dead_jaccard = float(dead_inter / max(dead_union, 1))
                alive_expected = _expected_random_jaccard(int(alive_i.sum()), int(alive_j.sum()), len(alive_i))
                dead_expected = _expected_random_jaccard(int(dead_i.sum()), int(dead_j.sum()), len(dead_i))

                alive_jaccards.append(alive_jaccard)
                dead_jaccards.append(dead_jaccard)
                alive_excess.append(alive_jaccard - alive_expected)
                dead_excess.append(dead_jaccard - dead_expected)

                pair_records.append(
                    {
                        "head_i": i,
                        "head_j": j,
                        "alive_jaccard": alive_jaccard,
                        "alive_expected_random": alive_expected,
                        "alive_excess_over_random": alive_jaccard - alive_expected,
                        "dead_jaccard": dead_jaccard,
                        "dead_expected_random": dead_expected,
                        "dead_excess_over_random": dead_jaccard - dead_expected,
                    }
                )

        results["architectures"][sae_type] = {
            "per_head": head_meta,
            "summary": {
                "mean_alive_pct": float(np.mean([m["alive_pct"] for m in head_meta])),
                "std_alive_pct": float(np.std([m["alive_pct"] for m in head_meta])),
                "mean_alive_jaccard": float(np.mean(alive_jaccards)),
                "mean_dead_jaccard": float(np.mean(dead_jaccards)),
                "mean_alive_excess_over_random": float(np.mean(alive_excess)),
                "mean_dead_excess_over_random": float(np.mean(dead_excess)),
            },
            "pairwise": pair_records,
        }

        print(
            f"{sae_type}: alive={results['architectures'][sae_type]['summary']['mean_alive_pct']:.1f}% "
            f"| alive J={results['architectures'][sae_type]['summary']['mean_alive_jaccard']:.4f} "
            f"| dead J={results['architectures'][sae_type]['summary']['mean_dead_jaccard']:.4f}"
        )

    return results

def exp_residual_correlation(
    states_dir: Path,
    checkpoint_dir: Path,
    layer: int = 9,
    n_features: int = 2048,
    k: int = 32,
    seed: int = 42,
    n_samples: int = 5000,
    batch_size: int = 64,
    coherence_subsets: int = 64,
    coherence_ks: list[int] | None = None,
) -> dict[str, Any]:
    """Measure cross-head residual alignment for flat vs bilinear."""
    if coherence_ks is None:
        coherence_ks = [2, 4, 8, 16]

    torch.set_grad_enabled(False)
    device = "cuda"
    d_in = 128 * 128
    eps = 1e-8

    def load_sae(sae_type: str, head: int):
        ckpt_dir = checkpoint_dir / f"{sae_type}_L{layer}_H{head}_nf{n_features}_k{k}_s{seed}"
        ckpt_path = ckpt_dir / "best.pt"
        cfg_path = ckpt_dir / "config.json"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        sae, _, _ = load_sae_checkpoint(
            ckpt_path,
            config_path=cfg_path if cfg_path.exists() else None,
            device=device,
            weights_only=True,
        )
        return sae, ckpt_dir

    def summarize_pair_matrix(matrix: np.ndarray) -> dict[str, Any]:
        pair_vals = []
        for i in range(N_HEADS):
            for j in range(i + 1, N_HEADS):
                pair_vals.append(float(matrix[i, j]))
        return {
            "mean_offdiag": float(np.mean(pair_vals)),
            "std_offdiag": float(np.std(pair_vals)),
            "min_offdiag": float(np.min(pair_vals)),
            "max_offdiag": float(np.max(pair_vals)),
            "matrix": matrix.tolist(),
        }

    state_root = states_dir

    def process_architecture(sae_type: str) -> dict[str, Any]:
        sample_shape = np.load(f"{state_root}/layer_{layer}/head_0.npy", mmap_mode="r").shape
        n_use = min(n_samples, int(sample_shape[0]))
        residual_path = Path(f"/tmp/{sae_type}_L{layer}_residuals.dat")
        norm_path = Path(f"/tmp/{sae_type}_L{layer}_norms.dat")
        residuals = np.memmap(
            residual_path,
            dtype=np.float16,
            mode="w+",
            shape=(N_HEADS, n_use, d_in),
        )
        norms = np.memmap(
            norm_path,
            dtype=np.float32,
            mode="w+",
            shape=(N_HEADS, n_use),
        )

        head_stats = []
        empirical_alive_masks: list[np.ndarray] = []
        for head in range(N_HEADS):
            sae, ckpt_dir = load_sae(sae_type, head)
            head_states = np.load(f"{state_root}/layer_{layer}/head_{head}.npy", mmap_mode="r")
            empirical_alive = np.zeros(n_features, dtype=bool)

            sqerr_sum = 0.0
            norm_sum = 0.0
            for start in range(0, n_use, batch_size):
                end = min(start + batch_size, n_use)
                batch_np = np.asarray(head_states[start:end], dtype=np.float32)
                batch = torch.from_numpy(batch_np).to(device)
                if sae_type == "flat":
                    out = sae(batch.reshape(batch.shape[0], -1))
                    recon = out.reconstruction.reshape(batch.shape[0], 128, 128)
                else:
                    out = sae(batch)
                    recon = out.reconstruction

                residual = batch - recon
                residual_flat = residual.reshape(residual.shape[0], -1)
                batch_norms = torch.linalg.vector_norm(residual_flat, dim=-1)
                empirical_alive |= out.coefficients.abs().gt(0).any(dim=0).detach().cpu().numpy()

                residuals[head, start:end] = residual_flat.detach().cpu().numpy().astype(np.float16)
                norms[head, start:end] = batch_norms.detach().cpu().numpy().astype(np.float32)

                sqerr_sum += float((residual_flat ** 2).sum().item())
                norm_sum += float(batch_norms.sum().item())

            head_stats.append(
                {
                    "head": head,
                    "ckpt_dir": str(ckpt_dir),
                    "mean_residual_mse": float(sqerr_sum / max(n_use * d_in, 1)),
                    "mean_residual_norm": float(norm_sum / max(n_use, 1)),
                    "empirical_n_alive": int(empirical_alive.sum()),
                    "empirical_alive_pct": float(100.0 * empirical_alive.sum() / max(len(empirical_alive), 1)),
                }
            )
            empirical_alive_masks.append(empirical_alive)
            print(
                f"{sae_type} head {head}: "
                f"MSE={head_stats[-1]['mean_residual_mse']:.4e}, "
                f"norm={head_stats[-1]['mean_residual_norm']:.4f}, "
                f"alive={head_stats[-1]['empirical_n_alive']}"
            )
            del sae
            torch.cuda.empty_cache()

        empirical_alive_jaccards = []
        empirical_dead_jaccards = []
        for i in range(N_HEADS):
            for j in range(i + 1, N_HEADS):
                alive_i = empirical_alive_masks[i]
                alive_j = empirical_alive_masks[j]
                dead_i = ~alive_i
                dead_j = ~alive_j
                empirical_alive_jaccards.append(
                    float(np.logical_and(alive_i, alive_j).sum() / max(np.logical_or(alive_i, alive_j).sum(), 1))
                )
                empirical_dead_jaccards.append(
                    float(np.logical_and(dead_i, dead_j).sum() / max(np.logical_or(dead_i, dead_j).sum(), 1))
                )

        signed_cos_sum = np.zeros((N_HEADS, N_HEADS), dtype=np.float64)
        abs_cos_sum = np.zeros((N_HEADS, N_HEADS), dtype=np.float64)
        sample_count = 0

        rng = np.random.default_rng(seed)
        subset_lookup = {int(k_val): _make_subset_list(int(k_val), rng, coherence_subsets) for k_val in coherence_ks}
        coherence_sum = {int(k_val): 0.0 for k_val in coherence_ks}
        coherence_sq_sum = {int(k_val): 0.0 for k_val in coherence_ks}
        coherence_count = {int(k_val): 0 for k_val in coherence_ks}

        for start in range(0, n_use, batch_size):
            end = min(start + batch_size, n_use)
            if end <= start:
                continue

            residual_batch = torch.from_numpy(np.asarray(residuals[:, start:end, :], dtype=np.float32)).to(device)
            norm_batch = torch.from_numpy(np.asarray(norms[:, start:end], dtype=np.float32)).to(device)
            denom = torch.clamp(torch.einsum("hb,jb->bhj", norm_batch, norm_batch), min=eps)

            gram = torch.einsum("hbd,jbd->bhj", residual_batch, residual_batch)
            cos = gram / denom
            signed_cos_sum += cos.sum(dim=0).detach().cpu().numpy()
            abs_cos_sum += cos.abs().sum(dim=0).detach().cpu().numpy()
            sample_count += (end - start)

            for k_val, subsets in subset_lookup.items():
                for subset in subsets:
                    subset_batch = residual_batch[subset]
                    summed = subset_batch.sum(dim=0)
                    numerator = torch.linalg.vector_norm(summed, dim=-1)
                    denominator = torch.sqrt(torch.clamp((subset_batch ** 2).sum(dim=(0, 2)), min=eps))
                    coherence = numerator / denominator
                    coherence_sum[k_val] += float(coherence.sum().item())
                    coherence_sq_sum[k_val] += float((coherence ** 2).sum().item())
                    coherence_count[k_val] += int(coherence.numel())

        signed_cos_mean = signed_cos_sum / max(sample_count, 1)
        abs_cos_mean = abs_cos_sum / max(sample_count, 1)

        coherence_summary = {}
        for k_val in coherence_ks:
            total = coherence_count[int(k_val)]
            mean_val = coherence_sum[int(k_val)] / max(total, 1)
            mean_sq = coherence_sq_sum[int(k_val)] / max(total, 1)
            variance = max(mean_sq - mean_val ** 2, 0.0)
            coherence_summary[str(int(k_val))] = {
                "mean": float(mean_val),
                "std": float(math.sqrt(variance)),
                "n_observations": int(total),
                "n_subsets": len(subset_lookup[int(k_val)]),
            }

        return {
            "n_samples": n_use,
            "batch_size": batch_size,
            "per_head": head_stats,
            "mean_residual_mse_across_heads": float(np.mean([x["mean_residual_mse"] for x in head_stats])),
            "mean_residual_norm_across_heads": float(np.mean([x["mean_residual_norm"] for x in head_stats])),
            "empirical_alive": {
                "mean_alive_pct": float(np.mean([x["empirical_alive_pct"] for x in head_stats])),
                "std_alive_pct": float(np.std([x["empirical_alive_pct"] for x in head_stats])),
                "mean_alive_jaccard": float(np.mean(empirical_alive_jaccards)),
                "mean_dead_jaccard": float(np.mean(empirical_dead_jaccards)),
            },
            "pairwise_signed_cosine": summarize_pair_matrix(signed_cos_mean),
            "pairwise_abs_cosine": summarize_pair_matrix(abs_cos_mean),
            "coherence": coherence_summary,
        }

    results = {
        "layer": layer,
        "n_heads": N_HEADS,
        "n_features": n_features,
        "k": k,
        "seed": seed,
        "n_samples": n_samples,
        "architectures": {},
    }

    t0 = time.time()
    for sae_type in ("flat", "bilinear"):
        arch_t0 = time.time()
        results["architectures"][sae_type] = process_architecture(sae_type)
        results["architectures"][sae_type]["elapsed_s"] = round(time.time() - arch_t0, 1)
        print(
            f"{sae_type}: abs-cos={results['architectures'][sae_type]['pairwise_abs_cosine']['mean_offdiag']:.4f}, "
            f"coh16={results['architectures'][sae_type]['coherence'].get('16', {}).get('mean')}"
        )

    flat = results["architectures"]["flat"]
    bilinear = results["architectures"]["bilinear"]
    results["comparison"] = {
        "flat_minus_bilinear_abs_cos_mean": (
            flat["pairwise_abs_cosine"]["mean_offdiag"] - bilinear["pairwise_abs_cosine"]["mean_offdiag"]
        ),
        "flat_minus_bilinear_signed_cos_mean": (
            flat["pairwise_signed_cosine"]["mean_offdiag"] - bilinear["pairwise_signed_cosine"]["mean_offdiag"]
        ),
        "flat_minus_bilinear_mean_residual_mse": (
            flat["mean_residual_mse_across_heads"] - bilinear["mean_residual_mse_across_heads"]
        ),
        "coherence_mean_deltas": {
            k_val: (
                flat["coherence"][k_val]["mean"] - bilinear["coherence"][k_val]["mean"]
            )
            for k_val in flat["coherence"].keys()
        },
    }
    results["elapsed_s"] = round(time.time() - t0, 1)
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-dir", type=Path, required=True,
                        help="Root containing layer_<L>/head_<H>.npy arrays")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Root containing <sae_type>_L{L}_H{H}_nf{nf}_k{k}_s{seed} subdirs")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where the combined JSON result is written")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B",
                        help="Unused here; kept for CLI consistency across scripts")
    parser.add_argument("--model-cache-dir", type=str, default=None)
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sequences", type=int, default=5000,
                        help="Number of state samples (aliased to --n-samples)")
    parser.add_argument("--n-features", type=int, default=2048)
    parser.add_argument("--k", type=int, default=32, help="Top-k used at training time")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--analysis", type=str, default="all",
                        choices=["all", "overlap", "residual"])
    parser.add_argument("--coherence-subsets", type=int, default=64)
    parser.add_argument("--coherence-ks", type=str, default="2,4,8,16")
    parser.add_argument("--output-name", type=str,
                        default="collective_residual_analysis_clean_08b.json")
    args = parser.parse_args()

    if args.model_cache_dir:
        import os
        os.environ.setdefault("HF_HOME", args.model_cache_dir)

    start = time.time()
    local_results: dict[str, Any] = {
        "metadata": {
            "analysis": args.analysis,
            "layer": args.layer,
            "n_features": args.n_features,
            "k": args.k,
            "seed": args.seed,
            "n_samples": args.n_sequences,
            "batch_size": args.batch_size,
        }
    }

    if args.analysis in {"all", "overlap"}:
        print("Running alive-slot overlap proxy...")
        overlap = exp_alive_overlap_proxy(
            checkpoint_dir=args.checkpoint_dir,
            layer=args.layer,
            n_features=args.n_features,
            k=args.k,
            seed=args.seed,
        )
        local_results["alive_overlap_proxy"] = overlap
        for sae_type, arch in overlap["architectures"].items():
            summary = arch["summary"]
            print(
                f"  {sae_type}: alive={summary['mean_alive_pct']:.1f}% "
                f"aliveJ={summary['mean_alive_jaccard']:.4f} "
                f"deadJ={summary['mean_dead_jaccard']:.4f}"
            )

    if args.analysis in {"all", "residual"}:
        print("Running residual correlation analysis...")
        coherence_ks = [int(x.strip()) for x in args.coherence_ks.split(",") if x.strip()]
        residual = exp_residual_correlation(
            states_dir=args.states_dir,
            checkpoint_dir=args.checkpoint_dir,
            layer=args.layer,
            n_features=args.n_features,
            k=args.k,
            seed=args.seed,
            n_samples=args.n_sequences,
            batch_size=args.batch_size,
            coherence_subsets=args.coherence_subsets,
            coherence_ks=coherence_ks,
        )
        local_results["residual_correlation"] = residual
        for sae_type, arch in residual["architectures"].items():
            print(
                f"  {sae_type}: abs-cos={arch['pairwise_abs_cosine']['mean_offdiag']:.4f}, "
                f"coh16={arch['coherence'].get('16', {}).get('mean'):.4f}, "
                f"mean MSE={arch['mean_residual_mse_across_heads']:.4e}"
            )

    local_results["elapsed_s"] = round(time.time() - start, 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_name
    out_path.write_text(json.dumps(local_results, indent=2))
    print(f"Saved local summary to {out_path}")

if __name__ == "__main__":
    main()
