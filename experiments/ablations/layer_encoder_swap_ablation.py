#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from core.train import train as train_sae_fn

def _code_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__) or ".",
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "unknown"

CODE_SHA = _code_sha()

# Experiment configuration

LAYERS = [1, 9, 17]
N_HEADS = 16
ALL_SAE_TYPES = ["flat", "rank1", "bilinear", "bilinear_tied", "bilinear_flat"]
SEEDS = [0, 1, 42]

SPECTRAL_LABELS = {
    1:  "~12 (concentrated)",
    9:  "6.8 (medium)",
    17: "3.0 (diffuse)",
}

SAE_N_FEATURES = 2048
SAE_K = 32
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 256
TRAIN_LR = 3e-4
TRAIN_LR_MIN = 3e-5
TRAIN_WARMUP_STEPS = 50
TRAIN_RESAMPLE_EVERY = 250

TOTAL_JOBS = len(LAYERS) * N_HEADS * len(ALL_SAE_TYPES) * len(SEEDS)  # 720

def train_sae(
    layer: int,
    head: int,
    sae_type: str,
    seed: int,
    states_dir: Path,
    ablation_dir: Path,
    n_features: int = SAE_N_FEATURES,
    k: int = SAE_K,
) -> dict[str, Any]:
    """Train one SAE on one (layer, head) pair. Skips if checkpoint exists."""
    ckpt_dir = ablation_dir / f"layer_{layer}" / f"head_{head}" / f"{sae_type}_s{seed}"
    best_path = ckpt_dir / "best.pt"

    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        ckpt_code_sha = str(cfg.get("code_sha", ckpt.get("code_sha", "unknown")))
        if (
            cfg.get("sae_type") == sae_type
            and int(cfg.get("layer", layer)) == layer
            and int(cfg.get("head", head)) == head
            and int(cfg.get("n_features", n_features)) == n_features
            and int(cfg.get("k", k)) == k
            and ckpt_code_sha == CODE_SHA
        ):
            val_mse = ckpt.get("val_mse", ckpt.get("best_val_mse"))
            print(f"L{layer} H{head} {sae_type} s{seed}: already exists (val_mse={val_mse})")
            return {
                "layer": layer, "head": head, "sae_type": sae_type, "seed": seed,
                "best_mse": val_mse, "skipped": True,
            }
        print(
            f"L{layer} H{head} {sae_type} s{seed}: checkpoint exists but metadata/code_sha "
            "mismatch; retraining"
        )

    data_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"State data not found: {data_path}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    result = train_sae_fn(
        sae_type=sae_type,
        data_dir=str(states_dir),
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
        f"val_mse={result.get('best_mse', '?')}"
    )
    return {"layer": layer, "head": head, "sae_type": sae_type, "seed": seed, **result}

def _load_checkpoint_mse(ckpt_path: Path) -> float | None:
    """Load val_mse from a best.pt checkpoint."""
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("val_mse", ckpt.get("best_val_mse"))

def analyze_results(ablation_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Load all checkpoints and compute layer-average MSE per SAE type."""
    all_mses: dict[int, dict[str, dict[int, list[float]]]] = {}
    missing: list[str] = []

    for layer in LAYERS:
        all_mses[layer] = {}
        for sae_type in ALL_SAE_TYPES:
            all_mses[layer][sae_type] = {}
            for head in range(N_HEADS):
                seed_vals = []
                for seed in SEEDS:
                    path = ablation_dir / f"layer_{layer}" / f"head_{head}" / f"{sae_type}_s{seed}" / "best.pt"
                    val = _load_checkpoint_mse(path)
                    if val is not None:
                        seed_vals.append(float(val))
                    else:
                        missing.append(f"L{layer} H{head} {sae_type} s{seed}")
                all_mses[layer][sae_type][head] = seed_vals

    if missing:
        sample = ", ".join(missing[:15])
        extra = f", ... (+{len(missing) - 15} more)" if len(missing) > 15 else ""
        raise RuntimeError(f"Missing {len(missing)} checkpoints: {sample}{extra}")

    per_head_mean: dict[int, dict[str, dict[int, float]]] = {}
    layer_avg: dict[int, dict[str, float]] = {}
    layer_std: dict[int, dict[str, float]] = {}

    for layer in LAYERS:
        per_head_mean[layer] = {}
        layer_avg[layer] = {}
        layer_std[layer] = {}
        for sae_type in ALL_SAE_TYPES:
            head_means = []
            per_head_mean[layer][sae_type] = {}
            for head in range(N_HEADS):
                m = float(np.mean(all_mses[layer][sae_type][head]))
                per_head_mean[layer][sae_type][head] = m
                head_means.append(m)
            layer_avg[layer][sae_type] = float(np.mean(head_means))
            layer_std[layer][sae_type] = float(np.std(head_means))

    print(f"\n{'=' * 100}")
    print("TABLE 1: LAYER-AVERAGE MSE PER SAE TYPE")
    print(f"{'=' * 100}")
    hdr = f"{'Layer':>8}  {'Spectrum':>22}"
    for t in ALL_SAE_TYPES:
        hdr += f"  {t:>14}"
    print(hdr)
    print("-" * 100)

    for layer in LAYERS:
        row = f"{'L' + str(layer):>8}  {SPECTRAL_LABELS[layer]:>22}"
        for t in ALL_SAE_TYPES:
            row += f"  {layer_avg[layer][t]:.4e}"
        print(row)

    print(f"\n{'=' * 100}")
    print("TABLE 2: % ADVANTAGE VS FLAT (positive = lower MSE than flat)")
    print(f"{'=' * 100}")
    hdr = f"{'Layer':>8}  {'Spectrum':>22}"
    for t in ALL_SAE_TYPES:
        if t == "flat":
            continue
        hdr += f"  {t:>14}"
    print(hdr)
    print("-" * 100)

    for layer in LAYERS:
        flat_val = layer_avg[layer]["flat"]
        row = f"{'L' + str(layer):>8}  {SPECTRAL_LABELS[layer]:>22}"
        for t in ALL_SAE_TYPES:
            if t == "flat":
                continue
            adv = (flat_val - layer_avg[layer][t]) / flat_val * 100.0
            row += f"  {adv:>+13.1f}%"
        print(row)

    def _paired_test(layer: int, type_a: str, type_b: str) -> tuple[float, float, str]:
        a_vals = [per_head_mean[layer][type_a][h] for h in range(N_HEADS)]
        b_vals = [per_head_mean[layer][type_b][h] for h in range(N_HEADS)]
        diffs = [b - a for a, b in zip(a_vals, b_vals)]
        mean_b = float(np.mean(b_vals))
        pct = float(np.mean(diffs)) / mean_b * 100.0 if abs(mean_b) > 1e-12 else 0.0
        _, p_val = stats.ttest_rel(b_vals, a_vals)
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"
        return pct, float(p_val), sig

    comparisons = [
        ("bilinear_flat", "flat",          "Encoder effect: bilinear_flat vs flat (same decoder, different encoder)"),
        ("bilinear",      "bilinear_flat", "Decoder effect: bilinear vs bilinear_flat (same encoder, different decoder)"),
        ("bilinear",      "flat",          "Full bilinear vs flat"),
        ("rank1",         "flat",          "rank1 vs flat (decoder-only change)"),
    ]

    print(f"\n{'=' * 100}")
    print("KEY HYPOTHESIS TESTS (paired t-test across 16 heads)")
    print("Does bilinear_flat beat flat on low-sv1/sv2 layers?")
    print(f"{'=' * 100}")

    for type_a, type_b, desc in comparisons:
        print(f"\n  {desc}:")
        for layer in LAYERS:
            pct, p_val, sig = _paired_test(layer, type_a, type_b)
            winner = type_a if pct > 0 else type_b
            print(f"    L{layer:>2} ({SPECTRAL_LABELS[layer]:>22}): "
                  f"{pct:>+6.2f}% (p={p_val:.4f} {sig})  [{winner} wins]")

    print(f"\n{'=' * 100}")
    print("PER-HEAD BREAKDOWN: L9 (paper's main layer)")
    print(f"{'=' * 100}")
    hdr = f"{'Head':>6}"
    for t in ALL_SAE_TYPES:
        hdr += f"  {t:>14}"
    hdr += f"  {'bf_vs_flat':>10}"
    print(hdr)
    print("-" * 100)

    for head in range(N_HEADS):
        row = f"{'H' + str(head):>6}"
        for t in ALL_SAE_TYPES:
            row += f"  {per_head_mean[9][t][head]:.4e}"
        flat_h = per_head_mean[9]["flat"][head]
        bf_h = per_head_mean[9]["bilinear_flat"][head]
        adv = (flat_h - bf_h) / flat_h * 100.0 if abs(flat_h) > 1e-12 else 0.0
        row += f"  {adv:>+9.1f}%"
        print(row)

    per_layer_results = []
    for layer in LAYERS:
        flat_val = layer_avg[layer]["flat"]
        type_results = {}
        for t in ALL_SAE_TYPES:
            adv = (flat_val - layer_avg[layer][t]) / flat_val * 100.0 if abs(flat_val) > 1e-12 else 0.0
            type_results[t] = {
                "layer_avg_mse": layer_avg[layer][t],
                "layer_std_mse": layer_std[layer][t],
                "advantage_vs_flat_pct": adv,
                "per_head_mean_mse": {str(h): per_head_mean[layer][t][h] for h in range(N_HEADS)},
            }
        per_layer_results.append({
            "layer": layer,
            "spectral_label": SPECTRAL_LABELS[layer],
            "mse_by_type": type_results,
        })

    hypothesis_tests = []
    for type_a, type_b, desc in comparisons:
        test_results = {}
        for layer in LAYERS:
            pct, p_val, sig = _paired_test(layer, type_a, type_b)
            test_results[f"L{layer}"] = {
                "pct_advantage": pct,
                "p_value": p_val,
                "significance": sig,
            }
        hypothesis_tests.append({
            "comparison": desc,
            "type_a": type_a,
            "type_b": type_b,
            "results": test_results,
        })

    output = {
        "experiment": "layer_encoder_swap_ablation",
        "hypothesis": (
            "bilinear encoder drives reconstruction gains on diffuse-spectrum layers; "
            "rank-1 decoder is secondary"
        ),
        "layers": LAYERS,
        "n_heads": N_HEADS,
        "sae_types": ALL_SAE_TYPES,
        "seeds": SEEDS,
        "spectral_labels": {str(k): v for k, v in SPECTRAL_LABELS.items()},
        "train_config": {
            "n_features": SAE_N_FEATURES,
            "k": SAE_K,
            "epochs": TRAIN_EPOCHS,
            "batch_size": TRAIN_BATCH_SIZE,
            "lr": TRAIN_LR,
            "lr_min": TRAIN_LR_MIN,
            "warmup_steps": TRAIN_WARMUP_STEPS,
            "resample_every": TRAIN_RESAMPLE_EVERY,
        },
        "per_layer": per_layer_results,
        "hypothesis_tests": hypothesis_tests,
        "code_sha": CODE_SHA,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "layer_encoder_swap_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output

def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-level encoder-swap ablation (3 layers x 16 heads).")
    parser.add_argument("--stage", default="all", choices=["train", "analyze", "all"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B",
                        help="Source model (for provenance; states should already be extracted).")
    parser.add_argument("--states-dir", type=Path, required=True,
                        help="Directory containing pre-extracted GDN states (layer_*/head_*.npy).")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Output directory for ablation SAE checkpoints.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for analysis JSON output.")
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=-1,
                        help="Run only this layer. -1 = all LAYERS.")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Run only this seed. -1 = all seeds.")
    parser.add_argument("--n-sequences", type=int, default=-1,
                        help="Unused for this script (states are pre-extracted).")
    args = parser.parse_args()

    states_dir = args.states_dir
    ablation_dir = args.checkpoint_dir

    target_layers = [args.layer] if args.layer >= 0 else LAYERS
    seeds_to_run = [args.seed] if args.seed >= 0 else SEEDS

    if args.stage in ("train", "all"):
        n_jobs = len(target_layers) * N_HEADS * len(ALL_SAE_TYPES) * len(seeds_to_run)
        print("=== Training layer-level encoder-swap SAEs ===")
        print(f"{len(target_layers)} layers x {N_HEADS} heads x {len(ALL_SAE_TYPES)} types x {len(seeds_to_run)} seeds = {n_jobs} jobs")

        jobs: list[dict[str, Any]] = []
        for ly in target_layers:
            for head in range(N_HEADS):
                for sae_type in ALL_SAE_TYPES:
                    for seed in seeds_to_run:
                        jobs.append({
                            "layer": ly,
                            "head": head,
                            "sae_type": sae_type,
                            "seed": seed,
                        })

        print(f"Running {len(jobs)}/{n_jobs} jobs sequentially")

        results = []
        failures = []
        for idx, job in enumerate(jobs, start=1):
            try:
                result = train_sae(job["layer"], job["head"], job["sae_type"], job["seed"],
                                   states_dir, ablation_dir)
                results.append(result)
                if idx % 50 == 0 or idx == len(jobs):
                    print(f"  [{idx}/{n_jobs}] completed ({len(failures)} failures so far)")
            except Exception as exc:
                failures.append({
                    "layer": job["layer"],
                    "head": job["head"],
                    "sae_type": job["sae_type"],
                    "seed": job["seed"],
                    "error": str(exc),
                })
                print(
                    f"  [{idx}/{n_jobs}] FAILED: L{job['layer']} H{job['head']} "
                    f"{job['sae_type']} s{job['seed']}: {exc}"
                )

        n_ok = len(results)
        print(f"\nTraining complete: {n_ok}/{n_jobs} succeeded, {len(failures)} failed")
        if failures:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            failures_path = args.output_dir / "layer_encoder_swap_training_failures.json"
            with open(failures_path, "w") as f:
                json.dump({"failures": failures, "code_sha": CODE_SHA}, f, indent=2)
            print("Failed jobs:")
            for fail in failures[:20]:
                print(f"  L{fail['layer']} H{fail['head']} {fail['sae_type']} s{fail['seed']}: {fail['error']}")
            if len(failures) > 20:
                print(f"  ... and {len(failures) - 20} more")
            print(f"Saved failure summary to {failures_path}")

    if args.stage in ("analyze", "all"):
        print("\n=== Analyzing layer-level encoder-swap results ===")
        output = analyze_results(ablation_dir, args.output_dir)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        local_path = args.output_dir / "layer_encoder_swap_results.json"
        with open(local_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved results to {local_path}")

if __name__ == "__main__":
    main()
