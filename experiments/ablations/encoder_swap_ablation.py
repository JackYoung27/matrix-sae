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

TARGET_PAIRS = [
    {"layer": 0,  "head": 2, "sv1_sv2": 16.4, "label": "concentrated"},
    {"layer": 9,  "head": 0, "sv1_sv2": 6.8,  "label": "medium"},
    {"layer": 17, "head": 2, "sv1_sv2": 3.0,  "label": "diffuse"},
]

ALL_SAE_TYPES = ["flat", "rank1", "bilinear", "bilinear_tied", "bilinear_flat"]
NEW_SAE_TYPES = ["bilinear", "bilinear_flat", "bilinear_tied"]
SEEDS = [0, 1, 42]

SAE_N_FEATURES = 2048
SAE_K = 32
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 256
TRAIN_LR = 3e-4
TRAIN_LR_MIN = 3e-5
TRAIN_WARMUP_STEPS = 50
TRAIN_RESAMPLE_EVERY = 250

def train_sae(
    layer: int,
    head: int,
    sae_type: str,
    seed: int,
    state_dir: Path,
    new_ckpt_dir: Path,
    n_features: int = SAE_N_FEATURES,
    k: int = SAE_K,
) -> dict[str, Any]:
    """Train one SAE on one (layer, head) pair. Skips if checkpoint exists."""
    ckpt_dir = new_ckpt_dir / f"layer_{layer}" / f"head_{head}" / f"{sae_type}_s{seed}"
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

    data_path = state_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"State data not found: {data_path}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    result = train_sae_fn(
        sae_type=sae_type,
        data_dir=str(state_dir),
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

def analyze_results(
    new_ckpt_dir: Path,
    existing_ckpt_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Load all checkpoints, compute mean MSE per (layer, type), print table."""
    results_by_pair: list[dict[str, Any]] = []
    missing_runs: list[dict[str, Any]] = []

    for pair_info in TARGET_PAIRS:
        layer = pair_info["layer"]
        head = pair_info["head"]
        sv1_sv2 = pair_info["sv1_sv2"]
        label = pair_info["label"]

        mse_by_type: dict[str, dict[str, Any]] = {}

        for sae_type in ALL_SAE_TYPES:
            seed_mses: dict[int, float] = {}
            for seed in SEEDS:
                new_path = new_ckpt_dir / f"layer_{layer}" / f"head_{head}" / f"{sae_type}_s{seed}" / "best.pt"
                old_path = existing_ckpt_dir / f"layer_{layer}" / f"head_{head}" / f"{sae_type}_s{seed}" / "best.pt"

                val_mse = _load_checkpoint_mse(new_path)
                if val_mse is None:
                    val_mse = _load_checkpoint_mse(old_path)
                if val_mse is not None:
                    seed_mses[seed] = float(val_mse)
                else:
                    missing_runs.append({
                        "layer": layer,
                        "head": head,
                        "sae_type": sae_type,
                        "seed": seed,
                    })

            if seed_mses:
                mse_vals = list(seed_mses.values())
                mse_by_type[sae_type] = {
                    "mean": float(np.mean(mse_vals)),
                    "std": float(np.std(mse_vals)),
                    "n_seeds": len(mse_vals),
                    "seed_mses": seed_mses,
                }

        flat_mean = mse_by_type.get("flat", {}).get("mean")
        advantages: dict[str, float | None] = {}
        for sae_type in ALL_SAE_TYPES:
            if sae_type == "flat":
                advantages[sae_type] = 0.0
                continue
            type_mean = mse_by_type.get(sae_type, {}).get("mean")
            if type_mean is not None and flat_mean is not None and abs(flat_mean) > 1e-12:
                advantages[sae_type] = (flat_mean - type_mean) / flat_mean * 100.0
            else:
                advantages[sae_type] = None

        results_by_pair.append({
            "layer": layer,
            "head": head,
            "sv1_sv2": sv1_sv2,
            "label": label,
            "mse_by_type": mse_by_type,
            "advantage_vs_flat_pct": advantages,
        })

    if missing_runs:
        details = ", ".join(
            f"L{x['layer']} H{x['head']} {x['sae_type']} s{x['seed']}" for x in missing_runs[:10]
        )
        if len(missing_runs) > 10:
            details += f", ... (+{len(missing_runs) - 10} more)"
        raise RuntimeError(
            "Encoder-swap analysis requires all checkpoints to be present. Missing: "
            f"{details}"
        )

    print(f"\n{'=' * 90}")
    print("ENCODER-SWAP ABLATION RESULTS")
    print(f"{'=' * 90}")
    print(f"{'Pair':>10}  {'sv1/sv2':>7}  {'Label':>12}  ", end="")
    for sae_type in ALL_SAE_TYPES:
        print(f"  {sae_type:>14}", end="")
    print()
    print("-" * 90)

    for entry in results_by_pair:
        pair_str = f"L{entry['layer']}H{entry['head']}"
        print(f"{pair_str:>10}  {entry['sv1_sv2']:>7.1f}  {entry['label']:>12}  ", end="")
        for sae_type in ALL_SAE_TYPES:
            mse_info = entry["mse_by_type"].get(sae_type)
            if mse_info:
                print(f"  {mse_info['mean']:.4e}", end="")
            else:
                print(f"  {'n/a':>14}", end="")
        print()

    print(f"\n{'=' * 90}")
    print("% ADVANTAGE vs FLAT (positive = better than flat)")
    print(f"{'=' * 90}")
    print(f"{'Pair':>10}  {'sv1/sv2':>7}  {'Label':>12}  ", end="")
    for sae_type in ALL_SAE_TYPES:
        if sae_type == "flat":
            continue
        print(f"  {sae_type:>14}", end="")
    print()
    print("-" * 90)

    for entry in results_by_pair:
        pair_str = f"L{entry['layer']}H{entry['head']}"
        print(f"{pair_str:>10}  {entry['sv1_sv2']:>7.1f}  {entry['label']:>12}  ", end="")
        for sae_type in ALL_SAE_TYPES:
            if sae_type == "flat":
                continue
            adv = entry["advantage_vs_flat_pct"].get(sae_type)
            if adv is not None:
                print(f"  {adv:>+13.1f}%", end="")
            else:
                print(f"  {'n/a':>14}", end="")
        print()

    print(f"\n{'=' * 90}")
    print("KEY COMPARISONS")
    print(f"{'=' * 90}")
    for entry in results_by_pair:
        layer, head = entry["layer"], entry["head"]
        mse = entry["mse_by_type"]
        pair_str = f"L{layer}H{head} ({entry['label']}, sv1/sv2={entry['sv1_sv2']})"

        bf_mean = mse.get("bilinear_flat", {}).get("mean")
        fl_mean = mse.get("flat", {}).get("mean")
        bi_mean = mse.get("bilinear", {}).get("mean")
        r1_mean = mse.get("rank1", {}).get("mean")

        print(f"\n{pair_str}:")
        if bf_mean is not None and fl_mean is not None:
            diff = (fl_mean - bf_mean) / fl_mean * 100.0
            winner = "bilinear_flat" if diff > 0 else "flat"
            print(f"  bilinear_flat vs flat: {diff:+.2f}% ({winner} wins)")
            print(f"    -> bilinear encoder {'helps' if diff > 0 else 'does not help'} with flat decoder")
        if bf_mean is not None and bi_mean is not None:
            diff = (bf_mean - bi_mean) / bf_mean * 100.0
            print(f"  bilinear vs bilinear_flat: {diff:+.2f}% gain from rank-1 decoder")
            if abs(diff) < 2.0:
                print("    -> rank-1 decoder adds little; encoder is the driver")
            else:
                print(f"    -> rank-1 decoder contributes meaningfully ({diff:+.2f}%)")
        if r1_mean is not None and fl_mean is not None:
            diff = (fl_mean - r1_mean) / fl_mean * 100.0
            print(f"  rank1 vs flat: {diff:+.2f}% (rank-1 decoder with linear encoder)")

    output = {
        "experiment": "encoder_swap_ablation",
        "hypothesis": (
            "bilinear encoder drives reconstruction gains on diffuse-spectrum layers; "
            "rank-1 decoder is secondary"
        ),
        "target_pairs": TARGET_PAIRS,
        "sae_types": ALL_SAE_TYPES,
        "seeds": SEEDS,
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
        "per_pair": results_by_pair,
        "code_sha": CODE_SHA,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "encoder_swap_ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output

def main() -> None:
    parser = argparse.ArgumentParser(description="Encoder-swap ablation (3 layer-head pairs).")
    parser.add_argument("--stage", default="all", choices=["train", "analyze", "all"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B",
                        help="Source model (for provenance; states should already be extracted).")
    parser.add_argument("--states-dir", type=Path, required=True,
                        help="Directory containing pre-extracted GDN states (layer_*/head_*.npy).")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Output directory for new SAE checkpoints.")
    parser.add_argument("--existing-checkpoint-dir", type=Path, default=None,
                        help="Directory with existing flat/rank1 checkpoints (for analyze stage).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for analysis JSON output.")
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=-1,
                        help="Run only this layer. -1 = all TARGET_PAIRS.")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Run only this seed. -1 = all seeds.")
    parser.add_argument("--n-sequences", type=int, default=-1,
                        help="Unused for this script (states are pre-extracted).")
    args = parser.parse_args()

    state_dir = args.states_dir
    new_ckpt_dir = args.checkpoint_dir
    existing_ckpt_dir = args.existing_checkpoint_dir if args.existing_checkpoint_dir is not None else new_ckpt_dir

    seeds_to_run = [args.seed] if args.seed >= 0 else SEEDS

    if args.stage in ("train", "all"):
        print("=== Training encoder-swap SAEs ===")
        pairs_to_run = [p for p in TARGET_PAIRS if args.layer < 0 or p["layer"] == args.layer]
        print(f"{len(pairs_to_run)} pairs x {len(NEW_SAE_TYPES)} types x {len(seeds_to_run)} seeds")

        jobs: list[dict[str, Any]] = []
        for pair_info in pairs_to_run:
            layer = pair_info["layer"]
            head = pair_info["head"]
            for sae_type in NEW_SAE_TYPES:
                for seed in seeds_to_run:
                    jobs.append({
                        "layer": layer,
                        "head": head,
                        "sae_type": sae_type,
                        "seed": seed,
                    })

        n_jobs = len(jobs)
        print(f"Running {n_jobs} training jobs sequentially")
        results = []
        failures = []
        for idx, job in enumerate(jobs, start=1):
            try:
                result = train_sae(job["layer"], job["head"], job["sae_type"], job["seed"],
                                   state_dir, new_ckpt_dir)
                results.append(result)
                mse = result.get("best_mse", "?")
                skipped = result.get("skipped", False)
                status = "skipped" if skipped else "done"
                print(
                    f"  [{idx}/{n_jobs}] L{job['layer']} H{job['head']} "
                    f"{job['sae_type']} s{job['seed']}: mse={mse} ({status})"
                )
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
            failures_path = args.output_dir / "encoder_swap_training_failures.json"
            with open(failures_path, "w") as f:
                json.dump({"failures": failures, "code_sha": CODE_SHA}, f, indent=2)
            print("Failed jobs:")
            for f in failures:
                print(f"  L{f['layer']} H{f['head']} {f['sae_type']} s{f['seed']}: {f['error']}")
            print(f"Saved failure summary to {failures_path}")

    if args.stage in ("analyze", "all"):
        print("\n=== Analyzing encoder-swap ablation results ===")
        output = analyze_results(new_ckpt_dir, existing_ckpt_dir, args.output_dir)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        local_path = args.output_dir / "encoder_swap_ablation.json"
        with open(local_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved results to {local_path}")

if __name__ == "__main__":
    main()
