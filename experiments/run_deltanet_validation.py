#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.train import train as train_sae_fn
from experiments.extraction.extract_states import (
    extract_states,
    get_gdn_layer_indices,
    load_corpus_tokens,
    probe_state_dims,
    setup_memmaps,
)

# Constants

DEFAULT_MODEL_NAME = "fla-hub/delta_net-1.3B-100B"
LAYERS = [1, 12, 22]  # early / mid / late
N_SAMPLES = 5000
SEQ_LEN = 1024
BATCH_SIZE = 16
N_FEATURES = 2048
K = 32
SEEDS = [0, 1, 2]
SAE_TYPES = ["flat", "bilinear"]
EPOCHS = 20

# Historical Modal volume for provenance: deltanet-validation-data

def _code_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "unknown"

CODE_SHA = _code_sha()

def extract_layer(
    layer: int,
    model_name: str,
    states_dir: Path,
    model_cache_dir: Path | None,
    n_samples: int,
    seq_len: int,
    batch_size: int,
) -> dict[str, Any]:
    """Extract DeltaNet recurrent states for one layer."""
    if model_cache_dir is not None:
        os.environ["HF_HOME"] = str(model_cache_dir)

    from fla.models.delta_net import DeltaNetForCausalLM
    from transformers import AutoTokenizer

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeltaNetForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    config = model.config
    gdn_layers = get_gdn_layer_indices(config)

    output_dir = states_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = output_dir / f"layer_{layer}"

    # Skip if already done
    layer_meta_path = layer_dir / "layer_metadata.json"
    if layer_meta_path.exists():
        existing = json.loads(layer_meta_path.read_text())
        if existing.get("n_samples", 0) >= n_samples and existing.get("model") == model_name:
            print(f"Layer {layer} already extracted ({existing['n_samples']} samples). Skipping.")
            return existing

    all_layers = gdn_layers
    print(f"All linear attention layers: {len(all_layers)} (model has {config.num_hidden_layers} total)")

    n_heads, key_dim, val_dim = probe_state_dims(model, layer, tokenizer, "cuda")
    print(f"Layer {layer}: {n_heads} heads x ({key_dim}, {val_dim})")

    batches = load_corpus_tokens(tokenizer, None, seq_len, n_samples, batch_size)
    actual_samples = sum(b.shape[0] for b in batches)
    print(f"Prepared {len(batches)} batches, {actual_samples} samples")

    memmaps = setup_memmaps(output_dir, [layer], n_heads, key_dim, val_dim, actual_samples)

    t0 = time.time()
    n_written = extract_states(model, config, batches, [layer], memmaps, "cuda")
    elapsed = time.time() - t0

    for mm in memmaps[layer]:
        mm.flush()

    layer_meta = {
        "model": model_name,
        "layer": layer,
        "n_samples": n_written,
        "n_heads": n_heads,
        "key_head_dim": key_dim,
        "value_head_dim": val_dim,
        "state_shape_per_head": [n_written, key_dim, val_dim],
        "dtype": "float16",
        "seq_len": seq_len,
        "extraction_time_s": round(elapsed, 1),
    }
    layer_dir.mkdir(parents=True, exist_ok=True)
    layer_meta_path.write_text(json.dumps(layer_meta, indent=2))

    print(f"Layer {layer}: {n_written} samples in {elapsed:.1f}s")
    return layer_meta

def spectral_audit(states_dir: Path, results_dir: Path) -> dict[str, Any]:
    """Compute sigma_1/sigma_2 ratio for each layer, head 0."""
    results: dict[str, Any] = {}

    for layer in LAYERS:
        data_path = states_dir / f"layer_{layer}" / "head_0.npy"
        if not data_path.exists():
            print(f"WARNING: {data_path} not found, skipping layer {layer}")
            continue

        states = np.load(str(data_path), mmap_mode="r").astype(np.float32)
        n, d_k, d_v = states.shape
        print(f"Layer {layer}: states shape = ({n}, {d_k}, {d_v})")

        sigma1_list, sigma2_list = [], []
        batch_sz = 512
        for i in range(0, n, batch_sz):
            j = min(i + batch_sz, n)
            S = torch.linalg.svdvals(torch.from_numpy(states[i:j]).cuda()).cpu().numpy()
            sigma1_list.append(S[:, 0])
            sigma2_list.append(S[:, 1] if S.shape[1] > 1 else np.zeros(j - i))

        sigma1 = np.concatenate(sigma1_list)
        sigma2 = np.concatenate(sigma2_list)
        ratio = sigma1 / np.clip(sigma2, 1e-12, None)

        layer_result = {
            "sigma1_mean": float(sigma1.mean()),
            "sigma2_mean": float(sigma2.mean()),
            "ratio_mean": float(ratio.mean()),
            "ratio_median": float(np.median(ratio)),
            "ratio_std": float(ratio.std()),
            "n_samples": int(n),
            "d_k": d_k,
            "d_v": d_v,
        }
        results[f"layer_{layer}"] = layer_result
        print(
            f"  Layer {layer}: sigma1/sigma2 = {layer_result['ratio_mean']:.2f} "
            f"(median {layer_result['ratio_median']:.2f})"
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "spectral_audit.json"
    out_path.write_text(json.dumps(results, indent=2))

    return results

def train_sae(
    sae_type: str,
    layer: int,
    seed: int,
    states_dir: Path,
    ckpt_dir: Path,
) -> dict[str, Any]:
    """Train one SAE configuration."""
    output_dir = str(ckpt_dir / f"{sae_type}_L{layer}_H0_nf{N_FEATURES}_k{K}_s{seed}")
    result = train_sae_fn(
        sae_type=sae_type,
        data_dir=str(states_dir),
        layer=layer,
        head=0,
        n_features=N_FEATURES,
        k=K,
        lr=3e-4,
        batch_size=256,
        epochs=EPOCHS,
        warmup_steps=50,
        resample_every=250,
        output_dir=output_dir,
        seed=seed,
        rank=1,
    )
    return result

def report(
    model_name: str,
    states_dir: Path,
    ckpt_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """Aggregate training results and spectral audit into a single report."""
    spectral_path = results_dir / "spectral_audit.json"
    spectral = json.loads(spectral_path.read_text()) if spectral_path.exists() else {}

    train_results: list[dict] = []
    for sae_type in SAE_TYPES:
        for layer in LAYERS:
            for seed in SEEDS:
                config_path = ckpt_dir / f"{sae_type}_L{layer}_H0_nf{N_FEATURES}_k{K}_s{seed}" / "config.json"
                if not config_path.exists():
                    print(f"Missing: {config_path}")
                    continue
                cfg = json.loads(config_path.read_text())
                train_results.append(cfg)

    comparison: dict[str, Any] = {}
    for layer in LAYERS:
        layer_key = f"layer_{layer}"
        flat_mses = []
        bilinear_mses = []
        for r in train_results:
            if r.get("layer") != layer or r.get("head", 0) != 0:
                continue
            tag = f"{r['sae_type']}_L{layer}_H0_nf{N_FEATURES}_k{K}_s{r.get('seed', 42)}"
            best_path = ckpt_dir / tag / "best.pt"
            if not best_path.exists():
                continue
            ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
            val_mse = ckpt.get("val_mse", float("inf"))
            if r["sae_type"] == "flat":
                flat_mses.append(val_mse)
            elif r["sae_type"] == "bilinear":
                bilinear_mses.append(val_mse)

        if flat_mses and bilinear_mses:
            flat_mean = float(np.mean(flat_mses))
            bilinear_mean = float(np.mean(bilinear_mses))
            advantage_pct = 100.0 * (flat_mean - bilinear_mean) / flat_mean if flat_mean > 0 else 0.0
            sigma_ratio = spectral.get(layer_key, {}).get("ratio_mean", float("nan"))

            comparison[layer_key] = {
                "flat_mse_mean": flat_mean,
                "flat_mse_std": float(np.std(flat_mses)),
                "bilinear_mse_mean": bilinear_mean,
                "bilinear_mse_std": float(np.std(bilinear_mses)),
                "bilinear_advantage_pct": round(advantage_pct, 2),
                "sigma1_sigma2_ratio": sigma_ratio,
                "n_seeds": len(flat_mses),
            }
            print(
                f"  Layer {layer}: flat={flat_mean:.4e}, bilinear={bilinear_mean:.4e}, "
                f"advantage={advantage_pct:.1f}%, sigma1/sigma2={sigma_ratio:.2f}"
            )
        else:
            print(f"  Layer {layer}: insufficient results (flat={len(flat_mses)}, bilinear={len(bilinear_mses)})")

    full_report = {
        "model": model_name,
        "layers": LAYERS,
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "k": K,
        "epochs": EPOCHS,
        "seeds": SEEDS,
        "spectral_audit": spectral,
        "comparison": comparison,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "deltanet_validation_report.json"
    out_path.write_text(json.dumps(full_report, indent=2))

    print("\n=== DeltaNet Validation Report ===")
    print(json.dumps(full_report, indent=2))
    return full_report

def main() -> None:
    parser = argparse.ArgumentParser(description="DeltaNet validation pipeline.")
    parser.add_argument("--stage", default="all", choices=["extract", "spectral", "train", "report", "all"])
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--states-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for report/spectral audit JSON.")
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=-1,
                        help="Run only this layer (extract stage). -1 = all layers.")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Run only this seed (train stage). -1 = all seeds.")
    parser.add_argument("--n-sequences", type=int, default=N_SAMPLES,
                        help="Number of sequences for state extraction.")
    args = parser.parse_args()

    states_dir = args.states_dir
    ckpt_dir = args.checkpoint_dir
    results_dir = args.output_dir

    if args.stage == "all":
        stages_to_run = ["extract", "spectral", "train", "report"]
    else:
        stages_to_run = [args.stage]

    layers_to_run = [args.layer] if args.layer >= 0 else LAYERS
    seeds_to_run = [args.seed] if args.seed >= 0 else SEEDS

    if "extract" in stages_to_run:
        print(f"\n{'='*60}")
        print(f"STAGE: extract ({len(layers_to_run)} layers sequentially)")
        print(f"{'='*60}")
        for layer in layers_to_run:
            meta = extract_layer(
                layer=layer,
                model_name=args.model,
                states_dir=states_dir,
                model_cache_dir=args.model_cache_dir,
                n_samples=args.n_sequences,
                seq_len=SEQ_LEN,
                batch_size=BATCH_SIZE,
            )
            print(f"  Layer {meta.get('layer', '?')}: {meta.get('n_samples', 0)} samples, "
                  f"({meta.get('key_head_dim', '?')}, {meta.get('value_head_dim', '?')})")

    if "spectral" in stages_to_run:
        print(f"\n{'='*60}")
        print("STAGE: spectral audit")
        print(f"{'='*60}")
        spectral_result = spectral_audit(states_dir, results_dir)
        for layer_key, vals in spectral_result.items():
            print(f"  {layer_key}: sigma1/sigma2 = {vals['ratio_mean']:.2f}")

    if "train" in stages_to_run:
        print(f"\n{'='*60}")
        configs = [(t, l, s) for t in SAE_TYPES for l in LAYERS for s in seeds_to_run]
        print(f"STAGE: train ({len(configs)} jobs)")
        print(f"{'='*60}")
        for sae_type, layer, seed in configs:
            result = train_sae(sae_type, layer, seed, states_dir, ckpt_dir)
            best_mse = result.get("best_mse", "?")
            total_time = result.get("total_time_s", "?")
            mse_str = f"{best_mse:.4e}" if isinstance(best_mse, (int, float)) else str(best_mse)
            print(f"  {sae_type} L{layer} s{seed}: best_mse={mse_str}, time={total_time}s")

    if "report" in stages_to_run:
        print(f"\n{'='*60}")
        print("STAGE: report")
        print(f"{'='*60}")
        report(args.model, states_dir, ckpt_dir, results_dir)

if __name__ == "__main__":
    main()
