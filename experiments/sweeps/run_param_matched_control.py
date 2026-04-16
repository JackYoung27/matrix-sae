#!/usr/bin/env python3
"""Parameter-matched FlatSAE control experiment.

Stage 1: Train FlatSAE at nf=256 to match BilinearSAE's ~8.4M parameter budget.
Stage 2: Run downstream single-head perplexity patching to compare FlatSAE variants
against existing BilinearSAE checkpoints.

No Modal: runs locally on the user's GPU. Paths are configurable via CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make flat-layout sub-modules (core/, experiments/analysis, experiments/extraction)
# importable both as packages and by their internal "from sae import ..." style.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (
    _REPO_ROOT,
    _REPO_ROOT / "core",
    _REPO_ROOT / "experiments" / "analysis",
    _REPO_ROOT / "experiments" / "extraction",
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from core.train import train as train_sae  # noqa: E402
from experiments.analysis.evaluate_downstream import (  # noqa: E402
    evaluate_downstream,
    format_results_table,
    load_sae_from_checkpoint,
)
from experiments.extraction.extract_states import (  # noqa: E402
    load_corpus_from_file,
    load_corpus_tokens,
    load_model_and_tokenizer,
)

# Parameter-matched configuration
# BilinearSAE at nf=16384, rank=1, d_k=d_v=128 has 8,421,376 params.
# FlatSAE params = 2 * d_in * nf + nf + d_in  where d_in = 128*128 = 16384.
# Solving: nf = (8_421_376 - 16384) / (2*16384 + 1) ≈ 256.5 → nf=256.
# FlatSAE at nf=256: 8,405,248 params (99.8% match).
PARAM_MATCHED_NF = 256

LAYER = 9
HEAD = 0
SEED = 42
K = 32
EPOCHS = 20
BATCH_SIZE = 256
LR = 3e-4
LR_MIN = 3e-5
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
N_DOWNSTREAM_SEQUENCES = 500
SEQ_LEN = 1024

def _corpus_slug(corpus_source: str) -> str:
    return corpus_source.strip().lower()

def _states_dir_for(data_root: Path, corpus_source: str) -> Path:
    slug = _corpus_slug(corpus_source)
    if slug == "openwebtext":
        return data_root / "states"
    return data_root / f"states_{slug}"

def _experiment_tag(model_name: str, seq_len: int, n_samples: int, corpus_source: str) -> str:
    model_slug = model_name.split("/")[-1].lower().replace(".", "_")
    tag = f"{model_slug}_sl{seq_len}_ns{n_samples}"
    slug = _corpus_slug(corpus_source)
    if slug != "openwebtext":
        tag = f"{tag}_{slug}"
    return tag

def train_param_matched_flat(
    *,
    data_root: Path,
    states_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    seed: int = SEED,
    layer: int = LAYER,
    head: int = HEAD,
    n_features: int = PARAM_MATCHED_NF,
    k: int = K,
    corpus_source: str = "openwebtext",
) -> dict:
    """Train FlatSAE with nf=256 to match BilinearSAE's 8.4M parameter budget."""
    states_dir = states_dir or _states_dir_for(data_root, corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())

    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root = checkpoint_dir or (data_root / "checkpoints")
    output_dir = (
        ckpt_root / exp_tag
        / f"flat_L{layer}_H{head}_nf{n_features}_k{k}_s{seed}"
    )

    result = train_sae(
        sae_type="flat",
        data_dir=str(states_dir),
        layer=layer,
        head=head,
        n_features=n_features,
        k=k,
        lr=LR,
        lr_min=LR_MIN,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        warmup_steps=50,
        resample_every=250,
        output_dir=str(output_dir),
        seed=seed,
        rank=1,
    )

    print(f"\nTrained parameter-matched FlatSAE: nf={n_features}")
    print(f"  val MSE = {result['best_mse']:.6e}")
    print(f"  dead features = {result['final_n_dead']}")
    print(f"  training time = {result['total_time_s']:.0f}s")
    return result

def evaluate_param_matched_downstream(
    *,
    data_root: Path,
    states_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    model_name: str = MODEL_NAME,
    model_cache_dir: Path | None = None,
    seed: int = SEED,
    layer: int = LAYER,
    head: int = HEAD,
    n_features: int = PARAM_MATCHED_NF,
    k: int = K,
    n_sequences: int = N_DOWNSTREAM_SEQUENCES,
    seq_len: int = SEQ_LEN,
    corpus_source: str = "openwebtext",
) -> dict:
    """Single-head downstream eval: parameter-matched FlatSAE vs existing BilinearSAE."""
    import torch

    if model_cache_dir is not None:
        os.environ["HF_HOME"] = str(model_cache_dir)

    t0 = time.time()

    states_dir = states_dir or _states_dir_for(data_root, corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}")
    meta = json.loads(meta_path.read_text())

    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root_base = checkpoint_dir or (data_root / "checkpoints")
    ckpt_root = ckpt_root_base / exp_tag

    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model load: {mem_gb:.1f} GB")

    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    corpus_path = states_dir / "corpus.npy"
    if corpus_path.exists():
        batches = load_corpus_from_file(str(corpus_path), 8, n_samples=n_sequences)
    else:
        batches = load_corpus_tokens(tokenizer, None, seq_len, n_sequences, 8)
    actual = sum(b.shape[0] for b in batches)
    print(f"Loaded {actual} sequences for downstream eval")

    sae_configs = []

    # 1. Parameter-matched FlatSAE (nf=256)
    pm_tag = f"flat_L{layer}_H{head}_nf{n_features}_k{k}_s{seed}"
    pm_dir = ckpt_root / pm_tag
    pm_cfg_path = pm_dir / "config.json"
    pm_best_path = pm_dir / "best.pt"
    if pm_cfg_path.exists() and pm_best_path.exists():
        sae, cfg, train_mse = load_sae_from_checkpoint(
            str(pm_best_path), str(pm_cfg_path), device="cuda",
        )
        n_params = sum(p.numel() for p in sae.parameters())
        sae_configs.append({
            "tag": f"flat_nf{n_features} ({n_params/1e6:.1f}M params)",
            "sae": sae,
            "sae_type": "flat",
            "train_mse": train_mse,
        })
        print(f"Loaded parameter-matched FlatSAE: {n_params:,} params, val_mse={train_mse:.4e}")
    else:
        print(f"WARNING: parameter-matched FlatSAE not found at {pm_dir}")

    # 2. Existing BilinearSAE checkpoints for comparison (try nf=2048 and nf=16384)
    for nf_compare in [2048, 16384]:
        for sae_type in ["bilinear", "bilinear_tied"]:
            tag = f"{sae_type}_L{layer}_H{head}_nf{nf_compare}_k{k}_s{seed}"
            d = ckpt_root / tag
            cfg_path = d / "config.json"
            best_path = d / "best.pt"
            if cfg_path.exists() and best_path.exists():
                try:
                    sae, cfg, train_mse = load_sae_from_checkpoint(
                        str(best_path), str(cfg_path), device="cuda",
                    )
                    n_params = sum(p.numel() for p in sae.parameters())
                    sae_configs.append({
                        "tag": f"{sae_type}_nf{nf_compare} ({n_params/1e6:.1f}M params)",
                        "sae": sae,
                        "sae_type": sae_type,
                        "train_mse": train_mse,
                    })
                    print(f"Loaded {sae_type} nf={nf_compare}: {n_params:,} params, val_mse={train_mse:.4e}")
                except Exception as e:
                    print(f"WARN: failed to load {tag}: {e}")

    # 3. Existing FlatSAE nf=2048 for reference
    for nf_compare in [2048, 16384]:
        flat_tag = f"flat_L{layer}_H{head}_nf{nf_compare}_k{k}_s{seed}"
        d = ckpt_root / flat_tag
        cfg_path = d / "config.json"
        best_path = d / "best.pt"
        if cfg_path.exists() and best_path.exists():
            try:
                sae, cfg, train_mse = load_sae_from_checkpoint(
                    str(best_path), str(cfg_path), device="cuda",
                )
                n_params = sum(p.numel() for p in sae.parameters())
                sae_configs.append({
                    "tag": f"flat_nf{nf_compare} ({n_params/1e6:.1f}M params)",
                    "sae": sae,
                    "sae_type": "flat",
                    "train_mse": train_mse,
                })
                print(f"Loaded flat nf={nf_compare}: {n_params:,} params, val_mse={train_mse:.4e}")
            except Exception as e:
                print(f"WARN: failed to load {flat_tag}: {e}")

    # Also try expansion-factor-based naming (ef1 = nf=16384)
    for sae_type in ["flat", "bilinear", "bilinear_tied", "rank1"]:
        ef_tag = f"{sae_type}_L{layer}_H{head}_ef1_k{k}_s{seed}"
        d = ckpt_root / ef_tag
        cfg_path = d / "config.json"
        best_path = d / "best.pt"
        if cfg_path.exists() and best_path.exists():
            try:
                sae, cfg, train_mse = load_sae_from_checkpoint(
                    str(best_path), str(cfg_path), device="cuda",
                )
                n_params = sum(p.numel() for p in sae.parameters())
                sae_configs.append({
                    "tag": f"{sae_type}_ef1 ({n_params/1e6:.1f}M params)",
                    "sae": sae,
                    "sae_type": cfg["sae_type"],
                    "train_mse": train_mse,
                })
                print(f"Loaded {sae_type} ef=1: {n_params:,} params, val_mse={train_mse:.4e}")
            except Exception as e:
                print(f"WARN: failed to load {ef_tag}: {e}")

    if not sae_configs:
        return {"error": "no SAE checkpoints found"}

    print(f"\nEvaluating {len(sae_configs)} checkpoints + baseline")

    results = evaluate_downstream(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=batches,
        layer_idx=layer,
        sae_configs=sae_configs,
        head_idx=head,
        split_fraction=0.5,
        device="cuda",
    )
    results["model"] = model_name
    results["experiment"] = "parameter_matched_control"
    results["param_matched_nf"] = n_features
    results["total_time_s"] = round(time.time() - t0, 1)

    table = format_results_table(results)
    print(f"\n{table}")

    out_dir = data_root / "downstream_eval" / exp_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"layer_{layer}_param_matched_control_s{seed}.json"
    table_path = out_dir / f"layer_{layer}_param_matched_control_s{seed}_table.txt"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    table_path.write_text(table)

    print(f"\nResults saved to {results_path}")
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["train", "evaluate", "all"], default="all",
                        help="Which stage to run (default: all)")
    parser.add_argument("--data-root", type=Path, default=Path("./data"),
                        help="Root directory containing states/ and checkpoints/ (default: ./data)")
    parser.add_argument("--states-dir", type=Path, default=None,
                        help="Override states directory; defaults to <data-root>/states[_corpus]")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="(Unused; kept for CLI consistency) top-level output override")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Override checkpoint root; defaults to <data-root>/checkpoints")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"HF model name (default: {MODEL_NAME})")
    parser.add_argument("--model-cache-dir", type=Path, default=None,
                        help="HF_HOME cache directory for model weights")
    parser.add_argument("--layer", type=int, default=LAYER,
                        help=f"Layer index (default: {LAYER})")
    parser.add_argument("--head", type=int, default=HEAD,
                        help=f"Head index (default: {HEAD})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--n-sequences", type=int, default=N_DOWNSTREAM_SEQUENCES,
                        help=f"Downstream eval sequences (default: {N_DOWNSTREAM_SEQUENCES})")
    parser.add_argument("--n-features", type=int, default=PARAM_MATCHED_NF,
                        help=f"FlatSAE dictionary size (default: {PARAM_MATCHED_NF})")
    parser.add_argument("--k", type=int, default=K,
                        help=f"TopK sparsity (default: {K})")
    parser.add_argument("--corpus-source", default="openwebtext",
                        help="Corpus source slug (default: openwebtext)")
    args = parser.parse_args()

    data_root: Path = args.data_root.resolve()
    t0 = time.time()
    print(f"Parameter-matched FlatSAE control: stage={args.stage}, seed={args.seed}")
    print(f"  nf={args.n_features}, layer={args.layer}, head={args.head}, k={args.k}")
    print(f"  FlatSAE params ≈ {2 * 16384 * args.n_features + args.n_features + 16384:,}")
    print(f"  BilinearSAE params ≈ {4 * 16384 * 128 + 16384 + 16384:,}")

    if args.stage in ("train", "all"):
        train_result = train_param_matched_flat(
            data_root=data_root,
            states_dir=args.states_dir,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
            layer=args.layer,
            head=args.head,
            n_features=args.n_features,
            k=args.k,
            corpus_source=args.corpus_source,
        )
        print("\nTraining result:")
        print(f"  val MSE = {train_result['best_mse']:.6e}")
        print(f"  dead features = {train_result['final_n_dead']}")
        print(f"  time = {train_result['total_time_s']:.0f}s")

    if args.stage in ("evaluate", "all"):
        eval_result = evaluate_param_matched_downstream(
            data_root=data_root,
            states_dir=args.states_dir,
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.model,
            model_cache_dir=args.model_cache_dir,
            seed=args.seed,
            layer=args.layer,
            head=args.head,
            n_features=args.n_features,
            k=args.k,
            n_sequences=args.n_sequences,
            seq_len=SEQ_LEN,
            corpus_source=args.corpus_source,
        )
        if "error" in eval_result:
            print(f"\nDownstream eval error: {eval_result['error']}")
        else:
            print(f"\nDownstream eval completed in {eval_result.get('total_time_s', 0):.0f}s")

    print(f"\nTotal wall clock: {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
