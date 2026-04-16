
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Put the repo root on sys.path so `from core...` and `from experiments...`
# resolve when the script is invoked as `python experiments/run_batchtopk_downstream.py`
# from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.train import train as train_sae
from experiments.analysis.evaluate_downstream import (
    evaluate_downstream_perhead_matched,
    load_sae_from_checkpoint,
)
from experiments.extraction.extract_states import load_model_and_tokenizer

MODEL_DEFAULT = "Qwen/Qwen3.5-0.8B"
LAYER = 9
N_FEATURES = 2048
K = 32
SEED = 42
N_HEADS = 16

def train_batchtopk_head(
    sae_type: str,
    head: int,
    states_dir: str,
    checkpoint_root: str,
    layer: int = LAYER,
    seed: int = SEED,
) -> dict[str, Any]:
    out_dir = (
        f"{checkpoint_root}/"
        f"{sae_type}_L{layer}_H{head}_nf{N_FEATURES}_k{K}_batchtopk_s{seed}"
    )
    print(f"Training {sae_type} BatchTopK, L{layer} H{head}, seed {seed}")
    t0 = time.time()
    out = train_sae(
        sae_type=sae_type,
        data_dir=states_dir,
        layer=layer,
        head=head,
        n_features=N_FEATURES,
        k=K,
        lr=3e-4,
        lr_min=3e-5,
        batch_size=256,
        epochs=20,
        warmup_steps=50,
        norm_every=100,
        resample_every=250,
        rank=1,
        seed=seed,
        use_batchtopk=True,
        output_dir=out_dir,
    )
    elapsed = time.time() - t0
    return {
        "sae_type": sae_type,
        "head": head,
        "seed": seed,
        "best_mse": out.get("best_mse", out.get("best_val_mse")),
        "n_dead": out.get("final_n_dead"),
        "time_s": elapsed,
        "checkpoint_dir": out_dir,
    }

def evaluate_batchtopk_downstream(
    states_dir: str,
    matched_checkpoint_root: str,
    batchtopk_checkpoint_root: str,
    output_dir: str,
    model_name: str,
    model_cache_dir: str | None,
    layer: int = LAYER,
    n_sequences: int = 500,
    seed: int = SEED,
) -> dict[str, Any]:
    import numpy as np
    import torch

    print(f"=== BatchTopK downstream, L{layer}, n_sequences={n_sequences} ===")

    if model_cache_dir:
        import os

        os.environ.setdefault("HF_HOME", model_cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", model_cache_dir)

    model, tokenizer, config = load_model_and_tokenizer(model_name, device="cuda")
    model.eval()

    corpus_ids = np.load(f"{states_dir}/corpus.npy", mmap_mode="r")
    actual = min(n_sequences, len(corpus_ids))
    batches = []
    batch_size = 8
    for start in range(0, actual, batch_size):
        end = min(start + batch_size, actual)
        batches.append(np.array(corpus_ids[start:end], dtype=np.int64))

    corpus_batches = [torch.tensor(b, dtype=torch.long) for b in batches]

    sae_type_configs: dict[str, dict[int, tuple]] = {}

    # Canonical TopK baselines from the clean matched cohort.
    for sae_type in ["flat", "rank1", "bilinear"]:
        head_saes: dict[int, tuple] = {}
        for head in range(N_HEADS):
            ckpt_dir = (
                Path(matched_checkpoint_root)
                / f"{sae_type}_L{layer}_H{head}_nf{N_FEATURES}_k{K}_s{seed}"
            )
            best_path = ckpt_dir / "best.pt"
            cfg_path = ckpt_dir / "config.json"
            sae, _, _ = load_sae_from_checkpoint(
                str(best_path),
                str(cfg_path),
                device="cuda",
            )
            head_saes[head] = (sae, sae_type)
        sae_type_configs[f"{sae_type}_topk"] = head_saes

    # Fresh BatchTopK checkpoints.
    for sae_type in ["flat", "rank1", "bilinear"]:
        head_saes = {}
        for head in range(N_HEADS):
            ckpt_dir = (
                Path(batchtopk_checkpoint_root)
                / f"{sae_type}_L{layer}_H{head}_nf{N_FEATURES}_k{K}_batchtopk_s{seed}"
            )
            best_path = ckpt_dir / "best.pt"
            cfg_path = ckpt_dir / "config.json"
            sae, _, _ = load_sae_from_checkpoint(
                str(best_path),
                str(cfg_path),
                device="cuda",
            )
            head_saes[head] = (sae, sae_type)
        sae_type_configs[f"{sae_type}_batchtopk"] = head_saes

    result = evaluate_downstream_perhead_matched(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=corpus_batches,
        layer_idx=layer,
        sae_type_configs=sae_type_configs,
        n_heads=N_HEADS,
        split_fraction=0.5,
        device="cuda",
    )
    result["experiment"] = "batchtopk_downstream"
    result["seed"] = seed
    result["n_features"] = N_FEATURES
    result["k"] = K
    result["n_sequences"] = actual

    out_dir = Path(output_dir) / "reviewer_experiments" / "batchtopk_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"L{layer}_s{seed}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result

def main():
    parser = argparse.ArgumentParser(description="BatchTopK downstream experiment")
    parser.add_argument("--states-dir", required=True, help="Path to extracted states directory (replaces /data/states).")
    parser.add_argument("--output-dir", default="results/data", help="Directory for JSON outputs.")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Root directory for SAE checkpoints (replaces /data/checkpoints).",
    )
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--model-cache-dir", default=None, help="HF cache dir (replaces /models).")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-sequences", type=int, default=500)
    args = parser.parse_args()

    matched_checkpoint_root = str(Path(args.checkpoint_dir) / "qwen3_5-0_8b_sl1024_ns5000_ultrachat_200k")
    batchtopk_checkpoint_root = str(Path(args.checkpoint_dir) / "review_batchtopk_downstream")

    t0 = time.time()

    # Sequential training. Parallelize with joblib or multiprocessing if desired.
    train_results: list[dict[str, Any]] = []
    failures: list[str] = []
    for sae_type in ["flat", "rank1", "bilinear"]:
        for head in range(N_HEADS):
            try:
                result = train_batchtopk_head(
                    sae_type=sae_type,
                    head=head,
                    states_dir=args.states_dir,
                    checkpoint_root=batchtopk_checkpoint_root,
                    layer=args.layer,
                    seed=args.seed,
                )
                train_results.append(result)
                print(
                    f"[ok] {sae_type} H{head}: "
                    f"MSE={result['best_mse']:.6e}, dead={result['n_dead']}, {result['time_s']:.0f}s"
                )
            except Exception as exc:
                failures.append(f"{sae_type} H{head}: {exc}")
                print(f"[fail] {sae_type} H{head}: {exc}")

    if failures:
        raise RuntimeError("BatchTopK training failures: " + "; ".join(failures))

    print("\nTraining complete. Running downstream evaluation...")
    downstream = evaluate_batchtopk_downstream(
        states_dir=args.states_dir,
        matched_checkpoint_root=matched_checkpoint_root,
        batchtopk_checkpoint_root=batchtopk_checkpoint_root,
        output_dir=args.output_dir,
        model_name=args.model,
        model_cache_dir=args.model_cache_dir,
        layer=args.layer,
        n_sequences=args.n_sequences,
        seed=args.seed,
    )

    summary = {
        "experiment": "batchtopk_downstream",
        "generated_at": time.time(),
        "train_results": train_results,
        "downstream": downstream,
    }

    out_path = Path(args.output_dir) / "batchtopk_downstream.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
