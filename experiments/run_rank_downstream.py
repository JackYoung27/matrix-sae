
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Put the repo root on sys.path so `from core...` and `from experiments...`
# resolve when the script is invoked directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.train import train as train_sae
from experiments.analysis.evaluate_downstream import (
    evaluate_downstream_perhead_matched,
    load_sae_from_checkpoint,
)
from experiments.extraction.extract_states import load_model_and_tokenizer

def _parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]

def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)

MODEL_DEFAULT = "Qwen/Qwen3.5-0.8B"
N_FEATURES = 2048
K = 32
SEED = 42
N_HEADS = 16

def _canonical_rank1_dir(matched_root: str, layer: int, head: int, seed: int) -> Path:
    return Path(matched_root) / f"bilinear_L{layer}_H{head}_nf{N_FEATURES}_k{K}_s{seed}"

def _rank_checkpoint_dir(rank_root: str, layer: int, head: int, rank: int, seed: int) -> Path:
    return Path(rank_root) / f"bilinear_r{rank}_L{layer}_H{head}_nf{N_FEATURES}_k{K}_s{seed}"

def _rank1_checkpoint_dir(matched_root: str, rank_root: str, layer: int, head: int, seed: int) -> Path:
    canonical = _canonical_rank1_dir(matched_root, layer=layer, head=head, seed=seed)
    if (canonical / "best.pt").exists() and (canonical / "config.json").exists():
        return canonical
    return _rank_checkpoint_dir(rank_root, layer=layer, head=head, rank=1, seed=seed)

def _load_existing_train_result(ckpt_dir: Path) -> dict[str, Any]:
    import torch

    best_path = ckpt_dir / "best.pt"
    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    return {
        "best_mse": payload.get("val_mse"),
        "config": payload.get("config", {}),
    }

def train_rank_head(
    layer: int,
    head: int,
    states_dir: str,
    matched_checkpoint_root: str,
    rank_checkpoint_root: str,
    rank: int = 2,
    seed: int = SEED,
) -> dict[str, Any]:
    if rank == 1:
        canonical_dir = _canonical_rank1_dir(matched_checkpoint_root, layer=layer, head=head, seed=seed)
        canonical_best = canonical_dir / "best.pt"
        canonical_cfg = canonical_dir / "config.json"
        if canonical_best.exists() and canonical_cfg.exists():
            existing = _load_existing_train_result(canonical_dir)
            return {
                "layer": layer,
                "head": head,
                "rank": rank,
                "seed": seed,
                "best_mse": existing["best_mse"],
                "checkpoint_dir": str(canonical_dir),
                "skipped": True,
                "source": "canonical",
            }

    ckpt_dir = _rank_checkpoint_dir(rank_checkpoint_root, layer=layer, head=head, rank=rank, seed=seed)
    best_path = ckpt_dir / "best.pt"
    cfg_path = ckpt_dir / "config.json"

    if best_path.exists() and cfg_path.exists():
        existing = _load_existing_train_result(ckpt_dir)
        return {
            "layer": layer,
            "head": head,
            "rank": rank,
            "seed": seed,
            "best_mse": existing["best_mse"],
            "checkpoint_dir": str(ckpt_dir),
            "skipped": True,
        }

    print(f"Training bilinear rank-{rank}, L{layer} H{head}, seed {seed}")
    t0 = time.time()
    out = train_sae(
        sae_type="bilinear",
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
        rank=rank,
        seed=seed,
        output_dir=str(ckpt_dir),
    )
    elapsed = time.time() - t0
    return {
        "layer": layer,
        "head": head,
        "rank": rank,
        "seed": seed,
        "best_mse": out.get("best_mse", out.get("best_val_mse")),
        "n_dead": out.get("final_n_dead"),
        "time_s": elapsed,
        "checkpoint_dir": str(ckpt_dir),
        "skipped": False,
    }

def evaluate_rank_downstream(
    layer: int,
    states_dir: str,
    matched_checkpoint_root: str,
    rank_checkpoint_root: str,
    output_dir: str,
    model_name: str,
    ranks_csv: str = "1,2",
    n_sequences: int = 500,
    seed: int = SEED,
) -> dict[str, Any]:
    import numpy as np
    import torch

    ranks = _parse_int_csv(ranks_csv)
    if 1 not in ranks:
        ranks = [1] + ranks

    model, tokenizer, _ = load_model_and_tokenizer(model_name, device="cuda")
    model.eval()

    corpus_ids = np.load(f"{states_dir}/corpus.npy", mmap_mode="r")
    actual = min(n_sequences, len(corpus_ids))
    batches = []
    batch_size = 8
    for start in range(0, actual, batch_size):
        end = min(start + batch_size, actual)
        batches.append(np.array(corpus_ids[start:end], dtype=np.int64))
    corpus_batches = [torch.tensor(batch, dtype=torch.long) for batch in batches]

    sae_type_configs: dict[str, dict[int, tuple]] = {}
    train_mse_summary: dict[str, dict[str, float | int | None]] = {}

    for rank in ranks:
        tag = f"bilinear_r{rank}"
        head_saes: dict[int, tuple] = {}
        train_mses: list[float] = []
        for head in range(N_HEADS):
            if rank == 1:
                ckpt_dir = _rank1_checkpoint_dir(
                    matched_checkpoint_root, rank_checkpoint_root, layer=layer, head=head, seed=seed
                )
            else:
                ckpt_dir = _rank_checkpoint_dir(
                    rank_checkpoint_root, layer=layer, head=head, rank=rank, seed=seed
                )

            best_path = ckpt_dir / "best.pt"
            cfg_path = ckpt_dir / "config.json"
            sae, _, train_mse = load_sae_from_checkpoint(
                str(best_path),
                str(cfg_path),
                device="cuda",
            )
            head_saes[head] = (sae, "bilinear")
            if train_mse is not None:
                train_mses.append(float(train_mse))

        sae_type_configs[tag] = head_saes
        train_mse_summary[tag] = {
            "rank": rank,
            "n_heads_loaded": len(head_saes),
            "mean_val_mse": _mean(train_mses),
        }

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
    result["experiment"] = "rank_downstream"
    result["seed"] = seed
    result["n_features"] = N_FEATURES
    result["k"] = K
    result["n_sequences"] = actual
    result["train_mse_summary"] = train_mse_summary
    result["ranks"] = ranks

    out_dir = Path(output_dir) / "reviewer_experiments" / "rank_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"L{layer}_ranks_{'-'.join(str(rank) for rank in ranks)}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result

def main():
    parser = argparse.ArgumentParser(description="Higher-rank bilinear SAE downstream experiment")
    parser.add_argument("--states-dir", required=True, help="Path to extracted states directory (replaces /data/states).")
    parser.add_argument("--output-dir", default="results/data", help="Directory for JSON outputs.")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Root directory for SAE checkpoints (replaces /data/checkpoints).",
    )
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--model-cache-dir", default=None, help="HF cache dir (replaces /models).")
    parser.add_argument("--layers", default="9,17", help="Comma-separated layers to sweep.")
    parser.add_argument("--ranks", default="2", help="Comma-separated ranks (rank 1 reuses canonical).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-sequences", type=int, default=500)
    args = parser.parse_args()

    if args.model_cache_dir:
        import os

        os.environ.setdefault("HF_HOME", args.model_cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.model_cache_dir)

    checkpoint_root = Path(args.checkpoint_dir)
    matched_checkpoint_root = str(checkpoint_root / "qwen3_5-0_8b_sl1024_ns5000_ultrachat_200k")
    rank_checkpoint_root = str(checkpoint_root / "review_rank_downstream")

    t0 = time.time()
    layers_list = _parse_int_csv(args.layers)
    requested_ranks = sorted(set(_parse_int_csv(args.ranks)))
    train_ranks = sorted({1, *[rank for rank in requested_ranks if rank > 1]})
    eval_ranks = sorted({1, *requested_ranks})

    # Sequential. Parallelize with joblib/multiprocessing if desired.
    train_results: list[dict[str, Any]] = []
    failures: list[str] = []
    total_jobs = len(layers_list) * len(train_ranks) * N_HEADS
    print(f"Running {total_jobs} higher-rank training jobs sequentially.")
    for layer in layers_list:
        for rank in train_ranks:
            for head in range(N_HEADS):
                try:
                    result = train_rank_head(
                        layer=layer,
                        head=head,
                        states_dir=args.states_dir,
                        matched_checkpoint_root=matched_checkpoint_root,
                        rank_checkpoint_root=rank_checkpoint_root,
                        rank=rank,
                        seed=args.seed,
                    )
                    train_results.append(result)
                    status = "skip" if result.get("skipped") else "done"
                    print(
                        f"[{status}] L{layer} r{rank} H{head}: "
                        f"MSE={result.get('best_mse'):.6e}"
                    )
                except Exception as exc:
                    failures.append(f"L{layer} r{rank} H{head}: {exc}")
                    print(f"[fail] L{layer} r{rank} H{head}: {exc}")

    if failures:
        raise RuntimeError("Higher-rank training failures: " + "; ".join(failures))

    print("\nTraining complete. Running downstream evaluations...")
    downstream_results: dict[str, Any] = {}
    for layer in layers_list:
        result = evaluate_rank_downstream(
            layer=layer,
            states_dir=args.states_dir,
            matched_checkpoint_root=matched_checkpoint_root,
            rank_checkpoint_root=rank_checkpoint_root,
            output_dir=args.output_dir,
            model_name=args.model,
            ranks_csv=",".join(str(rank) for rank in eval_ranks),
            n_sequences=args.n_sequences,
            seed=args.seed,
        )
        downstream_results[f"L{layer}"] = result

    summary = {
        "experiment": "rank_downstream",
        "generated_at": time.time(),
        "layers": layers_list,
        "ranks": eval_ranks,
        "train_results": train_results,
        "downstream": downstream_results,
    }

    out_path = Path(args.output_dir) / "rank_downstream.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
